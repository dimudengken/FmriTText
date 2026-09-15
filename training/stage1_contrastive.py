"""Stage 1：CLIP 风格多模态对比学习（复刻 BIT-LLM Stage 1）。

训练 encoder + per-subject ridge + contrastive head，对齐 fMRI↔图像 与 fMRI↔文本。
LLM 不加载（省 ~6.5GB 显存）；CLIP 视觉/文本编码器全程冻结，只当特征提供者。

数据流：
  voxels(B, n_voxels) → KeyValueEncoder → (B,128,1024)
    → SubjectRidge → (B,128,1024) → mean-pool → Linear(1024→768) → L2norm
  ↕ 双向 InfoNCE（BrainContrastiveLoss）
  图像/文本嵌入由 frozen CLIP ViT-L/14 提供。

本步是修复 MindLLM 编码器"对体素不敏感"（Real/Zero 余弦 0.9889）的关键：
对比学习强制编码器依赖体素值，而非只靠位置/分区嵌入。

跨被试机制（2026-08-31 换型，BIT-LLM 配方，MoCo bank 已证伪）：
  单被试 batch + per-subject ridge 是 BIT-LLM 原文机制的两处偏离——单被试 batch 的 in-batch
  InfoNCE 负样本永远同被试，无跨被试对齐压力 → 各被试语义空间漂移（S8 检索随机级根因）；
  per-subject ridge 让 head 训在 post-ridge 域、S8 零样本走 pre-ridge 域（但 2026-08-31
  ridge 换型诊断：S8 ridge 换入后 zero_cos 0.097 特征已敏感、检索仍随机级 3717 → 域失配
  不是主因，语义错位在编码器输出端本身）。
  `--no_ridge --mixed_batch` = BIT-LLM 全配方：去掉 ridge（读出头全共享）+ 每个 batch 混合
  全部训练被试（in-batch 负样本天然跨被试，无需任何额外损失/bank）。
  `--xsubj_lambda`（MoCo FIFO bank）已证伪：λ=0.05/0.2 单调反噬 cos(real,zero) 0.59→0.76，
  模型走"降体素依赖"逃生通道，勿再用。

用法（服务器，fmri 环境，项目根目录）：
  python training/stage1_contrastive.py \
      --data_path /root/autodl-tmp/MindEyeV2-main/data
  # 快速冒烟：python training/stage1_contrastive.py --max_steps 200 --epochs 1
  # 跨被试机制换型（BIT-LLM 配方，S8 shared1000 干净 held-out）：
  python training/stage1_contrastive.py \
      --data_path /root/autodl-tmp/MindEyeV2-main/data \
      --no_ridge --mixed_batch --exclude_shared1000

输出：{out_dir}/stage1_encoder.pt（encoder + ridge + head 的状态字典）
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from transformers import CLIPModel, CLIPTokenizer

from data.dataloader import build_mixed_train_loader, build_train_loader
from data.preprocessing import build_caption_map
from models.fmri_encoder import KeyValueEncoder
from models.ridge import SubjectRidge
from training.losses import BrainContrastiveLoss, info_nce_crosssubj
from utils.save_load import load_train_ckpt, save_train_ckpt

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def batch_topk_retrieval(sim, k=1):
    """sim (B, B)：第 i 行里真值在第 i 列。返回命中率。"""
    topk = torch.topk(sim, k=k, dim=1).indices
    hit = (topk == torch.arange(sim.size(0), device=sim.device).unsqueeze(1)).any(dim=1)
    return hit.float().mean().item()


class ContrastiveHead(nn.Module):
    """128 tokens → mean-pool → Linear(1024→768) → L2 归一化（CLIP 对比空间）。"""

    def __init__(self, in_dim=1024, out_dim=768):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, tokens):
        x = tokens.mean(dim=1)  # (B, in_dim)
        x = self.proj(x)
        return F.normalize(x, dim=-1)


class ContrastiveBrainModel(nn.Module):
    def __init__(self, n_subjects=8, anatomy_dir="data/anatomy_cache", out_dim=768,
                 use_ridge=True):
        super().__init__()
        self.use_ridge = use_ridge
        self.encoder = KeyValueEncoder(anatomy_dir=anatomy_dir)
        if use_ridge:
            self.ridge = SubjectRidge(n_subjects=n_subjects, dim=1024)
        self.head = ContrastiveHead(in_dim=1024, out_dim=out_dim)

    def forward(self, voxels, subj):
        x = self.encoder(voxels, subj)
        if self.use_ridge:
            x = self.ridge(x, subj)
        return self.head(x)


def push_bank(bank, brain_emb, nsd, subj, cap):
    """把当前 batch 的 head 嵌入（detach）推入 FIFO bank；(emb_cpu, nsd_idx, subj)。"""
    e = brain_emb.detach().cpu()
    for j in range(e.size(0)):
        bank.append((e[j], int(nsd[j]), subj))
    if len(bank) > cap:
        del bank[:len(bank) - cap]


def cross_subject_loss(brain_emb, image_emb, nsd, subj, bank, n_neg, logit_scale, device):
    """跨被试对比项：bank 里其他被试的 head 嵌入作负样本（同图排除）。bank 覆盖不足返回 None。"""
    others = [(e, i) for e, i, s in bank if s != subj]
    if len(others) < max(64, n_neg):
        return None
    embs = torch.stack([e for e, _ in others]).to(device)
    idxs = torch.tensor([i for _, i in others], device=device)
    return info_nce_crosssubj(brain_emb, image_emb, embs, idxs, nsd.to(device), n_neg, logit_scale)


def cross_subject_retrieval_diag(brain_emb, nsd, subj, bank):
    """训练时代理：shared1000 同图的其他被试嵌入在 bank 里的检索 rank（MedR）。

    跨被试对齐若在变好，同图跨被试嵌入应高排位 → MedR 小（随机级 ≈ K'/2）。返回 float 或 None。
    """
    others = [(e, i) for e, i, s in bank if s != subj]
    if len(others) < 256:
        return None
    oe = torch.stack([e for e, _ in others])  # (K',768) cpu
    oi = torch.tensor([i for _, i in others])  # (K',)
    sim = brain_emb.detach().cpu() @ oe.T  # (B,K')
    ranks = []
    for j in range(brain_emb.size(0)):
        tgt = int(nsd[j])
        hits = (oi == tgt).nonzero().squeeze(1)
        if hits.numel() == 0:
            continue
        best = sim[j, hits].max()
        ranks.append((sim[j] > best).sum().item() + 1)
    if not ranks:
        return None
    return sum(ranks) / len(ranks)


def main():
    ap = argparse.ArgumentParser(description="Stage 1 对比学习（BIT-LLM 复刻）")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--clip", default="openai/clip-vit-large-patch14")
    ap.add_argument("--anatomy_dir", default="data/anatomy_cache")
    ap.add_argument("--out_dir", default="checkpoints/stage1")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=None, help="快速冒烟用：跑够步数就停")
    ap.add_argument("--log_steps", type=int, default=20)
    ap.add_argument("--ckpt_freq", type=int, default=1000, help="每 N 步滚动保存 last.pt（断点续训）")
    ap.add_argument("--resume", default=None, help="从 {out_dir}/last.pt 续训")
    ap.add_argument("--init_encoder", default=None,
                    help="训练前用普通 state_dict（如 stage1_encoder.pt）初始化 encoder+ridge+head——"
                         "定向微调先导用：低 lr + 新目标在现役编码器上验证假设，不重训")
    ap.add_argument("--xsubj_lambda", type=float, default=0.0,
                    help="跨被试对比权重（0=关，baseline 行为不变）。bank 负样本 = 其他被试 head 嵌入")
    ap.add_argument("--bank_size", type=int, default=4096, help="跨被试 FIFO bank 容量")
    ap.add_argument("--xsubj_neg", type=int, default=128, help="每 query 从 bank 采的跨被试硬负样本数")
    ap.add_argument("--xsubj_warmup", type=int, default=500, help="开启跨被试项前的步数（让 bank 攒够跨被试覆盖）")
    ap.add_argument("--exclude_shared1000", action="store_true",
                    help="S1-7 训练只用 unique 图（split=train），剔除 shared1000——"
                         "BIT-LLM 协议：S8 shared1000 成为干净 held-out")
    ap.add_argument("--no_ridge", action="store_true",
                    help="去掉 per-subject ridge：读出头全共享（BIT-LLM 配方第 2 条）。"
                         "论文警告：stage1 用逐被试投影 → 跨被试检索崩，即使被试内正常。"
                         "与 --mixed_batch 同开 = BIT-LLM 跨被试机制全配方")
    ap.add_argument("--mixed_batch", action="store_true",
                    help="混合被试 batch：每个 batch 含全部训练被试（各 batch_size//n_subj 个），"
                         "in-batch InfoNCE 负样本天然跨被试（BIT-LLM Sec 3.4 配方第 1 条，无 bank）。"
                         "单被试模式保持原行为")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} | data={args.data_path} | clip={args.clip}")

    captions_by_nsd_idx = build_caption_map(args.data_path)
    print(f"captions built: {len(captions_by_nsd_idx)} images mapped")

    # 冻结 CLIP
    clip = CLIPModel.from_pretrained(args.clip).to(device).eval()
    for p in clip.parameters():
        p.requires_grad = False
    tokenizer = CLIPTokenizer.from_pretrained(args.clip)

    model = ContrastiveBrainModel(n_subjects=8, anatomy_dir=args.anatomy_dir,
                                  use_ridge=not args.no_ridge).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable: {n_trainable / 1e6:.1f}M (encoder+{'ridge+' if not args.no_ridge else ''}head)")
    if args.init_encoder:
        # strict=False：--no_ridge 模型无 ridge 键，旧的带 ridge ckpt 里有 → 只取 encoder+head
        model.load_state_dict(torch.load(args.init_encoder, map_location=device,
                                         weights_only=True), strict=False)
        print(f"[stage1] init encoder+head (no_ridge={args.no_ridge}) from {args.init_encoder} "
              f"(strict=False, ridge 键按需跳过)")
        if args.xsubj_lambda <= 0:
            print(f"[stage1] WARNING: --init_encoder 但 --xsubj_lambda 0 → 等同普通续训，目标未变")

    tr_splits = ("train",) if args.exclude_shared1000 else ("train", "new_test")
    if args.mixed_batch:
        loader = build_mixed_train_loader(args.data_path, list(range(1, 8)),
                                          captions_by_nsd_idx, args.batch_size,
                                          return_image=True, splits=tr_splits)
        print(f"train loader: {len(loader)} mixed-subject batches "
              f"(每批含全部 7 被试，各 {args.batch_size // 7} trial, splits={tr_splits})")
    else:
        loader = build_train_loader(args.data_path, list(range(1, 8)), captions_by_nsd_idx,
                                    args.batch_size, return_image=True, splits=tr_splits)
        print(f"train loader: {len(loader)} single-subject batches (splits={tr_splits})")

    opt = AdamW(model.parameters(), lr=args.lr)
    loss_fn = BrainContrastiveLoss()
    mean = CLIP_MEAN.to(device)
    std = CLIP_STD.to(device)

    bank = []  # 跨被试 FIFO bank：(emb_cpu, nsd_idx, subj)
    xs_logit_scale = loss_fn.logit_scale

    start_epoch, global_step = 0, 0
    if args.resume:
        start_epoch, global_step = load_train_ckpt(model, opt, args.resume)
        print(f"resumed from {args.resume}: epoch {start_epoch}, step {global_step}")
        # 防御：续训 ridge 架构不匹配（同 stage2 --lora 教训）。last.pt 是否含 ridge.* 键与
        # 本次 --no_ridge 必须一致——no_ridge run 的 last.pt 无 ridge 键，漏传 --no_ridge 会
        # 建随机 ridge 且 strict=False 静默跳过 → 随机 ridge 搅碎特征白测。
        _lk = torch.load(args.resume, map_location="cpu", weights_only=True)
        _st = _lk.get("model", _lk)
        _has_ridge = any(k.startswith("ridge.") for k in _st)
        if _has_ridge and args.no_ridge:
            print("WARNING: last.pt 含 ridge.* 权重但本次 --no_ridge → ridge 被丢弃，"
                  "读出头走共享路径。若原 run 是带 ridge 配方，架构不一致，勿混续。", flush=True)
        elif not _has_ridge and not args.no_ridge:
            print("WARNING: last.pt 无 ridge.* 权重但本次未加 --no_ridge → ridge 随机初始化，"
                  "会把特征搅碎白测。续训 no_ridge run 必须加 --no_ridge。", flush=True)
    elif args.init_encoder and args.xsubj_lambda > 0:
        print(f"[stage1] 定向微调先导：xsubj_lambda={args.xsubj_lambda} lr={args.lr} "
              f"bank={args.bank_size} n_neg={args.xsubj_neg} warmup={args.xsubj_warmup}")

    model.train()
    opt.zero_grad()
    running_loss = 0.0
    t0 = time.time()

    def to_groups(batch):
        """把任意 batch（单被试 or 混合被试）规范成 {subj: 组} 结构。"""
        if "groups" in batch:
            return batch["groups"], batch["subj_list"]
        return {batch["subj"]: batch}, [batch["subj"]]

    for epoch in range(start_epoch, args.epochs):
        for batch in loader:
            groups, subj_list = to_groups(batch)

            # 跨被试拼接图像/caption → 统一 CLIP 前向（embedding 顺序与 brain 拼接一致）
            img_parts = [groups[s]["image"].to(device) for s in subj_list]
            caps = []
            for s in subj_list:
                caps.extend(groups[s]["captions"])
            img_cat = torch.cat(img_parts, 0)          # (B_total, 3, 224, 224)
            nsd_cat = torch.cat([groups[s]["nsd_idx"] for s in subj_list], 0)

            pixel_values = (img_cat.float() - mean) / std
            tokens = tokenizer(caps, padding=True, truncation=True, max_length=77,
                               return_tensors="pt").to(device)

            with torch.no_grad():
                image_emb = F.normalize(clip.get_image_features(pixel_values=pixel_values), dim=-1)
                text_emb = F.normalize(clip.get_text_features(**tokens), dim=-1)

            # brain：按 subj 分组跑 encoder（体素数逐被试不同，无法跨被试堆叠），再拼接
            brain_parts = [model(groups[s]["voxels"].to(device), s) for s in subj_list]
            brain_emb = torch.cat(brain_parts, 0)      # (B_total, 768) L2 归一化
            loss = loss_fn(brain_emb, image_emb, text_emb) / args.grad_accum

            # MoCo bank 跨被试项（已证伪，仅单被试模式保留；混合 batch 的 in-batch 负样本
            # 已天然跨被试，无需 bank，且 bank 项曾诱导"降低体素依赖"逃生通道）
            if args.xsubj_lambda > 0 and global_step >= args.xsubj_warmup and not args.mixed_batch:
                xs = cross_subject_loss(brain_emb, image_emb, nsd_cat,
                                        subj_list[0], bank, args.xsubj_neg,
                                        xs_logit_scale, device)
                if xs is not None:
                    loss = loss + args.xsubj_lambda * xs / args.grad_accum
            loss.backward()

            if args.xsubj_lambda > 0 and not args.mixed_batch:
                push_bank(bank, brain_emb, nsd_cat, subj_list[0], args.bank_size)

            running_loss += loss.item() * args.grad_accum
            if (global_step + 1) % args.grad_accum == 0:
                opt.step()
                opt.zero_grad()

            global_step += 1
            if global_step % args.ckpt_freq == 0:
                save_train_ckpt(model, opt, epoch, global_step,
                                os.path.join(args.out_dir, "last.pt"))
                # 内嵌自诊断（复用本 batch 已算好的嵌入，只加一次 no_grad zero 前向，
                # zero 前向同样按 subj 分组）
                with torch.no_grad():
                    b = brain_emb.detach()
                    r1_bi = batch_topk_retrieval(b @ image_emb.T)
                    r1_bt = batch_topk_retrieval(b @ text_emb.T)
                    z_parts = [model(torch.zeros_like(groups[s]["voxels"]).to(device), s)
                               for s in subj_list]
                    brain_zero = torch.cat(z_parts, 0)
                    cz = F.cosine_similarity(b, brain_zero, dim=-1).mean().item()
                line = (f"  [diag] brain→image R@1={r1_bi:.3f} | brain→text R@1={r1_bt:.3f} "
                        f"| cos(real,zero)={cz:.3f}")
                if args.xsubj_lambda > 0:
                    xs_medr = cross_subject_retrieval_diag(b, nsd_cat, subj_list[0], bank)
                    n_other = len(set(s for _, _, s in bank))
                    line += (f" | bank_subj={n_other} "
                             f"| xs_sameimg_MedR={xs_medr if xs_medr is None else round(xs_medr, 0)}")
                print(line, flush=True)
            if global_step % args.log_steps == 0:
                el = time.time() - t0
                print(f"[epoch {epoch}] step {global_step} | loss {running_loss / args.log_steps:.4f} "
                      f"| {el / 60:.1f}min", flush=True)
                running_loss = 0.0

            if args.max_steps and global_step >= args.max_steps:
                break
        else:
            continue
        break

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    path = os.path.join(args.out_dir, "stage1_encoder.pt")
    torch.save(ckpt, path)
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
