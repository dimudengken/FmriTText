"""冻结编码器 + 只训 ContrastiveHead（匹配 scratch-head 双 cell，MLP-bypass 去混淆实验）。

回答 eval_subject_alignment v3 + bypass 消融留下的悬置：bypass 去掉共享 4 层残差 MLP 后 S8 的
CLIP 锚 / 逐图信号恢复、但检索持平（cross MedR 247→255、brain→CLIP MedR 301→283）——是冻结
离域 head（原 head 训在 mlp_out 几何，post≈正交）把 post 的可判别性封顶。这里给 head 公平机会：
在指定几何（bypass: head 吃 post / normal: head 吃 mlp_out）上用 stage1 同款配方重训 scratch
ContrastiveHead（编码器冻结、只训 head），再由 eval/eval_subject_alignment.py --probe_head 换入
测 S8 检索是否脱离 250~300。

关键 = 匹配双 cell：control（normal 几何，--tag mlpout）与 treatment（bypass 几何，--tag post）
用同一配方（同 loss / 同 loader / 同步数，唯一差异 = encoder.bypass_mlp 开关）→ 差异即隔离 MLP。

head 配方 = stage1 原样（镜像 stage1_contrastive.py mixed-batch 循环，剥离 encoder 梯度）：
  loader = build_mixed_train_loader(splits=("train",))：混合被试 batch → in-batch 跨被试 InfoNCE
    （no_ridge stage1 原 head 的训练分布；同图跨被试正样本被直接拉齐 = 恰是我们要测的 cross MedR；
    attnpool 的单被试协议为 S1 被试内设计，S8 跨被试不沿用）。splits=("train",) 对齐
    --exclude_shared1000 → shared1000(new_test) 评估全程干净。
  loss = BrainContrastiveLoss()（info_nce(brain,img) + info_nce(brain,txt)）作用于全混合 batch。
  head 输入 = 冻结 encoder(voxels, subj) 的 (B,128,1024) token（ContrastiveHead 内部 mean-pool），
    与 eval_subject_alignment 的 enc/head 位置同构。
  无 ridge（no_ridge 配方）；encoder 全程 no_grad；仅 scratch head ~0.79M 参数可训。

用法（服务器，fmri 环境，项目根目录；CLIP 本地路径 + HF_HUB_OFFLINE=1）：
  # control（normal 几何，head 输入 = mlp_out）
  HF_HUB_OFFLINE=1 python training/train_head_frozen.py \
      --data_path /root/autodl-tmp/MindEyeV2-main/data \
      --stage1 checkpoints/stage1/stage1_e84_noridge.pt \
      --clip /root/autodl-tmp/models/clip-vit-large-patch14 \
      --tag mlpout --epochs 3 --ckpt_freq 500
  # treatment（bypass 几何，head 输入 = post）
  HF_HUB_OFFLINE=1 python training/train_head_frozen.py \
      --data_path /root/autodl-tmp/MindEyeV2-main/data \
      --stage1 checkpoints/stage1/stage1_e84_noridge.pt \
      --clip /root/autodl-tmp/models/clip-vit-large-patch14 \
      --tag post --epochs 3 --ckpt_freq 500 --bypass_mlp
  # 冒烟：加 --max_steps 200 --epochs 1

架构改造（2026-09-08，见 CLAUDE.md"readout Subject-Robust Adapter"节）：post 几何上在 encoder 与
head 之间插 SubjectRobustAdapter（--adapter：token-wise norm→gated linear→residual，sigmoid gate
init 0.1 ≈ 近恒等起步），并可加 GRL 主体探针（--adv_lambda >0：pooled 适配 token 上线性主体分类器 +
梯度反转）显式去主体——回答 bypass 双 cell 的残余：S8 零样本 clip MedR 能否压过 treatment 175、
subject_probe(head) 能否 <0.442（SIFE 从 encoder 输出挪到 post-readout adapter，位置与证据不同）。
默认全关 = 上面 control/treatment 双 cell 原样（复现不受影响）。判据见 CLAUDE.md。
  # mode 2 cell A（post 几何 + adapter，--adv_lambda 0 = 隔离 adapter 结构本身）
  HF_HUB_OFFLINE=1 python training/train_head_frozen.py \
      --data_path ... --stage1 checkpoints/stage1/stage1_e84_noridge.pt \
      --clip ... --tag robust_noadv --bypass_mlp --adapter --epochs 3
  # mode 2 cell B（post 几何 + adapter + GRL 去主体，主 cell）
  HF_HUB_OFFLINE=1 python training/train_head_frozen.py \
      --data_path ... --stage1 checkpoints/stage1/stage1_e84_noridge.pt \
      --clip ... --tag robust_adv --bypass_mlp --adapter \
      --adv_lambda 1.0 --grl_alpha 1.0 --epochs 3
输出：{out_dir}/head_{tag}.pt（ContrastiveHead）+（--adapter 时）{out_dir}/adapter_{tag}.pt
      （SubjectRobustAdapter state_dict。评估：eval_subject_alignment.py --probe_head head_{tag}.pt
      --adapter adapter_{tag}.pt --bypass_mlp）
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import CLIPModel, CLIPTokenizer

from data.dataloader import build_mixed_train_loader, build_train_loader
from data.preprocessing import build_caption_map
from models.fmri_encoder import KeyValueEncoder
from models.subject_adapter import SubjectRobustAdapter, SubjectProbe
from training.losses import BrainContrastiveLoss
from training.stage1_contrastive import ContrastiveHead

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def load_frozen_encoder(ckpt_path, device, anatomy_dir):
    """no_ridge stage1 ckpt → 冻结 KeyValueEncoder（不经 ridge，不需要原 head）。"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    encoder = KeyValueEncoder(anatomy_dir=anatomy_dir).to(device).eval()
    enc = {k.replace("encoder.", ""): v for k, v in ckpt.items() if k.startswith("encoder.")}
    encoder.load_state_dict(enc, strict=False)
    print(f"[head_frozen] encoder {len(enc)} tensors loaded from {ckpt_path}")
    if len(enc) == 0:
        raise RuntimeError(f"stage1 ckpt 加载 0 张量（enc {len(enc)}），检查键前缀 encoder.*")
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


def batch_topk_retrieval(sim, k=1):
    """sim (B, B)：第 i 行真值在第 i 列 → 命中率。"""
    topk = torch.topk(sim, k=k, dim=1).indices
    hit = (topk == torch.arange(sim.size(0), device=sim.device).unsqueeze(1)).any(dim=1)
    return hit.float().mean().item()


def to_groups(batch):
    """任意 batch（单被试 or 混合）规范成 {subj: 组} 结构（镜像 stage1_contrastive）。"""
    if "groups" in batch:
        return batch["groups"], batch["subj_list"]
    return {batch["subj"]: batch}, [batch["subj"]]


def main():
    ap = argparse.ArgumentParser(description="冻结编码器只训 ContrastiveHead（匹配双 cell 消融）")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--stage1", default="checkpoints/stage1/stage1_encoder.pt")
    ap.add_argument("--clip", default="openai/clip-vit-large-patch14")
    ap.add_argument("--anatomy_dir", default="data/anatomy_cache")
    ap.add_argument("--out_dir", default="checkpoints/head_frozen")
    ap.add_argument("--tag", required=True,
                    help="几何标签（control=mlpout / treatment=post），存 head_{tag}.pt")
    ap.add_argument("--loader", default="mixed", choices=["mixed", "single"],
                    help="mixed=混合被试 batch（stage1 原 head 分布，默认，跨被试用）；"
                         "single=单被试 batch（为 S1 被试内设计的 attnpool 协议，跨被试不沿用）")
    ap.add_argument("--bypass_mlp", action="store_true",
                    help="跳过 encoder 共享 4 层残差 MLP：head 输入从 mlp_out 挪到 post（readout 经 "
                         "LN+GELU 后）。control 不传、treatment 传——唯一差异即隔离 MLP")
    ap.add_argument("--adapter", action="store_true",
                    help="在 encoder 输出与 head 之间插 SubjectRobustAdapter（token-wise 门控适配，"
                         "配合 --bypass_mlp 用 post 几何；默认关 = control/treatment 双 cell 原样）")
    ap.add_argument("--adv_lambda", type=float, default=0.0,
                    help=">0：pooled 适配 token 上 GRL 主体探针 CE 的权重（显式去主体）；须配 --adapter")
    ap.add_argument("--grl_alpha", type=float, default=1.0,
                    help="SubjectProbe 内 GRL 的梯度反转强度（DANN 式）")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=None, help="冒烟：跑够步数就停")
    ap.add_argument("--log_steps", type=int, default=20)
    ap.add_argument("--ckpt_freq", type=int, default=500,
                    help="每 N 步滚存 head_{tag}_last.pt（断点防护）；结束必存 head_{tag}.pt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} | tag={args.tag} | loader={args.loader} | bypass_mlp={args.bypass_mlp}")

    encoder = load_frozen_encoder(args.stage1, device, args.anatomy_dir)
    if args.bypass_mlp:
        encoder.bypass_mlp = True
        print("[head_frozen] ** MLP bypass 开启：head 输入 = post（readout 经 LN+GELU+Dropout 后）")
    head = ContrastiveHead().to(device)
    print(f"[head_frozen] encoder frozen; scratch ContrastiveHead "
          f"{sum(p.numel() for p in head.parameters()) / 1e3:.0f}K params")
    head.train()

    clip = CLIPModel.from_pretrained(args.clip).to(device).eval()
    for p in clip.parameters():
        p.requires_grad = False
    tokenizer = CLIPTokenizer.from_pretrained(args.clip)
    mean, std = CLIP_MEAN.to(device), CLIP_STD.to(device)

    captions_by_nsd_idx = build_caption_map(args.data_path)
    # splits=("train",) = 对齐 stage1 --exclude_shared1000（只训 unique 图，shared1000 评估干净）
    subj_list = list(range(1, 8))
    if args.loader == "mixed":
        loader = build_mixed_train_loader(args.data_path, subj_list, captions_by_nsd_idx,
                                          args.batch_size, return_image=True, seed=args.seed,
                                          splits=("train",))
        print(f"[head_frozen] train loader: {len(loader)} mixed-subject batches "
              f"(每批含全部 7 被试各 {args.batch_size // 7}, splits=train unique-only)")
    else:
        loader = build_train_loader(args.data_path, subj_list, captions_by_nsd_idx,
                                    args.batch_size, return_image=True, seed=args.seed,
                                    splits=("train",))
        print(f"[head_frozen] train loader: {len(loader)} single-subject batches "
              f"(splits=train unique-only)")

    adapter = probe = None
    if args.adapter:
        adapter = SubjectRobustAdapter().to(device)
        print(f"[head_frozen] ** SubjectRobustAdapter 开启：head 输入 = 适配 token "
              f"（norm→gated linear→residual，gate sigmoid init 0.1 近恒等起步）")
    if args.adv_lambda > 0:
        if not args.adapter:
            raise RuntimeError("--adv_lambda >0 必须配 --adapter（GRL 探针作用在适配 token 上）")
        probe = SubjectProbe(n_subjects=len(subj_list), grl_alpha=args.grl_alpha).to(device)
        print(f"[head_frozen] ** GRL 主体探针：n_subjects={len(subj_list)} "
              f"adv_lambda={args.adv_lambda} grl_alpha={args.grl_alpha}")

    params = list(head.parameters())
    if adapter is not None:
        params += list(adapter.parameters())
    if probe is not None:
        params += list(probe.parameters())
    opt = AdamW(params, lr=args.lr)
    loss_fn = BrainContrastiveLoss()
    os.makedirs(args.out_dir, exist_ok=True)

    global_step, run_loss, run_r1 = 0, 0.0, 0.0
    t0 = time.time()
    opt.zero_grad()
    for epoch in range(args.epochs):
        for batch in loader:
            groups, slist = to_groups(batch)
            # encoder + CLIP 全程冻结，no_grad 里算；head 单独在外吃梯度
            with torch.no_grad():
                px = (torch.cat([groups[s]["image"].to(device) for s in slist], 0).float() - mean) / std
                caps = []
                for s in slist:
                    caps.extend(groups[s]["captions"])
                tok = tokenizer(caps, padding=True, truncation=True, max_length=77,
                                return_tensors="pt").to(device)
                image_emb = F.normalize(clip.get_image_features(pixel_values=px), dim=-1)
                text_emb = F.normalize(clip.get_text_features(**tok), dim=-1)
                toks = [encoder(groups[s]["voxels"].to(device), s) for s in slist]
            cat_toks = torch.cat(toks, 0)                   # (B_total, 128, 1024)
            if adapter is not None:
                cat_toks = adapter(cat_toks)                # token-wise 门控适配（跨被试共享）
            brain_emb = head(cat_toks)                      # (B_total, 768) L2 归一化
            loss = loss_fn(brain_emb, image_emb, text_emb) / args.grad_accum
            if probe is not None:
                # pooled 适配 token 上的 GRL 主体探针：CE 经梯度反转 → 显式去主体
                subj_ids = torch.cat([torch.full((groups[s]["voxels"].size(0),), s,
                                                 dtype=torch.long, device=device)
                                      for s in slist])
                z_pool = F.normalize(cat_toks.mean(dim=1), dim=-1)
                loss = loss + args.adv_lambda * probe(z_pool, subj_ids) / args.grad_accum
            loss.backward()
            run_loss += loss.item() * args.grad_accum
            if (global_step + 1) % args.grad_accum == 0:
                opt.step()
                opt.zero_grad()
            global_step += 1
            with torch.no_grad():
                run_r1 += batch_topk_retrieval(brain_emb @ image_emb.T)
            if global_step % args.log_steps == 0:
                print(f"[epoch {epoch}] step {global_step} | loss {run_loss / args.log_steps:.3f} "
                      f"| brain↔img R@1 {run_r1 / args.log_steps:.3f} | "
                      f"{(time.time() - t0) / 60:.1f}min", flush=True)
                run_loss, run_r1 = 0.0, 0.0
            if args.ckpt_freq and global_step % args.ckpt_freq == 0:
                torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()},
                           os.path.join(args.out_dir, f"head_{args.tag}_last.pt"))
                print(f"[head_frozen] saved rolling head_{args.tag}_last.pt @ step {global_step}")
                if adapter is not None:
                    torch.save({k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                               os.path.join(args.out_dir, f"adapter_{args.tag}_last.pt"))
            if args.max_steps and global_step >= args.max_steps:
                break
        if args.max_steps and global_step >= args.max_steps:
            break

    head.eval()
    rpath = os.path.join(args.out_dir, f"head_{args.tag}.pt")
    torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, rpath)
    print(f"[head_frozen] saved: {rpath} ({len(head.state_dict())} tensors)")
    apath = None
    if adapter is not None:
        apath = os.path.join(args.out_dir, f"adapter_{args.tag}.pt")
        torch.save({k: v.detach().cpu() for k, v in adapter.state_dict().items()}, apath)
        print(f"[head_frozen] saved: {apath} (SubjectRobustAdapter, {len(adapter.state_dict())} tensors)")
    eval_extra = f"--adapter {apath} " if apath else ""
    eval_extra += "--bypass_mlp " if args.bypass_mlp else ""
    print(f"[head_frozen] 评估：eval/eval_subject_alignment.py --probe_head {rpath} "
          f"{eval_extra}(同几何) → 对比 control/treatment S8 检索")


if __name__ == "__main__":
    main()
