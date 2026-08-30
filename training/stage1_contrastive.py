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

用法（服务器，fmri 环境，项目根目录）：
  python training/stage1_contrastive.py \
      --data_path /root/autodl-tmp/MindEyeV2-main/data
  # 快速冒烟：python training/stage1_contrastive.py --max_steps 200 --epochs 1

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

from data.dataloader import build_train_loader
from data.preprocessing import build_caption_map
from models.fmri_encoder import KeyValueEncoder
from models.ridge import SubjectRidge
from training.losses import BrainContrastiveLoss
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
    def __init__(self, n_subjects=8, anatomy_dir="data/anatomy_cache", out_dim=768):
        super().__init__()
        self.encoder = KeyValueEncoder(anatomy_dir=anatomy_dir)
        self.ridge = SubjectRidge(n_subjects=n_subjects, dim=1024)
        self.head = ContrastiveHead(in_dim=1024, out_dim=out_dim)

    def forward(self, voxels, subj):
        x = self.encoder(voxels, subj)
        x = self.ridge(x, subj)
        return self.head(x)


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
    ap.add_argument("--exclude_shared1000", action="store_true",
                    help="S1-7 训练只用 unique 图（split=train），剔除 shared1000——"
                         "BIT-LLM 协议：S8 shared1000 成为干净 held-out")
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

    model = ContrastiveBrainModel(n_subjects=8, anatomy_dir=args.anatomy_dir).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable: {n_trainable / 1e6:.1f}M (encoder+ridge+head)")

    tr_splits = ("train",) if args.exclude_shared1000 else ("train", "new_test")
    loader = build_train_loader(args.data_path, list(range(1, 8)), captions_by_nsd_idx,
                                args.batch_size, return_image=True, splits=tr_splits)
    print(f"train loader: {len(loader)} single-subject batches (splits={tr_splits})")

    opt = AdamW(model.parameters(), lr=args.lr)
    loss_fn = BrainContrastiveLoss()
    mean = CLIP_MEAN.to(device)
    std = CLIP_STD.to(device)

    start_epoch, global_step = 0, 0
    if args.resume:
        start_epoch, global_step = load_train_ckpt(model, opt, args.resume)
        print(f"resumed from {args.resume}: epoch {start_epoch}, step {global_step}")

    model.train()
    opt.zero_grad()
    running_loss = 0.0
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        for batch in loader:
            voxels = batch["voxels"].to(device)
            subj = batch["subj"]
            images = batch["image"].to(device)
            captions = batch["captions"]

            pixel_values = (images.float() - mean) / std
            tokens = tokenizer(captions, padding=True, truncation=True, max_length=77,
                               return_tensors="pt").to(device)

            with torch.no_grad():
                image_emb = F.normalize(clip.get_image_features(pixel_values=pixel_values), dim=-1)
                text_emb = F.normalize(clip.get_text_features(**tokens), dim=-1)

            brain_emb = model(voxels, subj)
            loss = loss_fn(brain_emb, image_emb, text_emb) / args.grad_accum
            loss.backward()

            running_loss += loss.item() * args.grad_accum
            if (global_step + 1) % args.grad_accum == 0:
                opt.step()
                opt.zero_grad()

            global_step += 1
            if global_step % args.ckpt_freq == 0:
                save_train_ckpt(model, opt, epoch, global_step,
                                os.path.join(args.out_dir, "last.pt"))
                # 内嵌自诊断（复用本 batch 已算好的嵌入，只加一次 no_grad zero 前向）
                with torch.no_grad():
                    b = brain_emb.detach()
                    r1_bi = batch_topk_retrieval(b @ image_emb.T)
                    r1_bt = batch_topk_retrieval(b @ text_emb.T)
                    brain_zero = model(torch.zeros_like(voxels), subj)
                    cz = F.cosine_similarity(b, brain_zero, dim=-1).mean().item()
                print(f"  [diag] brain→image R@1={r1_bi:.3f} | brain→text R@1={r1_bt:.3f} "
                      f"| cos(real,zero)={cz:.3f}", flush=True)
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
