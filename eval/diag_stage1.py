"""诊断 Stage 1 编码器在训练被试（S1-7）上的真实质量。

与 eval/run_eval.py 的区别：这里用**训练被试 + 训练过的 ridge**，无跨被试 confound，
直接回答"Stage 1 是否把编码器练成对体素敏感、语义可检索"。

三个指标（batch 内，多 batch 聚合）：
- brain→image R@1 / brain→text R@1：语义检索（COCO caption 先验下，R@1 高=编码器真在编码图像内容）
- cos(real, zero)：实值 vs 全零体素的 head 特征余弦（目标接近 0 = 体素敏感）

判读：cos>0.6 = 结构主导、体素敏感性差 → Stage 1 未修复 MindLLM 缺陷（BIT-LLM 修到 -0.008）；
cos<0.2 = 编码器体素敏感 → 问题在下游/跨被试，Stage 1 无罪。

用法（服务器 fmri 环境）：
  python eval/diag_stage1.py \
      --data_path /root/autodl-tmp/MindEyeV2-main/data \
      --clip /root/autodl-tmp/models/clip-vit-large-patch14 \
      --stage1 checkpoints/stage1/stage1_encoder.pt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

from data.dataloader import build_train_loader
from data.preprocessing import build_caption_map
from training.stage1_contrastive import ContrastiveBrainModel, batch_topk_retrieval

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def main():
    ap = argparse.ArgumentParser(description="Stage 1 编码器训练被试诊断")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--clip", default="openai/clip-vit-large-patch14")
    ap.add_argument("--stage1", default="checkpoints/stage1/stage1_encoder.pt")
    ap.add_argument("--subjects", type=int, nargs="*", default=[1, 2, 5, 7])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--batches", type=int, default=32)
    ap.add_argument("--anatomy_dir", default="data/anatomy_cache")
    args = ap.parse_args()
    device = "cuda"

    model = ContrastiveBrainModel(n_subjects=8, anatomy_dir=args.anatomy_dir).to(device).eval()
    ckpt = torch.load(args.stage1, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):  # 嵌套训练 ckpt（last.pt）
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt, strict=False)
    print(f"loaded stage1: {len(ckpt)} tensors (encoder+ridge+head)")

    clip = CLIPModel.from_pretrained(args.clip).to(device).eval()
    tok = CLIPTokenizer.from_pretrained(args.clip)
    captions = build_caption_map(args.data_path)
    loader = build_train_loader(args.data_path, args.subjects, captions,
                                args.batch_size, return_image=True)
    print(f"subjects={args.subjects} | {len(loader)} batches available")

    mean, std = CLIP_MEAN.to(device), CLIP_STD.to(device)
    r1_i = r1_t = cz = n = 0.0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.batches:
                break
            v = batch["voxels"].to(device)
            s = batch["subj"]
            px = (batch["image"].to(device).float() - mean) / std
            toks = tok(batch["captions"], padding=True, truncation=True, max_length=77,
                       return_tensors="pt").to(device)
            im = F.normalize(clip.get_image_features(pixel_values=px), dim=-1)
            tx = F.normalize(clip.get_text_features(**toks), dim=-1)
            b = model(v, s)
            z = model(torch.zeros_like(v), s)
            bs = b.size(0)
            r1_i += batch_topk_retrieval(b @ im.T) * bs
            r1_t += batch_topk_retrieval(b @ tx.T) * bs
            cz += F.cosine_similarity(b, z, dim=-1).mean().item() * bs
            n += bs
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{min(len(loader), args.batches)} batches", flush=True)

    r1i, r1t, c = r1_i / n, r1_t / n, cz / n
    print(f"\n[stage1 diag] brain→image R@1={r1i:.4f} | brain→text R@1={r1t:.4f} "
          f"| cos(real,zero)={c:.4f} (n={int(n)})")
    if c > 0.6:
        print("VERDICT: cos>0.6 = 编码器结构主导、体素敏感性差 → Stage 1 未修复 MindLLM 缺陷，根因在编码器")
    elif c < 0.2:
        print("VERDICT: cos<0.2 = 编码器体素敏感 → 根因不在 Stage 1，在下游/跨被试 ridge")
    else:
        print(f"VERDICT: 中间态，结合 R@1 看（brain→image R@1 高=语义在编码，低=弱）")


if __name__ == "__main__":
    main()
