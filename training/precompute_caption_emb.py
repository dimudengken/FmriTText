"""预计算每张 NSD 图像的 caption CLIP 平均嵌入（Stage 2 辅助对齐损失用）。

背景：Stage 2 的 CE-through-LLM 信号太弱，接口 bootstrap 不动。辅助损失直接监督
projector 输出的语义汇总对齐 caption 语义。CLIP 只在预计算时用，训练时不占显存。

输出：{data_path}/caption_emb_mean_clip.pt
      float32 (73000, 768)，L2 归一化；row i = 第 i 张 NSD 图像的 5 个 caption 平均嵌入。

用法（服务器 fmri 环境）：
  python training/precompute_caption_emb.py \
      --data_path /root/autodl-tmp/MindEyeV2-main/data \
      --clip /root/autodl-tmp/models/clip-vit-large-patch14
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

from data.preprocessing import build_caption_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--clip", default="/root/autodl-tmp/models/clip-vit-large-patch14")
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()
    device = "cuda"

    captions_by_nsd_idx = build_caption_map(args.data_path)
    n = len(captions_by_nsd_idx)
    print(f"images: {n}")

    # caption 字符串 → 出现的 nsd_idx 列表（COCO caption 一般只对应一张图）
    cap_to_idx = {}
    for idx, caps in enumerate(captions_by_nsd_idx):
        for c in caps:
            cap_to_idx.setdefault(c, []).append(idx)
    unique_caps = list(cap_to_idx)
    print(f"unique captions: {len(unique_caps)}")

    clip = CLIPModel.from_pretrained(args.clip).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(args.clip)

    out = torch.zeros(n, 768)
    counts = torch.zeros(n)
    with torch.no_grad():
        for start in range(0, len(unique_caps), args.batch_size):
            batch_caps = unique_caps[start:start + args.batch_size]
            toks = tokenizer(batch_caps, padding=True, truncation=True,
                             max_length=77, return_tensors="pt").to(device)
            emb = F.normalize(clip.get_text_features(**toks), dim=-1).float().cpu()
            for i, cap in enumerate(batch_caps):
                idxs = cap_to_idx[cap]
                out[idxs] += emb[i]
                counts[idxs] += 1
            if start % (args.batch_size * 50) == 0:
                print(f"  {start}/{len(unique_caps)}", flush=True)

    mask = counts > 0
    out[mask] /= counts[mask].unsqueeze(1)
    out[mask] = F.normalize(out[mask], dim=-1)
    path = os.path.join(args.data_path, "caption_emb_mean_clip.pt")
    torch.save(out.float(), path)
    print(f"saved: {path} ({out.shape}, {mask.sum().item()} images have captions)")


if __name__ == "__main__":
    main()
