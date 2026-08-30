"""预计算每张 NSD 图像的 CLIP 图像嵌入（Stage2 方案 A 的 semclip 损失目标用）。

背景：方案 A 用 sem_head(projector pooled) 预测"看图时的 CLIP 语义摘要"，对真 CLIP
图像嵌入做 MSE。CLIP 只在预计算时用，训练时不占显存。按 nsd_idx 索引，训练时
`img_emb[batch["nsd_idx"]]` 取目标（同 caption 嵌入模式）。

输出：{data_path}/image_emb_mean_clip.pt
      float32 (73000, 768)，L2 归一化；row i = 第 i 张 NSD 图像（nsdId 0~72999）的 CLIP 嵌入。

用法（服务器 fmri 环境，离线）：
  HF_HUB_OFFLINE=1 python training/precompute_image_emb.py \
      --data_path /root/autodl-tmp/MindEyeV2-main/data \
      --clip /root/autodl-tmp/models/clip-vit-large-patch14
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import CLIPModel

from data.preprocessing import load_images_handle

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--clip", default="/root/autodl-tmp/models/clip-vit-large-patch14")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--out", default=None, help="输出 .pt 路径（缺省 {data_path}/image_emb_mean_clip.pt）")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    images = load_images_handle(args.data_path)["images"]
    n = images.shape[0]
    print(f"images: {n} x {images.shape[1:]} (float16 [0,1]，勿再 /255)")

    clip = CLIPModel.from_pretrained(args.clip).to(device).eval()
    mean, std = CLIP_MEAN.to(device), CLIP_STD.to(device)

    out = torch.zeros(n, clip.config.projection_dim)
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            imgs = torch.from_numpy(images[start:start + args.batch_size].astype("float32")).to(device)
            px = (imgs - mean) / std
            emb = F.normalize(clip.get_image_features(pixel_values=px), dim=-1).float().cpu()
            out[start:start + emb.size(0)] = emb
            if start % (args.batch_size * 50) == 0:
                print(f"  {start}/{n}", flush=True)

    path = args.out or os.path.join(args.data_path, "image_emb_mean_clip.pt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(out, path)
    print(f"saved: {path} ({out.shape})")


if __name__ == "__main__":
    main()
