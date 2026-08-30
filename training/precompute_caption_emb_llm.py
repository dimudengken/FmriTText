"""预计算每张 NSD 图像的 caption 的冻结 LLM 平均嵌入（Stage 2 辅助对齐损失用）。

背景：CLIP 空间对齐的 aux 与冻结 LLM 的 CE 目标冲突——CLIP 对齐的 brain tokens
对冻结 Llama 是 OOD 输入，CE(real) 被拉到 3.9 > 冻结先验 3.0（step 29990 实测）。
aux 的目标空间必须换成 LLM 自己的语义空间：用冻结 LLM 的 token embedding 对
caption 做 mean-pool 得锚点，brain 汇总在该空间对齐 = CE 天然会降，两个目标同向。

只取嵌入表（不 forward 全模型），bf16 下 ~0.8GB，比 CLIP 版更快。

输出：{data_path}/caption_emb_mean_llm.pt
      float32 (73000, 3072)，L2 归一化；row i = 第 i 张 NSD 图像的 5 个 caption
      在 Llama 嵌入空间里的平均向量。

用法（服务器 fmri 环境）：
  python training/precompute_caption_emb_llm.py \
      --data_path /root/autodl-tmp/MindEyeV2-main/data \
      --llm /root/autodl-tmp/models/Llama-3.2-3B-Instruct
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.preprocessing import build_caption_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--llm", default="/root/autodl-tmp/models/Llama-3.2-3B-Instruct")
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()
    device = "cuda"

    captions_by_nsd_idx = build_caption_map(args.data_path)
    n = len(captions_by_nsd_idx)
    # caption 字符串 → 出现的 nsd_idx 列表（COCO caption 一般只对应一张图）
    cap_to_idx = {}
    for idx, caps in enumerate(captions_by_nsd_idx):
        for c in caps:
            cap_to_idx.setdefault(c, []).append(idx)
    unique_caps = list(cap_to_idx)
    print(f"images: {n} | unique captions: {len(unique_caps)}")

    tok = AutoTokenizer.from_pretrained(args.llm)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained(args.llm, torch_dtype=torch.bfloat16).to(device).eval()
    emb = llm.get_input_embeddings().weight  # (vocab, 3072) bf16
    del llm
    torch.cuda.empty_cache()
    print(f"embedding table: {tuple(emb.shape)}")

    out = torch.zeros(n, emb.size(1))
    counts = torch.zeros(n)
    with torch.no_grad():
        for start in range(0, len(unique_caps), args.batch_size):
            batch_caps = unique_caps[start:start + args.batch_size]
            toks = tok([c + tok.eos_token for c in batch_caps], padding=True,
                       truncation=True, max_length=77, add_special_tokens=False,
                       return_tensors="pt").to(device)
            ids = toks["input_ids"]                       # (B, L)
            mask = (ids != tok.pad_token_id).float()      # (B, L)
            e = emb[ids]                                  # (B, L, 3072)
            pooled = (e * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            pooled = F.normalize(pooled, dim=-1).float().cpu()
            for i, cap in enumerate(batch_caps):
                for idx in cap_to_idx[cap]:
                    out[idx] += pooled[i]
                    counts[idx] += 1
            if start % (args.batch_size * 50) == 0:
                print(f"  {start}/{len(unique_caps)}", flush=True)

    mask = counts > 0
    out[mask] /= counts[mask].unsqueeze(1)
    out[mask] = F.normalize(out[mask], dim=-1)
    path = os.path.join(args.data_path, "caption_emb_mean_llm.pt")
    torch.save(out.float(), path)
    print(f"saved: {path} ({out.shape}, {mask.sum().item()} images have captions)")


if __name__ == "__main__":
    main()
