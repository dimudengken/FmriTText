"""Stage 1 诊断脚本：一次真实 batch 上验证三条关键链路。

问题背景：stage1 对比 loss 钉在随机水平下方 ~0.1 处不下降，LR(1e-4/1e-3) 与
负样本数(64/128) 均无效。此脚本在真实数据上一次性回答：
1. 梯度是否真的到达 encoder？（backward 后各模块 grad norm）
2. encoder 输出是否对体素敏感？（Real vs Shuffle / Noise / Zero 余弦）
3. loss 是否随 voxel 输入变化？（loss(real) vs loss(noise)）
4. batch 内有无对齐信号？（brain→image / brain→text 检索 R@1，正负 logit 差）

用法（服务器 fmri 环境）：
  python training/diag_stage1.py --data_path /root/autodl-tmp/MindEyeV2-main/data \
      --resume checkpoints/stage1/last.pt   # 不传则 fresh init
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
from training.losses import BrainContrastiveLoss
from training.stage1_contrastive import ContrastiveBrainModel, CLIP_MEAN, CLIP_STD


def topk_retrieval(sim, k=1):
    """sim (B, B)：第 i 行里真值在第 i 列。返回命中率。"""
    topk = torch.topk(sim, k=k, dim=1).indices
    hit = (topk == torch.arange(sim.size(0), device=sim.device).unsqueeze(1)).any(dim=1)
    return hit.float().mean().item()


def safe_cos(a, b):
    na = a.norm(dim=-1)
    nb = b.norm(dim=-1)
    if ((na > 1e-6) & (nb > 1e-6)).all():
        return F.cosine_similarity(a, b, dim=-1).mean().item()
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--clip", default="/root/autodl-tmp/models/clip-vit-large-patch14")
    ap.add_argument("--anatomy_dir", default="data/anatomy_cache")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--resume", default=None, help="last.pt 或 stage1_encoder.pt；缺省 fresh init")
    args = ap.parse_args()

    device = "cuda"
    clip = CLIPModel.from_pretrained(args.clip).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(args.clip)

    model = ContrastiveBrainModel(n_subjects=8, anatomy_dir=args.anatomy_dir).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        state = ckpt.get("model", ckpt)  # last.pt 有 "model" 键；stage1_encoder.pt 直接是状态字典
        state = {k: v for k, v in state.items()
                 if k.startswith(("encoder.", "ridge.", "head."))}
        model.load_state_dict(state, strict=False)
        print(f"model: loaded from {args.resume} (trainable {n_train / 1e6:.1f}M)")
    else:
        print(f"model: fresh random init (trainable {n_train / 1e6:.1f}M)")
    model.eval()

    caps = build_caption_map(args.data_path)
    loader = build_train_loader(args.data_path, list(range(1, 8)), caps,
                                args.batch_size, return_image=True)
    batch = next(iter(loader))
    voxels = batch["voxels"].to(device).requires_grad_(True)
    subj = batch["subj"]
    images = batch["image"].to(device)
    captions = batch["captions"]
    print(f"batch: subj={subj} voxels={tuple(voxels.shape)} images={tuple(images.shape)}")

    with torch.no_grad():
        pv = (images.float() - CLIP_MEAN.to(device)) / CLIP_STD.to(device)
        image_emb = F.normalize(clip.get_image_features(pixel_values=pv), dim=-1)
        tokens = tokenizer(captions, padding=True, truncation=True,
                           max_length=77, return_tensors="pt").to(device)
        text_emb = F.normalize(clip.get_text_features(**tokens), dim=-1)
    print(f"image_emb norm={image_emb.norm(dim=-1).mean().item():.4f} "
          f"| text_emb norm={text_emb.norm(dim=-1).mean().item():.4f}")
    sim_ii = (image_emb @ image_emb.T).mean().item()
    sim_tt = (text_emb @ text_emb.T).mean().item()
    sim_it = (image_emb @ text_emb.T).mean().item()
    print(f"CLIP 内部: image~image={sim_ii:.4f} text~text={sim_tt:.4f} image~text={sim_it:.4f}")
    print(f"CLIP image→image 检索 R@1={topk_retrieval(image_emb @ image_emb.T):.3f}（≈1.0 才正常）")

    loss_fn = BrainContrastiveLoss()
    brain_real = model(voxels, subj)
    loss_real = loss_fn(brain_real, image_emb, text_emb)
    rnd = torch.log(torch.tensor(float(args.batch_size))).item()
    print(f"\nloss(real)   = {loss_real.item():.4f}  (random={rnd:.4f})")

    noise = torch.randn_like(voxels) * voxels.std(dim=0, keepdim=True) + voxels.mean(dim=0, keepdim=True)
    shuffled = torch.stack([v[torch.randperm(v.numel())] for v in voxels])
    with torch.no_grad():
        brain_noise = model(noise, subj)
        brain_shuf = model(shuffled, subj)
        brain_zero = model(torch.zeros_like(voxels), subj)
        loss_noise = loss_fn(brain_noise, image_emb, text_emb)
    print(f"loss(noise)  = {loss_noise.item():.4f}  (|Δ|={abs(loss_noise.item() - loss_real.item()):.4f})")
    print(f"cos(real, shuffle)={safe_cos(brain_real, brain_shuf):.4f}  "
          f"cos(real, noise)={safe_cos(brain_real, brain_noise):.4f}  "
          f"cos(real, zero)={safe_cos(brain_real, brain_zero):.4f}")
    print("  三个 cos 若≈1.0 → encoder 输出忽略 voxel 内容；≈0 → 依赖 voxel 值")

    with torch.no_grad():
        sim_bi = brain_real @ image_emb.T
        sim_bt = brain_real @ text_emb.T
        r1_bi = topk_retrieval(sim_bi)
        r1_bt = topk_retrieval(sim_bt)
        pos_bi = torch.diag(sim_bi).mean().item()
        neg_bi = (sim_bi - torch.eye(sim_bi.size(0), device=device) * 1e9).max(dim=1).values.mean().item()
    print(f"brain→image R@1={r1_bi:.3f}（chance≈{1.0 / args.batch_size:.3f}）| "
          f"brain→text  R@1={r1_bt:.3f}")
    print(f"正样本 logit={pos_bi:.4f} vs 负样本 top={neg_bi:.4f}（差距应>2 才叫有对齐）")

    loss_real.backward()
    print("\n梯度 norm（loss.backward() 后）：")
    groups = {"encoder.neuro_informed_attn": 0.0, "encoder.region_feature_project": 0.0,
              "encoder.mlp": 0.0, "encoder.head": 0.0, "ridge": 0.0, "head": 0.0}
    for name, p in model.named_parameters():
        if p.grad is not None:
            for g in groups:
                if name.startswith(g):
                    groups[g] += p.grad.norm().item() ** 2
    for g, v in groups.items():
        print(f"  {g:34s} grad_norm={v ** 0.5:.4e}")
    print(f"  voxels.grad                          norm="
          f"{voxels.grad.norm().item() if voxels.grad is not None else 'None'}")


if __name__ == "__main__":
    main()
