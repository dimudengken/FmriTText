"""hard-negative 先导（修正版）：冻结 encoder+ridge，微调原始 ContrastiveHead，只扩 text 分支负样本。

问题（CLAUDE.md 2026-08-30）：S1 被试内 stage1 封闭集文本上界 CIDEr 15.3 ≈ stage2 生成 15.05
（×100 尺度）→ 接口已榨干编码器。剩的问题是：15.3 是"编码器特征信息墙"还是"stage1 in-batch
对比没榨出细粒度"。

前一轮失败（独立 text_head + text-only hard-neg，2026-08-30）：全指标退步 + image 检索崩（MedR 43→575）
→ 退化解，text-only 训练丢掉图像语义锚点。本版按用户要求消除 confound：
  - **复用原始 ContrastiveHead**（从 stage1 已收敛权重微调，不新建/不随机初始化）。
  - **双项 loss 与 stage1 完全同构**：loss = 0.5·info_nce(brain, image) + 0.5·info_nce(brain, text)。
  - **image 分支不动**（batch-local 负样本，stage1 原样）→ 强锚点锁死图像-CLIP 对齐。
  - **只改 text 分支**：负样本池 = batch 内（原样）+ 全局预缓存 text hard-neg bank（top-N，排除同图 caption）。

判据（S1 留出，与 15.3 同口径，eval 同脚本对比微调前后 head；CIDEr 输出 ×100）：
  - 抬升（caption R@1/top1_img_acc 翻倍+、上界 CIDEr >25、image MedR 保持 ~43 不崩）→ 特征里有细粒度，
    stage1 没榨出来 → 回头修接口/更狠训 head。
  - 持平或降（≤15-20，image 检索不崩）→ 特征墙坐实（真墙），被试内收口。

用法（服务器，fmri 环境，项目根目录）：
  # 冒烟：--max_steps 200
  HF_HUB_OFFLINE=1 python training/train_head_hardneg.py \
      --data_path /root/autodl-tmp/MindEyeV2-main/data \
      --stage1 checkpoints/stage1/stage1_encoder.pt \
      --clip /root/autodl-tmp/models/clip-vit-large-patch14 \
      --max_steps 200
  # 全量（默认 2 epochs，~20-30 min）
  HF_HUB_OFFLINE=1 python training/train_head_hardneg.py \
      --data_path /root/autodl-tmp/MindEyeV2-main/data \
      --stage1 checkpoints/stage1/stage1_encoder.pt \
      --clip /root/autodl-tmp/models/clip-vit-large-patch14

输出：
  {out_dir}/head_hardneg_text.pt（微调后的 head）
  {out_dir}/head_hardneg_eval.json（微调前后 head 在 S1 留出上的对比）
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import CLIPModel, CLIPTokenizer

from data.dataloader import build_holdout_loader, build_train_val_loaders
from data.preprocessing import build_caption_map
from eval.eval_stage1_retrieval import build_caption_gallery, caption_retrieval, load_stage1
from eval.run_eval import build_gallery, retrieval_metrics
from training.losses import info_nce
from utils.diagnostics import run_diagnostics
from utils.metrics import compute_text_metrics

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def info_nce_text_extended(q, pos, bank, bank_img, nsd, n_hard, logit_scale):
    """text 分支 InfoNCE：负样本 = batch 内（排除自身与同图行）+ 全局 top-n_hard 硬负样本。

    q/pos: (B,768) 归一化；bank: (K,768) 归一化；bank_img: (K,)；nsd: (B,)。返回标量 loss。
    """
    B = q.size(0)
    dev = q.device
    pos_logit = logit_scale * (q * pos).sum(-1, keepdim=True)          # (B,1)
    batch_logits = logit_scale * (q @ pos.T)                            # (B,B)
    same = nsd.unsqueeze(0) == nsd.unsqueeze(1)                         # (B,B) 同图行
    batch_neg = batch_logits.masked_fill(
        torch.eye(B, dtype=torch.bool, device=dev) | same, -1e9)
    own = bank_img.unsqueeze(0) == nsd.unsqueeze(1)                     # (B,K) 同图 caption
    sim = q @ bank.T
    hard = sim.masked_fill(own, -1e9).topk(n_hard, dim=1).indices       # (B,n_hard)
    hard_logits = logit_scale * torch.einsum("bd,bnd->bn", q, bank[hard])
    logits = torch.cat([pos_logit, batch_neg, hard_logits], dim=1)      # (B, 1+B+n_hard)
    return F.cross_entropy(logits, torch.zeros(B, dtype=torch.long, device=dev))


def main():
    ap = argparse.ArgumentParser(description="hard-negative 先导：微调原始 head，只扩 text 负样本")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--stage1", default="checkpoints/stage1/stage1_encoder.pt")
    ap.add_argument("--clip", default="openai/clip-vit-large-patch14")
    ap.add_argument("--anatomy_dir", default="data/anatomy_cache")
    ap.add_argument("--out_dir", default="checkpoints/head_hardneg")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4, help="微调收敛好的 head 用 stage1 同款 lr")
    ap.add_argument("--n_hard", type=int, default=128, help="text 分支每 query 额外采样的全局硬负样本数")
    ap.add_argument("--logit_scale", type=float, default=1.0 / 0.07, help="与 stage1 一致")
    ap.add_argument("--train_subjs", default="1,2,3,4,5,6,7")
    ap.add_argument("--eval_holdout", type=float, default=0.05)
    ap.add_argument("--splits", default="train,new_test")
    ap.add_argument("--test_subj", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--log_steps", type=int, default=500)
    ap.add_argument("--diag_batches", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_spice", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} | n_hard={args.n_hard} | lr={args.lr} | logit_scale={args.logit_scale:.2f}")

    # 冻结 encoder+ridge；ContrastiveHead 微调（复用 stage1 已收敛权重）
    encoder, ridge, head = load_stage1(args.stage1, device, args.anatomy_dir)
    for p in list(encoder.parameters()) + list(ridge.parameters()):
        p.requires_grad = False
    from training.stage1_contrastive import ContrastiveHead
    baseline_head = ContrastiveHead().to(device).eval()
    baseline_head.load_state_dict(head.state_dict())
    print(f"[hardneg] encoder+ridge frozen; head 微调 {sum(p.numel() for p in head.parameters()) / 1e3:.0f}K params")

    clip = CLIPModel.from_pretrained(args.clip).to(device).eval()
    for p in clip.parameters():
        p.requires_grad = False
    tokenizer = CLIPTokenizer.from_pretrained(args.clip)
    mean, std = CLIP_MEAN.to(device), CLIP_STD.to(device)

    captions_by_nsd_idx = build_caption_map(args.data_path)
    splits_pool = tuple(s.strip() for s in args.splits.split(","))
    subj_list = [int(s) for s in args.train_subjs.split(",")]

    train_loader, _ = build_train_val_loaders(
        args.data_path, subj_list, captions_by_nsd_idx, args.batch_size,
        val_holdout=args.eval_holdout, return_image=True, seed=args.seed,
        splits=splits_pool)
    print(f"[hardneg] train loader: {len(train_loader)} batches (S{subj_list})")

    bank, bank_img, _ = build_caption_gallery(train_loader, clip, tokenizer,
                                              captions_by_nsd_idx, device)
    bank = bank.to(device)
    bank_img = bank_img.to(device)
    print(f"[hardneg] global text bank: {bank.size(0)} captions")

    head.train()
    opt = AdamW(head.parameters(), lr=args.lr)
    t0 = time.time()
    global_step = 0
    run_loss, run_rank = 0.0, 0.0
    for epoch in range(args.epochs):
        for batch in train_loader:
            voxels = batch["voxels"].to(device)
            subj = batch["subj"]
            nsd = batch["nsd_idx"].to(device)
            with torch.no_grad():
                tokens = ridge(encoder(voxels, subj), subj)          # (B,128,1024) 冻结
                px = (batch["image"].to(device).float() - mean) / std
                img_pos = F.normalize(clip.get_image_features(pixel_values=px), dim=-1)
                tok = tokenizer(batch["captions"], padding=True, truncation=True,
                                max_length=77, return_tensors="pt").to(device)
                txt_pos = F.normalize(clip.get_text_features(**tok), dim=-1)
            brain = head(tokens)                                     # (B,768) 归一化
            loss = 0.5 * info_nce(brain, img_pos, args.logit_scale) \
                 + 0.5 * info_nce_text_extended(brain, txt_pos, bank, bank_img, nsd,
                                                args.n_hard, args.logit_scale)
            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                pos_l = (brain * txt_pos).sum(-1, keepdim=True)
                neg_l = brain @ txt_pos.T                             # 用 batch text 估正样本名次（诊断用）
                rank = (neg_l > pos_l).sum(1).float().mean().item()
            run_loss += loss.item()
            run_rank += rank
            global_step += 1
            if global_step % args.log_steps == 0:
                print(f"[epoch {epoch}] step {global_step} | loss {run_loss / args.log_steps:.3f} "
                      f"| batch_pos_rank {run_rank / args.log_steps:.2f} | "
                      f"{(time.time() - t0) / 60:.1f}min", flush=True)
                run_loss, run_rank = 0.0, 0.0
            if args.max_steps and global_step >= args.max_steps:
                break
        if args.max_steps and global_step >= args.max_steps:
            break

    head.eval()
    os.makedirs(args.out_dir, exist_ok=True)
    head_path = os.path.join(args.out_dir, "head_hardneg_text.pt")
    torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, head_path)
    print(f"[hardneg] saved head: {head_path}")

    # ---- 评估：微调前后 head 同协议对比（S1 留出，与 0.153 同口径）----
    val_loader = build_holdout_loader(
        args.data_path, args.test_subj, captions_by_nsd_idx, args.batch_size,
        val_holdout=args.eval_holdout, return_image=True, seed=args.seed,
        splits=splits_pool)
    gallery_cap, img_of_cap, items_cap = build_caption_gallery(
        val_loader, clip, tokenizer, captions_by_nsd_idx, device)
    gallery_img, col_of = build_gallery(val_loader, clip, device)

    def make_enc(h):
        def fn(voxels, subj):
            return h(ridge(encoder(voxels.to(device), subj), subj))
        return fn

    results = {"n_hard": args.n_hard, "lr": args.lr, "logit_scale": args.logit_scale,
               "test_subj": args.test_subj, "protocol": f"holdout({args.eval_holdout} of {splits_pool})",
               "heads": {}}
    for name, h in [("stage1_ContrastiveHead（微调前）", baseline_head),
                    ("head_hardneg_text（微调后）", head)]:
        queries, nsd_idxs = [], []
        for b in val_loader:
            with torch.no_grad():
                queries.append(make_enc(h)(b["voxels"], b["subj"]))
            nsd_idxs += b["nsd_idx"].tolist()
        queries = [q for qq in queries for q in qq]
        ret = caption_retrieval(queries, nsd_idxs, gallery_cap, img_of_cap)
        top1 = ret.pop("top1_cols")
        hyps = [items_cap[c][1] for c in top1]
        refs = [captions_by_nsd_idx[i] or [""] for i in nsd_idxs]
        text = compute_text_metrics(hyps, refs, use_spice=not args.no_spice)
        img_ret = retrieval_metrics(queries, nsd_idxs, gallery_img, col_of)
        diag = run_diagnostics(make_enc(h), val_loader, max_batches=args.diag_batches)
        results["heads"][name] = {"caption_retrieval": ret, "top1_caption_metrics": text,
                                  "image_retrieval": img_ret, "diagnostics": diag}
        print(f"\n=== {name} ===")
        print(f"  caption retrieval: R@1={ret['R@1']:.3f} R@10={ret['R@10']:.3f} "
              f"MedR={ret['MedR']:.1f} top1_img_acc={ret['top1_img_acc']:.3f}")
        print(f"  top-1 caption metrics（文本上界）: CIDEr {text['CIDEr']:.4f} "
              f"BLEU-4 {text.get('BLEU-4', 0):.4f} ROUGE-L {text.get('ROUGE-L', 0):.4f}")
        print(f"  image retrieval: R@1={img_ret['R@1']:.3f} R@10={img_ret['R@10']:.3f} "
              f"MedR={img_ret['MedR']:.1f}")
        print(f"  diagnostics: zero_cos={diag['zero_cos']:.3f}")

    rpath = os.path.join(args.out_dir, "head_hardneg_eval.json")
    with open(rpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[hardneg] saved: {rpath}")
    print(f"[hardneg] 判据：微调后 caption R@1/top1_img_acc 是否翻倍+、上界 CIDEr 是否 >25 "
          f"（×100；微调前 baseline 15.3）、image MedR 是否保持 ~43 不崩")


if __name__ == "__main__":
    main()
