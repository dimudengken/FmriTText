"""注意力池化探针：mean-pool vs attention-pool head（冻结 encoder+ridge，同预算同 loss）。

回答 CLAUDE.md 2026-08-27 "Stage1→Stage2 语义继承断层" 的下游问题：
stage1 的语义对齐发生在 mean-pool 后的 768 维，若 token 级还藏着 mean-pool 丢弃的
caption 信息，则（a）断层真实有损、（b）值得投 stage1 的 token 级对齐重训。
判据 = 注意力池化 head 在 S1 留出上的检索上界 vs mean-pool head（15.3 封闭集参照，×100）。

三个 head 同协议对比（S1 留出 5%、splits=train,new_test、seed=42，与 15.3 同口径；CIDEr ×100）：
  1. stage1_ContrastiveHead（原始，stage1 全量训练，封闭集参照 = 15.3）
  2. head_mean_scratch（mean-pool，随机初始化从零训，同预算）→ 同预算 mean-pool 参照
  3. head_attn_scratch（attention-pool，随机初始化从零训，同预算）→ 探针本体
  2 vs 3 隔离唯一的差异 = 池化机制（同 loss/同数据/同步数）。

loss 与 stage1 完全同构：0.5·info_nce(brain,image) + 0.5·info_nce(brain,text)，
batch-local 负样本，无 hard-neg（不引入第二个变量）。

AttentionPoolHead 设计：单可学习 query，query 零初始化 → 注意力初始近均匀（≈mean-pool），
训后可学出非均匀加权。若训完注意力熵仍 ≈ log(128)（均匀）→ 模型找不到值得加权的 token，
坐实 "token 不比 mean-pool 多携带信息"。

用法（服务器，fmri 环境，项目根目录）：
  # 冒烟：--max_steps 200 --epochs 1
  HF_HUB_OFFLINE=1 python training/train_head_attnpool.py \
      --data_path /root/autodl-tmp/MindEyeV2-main/data \
      --stage1 checkpoints/stage1/stage1_encoder.pt \
      --clip /root/autodl-tmp/models/clip-vit-large-patch14 \
      --max_steps 200
  # 全量（默认 2 epochs，~30-40 min）
  HF_HUB_OFFLINE=1 python training/train_head_attnpool.py \
      --data_path /root/autodl-tmp/MindEyeV2-main/data \
      --stage1 checkpoints/stage1/stage1_encoder.pt \
      --clip /root/autodl-tmp/models/clip-vit-large-patch14

输出：
  {out_dir}/head_attnpool_mean.pt + head_attnpool_attn.pt（两个 scratch head）
  {out_dir}/head_attnpool_eval.json（三 head 在 S1 留出上的对比）
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from transformers import CLIPModel, CLIPTokenizer

from data.dataloader import build_holdout_loader, build_train_val_loaders
from data.preprocessing import build_caption_map
from eval.eval_stage1_retrieval import build_caption_gallery, caption_retrieval, load_stage1
from eval.run_eval import build_gallery, retrieval_metrics
from training.losses import info_nce
from training.stage1_contrastive import ContrastiveHead
from utils.diagnostics import run_diagnostics
from utils.metrics import compute_text_metrics

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


class AttentionPoolHead(nn.Module):
    """128 tokens → 单可学习 query 注意力加权 → Linear(1024→768) → L2 归一化。

    query 零初始化 → 注意力初始近均匀（等价 mean-pool 起点），训练中学非均匀加权。
    """

    def __init__(self, in_dim=1024, out_dim=768):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(in_dim))
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, tokens):
        attn = F.softmax(tokens @ self.query / (tokens.size(-1) ** 0.5), dim=1)  # (B,128)
        x = (attn.unsqueeze(-1) * tokens).sum(dim=1)  # (B,1024)
        return F.normalize(self.proj(x), dim=-1)

    def attention(self, tokens):
        """返回注意力权重 (B,128)，诊断用：熵是否偏离均匀。"""
        return F.softmax(tokens @ self.query / (tokens.size(-1) ** 0.5), dim=1)


def attn_entropy(head, tokens):
    """注意力熵均值（bit）。均匀 128 路 = log2(128) ≈ 7.0，或 nats ≈ 4.85。"""
    a = head.attention(tokens)
    return -(a * (a + 1e-12).log()).sum(-1).mean().item()


def main():
    ap = argparse.ArgumentParser(description="注意力池化探针：mean vs attn 池化（冻结 encoder+ridge）")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--stage1", default="checkpoints/stage1/stage1_encoder.pt")
    ap.add_argument("--clip", default="openai/clip-vit-large-patch14")
    ap.add_argument("--anatomy_dir", default="data/anatomy_cache")
    ap.add_argument("--out_dir", default="checkpoints/head_attnpool")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4, help="与 stage1 同款 lr")
    ap.add_argument("--logit_scale", type=float, default=1.0 / 0.07, help="与 stage1 一致")
    ap.add_argument("--train_subjs", default="1,2,3,4,5,6,7")
    ap.add_argument("--eval_holdout", type=float, default=0.05)
    ap.add_argument("--splits", default="train,new_test")
    ap.add_argument("--test_subj", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--eval_only", action="store_true",
                    help="跳过训练，加载 {out_dir} 下已保存的 scratch head 只跑评估（省重训）")
    ap.add_argument("--no_ridge", action="store_true",
                    help="stage1 用 --no_ridge 训（读出头全共享）时探针必须同传：跳过 ridge，"
                         "enc = encoder(voxels)。漏传 = 随机 ridge 搅碎特征白测")
    ap.add_argument("--log_steps", type=int, default=500)
    ap.add_argument("--diag_batches", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_spice", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} | lr={args.lr} | logit_scale={args.logit_scale:.2f}")

    # 冻结 encoder+ridge；三个 head：原始参照 + 两个从零训 scratch head
    encoder, ridge, head_orig = load_stage1(args.stage1, device, args.anatomy_dir,
                                            no_ridge=args.no_ridge)
    for p in list(encoder.parameters()) + list(ridge.parameters()):
        p.requires_grad = False
    head_mean = ContrastiveHead().to(device)
    head_attn = AttentionPoolHead().to(device)

    def enc_tokens(voxels, subj):
        x = encoder(voxels, subj)
        if not args.no_ridge:
            x = ridge(x, subj)
        return x
    params = list(head_mean.parameters()) + list(head_attn.parameters())
    print(f"[attnpool] encoder+ridge frozen; scratch heads "
          f"{sum(p.numel() for p in params) / 1e3:.0f}K params")

    if args.eval_only:
        head_mean.load_state_dict(torch.load(
            os.path.join(args.out_dir, "head_attnpool_mean.pt"),
            map_location=device, weights_only=True))
        head_attn.load_state_dict(torch.load(
            os.path.join(args.out_dir, "head_attnpool_attn.pt"),
            map_location=device, weights_only=True))
        head_mean.eval()
        head_attn.eval()
        print(f"[attnpool] EVAL_ONLY: loaded saved scratch heads from {args.out_dir}")
    else:
        head_mean.train()
        head_attn.train()

    clip = CLIPModel.from_pretrained(args.clip).to(device).eval()
    for p in clip.parameters():
        p.requires_grad = False
    tokenizer = CLIPTokenizer.from_pretrained(args.clip)
    mean, std = CLIP_MEAN.to(device), CLIP_STD.to(device)

    captions_by_nsd_idx = build_caption_map(args.data_path)
    splits_pool = tuple(s.strip() for s in args.splits.split(","))
    subj_list = [int(s) for s in args.train_subjs.split(",")]

    if not args.eval_only:
        train_loader, _ = build_train_val_loaders(
            args.data_path, subj_list, captions_by_nsd_idx, args.batch_size,
            val_holdout=args.eval_holdout, return_image=True, seed=args.seed,
            splits=splits_pool)
        print(f"[attnpool] train loader: {len(train_loader)} batches (S{subj_list})")

        opt = AdamW(params, lr=args.lr)
    t0 = time.time()
    global_step = 0
    run_loss, run_rank = 0.0, 0.0
    for epoch in range(0 if args.eval_only else args.epochs):
        for batch in train_loader:
            voxels = batch["voxels"].to(device)
            subj = batch["subj"]
            with torch.no_grad():
                tokens = enc_tokens(voxels, subj)                    # (B,128,1024) 冻结
                px = (batch["image"].to(device).float() - mean) / std
                img_pos = F.normalize(clip.get_image_features(pixel_values=px), dim=-1)
                tok = tokenizer(batch["captions"], padding=True, truncation=True,
                                max_length=77, return_tensors="pt").to(device)
                txt_pos = F.normalize(clip.get_text_features(**tok), dim=-1)
            b_mean = head_mean(tokens)
            b_attn = head_attn(tokens)
            loss = 0.5 * info_nce(b_mean, img_pos, args.logit_scale) \
                 + 0.5 * info_nce(b_mean, txt_pos, args.logit_scale) \
                 + 0.5 * info_nce(b_attn, img_pos, args.logit_scale) \
                 + 0.5 * info_nce(b_attn, txt_pos, args.logit_scale)
            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                pos_l = (b_attn * txt_pos).sum(-1, keepdim=True)
                neg_l = b_attn @ txt_pos.T
                rank = (neg_l > pos_l).sum(1).float().mean().item()
            run_loss += loss.item()
            run_rank += rank
            global_step += 1
            if global_step % args.log_steps == 0:
                print(f"[epoch {epoch}] step {global_step} | loss {run_loss / args.log_steps:.3f} "
                      f"| attn_pos_rank {run_rank / args.log_steps:.2f} | "
                      f"{(time.time() - t0) / 60:.1f}min", flush=True)
                run_loss, run_rank = 0.0, 0.0
            if args.max_steps and global_step >= args.max_steps:
                break
        if args.max_steps and global_step >= args.max_steps:
            break

    if not args.eval_only:
        head_mean.eval()
        head_attn.eval()
        os.makedirs(args.out_dir, exist_ok=True)
        for h, name in [(head_mean, "head_attnpool_mean.pt"),
                        (head_attn, "head_attnpool_attn.pt")]:
            torch.save({k: v.detach().cpu() for k, v in h.state_dict().items()},
                       os.path.join(args.out_dir, name))
        print(f"[attnpool] saved scratch heads to {args.out_dir}")

    # ---- 评估：三 head 同协议对比（S1 留出，与 15.3 同口径，CIDEr ×100）----
    val_loader = build_holdout_loader(
        args.data_path, args.test_subj, captions_by_nsd_idx, args.batch_size,
        val_holdout=args.eval_holdout, return_image=True, seed=args.seed,
        splits=splits_pool)
    gallery_cap, img_of_cap, items_cap = build_caption_gallery(
        val_loader, clip, tokenizer, captions_by_nsd_idx, device)
    gallery_img, col_of = build_gallery(val_loader, clip, device)

    def make_enc(h):
        def fn(voxels, subj):
            return h(enc_tokens(voxels.to(device), subj))
        return fn

    results = {"lr": args.lr, "logit_scale": args.logit_scale,
               "test_subj": args.test_subj, "protocol": f"holdout({args.eval_holdout} of {splits_pool})",
               "reference": "stage1_ContrastiveHead 封闭集上界 CIDEr 15.3（×100，CLAUDE.md）",
               "heads": {}}
    heads = [("stage1_ContrastiveHead（原始，封闭集参照）", head_orig),
             ("head_mean_scratch（mean-pool，从零训同预算）", head_mean),
             ("head_attn_scratch（attention-pool，从零训同预算）", head_attn)]
    for name, h in heads:
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
        entry = {"caption_retrieval": ret, "top1_caption_metrics": text,
                 "image_retrieval": img_ret, "diagnostics": diag}
        results["heads"][name] = entry
        print(f"\n=== {name} ===")
        print(f"  caption retrieval: R@1={ret['R@1']:.3f} R@10={ret['R@10']:.3f} "
              f"MedR={ret['MedR']:.1f} top1_img_acc={ret['top1_img_acc']:.3f}")
        print(f"  top-1 caption metrics（文本上界）: CIDEr {text['CIDEr']:.4f} "
              f"BLEU-4 {text.get('BLEU-4', 0):.4f} ROUGE-L {text.get('ROUGE-L', 0):.4f}")
        print(f"  image retrieval: R@1={img_ret['R@1']:.3f} R@10={img_ret['R@10']:.3f} "
              f"MedR={img_ret['MedR']:.1f}")
        print(f"  diagnostics: zero_cos={diag['zero_cos']:.3f}")
        if isinstance(h, AttentionPoolHead):
            ents = []
            for b in val_loader:
                with torch.no_grad():
                    toks = enc_tokens(b["voxels"].to(device), b["subj"])
                    ents.append(attn_entropy(h, toks))
                if len(ents) >= 8:
                    break
            mean_ent = sum(ents) / len(ents)
            entry["attn_entropy"] = mean_ent
            print(f"  attention entropy: {mean_ent:.3f} nats（均匀 128 路 = {math.log(128):.3f}）"
                  f" → {'近均匀（未学到值得加权的 token）' if mean_ent > math.log(128) * 0.9 else '明显收窄（学到非均匀加权）'}")

    rpath = os.path.join(args.out_dir, "head_attnpool_eval.json")
    with open(rpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[attnpool] saved: {rpath}")
    print(f"[attnpool] 判据（×100）：attn-pool 上界 vs mean-pool 上界（15.3）——"
          f">15.3 则 token 有额外信息、断层值得修；≤15.3 则断层无害、token 级对齐不投")


if __name__ == "__main__":
    main()
