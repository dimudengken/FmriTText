"""Stage1 文本检索上界（隔离 stage2）：只加载 stage1_encoder.pt 的 Encoder + ridge + ContrastiveHead。

不加载 stage2（projector/cross_attn/LoRA/LLM）——衡量「若接口完美转文本，生成质量最多能到哪」，
用来判断生成瓶颈在 stage1 编码器语义，还是在 stage2 接口消费。

数据流：
  voxels → KeyValueEncoder → SubjectRidge → ContrastiveHead → (B,768) L2 归一化   [query]
  测试集内每张 unique 图的 5 条真参考 caption → CLIP 文本嵌入 → (K,768) L2 归一化 [gallery]
  每 trial 检索 top-1 caption，其 CIDEr/BLEU（vs 该图真 refs）= 文本生成上界。

对照参照（同协议 S1 holdout 5%、splits=train,new_test、seed=42、full trial）下 stage2 生成：
  CIDEr 15.05 / BLEU-4 8.66%（×100 尺度，可报数，见 CLAUDE.md 单被试条目）。
本脚本 retrieved_caption_metrics 的 CIDEr/BLEU 亦为 ×100（metrics.py 2026-09-05 起直出）。
判读：
  - 检索上界 >> 生成数 → 接口在浪费编码器语义（接口有提升空间，编码器无罪）
  - 检索上界 ≈ 生成数 → 编码器本身是瓶颈，接口再完美也救不了

用法（服务器，fmri 环境，项目根目录；CLIP 本地路径 + HF_HUB_OFFLINE=1）：
  # S1 被试内留出（复现对照协议，主数）
  HF_HUB_OFFLINE=1 python eval/eval_stage1_retrieval.py \
      --data_path /root/autodl-tmp/MindEyeV2-main/data \
      --stage1 checkpoints/stage1/stage1_encoder.pt \
      --clip /root/autodl-tmp/models/clip-vit-large-patch14 \
      --test_subj 1 --eval_holdout 0.05 --splits train,new_test --seed 42
  # S8 跨被试（无 holdout 划分，用 unique 图 held-out）：
  #   --test_subj 8 --eval_holdout 0 --split train
  # S8 ridge 域失配诊断：--s8_ridge checkpoints/stage_s8/stage_s8_ridge.pt（覆盖随机 S8 ridge）
  #   对照：S8 unique MedR 4139 / shared1000 MedR 421（随机级）；S1（训练过 ridge）MedR 43
  #   2026-08-31 结果：换入后 zero_cos 0.097（特征敏感）但 MedR 仍 3717 ≈ 随机 → ridge 域失配
  #   不是主因，语义错位在编码器输出端本身 → 上 BIT-LLM 配方（--no_ridge --mixed_batch）
  # 跨被试机制换型（BIT-LLM 配方）评估：--test_subj 8 --eval_holdout 0 --split train --no_ridge
  #   同传 --no_ridge（否则随机 ridge 搅碎特征白测）；对照旧配方 S8 MedR 4139
  # 被试内文本上界（口径 A，单被试 head）：--test_subj 1 --eval_holdout 0.05 ... --no_ridge
  #   --within_retrieval_head checkpoints/head_attnpool/head_attnpool_mean.pt（不传 = 原装
  #   mixed-batch 跨被试 head，被试内上界偏低一档——head 口径影响测量值，须在结果里标注）
  # 冒烟：--max_trials 128

输出：{out_dir}/stage1_caption_retrieval.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

from data.dataloader import build_holdout_loader, build_test_loader
from data.preprocessing import build_caption_map
from models.fmri_encoder import KeyValueEncoder
from models.ridge import SubjectRidge
from utils.diagnostics import run_diagnostics
from utils.metrics import compute_text_metrics


def load_stage1(stage1_ckpt, device, anatomy_dir, no_ridge=False):
    """加载 stage1 的 encoder(+ridge)+head，打印每块加载张量数。

    兼容两种 ckpt：最终 stage1_encoder.pt（plain state_dict）与训练 last.pt
    （嵌套 {"model": ...}，供中途 S1 被试内监控）。no_ridge 时允许 ridge 键为 0。
    """
    ckpt = torch.load(stage1_ckpt, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    encoder = KeyValueEncoder(anatomy_dir=anatomy_dir).to(device).eval()
    ridge = SubjectRidge(n_subjects=8, dim=1024).to(device).eval()
    from training.stage1_contrastive import ContrastiveHead
    head = ContrastiveHead().to(device).eval()

    enc = {k.replace("encoder.", ""): v for k, v in ckpt.items() if k.startswith("encoder.")}
    rg = {k.replace("ridge.", ""): v for k, v in ckpt.items() if k.startswith("ridge.")}
    hd = {k.replace("head.", ""): v for k, v in ckpt.items() if k.startswith("head.")}
    encoder.load_state_dict(enc, strict=False)
    ridge.load_state_dict(rg, strict=False)
    head.load_state_dict(hd, strict=False)
    print(f"[stage1] encoder {len(enc)} tensors | ridge {len(rg)} | head {len(hd)}"
          f" (no_ridge={no_ridge})")
    if len(enc) == 0 or len(hd) == 0:
        raise RuntimeError(f"stage1 ckpt 加载 0 张量（enc {len(enc)}/ridge {len(rg)}/head {len(hd)}），"
                           f"检查 {stage1_ckpt} 键前缀是否为 encoder./head. "
                           f"（训练中间 last.pt 应为嵌套 {{model:...}}，已自动解包）")
    if not no_ridge and len(rg) == 0:
        raise RuntimeError(f"stage1 ckpt 无 ridge 键但未用 --no_ridge → ridge 随机初始化白测。"
                           f"这是 --no_ridge 训练的模型，评估必须加 --no_ridge")
    return encoder, ridge, head


def load_probe_head(path, device):
    """加载单被试口径 ContrastiveHead（口径 A 被试内文本上界用）。

    兼容纯 ContrastiveHead state_dict（无前缀，如 train_head_attnpool.py 的 head_attnpool_mean.pt）
    与全量 stage1 ckpt（带 "head." 前缀）。单被试 batch 训练的 head 保留 per-subject 图像身份；
    原装 mixed-batch 跨被试 head 会抹掉它 → 被试内检索上界偏低。
    """
    from training.stage1_contrastive import ContrastiveHead
    head = ContrastiveHead().to(device).eval()
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    allowed = set(head.state_dict())
    if any(k.startswith("head.") for k in ckpt):
        hd = {k.replace("head.", ""): v for k, v in ckpt.items() if k.startswith("head.")}
    else:
        hd = {k: v for k, v in ckpt.items() if k in allowed}
    missing = sorted(allowed - set(hd))
    if missing or not hd:
        print(f"[stage1] WARNING: {path} 加载 {len(hd)}/{len(head.state_dict())} tensors"
              f"missing={missing} unexpected={sorted(set(ckpt) - allowed)}")
        if len(hd) == 0:
            raise RuntimeError(f"--within_retrieval_head {path} 加载 0 张量，检查是否 ContrastiveHead ckpt")
    head.load_state_dict(hd, strict=False)
    print(f"[stage1] loaded probe ContrastiveHead from {path} ({len(hd)} tensors)")
    return head


def build_caption_gallery(loader, clip, tokenizer, captions_by_nsd_idx, device):
    """收集 loader 内 unique 图的 5 条真 caption → CLIP 文本嵌入画廊 (K,768) 归一化。"""
    idxs = sorted({i for b in loader for i in b["nsd_idx"].tolist()})
    items, seen = [], set()
    for i in idxs:
        for cap in captions_by_nsd_idx[i] or []:
            if (i, cap) not in seen:
                seen.add((i, cap))
                items.append((i, cap))
    embs = []
    for s in range(0, len(items), 256):
        tok = tokenizer([c for _, c in items[s:s + 256]], padding=True,
                        truncation=True, max_length=77, return_tensors="pt").to(device)
        with torch.no_grad():
            e = F.normalize(clip.get_text_features(**tok), dim=-1)
        embs.append(e.cpu())
    gallery = F.normalize(torch.cat(embs), dim=-1)  # (K, 768) cpu
    img_of = torch.tensor([i for i, _ in items])    # (K,) cpu
    print(f"[stage1] caption gallery: {len(items)} captions from {len(idxs)} unique images")
    return gallery, img_of, items


def caption_retrieval(queries, nsd_idxs, gallery, img_of):
    """每 trial 检索 top-1 caption；返回检索指标 + top-1 caption 索引。

    检索 rank：trial 真图像 5 条 caption 中的最高相似度在画廊里的名次（1-based）。
    """
    Q = torch.stack([q.cpu() for q in queries])     # (T, 768)
    sim = Q @ gallery.T                              # (T, K)
    tgt = torch.tensor(nsd_idxs)
    ranks = []
    for t in range(sim.size(0)):
        correct = (img_of == tgt[t]).nonzero().squeeze(1)
        best = sim[t, correct].max()
        ranks.append((sim[t] > best).sum().item() + 1)
    ranks = torch.tensor(ranks, dtype=torch.float)
    top1 = sim.argmax(dim=1)
    return {
        "R@1": (ranks == 1).float().mean().item(),
        "R@10": (ranks <= 10).float().mean().item(),
        "MedR": ranks.median().item(),
        "top1_img_acc": (img_of[top1] == tgt).float().mean().item(),
        "top1_cols": top1.tolist(),
    }


def main():
    ap = argparse.ArgumentParser(description="Stage1 文本检索上界（隔离 stage2）")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--stage1", default="checkpoints/stage1/stage1_encoder.pt")
    ap.add_argument("--s8_ridge", default=None,
                    help="S8 诊断：用训练过的 S8 ridge 覆盖 stage1 的随机 S8 ridge（如 stage_s8_ridge.pt）。"
                         "只对 --test_subj 8 有意义——测 ridge 域失配假说：若换后 S8 检索脱离随机级 → "
                         "S8 崩是'head 训练在 post-ridge 域、S8 zero_shot 走 pre-ridge 域'的域失配造成")
    ap.add_argument("--no_ridge", action="store_true",
                    help="stage1 用 --no_ridge 训（读出头全共享）时评估必须同传：跳过 ridge，"
                         "enc_fn = head(encoder(voxels))。漏传 = 随机 ridge 搅碎特征白测")
    ap.add_argument("--within_retrieval_head", default=None,
                    help="口径 A 被试内（--test_subj 1/2/5/7）文本上界用单被试口径探针 head（如 "
                         "checkpoints/head_attnpool/head_attnpool_mean.pt）。不传 = 原装 mixed-batch "
                         "跨被试 head（被试内上界偏低一档，head 口径影响测量值）。对 --test_subj 8 忽略")
    ap.add_argument("--clip", default="openai/clip-vit-large-patch14")
    ap.add_argument("--anatomy_dir", default="data/anatomy_cache")
    ap.add_argument("--out_dir", default="eval_results")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--test_subj", type=int, default=1,
                    help="1/2/5/7 训练被试（配 --eval_holdout）；8=S8 跨被试（配 --eval_holdout 0 --split train）")
    ap.add_argument("--eval_holdout", type=float, default=0.05,
                    help=">0 用该被试 val-holdout 留出子集（须与 stage2 训练同 val_holdout/splits/seed）；"
                         "0 则用 build_test_loader 全量评估（--split 指定）")
    ap.add_argument("--split", default="train",
                    help="--eval_holdout 0 时用：train=S8 unique held-out（默认）；new_test=shared1000")
    ap.add_argument("--splits", default="train,new_test",
                    help="holdout 划分的 splits 池，须与 stage2 训练同传")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_trials", type=int, default=None, help="冒烟：限制总 trial 数")
    ap.add_argument("--diag_batches", type=int, default=4)
    ap.add_argument("--no_spice", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} | stage1={args.stage1} | clip={args.clip}")

    encoder, ridge, head = load_stage1(args.stage1, device, args.anatomy_dir,
                                       no_ridge=args.no_ridge)

    # 检索 head 口径切换（同 run_eval）：口径 A 被试内（test_subj≠8）= 单被试探针 head；
    # 口径 B 跨被试（test_subj 8）= stage1 原装 mixed-batch 跨被试 head（默认）。
    if args.within_retrieval_head and args.test_subj == 8:
        print(f"[stage1] WARNING: --within_retrieval_head 对 S8 跨被试口径 B 无意义，已忽略")
    elif args.within_retrieval_head:
        head = load_probe_head(args.within_retrieval_head, device)
    elif args.test_subj != 8:
        print(f"[stage1] WARNING: 被试内文本上界（口径 A）未传 --within_retrieval_head → 用原装 "
              f"mixed-batch 跨被试 head，上界偏低一档（head 口径影响测量值，结果里已标注）")

    if args.s8_ridge and args.no_ridge:
        print(f"[stage1] --no_ridge 时忽略 --s8_ridge（ridge 被跳过，覆盖无意义）")
    if args.s8_ridge and not args.no_ridge:
        rg = torch.load(args.s8_ridge, map_location="cpu", weights_only=True)
        s8 = {k.replace("ridge.linears.7.", ""): v for k, v in rg.items()
              if k.startswith("ridge.linears.7.")}
        if not s8:
            s8 = {k: v for k, v in rg.items() if k in ("weight", "bias")}
        miss = ridge.linears[7].load_state_dict(s8, strict=False)
        print(f"[stage1] S8 ridge 覆盖自 {args.s8_ridge}（{len(s8)} 张量，S8 随机 ridge 被替换）"
              f"missing={miss.missing_keys} unexpected={miss.unexpected_keys}")
        if len(s8) == 0:
            print(f"[stage1] WARNING: {args.s8_ridge} 无 ridge.linears.7.* 键 → S8 ridge 仍是随机")

    def enc_fn(voxels, subj):
        x = encoder(voxels.to(device), subj)
        if not args.no_ridge:
            x = ridge(x, subj)
        return head(x)

    clip = CLIPModel.from_pretrained(args.clip).to(device).eval()
    for p in clip.parameters():
        p.requires_grad = False
    tokenizer = CLIPTokenizer.from_pretrained(args.clip)

    captions_by_nsd_idx = build_caption_map(args.data_path)
    if args.eval_holdout > 0:
        splits_pool = tuple(s.strip() for s in args.splits.split(","))
        loader = build_holdout_loader(args.data_path, args.test_subj, captions_by_nsd_idx,
                                      args.batch_size, val_holdout=args.eval_holdout,
                                      return_image=False, seed=args.seed, splits=splits_pool)
        split_label = f"holdout({args.eval_holdout:.0%} of {splits_pool})"
        if args.test_subj == 8:
            print("[stage1] WARNING: --eval_holdout 仅对训练过的被试（S1-7）有意义，S8 无同构留出划分")
    else:
        loader = build_test_loader(args.data_path, args.test_subj, captions_by_nsd_idx,
                                   args.batch_size, return_image=False, split=args.split)
        split_label = args.split
    total = len(loader.dataset)
    if args.max_trials:
        total = min(total, args.max_trials)
    print(f"S{args.test_subj:02d} {split_label} loader: {len(loader)} batches ({total} trials)")

    # 早期健全性：训练被试 zero_cos 应 <0.2（~0.84 = 编码器/ridge 加载失败）
    early = run_diagnostics(enc_fn, loader, max_batches=args.diag_batches)
    print(f"[stage1] EARLY zero_cos={early['zero_cos']:.3f} shuffle_cos={early['shuffle_cos']:.3f}")
    if early["zero_cos"] > 0.5:
        print(f"[stage1] WARNING: zero_cos={early['zero_cos']:.3f} 异常偏高，编码器/ridge 可能加载失败")

    gallery, img_of, items = build_caption_gallery(loader, clip, tokenizer, captions_by_nsd_idx, device)

    queries, nsd_idxs = [], []
    t0 = time.time()
    for i, batch in enumerate(loader):
        voxels = batch["voxels"].to(device)
        with torch.no_grad():
            queries.append(enc_fn(voxels, batch["subj"]))
        nsd_idxs += batch["nsd_idx"].tolist()
        if args.max_trials and (i + 1) * batch["voxels"].size(0) >= args.max_trials:
            break
        if (i + 1) % 10 == 0:
            print(f"[stage1] {i + 1}/{len(loader)} batches | {(time.time() - t0) / 60:.1f}min", flush=True)

    queries = [q for qq in queries for q in qq]
    ret = caption_retrieval(queries, nsd_idxs, gallery, img_of)
    top1_cols = ret.pop("top1_cols")
    hyps = [items[c][1] for c in top1_cols]
    refs = [captions_by_nsd_idx[i] or [""] for i in nsd_idxs]
    text = compute_text_metrics(hyps, refs, use_spice=not args.no_spice)

    retrieval_head_label = (f"within:{os.path.basename(args.within_retrieval_head)}"
                            if (args.within_retrieval_head and args.test_subj != 8)
                            else f"cross_subject(stage1):{os.path.basename(args.stage1)}")
    results = {"task": "stage1_caption_retrieval", "stage1": args.stage1,
               "retrieval_head": retrieval_head_label,
               "subj": args.test_subj, "split": split_label, "n_trials": len(nsd_idxs),
               "n_unique_images": int(img_of.unique().numel()), "gallery_size": int(gallery.size(0)),
               "retrieval": ret, "retrieved_caption_metrics": text, "diagnostics": early,
               "generation_reference": {"CIDEr": 15.05, "BLEU4": 8.66,
                                        "note": "同协议 S1 holdout stage2 生成，×100 尺度（CLAUDE.md 单被试条目）"}}
    os.makedirs(args.out_dir, exist_ok=True)
    rpath = os.path.join(args.out_dir, "stage1_caption_retrieval.json")
    with open(rpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[stage1] retrieval: R@1={ret['R@1']:.3f} R@10={ret['R@10']:.3f} MedR={ret['MedR']:.1f} "
          f"| top1_img_acc={ret['top1_img_acc']:.3f}")
    print(f"[stage1] retrieved-caption metrics（文本上界）: {text}")
    print(f"[stage1] generation reference（同协议 stage2 生成）: CIDEr 15.05 / BLEU-4 8.66% (×100)")
    print(f"[stage1] saved: {rpath}")


if __name__ == "__main__":
    main()
