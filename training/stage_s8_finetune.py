"""Stage S8：留出被试的少样本 ridge 功能对齐（少样本臂主结果）。

对齐 MindEye2"每被试加一层线性映射"的最小改动，落在 Key/Value 之后：
  Key/Value编码器(冻结, 共享) → S8 ridge(训, 新层) → Projector(冻结/可选训) → Cross-Attn(冻结/可选训) → Frozen Llama

- 预训练 S1-7 时 S8 的 ridge 从未见过梯度（shared1000 里的 S8 数据全程留出）；
  本脚本只在 S8 的 shared1000 图上做 SFT（response-only masking，同 stage2_sft）。
- 默认只训 ridge（~1M 参数/被试），Projector/Cross-Attn 冻结——MindEye2 的"最小改动"；
  --train_proj_cross 可同时解冻 projector + cross_attn（数据量小时更稳）。
- 预训练三个阶段哪个作为起点皆可：
    --resume 的 ckpt 含 encoder+ridge(+projector+cross_attn)：
      stage1_encoder.pt 只含 encoder+ridge+head → 需要 stage2 的 projector/cross_attn
      stage2_sft.pt   含 encoder+ridge+projector+cross_attn → 完整，默认推荐
      stage3_grpo.pt  含 LoRA 合并？不，含 LoRA 权重；本脚本走 stage2 即可
- 推理复用：s8_finetune 状态（ridge_s8）再叠回 stage2/3 完整 ckpt 即可。

用法（服务器，fmri 环境，项目根目录）：
  python training/stage_s8_finetune.py \
      --data_path /root/autodl-tmp/MindEyeV2-main/data \
      --llm /root/autodl-tmp/models/Llama-3.2-3B-Instruct \
      --resume checkpoints/stage2/stage2_sft.pt
  # stage2 是带 LoRA 训的 → 必须加 --lora：LLM 挂 LoRA 再加载 lora_* 权重（冻结）。
  #   否则微调时的 LLM 是裸 Llama，CE 空间与评估不一致，ridge 对齐到错误目标
  #   （症状：S8 生成 diverse 但 0 命中——2026-08-28 diag_collapse 实测）。
  # 对比对齐模式（MindEye2 式，--contrastive）：S8 ridge 用 InfoNCE 直接对齐 CLIP
  #   图像嵌入（stage1 ContrastiveHead 映射到 CLIP 空间），绕开 CE 穿 gate=0.08 冻结
  #   接口的弱梯度。CE 微调（裸/LoRA）都救不动 S8 语义对齐（2026-08-28 实测）。
  #   --contrastive --train_proj_cross：loss=InfoNCE+λ_ce·CE，projector/cross_attn 一起适应
  #   S8（align() 只走 encoder→ridge→head，纯 contrastive 下接口在计算图外、必须加 CE 项训它）。
  # 快速冒烟：--epochs 1 --max_steps 20 --batch_size 32 --val_freq 0
  # CE 分支默认 held-out 验证早停（治"接口记忆 shared1000"）：训练用 shared1000（new_test），
  #   每 --val_freq 步用 S8 unique 图（train，真 held-out，从不进训练）验证 response-only CE，
  #   验证最优即存 stage_s8_ridge.pt（best），连续 --patience 次不创新低即早停。best 选在
  #   held-out CE 最优步 = run_eval --split train（unique 图 held-out）的直接代理。
  #   关闭：--val_freq 0（旧行为：最后一步存盘）。

输出：{out_dir}/stage_s8_ridge.pt（S8 ridge + 可选 projector/cross_attn 状态）
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functools import partial

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer

from data.dataloader import build_finetune_loader
from data.preprocessing import build_caption_map
from models.brain_llm import BrainLLM
from utils.save_load import load_into, save
from training.stage2_sft import MESSAGES, sft_collate


@torch.no_grad()
def run_validation(model, val_loader, device, max_batches=0):
    """S8 unique 图（真 held-out）上聚合 response-only CE（与训练同口径）。返回平均 CE。"""
    model.eval()
    ce_sum, n = 0.0, 0
    for i, b in enumerate(val_loader):
        if max_batches and i >= max_batches:
            break
        v = b["voxels"].to(device)
        s = b["subj"]
        out = model(v, s, input_ids=b["input_ids"].to(device),
                    attention_mask=b["attention_mask"].to(device),
                    labels=b["labels"].to(device))
        ce_sum += out.loss.item() * v.size(0)
        n += v.size(0)
    model.train()
    return ce_sum / n if n else float("inf")


def main():
    ap = argparse.ArgumentParser(description="Stage S8 少样本 ridge 功能对齐")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--llm", default="/root/autodl-tmp/models/Llama-3.2-3B-Instruct")
    ap.add_argument("--resume", default="checkpoints/stage2/stage2_sft.pt")
    ap.add_argument("--ridge_overlay", default="checkpoints/stage_s8/stage_s8_ridge.pt",
                    help="contrastive 分支热身：overlay 已训好的 S8 ridge（防从随机 S8 ridge 起步），"
                         "不存在则跳过")
    ap.add_argument("--subj", type=int, default=8, help="留出被试（默认 8）")
    ap.add_argument("--anatomy_dir", default="data/anatomy_cache")
    ap.add_argument("--out_dir", default="checkpoints/stage_s8")
    ap.add_argument("--train_proj_cross", action="store_true",
                    help="同时解冻 projector + cross_attn（默认只训 ridge）")
    ap.add_argument("--split", default="new_test",
                    help="S8 微调数据 split：默认 shared1000(new_test)；可换 train")
    ap.add_argument("--val_freq", type=int, default=200,
                    help="每 N 步用 S8 held-out（另一 split）验证 CE；0=不验证（旧行为，最后一步存盘）")
    ap.add_argument("--val_batches", type=int, default=64,
                    help="验证限制最多 N batch（0=全量；S8 unique ~19500 trial 全量慢，默认 64≈1024 trial）")
    ap.add_argument("--patience", type=int, default=3,
                    help="验证 CE 连续 N 次不创新低即早停（best 选 held-out 最优步，压制 shared1000 记忆）")
    ap.add_argument("--val_split", default=None,
                    help="验证集 split：默认与训练 split 互补（训练 new_test→验证 train，反之）")
    ap.add_argument("--batch_size", type=int, default=16,
                    help="S8 只有 1M 参数可训但梯度要回传整条 LLM 图，bs=16 更省显存")
    ap.add_argument("--lora", action="store_true",
                    help="resume ckpt 含 lora_* 权重时必加：给 LLM 挂 LoRA 再加载（冻结，仅作固定接口），"
                         "否则微调 CE 空间与评估不一致，ridge 对齐错目标")
    ap.add_argument("--lora_r", type=int, default=128)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--contrastive", action="store_true",
                    help="InfoNCE 对齐 CLIP 图像嵌入（MindEye2 式）：ridge→stage1 head→CLIP 空间，"
                         "绕开 CE 穿弱 gate 接口的弱梯度。纯对比，不用 LLM/LoRA")
    ap.add_argument("--stage1", default="checkpoints/stage1/stage1_encoder.pt",
                    help="--contrastive 用：取 ContrastiveHead（1024→768 CLIP 空间）")
    ap.add_argument("--clip", default="/root/autodl-tmp/models/clip-vit-large-patch14")
    ap.add_argument("--cont_temp", type=float, default=0.07, help="InfoNCE 温度")
    ap.add_argument("--lambda_ce", type=float, default=1.0,
                    help="--contrastive --train_proj_cross 时：loss = InfoNCE + λ_ce·CE "
                         "（CE 项训 projector/cross_attn；纯 contrastive 只训 ridge 时无用）")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--max_steps", type=int, default=None, help="快速冒烟用")
    ap.add_argument("--log_steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} | llm={args.llm} | subj=0{args.subj}")

    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = BrainLLM(llm_path=args.llm, n_subjects=8,
                     encoder_kwargs={"anatomy_dir": args.anatomy_dir},
                     torch_dtype=torch.bfloat16).to(device)

    n_enc = load_into(model, args.resume, prefix="encoder.")
    n_rg = load_into(model, args.resume, prefix="ridge.")
    n_pp = load_into(model, args.resume, prefix="projector.")
    n_ca = load_into(model, args.resume, prefix="cross_attn.")
    print(f"loaded {n_enc} encoder + {n_rg} ridge + {n_pp} projector + {n_ca} cross_attn "
          f"tensors from {args.resume}")

    # S8 ridge 热身：S8 ridge 在 stage2_sft 里是随机（S8 ridge 陷阱），直接 overlay 上一轮
    # 已对齐的对比 ridge，让 CE（含 --train_proj_cross 的接口适应）从对齐好的特征开始，
    # 不重头训对齐。仅 subj=8 且文件存在时生效（两种分支都适用）。
    if args.subj == 8 and os.path.exists(args.ridge_overlay):
        n_rg2 = load_into(model, args.ridge_overlay, prefix=f"ridge.linears.{args.subj - 1}.")
        print(f"[load] ridge overlay: {n_rg2} tensors from {args.ridge_overlay}", flush=True)

    # 防御（同 stage2_sft）：resume ckpt 含 lora_* 但没 --lora → 静默丢 LoRA → 微调错空间。
    # CE 项（非 contrastive，或 contrastive+--train_proj_cross）需 LLM 空间与评估一致 → 此分支挂 LoRA；
    # 纯 contrastive（只训 ridge，不走 LLM）时 LoRA 无关，跳过。
    if not args.contrastive or args.train_proj_cross:
        _lk = torch.load(args.resume, map_location="cpu", weights_only=True)
        _st = _lk.get("model", _lk)
        _has_lora = any("lora_" in k for k in _st)
        if _has_lora and not args.lora:
            print("WARNING: resume ckpt 含 lora_* 权重但本次未挂 --lora → LLM 是裸 Llama，"
                  "微调空间与评估不一致，S8 ridge 会对齐错目标。带 LoRA 的 stage2 ckpt 必须加 --lora。",
                  flush=True)
        elif not _has_lora and args.lora:
            print("WARNING: --lora 已传但 resume ckpt 无 lora_* 权重 → 挂的是全新零 LoRA（恒等），"
                  "不等于评估时的 LoRA 接口。确认 resume 指向带 LoRA 的 stage2 ckpt。", flush=True)

        if args.lora:
            from peft import LoraConfig, get_peft_model
            model.llm = get_peft_model(model.llm, LoraConfig(
                r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"]))
            n_lora = load_into(model, args.resume, prefix="llm.")
            print(f"[load] llm(lora) {n_lora} tensors（LoRA 冻结，仅作固定接口）", flush=True)

    # 冻结一切，再选择性解冻。named_parameters 返回完整路径：ridge 内部的
    # ModuleList 包装成 ridge.linears.{s}.weight，不是 ridge.{s}.weight。
    for p in model.parameters():
        p.requires_grad = False
    ridge_prefix = f"ridge.linears.{args.subj - 1}."
    for n, p in model.named_parameters():
        if n.startswith(ridge_prefix):
            p.requires_grad = True
    if args.train_proj_cross:
        for n, p in model.named_parameters():
            if n.startswith(("projector.", "cross_attn.")):
                p.requires_grad = True
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable: {n_train / 1e6:.3f}M ({ridge_prefix}*"
          f"{' + projector + cross_attn' if args.train_proj_cross else ''})")
    model.train()

    # ============ 对比对齐模式（MindEye2 式）============
    # S8 ridge 用 InfoNCE 直接对齐 CLIP 图像嵌入：ridge_out → stage1 ContrastiveHead
    # → CLIP 空间，拉近正确图像、推远 batch 内其余。绕开 CE 穿过 gate=0.08 冻结接口
    # 的弱梯度（CE 裸/LoRA 都救不动 S8 语义对齐，2026-08-28 实测）。
    # 判读：训练 batch R@1 爬到高位且评估 retrieval 脱离 0.037 → ridge 对齐是短板；
    #       否则接口 held-out 泛化是天花板。
    if args.contrastive:
        import torch.nn.functional as F
        from transformers import CLIPModel
        from training.stage1_contrastive import ContrastiveHead

        head = ContrastiveHead().to(device).eval()
        for p in head.parameters():
            p.requires_grad = False
        _ck = torch.load(args.stage1, map_location="cpu", weights_only=True)
        _hd = {k.replace("head.", ""): v for k, v in _ck.items() if k.startswith("head.")}
        head.load_state_dict(_hd, strict=False)
        clip = CLIPModel.from_pretrained(args.clip).to(device).eval()
        for p in clip.parameters():
            p.requires_grad = False
        print(f"[s8] contrastive: head({len(_hd)} tensors) + CLIP={args.clip} | temp={args.cont_temp}")

        captions_by_nsd_idx = build_caption_map(args.data_path)
        # gallery 必须覆盖全部 trial：shuffle=False + drop_last=False，否则与训练 pass
        # 的 shuffle 状态不同、drop_last 丢的 trial 不同 → 训练 batch 撞上 gallery 缺失的图
        gallery_loader = build_finetune_loader(args.data_path, args.subj, captions_by_nsd_idx,
                                               args.batch_size, split=args.split,
                                               return_image=True, shuffle=False, drop_last=False)
        # 训练 loader 用 sft_collate（含 input_ids/labels）：--train_proj_cross 时 CE 项
        # 训 projector/cross_attn（align() 只走 encoder→ridge→head，纯 contrastive 下接口收零梯度）。
        _collate = partial(sft_collate, tokenizer=tokenizer, max_len=args.max_len)
        loader = build_finetune_loader(args.data_path, args.subj, captions_by_nsd_idx,
                                       args.batch_size, split=args.split,
                                       return_image=False, collate_fn=_collate)
        print(f"S0{args.subj} {args.split} contrastive loader: {len(loader)} batches "
              f"(gallery {len(gallery_loader)} batches)")

        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
        embs = {}
        for batch in gallery_loader:
            px = (batch["image"].to(device).float() - mean) / std
            with torch.no_grad():
                e = F.normalize(clip.get_image_features(pixel_values=px), dim=-1)
            for idx, emb in zip(batch["nsd_idx"].tolist(), e):
                embs.setdefault(idx, []).append(emb)
        idxs = sorted(embs)
        gallery = F.normalize(torch.stack([torch.stack(embs[i]).mean(0) for i in idxs]), dim=-1).to(device)
        col_of = {i: c for c, i in enumerate(idxs)}
        print(f"[s8] gallery: {len(idxs)} unique images")

        def align(voxels, subj):
            x = model.encoder(voxels, subj)
            x = model.ridge(x, subj)
            return F.normalize(head(x), dim=-1)   # (B, 768)；head 冻结，梯度回流 ridge

        opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        opt.zero_grad()
        global_step, running = 0, 0.0
        t0 = time.time()
        for epoch in range(args.epochs):
            for batch in loader:
                voxels = batch["voxels"].to(device)
                subj = batch["subj"]
                B = voxels.size(0)
                q = align(voxels, subj)
                key_idx = [col_of[i] for i in batch["nsd_idx"].tolist()]
                keys = gallery[torch.tensor(key_idx, device=device)]       # (B, 768)
                sim = (q @ keys.T) / args.cont_temp
                labels = torch.arange(B, device=device)
                loss = F.cross_entropy(sim, labels)
                if args.train_proj_cross:
                    ce = model(voxels, subj, input_ids=batch["input_ids"].to(device),
                               attention_mask=batch["attention_mask"].to(device),
                               labels=batch["labels"].to(device))
                    loss = loss + args.lambda_ce * ce.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad()

                global_step += 1
                running += loss.item()
                if global_step % args.log_steps == 0:
                    acc = (sim.argmax(1) == labels).float().mean().item()
                    print(f"[epoch {epoch}] step {global_step} | loss {running / args.log_steps:.4f} "
                          f"| batch R@1 {acc:.3f} | {(time.time() - t0) / 60:.1f}min", flush=True)
                    running = 0.0
                if args.max_steps and global_step >= args.max_steps:
                    break
            else:
                continue
            break

        ckpt = {k: v for k, v in model.state_dict().items()
                if k.startswith(ridge_prefix)
                or (args.train_proj_cross and k.startswith(("projector.", "cross_attn.")))}
        path = save(ckpt, os.path.join(args.out_dir, "stage_s8_ridge.pt"))
        print(f"saved: {path} ({len(ckpt)} tensors)")
        return

    captions_by_nsd_idx = build_caption_map(args.data_path)
    collate = partial(sft_collate, tokenizer=tokenizer, max_len=args.max_len)
    loader = build_finetune_loader(args.data_path, args.subj, captions_by_nsd_idx,
                                   args.batch_size, split=args.split,
                                   return_image=False, collate_fn=collate)
    print(f"S0{args.subj} {args.split} loader: {len(loader)} batches ({len(loader) * args.batch_size} trials)")

    # held-out 验证集 = 与训练 split 互补的另一半（默认），S8 该 split 数据从不进训练 →
    # best checkpoint 选在真泛化最优步，直接压制"接口记忆 shared1000 caption"。
    val_loader = None
    if args.val_freq:
        val_split = args.val_split or ("new_test" if args.split == "train" else "train")
        val_loader = build_finetune_loader(args.data_path, args.subj, captions_by_nsd_idx,
                                           args.batch_size, split=val_split,
                                           return_image=False, collate_fn=collate,
                                           shuffle=False, drop_last=False)
        _vb = args.val_batches or "全量"
        print(f"val loader: {len(val_loader)} batches (S0{args.subj} {val_split} held-out, "
              f"每 {args.val_freq} 步验证限 {_vb} batch, patience={args.patience})")

    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    opt.zero_grad()
    best_val, patience_hits, saved_best = float("inf"), 0, False
    global_step, running_loss = 0, 0.0
    t0 = time.time()
    for epoch in range(args.epochs):
        for batch in loader:
            voxels = batch["voxels"].to(device)
            subj = batch["subj"]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(voxels, subj, input_ids=input_ids,
                        attention_mask=attention_mask, labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad()

            global_step += 1
            running_loss += out.loss.item()
            if global_step % args.log_steps == 0:
                print(f"[epoch {epoch}] step {global_step} | loss {running_loss / args.log_steps:.4f} "
                      f"| {(time.time() - t0) / 60:.1f}min", flush=True)
                running_loss = 0.0

            # held-out 验证：S8 互补 split 从不进训练 → best 选在真泛化最优步
            if val_loader and global_step % args.val_freq == 0:
                val_ce = run_validation(model, val_loader, device, args.val_batches)
                if val_ce < best_val - 1e-4:
                    best_val = val_ce
                    patience_hits = 0
                    saved_best = True
                    _ck = {k: v for k, v in model.state_dict().items()
                           if k.startswith(ridge_prefix)
                           or (args.train_proj_cross and k.startswith(("projector.", "cross_attn.")))}
                    _p = save(_ck, os.path.join(args.out_dir, "stage_s8_ridge.pt"))
                    print(f"  [val] CE={val_ce:.4f} | best={best_val:.4f} | 存 best → {_p}",
                          flush=True)
                else:
                    patience_hits += 1
                    print(f"  [val] CE={val_ce:.4f} | best={best_val:.4f} | "
                          f"patience={patience_hits}/{args.patience}", flush=True)
                if patience_hits >= args.patience:
                    print(f"[early stop] 验证 CE 连续 {args.patience} 次未创新低，停在第 "
                          f"{global_step} 步 (best CE={best_val:.4f})", flush=True)
                    break

            if args.max_steps and global_step >= args.max_steps:
                break
        else:
            continue
        break

    if saved_best:
        print(f"stage_s8_ridge.pt = held-out 验证最优 (CE={best_val:.4f})，训练中已保存", flush=True)
    else:
        ckpt = {k: v for k, v in model.state_dict().items()
                if k.startswith(ridge_prefix)
                or (args.train_proj_cross and k.startswith(("projector.", "cross_attn.")))}
        path = save(ckpt, os.path.join(args.out_dir, "stage_s8_ridge.pt"))
        print(f"saved: {path} ({len(ckpt)} tensors)  # 未触发验证（val_freq=0 或 max_steps 提前结束）")


if __name__ == "__main__":
    main()
