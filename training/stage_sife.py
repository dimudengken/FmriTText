"""Stage SIFE：零样本臂对抗预训练（ZEBRA SIFE 正则器）。

从 stage2 权重出发，解冻 encoder + projector + cross_attn，在 S1-7 上训
  SFT response CE + λ_adv * adv_loss + λ_recon * recon_loss
其中 adv = 主体判别器经 GRL 对抗（迫使编码器输出主体不变特征），
recon = 重建锚点（从 pooled tokens 重建归一化体素，防止对抗破坏信号）。

- ridge 全程冻结/不用（零样本臂；S8 测试时 use_ridge=False 直测）。
- SIFE 只用于训练期正则化，不入推理路径；判别器在 8 类里只见过 S1-7（8 类目标不会出现）。
- 编码器梯度两路汇合：SFT CE（经 projector→cross-attn→LLM）+ SIFE（直接读 encoder 输出），
  共享权重、梯度正常累加；encoder 会重复前向一次（代价小，LLM 才是大头）。

用法（服务器，fmri 环境，项目根目录）：
  python training/stage_sife.py \
      --data_path /root/autodl-tmp/MindEyeV2-main/data \
      --llm /root/autodl-tmp/models/Llama-3.2-3B-Instruct \
      --resume checkpoints/stage2/stage2_sft.pt
  # 快速冒烟：--epochs 1 --max_steps 20 --batch_size 8

输出：{out_dir}/stage_sife.pt（encoder + projector + cross_attn + sife 状态；无 ridge/llm）
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

from data.dataloader import build_train_loader
from data.preprocessing import build_caption_map
from models.brain_llm import BrainLLM
from models.sife import SIFE
from utils.save_load import load_into, load_train_ckpt, save, save_train_ckpt
from training.stage2_sft import MESSAGES, sft_collate


def main():
    ap = argparse.ArgumentParser(description="Stage SIFE 零样本臂对抗预训练")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--llm", default="/root/autodl-tmp/models/Llama-3.2-3B-Instruct")
    ap.add_argument("--resume", default="checkpoints/stage2/stage2_sft.pt")
    ap.add_argument("--anatomy_dir", default="data/anatomy_cache")
    ap.add_argument("--out_dir", default="checkpoints/stage_sife")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4, help="projector/cross_attn/sife")
    ap.add_argument("--lr_enc", type=float, default=1e-5, help="encoder（更低，防扰动）")
    ap.add_argument("--lambda_adv", type=float, default=1.0)
    ap.add_argument("--lambda_recon", type=float, default=1.0)
    ap.add_argument("--grl_alpha", type=float, default=1.0,
                    help="GRL 反转系数（消融：1.0 强去主体 vs 0.3~0.5 温和；α 大易使 SFT_CE 难收敛）")
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--max_steps", type=int, default=None, help="快速冒烟用")
    ap.add_argument("--log_steps", type=int, default=10)
    ap.add_argument("--ckpt_freq", type=int, default=1000, help="每 N 步滚动保存 last.pt（断点续训）")
    ap.add_argument("--resume_train", default=None, help="从 {out_dir}/last.pt 续训（--resume 是初始权重，勿混）")
    ap.add_argument("--lora", action="store_true",
                    help="resume ckpt 含 lora_* 权重时必加：LLM 挂 LoRA 再加载（冻结），"
                         "否则 SFT CE 在裸 Llama 空间算、与评估空间不一致")
    ap.add_argument("--lora_r", type=int, default=128)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} | llm={args.llm}")

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

    sife = SIFE(token_dim=1024, n_subjects=8, grl_alpha=args.grl_alpha).to(device)

    # LoRA 防御（同 stage2/s8）：resume ckpt 含 lora_* 但没 --lora → 静默丢 LoRA → SFT CE 错空间。
    _lk = torch.load(args.resume, map_location="cpu", weights_only=True)
    _st = _lk.get("model", _lk)
    _has_lora = any("lora_" in k for k in _st)
    if _has_lora and not args.lora:
        print("WARNING: resume ckpt 含 lora_* 权重但未挂 --lora → LLM 裸 Llama，SFT CE 空间与评估不一致。"
              "带 LoRA 的 stage2 ckpt 必须加 --lora。", flush=True)
    if args.lora:
        from peft import LoraConfig, get_peft_model
        model.llm = get_peft_model(model.llm, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
        n_lora = load_into(model, args.resume, prefix="llm.")
        print(f"[load] llm(lora) {n_lora} tensors（LoRA 冻结，仅作固定接口）", flush=True)

    # 冻结：ridge（零样本臂不用）+ LLM；解冻 encoder/projector/cross_attn；SIFE 默认可训
    for p in model.parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        if n.startswith(("encoder.", "projector.", "cross_attn.")):
            p.requires_grad = True
    enc_params = [p for n, p in model.named_parameters()
                  if n.startswith("encoder.") and p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith("encoder.")]
    head_params += list(sife.parameters())
    n_train = sum(p.numel() for p in enc_params + head_params)
    print(f"trainable: {n_train / 1e6:.1f}M "
          f"(encoder {sum(p.numel() for p in enc_params) / 1e6:.1f}M @lr_enc "
          f"+ projector/cross_attn/sife @lr)")

    opt = AdamW([{"params": enc_params, "lr": args.lr_enc},
                 {"params": head_params, "lr": args.lr}])

    start_epoch, global_step = 0, 0
    if args.resume_train:
        start_epoch, global_step = load_train_ckpt(model, opt, args.resume_train, extra_module=sife)
        print(f"resumed from {args.resume_train}: epoch {start_epoch}, step {global_step}")

    opt.zero_grad()
    model.train()
    sife.train()

    captions_by_nsd_idx = build_caption_map(args.data_path)
    collate = partial(sft_collate, tokenizer=tokenizer, max_len=args.max_len)
    loader = build_train_loader(args.data_path, list(range(1, 8)), captions_by_nsd_idx,
                                args.batch_size, return_image=False, collate_fn=collate)
    print(f"train loader: {len(loader)} single-subject batches")

    running_loss, running_sife = 0.0, 0.0
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        for batch in loader:
            voxels = batch["voxels"].to(device)
            subj = batch["subj"]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            features = model.encoder(voxels, subj)  # (B, 128, 1024) pre-ridge，有梯度
            out = model(voxels, subj, input_ids=input_ids,
                        attention_mask=attention_mask, labels=labels)
            sife_loss, adv, recon = sife.loss(features, subj, voxels,
                                              args.lambda_adv, args.lambda_recon)
            loss = (out.loss + sife_loss) / args.grad_accum
            loss.backward()

            running_loss += out.loss.item()
            running_sife += sife_loss.item()
            if (global_step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad] + list(sife.parameters()), 1.0)
                opt.step()
                opt.zero_grad()

            global_step += 1
            if global_step % args.ckpt_freq == 0:
                save_train_ckpt(model, opt, epoch, global_step,
                                os.path.join(args.out_dir, "last.pt"),
                                extra_state=sife.state_dict())
            if global_step % args.log_steps == 0:
                print(f"[epoch {epoch}] step {global_step} | sft {running_loss / args.log_steps:.4f} "
                      f"| sife {running_sife / args.log_steps:.4f} (adv {adv.item():.3f}, "
                      f"recon {recon.item():.4f}) | {(time.time() - t0) / 60:.1f}min", flush=True)
                running_loss = running_sife = 0.0

            if args.max_steps and global_step >= args.max_steps:
                break
        else:
            continue
        break

    ckpt = {k: v.detach().cpu() for k, v in model.state_dict().items()
            if k.startswith(("encoder.", "projector.", "cross_attn."))}
    ckpt.update({f"sife.{k}": v.detach().cpu() for k, v in sife.state_dict().items()})
    path = save(ckpt, os.path.join(args.out_dir, "stage_sife.pt"))
    print(f"saved: {path} ({len(ckpt)} tensors, 零样本臂：加载后 use_ridge=False 直测)")


if __name__ == "__main__":
    main()
