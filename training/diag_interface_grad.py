"""接口梯度测试：验证 loss 梯度能否穿过冻结 LLM 回到 cross-attn / projector。

背景：Stage 2 的 diag 里 gate 几乎不动（0.3→0.3008）、接口增益塌向 0，怀疑梯度在
冻结 LLM 的 25 层反传中衰减消失。本脚本做一次完整 forward+backward，打印：
1. 每层 dL/d(layer_output) 的范数（看梯度从 layer27 衰减到 layer3 的幅度）
2. encoder / ridge / projector / cross_attn 的参数梯度范数
3. 每个 adapter 的 gate 值及其梯度

用法（服务器 fmri 环境，项目根目录）：
  python training/diag_interface_grad.py \
      --data_path /root/autodl-tmp/MindEyeV2-main/data \
      --llm /root/autodl-tmp/models/Llama-3.2-3B-Instruct \
      --resume_encoder checkpoints/stage1/stage1_encoder.pt
"""
import argparse
import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer

from data.dataloader import build_train_loader
from data.preprocessing import build_caption_map
from models.brain_llm import BrainLLM
from training.stage2_sft import sft_collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--llm", default="/root/autodl-tmp/models/Llama-3.2-3B-Instruct")
    ap.add_argument("--resume_encoder", default=None)
    ap.add_argument("--anatomy_dir", default="data/anatomy_cache")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--gate_init", type=float, default=0.3)
    args = ap.parse_args()

    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = BrainLLM(llm_path=args.llm, n_subjects=8,
                     encoder_kwargs={"anatomy_dir": args.anatomy_dir},
                     gate_init=args.gate_init, torch_dtype=torch.bfloat16).to(device)
    if args.resume_encoder:
        ckpt = torch.load(args.resume_encoder, map_location="cpu", weights_only=True)
        # 剥前缀再传给子模块（同 stage2_sft 的修复）：带 "encoder." 前缀的键子模块不认，
        # 旧版静默加载 0 张量、编码器保持随机。
        enc = {k[len("encoder."):]: v for k, v in ckpt.items() if k.startswith("encoder.")}
        rg = {k[len("ridge."):]: v for k, v in ckpt.items() if k.startswith("ridge.")}
        model.encoder.load_state_dict(enc, strict=False)
        model.ridge.load_state_dict(rg, strict=False)
        n_enc = sum(1 for k in enc if k in model.encoder.state_dict())
        n_rg = sum(1 for k in rg if k in model.ridge.state_dict())
        print(f"resumed encoder ({n_enc}/{len(enc)}) + ridge ({n_rg}/{len(rg)})")
    model.train()
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.ridge.parameters():
        p.requires_grad = False
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"trainable ({sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M): {trainable}")

    # 每层 dL/d(output) 范数：观察梯度在冻结层间的衰减
    decay = {}

    def make_bhook(i):
        def bhook(module, grad_input, grad_output):
            decay[i] = grad_output[0].norm().item()
        return bhook

    for i, layer in enumerate(model.llm.model.layers):
        layer.register_full_backward_hook(make_bhook(i))

    caps = build_caption_map(args.data_path)
    collate = partial(sft_collate, tokenizer=tokenizer, max_len=160)
    loader = build_train_loader(args.data_path, [1], caps, args.batch_size,
                                return_image=False, collate_fn=collate)
    batch = next(iter(loader))
    voxels = batch["voxels"].to(device)
    subj = batch["subj"]
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    print(f"batch: subj={subj} voxels={tuple(voxels.shape)} seq_len={input_ids.size(1)}")

    model.zero_grad()
    out = model(voxels, subj, input_ids=input_ids,
                attention_mask=attention_mask, labels=labels)
    out.loss.backward()
    print(f"loss = {out.loss.item():.4f}")

    print("\n梯度衰减（dL/d(layer_output) 范数，layer27 → layer3）：")
    for i in sorted(decay, reverse=True):
        mark = "  <-- cross-attn 注入层" if i in (3, 7, 11, 15) else ""
        print(f"  layer{i:2d}: {decay[i]:.6e}{mark}")

    print("\n参数梯度范数：")

    def gn(prefix):
        tot = sum(p.grad.norm().item() ** 2 for n, p in model.named_parameters()
                  if n.startswith(prefix) and p.grad is not None)
        return tot ** 0.5

    for prefix in ["encoder.", "ridge.", "projector.", "cross_attn."]:
        print(f"  {prefix:14s} grad_norm = {gn(prefix):.3e}")

    print("\ngate 值及其梯度：")
    for i in (3, 7, 11, 15):
        g = model.cross_attn[i].gate
        gd = g.grad.item() if g.grad is not None else None
        print(f"  gate[{i:2d}] = {g.item():.4f}   grad = {gd}")


if __name__ == "__main__":
    main()
