"""brain_llm.py 接线冒烟测试：随机 voxels + 一句 prompt，跑通 encoder→LLM 前向与生成。

用法（在服务器项目根目录，anatomy_cache 已生成）：
  python smoke_test_brain_llm.py --llm /root/autodl-tmp/models/Llama-3.2-3B-Instruct

验证点：
  1. BrainLLM 实例化（encoder+ridge+projector+cross-attn+冻结 Llama）不报错；
  2. forward logits 形状 (B, L, vocab)；
  3. random vs 全零 voxels 的 logits 不同 —— 证明 cross-attn hook 真正注入；
  4. generate 能带着 brain tokens 走完整生成循环。
"""
import argparse
import os

import numpy as np
import torch
from transformers import AutoTokenizer

from models.brain_llm import BrainLLM

SYSTEM = ("You are a helpful agent that decodes the brain activity of a person "
          "looking at an image. Output exactly ONE short caption. Do NOT ask for "
          "images or more info. Do NOT mention fMRI/brain/limitations. No self-reference.")


def build_prompt_inputs(tokenizer, device):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Describe the image as simply as possible."},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    return ids, torch.ones_like(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="/root/autodl-tmp/models/Llama-3.2-3B-Instruct")
    ap.add_argument("--anatomy_dir", default="data/anatomy_cache")
    ap.add_argument("--subj", type=int, default=1)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    torch_dtype = getattr(torch, args.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 用真实 anatomy 拿 n_voxels，构造匹配长度的随机 voxels
    npz = np.load(os.path.join(args.anatomy_dir, f"subj0{args.subj}_anatomy.npz"))
    n_voxels = npz["region_mask"].shape[0]
    npz.close()
    print(f"subj0{args.subj}: {n_voxels} voxels | device={device} | dtype={args.dtype}")

    model = BrainLLM(
        llm_path=args.llm,
        n_subjects=8,
        encoder_kwargs={"anatomy_dir": args.anatomy_dir},
        torch_dtype=torch_dtype,
    ).to(device)
    model.eval()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model instantiated OK | trainable params: {n_trainable / 1e6:.1f}M")

    layers = model.llm.model.layers
    hooked = [i for i, lay in enumerate(layers) if lay._forward_hooks]
    print(f"cross-attn hooks on layers: {hooked} (expect {list(model.cross_attn.layer_indices)})")
    gates = [f"{i}:{model.cross_attn[i].gate.item():.4f}" for i in hooked]
    print(f"gate init values: {gates} (expect ~0.08)")

    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    input_ids, attention_mask = build_prompt_inputs(tokenizer, device)
    input_ids = input_ids.expand(args.batch, -1)
    attention_mask = attention_mask.expand(args.batch, -1)
    print(f"input_ids: {tuple(input_ids.shape)}  (batch={args.batch})")

    g = torch.Generator(device=device).manual_seed(0)
    voxels = torch.randn(args.batch, n_voxels, generator=g, device=device)
    zeros = torch.zeros_like(voxels)

    with torch.no_grad():
        logits_real = model(voxels, args.subj, input_ids=input_ids, attention_mask=attention_mask).logits
        logits_zero = model(zeros, args.subj, input_ids=input_ids, attention_mask=attention_mask).logits

    print(f"forward logits shape: {tuple(logits_real.shape)}  (expect (B, L, vocab))")
    diff = (logits_real - logits_zero).abs().max().item()
    print(f"max |logits(real) - logits(zero)| = {diff:.4f}")
    assert diff > 0, "cross-attn hook 未生效：random 与 zero voxels 的 logits 完全相同！"

    with torch.no_grad():
        out = model.generate(voxels, args.subj, input_ids=input_ids,
                             attention_mask=attention_mask, max_new_tokens=16)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(f"generate output tokens: {tuple(out.shape)}")
    print(f"--- decoded (随机 voxels，预计无意义) ---\n{text}\n---")

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
