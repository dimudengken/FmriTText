"""stage2 坍缩机理诊断（服务器跑）。

回答核心问题：接口（cross-attn）到底有没有在牵引生成，还是输出纯是 LLM 的
caption 先验（模板族来自 prompt + LoRA，与脑信号无关）？

1) 4 层 adapter 的 gate 值 + o_proj 范数 —— 接口是否学到非零输出。
   - gate ≈ 0.08（初始值）且 o_proj_norm ≈ 0 → 接口死（o_proj 从未离开零）
   - gate >> 0.08 或 o_proj_norm 大 → 接口已学到东西
2) 同一批 trial 上 4 条件生成对比：none（无脑）/ rand（随机脑）/ zero（零脑）/ real（真实脑）。
   真实脑用 S1（ridge 已训练，特征真实对齐）。若四条件输出几乎相同 →
   接口是 no-op，模板坍缩来自 LLM 先验，与脑信号无关（= 加粗接口已修也白搭）。

用法（项目根目录）：
  HF_HUB_OFFLINE=1 python utils/diag_collapse.py [--trials 8]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

from data.dataloader import build_test_loader
from data.preprocessing import build_caption_map
from models.brain_llm import BrainLLM
from training.stage2_sft import MESSAGES
from utils.save_load import load_into

LLM = "/root/autodl-tmp/models/Llama-3.2-3B-Instruct"
CKPT = "checkpoints/stage2/last.pt"
DATA = "/root/autodl-tmp/MindEyeV2-main/data"
TEST_SUBJ = 1  # S1 ridge 已训练，特征真实对齐，无需 S8 ridge overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--test_subj", type=int, default=TEST_SUBJ)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--s8_ridge", default="checkpoints/stage_s8/stage_s8_ridge.pt",
                    help="test_subj=8 时 overlay 该 S8 ridge（否则 S8 ridge 随机，特征被搅碎）")
    ap.add_argument("--no_lora", action="store_true", help="不挂 LoRA（对照：裸 LLM）")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} | ckpt={args.ckpt} | test_subj={args.test_subj} | trials={args.trials}")

    model = BrainLLM(llm_path=LLM, n_subjects=8,
                     encoder_kwargs={"anatomy_dir": "data/anatomy_cache"},
                     torch_dtype=torch.bfloat16).to(device).eval()
    for pre in ("encoder.", "ridge.", "projector.", "cross_attn."):
        n = load_into(model, args.ckpt, prefix=pre)
        print(f"[load] {pre} {n} tensors", flush=True)
    if args.test_subj == 8 and os.path.exists(args.s8_ridge):
        n = load_into(model, args.s8_ridge, prefix="ridge.linears.7.")
        print(f"[load] S8 ridge overlay: {n} tensors from {args.s8_ridge}", flush=True)

    if not args.no_lora:
        model.llm = get_peft_model(model.llm, LoraConfig(
            r=128, lora_alpha=128, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
        for p in model.llm.parameters():
            p.requires_grad = False
        n = load_into(model, args.ckpt, prefix="llm.")
        print(f"[load] llm(lora) {n} tensors", flush=True)

    print("\n=== 1) gate 值 / o_proj 范数（接口是否学到非零输出）===")
    for i in model.cross_attn.layer_indices:
        ad = model.cross_attn[i]
        g = ad.gate.detach().item()
        on = ad.o_proj.weight.detach().float().norm().item()
        print(f"  layer {i:2d}: gate={g:.4f}  o_proj_norm={on:.4f}")

    tokenizer = AutoTokenizer.from_pretrained(LLM)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    loader = build_test_loader(DATA, args.test_subj,
                               build_caption_map(DATA), args.trials, return_image=False)
    batch = next(iter(loader))
    voxels = batch["voxels"].to(device)
    subj = batch["subj"]
    B = voxels.size(0)
    print(f"\n=== 2) 4 条件生成对比（S{args.test_subj:02d}，{B} trial，subj={subj}）===")

    prompt = tokenizer.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    ids = enc.input_ids.to(device).expand(B, -1)
    attn = enc.attention_mask.to(device).expand(B, -1)
    Lp = ids.size(1)

    def gen(brain):
        model._brain_tokens = brain
        try:
            out = model.llm.generate(ids, attention_mask=attn,
                                     max_new_tokens=32, do_sample=False, num_beams=1)
            return [t.strip() for t in tokenizer.batch_decode(out[:, Lp:], skip_special_tokens=True)]
        finally:
            model._brain_tokens = None

    bt = model._brain_forward(voxels, subj, use_ridge=True)
    print(f"  brain token: shape={tuple(bt.shape)} | mean={bt.float().mean():.4f} | std={bt.float().std():.4f}")
    refs = batch["captions"]
    print("  该批 trial 的真实参考 caption（前 2 条）:")
    for i, r in enumerate(refs):
        rl = r if isinstance(r, list) else [r]
        print(f"    {i}: {rl[0] if rl else ''}")

    conds = [
        ("none（无脑）", None),
        ("rand（随机脑）", torch.randn_like(bt) * bt.float().std()),
        ("zero（零脑）", torch.zeros_like(bt)),
        ("real（真实脑）", bt),
    ]
    for name, brain in conds:
        caps = gen(brain)
        print(f"\n  --- {name} ---")
        for i, c in enumerate(caps):
            print(f"    {i}: {c}")

    # 逐 trial 对比 real vs none 是否相同
    caps_real = gen(bt)
    caps_none = gen(None)
    same = sum(1 for a, b in zip(caps_real, caps_none) if a == b)
    print(f"\n  real vs none 逐 trial 相同数: {same}/{B}  "
          f"（{B - same}/{B} 不同 = 接口在改变输出；0 相同 = 接口 no-op）")


if __name__ == "__main__":
    main()
