"""Stage 3：GRPO 奖励微调（复刻 BIT-LLM Stage 3）。

冻结 encoder + ridge + projector + cross-attn + LLM 主干，仅更新 LoRA（r=128, α=128）。
每组采样 G 个输出，优势按组内归一化；ref 用 LoRA 的 disable_adapter() 复用同一份权重。

  R_i   = 3*CIDEr + 0.3*BERTScore - 0.4*IncompletenessPenalty + w_spec*Specificity
  A_i   = (R_i - mean(R)) / std(R)                      （组内，G 个样本）
  loss  = -mean(A_i * log p_θ(o_i|q)) + kl_coef * KL(θ||ref)   （逐 token KL）

熵坍缩修复（2026-08-28）：Specificity = hyp 平均 unigram IDF（--w_spec，默认 0.3），
惩罚全功能词的模板家族；kl_coef 默认 0.05→0.1 收束策略漂移。

用法（服务器，fmri 环境，项目根目录）：
  python training/stage3_grpo.py \
      --data_path /root/autodl-tmp/MindEyeV2-main/data \
      --llm /root/autodl-tmp/models/Llama-3.2-3B-Instruct \
      --resume_stage2 checkpoints/stage2/stage2_sft.pt
  # 快速冒烟：--batch_size 2 --group 2 --max_steps 10

输出：{out_dir}/stage3_grpo.pt（全部非 llm 状态 + LoRA adapter 状态）
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functools import partial

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from transformers import AutoTokenizer

from data.dataloader import build_train_loader, collate_batch
from data.preprocessing import build_caption_map
from models.brain_llm import BrainLLM
from training.rewards import CiderGlobal, compute_rewards
from utils.save_load import load_train_ckpt, save_train_ckpt

SYSTEM = ("You are a helpful agent that decodes the brain activity of a person "
          "looking at an image. Output exactly ONE short caption. Do NOT ask for "
          "images or more info. Do NOT mention fMRI/brain/limitations. No self-reference.")
USER = "Describe the image as simply as possible."
MESSAGES = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": USER},
]


def tokenize_prompts(tokenizer, n, device):
    prompt = tokenizer.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    ids = enc.input_ids.to(device).expand(n, -1)
    attn = torch.ones_like(ids)
    return ids, attn


def sample_responses(model, voxels, subj, prompt_ids, prompt_attn, max_new, temp, top_p, pad_id):
    """每组 G 个采样：expand 到 (B*G) 独立序列，每行出 1 个样本。返回 (B*G, Lmax)。"""
    gen = model.generate(voxels, subj, input_ids=prompt_ids, attention_mask=prompt_attn,
                         max_new_tokens=max_new, do_sample=True, temperature=temp,
                         top_p=top_p, pad_token_id=pad_id)
    return gen


def response_logprobs(model, voxels, subj, gen, prompt_len, pad_id):
    """gen (B*G, Lmax)；返回响应区逐 token logprob（masked）+ 响应长度。"""
    Bn, Lmax = gen.shape
    attn = (gen != pad_id).long()
    out = model(voxels, subj, input_ids=gen, attention_mask=attn)
    logits = out.logits[:, :-1]  # (Bn, Lmax-1, V)
    logp = F.log_softmax(logits, dim=-1)
    tok = gen[:, 1:]
    per = logp.gather(-1, tok.unsqueeze(-1)).squeeze(-1)  # (Bn, Lmax-1)
    resp_logp = per[:, prompt_len - 1:]  # 响应 token 位置 prompt_len..Lmax-1
    resp_mask = (gen[:, prompt_len:] != pad_id).float()
    return resp_logp * resp_mask  # (Bn, new)


def decode_responses(tokenizer, gen, prompt_len):
    return [t.strip() for t in tokenizer.batch_decode(gen[:, prompt_len:],
                                                      skip_special_tokens=True)]


def main():
    ap = argparse.ArgumentParser(description="Stage 3 GRPO（BIT-LLM 复刻）")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--llm", default="/root/autodl-tmp/models/Llama-3.2-3B-Instruct")
    ap.add_argument("--resume_stage2", default="checkpoints/stage2/stage2_sft.pt")
    ap.add_argument("--anatomy_dir", default="data/anatomy_cache")
    ap.add_argument("--out_dir", default="checkpoints/stage3")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--group", type=int, default=4, help="每组采样数 G")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--kl_coef", type=float, default=0.1, help="KL 惩罚系数；旧 0.05 熵坍缩，0.1~0.2 收束漂移")
    ap.add_argument("--w_spec", type=float, default=0.3,
                    help="特异性奖励权重：hyp 平均 unigram IDF 加成（治模板化，0 关闭，0.3~0.5 试）")
    ap.add_argument("--max_new", type=int, default=32)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--use_bertscore", action="store_true",
                    help="奖励加入 BERTScore（会下载 roberta-large ~1.3GB，需设 HF_ENDPOINT 镜像）")
    ap.add_argument("--max_steps", type=int, default=None, help="快速冒烟用")
    ap.add_argument("--log_steps", type=int, default=5)
    ap.add_argument("--ckpt_freq", type=int, default=1000, help="每 N 步滚动保存 last.pt（断点续训）")
    ap.add_argument("--resume", default=None, help="从 {out_dir}/last.pt 续训（不再加载 stage2）")
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

    for p in model.parameters():
        p.requires_grad = False  # 冻结一切，只剩 LoRA 可训

    lora = LoraConfig(
        r=128, lora_alpha=128, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model.llm = get_peft_model(model.llm, lora)
    print(f"LoRA trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")
    model.train()

    captions_by_nsd_idx = build_caption_map(args.data_path)
    # 全量 COCO 语料固定 IDF：batch 内 8~16 条参考的微型 IDF 会强化通用模板（坍缩根因）
    t_corpus = time.time()
    cider_scorer = CiderGlobal(captions_by_nsd_idx)
    print(f"global CIDEr corpus: {cider_scorer.N} refs, df built in "
          f"{(time.time() - t_corpus):.1f}s", flush=True)
    loader = build_train_loader(args.data_path, list(range(1, 8)), captions_by_nsd_idx,
                                args.batch_size, return_image=False, collate_fn=collate_batch)
    print(f"train loader: {len(loader)} single-subject batches")

    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    pad_id = tokenizer.pad_token_id

    start_epoch, global_step = 0, 0
    if args.resume:
        # 续训：不加载 stage2，直接灌 last.pt（已含 encoder/ridge/projector/cross_attn + LoRA + opt）
        start_epoch, global_step = load_train_ckpt(model, opt, args.resume)
        print(f"resumed from {args.resume}: epoch {start_epoch}, step {global_step}")
    else:
        ckpt = torch.load(args.resume_stage2, map_location="cpu")
        model.load_state_dict({k: v for k, v in ckpt.items() if not k.startswith("llm.")},
                              strict=False)
        print(f"resumed stage2 weights from {args.resume_stage2}")

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        for batch in loader:
            voxels = batch["voxels"].to(device)      # (B, n_voxels)
            subj = batch["subj"]
            # 每 trial 用完整的 5 条 COCO 参考 caption（CIDEr 需要多 ref，单条 ref 区分度太差）
            refs = [captions_by_nsd_idx[int(i)] or [""] for i in batch["nsd_idx"]]

            B = voxels.size(0)
            G = args.group
            prompt_ids, prompt_attn = tokenize_prompts(tokenizer, B, device)
            Lp = prompt_ids.size(1)

            vx = voxels.repeat_interleave(G, 0)       # (B*G, n_voxels)
            pr_ids = prompt_ids.repeat_interleave(G, 0)
            pr_attn = prompt_attn.repeat_interleave(G, 0)

            with torch.no_grad():
                gen = sample_responses(model, vx, subj, pr_ids, pr_attn,
                                       args.max_new, args.temp, args.top_p, pad_id)
                hyps = decode_responses(tokenizer, gen, Lp)          # (B*G,)
                refs_flat = [r for r in refs for _ in range(G)]      # (B*G,)
                rewards = compute_rewards(hyps, refs_flat,
                                         use_bertscore=args.use_bertscore,
                                         cider_scorer=cider_scorer,
                                         w_spec=args.w_spec).to(device).view(B, G)

            resp_pol = response_logprobs(model, vx, subj, gen, Lp, pad_id)  # 有梯度
            with torch.no_grad():
                # 用 BaseTuner 的底层开关拨 LoRA 层（PeftModel 不暴露 disable/enable，
                # 而 transformers PeftAdapterMixin 的复数方法因 base.active_adapters=None 会报错）
                model.llm.base_model.disable_adapter_layers()
                resp_ref = response_logprobs(model, vx, subj, gen, Lp, pad_id)
                model.llm.base_model.enable_adapter_layers()

            # 组内优势
            mean = rewards.mean(-1, keepdim=True)
            std = rewards.std(-1, keepdim=True) + 1e-4
            adv = (rewards - mean) / std  # (B, G)

            pol_sum = resp_pol.sum(-1).view(B, G)
            ref_sum = resp_ref.sum(-1).view(B, G)

            policy_loss = -(adv * pol_sum).mean()
            kl = (torch.exp(resp_ref - resp_pol) - (resp_ref - resp_pol) - 1.0).sum(-1).mean()
            loss = policy_loss + args.kl_coef * kl

            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad()

            global_step += 1
            if global_step % args.ckpt_freq == 0:
                save_train_ckpt(model, opt, epoch, global_step,
                                os.path.join(args.out_dir, "last.pt"))
                print(f"    ckpt saved: {args.out_dir}/last.pt (step {global_step})", flush=True)
            if global_step % args.log_steps == 0:
                spec_mean = (sum(cider_scorer.specificity(h) for h in hyps) / len(hyps)
                             if hyps else 0.0)
                # CIDEr 分量单列：区分 mean_r 爬升是"图像匹配"还是"特异性注水"（reward hacking）
                cider_mean = (sum(cider_scorer.score(h, rl) for h, rl in zip(hyps, refs_flat))
                              / len(hyps) if hyps else 0.0)
                print(f"[epoch {epoch}] step {global_step} | loss {loss.item():.4f} "
                      f"| mean_r {rewards.mean().item():.3f} (cider {cider_mean:.3f} "
                      f"+ spec {spec_mean:.3f}) | {(time.time() - t0) / 60:.1f}min", flush=True)
                # 组内最高奖励样本：hyps[0] 常是泛化 miss，mean_r 高但看不到命中。
                # 若 group-best 带图像特异词且 r>0.4 → GRPO 在产出命中；一直泛化才是问题。
                bi = int(rewards.argmax().item())
                print(f"    group-best r={rewards.flatten()[bi].item():.3f}: {hyps[bi][:70]!r}")
                gbr = refs_flat[bi]
                print(f"    group-best ref: {gbr[0][:70]!r}" if gbr else "")
                print(f"    sample hyp: {hyps[0][:60]!r} | ref: {refs[0][:60]!r}")

            if args.max_steps and global_step >= args.max_steps:
                break
        else:
            continue
        break

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = {}
    for k, v in model.state_dict().items():
        if not k.startswith("llm.base_model.model.") or "lora_" in k:
            ckpt[k] = v.detach().cpu()
    path = os.path.join(args.out_dir, "stage3_grpo.pt")
    torch.save(ckpt, path)
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
