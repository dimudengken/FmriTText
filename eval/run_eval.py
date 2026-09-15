"""S8 双协议评估：少样本臂（stage2 + S8 ridge）与零样本臂（stage_sife, use_ridge=False）。

对 S8 held-out new_test（shared1000，~3000 trial）：
- 生成字幕 → 文本指标（BLEU-1/2/3/4、ROUGE-L、METEOR、CIDEr、SPICE，COCO 5 参考）
- fMRI→image 检索（R@1/R@10/MedR）：查询 = 编码器(→ridge)→Stage1 ContrastiveHead→L2；
  画廊 = CLIP ViT-L/14 图像嵌入（unique shared1000 图像）
- 扰动诊断（Zero/Shuffle/Gaussian，编码器对体素值敏感性与鲁棒性）

双协议警告（CLAUDE.md）：少样本臂用了 S8 数据（ridge 微调），零样本臂完全没见过 S8，
两者不可混入同一 baseline 表直接对比，须分列报告。

检索/诊断 head 口径（2026-09-03，MedR 747 假象教训）：
- 口径 A 被试内（--test_subj 1/2/5/7，配 --eval_holdout>0）：检索必须用单被试探针 head
  （--within_retrieval_head，如 head_attnpool_mean.pt），不能用 stage1 原装 mixed-batch 跨被试
  ContrastiveHead——后者把 per-subject 图像身份投影掉，被试内图像检索退到随机级（MedR 747 先例），
  是评估路径假象非 Stage2 退化。生成文本指标与 zero_cos/gaussian 诊断不受检索 head 影响。
- 口径 B 跨被试（--test_subj 8）：检索用原装跨被试 head（默认），衡量跨被试迁移；被试内检索差
  是预期，禁止拿它评价被试内 Stage2。脚本按 --test_subj 自动切换（被试内需显式给探针 head 路径，
  漏传只告警不退随机假象）。

用法（服务器，fmri 环境，项目根目录）：
  python eval/run_eval.py \
      --data_path /root/autodl-tmp/MindEyeV2-main/data \
      --llm /root/autodl-tmp/models/Llama-3.2-3B-Instruct \
      --stage1 checkpoints/stage1/stage1_encoder.pt \
      --resume_stage2 checkpoints/stage2/stage2_sft.pt \
      --s8_ridge checkpoints/stage_s8/stage_s8_ridge.pt \
      --sife checkpoints/stage_sife/stage_sife.pt
  # 快速冒烟：--arm both --max_trials 24 --diag_batches 2 --batch_size 8 --no_spice

输出：{out_dir}/{arm}_results.json（指标 + 检索 + 诊断）与 {arm}_hypotheses.json。
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functools import partial

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPModel
from data.dataloader import build_test_loader, build_holdout_loader
from data.preprocessing import build_caption_map
from models.brain_llm import BrainLLM
from training.stage2_sft import MESSAGES
from utils.diagnostics import run_diagnostics
from utils.metrics import compute_text_metrics
from utils.save_load import load_into

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def load_stage1_head(stage1_ckpt, device):
    """从 stage1_encoder.pt 取 ContrastiveHead（CLIP 对比空间，检索/诊断用）。"""
    from training.stage1_contrastive import ContrastiveHead
    head = ContrastiveHead().to(device).eval()
    ckpt = torch.load(stage1_ckpt, map_location="cpu", weights_only=True)
    hd = {k.replace("head.", ""): v for k, v in ckpt.items() if k.startswith("head.")}
    head.load_state_dict(hd, strict=False)
    print(f"[eval] loaded ContrastiveHead from {stage1_ckpt} ({len(hd)} tensors)")
    return head


def load_probe_head(path, device):
    """加载单被试口径 ContrastiveHead（口径 A 被试内检索用）。

    兼容两种 ckpt：纯 ContrastiveHead state_dict（无前缀，如 train_head_attnpool.py 保存的
    head_attnpool_mean.pt）与全量 stage1 ckpt（带 "head." 前缀）。单被试 batch 训练的 head
    保留 per-subject 图像身份；原装 mixed-batch 跨被试 head 会抹掉它（MedR 747 假象）。
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
        print(f"[eval] WARNING: {path} 加载 {len(hd)}/{len(head.state_dict())} tensors"
              f"missing={missing} unexpected={sorted(set(ckpt) - allowed)}")
        if len(hd) == 0:
            raise RuntimeError(f"--within_retrieval_head {path} 加载 0 张量，检查是否 ContrastiveHead ckpt")
    head.load_state_dict(hd, strict=False)
    print(f"[eval] loaded probe ContrastiveHead from {path} ({len(hd)} tensors)")
    return head


def load_arm_state(model, arm, args):
    """加载某臂权重，返回该臂 use_ridge 标志。

    打印每前缀实际加载的张量数——若 ckpt 格式不对（嵌套训练 ckpt / 缺编码器键），
    load_into 静默加载 0 个张量、模型保持随机初始化，评估结果全是假象
    （症状：训练被试 S1 评估 cos(real,zero) 高达 0.84、检索跌到随机，与 diag_stage1 矛盾）。
    """
    if args.no_ridge:
        # 跨被试机制换型（stage1/stage2 --no_ridge）：读出头全共享、无 per-subject ridge，
        # 全部前向跳过 ridge（use_ridge=False）。不 overlay S8 ridge / projector 适配。
        if arm == "few_shot":
            for pre in ("encoder.", "projector.", "cross_attn."):
                n = load_into(model, args.resume_stage2, prefix=pre)
                print(f"[eval] few_shot(no_ridge): {pre} loaded {n} tensors", flush=True)
            if args.lora:
                n = load_into(model, args.resume_stage2, prefix="llm.")
                print(f"[eval] few_shot(no_ridge): loaded LoRA ({n} tensors)")
        else:
            for pre in ("encoder.", "projector.", "cross_attn."):
                n = load_into(model, args.sife, prefix=pre)
                print(f"[eval] zero_shot(no_ridge): {pre} loaded {n} tensors", flush=True)
            if args.lora:
                n = load_into(model, args.sife, prefix="llm.")
                lora_src = args.sife if n else args.resume_stage2
                if not n:
                    n = load_into(model, args.resume_stage2, prefix="llm.")
                print(f"[eval] zero_shot(no_ridge): loaded LLM LoRA ({n} tensors) from {lora_src}")
        return False
    if arm == "few_shot":
        for pre in ("encoder.", "ridge.", "projector.", "cross_attn."):
            n = load_into(model, args.resume_stage2, prefix=pre)
            print(f"[eval] few_shot: {pre} loaded {n} tensors from {args.resume_stage2}", flush=True)
        if args.s8_ridge:
            n = load_into(model, args.s8_ridge, prefix="ridge.linears.7.")
            print(f"[eval] few_shot: overlaid S8 ridge ({n} tensors)")
            # contrastive+--train_proj_cross 的 joint run 把 S8 适应的 projector/cross_attn
            # 也存进了 s8_ridge ckpt → 一并 overlay，否则评估用 stage2 原版接口、适应白训。
            # 旧 CE-ridge ckpt 无这些键 → load_into 加载 0，no-op 安全。
            n_pp = load_into(model, args.s8_ridge, prefix="projector.")
            n_ca = load_into(model, args.s8_ridge, prefix="cross_attn.")
            if n_pp or n_ca:
                print(f"[eval] few_shot: overlaid S8-adapted projector ({n_pp}) + cross_attn ({n_ca})")
        if args.lora:
            n = load_into(model, args.resume_stage2, prefix="llm.")
            print(f"[eval] few_shot: loaded LoRA ({n} tensors)")
        return True
    for pre in ("encoder.", "projector.", "cross_attn."):
        n = load_into(model, args.sife, prefix=pre)
        print(f"[eval] zero_shot: {pre} loaded {n} tensors from {args.sife}", flush=True)
    if args.lora:
        # LoRA 优先从 --sife 取（stage3_grpo.pt 含 GRPO LoRA → 主线程评估走这条）；
        # --sife 无 lora 键（stage_sife/stage2 ckpt）则回退 --resume_stage2（stage_sife 冻结的就是它）。
        n = load_into(model, args.sife, prefix="llm.")
        lora_src = args.sife if n else args.resume_stage2
        if not n:
            n = load_into(model, args.resume_stage2, prefix="llm.")
        print(f"[eval] zero_shot: loaded LLM LoRA ({n} tensors) from {lora_src}")
    print(f"[eval] zero_shot: loaded {os.path.basename(args.sife)} encoder/projector/cross_attn (skip ridge)")
    return False


def build_gallery(loader, clip, device):
    """收集 loader 内 unique 图像的 CLIP 嵌入（repeats 取均值）→ (K, 768) L2 归一化。"""
    mean = CLIP_MEAN.to(device)
    std = CLIP_STD.to(device)
    embs = {}
    for batch in loader:
        px = (batch["image"].to(device).float() - mean) / std
        with torch.no_grad():
            e = F.normalize(clip.get_image_features(pixel_values=px), dim=-1)
        for idx, emb in zip(batch["nsd_idx"].tolist(), e):
            embs.setdefault(idx, []).append(emb)
    idxs = sorted(embs)
    gallery = F.normalize(torch.stack([torch.stack(embs[i]).mean(0) for i in idxs]), dim=-1)
    col_of = {i: c for c, i in enumerate(idxs)}
    print(f"[eval] gallery: {len(idxs)} unique images")
    return gallery, col_of


def make_enc_fn(model, head, use_ridge, device):
    def fn(voxels, subj):
        x = model.encoder(voxels.to(device), subj)
        if use_ridge:
            x = model.ridge(x, subj)
        return head(x)
    return fn


def retrieval_metrics(queries, nsd_idxs, gallery, col_of):
    """queries: list[(T,768) 归一化] 与每 trial 的 nsd_idx 对齐。"""
    Q = torch.stack(queries)  # (T, 768)
    sim = Q @ gallery.T                    # (T, K)
    correct = torch.tensor([col_of[i] for i in nsd_idxs], device=sim.device)
    self_sim = sim.gather(1, correct.unsqueeze(1)).squeeze(1)
    rank = (sim > self_sim.unsqueeze(1)).sum(1) + 1   # 1-based rank of correct
    return {
        "R@1": (rank == 1).float().mean().item(),
        "R@10": (rank <= 10).float().mean().item(),
        "MedR": rank.float().median().item(),
    }


def run_text_sensitivity(model, loader, tokenizer, prompt_ids, prompt_attn, Lp,
                         captions_by_nsd_idx, use_ridge, args, device, label):
    """生成级 voxel 敏感性诊断：同批 trial 在 real/zero/shuffle voxels 下 greedy 生成。

    greedy（do_sample=False, beams=1）= 确定性：若脑特征没进 LLM 采样，三条件输出应逐字
    相同。判据：
      real≈zero/shuf（CIDEr 相近、逐字一致率≈1）→ 冻结 encoder/projector 输出没驱动
        LLM 生成 → 接口死通道（支持"COCO Lottery"：LLM LoRA 忽略脑特征）；
      real>>zero/shuf → 脑输入确实在驱动生成，问题是 RL 奖励/优势推不动质量中心
        （advantage 噪声方向）。
    同批 trial、同 prompt、greedy 无随机 → 唯一变量是 voxels 内容。
    输出 {out_dir}/text_sens_{label}_{sife_stem}.json。
    """
    n_trials = args.text_sens_trials
    hyps_r, hyps_z, hyps_s, refs = [], [], [], []
    t0 = time.time()
    for i, batch in enumerate(loader):
        vox = batch["voxels"].to(device)
        subj = batch["subj"]
        B = vox.size(0)
        zero = torch.zeros_like(vox)
        # 每个 trial 独立打乱全部体素（保留边缘分布，破坏空间/取值对应）
        shuf = torch.stack([v[torch.randperm(v.numel(), device=v.device)] for v in vox])

        def gen(v):
            g = model.generate(v, subj, input_ids=prompt_ids.expand(B, -1),
                               attention_mask=prompt_attn.expand(B, -1),
                               max_new_tokens=args.max_new, use_ridge=use_ridge,
                               do_sample=False, num_beams=args.num_beams)
            return [t.strip() for t in tokenizer.batch_decode(g[:, Lp:], skip_special_tokens=True)]

        hyps_r += gen(vox)
        hyps_z += gen(zero)
        hyps_s += gen(shuf)
        for idx in batch["nsd_idx"].tolist():
            refs.append(captions_by_nsd_idx[idx] or [""])
        if len(hyps_r) >= n_trials or (i + 1) == len(loader):
            break
        if (i + 1) % 5 == 0:
            print(f"[text_sens:{label}] {len(hyps_r)}/{n_trials} trials | "
                  f"{(time.time() - t0) / 60:.1f}min", flush=True)

    hyps_r, hyps_z, hyps_s, refs = (hyps_r[:n_trials], hyps_z[:n_trials],
                                    hyps_s[:n_trials], refs[:n_trials])
    n = len(hyps_r)

    def agree(a, b):
        return sum(1 for x, y in zip(a, b) if x == y) / n

    m_r = compute_text_metrics(hyps_r, refs, use_spice=False)
    m_z = compute_text_metrics(hyps_z, refs, use_spice=False)
    m_s = compute_text_metrics(hyps_s, refs, use_spice=False)
    out = {
        "arm": label, "n_trials": n,
        "exact_agree_real_vs_zero": agree(hyps_r, hyps_z),
        "exact_agree_real_vs_shuf": agree(hyps_r, hyps_s),
        "exact_agree_zero_vs_shuf": agree(hyps_z, hyps_s),
        "CIDEr": {"real": m_r["CIDEr"], "zero": m_z["CIDEr"], "shuf": m_s["CIDEr"]},
        "BLEU4": {"real": m_r["BLEU-4"], "zero": m_z["BLEU-4"], "shuf": m_s["BLEU-4"]},
        "ROUGE_L": {"real": m_r["ROUGE-L"], "zero": m_z["ROUGE-L"], "shuf": m_s["ROUGE-L"]},
        "delta_CIDEr_real_minus_zero": m_r["CIDEr"] - m_z["CIDEr"],
        "delta_CIDEr_real_minus_shuf": m_r["CIDEr"] - m_s["CIDEr"],
    }
    stem = os.path.splitext(os.path.basename(args.sife))[0]
    rpath = os.path.join(args.out_dir, f"text_sens_{label}_{stem}.json")
    with open(rpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[text_sens:{label}] n={n} | CIDEr real/zero/shuf = "
          f"{out['CIDEr']['real']:.4f}/{out['CIDEr']['zero']:.4f}/{out['CIDEr']['shuf']:.4f} | "
          f"BLEU-4 {out['BLEU4']['real']:.4f}/{out['BLEU4']['zero']:.4f}/{out['BLEU4']['shuf']:.4f} | "
          f"exact-agree r-vs-z={out['exact_agree_real_vs_zero']:.3f} "
          f"r-vs-shuf={out['exact_agree_real_vs_shuf']:.3f} | saved: {rpath}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description="S8 双协议评估")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--llm", default="/root/autodl-tmp/models/Llama-3.2-3B-Instruct")
    ap.add_argument("--stage1", default="checkpoints/stage1/stage1_encoder.pt")
    ap.add_argument("--resume_stage2", default="checkpoints/stage2/stage2_sft.pt")
    ap.add_argument("--s8_ridge", default="checkpoints/stage_s8/stage_s8_ridge.pt")
    ap.add_argument("--sife", default="checkpoints/stage_sife/stage_sife.pt")
    ap.add_argument("--arm", choices=["few_shot", "zero_shot", "both"], default="both")
    ap.add_argument("--clip", default="openai/clip-vit-large-patch14",
                    help="CLIP 模型：HF 名称或本地目录（服务器离线时传本地路径）")
    ap.add_argument("--anatomy_dir", default="data/anatomy_cache")
    ap.add_argument("--out_dir", default="eval_results")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new", type=int, default=32)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--test_subj", type=int, default=8,
                    help="测哪个被试。默认 8(S8 held-out)；传 1/2/5/7 可在训练过的被试上做接口纯净测试"
                         "（其 ridge 已训练，无 S8 随机 ridge confound）")
    ap.add_argument("--max_trials", type=int, default=None, help="冒烟：限制总 trial 数")
    ap.add_argument("--split", default="new_test",
                    help="评估数据 split：new_test=shared1000（默认，S8 微调同源）；"
                         "train=S8 unique 图（从未进 S8 微调 → held-out 泛化检查）")
    ap.add_argument("--eval_holdout", type=float, default=0.0,
                    help=">0 时用该被试的 val-holdout 留出子集评估（被试内 held-out，仅训练过的被试 1/2/5/7）。"
                         "划分与 build_train_val_loaders 完全同构（同 seed+s randperm、同 val_holdout），"
                         "必须与 stage2 训练同传 val_holdout 值、--splits 与 --seed，否则留出子集与训练重叠泄漏")
    ap.add_argument("--splits", default="train,new_test",
                    help="holdout 划分的 splits 池（逗号分隔），须与 stage2 训练同传（默认 train∪new_test）")
    ap.add_argument("--diag_batches", type=int, default=None, help="冒烟：限制诊断 batch 数")
    ap.add_argument("--no_spice", action="store_true", help="跳过 SPICE（需 Java，慢）")
    ap.add_argument("--lora", action="store_true",
                    help="resume_stage2 含 LoRA 权重：给 LLM 挂 LoRA 再加载，否则 LoRA 模型会被配冻结 LLM 白测")
    ap.add_argument("--lora_r", type=int, default=128)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--proj_mode", default="rmsnorm", choices=["rmsnorm", "norm_mag"],
                    help="必须与 stage2 训练同传：norm_mag=投影器末通道拼 L2 范数的 ckpt，"
                         "否则 rmsnorm 前向丢掉幅值通道白测")
    ap.add_argument("--gate_mode", default="scalar", choices=["scalar", "per_token"],
                    help="必须与 stage2 训练同传：per_token=逐脑 token gate 的 ckpt，"
                         "否则 scalar 架构加载失败白测")
    ap.add_argument("--sem_kv", action="store_true",
                    help="必须与 stage2 训练同传：方案 A 的 ckpt 在 KV 里追加了第 129 个语义摘要"
                         "token，漏传 = 129-token 模型用 128-token 前向白测")
    ap.add_argument("--no_ridge", action="store_true",
                    help="必须与 stage1/stage2 训练同传：跨被试机制换型（读出头全共享、无 "
                         "per-subject ridge）。全程 use_ridge=False，否则随机 ridge 搅碎特征白测")
    ap.add_argument("--bypass_mlp", action="store_true",
                    help="诊断消融：跳过 encoder 共享 4 层残差 MLP（linear-vs-nonlinear）。"
                         "head/projector/cross-attn 均未经 readout 域调参 → S1/S8 都会因接口失配"
                         "下降，判读只看 S8 相对 S1 是否拉回（encoding 级判据见 "
                         "eval_subject_alignment.py --bypass_mlp）")
    ap.add_argument("--within_retrieval_head", default=None,
                    help="口径 A 被试内（--test_subj 1/2/5/7）检索用单被试口径探针 head（如 "
                         "checkpoints/head_attnpool/head_attnpool_mean.pt）。不传 = 用 stage1 原装 "
                         "mixed-batch 跨被试 head → 被试内图像检索退随机级（MedR 747 假象，非 Stage2 "
                         "退化）；对 --test_subj 8（跨被试口径 B）自动忽略。生成/诊断不受影响")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--text_sens_trials", type=int, default=0,
                    help=">0 时追加生成级敏感性诊断：同批 trial 在 real/zero/shuffle voxels 下 "
                         "greedy 生成，比 CIDEr 与逐字一致率。greedy=确定性 → 若脑特征没进 LLM，"
                         "三条件输出逐字相同。real≈zero/shuf → LLM 忽略脑特征（接口问题）；"
                         "real>>zero → 脑输入驱动生成，问题在 RL 优化。")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} | arm={args.arm} | clip={args.clip}")

    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = BrainLLM(llm_path=args.llm, n_subjects=8,
                     encoder_kwargs={"anatomy_dir": args.anatomy_dir},
                     torch_dtype=torch.bfloat16,
                     proj_mode=args.proj_mode, gate_mode=args.gate_mode,
                     sem_kv=args.sem_kv).to(device).eval()
    if args.bypass_mlp:
        model.encoder.bypass_mlp = True
        print("[eval] ** MLP bypass 开启：跳过共享 4 层残差 MLP（接口未经 readout 域调参，"
              "S1/S8 均会因接口失配下降；只读 S8 相对 S1 的拉回幅度）", flush=True)
    if args.lora:
        from peft import LoraConfig, get_peft_model
        # 7 类超集：GRPO(stage3) 的 LoRA 训在 q/k/v/o/gate/up/down；stage2 只有 q/k/v/o，
        # 缺的模块 PEFT 零初始化 lora_B = 无操作，加载 strict=False 静默跳过，不污染 stage2 评估。
        lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                              lora_dropout=args.lora_dropout,
                              target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                              "gate_proj", "up_proj", "down_proj"])
        model.llm = get_peft_model(model.llm, lora_cfg)
        for p in model.llm.parameters():
            p.requires_grad = False
        print(f"[eval] LLM wrapped with LoRA (r={args.lora_r})")
    head = load_stage1_head(args.stage1, device)

    # 检索/诊断 head 口径切换（生成不走 head，不受影响）：
    #   口径 A 被试内（--test_subj≠8）= 单被试探针 head（--within_retrieval_head）；
    #   口径 B 跨被试（--test_subj 8）= stage1 原装 mixed-batch 跨被试 head（默认，衡量跨被试迁移）。
    retrieval_head = head
    if args.within_retrieval_head and args.test_subj == 8:
        print("[eval] WARNING: --within_retrieval_head 对 S8 跨被试口径 B 无意义，已忽略（S8 用原装 head）")
    elif args.within_retrieval_head:
        retrieval_head = load_probe_head(args.within_retrieval_head, device)
    elif args.test_subj != 8:
        print("[eval] WARNING: 被试内评估（口径 A）未传 --within_retrieval_head → 检索用 stage1 原装 "
              "mixed-batch 跨被试 head，被试内图像检索会退随机级（口径错配假象，非 Stage2 退化）；"
              "生成 CIDEr/BLEU 与 zero_cos/gaussian 诊断不受影响")

    clip = CLIPModel.from_pretrained(args.clip).to(device).eval()
    for p in clip.parameters():
        p.requires_grad = False

    captions_by_nsd_idx = build_caption_map(args.data_path)
    if args.eval_holdout > 0:
        splits_pool = tuple(s.strip() for s in args.splits.split(","))
        loader = build_holdout_loader(args.data_path, args.test_subj, captions_by_nsd_idx,
                                      args.batch_size, val_holdout=args.eval_holdout,
                                      return_image=True, seed=args.seed, splits=splits_pool)
        split_label = f"holdout({args.eval_holdout:.0%} of {splits_pool})"
        if args.test_subj == 8:
            print(f"[eval] WARNING: --eval_holdout 仅对训练过的被试（S1-7）有意义，S8 无同构留出划分")
    else:
        loader = build_test_loader(args.data_path, args.test_subj, captions_by_nsd_idx,
                                   args.batch_size, return_image=True, split=args.split)
        split_label = args.split
    total = len(loader.dataset)
    if args.max_trials:
        total = min(total, args.max_trials)
    print(f"S{args.test_subj:02d} {split_label} loader: {len(loader)} batches ({total} trials)")

    gallery, col_of = build_gallery(loader, clip, device)
    os.makedirs(args.out_dir, exist_ok=True)

    prompt = tokenizer.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    prompt_ids, prompt_attn = enc.input_ids.to(device), enc.attention_mask.to(device)
    Lp = prompt_ids.size(1)

    arms = ["few_shot", "zero_shot"] if args.arm == "both" else [args.arm]
    for arm in arms:
        use_ridge = load_arm_state(model, arm, args)
        enc_fn = make_enc_fn(model, retrieval_head, use_ridge, device)

        # 早期健全性：训练被试（S1/2/5/7）的 ridge 已训练，zero_cos 应 <0.2。
        # 若仍 ~0.84 = 编码器/ridge 实际没加载对（随机初始化）→ 整场结果无效，
        # 在跑昂贵的生成前先暴露，省得白烧一小时。
        early = run_diagnostics(enc_fn, loader, max_batches=args.diag_batches or 4)
        print(f"[{arm}] EARLY zero_cos={early['zero_cos']:.3f} "
              f"shuffle_cos={early['shuffle_cos']:.3f} (训练被试应 <0.2；"
              f"~0.84 = 编码器/ridge 加载失败或 S8 随机 ridge confound)")
        if args.test_subj in (1, 2, 5, 7) and early["zero_cos"] > 0.5:
            print(f"[{arm}] WARNING: 训练被试 zero_cos={early['zero_cos']:.3f} 异常偏高，"
                  f"本臂结果无效——检查上面各前缀的实际加载张量数", flush=True)

        hyps, nsd_idxs, queries, refs = [], [], [], []
        t0 = time.time()
        for i, batch in enumerate(loader):
            voxels = batch["voxels"].to(device)
            subj = batch["subj"]
            B = voxels.size(0)
            gen = model.generate(voxels, subj, input_ids=prompt_ids.expand(B, -1),
                                 attention_mask=prompt_attn.expand(B, -1),
                                 max_new_tokens=args.max_new, use_ridge=use_ridge,
                                 do_sample=False, num_beams=args.num_beams)
            hyps += [t.strip() for t in tokenizer.batch_decode(gen[:, Lp:], skip_special_tokens=True)]
            for idx in batch["nsd_idx"].tolist():
                nsd_idxs.append(idx)
                refs.append(captions_by_nsd_idx[idx] or [""])
            with torch.no_grad():
                q = enc_fn(voxels, subj)  # (B, 768) 归一化，与 trial 对齐
            queries.append(q)
            if args.max_trials and (i + 1) * B >= args.max_trials:
                break
            if (i + 1) % 10 == 0:
                print(f"[{arm}] {i + 1}/{len(loader)} batches | {(time.time() - t0) / 60:.1f}min",
                      flush=True)

        queries = [q for qq in queries for q in qq]  # 摊平成 (T, 768) 列表
        metrics = compute_text_metrics(hyps, refs, use_spice=not args.no_spice)
        ret = retrieval_metrics(queries, nsd_idxs, gallery, col_of)
        diag = run_diagnostics(enc_fn, loader, max_batches=args.diag_batches)

        retrieval_head_label = (f"within:{os.path.basename(args.within_retrieval_head)}"
                                if (args.within_retrieval_head and args.test_subj != 8)
                                else f"cross_subject(stage1):{os.path.basename(args.stage1)}")
        results = {"arm": arm, "n_trials": len(hyps), "use_ridge": use_ridge,
                   "retrieval_head": retrieval_head_label,
                   "text_metrics": metrics, "retrieval": ret, "diagnostics": diag}
        rpath = os.path.join(args.out_dir, f"{arm}_results.json")
        with open(rpath, "w") as f:
            json.dump(results, f, indent=2)
        hpath = os.path.join(args.out_dir, f"{arm}_hypotheses.json")
        with open(hpath, "w") as f:
            json.dump([{"nsd_idx": i, "hyp": h, "refs": r} for i, h, r in zip(nsd_idxs, hyps, refs)],
                      f, indent=2)
        print(f"[{arm}] text_metrics: {metrics}")
        print(f"[{arm}] retrieval: {ret}")
        print(f"[{arm}] diagnostics: {diag}")
        print(f"[{arm}] saved: {rpath} / {hpath}")
        if args.text_sens_trials > 0:
            run_text_sensitivity(model, loader, tokenizer, prompt_ids, prompt_attn, Lp,
                                 captions_by_nsd_idx, use_ridge, args, device, arm)


if __name__ == "__main__":
    main()
