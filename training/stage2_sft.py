"""Stage 2：SFT 监督微调（复刻 BIT-LLM Stage 2）。

冻结 encoder + ridge + LLM 主干，仅训练 Projector + Cross-Attn adapters。
response-only loss masking：prompt 位置 label=-100，只对 caption 部分算 CE。

数据流：
  voxels → encoder(冻结) → ridge(冻结) → projector(训) → cross-attn(训) → Frozen Llama → logits

早停策略：每被试随机划 val_holdout 比例做独立验证集（不碰 S8），每 val_freq 步
跑验证 response-only CE；验证最优即存 stage2_sft.pt（best），连续 patience 次
不创新低（且过 min_steps）即 early stop。last.pt 仍滚动存供 --resume 续训。

用法（服务器，fmri 环境，项目根目录）：
  python training/stage2_sft.py \
      --data_path /root/autodl-tmp/MindEyeV2-main/data \
      --llm /root/autodl-tmp/models/Llama-3.2-3B-Instruct \
      --resume_encoder checkpoints/stage1/stage1_encoder.pt
  # 快速冒烟：--epochs 1 --max_steps 100 --batch_size 8

输出：{out_dir}/stage2_sft.pt（验证最优的 projector + cross_attn 状态字典）
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from transformers import AutoTokenizer

from data.dataloader import build_train_val_loaders, collate_batch
from data.preprocessing import build_caption_map
from models.brain_llm import BrainLLM
from utils.save_load import load_train_ckpt, save_train_ckpt

SYSTEM = ("You are a helpful agent that decodes the brain activity of a person "
          "looking at an image. Output exactly ONE short caption. Do NOT ask for "
          "images or more info. Do NOT mention fMRI/brain/limitations. No self-reference.")
USER = "Describe the image as simply as possible."
MESSAGES = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": USER},
]


def batch_topk_retrieval(sim, k=1):
    """sim (B, B)：第 i 行真值在第 i 列。返回命中率。"""
    topk = torch.topk(sim, k=k, dim=1).indices
    hit = (topk == torch.arange(sim.size(0), device=sim.device).unsqueeze(1)).any(dim=1)
    return hit.float().mean().item()


def sft_collate(batch, tokenizer, max_len):
    """分词 prompt+caption，右侧 padding，response-only 的 labels。"""
    out = collate_batch(batch)
    input_ids_list, labels_list = [], []
    for cap in out["captions"]:
        prompt = tokenizer.apply_chat_template(MESSAGES, tokenize=False,
                                               add_generation_prompt=True)
        pids = tokenizer(prompt, add_special_tokens=False).input_ids
        cids = tokenizer(cap + tokenizer.eos_token, add_special_tokens=False).input_ids
        total = pids + cids
        if len(total) > max_len:
            # 从 prompt 头部截，保住 caption
            drop = len(total) - max_len
            pids = pids[drop:]
            total = pids + cids
        input_ids_list.append(total)
        labels_list.append([-100] * len(pids) + cids)

    L = max(len(x) for x in input_ids_list)
    input_ids = torch.full((len(batch), L), tokenizer.pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), L), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), L), dtype=torch.long)
    for i, (ids, lab) in enumerate(zip(input_ids_list, labels_list)):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        labels[i, : len(lab)] = torch.tensor(lab, dtype=torch.long)
        attn[i, : len(ids)] = 1

    out["input_ids"] = input_ids
    out["attention_mask"] = attn
    out["labels"] = labels
    return out


@torch.no_grad()
def run_validation(model, val_loader, device, max_batches=0, use_ridge=True):
    """验证集上聚合 response-only CE（与训练 CE 同口径）。返回平均 CE。"""
    model.eval()
    ce_sum, n = 0.0, 0
    for i, b in enumerate(val_loader):
        if max_batches and i >= max_batches:
            break
        v = b["voxels"].to(device)
        s = b["subj"]
        ids = b["input_ids"].to(device)
        m = b["attention_mask"].to(device)
        l = b["labels"].to(device)
        out = model(v, s, input_ids=ids, attention_mask=m, labels=l, use_ridge=use_ridge)
        ce_sum += out.loss.item() * v.size(0)
        n += v.size(0)
    model.train()
    return ce_sum / n if n else float("inf")


def main():
    ap = argparse.ArgumentParser(description="Stage 2 SFT（BIT-LLM 复刻）")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--llm", default="/root/autodl-tmp/models/Llama-3.2-3B-Instruct")
    ap.add_argument("--resume_encoder", default=None, help="Stage1 的 encoder.pt（可选）")
    ap.add_argument("--anatomy_dir", default="data/anatomy_cache")
    ap.add_argument("--out_dir", default="checkpoints/stage2")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--gate_init", type=float, default=0.08,
                    help="cross-attn 门控初值；0.08 偏小会把 projector 学习信号压死，可试 0.3")
    ap.add_argument("--aux_lambda", type=float, default=0.5,
                    help="辅助对齐损失权重（对齐 caption 语义；设为 0 关闭）")
    ap.add_argument("--aux_scale", type=float, default=10.0,
                    help="辅助 InfoNCE 的 logit 缩放")
    ap.add_argument("--lora", action="store_true",
                    help="给冻结 LLM 挂 LoRA(q/k/v/o) 一起训——冻结 LLM 无法向脑 token 靠拢，"
                         "LoRA 让 LLM 自适应，CE 梯度强且直接（论文 Stage 3 机制）")
    ap.add_argument("--lora_r", type=int, default=128)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--caption_emb", default=None,
                    help="预计算 caption 平均嵌入 .pt；缺省 CLIP 空间 {data_path}/caption_emb_mean_clip.pt；"
                         "LLM 空间用 caption_emb_mean_llm.pt（推荐：对齐冻结 LLM 自身语义空间）")
    ap.add_argument("--warmstart_projector", action="store_true",
                    help="projector 语义热启动：前 768 维 = stage1 head（CLIP 语义空间），其余压低到 1e-3；"
                         "aux_head 初始化为选择器。让脑 token 从 step 0 就带 stage1 级语义，"
                         "直接检验'接口能否消费语义 token'（诊断投影器 vs 消费端瓶颈）")
    # --- 四项低成本改进（2026-08-28，全部默认关，逐个实验，勿同开）---
    ap.add_argument("--o_proj_init", type=float, default=0.0,
                    help="cross-attn o_proj 高斯初始化 std（0=零初始化 baseline；>0 如 1e-3 "
                         "给训练早期投影器一条梯度回流通道）。看收敛后性能，不看早期曲线")
    ap.add_argument("--proj_mode", default="rmsnorm", choices=["rmsnorm", "norm_mag"],
                    help="projector 输出：rmsnorm=当前（抹幅值）；norm_mag=保留 RMSNorm + "
                         "末通道拼每 token L2 范数（幅值通道实验）。评估须同传 --proj_mode")
    ap.add_argument("--gate_mode", default="scalar", choices=["scalar", "per_token"],
                    help="cross-attn 门控：scalar=全局标量（当前）；per_token=逐脑 token "
                         "value gate（sigmoid 0-1）。评估须同传 --gate_mode")
    ap.add_argument("--gate_reg", type=float, default=0.0,
                    help="per_token gate 的 (g-0.5)² 正则权重（推离 0/1 两端防坍缩；0=关，"
                         "建议 0.1）")
    ap.add_argument("--sem_lambda", type=float, default=0.0,
                    help="语义继承 MSE 权重：projector pooled → stage1 ContrastiveHead 空间对齐"
                         "（0=关；建议 0.01~0.05；测此项时须 --aux_lambda 0 隔离旧 InfoNCE aux）")
    # --- 方案 A：CLIP 语义摘要 token（2026-08-30，默认关；与 --sem_lambda 共用 sem_head，勿同开）---
    ap.add_argument("--sem_kv", action="store_true",
                    help="把 projector 输出的 mean 作为第 129 个 KV token 注入 cross-attn"
                         "（语义摘要位置；softmax 可给 0 权重忽略噪声）。架构 flag，评估须同传 --sem_kv")
    ap.add_argument("--semclip_lambda", type=float, default=0.0,
                    help="sem_head 对真 CLIP 图像嵌入的 MSE 权重（0=关；建议 0.03；比 #4 的 "
                         "stage1_head 目标更干净；与 --sem_lambda 互斥，此 flag 优先）")
    ap.add_argument("--image_emb", default=None,
                    help="预计算图像 CLIP 嵌入 .pt；缺省 {data_path}/image_emb_mean_clip.pt"
                         "（(73000,768) fp32 L2 归一化，按 nsd_idx 索引）")
    # --- 数据划分复刻 BIT-LLM（2026-08-30）：S1-7 训练剔除 shared1000 ---
    ap.add_argument("--exclude_shared1000", action="store_true",
                    help="S1-7 训练只用 unique 图（split=train），剔除 shared1000（new_test）——"
                         "BIT-LLM 协议：S8 shared1000 成为干净 held-out，评估用 --split new_test")
    # --- 跨被试机制换型（2026-08-31，BIT-LLM 配方）---
    ap.add_argument("--no_ridge", action="store_true",
                    help="stage1 编码器用 --no_ridge 训（读出头全共享、无 per-subject ridge）时，"
                         "stage2 下游必须同步去 ridge：use_ridge=False 贯穿 forward/验证/diag。"
                         "否则 BrainLLM 的 ridge 随机初始化会把 pre-ridge 特征搅碎白测。"
                         "评估 run_eval 也须 --no_ridge 同传")
    ap.add_argument("--max_steps", type=int, default=None, help="快速冒烟用：跑够步数就停")
    ap.add_argument("--log_steps", type=int, default=10)
    ap.add_argument("--diag_batches", type=int, default=8,
                    help="聚合 diag 的 batch 数：单 batch(bs=16) 的 R@1/增益全是噪声，聚合 N 个 batch 出可信读数")
    ap.add_argument("--ckpt_freq", type=int, default=1000, help="每 N 步滚动保存 last.pt（断点续训）")
    ap.add_argument("--val_holdout", type=float, default=0.05,
                    help="每被试从训练数据随机划出的验证比例（不碰 S8）")
    ap.add_argument("--val_freq", type=int, default=500, help="每 N 步验证一次（判定 best/early stop）")
    ap.add_argument("--val_batches", type=int, default=0,
                    help="验证集最多跑 N batch；0 = 验证集全量")
    ap.add_argument("--patience", type=int, default=3,
                    help="验证 CE 连续 N 次不创新低即 early stop")
    ap.add_argument("--min_steps", type=int, default=0,
                    help="早停生效的最小步数（防早期误停）")
    ap.add_argument("--resume", default=None, help="从 {out_dir}/last.pt 续训")
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
                     gate_init=args.gate_init, torch_dtype=torch.bfloat16,
                     proj_mode=args.proj_mode, o_proj_init=args.o_proj_init,
                     gate_mode=args.gate_mode, sem_kv=args.sem_kv).to(device)
    if args.proj_mode != "rmsnorm" or args.o_proj_init or args.gate_mode != "scalar" \
            or args.sem_kv or args.no_ridge:
        print(f"[stage2] 改进 flag 生效：proj_mode={args.proj_mode} "
              f"o_proj_init={args.o_proj_init} gate_mode={args.gate_mode} "
              f"gate_reg={args.gate_reg} sem_lambda={args.sem_lambda} "
              f"sem_kv={args.sem_kv} semclip_lambda={args.semclip_lambda} "
              f"no_ridge={args.no_ridge}", flush=True)

    if args.resume_encoder:
        ckpt = torch.load(args.resume_encoder, map_location="cpu")
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]  # last.pt 嵌套 {"model":...} 解包（stage1_encoder.pt 为 plain）
        # 必须剥前缀：子模块 load_state_dict 只认无前缀键（"region_feature_project.weight"）。
        # 旧版把带 "encoder." 前缀的键直接传给子模块 → strict=False 静默跳过全部 → 编码器
        # 保持随机初始化 —— 这是所有 stage2 训练/评估结果全废的真正根因（"23 tensors" 是
        # 字典大小假象，实际加载 0）。数字打印改为 n_loaded/total，0/total 一眼可见。
        enc = {k[len("encoder."):]: v for k, v in ckpt.items() if k.startswith("encoder.")}
        rg = {k[len("ridge."):]: v for k, v in ckpt.items() if k.startswith("ridge.")}
        model.encoder.load_state_dict(enc, strict=False)
        model.ridge.load_state_dict(rg, strict=False)
        n_enc = sum(1 for k in enc if k in model.encoder.state_dict())
        n_rg = sum(1 for k in rg if k in model.ridge.state_dict())
        print(f"resumed encoder ({n_enc}/{len(enc)} tensors) + ridge ({n_rg}/{len(rg)} tensors)")
        if args.warmstart_projector:
            hw = ckpt.get("head.proj.weight")
            assert hw is not None, "--warmstart_projector 需要 stage1 ckpt 含 head.proj.weight"
            with torch.no_grad():
                model.projector.linear.weight.data[: hw.size(0)] = hw.float()
                model.projector.linear.weight.data[hw.size(0):].mul_(1e-3)
            print(f"warmstart projector: [0:{hw.size(0)}] = stage1 head, 其余压低 1e-3")

    # stage1 ContrastiveHead（Linear 1024→768，冻结）：startup cos 检查 + #4 语义继承
    # MSE 共用。只用 head.proj 裸线性（不带 L2norm），MSE 尺度才有意义。
    stage1_head = None
    if args.resume_encoder:
        _h_ckpt = torch.load(args.resume_encoder, map_location="cpu")
        if "model" in _h_ckpt and isinstance(_h_ckpt["model"], dict):
            _h_ckpt = _h_ckpt["model"]  # 同上：嵌套 last.pt 解包，head 键才在顶层
        if "head.proj.weight" in _h_ckpt:
            stage1_head = nn.Linear(_h_ckpt["head.proj.weight"].size(1),
                                    _h_ckpt["head.proj.weight"].size(0)).to(device).float()
            stage1_head.load_state_dict({"weight": _h_ckpt["head.proj.weight"].float(),
                                         "bias": _h_ckpt["head.proj.bias"].float()}, strict=False)
            for p in stage1_head.parameters():
                p.requires_grad = False
            print(f"[stage2] stage1 head 常驻：{stage1_head.weight.shape} (startup cos + sem_loss 共用)")

    # 冻结 encoder + ridge；只训 projector + cross_attn（LLM 冻结在 BrainLLM 内）
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.ridge.parameters():
        p.requires_grad = False

    # 可选：给冻结 LLM 挂 LoRA。证据链：纯 CE 信号太弱（o_proj 零初始化时 projector
    # 梯度 ~0.3），CLIP/LLM 空间 aux 都失败（前者 hijack projector、后者无可学习信号）。
    # LoRA 让 LLM 自身适应脑 token，CE 梯度强且直接，interface 才学得起来。
    if args.lora:
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                              lora_dropout=args.lora_dropout,
                              target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        model.llm = get_peft_model(model.llm, lora_cfg)
        for n, p in model.llm.named_parameters():
            p.requires_grad = "lora_" in n
        n_lora = sum(p.numel() for p in model.llm.parameters() if p.requires_grad)
        print(f"LoRA applied: {n_lora / 1e6:.1f}M trainable (q/k/v/o, r={args.lora_r})")

    # 可选辅助对齐损失（--aux_lambda 0 关闭）：离线预计算的 caption 嵌入作 projector
    # 的语义监督。注意：对齐空间必须和 LLM 需要什么空间一致，否则 CE 会变差。
    cap_emb = None
    model.aux_head = None
    if args.aux_lambda > 0:
        cap_path = args.caption_emb or os.path.join(args.data_path, "caption_emb_mean_clip.pt")
        cap_emb = torch.load(cap_path, map_location="cpu", weights_only=True).float()
        if cap_emb.size(1) == model.projector.linear.out_features:
            model.aux_head = None
            print(f"caption embeddings: {cap_emb.shape} | aux 无头，直接余弦对齐")
        else:
            model.aux_head = nn.Linear(model.projector.linear.out_features,
                                       cap_emb.size(1)).to(device)  # float32；输入统一 .float()
            print(f"caption embeddings: {cap_emb.shape} | aux_head: {model.aux_head.weight.shape}")
            if args.warmstart_projector and cap_emb.size(1) == 768:
                with torch.no_grad():
                    model.aux_head.weight.data.zero_()
                    model.aux_head.weight.data[: cap_emb.size(1), : cap_emb.size(1)] = torch.eye(768)
                    model.aux_head.bias.data.zero_()
                print("aux_head 初始化为选择器（取 projector 前 768 维 = stage1 head 语义）")

    # 语义继承 MSE：sem_head(projector pooled 3072) → CLIP 空间 (768)。目标二选一：
    #   #4   --sem_lambda>0      → stage1_head(enc_sem)（旧，目标本身带噪声）
    #   方案A --semclip_lambda>0 → 真 CLIP 图像嵌入（更干净，优先；二者互斥共用 sem_head）
    model.sem_head = None
    if args.sem_lambda > 0 or args.semclip_lambda > 0:
        if args.semclip_lambda > 0:
            model.sem_head = nn.Linear(model.projector.linear.out_features, 768).to(device).float()
            print(f"sem_head: {model.sem_head.weight.shape} | λ_semclip={args.semclip_lambda} "
                  f"(MSE 对齐真 CLIP 图像嵌入；建议 --aux_lambda 0 隔离旧 InfoNCE)", flush=True)
        else:
            assert stage1_head is not None, "--sem_lambda 需要 --resume_encoder 提供 stage1 head"
            model.sem_head = nn.Linear(model.projector.linear.out_features,
                                       stage1_head.weight.size(0)).to(device).float()
            print(f"sem_head: {model.sem_head.weight.shape} | λ_sem={args.sem_lambda} "
                  f"(MSE 对齐 stage1 head；测此项须 --aux_lambda 0 隔离旧 InfoNCE)", flush=True)

    # 方案 A 的真 CLIP 图像嵌入缓存（--semclip_lambda 0 关闭）：(73000,768) fp32 L2 归一化，
    # 按 nsd_idx 索引，由 training/precompute_image_emb.py 预计算。
    img_emb = None
    if args.semclip_lambda > 0:
        img_path = args.image_emb or os.path.join(args.data_path, "image_emb_mean_clip.pt")
        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"--semclip_lambda 需要图像 CLIP 嵌入缓存 {img_path}。"
                f"先用 python training/precompute_image_emb.py 生成（~10 min）。")
        img_emb = torch.load(img_path, map_location="cpu", weights_only=True).float()
        assert img_emb.dim() == 2 and img_emb.size(1) == 768, f"image_emb 形状应 (73000,768)，实际 {img_emb.shape}"
        print(f"image embeddings: {img_emb.shape} (L2 归一化, 按 nsd_idx)", flush=True)

    def aux_embed(sem):
        x = sem.float()
        if model.aux_head is not None:
            x = model.aux_head(x)
        return F.normalize(x, dim=-1)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable: {n_train / 1e6:.1f}M -> {trainable}")

    # 不用 gradient checkpointing：cross-attn 经 hook 注入，checkpoint 重算会把
    # brain tokens 当常量，projector 的梯度被静默吞掉（Stage2 里 projector 只喂
    # cross-attn，等于梯度归零）；且活张量被重算二次 backward 会直接报错。
    # bs=32 不 checkpoint 实测峰值 ~28GB（2026-08-29 OOM 实测），32GB 卡勉强、波动即崩；
    # 用 bs=16×grad_accum=4（有效 batch 64 不变，训练动力学等价）。
    model.train()

    captions_by_nsd_idx = build_caption_map(args.data_path)
    collate = partial(sft_collate, tokenizer=tokenizer, max_len=args.max_len)
    tr_splits = ("train",) if args.exclude_shared1000 else ("train", "new_test")
    loader, val_loader = build_train_val_loaders(
        args.data_path, list(range(1, 8)), captions_by_nsd_idx,
        args.batch_size, val_holdout=args.val_holdout,
        return_image=False, collate_fn=collate, seed=args.seed, splits=tr_splits)
    print(f"train loader: {len(loader)} | val loader: {len(val_loader)} "
          f"(holdout {args.val_holdout:.0%}/被试, 每 {args.val_freq} 步验证, "
          f"splits={tr_splits})")

    opt = AdamW(model.parameters(), lr=args.lr)

    def _gn(mod):
        tot = sum(p.grad.norm().item() ** 2 for p in mod.parameters()
                  if p.grad is not None)
        return tot ** 0.5

    pgn_sum, cgn_sum, gn_n = 0.0, 0.0, 0

    start_epoch, global_step = 0, 0
    if args.resume:
        start_epoch, global_step = load_train_ckpt(model, opt, args.resume)
        print(f"resumed from {args.resume}: epoch {start_epoch}, step {global_step}")
        # 防御：续训静默丢 LoRA（2026-08-28 实测踩坑）。last.pt 含 lora_* 但本次
        # 未挂 --lora 时，load_train_ckpt 的 strict=False 会静默丢弃 LoRA 权重，接口
        # 比旧权重弱一截（S8 CIDEr 0.099 vs 旧带 LoRA 0.121）。
        _lk = torch.load(args.resume, map_location="cpu", weights_only=True)
        _st = _lk.get("model", _lk)
        _has_lora = any("lora_" in k for k in _st)
        if _has_lora and not args.lora:
            print("WARNING: last.pt 含 lora_* 权重但本次未挂 --lora → LoRA 被静默丢弃，"
                  "接口会比旧权重弱一截。续训带 LoRA 的 run 必须加 --lora。", flush=True)
        elif not _has_lora and args.lora:
            print("INFO: last.pt 无 lora_* 权重，本次 --lora 从头挂新 LoRA。", flush=True)

    # 关键保护：load_train_ckpt 会把 last.pt 的全部非 llm 权重（含 encoder.*/ridge.*）
    # 灌进模型，静默覆盖上面 --resume_encoder 刚加载的 stage1 编码器 → 冻结的编码器变成
    # 随机权重，整个 stage2 run 白训（既往所有 stage2 ckpt 的编码器全是随机的根因）。
    # 编码器+ridge 冻结，只该来自 stage1；resume 只接 projector/cross_attn/LoRA。
    if args.resume and args.resume_encoder:
        _ckpt = torch.load(args.resume_encoder, map_location="cpu")
        _enc = {k[len("encoder."):]: v for k, v in _ckpt.items() if k.startswith("encoder.")}
        _rg = {k[len("ridge."):]: v for k, v in _ckpt.items() if k.startswith("ridge.")}
        model.encoder.load_state_dict(_enc, strict=False)
        model.ridge.load_state_dict(_rg, strict=False)
        print("[stage2] --resume 之后重载 stage1 encoder+ridge（冻结权重不被 resume 覆盖）")

    # 启动健全性：编码器是否对体素值敏感。走完整路径 encoder→ridge→head（stage1 的
    # ContrastiveHead，fp32）测 cos(real,zero)，与 diag_stage1 同口径：<0.3 好；
    # ~0.8 = 随机/未带 stage1 编码器。注意 raw-token 层级（好编码器 ~0.70）是弱信号、
    # 不是判据——判别力由训练 head 放大，必须带上 head 测。
    model.eval()
    _cz_sum, _cz_n = 0.0, 0
    with torch.no_grad():
        for _i, _b in enumerate(loader):
            if _i >= 3:
                break
            _v = _b["voxels"].to(device)
            _s = _b["subj"]
            _r = model.encoder(_v, _s)
            _z = model.encoder(torch.zeros_like(_v), _s)
            if not args.no_ridge:
                _r, _z = model.ridge(_r, _s), model.ridge(_z, _s)
            _r, _z = _r.mean(1).float(), _z.mean(1).float()
            if stage1_head is not None:
                _r, _z = stage1_head(_r), stage1_head(_z)
            _cz_sum += F.cosine_similarity(_r, _z, dim=-1).mean().item() * _v.size(0)
            _cz_n += _v.size(0)
    model.train()
    _cz = _cz_sum / _cz_n
    _path = "encoder→head" if args.no_ridge else "encoder→ridge→head"
    print(f"[stage2] STARTUP encoder cos(real,zero)={_cz:.3f} "
          f"({_path}, fp32；<0.3 好；~0.8 = 随机/未带 stage1 编码器)")
    if _cz > 0.5:
        print("WARNING: 编码器对体素值不敏感，stage2 训练无脑信号可学。确认 "
              "--resume_encoder 指向 stage1_encoder.pt，且未被 --resume 覆盖。")

    opt.zero_grad()
    running_loss = 0.0
    t0 = time.time()
    best_val = float("inf")
    patience_hits = 0
    saved_best = False
    for epoch in range(start_epoch, args.epochs):
        for batch in loader:
            voxels = batch["voxels"].to(device)
            subj = batch["subj"]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out, sem, enc_sem = model(voxels, subj, input_ids=input_ids,
                                      attention_mask=attention_mask, labels=labels,
                                      return_brain_sem=True, use_ridge=not args.no_ridge)
            ce = out.loss / args.grad_accum
            # 辅助对齐：brain 语义汇总 vs caption CLIP 嵌入（batch 内 InfoNCE）
            bs = sem.size(0)
            if cap_emb is not None:
                brain_sem = aux_embed(sem)
                cap_sem = F.normalize(cap_emb[batch["nsd_idx"]].to(device).float(), dim=-1)
                if (cap_sem.norm(dim=-1) > 0).all():
                    sim = brain_sem @ cap_sem.T * args.aux_scale
                    aux = F.cross_entropy(sim, torch.arange(bs, device=device)) / args.grad_accum
                else:
                    aux = torch.zeros((), device=device)
                loss = ce + args.aux_lambda * aux
            else:
                loss = ce
            # 语义继承 MSE：目标二选一（semclip 优先，二者互斥共用 sem_head）
            if model.sem_head is not None:
                if args.semclip_lambda > 0:      # 方案 A：对真 CLIP 图像嵌入（pred 先归一化 ≈ 余弦距离）
                    sem_target = img_emb[batch["nsd_idx"]].to(device).float()            # (B,768) 已归一化
                    sem_pred = F.normalize(model.sem_head(sem.float()), dim=-1)
                    loss = loss + args.semclip_lambda * F.mse_loss(sem_pred, sem_target) / args.grad_accum
                elif args.sem_lambda > 0:        # #4：对 stage1 head 空间（target 冻结）
                    sem_target = stage1_head(enc_sem.float()).detach()                   # (B, 768)
                    loss = loss + args.sem_lambda * F.mse_loss(model.sem_head(sem.float()), sem_target) / args.grad_accum
            # #3 per-token gate 防坍缩正则：(g-0.5)² 推离 0/1（仅 gate_mode=per_token 生效）
            if args.gate_reg > 0:
                gregs = [(model.cross_attn[i].last_gate - 0.5).pow(2).mean()
                         for i in model.cross_attn.layer_indices
                         if getattr(model.cross_attn[i], "last_gate", None) is not None]
                if gregs:
                    loss = loss + args.gate_reg * torch.stack(gregs).mean()
            loss.backward()
            pgn_sum += _gn(model.projector)
            cgn_sum += _gn(model.cross_attn)
            gn_n += 1

            running_loss += loss.item() * args.grad_accum
            if (global_step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()

            global_step += 1
            if global_step % args.val_freq == 0:
                val_ce = run_validation(model, val_loader, device, args.val_batches,
                                        use_ridge=not args.no_ridge)
                if val_ce < best_val - 1e-4:
                    best_val = val_ce
                    patience_hits = 0
                    os.makedirs(args.out_dir, exist_ok=True)
                    _best_ckpt = {k: v.detach().cpu() for k, v in model.state_dict().items()
                                  if not k.startswith("llm.") or "lora_" in k}
                    torch.save(_best_ckpt, os.path.join(args.out_dir, "stage2_sft.pt"))
                    saved_best = True
                    print(f"  [val] CE={val_ce:.4f} | best={best_val:.4f} | 存 best → stage2_sft.pt",
                          flush=True)
                else:
                    patience_hits += 1
                    print(f"  [val] CE={val_ce:.4f} | best={best_val:.4f} | "
                          f"patience={patience_hits}/{args.patience}", flush=True)
                if patience_hits >= args.patience and global_step >= args.min_steps:
                    print(f"[early stop] 验证 CE 连续 {args.patience} 次未创新低，停在第 "
                          f"{global_step} 步 (best CE={best_val:.4f})", flush=True)
                    break
            if global_step % args.ckpt_freq == 0:
                save_train_ckpt(model, opt, epoch, global_step,
                                os.path.join(args.out_dir, "last.pt"))
                # 聚合 diag：单 batch(bs=16) 的 R@1/接口增益全是噪声（R@1 0.06↔0.25、
                # 增益 +0.02↔+0.17 乱跳），用连续 N 个 batch 聚合出可信读数
                # （R@1 在 ~128 图上才有意义，增益噪声随 √N 下降）。
                with torch.no_grad():
                    ce_r = ce_z = 0.0
                    hit_r = hit_z = n_img = 0
                    diter = iter(loader)
                    for _ in range(args.diag_batches):
                        db = next(diter)
                        dv = db["voxels"].to(device)
                        ds = db["subj"]
                        di = db["input_ids"].to(device)
                        da = db["attention_mask"].to(device)
                        dl = db["labels"].to(device)
                        dn = db["nsd_idx"]
                        r_out, b_sem, _ = model(dv, ds, input_ids=di, attention_mask=da,
                                                labels=dl, return_brain_sem=True,
                                                use_ridge=not args.no_ridge)
                        z_out, z_sem, _ = model(torch.zeros_like(dv), ds, input_ids=di,
                                                attention_mask=da, labels=dl,
                                                return_brain_sem=True,
                                                use_ridge=not args.no_ridge)
                        n_img += b_sem.size(0)
                        ce_r += r_out.loss.item() * b_sem.size(0)
                        ce_z += z_out.loss.item() * b_sem.size(0)
                        if cap_emb is not None:
                            b_real = aux_embed(b_sem.detach())
                            b_zero = aux_embed(z_sem)
                            cs = F.normalize(cap_emb[dn].to(device).float(), dim=-1)
                            hit_r += int(batch_topk_retrieval(b_real @ cs.T) * b_sem.size(0))
                            hit_z += int(batch_topk_retrieval(b_zero @ cs.T) * b_sem.size(0))
                print(f"  [diag] CE(real)={ce_r / n_img:.4f} | CE(zero)={ce_z / n_img:.4f} "
                      f"| 接口增益={(ce_z - ce_r) / n_img:+.4f} (n={n_img})", flush=True)
                if cap_emb is not None:
                    print(f"  [diag] aux R@1 real→cap={hit_r / n_img:.3f} | zero→cap={hit_z / n_img:.3f} "
                          f"| 语义分离={(hit_r - hit_z) / n_img:+.3f}", flush=True)
                # gate 是否在增长 = 模型是否承诺使用脑 tokens（初始 0.08，涨起来才健康）
                # per_token 模式无 self.gate 参数（被 gate_net 取代），报 last_gate 均值
                gates = []
                for i in (3, 7, 11, 15):
                    _ad = model.cross_attn[i]
                    if hasattr(_ad, "gate"):
                        gates.append(_ad.gate.item())
                    else:
                        _lg = _ad.last_gate
                        gates.append(_lg.mean().item() if _lg is not None else float("nan"))
                print(f"  [diag] gate={['%.4f' % g for g in gates]} "
                      f"(mean={sum(gates) / len(gates):.4f})", flush=True)
                # 梯度健康度：最近窗口 projector/cross_attn 的平均梯度范数
                # （在 zero_grad 前逐 step 累计，避免读到清空后的 0）
                print(f"  [diag] grad_norm(avg) projector={pgn_sum / max(gn_n, 1):.3e} "
                      f"| cross_attn={cgn_sum / max(gn_n, 1):.3e}", flush=True)
                # 释放 diag 前向预留的显存块，避免碎片累积诱发后续 backward OOM
                torch.cuda.empty_cache()
            if global_step % args.log_steps == 0:
                print(f"[epoch {epoch}] step {global_step} | loss {running_loss / args.log_steps:.4f} "
                      f"| {(time.time() - t0) / 60:.1f}min", flush=True)
                running_loss = 0.0

            if args.max_steps and global_step >= args.max_steps:
                break
        else:
            continue
        break

    if not saved_best:
        # 从未跑过验证（如 max_steps < val_freq 的冒烟）→ 兜底存当前权重
        os.makedirs(args.out_dir, exist_ok=True)
        ckpt = {k: v.detach().cpu() for k, v in model.state_dict().items()
                if not k.startswith("llm.") or "lora_" in k}
        path = os.path.join(args.out_dir, "stage2_sft.pt")
        torch.save(ckpt, path)
        print(f"saved: {path}")
    else:
        print(f"stage2_sft.pt = 验证最优 (CE={best_val:.4f})，训练中已保存；last.pt 供 --resume 续训")


if __name__ == "__main__":
    main()
