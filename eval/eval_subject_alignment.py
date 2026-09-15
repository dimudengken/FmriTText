"""跨被试 subject-alignment 定位实验（专家 Exp 1/3/4，2026-09-05 v3）。

对 shared1000 同图，测不同被试（S1/S2 训练内控制 vs S1/S8 held-out）在同一张图上的
representation geometry，在 encoder 深度节点 / stage1 ContrastiveHead 定位跨被试错位。

判据（位置 ℓ，subject 对 a,b，共同图集 C）：
  within_same[a]      = 被试 a 内部同一张图 split-half 余弦（噪声上限）
  cross_same(a,b)     = a、b 对同一张图的质心余弦（对齐时 ≈ noise_ceiling）
  cross_diff(a,b)     = a、b 对不同随机图的质心余弦（随机基线）        [Exp 4 同图判别]
  cross MedR(a,b)     = a 每张图质心在 b 图库里的检索名次（随机 ≈ |C|/2）
  noise_ceiling       = sqrt(within_same[a] × within_same[b])（该对被试几何对齐的理论上限）
  align_eff           = cross_same / noise_ceiling（1.0 = 已达被试间噪声上限）
  gap_vs_within       = cross_same / within_same[a]（对齐保留比例）
  clip_cos[s]         = 被试 s head 输出 vs CLIP 图像嵌入余弦（head 位置个体锚定度）

v2 新增（专家 Exp 1/3）：
  token_mean_cos(a,b) = 逐 token 位置平均的跨被试余弦（encoder tokens，位置匹配）
    —— 与 global（mean-pool）cross_same 对比：global 高 + token 低 = 全局对齐掩盖
    token-level subject shift。注意位置匹配是下界：token 语义对应可能跨被试置换。
  subject_probe        = leave-one-out nearest-centroid 线性探针：encoder/head 输出上
    分被试精度（chance = 1/K）。≈100% = 输出里 subject 身份主导。

v3 新增（用户三项诊断，2026-09-05）：
  (1) 深度剖面（Key/Value/final-token 三层）：
      架构澄清（fmri_encoder.py:90-105 读码确认）——KeyValueEncoder 无独立 K/V token 流。
      Key = region_feature_project(脑区分区+位置) 是逐被试固定、与图像内容无关的解剖函数；
      Value = 原生 BOLD（S1≈15.7k vs S8≈13.0k 体素长度不同，原生空间不可向量比）；二者在
      neuro_informed_attn 里 softmax(query·K)·V matmul 纠缠。故最早可比点 = attn 输出。
      encode_stages() 复刻 forward，取四位置：
        readout (1024) = 神经科学注意力读出 = K×V 交互结果（subject shift 若源自解剖 K
                        selection，此处已现）
        mlp_out (1024) = 共享 4 层残差 MLP 后（shift 若在共享非线性处被放大，此处现）
        enc     (1024) = head Linear 展开 128 token 的 mean（= v1 的 enc 位置）
        head    (768)  = ContrastiveHead（CLIP 语义空间，= v1 的 head 位置）
      判读：readout 已低（≈final）→ shift 出生在 K 选 V 的读出端，后段共享层不放大；
            readout 高 + enc 低 → 共享 MLP/head 放大 subject shift（nonlinear）；
            各位置 cross_same 都 ≈ noise_ceiling×低 → 读出端本身噪声/解剖域不匹配。
  (2) Procrustes / 正交对齐诊断（--procrustes_pairs）：
      对 (a,b) 在 fit 一半 shared 图上拟合 a→b 的线性映射（正交 Procrustes R=UV^T 与
      LS 线性两版），held-out 半测 cross MedR。若 MedR 大幅降（247→<100）→ 几何旋转为主；
      几乎不降 → nonlinear / token-specific mismatch。注意 fit/test 是同一批图的不同被试
      trial = 只测映射线性度，不是到 unseen 图的零样本迁移。
  (3) S8 zero-shot CLIP retrieval 的 image-level 分布（category 标签当前数据没有，
      preprocessing.py 只给 nsdId→cocoId→captions）：
      brain→CLIP 图级检索（head 质心 query vs CLIP 图嵌入 gallery，仅 head 与 CLIP 同空间）
      + 逐图 clip_cos 分布（mean/median/p10/p90/top-decile/frac>阈值）+ S1↔S8 逐图相关
      （可读子集是否重合）+ top 可读图 ref captions（定性类别读法）。判读：S8 全分布统一
      偏低 ≈ S1×0.4 → "全局弱"；S8 top-decile 接近 S1 median 而多数≈0 → "少数图可读"。

架构澄清（v2/v3 一致性）：no_ridge 已去掉唯一 per-subject 线性层，readout 之后全共享；
subject-split 只能落在 readout 输出端（K/解剖读出行 or 共享权重未桥接 V 域）。

用法（服务器，fmri 环境，项目根目录；CLIP 本地路径 + HF_HUB_OFFLINE=1）：
  # 全量（S1/S2/S8，shared1000=new_test；默认含 Procrustes 1↔8 + CLIP retrieval）
  HF_HUB_OFFLINE=1 python eval/eval_subject_alignment.py \
      --data_path /root/autodl-tmp/MindEyeV2-main/data \
      --stage1 checkpoints/stage1/stage1_e84_noridge.pt \
      --clip /root/autodl-tmp/models/clip-vit-large-patch14
  # 冒烟：--subjects 1,8 --max_trials 300 --max_token_images 0
  # 公平 scratch-head 去混淆（train_head_frozen.py 产物；head 训在哪个几何评估须同传 bypass）：
  #   control（normal 几何 + mlpout head）：--probe_head head_mlpout.pt
  #   treatment（bypass 几何 + post head）：--probe_head head_post.pt --bypass_mlp
  # mode 2（post 几何 + SubjectRobustAdapter，train_head_frozen.py --adapter 产物）：
  #   cell A（无 GRL，隔离 adapter 结构本身）：--probe_head head_robust_noadv.pt \
  #       --adapter adapter_robust_noadv.pt --bypass_mlp --tag robust_noadv
  #   cell B（adapter + GRL 去主体，主 cell）：--probe_head head_robust_adv.pt \
  #       --adapter adapter_robust_adv.pt --bypass_mlp --tag robust_adv
输出：{out_dir}/subject_alignment.json
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

from data.dataloader import build_test_loader
from data.preprocessing import build_caption_map
from models.fmri_encoder import KeyValueEncoder

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

POSITIONS = ("readout", "mlp_out", "enc", "head")   # 深度剖面四位置（内容相关共享维度）


def l2norm(x):
    return F.normalize(x, dim=-1)


def load_encoder_head(ckpt_path, device, anatomy_dir):
    """no_ridge stage1 ckpt → encoder + ContrastiveHead（不经 ridge）。"""
    from training.stage1_contrastive import ContrastiveHead
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    encoder = KeyValueEncoder(anatomy_dir=anatomy_dir).to(device).eval()
    head = ContrastiveHead().to(device).eval()
    enc = {k.replace("encoder.", ""): v for k, v in ckpt.items() if k.startswith("encoder.")}
    hd = {k.replace("head.", ""): v for k, v in ckpt.items() if k.startswith("head.")}
    encoder.load_state_dict(enc, strict=False)
    head.load_state_dict(hd, strict=False)
    print(f"[align] encoder {len(enc)} tensors | head {len(hd)} (no_ridge 直测，无需 ridge)")
    if len(enc) == 0 or len(hd) == 0:
        raise RuntimeError(f"stage1 ckpt 加载 0 张量（enc {len(enc)}/head {len(hd)}），"
                           f"检查 {ckpt_path} 键前缀 encoder./head.（嵌套 last.pt 已自动解包）")
    return encoder, head


def load_probe_head(path, device):
    """加载 scratch ContrastiveHead（train_head_frozen.py 产物）替换原 stage1 head。

    兼容纯 ContrastiveHead state_dict（keys = proj.weight/bias）与带 "head." 前缀的 ckpt。
    0 张量直接 raise（防白测）。返回 (head, n_loaded)。
    """
    from training.stage1_contrastive import ContrastiveHead
    h = ContrastiveHead().to(device).eval()
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if any(k.startswith("head.") for k in ckpt):
        hd = {k.replace("head.", ""): v for k, v in ckpt.items() if k.startswith("head.")}
    else:
        allowed = set(h.state_dict())
        hd = {k: v for k, v in ckpt.items() if k in allowed}
    if not hd:
        raise RuntimeError(f"--probe_head {path} 加载 0 张量，检查是否 ContrastiveHead ckpt")
    h.load_state_dict(hd, strict=False)
    return h, len(hd)


def load_adapter(path, device):
    """加载 SubjectRobustAdapter state_dict（train_head_frozen.py --adapter 产物）。

    兼容纯 adapter state_dict（keys = norm.weight/bias, fc.weight, gate）与带 "adapter." 前缀的
    ckpt。0 张量直接 raise（防白测）。返回 (adapter, n_loaded)。
    """
    from models.subject_adapter import SubjectRobustAdapter
    a = SubjectRobustAdapter().to(device).eval()
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if any(k.startswith("adapter.") for k in ckpt):
        sd = {k.replace("adapter.", ""): v for k, v in ckpt.items() if k.startswith("adapter.")}
    else:
        allowed = set(a.state_dict())
        sd = {k: v for k, v in ckpt.items() if k in allowed}
    if not sd:
        raise RuntimeError(f"--adapter {path} 加载 0 张量，检查是否 SubjectRobustAdapter ckpt")
    a.load_state_dict(sd, strict=False)
    return a, len(sd)


def encode_stages(encoder, voxels, subj):
    """复刻 KeyValueEncoder.forward 的共享维度节点（eval/no_grad 数值等价 checkpoint 路径）。

    KeyValueEncoder 无独立 K/V token 流：K = region_feature_project(解剖+位置) 逐被试固定、
    与图像无关；V = 原生 BOLD；二者在 neuro_informed_attn 纠缠。最早可比点 = attn 输出。
    返回 (attn_out, post, mlp_out, tokens)：
      attn_out (B,1024)  neuro 注意力读出 = K×V 交互结果
      post     (B,1024)  LN+GELU(attn_out)（bypass 时即 head 输入）
      mlp_out  (B,1024)  共享 4 层残差 MLP 之后（bypass 时 ≡ post）
      tokens   (B,128,1024) head Linear 展开（与 encoder(voxels,subj) 逐位一致）
    bypass_mlp = True 时跳过 MLP（encoder.bypass_mlp 控制，两处分支须一致）。
    """
    B, L = voxels.shape
    key_feats = encoder._get_region_mask(subj).to(voxels.device)
    keys = encoder.region_feature_project(key_feats)
    keys = keys.unsqueeze(0).expand(B, -1, -1)
    attn_out = encoder.neuro_informed_attn(voxels, keys)     # K×V 读出
    post = encoder.neuro_informed_attn_post(attn_out)
    if encoder.bypass_mlp:
        mlp_out = post
    else:
        x = post
        residual = x
        for block in encoder.mlp:
            x = block(x) + residual
            residual = x
        mlp_out = x
    flat = encoder.head(mlp_out)
    toks = flat.reshape(B, encoder.n_fmri_tokens, encoder.token_dim)
    return attn_out, post, mlp_out, toks


def split_half_cos(vectors):
    """一被试一张图 n 个 trial 向量的 split-half 余弦（n>=2）。"""
    n = len(vectors)
    if n < 2:
        return None
    E = torch.stack(vectors)
    if n == 2:
        a, b = E[0], E[1]
    else:
        k = n // 2
        a = l2norm(E[:k].sum(0))
        b = l2norm(E[k:].sum(0))
    return float(F.cosine_similarity(a, b, dim=-1).item())


def collect_subject(subj, loader, encoder, head, clip, device,
                    max_trials=None, max_token_images=200, adapter=None):
    """跑完该被试 shared1000，返回 (per_nsd_means, within_same, token_centroids, diag)。

    means[n][pos]（pos ∈ POSITIONS）= l2norm 质心；means[n]["clip"] = CLIP 图嵌入（768 已归一）。
    within_same[pos] = 逐图 split-half 均值。token_centroids 仅保留 max_token_images 张图。
    diag = {cos_post_mlp}: MLP bypass 的 confound 量化 = 正常模式下 head 输入从 mlp_out
    挪到 post 会动多远（cos 越低 = bypass 越改变 head 输入分布 → 解释须留余地）。
    adapter 给定时：encode_stages 后先过 SubjectRobustAdapter 再进 enc/head/token 诊断
    （readout/mlp_out 保持冻结原样 = 测适配 token 流里的主体残留）。
    """
    per_pos_reps = {k: [] for k in POSITIONS}
    clip_embs, nsd_list = [], []
    clip_cache = {}
    tok_sum, tok_n = {}, {}   # capped 的逐 token 质心累加（token-level 诊断）
    t0 = time.time()
    n_done = 0
    cpm_sum, cpm_n = 0.0, 0    # cos(post, mlp_out) 累加（仅非 bypass 时有效）
    for i, batch in enumerate(loader):
        vox = batch["voxels"].to(device)
        nsds = batch["nsd_idx"].tolist()
        with torch.no_grad():
            attn_out, post, mlp_out, toks = encode_stages(encoder, vox, subj)
            if adapter is not None:
                toks = adapter(toks)         # 适配 token 流（共享权重，readout/mlp_out 不动）
            mean_tok = toks.mean(dim=1)      # (B,1024)
            h = head(toks)                   # (B,768) 已 L2 归一化
        per_pos_reps["readout"].append(l2norm(attn_out).cpu())
        per_pos_reps["mlp_out"].append(l2norm(mlp_out).cpu())
        per_pos_reps["enc"].append(l2norm(mean_tok).cpu())
        per_pos_reps["head"].append(h.cpu())
        if not encoder.bypass_mlp:
            cpm_sum += float(F.cosine_similarity(post, mlp_out, dim=-1).sum().item())
            cpm_n += post.size(0)
        if max_token_images > 0:
            tcpu = toks.cpu()                # (B,128,1024)
            for j, n in enumerate(nsds):
                if n in tok_sum:
                    tok_sum[n] = tok_sum[n] + tcpu[j]
                    tok_n[n] += 1
                elif len(tok_sum) < max_token_images:
                    tok_sum[n] = tcpu[j].clone()
                    tok_n[n] = 1
        need = [k for k, n in enumerate(nsds) if n not in clip_cache]
        if need:
            px = (batch["image"][need].float().to(device) - CLIP_MEAN.to(device)) / CLIP_STD.to(device)
            with torch.no_grad():
                ce = l2norm(clip.get_image_features(pixel_values=px)).cpu()
            for local, n in enumerate([nsds[k] for k in need]):
                clip_cache[n] = ce[local]
        clip_embs.append(torch.stack([clip_cache[n] for n in nsds]))
        nsd_list.extend(nsds)
        n_done += len(nsds)
        if max_trials and n_done >= max_trials:
            break
        if (i + 1) % 10 == 0:
            print(f"[align] S{subj} {i + 1}/{len(loader)} batches | {(time.time() - t0) / 60:.1f}min",
                  flush=True)

    reps = {k: torch.cat(v, 0) for k, v in per_pos_reps.items()}
    cr = torch.cat(clip_embs, 0)
    groups = {}
    for j, n in enumerate(nsd_list):
        groups.setdefault(n, []).append(j)
    means, within = {}, {k: [] for k in reps}
    for n, idxs in groups.items():
        entry = {"clip": cr[idxs[0]]}
        for k, R in reps.items():
            V = R[idxs]
            entry[k] = l2norm(V.sum(0))      # 质心（已归一）
            wh = split_half_cos([V[jj] for jj in range(len(idxs))])
            if wh is not None:
                within[k].append(wh)
        means[n] = entry
    for k in within:
        within[k] = (sum(within[k]) / len(within[k])) if within[k] else None
    tok_centroids = {}
    for n in tok_sum:
        c = l2norm(tok_sum[n] / tok_n[n])    # (128,1024)，逐 token 归一
        tok_centroids[n] = c
    diag = {"cos_post_mlp": (cpm_sum / cpm_n) if cpm_n else None}
    ws = " ".join(f"{k}={within[k]:.3f}" for k in POSITIONS if within[k] is not None)
    print(f"[align] S{subj}: {len(nsd_list)} trials / {len(means)} unique images "
          f"(token-level kept {len(tok_centroids)}) | within_same {ws}", flush=True)
    if diag["cos_post_mlp"] is not None:
        print(f"[align] S{subj} cos(post,mlp_out)={diag['cos_post_mlp']:.3f} "
              f"(MLP bypass 的 head 输入挪动幅度；越低 = confound 越大)", flush=True)
    return means, within, tok_centroids, diag


def token_metrics(a_tok, b_tok, common, n_pos=128):
    """逐 token 位置平均的跨被试余弦（位置匹配；下界）。"""
    common_tok = [i for i in common if i in a_tok and i in b_tok]
    if len(common_tok) < 5:
        return None
    tot = 0.0
    for i in common_tok:
        A = a_tok[i]  # (n_pos,1024) 行归一
        B = b_tok[i]
        tot += float((A * B).sum(1).mean().item())  # 128 个 token 位置的平均 cos
    return {"n_token_images": len(common_tok), "n_pos": n_pos,
            "token_mean_cos": tot / len(common_tok)}


def subject_probe(pos_means, subj_list, common):
    """encoder/head 输出上 leave-one-out nearest-centroid 分被试精度。

    pos_means[s] = {nsd: vec}；common 为各被试共同图。返回 {acc, chance, n_queries}。
    """
    if len(subj_list) < 2:
        return None
    V = {s: torch.stack([pos_means[s][i] for i in common]) for s in subj_list}
    sums = {s: V[s].sum(0) for s in subj_list}
    n_correct, n_q = 0, 0
    for s in subj_list:
        for idx in range(V[s].size(0)):
            q = V[s][idx]                                   # 已归一
            cent = {}
            for c in subj_list:
                if c == s:
                    cent[c] = l2norm((sums[c] - V[c][idx]).unsqueeze(0))[0]  # leave-one-out
                else:
                    cent[c] = l2norm(sums[c].unsqueeze(0))[0]
            scores = {c: float((q * cent[c]).sum().item()) for c in subj_list}
            pred = max(scores, key=scores.get)
            n_correct += (pred == s)
            n_q += 1
    return {"acc": n_correct / n_q, "chance": 1.0 / len(subj_list),
            "n_queries": n_q, "subjects": subj_list}


def pair_metrics(a_means, b_means, a_within, b_within, a_tok=None, b_tok=None,
                 n_diff=3000, seed=42):
    """对 (a,b) 的跨被试指标（POSITIONS 每位置）；b_means=None → 只算 a 的 within_diff。"""
    g = torch.Generator().manual_seed(seed)
    common = sorted(set(a_means) & set(b_means)) if b_means is not None else sorted(a_means)
    out = {}
    if b_means is not None:
        out["n_common_images"] = len(common)
        for pos in POSITIONS:
            A = torch.stack([a_means[i][pos] for i in common])   # (N,D)
            B = torch.stack([b_means[i][pos] for i in common])
            sim = A @ B.T                                        # (N,N)
            N = sim.size(0)
            diag = sim.diag()
            ranks = (sim > diag.unsqueeze(1)).sum(1) + 1
            cross_same = float(diag.mean().item())
            ia = torch.randint(0, N, (n_diff,), generator=g)
            ib = torch.randint(0, N, (n_diff,), generator=g)
            keep = ia != ib
            d = float(sim[ia[keep], ib[keep]].mean().item())
            wa = a_within[pos] if a_within else None
            wb = b_within[pos] if b_within else None
            nc = (wa * wb) ** 0.5 if wa and wb else None
            out[pos] = {
                "cross_same": cross_same, "cross_diff": d,
                "gap_vs_within": cross_same / wa if wa else None,
                "noise_ceiling": nc,
                "align_eff": cross_same / nc if nc else None,
                "MedR": float(ranks.median().item()),
                "R@1": float((ranks == 1).float().mean().item()),
                "R@10": float((ranks <= 10).float().mean().item()),
            }
        tm = token_metrics(a_tok, b_tok, common)
        if tm:
            out["token"] = tm
            if "enc" in out:
                out["token"]["global_same_enc"] = out["enc"]["cross_same"]
        for sname, m in (("a", a_means), ("b", b_means)):
            ccs = [float(F.cosine_similarity(m[i]["head"], m[i]["clip"], dim=0).item())
                   for i in common]
            out[f"clip_cos_{sname}"] = sum(ccs) / len(ccs)
    else:
        out["n_images"] = len(common)
        for pos in POSITIONS:
            V = torch.stack([a_means[i][pos] for i in common])
            N = V.size(0)
            ia = torch.randint(0, N, (n_diff,), generator=g)
            ib = torch.randint(0, N, (n_diff,), generator=g)
            keep = ia != ib
            sim = V @ V.T
            out[pos] = {"within_diff": float(sim[ia[keep], ib[keep]].mean().item())}
    return out


def procrustes_test(a_means, b_means, common, pos, fit_frac=0.5, seed=42):
    """正交 Procrustes / LS 线性：fit 半 shared 图 a→b 映射，held-out 半测检索 MedR。

    若 held-out MedR 相对 unaligned 大幅降 → subject 错位大部分是几何旋转（线性可救）；
    几乎不降 → nonlinear / token-specific mismatch。fit/test 同图不同被试 trial =
    只测映射线性度，非到 unseen 图迁移。
    """
    g = torch.Generator().manual_seed(seed)
    A = torch.stack([a_means[i][pos] for i in common])   # (N,D) 行已归一
    B = torch.stack([b_means[i][pos] for i in common])
    N = A.size(0)
    if N < 20:
        return None
    perm = torch.randperm(N, generator=g)
    nf = max(10, int(N * fit_frac))
    fit_i, test_i = perm[:nf], perm[nf:]
    Af, Bf = A[fit_i], B[fit_i]
    At, Bt = A[test_i], B[test_i]

    def retrieval(Q, gal):
        sim = Q @ gal.T
        diag = sim.diag()
        ranks = (sim > diag.unsqueeze(1)).sum(1) + 1
        return ranks

    r0 = retrieval(At, Bt)
    U, _, Vt = torch.linalg.svd(Af.T @ Bf)
    R_ortho = U @ Vt                                        # 正交 Procrustes
    r1 = retrieval(l2norm(At @ R_ortho), Bt)
    R_lin = torch.linalg.lstsq(Af, Bf).solution            # LS 线性（全等距外任意线性）
    r2 = retrieval(l2norm(At @ R_lin), Bt)

    def medr(r):
        return float(r.median().item())

    return {"n_fit": int(nf), "n_test": int(N - nf),
            "MedR_unaligned": medr(r0), "MedR_procrustes": medr(r1),
            "MedR_linear": medr(r2),
            "R@1_unaligned": float((r0 == 1).float().mean().item()),
            "R@1_procrustes": float((r1 == 1).float().mean().item()),
            "R@1_linear": float((r2 == 1).float().mean().item()),
            "cross_same_unaligned": float((At * Bt).sum(1).mean().item()),
            "cross_same_procrustes": float(((At @ R_ortho) * Bt).sum(1).mean().item()),
            "cross_same_linear": float((l2norm(At @ R_lin) * Bt).sum(1).mean().item()),
            "pos": pos}


def brain_to_clip_retrieval(means, common):
    """head 质心 query vs CLIP 图嵌入 gallery 的图级检索（仅 head 与 CLIP 同空间）。"""
    if len(common) < 5:
        return None
    C = torch.stack([means[i]["clip"] for i in common])    # (N,768) 已归一
    Q = torch.stack([means[i]["head"] for i in common])    # (N,768) 已归一
    sim = Q @ C.T
    diag = sim.diag()
    ranks = (sim > diag.unsqueeze(1)).sum(1) + 1
    return {"R@1": float((ranks == 1).float().mean().item()),
            "R@10": float((ranks <= 10).float().mean().item()),
            "MedR": float(ranks.median().item()),
            "n_gallery": len(common)}


def clip_cos_dist(means, subj_name):
    """逐图 head↔CLIP 余弦分布（image-level 可读性：全局弱 vs 少数可读）。"""
    keys = sorted(means)
    if len(keys) < 5:
        return None
    v = torch.stack([F.cosine_similarity(means[i]["head"], means[i]["clip"], dim=0)
                     for i in keys])
    sv, _ = torch.sort(v)
    n = v.numel()

    def pct(p):
        return float(sv[min(n - 1, max(0, int(n * p) - 1))])

    return {"subj": subj_name, "n_images": n,
            "mean": float(v.mean().item()), "median": float(v.median().item()),
            "p10": pct(0.10), "p90": pct(0.90),
            "top10_mean": float(sv[-n // 10:].mean().item()) if n >= 10 else float(v.mean().item()),
            "frac_gt_0p10": float((v > 0.10).float().mean().item()),
            "frac_gt_0p15": float((v > 0.15).float().mean().item()),
            "frac_gt_0p20": float((v > 0.20).float().mean().item()),
            }


def main():
    ap = argparse.ArgumentParser(description="跨被试 subject-alignment 定位实验（v3 深度剖面 + "
                                             "Procrustes + CLIP image-level）")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--stage1", default="checkpoints/stage1/stage1_encoder.pt")
    ap.add_argument("--clip", default="openai/clip-vit-large-patch14")
    ap.add_argument("--anatomy_dir", default="data/anatomy_cache")
    ap.add_argument("--out_dir", default="eval_results")
    ap.add_argument("--subjects", default="1,2,8",
                    help="逗号分隔：训练被试对照（如 1,2）+ held-out（8）。")
    ap.add_argument("--pairs", default="1-8,1-2",
                    help="跨被试对，逗号分隔（如 1-8 主对、1-2 训练内控制）。")
    ap.add_argument("--procrustes_pairs", default="1-8",
                    help="Procrustes/线性对齐诊断跑在这些对上（1-8 主对）。")
    ap.add_argument("--split", default="new_test", help="shared1000=new_test")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_trials", type=int, default=None, help="冒烟：每被试限总 trial 数")
    ap.add_argument("--max_token_images", type=int, default=200,
                    help="token-level 诊断保留的图数上限（控制内存，~200 图 ≈ 100MB/被试）")
    ap.add_argument("--n_diff_pairs", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bypass_mlp", action="store_true",
                    help="跳过 encoder 共享 4 层残差 MLP（linear-vs-nonlinear 消融）。head 未经 "
                         "readout 域调参 → confound，幅度见每被试 cos(post,mlp_out)。")
    ap.add_argument("--probe_head", default=None,
                    help="用 scratch ContrastiveHead（train_head_frozen.py 的 head_{tag}.pt）替换原 "
                         "stage1 head——公平测：head 训在哪个几何（bypass=post / normal=mlp_out），"
                         "评估就须同传 --bypass_mlp 匹配，否则离域。")
    ap.add_argument("--adapter", default=None,
                    help="用 SubjectRobustAdapter state_dict（train_head_frozen.py --adapter 产物 "
                         "adapter_{tag}.pt）插在 encoder 输出与 head 之间——适配 token 流再测 "
                         "enc/head/subject-probe。通常配 --bypass_mlp（adapter 训在 post 几何）。")
    ap.add_argument("--tag", default="",
                    help="输出文件后缀：subject_alignment{_tag}.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} | stage1={args.stage1} | clip={args.clip} | split={args.split}")

    encoder, head = load_encoder_head(args.stage1, device, args.anatomy_dir)
    head_label = "original"
    if args.probe_head:
        head, n_hd = load_probe_head(args.probe_head, device)
        head_label = f"probe:{os.path.basename(args.probe_head)}"
        print(f"[align] ** --probe_head：scratch ContrastiveHead 换入（{n_hd} 张量）"
              f"——head 训在哪个几何评估须同传 --bypass_mlp 匹配，否则离域")
    if args.bypass_mlp:
        encoder.bypass_mlp = True
        print("[align] ** MLP bypass 开启：跳过共享 4 层残差 MLP（head 未经 readout 域调参，"
              "confound 幅度看 cos_post_mlp——bypass 时 mlp_out≡post 该项无意义）")
    adapter = None
    if args.adapter:
        adapter, n_ad = load_adapter(args.adapter, device)
        print(f"[align] ** --adapter：SubjectRobustAdapter 插在 encoder 输出与 head 之间"
              f"（{n_ad} 张量）——enc/head/token 诊断测适配后 token 流，readout/mlp_out 保持原样")
        if not args.bypass_mlp:
            print("[align] ** WARNING：--adapter 通常配 --bypass_mlp（adapter 训在 post 几何）；"
                  "当前 normal 几何 = adapter 直接吃 mlp_out token，训练/eval 几何可能不一致")
    clip = CLIPModel.from_pretrained(args.clip).to(device).eval()
    for p in clip.parameters():
        p.requires_grad = False

    captions_by_nsd_idx = build_caption_map(args.data_path)
    subs = [int(s) for s in args.subjects.split(",")]

    collect, within, tok, hdiag = {}, {}, {}, {}
    for s in subs:
        loader = build_test_loader(args.data_path, s, captions_by_nsd_idx,
                                   args.batch_size, return_image=True, split=args.split)
        print(f"[align] S{s} loader: {len(loader)} batches ({args.split:>9})")
        collect[s], within[s], tok[s], hdiag[s] = collect_subject(
            s, loader, encoder, head, clip, device, args.max_trials, args.max_token_images,
            adapter)
    del clip
    if device == "cuda":
        torch.cuda.empty_cache()

    results = {"stage1": args.stage1, "split": args.split, "subjects": subs,
               "head": head_label,
               "adapter": (os.path.basename(args.adapter) if args.adapter else None),
               "bypass_mlp": args.bypass_mlp,
               "head_diag": {str(s): hdiag[s] for s in subs},
               "within_same": {str(s): within[s] for s in subs},
               "within_diff": {}, "pairs": {}}
    for s in subs:
        results["within_diff"][str(s)] = pair_metrics(collect[s], None, within[s], None)
    for pr in args.pairs.split(","):
        a, b = (int(x) for x in pr.split("-"))
        pm = pair_metrics(collect[a], collect[b], within[a], within[b],
                          tok.get(a), tok.get(b), args.n_diff_pairs, args.seed)
        results["pairs"][pr] = pm
        print(f"\n===== pair {a}↔{b} ({pm['n_common_images']} common images) =====")
        for pos in POSITIONS:
            m = pm[pos]
            print(f"  [{pos:>7}] cross_same={m['cross_same']:.3f} cross_diff={m['cross_diff']:.3f} "
                  f"| MedR={m['MedR']:.0f} R@1={m['R@1']:.3f} R@10={m['R@10']:.3f}")
            if m["noise_ceiling"]:
                print(f"            within a/b = {within[a][pos]:.3f}/{within[b][pos]:.3f} "
                      f"| ceiling={m['noise_ceiling']:.3f} align_eff={m['align_eff']:.2f} "
                      f"gap/within={m['gap_vs_within']:.2f}")
        if "token" in pm:
            t = pm["token"]
            print(f"  [token ] token_mean_cos={t['token_mean_cos']:.3f} "
                  f"(global_same_enc={t['global_same_enc']:.3f}, {t['n_token_images']} imgs) "
                  f"→ token/global 揭示全局是否掩盖 token shift")
        print(f"  clip_cos a(S{a})={pm['clip_cos_a']:.3f} b(S{b})={pm['clip_cos_b']:.3f}")

    # subject 探针：encoder/head 输出能多准分被试（chance = 1/K）
    if len(subs) >= 2:
        common_all = sorted(set.intersection(*[set(collect[s].keys()) for s in subs]))
        for pos in POSITIONS:
            pos_means = {s: {i: collect[s][i][pos] for i in common_all} for s in subs}
            sp = subject_probe(pos_means, subs, common_all)
            if sp:
                results.setdefault("subject_probe", {})[pos] = sp
                print(f"[probe {pos:>7}] subject acc={sp['acc']:.3f} "
                      f"(chance={sp['chance']:.2f}, {sp['n_queries']} queries)")

    # (2) Procrustes / 正交对齐诊断（held-out 半测 MedR 是否随线性映射大幅降）
    procrustes_out = {}
    for pr in args.procrustes_pairs.split(","):
        a, b = (int(x) for x in pr.split("-"))
        if b not in collect:
            continue
        common = sorted(set(collect[a]) & set(collect[b]))
        procrustes_out[pr] = {}
        for pos in POSITIONS:
            pt = procrustes_test(collect[a], collect[b], common, pos, seed=args.seed)
            if pt is None:
                continue
            procrustes_out[pr][pos] = pt
            print(f"\n[procrustes {a}→{b} @ {pos:>7}] unaligned MedR={pt['MedR_unaligned']:.0f} "
                  f"→ procrustes {pt['MedR_procrustes']:.0f} → linear {pt['MedR_linear']:.0f} "
                  f"(fit {pt['n_fit']}/test {pt['n_test']})")
            print(f"            cross_same: {pt['cross_same_unaligned']:.3f} → "
                  f"{pt['cross_same_procrustes']:.3f} → {pt['cross_same_linear']:.3f} | "
                  f"R@1: {pt['R@1_unaligned']:.3f}/{pt['R@1_procrustes']:.3f}/{pt['R@1_linear']:.3f}")
    results["procrustes"] = procrustes_out

    # (3) S8 zero-shot CLIP retrieval image-level 分布
    clip_out = {"retrieval": {}, "per_image_cos": {}, "pair_corr": {}}
    for s in subs:
        keys = sorted(collect[s])
        r = brain_to_clip_retrieval(collect[s], keys)
        d = clip_cos_dist(collect[s], s)
        clip_out["retrieval"][str(s)] = r
        clip_out["per_image_cos"][str(s)] = d
        if r:
            print(f"\n[clip retrieval S{s}] head→CLIP 图库: R@1={r['R@1']:.3f} "
                  f"R@10={r['R@10']:.3f} MedR={r['MedR']:.0f} (n={r['n_gallery']})")
        if d:
            print(f"[clip_cos S{s}] mean={d['mean']:.3f} med={d['median']:.3f} "
                  f"p10={d['p10']:.3f} p90={d['p90']:.3f} top10_mean={d['top10_mean']:.3f} | "
                  f"frac>0.10={d['frac_gt_0p10']:.3f} >0.15={d['frac_gt_0p15']:.3f} "
                  f">0.20={d['frac_gt_0p20']:.3f}")
    # 逐图相关（可读子集是否重合）+ top 可读图 caption（定性类别读法，category 标签当前无）
    for pr in args.pairs.split(","):
        a, b = (int(x) for x in pr.split("-"))
        common = sorted(set(collect[a]) & set(collect[b]))
        if len(common) < 5:
            continue
        x = torch.stack([F.cosine_similarity(collect[a][i]["head"], collect[a][i]["clip"], dim=0)
                         for i in common])
        y = torch.stack([F.cosine_similarity(collect[b][i]["head"], collect[b][i]["clip"], dim=0)
                         for i in common])
        pearson = float(torch.corrcoef(torch.stack([x, y]))[0, 1].item())
        rx = torch.argsort(torch.argsort(x)).float()
        ry = torch.argsort(torch.argsort(y)).float()
        spearman = float(torch.corrcoef(torch.stack([rx, ry]))[0, 1].item())
        clip_out["pair_corr"][pr] = {"pearson": pearson, "spearman": spearman,
                                     "n_images": len(common)}
        print(f"[clip_cos corr {a}↔{b}] pearson={pearson:.3f} spearman={spearman:.3f} "
              f"({len(common)} imgs) → 两被试可读子集{'重合' if spearman > 0.3 else '不重合/错位'}")
    # top 可读图 captions（S1 vs S8 各自最可读的 8 张）
    top_out = {}
    for s in subs:
        keys = sorted(collect[s])
        vals = torch.stack([F.cosine_similarity(collect[s][i]["head"], collect[s][i]["clip"], dim=0)
                            for i in keys])
        topk = torch.topk(vals, min(8, len(keys))).indices.tolist()
        top_out[str(s)] = [{"nsd": keys[j],
                            "clip_cos": round(float(vals[j].item()), 3),
                            "ref": (captions_by_nsd_idx[keys[j]] or ["(no caption)"])[0]}
                           for j in topk]
        print(f"\n[top readable S{s}]")
        for t in top_out[str(s)]:
            print(f"  nsd={t['nsd']} cos={t['clip_cos']:.3f} | {t['ref']}")
    clip_out["top_readable"] = top_out
    results["clip"] = clip_out

    os.makedirs(args.out_dir, exist_ok=True)
    fname = "subject_alignment" + (f"_{args.tag}" if args.tag else "") + ".json"
    rpath = os.path.join(args.out_dir, fname)
    with open(rpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[align] saved: {rpath}")


if __name__ == "__main__":
    main()
