"""扰动诊断（BIT-LLM 风格）：编码器对体素值敏感性与鲁棒性。

在「编码器 → (ridge) → ContrastiveHead → L2 归一化嵌入」路径上测量：
- Zero：全零体素 vs 真实 → 余弦（目标接近 0，表示依赖体素值而非仅位置/分区嵌入）。
  修复前的 MindLLM 编码器此值为 0.9889（BIT-LLM Table 8），Stage1 对比后降至 ~0。
- Shuffle：体内素打乱（保留边缘分布）vs 真实 → 余弦（低 = 依赖体素排布/结构）。
- Gaussian：加噪曲线（std 递增 → 余弦递减，显示退化率）。

enc_fn: callable(voxels, subj) → (B, D) 归一化嵌入；subj 为 1-based int，单被试 batch。
"""
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def _pairwise_cosine(enc_fn, real, perturb, subj):
    e_real = enc_fn(real, subj)
    e_pert = enc_fn(perturb, subj)
    return torch.nn.functional.cosine_similarity(e_real, e_pert, dim=-1)  # (B,)


@torch.no_grad()
def zero_cosine(enc_fn, voxels, subj):
    return _pairwise_cosine(enc_fn, voxels, torch.zeros_like(voxels), subj)


@torch.no_grad()
def shuffle_cosine(enc_fn, voxels, subj, seed=42):
    B, L = voxels.shape
    g = torch.Generator(device=voxels.device).manual_seed(seed)
    idx = torch.stack([torch.randperm(L, generator=g) for _ in range(B)])
    return _pairwise_cosine(enc_fn, voxels, voxels.gather(1, idx.to(voxels.device)), subj)


@torch.no_grad()
def gaussian_cosines(enc_fn, voxels, subj, stds=(0.01, 0.05, 0.1, 0.5)):
    out = []
    for s in stds:
        noise = torch.randn_like(voxels) * s
        out.append(_pairwise_cosine(enc_fn, voxels, voxels + noise, subj).mean().item())
    return out


@torch.no_grad()
def run_diagnostics(enc_fn, loader, max_batches=None):
    """遍历 loader 聚合诊断。loader 需产出 voxels/subj（见 build_test_loader）。"""
    zero_sum, shuf_sum, n = 0.0, 0.0, 0
    gauss_sum = [0.0] * 4
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        voxels = batch["voxels"]
        subj = batch["subj"]
        z = zero_cosine(enc_fn, voxels, subj)
        s = shuffle_cosine(enc_fn, voxels, subj)
        g = gaussian_cosines(enc_fn, voxels, subj)
        B = voxels.size(0)
        zero_sum += z.mean().item() * B
        shuf_sum += s.mean().item() * B
        for j in range(4):
            gauss_sum[j] += g[j] * B
        n += B
    return {
        "zero_cos": zero_sum / n,
        "shuffle_cos": shuf_sum / n,
        "gaussian_cos_std001": gauss_sum[0] / n,
        "gaussian_cos_std005": gauss_sum[1] / n,
        "gaussian_cos_std010": gauss_sum[2] / n,
        "gaussian_cos_std050": gauss_sum[3] / n,
    }
