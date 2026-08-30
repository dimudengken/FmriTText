"""MindLLM 风格的 Key/Value 解耦 fMRI 编码器。

Key = [脑区分区特征（V1/V2/V3/V4/early_vis/higher_vis 6 类）+ Fourier 体素位置编码] 经
Linear 投影，Value = BOLD 信号。分区特征由 data/anatomy.py 从 brain_region_masks.hdf5
提取并缓存；位置编码逐体素，使注意力权重随体素变化、保留区内精细结构。

为什么必须有位置编码：仅用 6 个脑区掩膜时，keys 只有 ~6 个不同取值，注意力输出退化为
6 个区域均值的加权和，每个 trial 只剩 ~6 个统计量，对比学习撞上信息天花板而走平
（诊断实测：stage1 loss 在随机上方 0.2 处钉死）。逐体素位置编码解掉这个瓶颈。

形状：输入 (B, n_voxels) → 输出 (B, n_fmri_tokens, token_dim) = (B, 128, 1024)。

注意：forward 假设 batch 内为同一被试（体素数一致、分区特征一致）。混合被试需按 subj 分组。
"""
import os
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class NeuroscienceInformedAttentionLayer(nn.Module):
    """learnable query × 体素 Key 交叉注意力，Value 为 BOLD。"""

    def __init__(self, size, rank):
        super().__init__()
        self.query_embeddings = nn.Parameter(torch.empty(1, size, rank))
        nn.init.xavier_uniform_(self.query_embeddings)

    def forward(self, values, keys):
        # values (B, L)；keys (B, L, rank)。attention 在 bf16 下算：weights (B, size, L)
        # 是内存大户（bs=128 时 fp32 ~7.3GB），bf16 减半、backward 峰值 ~7.4GB 才装得下；
        # bf16 有 ~8 位尾数，softmax 分布精度损失可忽略。输出回 fp32 供下游。
        values = values.bfloat16()
        keys = keys.bfloat16()
        query = self.query_embeddings.bfloat16().expand(values.size(0), -1, -1)  # (B, size, rank)
        weights = (query @ keys.transpose(1, 2)).softmax(-1)  # (B, size, L)
        out = values.unsqueeze(-2) @ weights.transpose(1, 2)  # (B, size, 1)
        return out.squeeze(1).float()  # (B, size)


class KeyValueEncoder(nn.Module):
    """Key/Value 解耦编码器（MindLLM NeuroscienceInformedAttention 的适配版）。"""

    def __init__(self, n_fmri_tokens=128, token_dim=1024, rank=128, hidden_dim=1024,
                 n_mlp_layers=4, n_regions=6, n_pos=32, anatomy_dir="data/anatomy_cache",
                 norm_type="ln"):
        super().__init__()
        self.n_fmri_tokens = n_fmri_tokens
        self.token_dim = token_dim
        self.rank = rank
        self.n_pos = n_pos
        self.anatomy_dir = anatomy_dir
        out_dim = n_fmri_tokens * token_dim

        norm_func = (partial(nn.BatchNorm1d, num_features=hidden_dim) if norm_type == "bn"
                     else partial(nn.LayerNorm, normalized_shape=hidden_dim))
        act_fn = partial(nn.ReLU, inplace=True) if norm_type == "bn" else nn.GELU

        # Key 投影：[分区特征 (n_regions,) + Fourier 位置 (n_pos,)] → rank
        self.region_feature_project = nn.Linear(n_regions + n_pos, rank)

        self.neuro_informed_attn = NeuroscienceInformedAttentionLayer(size=hidden_dim, rank=rank)
        self.neuro_informed_attn_post = nn.Sequential(norm_func(), act_fn(), nn.Dropout(0.5))

        self.mlp = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), norm_func(), act_fn(), nn.Dropout(0.15))
            for _ in range(n_mlp_layers)
        ])
        self.head = nn.Linear(hidden_dim, out_dim)

        self._key_feats = {}  # subj -> tensor (n_voxels, n_regions + n_pos)

    def _get_region_mask(self, subj):
        if subj not in self._key_feats:
            npz = np.load(os.path.join(self.anatomy_dir, f"subj0{subj}_anatomy.npz"))
            region = torch.from_numpy(npz["region_mask"].astype(np.float32))  # (L, n_regions)
            npz.close()
            n = region.size(0)
            # Fourier 逐体素位置编码（保留区内精细结构，解 6 脑区均值的信息天花板）
            idx = torch.arange(n, dtype=torch.float32) / n
            freqs = 2 * torch.pi * torch.arange(1, self.n_pos // 2 + 1, dtype=torch.float32)
            ang = idx.unsqueeze(1) * freqs.unsqueeze(0)  # (L, n_pos/2)
            pos = torch.cat([ang.sin(), ang.cos()], dim=1)  # (L, n_pos)
            self._key_feats[subj] = torch.cat([region, pos], dim=1)
        return self._key_feats[subj]

    def forward(self, voxels, subj):
        B, L = voxels.shape
        key_feats = self._get_region_mask(subj).to(voxels.device)  # (L, n_regions + n_pos)
        keys = self.region_feature_project(key_feats)  # (L, rank)
        keys = keys.unsqueeze(0).expand(B, -1, -1)  # (B, L, rank)

        # gradient checkpoint：注意力权重 (B, size, L) 太大（L≈14k 体素），backward 同时
        # 持有 logits/权重/多份梯度会顶穿显存；重算而非保留，把峰值砍到 ~1/3。
        x = checkpoint(self.neuro_informed_attn, voxels, keys, use_reentrant=False)
        x = self.neuro_informed_attn_post(x)
        residual = x
        for block in self.mlp:
            x = block(x) + residual
            residual = x
        x = self.head(x)  # (B, 128 * token_dim)
        return x.reshape(B, self.n_fmri_tokens, self.token_dim)
