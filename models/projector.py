"""投影层：fMRI token → LLM 输入维度。

CLAUDE.md：Linear(1024→3072, no bias) + RMSNorm，128 个 brain tokens。
放 Key/Value 编码器 + 每被试 ridge 之后，Cross-Attention 之前。
"""
import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class Projector(nn.Module):
    """Linear(in_dim→out_dim, no bias) + RMSNorm。

    mode="rmsnorm"（默认）：Linear + RMSNorm。RMSNorm 逐 token 归一，抹掉每 token 的
    L2 幅值（只留方向）。
    mode="norm_mag"：保留 RMSNorm，把每 token 原始 L2 范数拼进末通道（out_dim 不变），
    幅值作为显式可学习信号通道（实验开关，默认关）。
    """

    def __init__(self, in_dim=1024, out_dim=3072, mode="rmsnorm"):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = RMSNorm(out_dim)
        self.mode = mode

    def forward(self, x):
        # x: (B, n_tokens, in_dim) → (B, n_tokens, out_dim)
        y = self.linear(x)
        if self.mode == "rmsnorm":
            return self.norm(y)
        mag = y.pow(2).mean(-1, keepdim=True).sqrt()   # (B, T, 1) 每 token L2 范数
        yn = self.norm(y)
        return torch.cat([yn[..., :-1], mag], dim=-1)  # 末通道=幅值，out_dim 不变
