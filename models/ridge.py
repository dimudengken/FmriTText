"""每被试 ridge 功能对齐层（少样本臂）。

CLAUDE.md：ridge 放 Key/Value 编码器之后、Projector 之前；形式 Linear(1024→1024)，
128 个 token 共享同一套权重（~1M/被试），勿做 per-token 独立。S8 微调时只训 S8 的 ridge。
"""
import torch
from torch import nn


class SubjectRidge(nn.Module):
    """每被试一个 Linear(dim→dim)，128 个 token 共享权重。"""

    def __init__(self, n_subjects=8, dim=1024):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_subjects)])

    def forward(self, x, subj):
        """x: (B, n_tokens, dim)；subj: int（1-based，单被试 batch）。"""
        if isinstance(subj, torch.Tensor):
            assert (subj == subj[0]).all(), "ridge 需单被试 batch"
            subj = int(subj[0].item())
        return self.linears[subj - 1](x)
