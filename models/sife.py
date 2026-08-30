"""SIFE（ZEBRA 零样本臂）：主体判别器 + 梯度反转层 + 表示保留重建锚点。

用途：预训练 S1-7 时把判别器当作对抗正则器——GRL 反转判别器梯度，迫使
编码器输出主体不变特征（跨被试一致）；重建锚点把特征投影回归一化体素，
防止对抗梯度破坏体素信号（表示保留）。

CLAUDE.md：零样本臂用 SIFE、不用 ridge；SIFE 只在预训练做正则化，S8 微调时关闭。
因此 SIFE 只挂在训练脚本里，不进入 BrainLLM 推理路径；零样本推理用
`BrainLLM(..., use_ridge=False)` 冻结编码器直出。

判别器读 pre-ridge 编码器特征（encoder 输出，1024 空间）——ridge 是被试特异层，
若让判别器看 ridge 之后的特征，主体信息已被显式对齐，判别器无从分类，对抗失效。
"""
import torch
import torch.nn.functional as F
from torch import nn


class _GradientReversal(torch.autograd.Function):
    """forward 恒等，backward 梯度乘以 -alpha。"""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    """梯度反转层：把梯度反转系数当模块参数，便于整体注册。"""

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return _GradientReversal.apply(x, self.alpha)


class SubjectDiscriminator(nn.Module):
    """从 pooled brain tokens 分类主体（1-8 分类）。"""

    def __init__(self, token_dim=1024, n_subjects=8, hidden=512, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_subjects),
        )

    def forward(self, features, subj):
        """features: (B, n_tokens, token_dim)；返回 CE loss（subj 为 1-based，单被试 batch）。"""
        pooled = features.mean(1)                       # (B, token_dim)
        logits = self.mlp(pooled)                       # (B, n_subjects)
        s = int(subj.item()) if isinstance(subj, torch.Tensor) else int(subj)
        labels = torch.full((logits.size(0),), s - 1, dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)


class ReconstructionHead(nn.Module):
    """表示保留锚点：从 pooled tokens 重建归一化体素，防止对抗训练破坏信号。

    每被试一个 Linear(token_dim → n_voxels)，按首次见到的 voxels 形状懒创建
    （体素数跨被试不同，13039~17907）。只用于训练期正则化，测试时丢弃。
    """

    def __init__(self, token_dim=1024):
        super().__init__()
        self.token_dim = token_dim
        self.norm = nn.LayerNorm(token_dim)
        self.heads = nn.ModuleDict()  # str(subj) -> Linear(token_dim, n_voxels)

    def forward(self, features, subj, voxels):
        """features: (B, n_tokens, token_dim)；voxels: (B, n_voxels)。返回 MSE loss。"""
        key = str(int(subj.item()) if isinstance(subj, torch.Tensor) else int(subj))
        if key not in self.heads:
            # 懒建头默认在 CPU，需跟随输入设备（GPU 训练时直接建会设备不匹配）
            self.heads[key] = nn.Linear(self.token_dim, voxels.shape[1]).to(features.device)
        pooled = features.mean(1)                                  # (B, token_dim)
        pred = self.heads[key](self.norm(pooled))                  # (B, n_voxels)
        return F.mse_loss(pred, voxels.float())


class SIFE(nn.Module):
    """主体判别器 + GRL + 重建锚点。forward 返回 (adv_loss, recon_loss)。"""

    def __init__(self, token_dim=1024, n_subjects=8, grl_alpha=1.0, disc_hidden=512):
        super().__init__()
        self.grl = GradientReversalLayer(alpha=grl_alpha)
        self.discriminator = SubjectDiscriminator(token_dim, n_subjects, disc_hidden)
        self.reconstruction = ReconstructionHead(token_dim)

    def forward(self, features, subj, voxels):
        """features 为 encoder 输出 (B, 128, 1024)（pre-ridge）。"""
        adv = self.discriminator(self.grl(features), subj)
        recon = self.reconstruction(features, subj, voxels)
        return adv, recon

    def loss(self, features, subj, voxels, lambda_adv=1.0, lambda_recon=10.0):
        adv, recon = self(features, subj, voxels)
        return lambda_adv * adv + lambda_recon * recon, adv, recon
