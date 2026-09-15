"""Subject-Robust Adapter：post-readout 门控适配 + 显式去主体（MLP-bypass 后续实验）。

架构（2026-09-08，见 CLAUDE.md"readout Subject-Robust Adapter"节；专家图）：
  encoder(post 几何, frozen) ── tokens [128,1024]
      → SubjectRobustAdapter（norm → gated linear → residual，token-wise 共享）
      → Cross-subject tokens [128,1024]
          ├─ Semantic branch：ContrastiveHead mean-pool → Linear → CLIP（InfoNCE 锚）
          └─ LLM branch：stage2 Projector → Cross-Attn → LLM（本探针实验不涉及）

动机（数据证据）：
  * MLP-bypass matched scratch-head 双 cell：post 几何 + 匹配 head 把 S8 零样本 clip MedR
    拉到 175（control 319），但残余主体位移仍在（subject_probe head 0.442 > chance、
    pair 1↔8 head MedR 167 仍 3-5× S1 内量级）。
  * 隐式 InfoNCE 不消除主体身份：stage1 就是 mixed-batch InfoNCE，训完 mlp_out 上 subject
    probe 仍 0.965 → 跨被试对齐不能靠隐式 CLIP 锚，必须显式对抗去主体。
  * procrustes 旁证：bypass 后残余是"近线性每被试旋转"（拟合对齐 MedR 61-63）——共享
    SubjectRobustAdapter 能否在零样本消掉它取决于 S8 的旋转是否落在 S1-7 学到的公共主体
    子空间（本实验要测的正是这个假说）。

实现：
  * SubjectRobustAdapter：对每个 brain token（共享同一套权重）做
      y = x + sigmoid(gate) ⊙ tanh(Linear(LayerNorm(x)))
    gate 初始化 logit(0.1) ≈ sigmoid=0.1 → 起于近恒等，训练中学非均匀门控注入。
    LayerNorm 逐 token 归一（顺带去掉每被试的 domain scale/shift）。
  * SubjectProbe：pooled 适配 token（mean over 128 → normalize）上线性主体分类器，前接
    GradientReversalLayer（复用 models/sife.py）→ 对抗梯度迫使适配输出在 pooled 特征上
    线性不可分主体。CE loss 按混合批逐行 subject 标签。
  判别语义分支 InputNCE 由外部训练脚本提供（BrainContrastiveLoss）。
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from models.sife import GradientReversalLayer


class SubjectRobustAdapter(nn.Module):
    """token-wise 门控适配：y = x + sigmoid(gate) ⊙ tanh(Linear(LayerNorm(x)))。"""

    def __init__(self, d=1024, gate_init=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc = nn.Linear(d, d, bias=False)
        self.act = nn.Tanh()
        self.gate = nn.Parameter(torch.full((d,), math.log(gate_init / (1.0 - gate_init))))

    def forward(self, x):
        # x: (B, 128, d)；gate 逐通道、128 token 与 batch 共享
        g = torch.sigmoid(self.gate).view(1, 1, -1)
        h = self.act(self.fc(self.norm(x)))
        return x + g * h


class SubjectProbe(nn.Module):
    """pooled 适配 token 上的 GRL 线性主体探针（对抗去主体）。"""

    def __init__(self, d=1024, n_subjects=7, grl_alpha=1.0):
        super().__init__()
        self.grl = GradientReversalLayer(alpha=grl_alpha)
        self.classifier = nn.Linear(d, n_subjects)

    def forward(self, pooled, subj):
        """pooled: (B, d) 已归一（mean over 128 → L2）；subj: (B,) 1-based。返回 CE（GRL 反转）。"""
        logits = self.classifier(self.grl(pooled))
        labels = subj.long() - 1
        return F.cross_entropy(logits, labels)
