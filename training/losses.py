"""训练损失：CLIP 风格 InfoNCE（fMRI↔图像 / fMRI↔文本 双对齐）。"""
import torch
import torch.nn.functional as F


def info_nce(a, b, logit_scale):
    """双向 InfoNCE。a, b: (B, D)，均已 L2 归一化。

    返回两个方向的交叉熵均值：正向 logits = a @ b.T，反向取转置。
    对角线为匹配对（batch 内 i↔i）。
    """
    logits = logit_scale * (a @ b.T)
    labels = torch.arange(a.size(0), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


class BrainContrastiveLoss:
    """fMRI 同时对齐图像与文本：loss = (fMRI↔图像 + fMRI↔文本) / 2。"""

    def __init__(self, logit_scale=1.0 / 0.07):
        self.logit_scale = logit_scale

    def __call__(self, brain, image, text):
        # 三者均为 (B, D) 归一化嵌入
        loss_img = info_nce(brain, image, self.logit_scale)
        loss_txt = info_nce(brain, text, self.logit_scale)
        return 0.5 * (loss_img + loss_txt)
