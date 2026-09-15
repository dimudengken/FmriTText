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


def info_nce_crosssubj(q, pos, bank, bank_img, nsd, n_neg, logit_scale):
    """跨被试对比：正对 = q 与本图 CLIP 嵌入；负样本 = 其他被试的 brain 嵌入（bank 内最难 top-n）。

    强制每个被试的语义空间对齐到同一 CLIP 空间：其他被试的 brain 嵌入若与 q 高度可混淆
    （=空间漂移），交叉熵惩罚把它们推开。q, pos, bank: 均已 L2 归一化。

    q: (B,768) 本被试 head 嵌入；pos: (B,768) 本图 CLIP 图像嵌入；
    bank: (K,768) 其他被试 head 嵌入（detach）；bank_img: (K,) 各行 nsd_idx；
    nsd: (B,) 本 batch 各行 nsd_idx（排除同图跨被试正例，不当作负样本）。
    """
    B = q.size(0)
    dev = q.device
    pos_logit = logit_scale * (q * pos).sum(-1, keepdim=True)          # (B,1)
    batch_logits = logit_scale * (q @ pos.T)                            # (B,B) batch CLIP 负
    same = nsd.unsqueeze(0) == nsd.unsqueeze(1)
    batch_neg = batch_logits.masked_fill(
        torch.eye(B, dtype=torch.bool, device=dev) | same, -1e9)
    own = bank_img.unsqueeze(0) == nsd.unsqueeze(1)                     # (B,K) 同图跨被试正例排除
    sim = (q @ bank.T).masked_fill(own, -1e9)
    hard = sim.topk(n_neg, dim=1).indices                               # (B,n_neg) 最难区分
    hard_logits = logit_scale * torch.einsum("bd,bnd->bn", q, bank[hard])
    logits = torch.cat([pos_logit, batch_neg, hard_logits], dim=1)      # (B, 1+B+n_neg)
    return F.cross_entropy(logits, torch.zeros(B, dtype=torch.long, device=dev))


class BrainContrastiveLoss:
    """fMRI 同时对齐图像与文本：loss = (fMRI↔图像 + fMRI↔文本) / 2。"""

    def __init__(self, logit_scale=1.0 / 0.07):
        self.logit_scale = logit_scale

    def __call__(self, brain, image, text):
        # 三者均为 (B, D) 归一化嵌入
        loss_img = info_nce(brain, image, self.logit_scale)
        loss_txt = info_nce(brain, text, self.logit_scale)
        return 0.5 * (loss_img + loss_txt)
