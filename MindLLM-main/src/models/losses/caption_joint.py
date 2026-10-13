"""Small, testable metrics for M4/M5's caption-stage auxiliary objective."""

import numbers

import torch
import torch.nn.functional as F

from src.models.losses.contrastive import multi_positive_infonce, similarity_diagnostics


def lookup_caption_targets(bank, index, sample_ids, device, mode):
    """Resolve per-stimulus targets on CPU, then transfer only this batch."""
    rows, ids = [], []
    for value in sample_ids:
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise TypeError("M4/M5 require integer coco_ids; use data.group_by_coco=true")
        coco_id = int(value)
        if coco_id not in index:
            raise KeyError(f"no {mode} CLIP target for coco_id={coco_id}")
        rows.append(index[coco_id])
        ids.append(coco_id)
    return bank[rows].to(device), torch.as_tensor(ids, dtype=torch.long, device=device)


def warmup_weight(step, maximum, warmup_steps):
    if warmup_steps < 0 or maximum < 0:
        raise ValueError("weight and warmup_steps must be nonnegative")
    return maximum if warmup_steps == 0 else maximum * min(step / warmup_steps, 1.0)


def caption_contrastive_metrics(brain_z, target_z, coco_ids, temperature):
    """Reuse Stage 1's symmetric multi-positive objective and score matching IDs."""
    loss = multi_positive_infonce(brain_z, target_z, coco_ids, temperature)
    similarity = F.normalize(brain_z.float(), dim=-1) @ F.normalize(target_z.float(), dim=-1).T
    predicted_ids = coco_ids[similarity.argmax(dim=1)]
    top1 = (predicted_ids == coco_ids).float().mean()
    positive, negative = similarity_diagnostics(brain_z, target_z, coco_ids)
    return loss, top1, positive, negative
