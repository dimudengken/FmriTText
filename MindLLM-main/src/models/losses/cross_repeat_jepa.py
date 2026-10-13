"""Losses for predicting one fMRI repeat in a CLIP-anchored latent space."""

import torch
import torch.nn.functional as F


def cosine_distance(prediction, target):
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction and target must have the same [batch, dim] shape")
    prediction = F.normalize(prediction.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    return (1.0 - (prediction * target).sum(dim=-1)).mean()


@torch.no_grad()
def ema_update(teacher, online, decay):
    """Update parameters after an optimizer step; copy non-trainable buffers."""
    if not 0.0 <= decay < 1.0:
        raise ValueError("EMA decay must lie in [0, 1)")
    teacher_params = dict(teacher.named_parameters())
    online_params = dict(online.named_parameters())
    teacher_buffers = dict(teacher.named_buffers())
    online_buffers = dict(online.named_buffers())
    if teacher_params.keys() != online_params.keys() or teacher_buffers.keys() != online_buffers.keys():
        raise ValueError("EMA teacher and online module structures differ")
    for name, parameter in teacher_params.items():
        parameter.lerp_(online_params[name], 1.0 - decay)
    for name, buffer in teacher_buffers.items():
        buffer.copy_(online_buffers[name])
