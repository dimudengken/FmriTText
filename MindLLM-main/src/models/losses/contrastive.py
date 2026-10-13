import torch
import torch.nn.functional as F


def _normalize(x):
    return F.normalize(x.float(), dim=-1)


def multi_positive_infonce(brain_z, target_z, group_ids, temperature=0.07):
    """Symmetric InfoNCE in which every sample sharing a `group_id` is a positive.

    Two fMRI repeats of the same stimulus are therefore never pushed apart (an
    ordinary diagonal InfoNCE would treat them as false negatives). When all group
    ids are unique this reduces exactly to the standard diagonal InfoNCE.

    Args:
        brain_z:   [B, D] brain-side embeddings, un-normalized is fine.
        target_z:  [B, D] frozen semantic target embeddings.
        group_ids: [B] stimulus identity; positives of i are {j: group_ids[j] == group_ids[i]}.
        temperature: logit scale divisor.

    Returns:
        scalar loss.
    """
    brain_z = _normalize(brain_z)
    target_z = _normalize(target_z)

    logits = brain_z @ target_z.T / temperature

    pos_mask = group_ids.reshape(-1, 1).eq(group_ids.reshape(1, -1)).float()
    pos_count = pos_mask.sum(dim=1).clamp_min(1.0)

    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    loss_b2t = -(log_prob * pos_mask).sum(dim=1) / pos_count

    log_prob_t = logits.T - torch.logsumexp(logits.T, dim=1, keepdim=True)
    loss_t2b = -(log_prob_t * pos_mask).sum(dim=1) / pos_count

    return 0.5 * (loss_b2t.mean() + loss_t2b.mean())


@torch.no_grad()
def similarity_diagnostics(brain_z, target_z, group_ids):
    """Mean cosine similarity of positive and negative brain/target pairs.

    The contrastive loss can go down for the wrong reason, so training logs these
    two numbers: a healthy run shows positive similarity rising while negative
    similarity stays flat or falls.
    """
    brain_z = _normalize(brain_z)
    target_z = _normalize(target_z)
    sim = brain_z @ target_z.T

    pos_mask = group_ids.reshape(-1, 1).eq(group_ids.reshape(1, -1))
    pos_sim = sim[pos_mask].mean()

    neg = sim[~pos_mask]
    neg_sim = neg.mean() if neg.numel() > 0 else sim.new_zeros(())

    return pos_sim, neg_sim
