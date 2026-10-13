import torch
import torch.nn.functional as F


@torch.no_grad()
def retrieval_metrics(query_z, target_z, ks=(1, 5, 10)):
    """Rank of the matching target for every query, i.e. Brain -> Image retrieval.

    The candidate bank is the set of target_z rows themselves, so this is
    within-set retrieval and both sides must be aligned row by row.

    Ties are counted as worse than a strict win (`sim > sim.diag()`), which is the
    conservative direction when duplicate stimuli share a target embedding.
    """
    query_z = F.normalize(query_z.float(), dim=-1)
    target_z = F.normalize(target_z.float(), dim=-1)
    assert query_z.size(0) == target_z.size(0), "query and target bank must align"

    sim = query_z @ target_z.T
    ranks = 1 + (sim > sim.diag().unsqueeze(1)).sum(dim=1)

    metrics = {f"r@{k}": (ranks <= k).float().mean().item() for k in ks}
    metrics["median_rank"] = ranks.float().median().item()
    metrics["mean_rank"] = ranks.float().mean().item()
    return metrics


@torch.no_grad()
def retrieval_metrics_unique_gallery(query_z, target_z, coco_ids, ks=(1, 5, 10)):
    """Rank each query against one target per stimulus, including cross-subject repeats."""
    query_z = F.normalize(query_z.float(), dim=-1)
    target_z = F.normalize(target_z.float(), dim=-1)
    ids = [int(value) for value in coco_ids.tolist()]
    first = {}
    for row, coco_id in enumerate(ids):
        if coco_id in first:
            if not torch.allclose(target_z[row], target_z[first[coco_id]], atol=0, rtol=0):
                raise ValueError(f"target differs for repeated coco_id={coco_id}")
        else:
            first[coco_id] = row
    gallery_ids = list(first)
    gallery = target_z[list(first.values())]
    gallery_index = {coco_id: row for row, coco_id in enumerate(gallery_ids)}
    correct = torch.tensor([gallery_index[coco_id] for coco_id in ids], device=query_z.device)
    sim = query_z @ gallery.T
    correct_sim = sim.gather(1, correct[:, None])
    ranks = 1 + (sim > correct_sim).sum(1)
    metrics = {f"r@{k}": (ranks <= k).float().mean().item() for k in ks}
    metrics["median_rank"] = ranks.float().median().item()
    metrics["mean_rank"] = ranks.float().mean().item()
    return metrics
