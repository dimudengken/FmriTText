"""Training-only functional voxel features and cluster-bank split audits.

The feature of a voxel is its ridge encoding weight in a common CLIP space.
This permits joint clustering across subjects without aligning voxel indices.
"""

import random

import numpy as np


def by_coco(group):
    grouped = {}
    for key in group:
        coco_id = int(group[key]["coco_id"][()])
        grouped.setdefault(coco_id, []).append(key)
    return grouped


def validation_ids(group, val_coco, seed):
    """Match dataset.train_val_split_by_coco_id, including insertion order."""
    if val_coco < 0:
        raise ValueError("val_coco must be nonnegative")
    ids = list(by_coco(group))
    random.Random(seed).shuffle(ids)
    return set(ids[:val_coco])


def heldout_ids(handles, val_coco, split_seed):
    """Union held-out stimuli across subjects before fitting any voxel features."""
    excluded = set()
    for handle in handles.values():
        excluded.update(validation_ids(handle["train"], val_coco, split_seed))
        for split in ("val", "test"):
            if split in handle:
                excluded.update(by_coco(handle[split]))
    return excluded


def training_responses(group, excluded_ids, feature_index):
    """Average repeats per training stimulus; never read held-out responses."""
    coco_ids, responses = [], []
    width = None
    for coco_id, keys in by_coco(group).items():
        if coco_id in excluded_ids:
            continue
        if coco_id not in feature_index:
            raise ValueError(f"CLIP bank is missing training coco_id={coco_id}")
        repeats = []
        for key in keys:
            values = np.asarray(group[key]["visual_voxels"][()], dtype=np.float32)
            if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
                raise ValueError(f"sample {key} must have finite 1D visual_voxels")
            if width is None:
                width = values.size
            if values.size != width:
                raise ValueError("visual_voxels width differs within a subject")
            repeats.append(values)
        responses.append(np.mean(repeats, axis=0))
        coco_ids.append(coco_id)
    if len(coco_ids) < 2:
        raise ValueError("at least two training stimuli are required after exclusions")
    return coco_ids, np.stack(responses)


def encoding_features(image_features, responses, ridge_alpha=100.0):
    """L2-normalized voxel encoding weights, fit solely on training stimuli."""
    if not np.isfinite(ridge_alpha) or ridge_alpha <= 0:
        raise ValueError("ridge_alpha must be positive and finite")
    x = np.asarray(image_features, dtype=np.float64)
    y = np.asarray(responses, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise ValueError("features and responses must have matching training rows")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("encoding inputs must be finite")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    y = y / np.maximum(y.std(axis=0, keepdims=True), 1e-6)
    gram = x.T @ x
    gram.flat[:: gram.shape[0] + 1] += ridge_alpha
    weights = np.linalg.solve(gram, x.T @ y).T
    weights /= np.maximum(np.linalg.norm(weights, axis=1, keepdims=True), 1e-12)
    return weights.astype(np.float32)


def audit_cluster_split(bank, handles, val_coco, split_seed):
    """Reject cluster fits that would contaminate this experiment's held-out set."""
    if bank.get("version") != 1 or not bank.get("subjects"):
        raise ValueError("unsupported or empty cluster bank")
    excluded = heldout_ids(handles, val_coco, split_seed)
    # Also audit bank subjects outside the requested training subset. Their
    # responses affected the shared centroids, so they too must be disjoint.
    all_fit_ids = set()
    for entry in bank["subjects"].values():
        if "fit_coco_ids" not in entry:
            raise ValueError("cluster bank lacks training provenance")
        all_fit_ids.update(int(value) for value in entry["fit_coco_ids"])
    overlap = all_fit_ids & excluded
    if overlap:
        raise ValueError(f"cluster fit overlaps held-out coco_ids: {sorted(overlap)[:10]}")
    for subject, handle in handles.items():
        if subject not in bank["subjects"]:
            raise ValueError(f"cluster bank is missing {subject}")
        fitted = set(int(value) for value in bank["subjects"][subject]["fit_coco_ids"])
        available = set(by_coco(handle["train"])) - excluded
        if not fitted or not fitted <= available:
            raise ValueError(f"cluster fit for {subject} is not a subset of this training split")
    return {"fitted_stimuli": len(all_fit_ids), "heldout_stimuli": len(excluded)}
