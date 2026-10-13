"""Build a shared functional voxel cluster bank without a tokenizer or LLM."""

import argparse
from contextlib import ExitStack
import logging
from pathlib import Path
import sys

import h5py
import nibabel as nib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.voxel_clustering import (
    audit_cluster_split,
    encoding_features,
    heldout_ids,
    training_responses,
)

logger = logging.getLogger(__name__)


def build_bank(handles, feature_bank, roi_dir, n_clusters=128, val_coco=0,
               split_seed=42, seed=42, ridge_alpha=100.0, random_clusters=False):
    if n_clusters < 1:
        raise ValueError("n_clusters must be positive")
    if feature_bank.get("modality", "image") != "image":
        raise ValueError("functional voxel clustering requires an image feature bank")
    embeddings = feature_bank["embeddings"].float().cpu().numpy()
    image_ids = [int(value) for value in feature_bank["coco_ids"]]
    if (embeddings.ndim != 2 or embeddings.shape[0] != len(image_ids)
            or len(set(image_ids)) != len(image_ids) or not np.isfinite(embeddings).all()):
        raise ValueError("CLIP bank must have one finite feature per unique coco_id")
    feature_index = {coco_id: row for row, coco_id in enumerate(image_ids)}
    excluded = heldout_ids(handles, val_coco, split_seed)
    entries, voxel_features = {}, []
    for subject, handle in handles.items():
        coco_ids, responses = training_responses(handle["train"], excluded, feature_index)
        x = embeddings[[feature_index[coco_id] for coco_id in coco_ids]]
        features = encoding_features(x, responses, ridge_alpha)
        subject_number = int(subject.removeprefix("subject_"))
        mask_path = Path(roi_dir) / f"subj{subject_number:02d}" / "nsdgeneral.nii.gz"
        # Match MaskRegistered exactly: integer mask, then > 0, C-order flatten.
        mask = nib.load(str(mask_path)).get_fdata().astype(np.int64) > 0
        flat_indices = np.flatnonzero(mask.reshape(-1))
        if responses.shape[1] != len(flat_indices):
            raise ValueError(f"{subject}: visual_voxels width does not match nsdgeneral mask")
        entries[subject] = {
            "fit_coco_ids": coco_ids,
            "mask_shape": list(mask.shape),
            "flat_indices": torch.from_numpy(flat_indices.copy()),
        }
        voxel_features.append(features)
        logger.info("%s: fit_stimuli=%d voxels=%d", subject, len(coco_ids), len(features))
    if not voxel_features:
        raise ValueError("at least one subject is required")
    features = np.concatenate(voxel_features)
    if len(features) < n_clusters:
        raise ValueError("n_clusters exceeds the total voxel count")
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init=10,
                            batch_size=4096, reassignment_ratio=0.0)
    labels = kmeans.fit_predict(features)
    if len(np.unique(labels)) != n_clusters:
        raise ValueError("functional features produced empty global clusters; reduce n_clusters")
    rng = np.random.default_rng(seed)
    start = 0
    for (subject, entry), features in zip(entries.items(), voxel_features):
        assigned = labels[start:start + len(features)].copy()
        start += len(features)
        if random_clusters:
            # Same per-subject cluster sizes as the functional fit: change only
            # membership, not token capacity or the number of absent clusters.
            rng.shuffle(assigned)
        entry["cluster_ids"] = torch.from_numpy(assigned).long()
        counts = np.bincount(assigned, minlength=n_clusters)
        logger.info("%s: occupied_clusters=%d size_min/max=%d/%d",
                    subject, np.count_nonzero(counts), counts.min(), counts.max())
    bank = {
        "version": 1,
        "n_clusters": n_clusters,
        "method": "random_size_matched" if random_clusters else "encoding_kmeans",
        "seed": seed,
        "ridge_alpha": ridge_alpha,
        "val_coco": val_coco,
        "split_seed": split_seed,
        "feature_model_id": feature_bank.get("model_id"),
        "excluded_coco_ids": sorted(excluded),
        "centroids": torch.from_numpy(kmeans.cluster_centers_),
        "subjects": entries,
    }
    audit_cluster_split(bank, handles, val_coco, split_seed)
    return bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5-dir", default="data/nsd0")
    parser.add_argument("--roi-dir", default="data/roi")
    parser.add_argument("--feature-bank", required=True)
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    parser.add_argument("--n-clusters", type=int, default=128)
    parser.add_argument("--val-coco", type=int, default=0,
                        help="Use 900 for data.split_val=true; 0 for split_val=false.")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--random-clusters", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(set(args.subjects)) != len(args.subjects) or any(s not in range(1, 9) for s in args.subjects):
        parser.error("subjects must be distinct NSD subject numbers (1-8)")
    output = Path(args.output)
    if output.exists():
        parser.error(f"output already exists: {output}; choose a new bank path")
    feature_bank = torch.load(args.feature_bank, map_location="cpu", weights_only=True)
    with ExitStack() as stack:
        handles = {
            f"subject_{s}": stack.enter_context(h5py.File(Path(args.h5_dir) / f"subject_{s}.h5", "r"))
            for s in args.subjects
        }
        bank = build_bank(handles, feature_bank, args.roi_dir, args.n_clusters,
                          args.val_coco, args.split_seed, args.seed,
                          args.ridge_alpha, args.random_clusters)
    bank["feature_bank_path"] = str(Path(args.feature_bank).resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, output)
    logger.info("Saved %s", output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
