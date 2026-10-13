"""Align an existing text CLIP bank to the exact stimulus set of an M1 image bank.

This only selects and reorders precomputed text vectors. It never changes the
image bank, recomputes CLIP features, or fills missing targets with guesses.
"""

import argparse
import os

import torch


def ids_and_embeddings(bank, label):
    ids = [int(value) for value in bank["coco_ids"].tolist()]
    embeddings = bank["embeddings"].float()
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} bank contains duplicate coco_ids")
    if embeddings.ndim != 2 or embeddings.size(0) != len(ids):
        raise ValueError(f"{label} bank has an invalid embedding shape")
    if not torch.isfinite(embeddings).all():
        raise ValueError(f"{label} bank contains NaN or Inf")
    if not torch.allclose(embeddings.norm(dim=1), torch.ones(len(ids)), atol=1e-3):
        raise ValueError(f"{label} bank embeddings are not L2 normalized")
    return ids, embeddings


def check_train_coverage(image_ids, h5_dir, subjects, coco_dir=None):
    import h5py

    train_ids = set()
    image_id_set = set(image_ids)
    for subject in subjects:
        path = os.path.join(h5_dir, f"subject_{subject}.h5")
        with h5py.File(path, "r") as handle:
            subject_ids = {
                int(handle["train"][key]["coco_id"][()])
                for key in handle["train"]
            }
        train_ids.update(subject_ids)
        subject_missing = subject_ids - image_id_set
        print(
            f"subject {subject}: train stimuli={len(subject_ids)}, "
            f"missing from image bank={len(subject_missing)}, "
            f"first={sorted(subject_missing)[:10]}",
            flush=True,
        )
    missing = train_ids - image_id_set
    print(f"train stimuli={len(train_ids)}, missing from image bank={len(missing)}")
    if missing and coco_dir and any(
        os.path.isdir(os.path.join(coco_dir, f"{split}2017"))
        for split in ("train", "val", "test")
    ):
        available = sum(
            any(
                os.path.isfile(os.path.join(coco_dir, f"{split}2017", f"{coco_id:012}.jpg"))
                for split in ("train", "val", "test")
            ) or os.path.isfile(os.path.join(coco_dir, f"{coco_id:012}.jpg"))
            for coco_id in missing
        )
        print(f"missing-bank stimuli with local COCO image={available}/{len(missing)}", flush=True)
    if missing:
        raise ValueError(
            f"M1 image bank lacks train stimuli for these subjects; first IDs: {sorted(missing)[:10]}. "
            "Use the image bank and subject list from the completed M1 run."
        )


def align(image_path, text_path, out_path, h5_dir=None, subjects=None, coco_dir=None):
    if os.path.abspath(out_path) == os.path.abspath(text_path):
        raise ValueError("--out must differ from --text-bank to preserve the original")
    image = torch.load(image_path, map_location="cpu")
    text = torch.load(text_path, map_location="cpu")
    image_ids, image_embeddings = ids_and_embeddings(image, "image")
    text_ids, text_embeddings = ids_and_embeddings(text, "text")
    if text.get("modality") != "text" or text.get("caption_aggregation") != "mean_then_l2":
        raise ValueError("text bank lacks modality=text or caption_aggregation=mean_then_l2")
    if image.get("model_id") != text.get("model_id"):
        raise ValueError("image and text banks use different CLIP model_id values")
    if image_embeddings.size(1) != text_embeddings.size(1):
        raise ValueError("image and text embedding dimensions differ")

    image_set, text_set = set(image_ids), set(text_ids)
    image_only, text_only = image_set - text_set, text_set - image_set
    print(f"image IDs={len(image_ids)}, text IDs={len(text_ids)}")
    print(f"image-only IDs={len(image_only)} first={sorted(image_only)[:10]}")
    print(f"text-only IDs={len(text_only)} first={sorted(text_only)[:10]}")
    if image_only:
        raise ValueError(
            "text bank is missing image-bank targets; regenerate text features for "
            "the missing COCO captions before training"
        )
    if subjects is not None:
        check_train_coverage(image_ids, h5_dir, subjects, coco_dir)

    text_index = {coco_id: row for row, coco_id in enumerate(text_ids)}
    aligned = dict(text)
    aligned["coco_ids"] = torch.tensor(image_ids, dtype=torch.long)
    aligned["embeddings"] = text_embeddings[[text_index[coco_id] for coco_id in image_ids]].contiguous()
    directory = os.path.dirname(out_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(aligned, out_path)
    print(f"saved aligned text bank: {len(image_ids)} x {aligned['embeddings'].size(1)} -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-bank", required=True)
    parser.add_argument("--text-bank", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--h5-dir", default="data/nsd0")
    parser.add_argument("--coco-dir", default="data/coco")
    parser.add_argument("--subjects", help="comma-separated M1/M2 subject IDs; also checks train coverage")
    args = parser.parse_args()
    subjects = [int(value) for value in args.subjects.split(",")] if args.subjects else None
    align(args.image_bank, args.text_bank, args.out, args.h5_dir, subjects, args.coco_dir)


if __name__ == "__main__":
    main()
