"""Precompute frozen CLIP image embeddings for every NSD stimulus.

The CLIP target of the global InfoNCE is fixed and the stimulus set is finite, so
the embeddings are computed once and stored as a `coco_id -> embedding` bank
instead of being recomputed during training.

Output format (torch.save):
    {"coco_ids": LongTensor[N], "embeddings": FloatTensor[N, D], "model_id": str}

The expected image layout is the one `src.dataset.load_coco_image` uses, i.e.
`<coco-dir>/{train,val,test}2017/<coco_id:012>.jpg`. Run from the repository root:

    uv run python scripts/precompute_clip_features.py \
        --out data/clip_features/coco_clip_vitl14.pt

`--subjects` must match `data.subjects`; only those subject h5 files define the
images that are actually needed.
"""

import argparse
import json
import logging
import os
from collections import defaultdict

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

logging.basicConfig(level=logging.INFO)

COCO_SPLITS = ["train", "val", "test"]
# The datamodule never reads the h5 "val" group (val is carved out of "train"), so
# scanning it here would only add images that training can never ask for.
H5_SPLITS = ["train", "test"]


def coco_image_path(coco_id, coco_dir):
    """Mirror of src.dataset.get_coco_path / load_coco_image.

    The flat `{coco_id:012}.jpg` layout is accepted as a last resort because the
    filename is unambiguous; the split subfolders are preferred when present.
    """
    candidates = [
        os.path.join(coco_dir, f"{split}2017", f"{coco_id:012}.jpg")
        for split in COCO_SPLITS
    ]
    candidates.append(os.path.join(coco_dir, f"{coco_id:012}.jpg"))
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(candidates[0])


def collect_coco_ids(h5_dir, subjects):
    """All coco ids the datamodule can ask for, with the configured subjects.

    Preferred source is the NSD h5 files actually read by the datamodule; the
    stim-info csv is a fallback for setups that keep the voxels elsewhere.
    """
    ids = set()
    files = [
        path
        for path in (
            os.path.join(h5_dir, f"subject_{subject}.h5") for subject in subjects
        )
        if os.path.isfile(path)
    ]
    if not files:
        logging.warning(f"no subject h5 file found under {h5_dir}")
    for path in files:
        import h5py

        before = len(ids)
        with h5py.File(path, "r") as f:
            for split in H5_SPLITS:
                if split not in f:
                    continue
                for key in f[split]:
                    ids.add(int(f[split][key]["coco_id"][()]))
        logging.info(f"{path}: +{len(ids) - before} ids")

    if not ids:
        info = pd.read_csv(os.path.join("data", "nsd_stim_info_merged.csv"))
        ids = {int(x) for x in info["cocoId"].dropna().unique()}
        logging.info(f"collected {len(ids)} coco ids from nsd_stim_info_merged.csv")
    return sorted(ids)


def load_captions(annotation_dir):
    """Use the same local COCO annotations as src.dataset.from_coco_caption."""
    captions = defaultdict(list)
    for split in ("train", "val"):
        path = os.path.join(annotation_dir, f"captions_{split}2017.json")
        with open(path, encoding="utf-8") as handle:
            for item in json.load(handle)["annotations"]:
                captions[int(item["image_id"])].append(item["caption"])
    return captions


def encode_text_bank(coco_ids, captions, model, tokenizer, batch_size, device):
    missing = [coco_id for coco_id in coco_ids if not captions.get(coco_id)]
    if missing:
        raise ValueError(f"{len(missing)} COCO ids have no human captions; first: {missing[:10]}")
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(coco_ids), batch_size):
            ids = coco_ids[start : start + batch_size]
            texts = [caption for coco_id in ids for caption in captions[coco_id]]
            counts = [len(captions[coco_id]) for coco_id in ids]
            encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            features = F.normalize(model.get_text_features(**encoded).float(), dim=-1)
            means = [part.mean(0) for part in features.split(counts)]
            embeddings.append(F.normalize(torch.stack(means), dim=-1).cpu())
            logging.info(f"encoded text {start + len(ids)}/{len(coco_ids)}")
    return torch.cat(embeddings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=("image", "text", "both"), default="image")
    parser.add_argument("--model-id", default="openai/clip-vit-large-patch14")
    parser.add_argument("--out", default="data/clip_features/coco_clip_vitl14.pt")
    parser.add_argument("--text-out", default="features/clip_text_features.pt")
    parser.add_argument("--annotation-dir", default="data/coco/annotations")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--coco-dir", default="data/coco")
    parser.add_argument("--h5-dir", default="data/nsd0")
    parser.add_argument(
        "--subjects",
        default="1",
        help="comma separated subject ids, must match data.subjects",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--limit", type=int, default=-1, help="only encode the first N ids"
    )
    args = parser.parse_args()

    from transformers import CLIPImageProcessor, CLIPModel

    subjects = [int(s) for s in str(args.subjects).split(",") if s.strip()]
    coco_ids = collect_coco_ids(args.h5_dir, subjects)
    if args.limit > 0:
        coco_ids = coco_ids[: args.limit]

    roots = {
        split: os.path.join(args.coco_dir, f"{split}2017") for split in COCO_SPLITS
    }
    logging.info(
        "coco image roots: "
        + ", ".join(
            f"{path}={'ok' if os.path.isdir(path) else 'MISSING'}"
            for path in roots.values()
        )
    )
    if args.modality in ("image", "both") and not any(os.path.isdir(path) for path in roots.values()):
        raise FileNotFoundError(
            f"no COCO image folder under {os.path.abspath(args.coco_dir)!r}; expected "
            f"{roots['train']} and friends. Pass --coco-dir pointing at the folder "
            "that contains train2017/."
        )

    logging.info(
        f"encoding {len(coco_ids)} images with {args.model_id} on {args.device}"
    )

    model = (
        CLIPModel.from_pretrained(args.model_id, torch_dtype=torch.float32)
        .to(args.device)
        .eval()
    )
    if args.modality in ("text", "both"):
        from transformers import CLIPTokenizer

        captions = load_captions(args.annotation_dir)
        tokenizer = CLIPTokenizer.from_pretrained(args.model_id)
        text_embeddings = encode_text_bank(
            coco_ids, captions, model, tokenizer, args.batch_size, args.device
        )
        text_dir = os.path.dirname(args.text_out)
        if text_dir:
            os.makedirs(text_dir, exist_ok=True)
        text_payload = {
            "coco_ids": torch.tensor(coco_ids, dtype=torch.long),
            "embeddings": text_embeddings,
            "model_id": args.model_id,
            "modality": "text",
            "caption_aggregation": "mean_then_l2",
        }
    if args.modality == "text":
        torch.save(text_payload, args.text_out)
        logging.info(f"saved {len(coco_ids)} text targets to {args.text_out}")
        return
    image_processor = CLIPImageProcessor.from_pretrained(args.model_id)

    embeddings = []
    kept_ids = []
    missing = []
    first_path = None
    first_miss = None
    with torch.no_grad():
        for start in range(0, len(coco_ids), args.batch_size):
            chunk = coco_ids[start : start + args.batch_size]
            images = []
            chunk_ids = []
            for coco_id in chunk:
                try:
                    image_path = coco_image_path(coco_id, args.coco_dir)
                except FileNotFoundError as exc:
                    missing.append(coco_id)
                    if first_miss is None:
                        first_miss = str(exc)
                    continue
                images.append(Image.open(image_path).convert("RGB"))
                chunk_ids.append(coco_id)
                if first_path is None:
                    first_path = image_path
            if not images:
                continue

            pixel_values = image_processor(images=images, return_tensors="pt")[
                "pixel_values"
            ].to(args.device)
            features = model.get_image_features(pixel_values=pixel_values)
            embeddings.append(F.normalize(features.float(), dim=-1).cpu())
            kept_ids.extend(chunk_ids)
            logging.info(f"encoded {start + len(chunk)}/{len(coco_ids)}")

    if not embeddings:
        raise RuntimeError(
            f"no image could be encoded; first lookup missed {first_miss}. Check that "
            f"--coco-dir holds train2017/<coco_id:012>.jpg"
        )

    if args.modality == "both" and kept_ids != coco_ids:
        raise ValueError("image/text bank ids differ because some COCO images are missing")
    if args.modality == "both":
        torch.save(text_payload, args.text_out)
        logging.info(f"saved {len(coco_ids)} text targets to {args.text_out}")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {
            "coco_ids": torch.tensor(kept_ids, dtype=torch.long),
            "embeddings": torch.cat(embeddings, dim=0),
            "model_id": args.model_id,
        },
        args.out,
    )
    logging.info(
        f"saved {len(kept_ids)} x {embeddings[0].size(-1)} embeddings to {args.out} "
        f"(first image read from {first_path}; {len(missing)} of {len(coco_ids)} "
        f"coco ids had no image)"
    )
    if missing:
        logging.warning(f"missing coco ids: {sorted(missing)[:20]}")


if __name__ == "__main__":
    main()
