"""Caption generation under Real, Zero and subject-local Shuffle, relative to M0.

Run from the project root with python -m scripts.evaluate_neural_dependence.
Raw voxels are perturbed before the encoder; prompts/references remain fixed.
"""
import argparse
import csv
import gc
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from lightning import seed_everything
from lightning.fabric.utilities.apply_func import move_data_to_device
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge

from scripts.neural_dependence_protocol import shuffle_donors
from src.dataset import create_tokenizer
from src.metrics import coco_tokenize
from src.models.mindllm import MindLLM
from src.utils import adapt_voxels


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap(values, keys, count, seed=42):
    """Paired per-image contributions, stratified by subject, fixed scorer IDF."""
    if count < 1:
        return None
    rng = np.random.default_rng(seed)
    subjects = np.array([key[0] for key in keys])
    groups = [np.flatnonzero(subjects == subject) for subject in sorted(set(subjects))]
    means = []
    for _ in range(count):
        rows = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        means.append(float(values[rows].mean()))
    return np.quantile(means, [0.025, 0.975]).tolist()


@torch.inference_mode()
def generate(model, batches, keys, donors, condition, path, metadata):
    records = []
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["metadata"] != metadata:
            raise ValueError(f"existing results use different settings: {path}; use a new output directory")
        records = saved["records"]
        if len(records) > len(keys):
            raise ValueError("cached prediction count exceeds sample count")
        for i, record in enumerate(records):
            if record["subject"] != keys[i][0] or record["coco_id"] != keys[i][1] or record["donor_coco_id"] != keys[donors[i]][1]:
                raise ValueError("cached prediction order or donor mapping differs")
    try:
        for i in range(len(records), len(batches)):
            batch = dict(batches[i])
            voxels = batches[donors[i]]["voxels"] if condition.startswith("shuffle") else batch["voxels"]
            batch["voxels"] = torch.zeros_like(voxels) if condition == "zero" else voxels
            batch = move_data_to_device(batch, torch.device("cuda"))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                tokens = model.generate(**batch)
            prediction = model.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
            records.append({"subject": keys[i][0], "coco_id": keys[i][1],
                "donor_coco_id": keys[donors[i]][1], "prediction": prediction})
            if (i + 1) % 50 == 0:
                save_json(path, {"metadata": metadata, "records": records})
                logging.info("%s %s: %d/%d", metadata["model"], condition, i + 1, len(batches))
    finally:
        save_json(path, {"metadata": metadata, "records": records})
    return [record["prediction"] for record in records]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="M0=PATH", help="supply M0 and one or more of M1, M2, M3")
    parser.add_argument("--subjects", default="1")
    parser.add_argument("--shuffle-seeds", default="42", help="use 42,43,44 for permutation robustness")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--out-dir", default="outputs/neural_dependence")
    args = parser.parse_args()
    checkpoints = {}
    for item in args.checkpoint:
        name, path = item.split("=", 1)
        if name in checkpoints:
            parser.error(f"duplicate checkpoint label: {name}")
        checkpoints[name] = path
    if "M0" not in checkpoints or len(checkpoints) < 2 or not set(checkpoints).issubset({"M0", "M1", "M2", "M3"}):
        parser.error("supply M0 and at least one of M1, M2, M3")
    model_names = [name for name in ("M0", "M1", "M2", "M3") if name in checkpoints]
    for path in checkpoints.values():
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    subjects = [int(value) for value in args.subjects.split(",")]
    shuffle_seeds = [int(value) for value in args.shuffle_seeds.split(",")]
    if len(subjects) != len(set(subjects)) or len(shuffle_seeds) != len(set(shuffle_seeds)) or args.bootstrap < 0:
        parser.error("subjects/seeds must be unique and bootstrap nonnegative")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "config"), version_base=None):
        cfg = compose(config_name="default")
    cfg.data.task, cfg.data.subjects = "coco-caption", subjects
    cfg.data.group_by_coco, cfg.data.split_val, cfg.data.num_workers, cfg.data.mixup = True, True, 0, False
    cfg.global_infonce.enabled, cfg.global_infonce.feature_path = False, None
    cfg.contrastive.enabled = False
    seed_everything(42, workers=True)
    tokenizer = create_tokenizer(cfg.model_id)
    data = instantiate(cfg.data, tokenizer=tokenizer)
    data.setup("test")
    batches, keys, references = [], [], []
    data_digest = hashlib.sha256()
    for batch in data.test_dataloader():
        if batch["voxels"].size(0) != 1:
            raise ValueError("caption evaluation requires batch size 1")
        batch["voxels"] = adapt_voxels(batch["voxels"]).cpu()
        key = (str(batch["subject"][0]), int(batch["sample_or_coco_id"][0]))
        refs = batch["answers"][0]
        refs = [refs] if isinstance(refs, str) else list(refs)
        if not refs or not all(isinstance(ref, str) for ref in refs):
            raise ValueError("each image needs caption references")
        data_digest.update(json.dumps([key, refs], ensure_ascii=False).encode())
        for field in ("voxels",):
            tensor = batch[field].contiguous()
            data_digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            data_digest.update(tensor.numpy().tobytes())
        for field in ("prompt", "first_chat"):
            for name, tensor in sorted(batch[field].items()):
                if isinstance(tensor, torch.Tensor):
                    data_digest.update((field + name + str(tuple(tensor.shape))).encode())
                    data_digest.update(tensor.contiguous().numpy().tobytes())
        batches.append(batch)
        keys.append(key)
        references.append(refs)
    if len(set(keys)) != len(keys):
        raise ValueError("evaluation must contain one caption query per subject/stimulus")
    for subject in data.file["train"]:
        for split in ("train", "val"):
            split_ids = {int(data.file[split][subject][k]["coco_id"][()]) for k in data.file[split][subject]}
            if split_ids & {coco_id for subj, coco_id in keys if subj == str(subject)}:
                raise ValueError(f"evaluation stimuli overlap {split} stimuli")
    conditions = {"real": list(range(len(keys))), "zero": list(range(len(keys)))}
    conditions.update({f"shuffle_{seed}": shuffle_donors(keys, seed) for seed in shuffle_seeds})
    manifest = {"protocol_version": 1, "samples_sha256": data_digest.hexdigest(), "keys": keys,
        "shuffle_donors": conditions, "raw_voxel_intervention": True, "model_id": cfg.model_id,
        "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": 128},
        "checkpoint_sha256": {name: file_hash(path) for name, path in checkpoints.items()},
        "evaluation_source": "shared1000; previously used for caption checkpoint selection"}
    manifest_path = out / "manifest.json"
    # Normalize tuples to the JSON representation for restart comparisons.
    manifest = json.loads(json.dumps(manifest))
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("output directory contains a different protocol; choose a new out-dir")
    save_json(manifest_path, manifest)
    all_scores, summary, metric_rows = {}, {}, []
    for name in model_names:
        seed_everything(42, workers=True)
        saved = torch.load(checkpoints[name], map_location="cpu")
        if "peft_model" in saved:
            raise ValueError("this protocol currently requires the frozen-LLM caption checkpoints (lora=false)")
        saved_keys = set(saved["state_dict"])
        del saved
        model = MindLLM.load_from_checkpoint(checkpoints[name], strict=False, encoder=cfg.encoder,
            model_id=cfg.model_id, tokenizer=tokenizer, peft_config=None, learning_rate=cfg.lr,
            lm_checkpointing="off", global_infonce=cfg.global_infonce, contrastive=cfg.contrastive)
        required = {key for key in model.state_dict() if key.startswith(("fmri_encoder.", "projector."))}
        if not required.issubset(saved_keys):
            raise ValueError(f"{name} is missing trained encoder/projector weights; use a complete caption checkpoint")
        model.eval().to("cuda")
        model.lm.generation_config.do_sample = False
        model.lm.generation_config.num_beams = 1
        all_scores[name], summary[name] = {}, {}
        for condition, donors in conditions.items():
            metadata = {"model": name, "checkpoint_sha256": manifest["checkpoint_sha256"][name],
                "samples_sha256": manifest["samples_sha256"], "condition": condition,
                "generation": manifest["generation"], "donors": donors}
            predictions = generate(model, batches, keys, donors, condition, out / f"{name}_{condition}.json", metadata)
            refs, preds = coco_tokenize(references, predictions)
            cider, per_image = Cider().compute_score(refs, preds)
            bleu, _ = Bleu(4).compute_score(refs, preds)
            rouge, _ = Rouge().compute_score(refs, preds)
            all_scores[name][condition] = np.asarray(per_image)
            summary[name][condition] = {"CIDEr": float(cider), "CIDEr_x100": float(cider) * 100,
                "BLEU4": float(bleu[3]), "ROUGE_L": float(rouge)}
            metric_rows.append({"model": name, "condition": condition, **summary[name][condition]})
            save_json(out / "metrics_partial.json", summary)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    for name in all_scores:
        real = all_scores[name]["real"]
        shuffle = np.mean([all_scores[name][f"shuffle_{seed}"] for seed in shuffle_seeds], axis=0)
        for label, control in (("zero", all_scores[name]["zero"]), ("shuffle", shuffle)):
            delta = real - control
            baseline_delta = all_scores["M0"]["real"] - (all_scores["M0"]["zero"] if label == "zero" else
                np.mean([all_scores["M0"][f"shuffle_{seed}"] for seed in shuffle_seeds], axis=0))
            summary[name][f"delta_{label}"] = {"CIDEr": float(delta.mean()), "CIDEr_x100": float(delta.mean()) * 100,
                "paired_CI95": bootstrap(delta, keys, args.bootstrap),
                "difference_from_M0": float((delta - baseline_delta).mean()),
                "difference_from_M0_CI95": bootstrap(delta - baseline_delta, keys, args.bootstrap)}
    summary["interpretation"] = {"shuffle_seeds": shuffle_seeds,
        "CI": "paired subject-stratified image bootstrap, fixed CIDEr reference IDF; conditional on checkpoints and shuffle seeds",
        "zero_caveat": "zero voxels are out of distribution; a drop alone does not establish useful neural information",
        "claim": "larger Shuffle drop supports stronger dependence on stimulus-matched brain input; verify permutation robustness"}
    save_json(out / "summary.json", summary)
    np.savez_compressed(out / "per_image_cider.npz", **{f"{name}_{condition}": values
        for name, scores in all_scores.items() for condition, values in scores.items()})
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    with (out / "dependence.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "real_CIDEr_x100", "zero_CIDEr_x100",
            "shuffle_mean_CIDEr_x100", "delta_zero_x100", "delta_shuffle_x100",
            "delta_zero_minus_M0_x100", "delta_shuffle_minus_M0_x100"])
        writer.writeheader()
        for name in model_names:
            arm = summary[name]
            writer.writerow({"model": name, "real_CIDEr_x100": arm["real"]["CIDEr_x100"],
                "zero_CIDEr_x100": arm["zero"]["CIDEr_x100"],
                "shuffle_mean_CIDEr_x100": float(np.mean([arm[f"shuffle_{seed}"]["CIDEr_x100"] for seed in shuffle_seeds])),
                "delta_zero_x100": arm["delta_zero"]["CIDEr_x100"], "delta_shuffle_x100": arm["delta_shuffle"]["CIDEr_x100"],
                "delta_zero_minus_M0_x100": arm["delta_zero"]["difference_from_M0"] * 100,
                "delta_shuffle_minus_M0_x100": arm["delta_shuffle"]["difference_from_M0"] * 100})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
