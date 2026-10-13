"""Bounded, sequential baseline/functional/random COCO-caption experiment."""

import argparse
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
ARMS = ("baseline", "functional", "random")


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def child(args, mode, arm=None):
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--mode", mode,
               "--output", str(args.output), "--init-checkpoint", str(args.init_checkpoint),
               "--feature-bank", str(args.feature_bank), "--epochs", str(args.epochs),
               "--batch-size", str(args.batch_size), "--accumulate", str(args.accumulate),
               "--seed", str(args.seed), "--patience", str(args.patience)]
    if arm:
        command += ["--arm", arm]
    return command


def campaign(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists():
        raise ValueError("experiment already exists; choose a new --output")
    protocol = {
        "source_sha256": {name: file_hash(PROJECT / name) for name in
            ("scripts/run_cluster_ablation.py", "src/dataset_cluster_ablation.py",
             "src/modules/clustered_neuro_informed_attn.py", "src/modules/neuro_informed_attn.py",
             "src/models/mindllm.py", "src/dataset.py", "main.py", "config/default.yaml")},
        "initial_checkpoint": str(args.init_checkpoint.resolve()),
        "initial_checkpoint_sha256": file_hash(args.init_checkpoint),
        "feature_bank": str(args.feature_bank.resolve()),
        "feature_bank_sha256": file_hash(args.feature_bank),
        "arms": list(ARMS), "subject": 1, "seed": args.seed,
        "task": "coco-caption", "validation_stimuli": 900,
        "split": "validation from HDF5 train; HDF5 test only after checkpoint selection",
        "learning_rate": 1e-4, "batch_size": args.batch_size,
        "accumulate_grad_batches": args.accumulate, "max_epochs": args.epochs,
        "early_stop_patience": args.patience, "selection": "min val/token_loss",
        "precision": "bf16-mixed", "lm_checkpointing": "off", "lora": False,
        "global_infonce": False, "caption_contrastive": False,
        "primary_metric": "held-out test CIDEr; functional minus baseline",
        "controls": "size-matched random clusters; real/zero/subject-local shuffled fMRI",
        "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": 128},
        "bootstrap": "2000 paired image resamples, fixed CIDEr IDF",
        "interpretation": "single-seed exploratory comparison; not a multi-seed significance claim",
        "max_hours": args.max_hours, "policy": "stop on first failed child; no automatic retry",
    }
    save_json(output / "protocol.json", protocol)
    deadline = time.monotonic() + args.max_hours * 3600
    status = {"pid": os.getpid(), "state": "running", "completed": [], "started_unix": time.time()}

    def run(command, label):
        for name, digest in protocol["source_sha256"].items():
            if file_hash(PROJECT / name) != digest:
                raise RuntimeError("experiment source changed: " + name)
        status.update(phase=label, command=command, phase_started_unix=time.time())
        save_json(output / "status.json", status)
        log_path = output / f"{label}.log"
        with log_path.open("w") as log:
            process = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            status.update(child_pid=process.pid, log=str(log_path))
            save_json(output / "status.json", status)
            try:
                code = process.wait(timeout=max(1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError(f"experiment time budget reached during {label}")
        if code:
            raise RuntimeError(f"{label} failed with exit code {code}; see {log_path}")
        status["completed"].append(label)
        save_json(output / "status.json", status)

    try:
        run([sys.executable, "-u", "scripts/build_voxel_clusters.py", "--subjects", "1",
             "--feature-bank", str(args.feature_bank), "--val-coco", "900",
             "--split-seed", str(args.seed), "--seed", str(args.seed),
             "--output", str(output / "functional.pt")], "build_functional")
        run(child(args, "random-bank"), "build_random")
        for arm in ARMS:
            run(child(args, "smoke", arm), f"smoke_{arm}")
        for arm in ARMS:
            run(child(args, "train", arm), f"train_{arm}")
        split_hashes = {json.loads((output / arm / "split.json").read_text())["sha256"] for arm in ARMS}
        if len(split_hashes) != 1:
            raise RuntimeError("training arms used different splits")
        for arm in ARMS:
            run(child(args, "evaluate", arm), f"evaluate_{arm}")
        run(child(args, "compare"), "compare")
        status.update(state="complete", finished_unix=time.time())
    except BaseException as error:
        status.update(state="failed", error=str(error), finished_unix=time.time())
        raise
    finally:
        save_json(output / "status.json", status)


def configuration(args):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(PROJECT / "config"), version_base=None):
        cfg = compose(config_name="default" if args.arm == "baseline" else "clustered")
    OmegaConf.set_struct(cfg, False)
    run = args.output / (f"smoke_{args.arm}" if args.mode == "smoke" else args.arm)
    cfg.seed, cfg.lr, cfg.checkpoint = args.seed, 1e-4, str(args.init_checkpoint.resolve())
    cfg.stage, cfg.test_after_fit, cfg.resume, cfg.early_stop = "fit", False, False, False
    cfg.output_dir, cfg.run_name = str(run), args.arm
    cfg.lora, cfg.lm_checkpointing = False, "off"
    cfg.global_infonce.enabled, cfg.global_infonce.feature_path = False, None
    cfg.contrastive.enabled = False
    cfg.data._target_ = "src.dataset_cluster_ablation.ClusterAblationDataModule"
    cfg.data.task, cfg.data.subjects = "coco-caption", [1]
    cfg.data.split_val, cfg.data.split_seed = True, args.seed
    cfg.data.batch_size, cfg.data.num_workers = args.batch_size, 0
    cfg.data.group_by_coco, cfg.data.mixup = True, False
    cfg.data.split_audit_path = str(run / "split.json")
    cfg.trainer.accelerator, cfg.trainer.devices, cfg.trainer.precision = "gpu", 1, "bf16-mixed"
    cfg.trainer.max_epochs = args.epochs
    cfg.trainer.accumulate_grad_batches = args.accumulate
    cfg.trainer.log_every_n_steps = 10
    cfg.trainer.enable_progress_bar = False
    cfg.trainer.num_sanity_val_steps = 2
    cfg.trainer.logger = {"_target_": "lightning.pytorch.loggers.CSVLogger",
                          "save_dir": str(run), "name": "csv", "version": "metrics"}
    cfg.trainer.callbacks = [callback for callback in cfg.trainer.callbacks
                            if callback._target_ != "src.utils.callbacks.WatchModel"]
    cfg.trainer.callbacks.append({"_target_": "lightning.pytorch.callbacks.EarlyStopping",
                                  "monitor": "val/token_loss", "mode": "min", "patience": args.patience})
    if args.arm != "baseline":
        cfg.encoder.cluster_path = str(args.output / f"{args.arm}.pt")
    if args.mode == "smoke":
        cfg.trainer.max_epochs = 1
        cfg.trainer.limit_train_batches = args.accumulate
        cfg.trainer.limit_val_batches = 2
        cfg.trainer.enable_checkpointing = False
        cfg.trainer.callbacks = [c for c in cfg.trainer.callbacks
                                if c._target_ != "lightning.pytorch.callbacks.ModelCheckpoint"]
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    return cfg, run


def train(args):
    import torch
    from main import main as train_mindllm
    from src.voxel_clustering import audit_cluster_split
    cfg, run = configuration(args)
    if args.arm != "baseline":
        import h5py
        with h5py.File(PROJECT / "data/nsd0/subject_1.h5", "r") as handle:
            bank = torch.load(cfg.encoder.cluster_path, map_location="cpu", weights_only=True)
            audit_cluster_split(bank, {"subject_1": handle}, 900, args.seed)
    train_mindllm(cfg)
    results = {"arm": args.arm, "mode": args.mode,
               "max_memory_gib": torch.cuda.max_memory_allocated() / 1024**3}
    if args.mode == "train":
        last = run / "coco-caption/checkpoints/last.ckpt"
        checkpoint = torch.load(last, map_location="cpu")
        records = [v for k, v in checkpoint["callbacks"].items() if "ModelCheckpoint" in k]
        best = records[0]["best_model_path"]
        if not Path(best).is_file():
            raise RuntimeError("best validation-selected checkpoint is missing")
        results.update(best_checkpoint=best, best_val_loss=float(records[0]["best_model_score"]),
                       epochs=int(checkpoint["epoch"]) + 1, global_step=int(checkpoint["global_step"]))
    save_json(run / "result.json", results)


def evaluate(args):
    import numpy as np
    import torch
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
    cfg, run = configuration(args)
    seed_everything(args.seed, workers=True)
    tokenizer = create_tokenizer(cfg.model_id)
    data = instantiate(cfg.data, tokenizer=tokenizer)
    data.setup("test")
    batches, keys, references = [], [], []
    for batch in data.test_dataloader():
        batch["voxels"] = adapt_voxels(batch["voxels"]).cpu()
        keys.append([str(batch["subject"][0]), int(batch["sample_or_coco_id"][0])])
        references.append(list(batch["answers"][0]))
        batches.append(batch)
    if len({tuple(key) for key in keys}) != len(keys):
        raise ValueError("test must have one caption query per stimulus")
    selected = json.loads((run / "result.json").read_text())["best_checkpoint"]
    model = MindLLM.load_from_checkpoint(selected, strict=False, encoder=cfg.encoder,
        model_id=cfg.model_id, tokenizer=tokenizer, peft_config=None, learning_rate=cfg.lr,
        global_infonce=cfg.global_infonce, contrastive=cfg.contrastive, lm_checkpointing="off")
    model.eval().to("cuda")
    model.lm.generation_config.do_sample, model.lm.generation_config.num_beams = False, 1
    donors = shuffle_donors([tuple(key) for key in keys], args.seed)
    scores, metrics = {}, {}
    for condition in ("real", "zero", "shuffle"):
        predictions = []
        with torch.inference_mode():
            for i, original in enumerate(batches):
                batch = dict(original)
                voxels = batches[donors[i]]["voxels"] if condition == "shuffle" else original["voxels"]
                batch["voxels"] = torch.zeros_like(voxels) if condition == "zero" else voxels
                batch = move_data_to_device(batch, torch.device("cuda"))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    tokens = model.generate(**batch)
                predictions.append(tokenizer.batch_decode(tokens, skip_special_tokens=True)[0])
                if (i + 1) % 50 == 0:
                    save_json(run / f"predictions_{condition}.json", {"keys": keys, "predictions": predictions})
                    logging.info("%s %s %d/%d", args.arm, condition, i + 1, len(batches))
        save_json(run / f"predictions_{condition}.json", {"keys": keys, "predictions": predictions})
        refs, preds = coco_tokenize(references, predictions)
        cider, per_image = Cider().compute_score(refs, preds)
        bleu, _ = Bleu(4).compute_score(refs, preds)
        rouge, _ = Rouge().compute_score(refs, preds)
        scores[condition] = np.asarray(per_image)
        metrics[condition] = {"CIDEr": float(cider), "CIDEr_x100": float(cider) * 100,
                              "BLEU4": float(bleu[3]), "ROUGE_L": float(rouge)}
        save_json(run / "test_metrics.json", metrics)
    np.savez_compressed(run / "per_image_cider.npz", **scores)
    save_json(run / "evaluation.json", {"keys": keys, "donors": donors,
              "references": references, "checkpoint": selected, "checkpoint_sha256": file_hash(selected)})


def compare(args):
    import numpy as np
    evaluations = [json.loads((args.output / arm / "evaluation.json").read_text()) for arm in ARMS]
    if any(e["keys"] != evaluations[0]["keys"] or e["references"] != evaluations[0]["references"]
           for e in evaluations[1:]):
        raise RuntimeError("test sample order or references differ between arms")
    scores = {arm: dict(np.load(args.output / arm / "per_image_cider.npz")) for arm in ARMS}
    rng = np.random.default_rng(args.seed)
    summary = {"exploratory_single_seed": True, "metrics": {}, "paired_comparisons": {}}
    for arm in ARMS:
        metrics = json.loads((args.output / arm / "test_metrics.json").read_text())
        metrics["brain_dependence_shuffle_CIDEr_x100"] = float((scores[arm]["real"] - scores[arm]["shuffle"]).mean() * 100)
        summary["metrics"][arm] = metrics
    for first, second in (("functional", "baseline"), ("functional", "random"), ("random", "baseline")):
        delta = scores[first]["real"] - scores[second]["real"]
        boot = [float(delta[rng.integers(0, len(delta), len(delta))].mean()) for _ in range(2000)]
        summary["paired_comparisons"][f"{first}_minus_{second}"] = {
            "delta_CIDEr_x100": float(delta.mean() * 100),
            "CI95_x100": (np.quantile(boot, [.025, .975]) * 100).tolist()}
    summary["interpretation"] = "Image bootstrap is conditional on this seed/checkpoints; repeat training seeds before a general superiority claim. Zero-fMRI is out of distribution."
    save_json(args.output / "summary.json", summary)
    with (args.output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["arm", "real_CIDEr_x100", "BLEU4", "ROUGE_L", "shuffle_drop_CIDEr_x100"])
        for arm in ARMS:
            metrics = summary["metrics"][arm]
            writer.writerow([arm, metrics["real"]["CIDEr_x100"], metrics["real"]["BLEU4"],
                             metrics["real"]["ROUGE_L"], metrics["brain_dependence_shuffle_CIDEr_x100"]])
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["campaign", "random-bank", "smoke", "train", "evaluate", "compare"], default="campaign")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--output", type=Path, default=Path("outputs/cluster_ablation_s1_seed42_20261013"))
    parser.add_argument("--init-checkpoint", type=Path, default=Path("ckpts/MindLLM/mindllm-base.ckpt"))
    parser.add_argument("--feature-bank", type=Path, default=Path("data/clip_features/coco_clip_vitl14.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-hours", type=float, default=24)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.mode in ("smoke", "train", "evaluate") and not args.arm:
        parser.error("--arm is required for model runs")
    if args.mode == "campaign":
        campaign(args)
    elif args.mode == "random-bank":
        import numpy as np
        import torch
        bank = torch.load(args.output / "functional.pt", map_location="cpu", weights_only=True)
        rng = np.random.default_rng(args.seed)
        for entry in bank["subjects"].values():
            labels = entry["cluster_ids"].numpy().copy()
            rng.shuffle(labels)
            entry["cluster_ids"] = torch.from_numpy(labels)
        bank["method"] = "random_size_matched"
        torch.save(bank, args.output / "random.pt")
    elif args.mode in ("smoke", "train"):
        train(args)
    elif args.mode == "evaluate":
        evaluate(args)
    else:
        compare(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
