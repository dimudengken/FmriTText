"""Frozen brain representation retrieval: identical ridge probes plus native head.

Run from the repository root with python -m scripts.compare_global_retrieval.
No encoder or caption parameters are updated. Ridge alpha is fixed in advance.
"""
import argparse
import gc
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from lightning import seed_everything
from lightning.fabric.utilities.apply_func import move_data_to_device

from scripts.compare_m0_m4_diagnostics import audit_data, checksum
from src.dataset import create_tokenizer
from src.models.mindllm import MindLLM, adapt_voxels


def score(queries, gallery, query_ids, gallery_ids):
    """Use strict argmax Top-1 and deterministic ID ordering to break ties."""
    order = torch.argsort(gallery_ids)
    gallery, gallery_ids = gallery[order], gallery_ids[order]
    index = {int(v): i for i, v in enumerate(gallery_ids)}
    correct = torch.tensor([index[int(v)] for v in query_ids])
    similarities = F.normalize(queries.float(), dim=1) @ F.normalize(gallery.float(), dim=1).T
    ranked = torch.argsort(similarities, dim=1, descending=True, stable=True)
    ranks = (ranked == correct[:, None]).nonzero()[:, 1] + 1
    return {**{f"top{k}": float((ranks <= k).float().mean()) for k in (1, 5, 10)},
            "median_rank": float(ranks.float().median()), "mean_rank": float(ranks.float().mean())}


def ridge_predict(train_x, train_y, test_x, alpha):
    # Identical train-only preprocessing and fixed regularization in both arms.
    train_x, test_x = F.normalize(train_x.double(), dim=1), F.normalize(test_x.double(), dim=1)
    x_mean, y_mean = train_x.mean(0), train_y.double().mean(0)
    x, y = train_x - x_mean, train_y.double() - y_mean
    gram = x.T @ x
    gram.diagonal().add_(alpha)
    weights = torch.linalg.solve(gram, x.T @ y)
    return ((test_x - x_mean) @ weights + y_mean).float()


@torch.inference_mode()
def extract(model, loader, limit, native=False):
    features, native_z, ids = [], [], []
    captured = []
    def capture(module, inputs, output):
        tokens = output["fmri_token_embeds"] if isinstance(output, dict) else output
        captured.append(tokens.detach().float().mean(1).cpu())
    hook = model.fmri_encoder.register_forward_hook(capture)
    try:
        for batch_index, batch in enumerate(loader):
            if limit and batch_index >= limit:
                break
            batch = move_data_to_device(batch, torch.device("cuda"))
            batch["voxels"] = adapt_voxels(batch["voxels"], True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                projected = model.forward(**batch, return_fmri_tokens=True)
                if native:
                    native_z.append(model.global_head(projected.float()).float().cpu())
            features.append(captured.pop())
            ids.extend(int(v) for v in batch["sample_or_coco_id"])
            if batch_index % 100 == 0:
                logging.info("extracted batches=%d images=%d native=%s", batch_index + 1, len(ids), native)
    finally:
        hook.remove()
    return torch.cat(features), torch.tensor(ids), torch.cat(native_z) if native_z else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True)
    p.add_argument("--global-checkpoint", required=True)
    p.add_argument("--image-bank", required=True)
    p.add_argument("--subject", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--train-batches", type=int, default=0, help="0 uses all training stimuli; fixed for both arms")
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--out", default="outputs/global_retrieval_comparison.json")
    args = p.parse_args()
    if args.ridge_alpha <= 0 or args.batch_size < 1 or args.train_batches < 0:
        p.error("positive alpha/batch-size and nonnegative train-batches required")
    for path in (args.baseline, args.global_checkpoint, args.image_bank):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "config"), version_base=None):
        cfg = compose(config_name="default")
    cfg.data.task, cfg.data.subjects = "coco-caption", [args.subject]
    cfg.data.group_by_coco, cfg.data.split_val = True, True
    cfg.data.batch_size, cfg.data.num_workers, cfg.data.mixup = args.batch_size, 0, False
    cfg.contrastive.enabled = False
    tokenizer = create_tokenizer(cfg.model_id)
    data = instantiate(cfg.data, tokenizer=tokenizer)
    audit = audit_data(data, args.batch_size)
    bank = torch.load(args.image_bank, map_location="cpu")
    bank_ids, targets = bank["coco_ids"].long(), bank["embeddings"].float()
    if len(set(bank_ids.tolist())) != len(bank_ids) or targets.ndim != 2 or len(targets) != len(bank_ids):
        raise ValueError("image bank must contain one embedding per unique COCO ID")
    if bank.get("modality") == "text":
        raise ValueError("an image bank is required")
    bank_index = {int(v): i for i, v in enumerate(bank_ids)}
    def lookup(ids):
        missing = sorted(set(ids.tolist()) - bank_index.keys())
        if missing:
            raise ValueError(f"image bank lacks {len(missing)} required images, first={missing[:10]}; rebuild it including held-out stimuli")
        return targets[[bank_index[int(v)] for v in ids]]
    result = {"audit": audit, "protocol": "frozen mean-pooled raw encoder tokens; same train-only ridge probe",
              "ridge_alpha": args.ridge_alpha, "selection_caveat": "shared1000 was also used for caption checkpoint selection"}
    for name, checkpoint in (("baseline", args.baseline), ("global", args.global_checkpoint)):
        seed_everything(42, workers=True)
        cfg.global_infonce.enabled = name == "global"
        cfg.global_infonce.feature_path = args.image_bank if name == "global" else None
        model = MindLLM.load_from_checkpoint(checkpoint, strict=False, encoder=cfg.encoder,
            model_id=cfg.model_id, lm_checkpointing="off", tokenizer=tokenizer, peft_config=None,
            learning_rate=cfg.lr, global_infonce=cfg.global_infonce, contrastive=cfg.contrastive)
        # strict=False permits omitted frozen LLM weights, never random probe inputs.
        saved_checkpoint = torch.load(checkpoint, map_location="cpu")
        if "peft_model" in saved_checkpoint:
            raise ValueError("this comparison requires frozen-LLM checkpoints (lora=false)")
        saved = saved_checkpoint["state_dict"]
        prefixes = ("fmri_encoder.", "projector.") + (("global_head.",) if name == "global" else ())
        required = {key for key in model.state_dict() if key.startswith(prefixes)}
        missing = sorted(required - saved.keys())
        if missing:
            raise ValueError(f"{name} checkpoint lacks trained weights: {missing[:10]}")
        del saved, saved_checkpoint
        model.eval().to("cuda")
        seed_everything(42, workers=True)
        train_x, train_ids, _ = extract(model, data.train_dataloader(), args.train_batches)
        test_x, test_ids, native_z = extract(model, data.val_dataloader(), 0, name == "global")
        if len(set(test_ids.tolist())) != audit["shared1000_stimuli"]:
            raise ValueError("evaluation did not cover every shared1000 image")
        if set(train_ids.tolist()) & set(test_ids.tolist()):
            raise ValueError("probe training and evaluation overlap")
        train_y = lookup(train_ids)
        gallery_ids = torch.tensor(sorted(set(test_ids.tolist())))
        gallery = lookup(gallery_ids)
        probe_z = ridge_predict(train_x, train_y, test_x, args.ridge_alpha)
        arm = {"checkpoint": checkpoint, "train_queries": len(train_ids), "eval_queries": len(test_ids),
               "gallery_images": len(gallery_ids), "chance_top1": 1 / len(gallery_ids),
               "train_ids_sha256": checksum(train_ids.tolist()), "eval_ids_sha256": checksum(test_ids.tolist()),
               "common_ridge_probe": score(probe_z, gallery, test_ids, gallery_ids)}
        if native_z is not None:
            arm["native_global_head"] = score(native_z, gallery, test_ids, gallery_ids)
        result[name] = arm
        logging.info("%s results: %s", name, json.dumps(arm))
        del model, train_x, test_x, native_z, probe_z
        gc.collect()
        torch.cuda.empty_cache()
    for key in ("train_ids_sha256", "eval_ids_sha256"):
        if result["baseline"][key] != result["global"][key]:
            raise ValueError(f"baseline/global sample mismatch: {key}")
    result["global_minus_baseline_probe"] = {key: result["global"]["common_ridge_probe"][key] - value
        for key, value in result["baseline"]["common_ridge_probe"].items()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
