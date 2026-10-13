"""Compare M0/M4 on identical held-out caption batches without training.

The validation loader currently reads NSD's shared1000 test group, including
when data.split_val=true. This script reports that provenance explicitly. It
loads the two checkpoints sequentially to fit a single 24 GB GPU.
"""

import argparse
import gc
import hashlib
import json
import logging
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from lightning.pytorch import Callback, Trainer

from src.dataset import MultiSourceBatchSampler, create_tokenizer
from src.models.mindllm import MindLLM
from src.utils.retrieval import retrieval_metrics_unique_gallery

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m0", required=True, help="M0 caption checkpoint")
    parser.add_argument("--m4", required=True, help="M4 caption checkpoint")
    parser.add_argument("--image-bank", required=True, help="M4 CLIP image bank")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--val-batches", type=int, default=250,
        help="same leading shared1000 batches for both arms; 250 x 4 = 1000",
    )
    parser.add_argument("--out", default="outputs/m0_m4_diagnostic.json")
    return parser.parse_args()


def checksum(ids):
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


def source_ids(group):
    return {
        int(group[key]["coco_id"][()])
        for key in group
    }


def audit_data(data, batch_size):
    """Check holdout provenance and the exact equality mask seen by M4 training."""
    subject = next(iter(data.file["train"]))
    train_ids = source_ids(data.file["train"][subject])
    reserved_ids = source_ids(data.file["val"][subject])
    shared_ids = source_ids(data.file["test"][subject])
    if train_ids & shared_ids or reserved_ids & shared_ids or train_ids & reserved_ids:
        raise ValueError("train, reserved900, and shared1000 COCO IDs must be disjoint")

    data.setup("fit")
    sampler = MultiSourceBatchSampler(
        data.train_data, batch_size=batch_size, shuffle=False,
        drop_last=False, by="coco",
    )
    total_batches = 0
    off_diagonal_positives = 0
    batches_with_multiple_positives = 0
    multi_repeat_stimuli = 0
    for rows in sampler:
        ids = torch.tensor([int(row[2]) for row in rows], dtype=torch.long)
        mask = ids[:, None].eq(ids[None, :])
        if not torch.all(mask.diag()):
            raise ValueError("positive mask has a missing diagonal positive")
        off_diagonal = int(mask.sum().item()) - len(ids)
        total_batches += 1
        off_diagonal_positives += off_diagonal
        batches_with_multiple_positives += int(off_diagonal > 0)
        multi_repeat_stimuli += sum(
            isinstance(row[1], tuple) and len(row[1]) > 1 for row in rows
        )
    return {
        "training_stimuli": len(train_ids),
        "reserved900_stimuli": len(reserved_ids),
        "shared1000_stimuli": len(shared_ids),
        "train_vs_shared1000_overlap": len(train_ids & shared_ids),
        "train_batches": total_batches,
        "train_batch_off_diagonal_positive_mask_entries": off_diagonal_positives,
        "train_batches_with_multiple_positives": batches_with_multiple_positives,
        "training_items_with_multiple_fmri_repeats": multi_repeat_stimuli,
        "mask_definition": "coco_ids[:, None] == coco_ids[None, :]",
        "repeat_handling": "one randomly chosen repeat per training stimulus",
        "validation_source": "shared1000 (file['test'])",
        "checkpoint_selection_source": "shared1000 (file['test'])",
    }


class CollectValidation(Callback):
    def __init__(self, caption_head=None):
        self.ids = []
        self.z = []
        self.off_diagonal_positives = 0
        self._hook = None
        if caption_head is not None:
            self._hook = caption_head.register_forward_hook(self._capture_z)

    def _capture_z(self, module, inputs, output):
        self.z.append(output.detach().float().cpu())

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        ids = [int(value) for value in batch["sample_or_coco_id"]]
        mask = torch.tensor(ids)[:, None].eq(torch.tensor(ids)[None, :])
        self.off_diagonal_positives += int(mask.sum().item()) - len(ids)
        self.ids.extend(ids)

    def close(self):
        if self._hook is not None:
            self._hook.remove()


def run_arm(name, checkpoint, cfg, data, tokenizer, val_batches):
    model_kwargs = {
        "encoder": cfg.encoder,
        "model_id": cfg.model_id,
        "lm_checkpointing": "off",
        "tokenizer": tokenizer,
        "peft_config": None,
        "learning_rate": 1e-4,
        "global_infonce": cfg.global_infonce,
        "contrastive": cfg.contrastive,
    }
    logger.info("[%s] loading %s", name, checkpoint)
    model = MindLLM.load_from_checkpoint(checkpoint, strict=False, **model_kwargs)
    collector = CollectValidation(model.caption_head if name == "M4" else None)
    trainer = Trainer(
        accelerator="gpu", devices=1, precision="bf16-mixed",
        logger=False, enable_checkpointing=False, callbacks=[collector],
        limit_val_batches=val_batches, num_sanity_val_steps=0,
    )
    try:
        metrics = trainer.validate(model=model, datamodule=data)[0]
        result = {
            "checkpoint": str(checkpoint),
            "n_queries": len(collector.ids),
            "ordered_coco_ids_sha256": checksum(collector.ids),
            "val/token_loss": float(metrics["val/token_loss"]),
            "val_positive_mask_off_diagonal_entries": collector.off_diagonal_positives,
        }
        if name == "M4":
            if not collector.z:
                raise ValueError("M4 caption head was not called during validation")
            query_z = torch.cat(collector.z)
            if query_z.size(0) != len(collector.ids):
                raise ValueError("caption head hook and validation loader disagree on row count")
            ids = torch.tensor(collector.ids, dtype=torch.long)
            targets = model.caption_bank[[model.caption_index[int(x)] for x in ids]]
            retrieval = retrieval_metrics_unique_gallery(query_z, targets, ids)
            result.update({
                "heldout_gallery_unique_images": len(set(collector.ids)),
                "retrieval_top1": retrieval["r@1"],
                "retrieval_top5": retrieval["r@5"],
                "chance_top1": 1 / len(set(collector.ids)),
                "val/contrastive_loss": float(metrics["val/contrastive_loss"]),
                "val/in_batch_contrastive_top1": float(metrics["val/contrastive_top1"]),
            })
        return result
    finally:
        collector.close()
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    if args.val_batches < 1 or args.batch_size < 1:
        raise ValueError("val-batches and batch-size must be positive")
    for path in (args.m0, args.m4, args.image_bank):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "config"), version_base=None
    ):
        cfg = compose(config_name="default")
    cfg.data.task = "coco-caption"
    cfg.data.subjects = [args.subject]
    cfg.data.group_by_coco = True
    cfg.data.split_val = True
    cfg.data.batch_size = args.batch_size
    cfg.data.num_workers = 0
    cfg.data.mixup = False
    cfg.contrastive.enabled = False
    tokenizer = create_tokenizer(cfg.model_id)
    data = instantiate(cfg.data, tokenizer=tokenizer)
    audit = audit_data(data, args.batch_size)
    if len(data.val_dataloader()) < args.val_batches:
        raise ValueError("val-batches exceeds the shared1000 validation loader")
    result = {"audit": audit, "eval_batches": args.val_batches}
    result["M0"] = run_arm("M0", args.m0, cfg, data, tokenizer, args.val_batches)
    cfg.contrastive.enabled = True
    cfg.contrastive.mode = "image"
    cfg.contrastive.target_path = args.image_bank
    result["M4"] = run_arm("M4", args.m4, cfg, data, tokenizer, args.val_batches)
    if result["M0"]["ordered_coco_ids_sha256"] != result["M4"]["ordered_coco_ids_sha256"]:
        raise ValueError("M0 and M4 did not evaluate the same ordered COCO IDs")
    if result["M0"]["n_queries"] != result["M4"]["n_queries"]:
        raise ValueError("M0 and M4 evaluated different numbers of queries")
    if args.val_batches == len(data.val_dataloader()):
        if result["M4"]["heldout_gallery_unique_images"] != audit["shared1000_stimuli"]:
            raise ValueError("the full validation pass did not cover every shared1000 image")
    result["delta_M4_minus_M0_token_loss"] = (
        result["M4"]["val/token_loss"] - result["M0"]["val/token_loss"]
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
