"""Standalone contrastive stage: align brain representations with frozen CLIP targets.

No language model is built here. That is the point: the joint LM + InfoNCE run kept
faulting inside the LM's gradient-checkpoint recompute (a `cudaLaunchKernel` fault in
`libcuda.so.1` reached from `CheckpointFunction.backward`), and dropping the language
model removes that site entirely.
"""

import logging
import math
import os

import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate
from lightning.pytorch import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.models.losses.contrastive import multi_positive_infonce, similarity_diagnostics
from src.modules.contrastive_projector import BrainContrastiveProjector
from src.utils.retrieval import retrieval_metrics

logger = logging.getLogger(__name__)


def load_encoder_weights(encoder, checkpoint):
    """Restore the encoder out of a MindLLM checkpoint.

    `MindLLM.load_from_checkpoint` is avoided on purpose: it would build Vicuna-7B
    just to reach an encoder that never touches the language model.
    """
    state_dict = torch.load(checkpoint, map_location="cpu")["state_dict"]
    weights = {
        name[len("fmri_encoder.") :]: value
        for name, value in state_dict.items()
        if name.startswith("fmri_encoder.")
    }
    if not weights:
        raise ValueError(f"{checkpoint} holds no fmri_encoder.* tensors")
    encoder.load_state_dict(weights, strict=True)
    logger.info(
        f"[stage1] encoder init: loaded {len(weights)} tensors from {checkpoint}"
    )


class BrainContrastive(LightningModule):
    def __init__(
        self,
        encoder,
        target_cache,
        encoder_init=None,
        hidden_dim=1024,
        temperature_init=0.07,
        learnable_temperature=True,
        lr_encoder=1e-5,
        lr_projector=1e-4,
        weight_decay=0.01,
        warmup_ratio=0.05,
        encoder_out=None,
        mode="image",
        image_feature_path=None,
        text_feature_path=None,
        image_weight=1.0,
        text_weight=1.0,
        loss_reduction="sum",
        require_disjoint_stimuli=False,
        target_bank_on_cpu=True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])
        # `save_hyperparameters` only fills `self.hparams`, so the values the rest of
        # the class reads back off `self` are assigned explicitly here.
        self.learnable_temperature = learnable_temperature
        self.encoder_out = encoder_out
        self.lr_encoder = lr_encoder
        self.lr_projector = lr_projector
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        if mode not in ("image", "text", "image_text"):
            raise ValueError(f"unknown contrastive mode: {mode}")
        if loss_reduction not in ("sum", "mean"):
            raise ValueError(f"unknown loss_reduction: {loss_reduction}")
        self.mode = mode
        self.image_weight = image_weight
        self.text_weight = text_weight
        self.loss_reduction = loss_reduction
        self.require_disjoint_stimuli = require_disjoint_stimuli
        self.target_bank_on_cpu = target_bank_on_cpu
        self._used_train_ids = set()

        if mode == "text" and not text_feature_path:
            raise ValueError("text_feature_path is required for text mode")
        bank_path = text_feature_path if mode == "text" else (image_feature_path or target_cache)
        bank = torch.load(bank_path, map_location="cpu")
        embeddings = bank["embeddings"].float()
        coco_ids = [int(x) for x in bank["coco_ids"].tolist()]
        assert embeddings.size(0) == len(coco_ids)
        if mode != "image":
            if len(set(coco_ids)) != len(coco_ids):
                raise ValueError("target bank has duplicate coco_ids")
            if not torch.isfinite(embeddings).all() or not torch.allclose(
                embeddings.norm(dim=1), torch.ones(len(coco_ids)), atol=1e-3
            ):
                raise ValueError("target bank must contain finite unit-length embeddings")
        # New modes keep the large offline bank on CPU and move only selected
        # batch targets to the accelerator. M1 retains its original buffer path.
        if mode != "image" and target_bank_on_cpu:
            self.clip_bank = embeddings
        else:
            self.register_buffer("clip_bank", embeddings, persistent=False)
        self.clip_index = {coco_id: i for i, coco_id in enumerate(coco_ids)}
        if mode != "image":
            if not text_feature_path:
                raise ValueError("text_feature_path is required for text and image_text modes")
            text_bank = bank if mode == "text" else torch.load(text_feature_path, map_location="cpu")
            if text_bank.get("modality") != "text" or text_bank.get("caption_aggregation") != "mean_then_l2":
                raise ValueError("text bank must contain aggregated CLIP text targets")
            text_ids = [int(x) for x in text_bank["coco_ids"].tolist()]
            text_embeddings = text_bank["embeddings"].float()
            if len(set(text_ids)) != len(text_ids) or not torch.isfinite(text_embeddings).all() or not torch.allclose(
                text_embeddings.norm(dim=1), torch.ones(len(text_ids)), atol=1e-3
            ):
                raise ValueError("text bank must have unique ids and finite unit-length embeddings")
            if mode == "text":
                image_bank = torch.load(image_feature_path or target_cache, map_location="cpu")
                if set(int(x) for x in image_bank["coco_ids"].tolist()) != set(coco_ids):
                    raise ValueError("image and text banks have different coco_id sets")
                if image_bank.get("model_id") != bank.get("model_id"):
                    raise ValueError("image and text banks must use the same CLIP model_id")
                if image_bank["embeddings"].shape != embeddings.shape:
                    raise ValueError("image and text bank shapes differ")
            if mode == "image_text":
                if set(text_ids) != set(coco_ids):
                    raise ValueError("image and text banks have different coco_id sets")
                if bank.get("model_id") != text_bank.get("model_id"):
                    raise ValueError("image and text banks must use the same CLIP model_id")
                if text_embeddings.shape != embeddings.shape:
                    raise ValueError("image and text bank shapes differ")
                if target_bank_on_cpu:
                    self.text_bank = text_embeddings
                else:
                    self.register_buffer("text_bank", text_embeddings, persistent=False)
                self.text_index = {coco_id: i for i, coco_id in enumerate(text_ids)}
        logger.info(
            f"[stage1] {len(coco_ids)} targets of dim {embeddings.size(-1)} from {bank_path}"
        )

        self.fmri_encoder = instantiate(encoder)
        if encoder_init:
            load_encoder_weights(self.fmri_encoder, encoder_init)
        # The output dim is the bank's, not a free choice: the loss and the retrieval
        # both compare the two sides with a bare dot product, so any other dim is a
        # shape error somewhere deep instead of a config error here.
        self.contrastive_projector = BrainContrastiveProjector(
            brain_dim=encoder.token_dim,
            projection_dim=embeddings.size(-1),
            hidden_dim=hidden_dim,
        )

        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature_init)))
        else:
            self.register_buffer(
                "fixed_log_temperature",
                torch.tensor(math.log(temperature_init)),
                persistent=False,
            )

        self._val_z = []
        self._val_coco_ids = []
        self._val_subjects = []
        # Seeded from the score recorded in the existing artifact, so the "best" this
        # run tracks is the best the file has ever held rather than the best of the
        # current process. Without it every restart resets the bar to -1.0 and the
        # first post-resume validation overwrites `encoder_out` unconditionally --
        # possibly with a slightly worse encoder than the one already on disk.
        self._best_r1 = -1.0
        if encoder_out and os.path.isfile(encoder_out):
            metric_key = "val_brain2text_r@1" if mode == "text" else "val_brain2image_r@1"
            self._best_r1 = float(
                torch.load(encoder_out, map_location="cpu").get(
                    metric_key, -1.0
                )
            )

    @property
    def temperature(self):
        log_temperature = (
            self.log_temperature
            if self.learnable_temperature
            else self.fixed_log_temperature
        )
        return log_temperature.exp().clamp(min=1e-3, max=1.0)

    def _targets(self, coco_ids, modality="image"):
        bank = self.text_bank if modality == "text" and self.mode == "image_text" else self.clip_bank
        lookup = self.text_index if modality == "text" and self.mode == "image_text" else self.clip_index
        rows = []
        for coco_id in coco_ids.tolist():
            if coco_id not in lookup:
                raise KeyError(
                    f"no precomputed target for coco_id={coco_id}; rebuild the bank with "
                    f"scripts/precompute_clip_features.py --subjects ..."
                )
            rows.append(lookup[coco_id])
        index = torch.as_tensor(rows, dtype=torch.long, device=bank.device)
        return bank[index].to(self.device)

    def _modalities(self):
        return ("image", "text") if self.mode == "image_text" else (self.mode,)

    def _losses(self, brain_z, coco_ids, prefix):
        losses = {}
        for modality in self._modalities():
            target = self._targets(coco_ids, modality)
            loss = multi_positive_infonce(brain_z, target, coco_ids, self.temperature)
            pos, neg = similarity_diagnostics(brain_z, target, coco_ids)
            losses[modality] = loss
            self.log(f"{prefix}/{modality}_infonce", loss, on_step=prefix == "train", on_epoch=True)
            self.log(f"{prefix}/{modality}_pos_sim", pos, on_step=prefix == "train", on_epoch=True)
            self.log(f"{prefix}/{modality}_neg_sim", neg, on_step=prefix == "train", on_epoch=True)
        total = sum((self.image_weight if key == "image" else self.text_weight) * value for key, value in losses.items())
        if self.mode == "image_text" and self.loss_reduction == "mean":
            total = total / 2
        self.log(f"{prefix}/contrastive_total", total, on_step=prefix == "train", on_epoch=True)
        if prefix == "train":
            self.log("train/total_loss", total, on_step=True, on_epoch=True)
        return total

    def _embed(self, batch):
        brain_tokens = self.fmri_encoder(batch["voxels"], subject=batch["subject"])
        return self.contrastive_projector(brain_tokens)

    def training_step(self, batch, batch_idx):
        brain_z = self._embed(batch)
        if self.mode != "image":
            self._used_train_ids.update(int(x) for x in batch["coco_id"].tolist())
            return self._losses(brain_z, batch["coco_id"], "train")
        targets = self._targets(batch["coco_id"])
        loss = multi_positive_infonce(
            brain_z, targets, batch["coco_id"], self.temperature
        )
        pos_sim, neg_sim = similarity_diagnostics(brain_z, targets, batch["coco_id"])
        self.log("train/contrastive_loss", loss, prog_bar=True)
        self.log("train/pos_sim", pos_sim)
        self.log("train/neg_sim", neg_sim)
        self.log("train/temperature", self.temperature.detach())
        return loss

    def validation_step(self, batch, batch_idx):
        brain_z = self._embed(batch)
        if self.mode != "image":
            self._losses(brain_z, batch["coco_id"], "val")
        self._val_z.append(brain_z.float().detach().cpu())
        self._val_coco_ids.append(batch["coco_id"].detach().cpu())
        self._val_subjects.extend(batch["subject"])

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking or not self._val_z:
            self._val_z.clear()
            self._val_coco_ids.clear()
            self._val_subjects.clear()
            return

        query_z = torch.cat(self._val_z, dim=0)
        coco_ids = torch.cat(self._val_coco_ids, dim=0)
        if self.mode != "image":
            from src.utils.retrieval import retrieval_metrics_unique_gallery
            modality_metrics = {}
            for modality in self._modalities():
                target = self._targets(coco_ids, modality).float().cpu()
                result = retrieval_metrics_unique_gallery(query_z, target, coco_ids)
                modality_metrics[modality] = result
                for name, value in result.items():
                    self.log(f"val/brain2{modality}_{name}", value, on_step=False, on_epoch=True)
                logger.info(f"[retrieval/{modality}] epoch={self.current_epoch} " + " ".join(f"{name}={value:.4f}" for name, value in result.items()))
            primary = "text" if self.mode == "text" else "image"
            score = modality_metrics[primary]["r@1"]
            self._save_encoder(score, f"val_brain2{primary}_r@1")
            self._val_z.clear()
            self._val_coco_ids.clear()
            self._val_subjects.clear()
            return
        targets = self._targets(coco_ids).float().cpu()
        metrics = retrieval_metrics(query_z, targets)
        for name, value in metrics.items():
            self.log(f"val/brain2image_{name}", value, on_step=False, on_epoch=True)
        logger.info(
            f"[retrieval] epoch={self.current_epoch} n={query_z.size(0)} "
            + " ".join(f"{name}={value:.4f}" for name, value in metrics.items())
        )

        # One subject's representation can collapse while the pooled number still
        # looks healthy, and the pooled one is the only figure the run reports.
        subjects = np.array(self._val_subjects)
        for subject in sorted(set(self._val_subjects)):
            rows = torch.as_tensor(subjects == subject)
            per_subject = retrieval_metrics(query_z[rows], targets[rows])
            logger.info(
                f"[retrieval] epoch={self.current_epoch} {subject} n={int(rows.sum())} "
                + " ".join(f"{name}={value:.4f}" for name, value in per_subject.items())
            )

        # `global_step == 0` means this validation ran before any training, i.e. the
        # standalone `trainer.validate()` that `train_contrastive.py` calls ahead of
        # `fit()`. The checkpoint restore happens inside `fit()`, so at that point the
        # encoder still holds its `encoder_init` weights and saving would overwrite a
        # trained `encoder_out` with the untrained starting point.
        if (
            self.encoder_out
            and self.global_step > 0
            and metrics["r@1"] > self._best_r1
        ):
            self._best_r1 = metrics["r@1"]
            directory = os.path.dirname(self.encoder_out)
            if directory:
                os.makedirs(directory, exist_ok=True)
            torch.save(
                {
                    "fmri_encoder": self.fmri_encoder.state_dict(),
                    "epoch": self.current_epoch,
                    "val_brain2image_r@1": metrics["r@1"],
                },
                self.encoder_out,
            )
            logger.info(
                f"[stage1] saved encoder to {self.encoder_out} "
                f"(r@1={metrics['r@1']:.4f})"
            )

        self._val_z.clear()
        self._val_coco_ids.clear()
        self._val_subjects.clear()

    def _save_encoder(self, score, metric_key):
        if self.encoder_out and self.global_step > 0 and score > self._best_r1:
            self._best_r1 = score
            directory = os.path.dirname(self.encoder_out)
            if directory:
                os.makedirs(directory, exist_ok=True)
            torch.save({"fmri_encoder": self.fmri_encoder.state_dict(),
                        "epoch": self.current_epoch, metric_key: score}, self.encoder_out)
            logger.info(f"[contrastive-pretrain] saved encoder to {self.encoder_out} ({metric_key}={score:.4f})")

    def on_train_epoch_end(self):
        if self.mode == "image":
            return
        used = self._used_train_ids
        data = self.trainer.datamodule
        held_out = getattr(data, "test_coco_ids", set())
        val_ids = getattr(data, "val_coco_ids", set())
        overlap = used & held_out
        val_overlap = used & val_ids
        logger.info(f"[contrastive-pretrain] mode={self.mode} used_unique_coco_ids={len(used)} test_overlap={len(overlap)} val_overlap={len(val_overlap)}")
        if overlap and self.require_disjoint_stimuli:
            raise ValueError(f"training used {len(overlap)} held-out/test coco_ids; first: {sorted(overlap)[:10]}")
        if val_overlap and self.require_disjoint_stimuli:
            raise ValueError(f"training used {len(val_overlap)} validation coco_ids; first: {sorted(val_overlap)[:10]}")
        self._used_train_ids.clear()

    def configure_optimizers(self):
        groups = [
            {"params": self.fmri_encoder.parameters(), "lr": self.lr_encoder},
            {"params": self.contrastive_projector.parameters(), "lr": self.lr_projector},
        ]
        if self.learnable_temperature:
            groups.append(
                {
                    "params": [self.log_temperature],
                    "lr": self.lr_projector,
                    "weight_decay": 0.0,
                }
            )
        optimizer = AdamW(groups, weight_decay=self.weight_decay)

        total_steps = max(1, self.trainer.estimated_stepping_batches)
        warmup_steps = max(1, int(total_steps * self.warmup_ratio))

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": LambdaLR(optimizer, lr_lambda),
                "interval": "step",
            },
        }
