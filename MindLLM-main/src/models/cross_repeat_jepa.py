"""Standalone cross-repeat JEPA with an EMA target and frozen CLIP image anchor."""

import copy
import logging
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from lightning.pytorch import LightningModule
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.models.brain_contrastive import load_encoder_weights
from src.models.losses.cross_repeat_jepa import cosine_distance, ema_update
from src.modules.contrastive_projector import BrainContrastiveProjector
from src.utils.retrieval import retrieval_metrics_unique_gallery

logger = logging.getLogger(__name__)


class CrossRepeatSemanticJEPA(LightningModule):
    def __init__(
        self,
        encoder,
        image_feature_path,
        encoder_init,
        encoder_out,
        model_id="openai/clip-vit-large-patch14",
        hidden_dim=1024,
        predictor_hidden_dim=1024,
        jepa_weight=1.0,
        semantic_weight=1.0,
        ema_decay=0.996,
        lr_encoder=1e-5,
        lr_head=1e-4,
        weight_decay=0.01,
        warmup_ratio=0.05,
        require_disjoint_stimuli=True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])
        if not encoder_init:
            raise ValueError("encoder_init must point to the original MindLLM checkpoint")
        if jepa_weight <= 0 or semantic_weight <= 0:
            raise ValueError("both JEPA and semantic loss weights must be positive")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must lie in [0, 1)")
        if not 0.0 <= warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must lie in [0, 1)")

        bank = torch.load(image_feature_path, map_location="cpu")
        ids = [int(value) for value in bank["coco_ids"].tolist()]
        embeddings = bank["embeddings"].float()
        if bank.get("model_id") != model_id or bank.get("modality") == "text":
            raise ValueError("semantic targets must be from the configured CLIP image model")
        if len(ids) != len(set(ids)) or embeddings.ndim != 2 or embeddings.size(0) != len(ids):
            raise ValueError("CLIP image bank must have one embedding per unique coco_id")
        if not torch.isfinite(embeddings).all() or not torch.allclose(
            embeddings.norm(dim=1), torch.ones(len(ids)), atol=1e-3
        ):
            raise ValueError("CLIP image bank must contain finite unit-length vectors")
        self.image_bank = embeddings  # kept on CPU and excluded from checkpoints
        self.image_index = {coco_id: row for row, coco_id in enumerate(ids)}

        self.fmri_encoder = instantiate(encoder)
        load_encoder_weights(self.fmri_encoder, encoder_init)
        self.semantic_projector = BrainContrastiveProjector(
            brain_dim=encoder.token_dim,
            projection_dim=embeddings.size(1),
            hidden_dim=hidden_dim,
        )
        self.predictor = nn.Sequential(
            nn.Linear(embeddings.size(1), predictor_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(predictor_hidden_dim),
            nn.Linear(predictor_hidden_dim, embeddings.size(1)),
        )
        self.target_encoder = copy.deepcopy(self.fmri_encoder)
        self.target_projector = copy.deepcopy(self.semantic_projector)
        self.target_encoder.requires_grad_(False).eval()
        self.target_projector.requires_grad_(False).eval()

        self.jepa_weight = float(jepa_weight)
        self.semantic_weight = float(semantic_weight)
        self.ema_decay = float(ema_decay)
        self.lr_encoder = float(lr_encoder)
        self.lr_head = float(lr_head)
        self.weight_decay = float(weight_decay)
        self.warmup_ratio = float(warmup_ratio)
        self.encoder_out = encoder_out
        self.require_disjoint_stimuli = require_disjoint_stimuli
        self._used_train_ids = set()
        self._val_z = []
        self._val_ids = []
        self._best_r1 = -1.0
        if encoder_out and os.path.isfile(encoder_out):
            artifact = torch.load(encoder_out, map_location="cpu")
            self._best_r1 = float(artifact.get("val_brain2image_r@1", -1.0))
        logger.info(
            "[cross-repeat-jepa] image_targets=%d dim=%d "
            "jepa_weight=%.4f semantic_weight=%.4f ema_decay=%.5f",
            len(ids), embeddings.size(1), self.jepa_weight,
            self.semantic_weight, self.ema_decay,
        )

    def train(self, mode=True):
        super().train(mode)
        # The online encoder uses its usual dropout; EMA targets remain stable.
        self.target_encoder.eval()
        self.target_projector.eval()
        return self

    def _image_targets(self, coco_ids, device):
        rows = []
        for value in coco_ids.tolist():
            coco_id = int(value)
            if coco_id not in self.image_index:
                raise KeyError(f"CLIP image bank lacks coco_id={coco_id}")
            rows.append(self.image_index[coco_id])
        return self.image_bank[rows].to(device)

    def _losses(self, batch):
        context = batch["context_voxels"]
        target = batch["target_voxels"]
        subject = batch["subject"]
        online_z = self.semantic_projector(self.fmri_encoder(context, subject=subject))
        with torch.no_grad():
            target_z = self.target_projector(self.target_encoder(target, subject=subject))
        predicted_z = F.normalize(self.predictor(online_z.float()), dim=-1)
        image_z = self._image_targets(batch["coco_id"], online_z.device)
        jepa_loss = cosine_distance(predicted_z, target_z)
        semantic_loss = cosine_distance(online_z, image_z)
        total = self.jepa_weight * jepa_loss + self.semantic_weight * semantic_loss
        return total, jepa_loss, semantic_loss, online_z

    def training_step(self, batch, batch_idx):
        total, jepa_loss, semantic_loss, _ = self._losses(batch)
        batch_size = batch["coco_id"].numel()
        self.log("train/loss", total, prog_bar=True, batch_size=batch_size)
        self.log("train/jepa_loss", jepa_loss, batch_size=batch_size)
        self.log("train/semantic_loss", semantic_loss, batch_size=batch_size)
        self._used_train_ids.update(int(value) for value in batch["coco_id"].tolist())
        return total

    def validation_step(self, batch, batch_idx):
        total, jepa_loss, semantic_loss, online_z = self._losses(batch)
        batch_size = batch["coco_id"].numel()
        self.log("val/loss", total, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("val/jepa_loss", jepa_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("val/semantic_loss", semantic_loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self._val_z.append(online_z.detach().float().cpu())
        self._val_ids.append(batch["coco_id"].detach().cpu())

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking or not self._val_z:
            self._val_z.clear()
            self._val_ids.clear()
            return
        query_z = torch.cat(self._val_z)
        coco_ids = torch.cat(self._val_ids)
        target_z = self._image_targets(coco_ids, torch.device("cpu"))
        metrics = retrieval_metrics_unique_gallery(query_z, target_z, coco_ids)
        for name, value in metrics.items():
            self.log(f"val/brain2image_{name}", value, on_step=False, on_epoch=True)
        logger.info(
            "[cross-repeat-jepa] epoch=%d val_queries=%d val_stimuli=%d %s",
            self.current_epoch, len(coco_ids), len(set(coco_ids.tolist())),
            " ".join(f"{name}={value:.4f}" for name, value in metrics.items()),
        )
        score = metrics["r@1"]
        if self.encoder_out and self.global_step > 0 and score > self._best_r1:
            self._best_r1 = score
            directory = os.path.dirname(self.encoder_out)
            if directory:
                os.makedirs(directory, exist_ok=True)
            data = self.trainer.datamodule
            torch.save(
                {
                    "fmri_encoder": self.fmri_encoder.state_dict(),
                    "epoch": self.current_epoch,
                    "val_brain2image_r@1": score,
                    "method": "cross_repeat_semantic_jepa",
                    "subjects": list(data.subjects),
                    "split_seed": data.split_seed,
                    "val_coco_ids": sorted(data.val_coco_ids),
                },
                self.encoder_out,
            )
            logger.info("[cross-repeat-jepa] saved best encoder to %s", self.encoder_out)
        self._val_z.clear()
        self._val_ids.clear()

    def on_train_epoch_end(self):
        data = self.trainer.datamodule
        val_overlap = self._used_train_ids & getattr(data, "val_coco_ids", set())
        test_overlap = self._used_train_ids & getattr(data, "test_coco_ids", set())
        logger.info(
            "[cross-repeat-jepa] train_stimuli=%d val_overlap=%d test_overlap=%d",
            len(self._used_train_ids), len(val_overlap), len(test_overlap),
        )
        self._used_train_ids.clear()
        if self.require_disjoint_stimuli and (val_overlap or test_overlap):
            raise ValueError("train stimuli overlap validation or test stimuli")

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer.step(closure=optimizer_closure)
        ema_update(self.target_encoder, self.fmri_encoder, self.ema_decay)
        ema_update(self.target_projector, self.semantic_projector, self.ema_decay)

    def configure_optimizers(self):
        optimizer = AdamW(
            [
                {"params": self.fmri_encoder.parameters(), "lr": self.lr_encoder},
                {"params": self.semantic_projector.parameters(), "lr": self.lr_head},
                {"params": self.predictor.parameters(), "lr": self.lr_head},
            ],
            weight_decay=self.weight_decay,
        )
        total_steps = max(1, self.trainer.estimated_stepping_batches)
        warmup_steps = max(1, int(total_steps * self.warmup_ratio))

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": LambdaLR(optimizer, lr_lambda), "interval": "step"},
        }
