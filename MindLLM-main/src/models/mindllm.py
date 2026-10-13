import logging
import numbers
import os
import torch
import itertools
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AutoModelForCausalLM,
)
from src.dataset import IGNORE_INDEX
from src.metrics import *
from src.dataset import FMRI_TOKEN
from src.models.losses.contrastive import (
    multi_positive_infonce,
    similarity_diagnostics,
)
from src.models.losses.caption_joint import (
    caption_contrastive_metrics,
    lookup_caption_targets,
    warmup_weight,
)
from src.modules.global_semantic_head import GlobalSemanticHead
from src.modules.contrastive_projector import BrainContrastiveProjector
from src.utils import adapt_voxels, safe_get_source
from src.utils.retrieval import retrieval_metrics
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from accelerate.hooks import AlignDevicesHook
from torchmetrics.text import Perplexity

from hydra.utils import instantiate


def remove_hook_from_module(
    module: torch.nn.Module, recurse=False, hook_cls=AlignDevicesHook
):

    if hasattr(module, "_hf_hook") and isinstance(module._hf_hook, hook_cls):
        module._hf_hook.detach_hook(module)
        delattr(module, "_hf_hook")

        if hasattr(module, "_old_forward"):
            module.forward = module._old_forward
            delattr(module, "_old_forward")

    if recurse:
        for child in module.children():
            remove_hook_from_module(child, recurse)

    return module


class WarmupScheduler(LambdaLR):
    def __init__(self, optimizer, warmup_steps, base_lr):
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        super().__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return 1.0


class MindLLM(MetricsRegistered):
    def __init__(
        self,
        encoder,
        model_id,
        tokenizer,
        learning_rate=1e-3,
        peft_config=None,
        scheduler_type=None,
        global_infonce=None,
        contrastive=None,
        lm_checkpointing="off",
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.peft_config = peft_config
        self.scheduler_type = scheduler_type

        self.save_hyperparameters()

        self.model_id = model_id
        attn_implementation = (
            "flash_attention_2"
            if (
                os.environ.get("WANDB_MODE") != "disabled" and torch.cuda.is_available()
            )
            else "sdpa"
        )

        self.lm = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        remove_hook_from_module(self.lm, recurse=True)

        if lm_checkpointing not in ("reentrant", "non_reentrant", "off"):
            raise ValueError(f"unknown lm_checkpointing mode: {lm_checkpointing}")
        if lm_checkpointing != "off":
            self.lm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": lm_checkpointing == "reentrant"
                }
            )
        logging.info("[lm] gradient_checkpointing=%s", lm_checkpointing)
        self.lm.resize_token_embeddings(len(tokenizer))
        self.lm.generation_config.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer
        if self.peft_config is None:
            for param in self.lm.parameters():
                param.requires_grad = False
        else:
            self.lm = get_peft_model(self.lm, self.peft_config)

        self.fmri_encoder = instantiate(encoder)

        self.projector = nn.Linear(
            encoder.token_dim, self.lm.get_input_embeddings().embedding_dim
        )

        # Built after self.projector on purpose: constructing it consumes torch RNG,
        # so any earlier placement would change the projector init and make
        # `global_infonce.enabled=false` runs diverge from the reproduced baseline.
        self.global_infonce_cfg = dict(global_infonce or {})
        self.global_lambda = float(self.global_infonce_cfg.get("lambda_loss", 0.05))
        self.global_temperature = float(
            self.global_infonce_cfg.get("temperature", 0.07)
        )
        self.global_head = None
        self.clip_index = None
        self.use_global_infonce = bool(
            self.global_infonce_cfg.get("enabled", False)
        )
        self._setup_global_infonce()

        self._val_global_z = []
        self._val_global_ids = []
        self._global_diag = None

        self.caption_contrastive_cfg = dict(contrastive or {})
        self.use_caption_contrastive = bool(
            self.caption_contrastive_cfg.get("enabled", False)
        )
        self.caption_head = None
        if self.use_caption_contrastive:
            self._setup_caption_contrastive(encoder)
            logging.info(
                "[caption-contrastive] trainable: fmri_encoder=%d "
                "projector=%d contrastive_head=%d; frozen: llm=%d CLIP=offline",
                sum(p.numel() for p in self.fmri_encoder.parameters() if p.requires_grad),
                sum(p.numel() for p in self.projector.parameters() if p.requires_grad),
                sum(p.numel() for p in self.caption_head.parameters() if p.requires_grad),
                sum(p.numel() for p in self.lm.parameters() if not p.requires_grad),
            )

        self.perplexity = Perplexity(ignore_index=IGNORE_INDEX)

    def _setup_caption_contrastive(self, encoder):
        """M4/M5 branch on raw brain tokens; the frozen target bank stays on CPU."""
        cfg = self.caption_contrastive_cfg
        if self.use_global_infonce:
            raise ValueError("global_infonce and caption contrastive cannot both be enabled")
        if self.peft_config is not None:
            raise ValueError("M4/M5 require a frozen LLM; set lora=false")
        mode = cfg.get("mode")
        if mode not in ("image", "text"):
            raise ValueError(f"caption contrastive mode must be image or text, got {mode!r}")
        if cfg.get("pooling", "mean") != "mean":
            raise ValueError("M4/M5 currently require mean pooling")
        if not cfg.get("symmetric", True) or not cfg.get("multi_positive", True):
            raise ValueError("M4/M5 require symmetric multi-positive InfoNCE")
        target_path = cfg.get("target_path")
        if not target_path:
            raise ValueError("contrastive.target_path is required when enabled")
        bank = torch.load(target_path, map_location="cpu")
        ids = [int(value) for value in bank["coco_ids"].tolist()]
        embeddings = bank["embeddings"].float()
        if len(ids) != len(set(ids)) or embeddings.ndim != 2 or embeddings.size(0) != len(ids):
            raise ValueError("contrastive bank must have one embedding per unique coco_id")
        if bank.get("model_id") != cfg.get("model_id", "openai/clip-vit-large-patch14"):
            raise ValueError("contrastive bank uses a different CLIP model_id")
        if embeddings.size(1) != int(cfg.get("embed_dim", 768)):
            raise ValueError("contrastive bank dimension differs from contrastive.embed_dim")
        if not torch.isfinite(embeddings).all() or not torch.allclose(
            embeddings.norm(dim=1), torch.ones(len(ids)), atol=1e-3
        ):
            raise ValueError("contrastive bank must have finite L2-normalized embeddings")
        if mode == "text" and (
            bank.get("modality") != "text"
            or bank.get("caption_aggregation") != "mean_then_l2"
        ):
            raise ValueError("M5 requires a mean_then_l2 COCO text bank")
        if mode == "image" and bank.get("modality") == "text":
            raise ValueError("M4 requires an image CLIP bank")
        self.caption_bank = embeddings  # plain CPU tensor, excluded from state_dict
        self.caption_index = {coco_id: row for row, coco_id in enumerate(ids)}
        self.caption_mode = mode
        self.caption_temperature = float(cfg.get("temperature", 0.07))
        self.caption_loss_weight = float(cfg.get("loss_weight", 0.05))
        self.caption_warmup_steps = int(cfg.get("warmup_steps", 1000))
        if self.caption_temperature <= 0 or self.caption_loss_weight < 0 or self.caption_warmup_steps < 0:
            raise ValueError("temperature must be positive; weight and warmup must be nonnegative")
        self.caption_head = BrainContrastiveProjector(
            brain_dim=encoder.token_dim,
            projection_dim=embeddings.size(1),
            hidden_dim=int(cfg.get("hidden_dim", 1024)),
        )
        logging.info(
            f"[caption-contrastive] mode={mode} target={target_path} "
            f"rows={len(ids)} dim={embeddings.size(1)} target_unit=coco_id "
            f"aggregation={bank.get('caption_aggregation', 'none')} "
            f"weight={self.caption_loss_weight} warmup_steps={self.caption_warmup_steps}"
        )

    def _caption_targets(self, sample_or_coco_id, device):
        return lookup_caption_targets(
            self.caption_bank,
            self.caption_index,
            sample_or_coco_id,
            device,
            self.caption_mode,
        )

    def _caption_weight(self):
        return warmup_weight(
            self.global_step, self.caption_loss_weight, self.caption_warmup_steps
        )

    def _setup_global_infonce(self):
        """Load the frozen target bank and build the projection head.

        The head is built even when the loss is disabled as long as a feature bank
        is configured, so that "head present but switched off" can be run as a
        regression check on top of the baseline.
        """
        feature_path = self.global_infonce_cfg.get("feature_path")
        if self.use_global_infonce and not feature_path:
            raise ValueError(
                "global_infonce.enabled=true requires global_infonce.feature_path"
            )
        if not feature_path:
            self.use_global_infonce = False
            return

        bank = torch.load(feature_path, map_location="cpu")
        embeddings = bank["embeddings"].float()
        coco_ids = [int(x) for x in bank["coco_ids"].tolist()]
        assert embeddings.size(0) == len(coco_ids)

        self.register_buffer("clip_bank", embeddings, persistent=False)
        self.clip_index = {coco_id: i for i, coco_id in enumerate(coco_ids)}

        # Constructing the head draws from the torch RNG, and the encoder's dropout
        # masks are drawn from that same stream during training. Restoring the state
        # around the construction keeps enabled/disabled runs on the baseline's RNG
        # stream, so `enabled=false` reproduces the baseline bit for bit instead of
        # only on average.
        rng_state = torch.get_rng_state()
        self.global_head = GlobalSemanticHead(
            brain_dim=self.lm.get_input_embeddings().embedding_dim,
            out_dim=embeddings.size(-1),
            hidden_dim=self.global_infonce_cfg.get("hidden_dim"),
            pooling=self.global_infonce_cfg.get("pooling", "mean"),
        )
        torch.set_rng_state(rng_state)
        logging.info(
            f"[global_infonce] {len(coco_ids)} targets of dim {embeddings.size(-1)} "
            f"from {feature_path}, enabled={self.use_global_infonce}"
        )

    def resolve_global_targets(self, sample_or_coco_id, device):
        """Look up the frozen semantic target of each sample in the batch."""
        rows = []
        group_ids = []
        for coco_id in sample_or_coco_id:
            if isinstance(coco_id, bool) or not isinstance(coco_id, numbers.Integral):
                raise TypeError(
                    "global_infonce needs integer stimulus ids, got "
                    f"{coco_id!r}; run with data.group_by_coco=true."
                )
            coco_id = int(coco_id)
            if coco_id not in self.clip_index:
                raise KeyError(
                    f"no precomputed semantic target for coco_id={coco_id}; re-run "
                    f"scripts/precompute_clip_features.py"
                )
            rows.append(self.clip_index[coco_id])
            group_ids.append(coco_id)

        rows = torch.as_tensor(rows, dtype=torch.long, device=self.clip_bank.device)
        target_embeds = self.clip_bank[rows].to(device)
        group_ids = torch.as_tensor(group_ids, dtype=torch.long, device=device)
        return target_embeds, group_ids

    def on_save_checkpoint(self, checkpoint):
        new_state_dict = {}
        for name, param in checkpoint["state_dict"].items():
            if name.startswith("lm."):
                continue
            new_state_dict[name] = param
        checkpoint["state_dict"] = new_state_dict

        if self.peft_config is not None:
            peft_params = get_peft_model_state_dict(self.lm)
            checkpoint["peft_model"] = peft_params

        super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint):
        if "peft_model" in checkpoint:
            set_peft_model_state_dict(self.lm, checkpoint.pop("peft_model"))
        super().on_load_checkpoint(checkpoint)

    def forward(self, voxels, chat, image, return_fmri_tokens=False, **kwargs):
        loss = {}
        outputs = {}

        first_chat = kwargs["first_chat"]
        first_inputs_embeds = self.lm.get_input_embeddings()(
            first_chat["input_ids"]
        ).float()
        encoder_outputs = self.fmri_encoder(
            voxels,
            subject=kwargs["subject"],
            prompt_inputs_embeds=first_inputs_embeds,
            prompt_attention_mask=first_chat["attention_mask"],
            padding_side=self.tokenizer.padding_side,
        )
        if isinstance(encoder_outputs, dict):
            fmri_token_embeds = encoder_outputs["fmri_token_embeds"]
        elif isinstance(encoder_outputs, torch.Tensor):
            fmri_token_embeds = encoder_outputs
        else:
            raise

        caption_z = None
        if self.use_caption_contrastive and not return_fmri_tokens:
            caption_z = self.caption_head(fmri_token_embeds.float())

        fmri_token_embeds = self.projector(fmri_token_embeds)

        global_embeds = None
        if self.use_global_infonce and not return_fmri_tokens:
            global_embeds = self.global_head(fmri_token_embeds.float())

        # ---- baseline path: unchanged generation branch ----
        fmri_token_embeds = fmri_token_embeds.to(self.lm.dtype)

        if return_fmri_tokens:
            # caption inference never touches the auxiliary branch
            return fmri_token_embeds

        inputs_embeds, attention_mask, labels, position_ids, fmri_mask = (
            self.prepare_multimodal_inputs(
                chat["input_ids"],
                chat["attention_mask"],
                chat["labels"],
                fmri_token_embeds,
            )
        )

        lm_outputs = self.lm(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=not (self.use_caption_contrastive and self.training),
        )

        loss["token_loss"] = lm_outputs.loss

        if self.use_caption_contrastive:
            target_z, coco_ids = self._caption_targets(
                kwargs["sample_or_coco_id"], caption_z.device
            )
            aux_loss, top1, pos_sim, neg_sim = caption_contrastive_metrics(
                caption_z, target_z, coco_ids, self.caption_temperature
            )
            prefix = "train" if self.training else "val"
            self.log(f"{prefix}/contrastive_loss", aux_loss, on_epoch=True)
            self.log(f"{prefix}/contrastive_top1", top1, on_epoch=True)
            if self.training:
                weight = self._caption_weight()
                loss["caption_contrastive"] = weight * aux_loss
                self.log("train/lm_loss", lm_outputs.loss, on_epoch=True)
                self.log("train/contrastive_weight", weight, on_epoch=True)
                self.log("train/positive_similarity", pos_sim, on_epoch=True)
                self.log("train/negative_similarity", neg_sim, on_epoch=True)
            else:
                self.log("val/lm_loss", lm_outputs.loss, on_epoch=True)
                self.log("val/positive_similarity", pos_sim, on_epoch=True)
                self.log("val/negative_similarity", neg_sim, on_epoch=True)

        # ---- auxiliary branch: global semantic alignment ----
        if self.use_global_infonce:
            if self.training:
                target_embeds, group_ids = self.resolve_global_targets(
                    kwargs["sample_or_coco_id"], global_embeds.device
                )
                global_loss = multi_positive_infonce(
                    global_embeds,
                    target_embeds,
                    group_ids,
                    self.global_temperature,
                )
                loss["global_infonce"] = self.global_lambda * global_loss

                pos_sim, neg_sim = similarity_diagnostics(
                    global_embeds, target_embeds, group_ids
                )
                self.log("train/global_infonce_raw", global_loss)
                self.log("train/global_pos_sim", pos_sim)
                self.log("train/global_neg_sim", neg_sim)
                self._global_diag = (
                    global_loss.detach().item(),
                    pos_sim.item(),
                    neg_sim.item(),
                )
            else:
                outputs["global_embeds"] = global_embeds.detach()

        if self.training:
            return loss

        outputs["logits"] = lm_outputs.logits.detach().clone()
        outputs["labels"] = labels
        return loss, outputs

    @torch.no_grad()
    def generate(self, **batch):
        fmri_token_embeds = self.forward(**batch, return_fmri_tokens=True)
        chat = batch["prompt"]
        inputs_embeds, attention_mask, _, position_ids, _ = (
            self.prepare_multimodal_inputs(
                chat["input_ids"], chat["attention_mask"], None, fmri_token_embeds
            )
        )
        generated_tokens = []
        for i in range(batch["voxels"].size(0)):
            generated_tokens.append(
                self.lm.generate(
                    input_ids=None,
                    inputs_embeds=inputs_embeds[i : i + 1],
                    attention_mask=attention_mask[i : i + 1],
                    max_new_tokens=128,
                )
                .squeeze(0)
                .tolist()
            )
        return generated_tokens

    def training_step(self, batch, batch_idx):
        batch["voxels"] = adapt_voxels(batch["voxels"], True)
        losses = self.forward(**batch)
        loss = sum(losses.values())
        self.log("train/loss", loss, prog_bar=True)
        if self.use_caption_contrastive:
            self.log("train/total_loss", loss, on_epoch=True)

        for key in losses.keys():
            self.log(f"train/{key}", losses[key])

        # Guaranteed to land in the nohup log: with WANDB_MODE=disabled the
        # self.log() values above are not necessarily visible, and these are the
        # numbers used to pick lambda and to check the contrastive loss is real.
        if self.use_global_infonce and self.global_step % 200 == 0:
            raw, pos_sim, neg_sim = self._global_diag
            logging.info(
                f"[global_infonce] step={self.global_step} "
                f"lm={losses['token_loss'].item():.4f} "
                f"raw={raw:.4f} "
                f"lam_raw={losses['global_infonce'].item():.4f} "
                f"pos_cos={pos_sim:.4f} neg_cos={neg_sim:.4f}"
            )
        if self.use_caption_contrastive and self.global_step % 200 == 0:
            logging.info(
                f"[caption-contrastive] step={self.global_step} "
                f"mode={self.caption_mode} lm={losses['token_loss'].item():.4f} "
                f"weighted_aux={losses['caption_contrastive'].item():.4f} "
                f"total={loss.item():.4f} weight={self._caption_weight():.5f}"
            )

        return loss

    def validation_step(self, batch, batch_idx):
        batch['voxels'] = adapt_voxels(batch['voxels'], True)
        losses, predictions = self.forward(**batch)
        loss = sum(losses.values())
        self.log('val/loss', loss, prog_bar=True)

        self.perplexity(preds=predictions['logits'][:, :-1], target=predictions['labels'][:, 1:])
        self.log('val/ppl', self.perplexity)

        for key in losses.keys():
            self.log(f'val/{key}', losses[key])

        subject = safe_get_source(batch['subject'])
        source = safe_get_source(batch['source'])
        self.log(f'{source}/token_loss', losses['token_loss'])
        self.log(f'{subject}/token_loss', losses['token_loss'])

        if self.use_global_infonce:
            self._val_global_z.append(predictions['global_embeds'].float().cpu())
            self._val_global_ids.extend(int(x) for x in batch['sample_or_coco_id'])

    def on_validation_epoch_end(self):
        """Brain -> Image retrieval, the representation-level view of the ablation.

        Caption metrics alone can stay flat while the brain representation gets
        measurably better, so this is computed over the whole validation epoch
        against the frozen target bank.
        """
        if not self.use_global_infonce:
            return
        if self.trainer.sanity_checking:
            self._val_global_z.clear()
            self._val_global_ids.clear()
            return
        if len(self._val_global_z) == 0:
            return

        query_z = torch.cat(self._val_global_z, dim=0)
        rows = torch.as_tensor(
            [self.clip_index[coco_id] for coco_id in self._val_global_ids],
            dtype=torch.long,
            device=self.clip_bank.device,
        )
        target_z = self.clip_bank[rows].float().cpu()

        metrics = retrieval_metrics(query_z, target_z)
        for name, value in metrics.items():
            self.log(f"val/brain2image_{name}", value, on_step=False, on_epoch=True)

        logging.info(
            f"[retrieval] epoch={self.current_epoch} n={query_z.size(0)} "
            + " ".join(f"{name}={value:.4f}" for name, value in metrics.items())
        )

        self._val_global_z.clear()
        self._val_global_ids.clear()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        batch["voxels"] = adapt_voxels(batch["voxels"])
        assert batch["voxels"].size(0) == 1, "only batch_size = 1 is supported"
        # generated evalution
        generated_tokens_batch = self.generate(**batch)
        preds = self.tokenizer.batch_decode(
            generated_tokens_batch, skip_special_tokens=True
        )
        targets = batch.get("answers")
        assert isinstance(targets, list)
        assert (
            isinstance(targets[0], str)
            or isinstance(targets[0], list)
            or isinstance(targets[0], tuple)
        )
        source = safe_get_source(batch["source"])
        self.update_metrics(preds, targets, source)

    def on_test_epoch_end(self):
        self.log_metrics()
        self.reset_metrics()

    def on_predict_start(self):
        self.count = 0

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        voxels = adapt_voxels(batch["voxels"])
        assert voxels.size(0) == 1, "only batch_size = 1 is supported"
        # generated evalution
        generated_tokens_batch = self.generate(**batch)
        preds = self.tokenizer.batch_decode(
            generated_tokens_batch, skip_special_tokens=True
        )
        targets = batch["answers"]
        safe_get_source(batch["source"])

        return preds, targets, batch["sample_or_coco_id"]

    def prepare_multimodal_inputs(
        self, input_ids, attention_mask, labels, fmri_token_embeds, signal_token=None
    ):
        inputs_embeds = self.lm.get_input_embeddings()(input_ids)
        if True:
            if signal_token is None:
                image_token_index = self.tokenizer.convert_tokens_to_ids(FMRI_TOKEN)
            else:
                image_token_index = signal_token
            image_features = fmri_token_embeds
            num_images, num_image_patches, embed_dim = image_features.shape
            batch_size, sequence_length = input_ids.shape
            left_padding = self.tokenizer.padding_side == "left"
            # 1. Create a mask to know where special image tokens are
            special_image_token_mask = input_ids == image_token_index
            num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
            # Compute the maximum embed dimension
            max_embed_dim = (
                num_special_image_tokens.max() * (num_image_patches - 1)
            ) + sequence_length
            batch_indices, non_image_indices = torch.where(
                input_ids != image_token_index
            )

            # 2. Compute the positions where text should be written
            # Calculate new positions for text tokens in merged image-text sequence.
            # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
            # `torch.cumsum` computes how each image token shifts subsequent text token positions.
            # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
            new_token_positions = (
                torch.cumsum(
                    (special_image_token_mask * (num_image_patches - 1) + 1), -1
                )
                - 1
            )
            nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
            if left_padding:
                new_token_positions += nb_image_pad[:, None]  # offset for left padding
            text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

            # 3. Create the full embedding, already padded to the maximum position
            final_embedding = torch.zeros(
                batch_size,
                max_embed_dim,
                embed_dim,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )
            final_attention_mask = torch.zeros(
                batch_size,
                max_embed_dim,
                dtype=attention_mask.dtype,
                device=inputs_embeds.device,
            )
            if labels is not None:
                final_labels = torch.full(
                    (batch_size, max_embed_dim),
                    IGNORE_INDEX,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
            # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
            # set the corresponding tensors into their correct target device.
            target_device = inputs_embeds.device
            batch_indices, non_image_indices, text_to_overwrite = (
                batch_indices.to(target_device),
                non_image_indices.to(target_device),
                text_to_overwrite.to(target_device),
            )
            attention_mask = attention_mask.to(target_device)

            # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
            # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
            final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[
                batch_indices, non_image_indices
            ]
            final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[
                batch_indices, non_image_indices
            ]
            if labels is not None:
                final_labels[batch_indices, text_to_overwrite] = labels[
                    batch_indices, non_image_indices
                ]

            # 5. Fill the embeddings corresponding to the images. Anything that is not `text_positions` needs filling (#29835)
            image_to_overwrite = torch.full(
                (batch_size, max_embed_dim),
                True,
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
            image_to_overwrite[batch_indices, text_to_overwrite] = False
            image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[
                :, None
            ].to(target_device)

            if image_to_overwrite.sum() != image_features.shape[:-1].numel():
                raise ValueError(
                    f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                    f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
                )
            final_embedding[image_to_overwrite] = (
                image_features.contiguous().reshape(-1, embed_dim).to(target_device)
            )
            final_attention_mask |= image_to_overwrite
            position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_(
                (final_attention_mask == 0), 1
            )

            # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
            batch_indices, pad_indices = torch.where(
                input_ids == self.tokenizer.pad_token_id
            )
            indices_to_mask = new_token_positions[batch_indices, pad_indices]

            final_embedding[batch_indices, indices_to_mask] = 0

            if labels is None:
                final_labels = None

            return (
                final_embedding,
                final_attention_mask,
                final_labels,
                position_ids,
                image_to_overwrite,
            )

    def configure_optimizers(self):
        named_params_list = [
            self.fmri_encoder.named_parameters(),
            self.projector.named_parameters(),
        ]
        if self.global_head is not None:
            named_params_list.append(self.global_head.named_parameters())
        if self.caption_head is not None:
            named_params_list.append(self.caption_head.named_parameters())
        if self.peft_config is not None:
            named_params_list.append(self.lm.named_parameters())

        itertools.chain(*named_params_list)

        def get_params_iter(*iterables):
            for it in iterables:
                for x in it:
                    yield x[1]

        params_iter = get_params_iter(*named_params_list)
        optimizer = torch.optim.AdamW(params_iter, lr=self.learning_rate)

        if self.scheduler_type is not None:
            if self.scheduler_type == "cosine_annealing_restart":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=8000, T_mult=1
                )
            else:
                raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

        if self.scheduler_type is None:
            return optimizer
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",  # called after every training step
                    "frequency": 1,
                },
            }
