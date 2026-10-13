"""Train the standalone brain<->image contrastive stage.

No LLM, no tokenizer, no instruction data: this stage only needs voxels and the
stimulus id that keys the frozen target. Mirrors main.py closely enough to keep
the same resume/validate conventions.
"""

import logging
import os

import hydra
from hydra.utils import instantiate
from lightning import seed_everything

from src.models.brain_contrastive import BrainContrastive

logging.basicConfig(level=logging.INFO)


@hydra.main(config_path="config", config_name="contrastive", version_base=None)
def train_contrastive(cfg):
    seed_everything(cfg.seed)
    contrastive = cfg.contrastive

    model_kwargs = {
        key: value
        for key, value in contrastive.items()
        if key not in ("data", "trainer", "ckpt_dir")
    }
    model = BrainContrastive(encoder=cfg.encoder, **model_kwargs)
    data_module = instantiate(
        contrastive.data, audit_test_ids=True
    ) if contrastive.mode != "image" else instantiate(contrastive.data)
    if contrastive.mode == "text":
        contrastive.trainer.callbacks[0].monitor = "val/brain2text_r@1"
    if contrastive.mode != "image":
        encoder_params = sum(p.numel() for p in model.fmri_encoder.parameters() if p.requires_grad)
        head_params = sum(p.numel() for p in model.contrastive_projector.parameters() if p.requires_grad)
        temp_params = int(model.learnable_temperature)
        logging.info(
            f"[contrastive-pretrain] mode={contrastive.mode} trainable: "
            f"fmri_encoder={encoder_params} global_head={head_params} "
            f"temperature={temp_params}; frozen: llm=not instantiated, "
            f"clip=offline banks; targets={','.join(model._modalities())}; "
            f"target_bank_on_cpu={model.target_bank_on_cpu}; "
            f"weights=image:{model.image_weight},text:{model.text_weight}; "
            f"reduction={model.loss_reduction}"
        )
    trainer = instantiate(contrastive.trainer)

    # The untrained encoder's retrieval number is what this stage has to beat, and
    # it doubles as the end-to-end check that the target bank and the dataset ids
    # line up (a missing id raises instead of scoring silently).
    trainer.validate(model=model, datamodule=data_module)

    last_ckpt = os.path.join(contrastive.ckpt_dir, "last.ckpt")
    resume_path = last_ckpt if (cfg.resume and os.path.isfile(last_ckpt)) else None
    logging.info(
        f"[resume] found {resume_path}, resuming"
        if resume_path
        else f"[resume] no {last_ckpt}, training from scratch"
    )
    trainer.fit(model=model, datamodule=data_module, ckpt_path=resume_path)


if __name__ == "__main__":
    train_contrastive()
