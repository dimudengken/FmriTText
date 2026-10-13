"""Train a standalone cross-repeat semantic JEPA encoder, without an LLM."""

import logging
import os

import hydra
from hydra.utils import instantiate
from lightning import seed_everything

from src.models.cross_repeat_jepa import CrossRepeatSemanticJEPA

logging.basicConfig(level=logging.INFO)


@hydra.main(config_path="config", config_name="jepa", version_base=None)
def train_jepa(cfg):
    seed_everything(cfg.seed)
    model = CrossRepeatSemanticJEPA(encoder=cfg.encoder, **cfg.jepa)
    data = instantiate(cfg.data, audit_test_ids=True)
    trainer = instantiate(cfg.trainer)

    # Validate the untrained initialization and the CLIP bank before any update.
    trainer.validate(model=model, datamodule=data)

    last_ckpt = os.path.join(cfg.ckpt_dir, "last.ckpt")
    resume_path = last_ckpt if cfg.resume and os.path.isfile(last_ckpt) else None
    logging.info("[cross-repeat-jepa] resume checkpoint: %s", resume_path)
    trainer.fit(model=model, datamodule=data, ckpt_path=resume_path)


if __name__ == "__main__":
    train_jepa()
