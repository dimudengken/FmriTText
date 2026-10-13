from lightning import seed_everything
import os
import hydra
from hydra.utils import instantiate
from peft import LoraConfig, TaskType

from src.dataset import create_tokenizer
from src.models.mindllm import MindLLM

import logging

logging.basicConfig(level=logging.INFO)


@hydra.main(config_path="config", config_name="default", version_base=None)
def main(cfg):
    seed_everything(cfg.seed)

    if cfg.contrastive.enabled:
        if cfg.data.task != "coco-caption" or not cfg.data.group_by_coco:
            raise ValueError("M4/M5 require data.task=coco-caption and data.group_by_coco=true")
        if cfg.checkpoint is None:
            raise ValueError("M4/M5 must initialize from the original mindllm-base.ckpt")
        logging.info("[caption-contrastive] initialization checkpoint=%s", cfg.checkpoint)

    if cfg.early_stop:
        cfg.trainer.callbacks.append(
            {
                "_target_": "lightning.pytorch.callbacks.EarlyStopping",
                "monitor": "val/token_loss",
                "patience": 10,
            }
        )
    trainer = instantiate(cfg.trainer)

    tokenizer = create_tokenizer(cfg.model_id)
    data_module = instantiate(cfg.data, tokenizer=tokenizer)

    if cfg.lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=32,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
    else:
        peft_config = None

    model_kwargs = {
        "encoder": cfg.encoder,
        "model_id": cfg.model_id,
        "lm_checkpointing": cfg.lm_checkpointing,
        "tokenizer": tokenizer,
        "peft_config": peft_config,
        "learning_rate": cfg.lr,
        "global_infonce": cfg.get("global_infonce", None),
        "contrastive": cfg.get("contrastive", None),
    }

    if cfg.checkpoint is None:
        model = MindLLM(**model_kwargs)
    else:
        model = MindLLM.load_from_checkpoint(
            cfg.checkpoint, strict=False, **model_kwargs
        )

    model.strict_loading = False
    if cfg.stage == "fit":
        last_ckpt = os.path.join(cfg.ckpt_dir, "last.ckpt")
        resume_path = last_ckpt if (cfg.resume and os.path.isfile(last_ckpt)) else None
        logging.info(
            f"[resume] found {resume_path}, resuming"
            if resume_path
            else f"[resume] no {last_ckpt}, training from scratch"
        )
        trainer.fit(model=model, datamodule=data_module, ckpt_path=resume_path)
        if cfg.test_after_fit:
            trainer.test(model=model, datamodule=data_module, ckpt_path="best")
    elif cfg.stage == "validate":
        trainer.validate(model=model, datamodule=data_module)
    elif cfg.stage == "test":
        trainer.test(model=model, datamodule=data_module)
    elif cfg.stage == "predict":
        trainer.predict(
            model=model,
            datamodule=data_module,
        )
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    main()
