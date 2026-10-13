import logging
import random

import numpy as np
import torch
from lightning import Trainer, LightningModule
from lightning.pytorch.callbacks import Callback


class WatchModel(Callback):
    def __init__(self, log_freq: int = 100):
        super().__init__()
        self.log_freq = log_freq

    def on_sanity_check_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_fit_start(trainer, pl_module)
        trainer.logger.watch(pl_module, log="all", log_freq=self.log_freq)


class RNGState(Callback):
    """Carry the process-global RNG state across a checkpoint so a resumed run continues the
    same shuffle instead of replaying epoch 0's."""

    def __init__(self):
        super().__init__()
        self._pending = None

    def on_save_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict
    ) -> None:
        checkpoint["rng_state"] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }

    def on_load_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict
    ) -> None:
        self._pending = checkpoint.pop("rng_state", None)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        state, self._pending = self._pending, None
        if state is None:
            return
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if state["cuda"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
        logging.info("[resume] restored RNG state from checkpoint")
