"""Audit the functional cluster bank, then run the original MindLLM entrypoint."""

from contextlib import ExitStack
import logging
from pathlib import Path

import h5py
import hydra
import torch

from src.voxel_clustering import audit_cluster_split


@hydra.main(config_path="config", config_name="clustered", version_base=None)
def train_clustered(cfg):
    if cfg.data.nsd_version != 0:
        raise ValueError("cluster split auditing currently supports data.nsd_version=0")
    if cfg.data.train_samples != -1:
        raise ValueError("cluster bank auditing requires data.train_samples=-1")
    if cfg.encoder.get("cluster_path") is None:
        raise ValueError("use the clustered encoder with a training-only cluster bank")
    bank = torch.load(cfg.encoder.cluster_path, map_location="cpu", weights_only=True)
    with ExitStack() as stack:
        handles = {
            f"subject_{s}": stack.enter_context(h5py.File(Path("data/nsd0") / f"subject_{s}.h5", "r"))
            for s in cfg.data.subjects
        }
        audit = audit_cluster_split(bank, handles, 900 if cfg.data.split_val else 0, cfg.data.split_seed)
    logging.info("[cluster-bank] split audit passed: %s", audit)
    # Import only after auditing, so a bad bank fails before tokenizer/LLM loading.
    from main import main as train_mindllm
    train_mindllm(cfg)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_clustered()
