"""Run a controlled downstream readout ablation through the original entrypoint."""

import hashlib
import json
import logging
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import torch

PROJECT = Path(__file__).resolve().parent
MODES = {"baseline": None, "adaptive": 1, "mlp_control": 2}
SOURCE_FILES = (
    "train_readout.py", "main.py", "src/modules/adaptive_neuro_informed_attn.py",
    "src/modules/neuro_informed_attn.py", "src/models/mindllm.py",
    "src/dataset.py", "src/dataset_cluster_ablation.py",
    "config/default.yaml", "config/readout_baseline.yaml",
    "config/readout_adaptive.yaml", "config/readout_mlp_control.yaml",
    "config/encoder/neuro_informed_attn.yaml",
    "config/encoder/adaptive_neuro_informed_attn.yaml",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_config(cfg):
    if cfg.readout_arm not in MODES:
        raise ValueError("unknown readout_arm")
    if cfg.readout_arm == "baseline":
        if cfg.encoder._target_ != "src.modules.NeuroscienceInformedAttention":
            raise ValueError("baseline arm must use the original encoder")
    elif (cfg.encoder._target_ !=
          "src.modules.adaptive_neuro_informed_attn.AdaptiveNeuroscienceInformedAttention"
          or cfg.encoder.mode != cfg.readout_arm):
        raise ValueError("readout_arm and encoder mode must agree")
    if cfg.lora or cfg.global_infonce.enabled or cfg.contrastive.enabled:
        raise ValueError("controlled readout runs require frozen LLM and no auxiliary losses")
    if (cfg.data._target_ != "src.dataset_cluster_ablation.ClusterAblationDataModule"
            or not cfg.data.split_val or cfg.data.nsd_version != 0):
        raise ValueError("use the audited NSD0 validation-from-training data module")
    if cfg.resume and cfg.stage != "fit":
        raise ValueError("resume=true applies only to stage=fit")
    if not cfg.checkpoint or not Path(cfg.checkpoint).is_file():
        raise ValueError("checkpoint must point to an existing initialization or trained checkpoint")
    # Hydra merges mappings; replace the logger to remove inherited Wandb keys.
    cfg.trainer.logger = OmegaConf.create({
        "_target_": "lightning.pytorch.loggers.CSVLogger",
        "save_dir": str(cfg.output_dir), "name": "csv",
    })
    return cfg


def audit_checkpoint(path, arm, allow_baseline):
    # These are user-owned Lightning checkpoints, including callback metadata.
    saved = torch.load(path, map_location="cpu", weights_only=False)
    state = saved.get("state_dict", {})
    prefix = "fmri_encoder.neuro_informed_attn."
    if prefix + "query_embeddings" not in state or "projector.weight" not in state:
        raise ValueError("expected a full MindLLM checkpoint with encoder and projector")
    marker = state.get(prefix + "variant_id")
    marker = None if marker is None else int(marker)
    expected = MODES[arm]
    if marker != expected and not (marker is None and allow_baseline):
        raise ValueError("checkpoint readout mode does not match the requested arm")
    if marker is None:
        # Clustered/other attention checkpoints are not the shared baseline.
        extra = [k for k in state if k.startswith(prefix) and k != prefix + "query_embeddings"]
        if extra:
            raise ValueError("initialization contains another attention experiment")
    else:
        names = ["gate", "context_norm.weight", "context_norm.bias"]
        names += (["context_projection.weight", "context_projection.bias", "query_factors", "key_factors"]
                  if marker == 1 else ["mlp_down.weight", "mlp_down.bias", "mlp_up.weight", "mlp_up.bias"])
        if any(prefix + name not in state for name in names):
            raise ValueError("readout checkpoint is incomplete")
    return {"path": str(Path(path).resolve()), "sha256": sha256(path), "variant_id": marker}


def training_identity(cfg):
    identity = OmegaConf.to_container(cfg, resolve=True)
    identity.pop("resume", None)
    return identity


def preflight(cfg):
    cfg = prepare_config(cfg)
    initial = audit_checkpoint(cfg.checkpoint, cfg.readout_arm, allow_baseline=cfg.stage == "fit")
    if cfg.stage != "fit":
        return cfg
    output = Path(cfg.output_dir)
    manifest_path = output / "readout_manifest.json"
    source_hashes = {name: sha256(PROJECT / name) for name in SOURCE_FILES}
    identity = training_identity(cfg)
    manifest = {"schema_version": 1, "arm": cfg.readout_arm, "initial_checkpoint": initial,
                "source_sha256": source_hashes, "config": identity,
                "status": "prepared", "performance_claim": "not evaluated"}
    if cfg.resume:
        last = Path(cfg.ckpt_dir) / "last.ckpt"
        if not manifest_path.is_file() or not last.is_file():
            raise ValueError("resume requires this experiment's manifest and last.ckpt")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for field in ("config", "source_sha256", "initial_checkpoint"):
            if previous[field] != manifest[field]:
                raise ValueError("resume provenance differs: " + field)
        audit_checkpoint(last, cfg.readout_arm, allow_baseline=False)
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("output_dir is not empty; choose a new directory or resume=true")
        output.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (output / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    return cfg


@hydra.main(config_path="config", config_name="readout_adaptive", version_base=None)
def main(cfg):
    cfg = preflight(cfg)
    logging.info("[readout] arm=%s checkpoint=%s output=%s", cfg.readout_arm, cfg.checkpoint, cfg.output_dir)
    from main import main as train_mindllm
    train_mindllm(cfg)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
