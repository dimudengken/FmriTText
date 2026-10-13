"""Compare encoder/projector output statistics of two MindLLM checkpoints.

Stage 1 changes the encoder's outputs, and MindLLM's projector was trained against
the baseline encoder's outputs. That mismatch is part of the experiment rather than
a confound, but a scale blow-up would make Stage 2 look broken for a reason that has
nothing to do with the representation, so this records the numbers beforehand.

Both sides are built straight from the checkpoints (`fmri_encoder.*` plus a plain
`nn.Linear` for `projector.*`), because the comparison needs neither the language
model nor the tokenizer, and loading Vicuna just to read an encoder's activations
would cost minutes per checkpoint.

The forward is run the way `BrainContrastive._embed` runs it: real voxels at their
native width, one subject per call.
"""

import argparse
import os
import sys

import h5py
import numpy as np
import torch
import yaml
from hydra.utils import instantiate

# Running a file inside `scripts/` makes that directory `sys.path[0]`, so the repo root
# holding `src/` is not importable and Hydra cannot resolve `_target_:
# src.modules.NeuroscienceInformedAttention`. Being in the repo root is not enough.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--h5", default="data/nsd0/subject_1.h5")
    parser.add_argument("--subject", default="subject_1")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--encoder-config", default="config/encoder/neuro_informed_attn.yaml")
    return parser.parse_args()


def stats(name, tensor):
    flat = tensor.float()
    finite = torch.isfinite(flat).float().mean().item()
    per_row_norm = flat.flatten(1).norm(dim=1)
    row = {
        "mean": flat.mean().item(),
        "std": flat.std().item(),
        "absmax": flat.abs().max().item(),
        "row_norm": per_row_norm.mean().item(),
        "finite": finite,
    }
    print(
        f"    {name:<14} mean={row['mean']:+.4f} std={row['std']:.4f} "
        f"absmax={row['absmax']:.3f} row_norm={row['row_norm']:.4f} "
        f"finite={row['finite']:.4f}"
    )
    return row


def main():
    args = parse_args()

    with h5py.File(args.h5, "r") as handle:
        keys = list(handle["train"])[: args.n]
        voxels = np.stack([handle["train"][k]["visual_voxels"][()] for k in keys])
    voxels = torch.as_tensor(voxels, dtype=torch.float32)
    print(f"[init-stats] {voxels.shape[0]} samples of {voxels.shape[1]} voxels from {args.h5}")

    encoder_cfg = yaml.safe_load(open(args.encoder_config))
    collected = []
    for path in args.checkpoint:
        state = torch.load(path, map_location="cpu")["state_dict"]
        encoder = instantiate(encoder_cfg)
        encoder.load_state_dict(
            {k[len("fmri_encoder."):]: v for k, v in state.items() if k.startswith("fmri_encoder.")},
            strict=True,
        )
        encoder.eval()
        projector = torch.nn.Linear(
            state["projector.weight"].shape[1], state["projector.weight"].shape[0]
        )
        projector.load_state_dict(
            {"weight": state["projector.weight"], "bias": state["projector.bias"]}
        )

        print(f"[init-stats] {path}")
        with torch.no_grad():
            tokens = encoder(voxels, subject=[args.subject] * voxels.size(0))
            projected = projector(tokens)
        collected.append((path, stats("encoder out", tokens), stats("projector out", projected)))

    if len(collected) == 2:
        (path_a, enc_a, proj_a), (path_b, enc_b, proj_b) = collected
        print(f"[init-stats] ratios {path_b} / {path_a} (1.0 means the same scale)")
        for label, a, b in (("encoder out", enc_a, enc_b), ("projector out", proj_a, proj_b)):
            print(
                f"    {label:<14} std={b['std'] / a['std']:.4f} "
                f"row_norm={b['row_norm'] / a['row_norm']:.4f} "
                f"absmax={b['absmax'] / a['absmax']:.4f} finite={b['finite']:.4f}"
            )


if __name__ == "__main__":
    main()
