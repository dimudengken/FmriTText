"""Merge Stage 1's contrastive encoder into a copy of MindLLM's baseline checkpoint.

The comparison this serves is single-variable by construction: Stage 2 runs the
baseline command with a different `checkpoint=` file and nothing else moves, so
the only difference between the two arms is the fMRI encoder's initialization.

Only the encoder is swapped. Stage 1's own projector is a CLIP-alignment head
(1024 -> 1024 -> 768) while MindLLM's projector is a single `nn.Linear(token_dim,
lm_hidden)` (1024 -> 4096), so there is no weight to carry across; the baseline's
projector is kept on purpose, because replacing it with a random one would add a
second variable. Stage 1's `BrainContrastiveProjector` is discarded entirely.

Every non-encoder entry is copied through bit-for-bit, including the training
state (`optimizer_states` / `epoch` / `global_step`). Those are inert here rather
than stripped: `main.py` reads `checkpoint=` through `load_from_checkpoint` for
initialization only, while the resume path is `ckpt_dir/last.ckpt`, which is a
different file entirely. Keeping them is also what makes "no non-encoder entry
changed" a checkable claim.
"""

import argparse

import torch

# Keys that would mean the encoder swap is not the only difference, or that Stage 1
# leaked its own auxiliary modules into the composite.
FORBIDDEN = ("global_head.", "contrastive_projector.", "log_temperature", "clip_bank")
REPLACEABLE = {"encoder": "fmri_encoder.", "projector": "projector."}
TRAINING_STATE = ("optimizer_states", "lr_schedulers", "loops", "epoch", "global_step")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base", required=True, help="the baseline MindLLM checkpoint")
    parser.add_argument("--stage1", required=True, help="Stage 1's encoder.pt")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--base-prefix",
        default="fmri_encoder.",
        help="prefix of the encoder tensors inside --base",
    )
    parser.add_argument(
        "--stage1-key",
        default="fmri_encoder",
        help="top-level key in --stage1 whose value holds the encoder state dict",
    )
    parser.add_argument(
        "--stage1-prefix",
        default="",
        help="prefix to strip from the keys inside that state dict",
    )
    parser.add_argument(
        "--replace",
        default="encoder",
        help=f"comma-separated subset of {sorted(REPLACEABLE)}; only 'encoder' can "
        f"actually be satisfied, since Stage 1 produces no LLM-side projector",
    )
    return parser.parse_args()


def stage1_pool(obj, key, prefix):
    """Bare-keyed encoder tensors out of Stage 1's artifact."""
    if key not in obj:
        raise ValueError(
            f"--stage1-key {key!r} not found; the file holds {sorted(obj)[:8]}"
        )
    pool = obj[key]
    if not isinstance(pool, dict):
        raise ValueError(f"--stage1-key {key!r} holds {type(pool).__name__}, not a state dict")
    if not prefix:
        return dict(pool)
    return {
        key[len(prefix):]: value for key, value in pool.items() if key.startswith(prefix)
    }


def main():
    args = parse_args()
    targets = [t.strip() for t in args.replace.split(",") if t.strip()]
    unknown = set(targets) - set(REPLACEABLE)
    if unknown:
        raise ValueError(f"--replace got {sorted(unknown)}, known: {sorted(REPLACEABLE)}")

    base = torch.load(args.base, map_location="cpu")
    base_state = base["state_dict"]
    stage1 = torch.load(args.stage1, map_location="cpu")
    pool = stage1_pool(stage1, args.stage1_key, args.stage1_prefix)

    merged = dict(base_state)
    per_target = {}
    for target in targets:
        base_prefix = args.base_prefix if target == "encoder" else REPLACEABLE[target]
        incoming = {
            f"{base_prefix}{key}": value
            for key, value in pool.items()
            if f"{base_prefix}{key}" in base_state
        }
        if not incoming:
            raise ValueError(
                f"--replace {target}: Stage 1 holds no {base_prefix}* tensors "
                f"(its {len(pool)} keys look like {sorted(pool)[:3]})"
            )

        existing = {k for k in base_state if k.startswith(base_prefix)}
        missing = sorted(existing - set(incoming))
        extra = sorted(set(incoming) - existing)
        if missing or extra:
            raise ValueError(
                f"--replace {target}: key sets differ -- missing_from_stage1={missing[:5]} "
                f"unexpected_in_stage1={extra[:5]}"
            )

        bad_shape = [k for k in incoming if tuple(incoming[k].shape) != tuple(base_state[k].shape)]
        if bad_shape:
            raise ValueError(
                f"--replace {target}: shape mismatch on {bad_shape[:3]} "
                f"({tuple(incoming[bad_shape[0]].shape)} vs "
                f"{tuple(base_state[bad_shape[0]].shape)})"
            )
        bad_dtype = [k for k in incoming if incoming[k].dtype != base_state[k].dtype]
        if bad_dtype:
            raise ValueError(
                f"--replace {target}: dtype mismatch on {bad_dtype[:3]} "
                f"({incoming[bad_dtype[0]].dtype} vs {base_state[bad_dtype[0]].dtype})"
            )

        merged.update(incoming)
        per_target[target] = list(incoming)
        print(f"[merge] replaced {len(incoming)} {base_prefix}* tensors from Stage 1")

    # Nothing from Stage 1 but the encoder may arrive, whether via the swap or because
    # --base was not a baseline checkpoint.
    replaced = {k for keys in per_target.values() for k in keys}
    for key in merged:
        if any(key.startswith(bad) or key == bad.rstrip(".") for bad in FORBIDDEN):
            raise ValueError(f"{key} must not appear in the composite checkpoint")

    # Every non-replaced entry has to be the same object, not merely equal: a copied
    # tensor would still compare equal and hide an accidental mutation.
    same_object = sum(1 for k, v in merged.items() if k not in replaced and v is base_state[k])
    if same_object != len(merged) - len(replaced):
        raise ValueError(
            f"{len(merged) - len(replaced) - same_object} non-encoder entries are copies "
            f"rather than the original tensors"
        )

    changed = [k for k in replaced if not torch.equal(merged[k], base_state[k])]
    l1 = sum((merged[k].float() - base_state[k].float()).abs().sum().item() for k in replaced)
    if not changed:
        raise ValueError("no encoder tensor changed; the prefix mapping is probably wrong")

    base["state_dict"] = merged
    torch.save(base, args.out)

    print(f"[merge] unchanged non-encoder entries: {len(merged) - len(replaced)} "
          f"(all identical objects)")
    print(f"[merge] missing keys: 0, shape mismatches: 0, dtype mismatches: 0")
    print(f"[merge] encoder changed keys: {len(changed)}/{len(replaced)} "
          f"L1(merged - base) = {l1:.4f}")
    print(f"[merge] encoder checksum base={sum(base_state[k].float().sum().item() for k in replaced):.4f} "
          f"merged={sum(merged[k].float().sum().item() for k in replaced):.4f}")
    print(f"[merge] stage1 epoch={stage1.get('epoch')} "
          f"val_brain2image_r@1={stage1.get('val_brain2image_r@1')}")
    print(f"[merge] training state present but inert ({', '.join(k for k in TRAINING_STATE if k in base)}): "
          f"main.py reads checkpoint= via load_from_checkpoint (init) and resumes from "
          f"ckpt_dir/last.ckpt, a different file")
    print(f"[merge] wrote {args.out}")


if __name__ == "__main__":
    main()
