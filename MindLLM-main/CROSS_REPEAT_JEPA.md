# Cross-Repeat Semantic JEPA

This experiment is an independent encoder pretraining stage. It does not load
Vicuna, alter the caption loss, or modify M0-M5. Its encoder artifact can be
merged into a copy of `mindllm-base.ckpt` for the original caption training.

## Definition

- A training item is two **different fMRI presentations** of the same COCO image
  within one subject. Stimuli with fewer than two presentations are excluded.
- The online fMRI encoder and semantic projector map the context repeat into
  the CLIP ViT-L/14 image embedding space. A predictor maps that online vector
  to the EMA encoder's representation of the other repeat.
- The target encoder and target projector are copied from the online modules at
  initialization, receive no gradients, run in evaluation mode, and update by
  EMA **after each optimizer step**.
- The objective is
  `jepa_weight * cosine_distance(predicted_repeat, EMA_target_repeat)
   + semantic_weight * cosine_distance(context, frozen_CLIP_image)`.
  Both distances are computed in float32. CLIP features are read from a
  precomputed image bank; CLIP itself is not loaded in this stage.
- Train/validation splitting is global across subjects by COCO ID, with a fixed
  seed. Test IDs are audited before training. Validation retrieval uses one CLIP
  image vector per unique COCO ID, even if multiple subjects saw the image.
- The best encoder is selected by `val/brain2image_r@1` and saved as
  `outputs/jepa_cross_repeat_s1-7/encoder.pt`. Full Lightning checkpoints,
  including the EMA teacher and optimizer state, are saved separately under
  `outputs/jepa_cross_repeat_s1-7/checkpoints/` for resuming.

The default is subject 1-7, `stimuli_per_batch=12`, bf16 mixed precision,
`ema_decay=0.996`, and equal loss weights. These are new experiment settings,
not claims about an established Cross-Repeat JEPA benchmark.

This global split and the exclusion of single-presentation stimuli differ from
the existing Stage 1 validation population. Compare downstream caption results
on the same test protocol; do not directly compare the Stage 1 validation R@1
numbers as if the candidate galleries were identical.

## Commands (after the Python import instability is resolved)

Run from the repository root. The following use the existing clean virtual
environment and save all output to logs.

```bash
nohup env WANDB_MODE=disabled VIRTUAL_ENV="$PWD/.venv-torch-probe" \
  uv run --active python -m unittest discover -s tests \
  -p test_cross_repeat_jepa.py \
  > jepa_tests.log 2>&1 &
```

```bash
nohup env WANDB_MODE=disabled PYTHONFAULTHANDLER=1 \
  VIRTUAL_ENV="$PWD/.venv-torch-probe" \
  uv run --active python train_jepa.py \
  > jepa_cross_repeat_s1-7.log 2>&1 &
```

On an interrupted run, `resume=true` restores
`outputs/jepa_cross_repeat_s1-7/checkpoints/last.ckpt` automatically. Use a
different `output_dir` and `resume=false` for a fresh run or an ablation.

After pretraining, merge **only** the exported fMRI encoder into a copy of the
baseline checkpoint:

```bash
nohup env WANDB_MODE=disabled VIRTUAL_ENV="$PWD/.venv-torch-probe" \
  uv run --active python scripts/merge_stage1_encoder_into_mindllm_ckpt.py \
  --base ckpts/MindLLM/mindllm-base.ckpt \
  --stage1 outputs/jepa_cross_repeat_s1-7/encoder.pt \
  --out outputs/jepa_cross_repeat_s1-7/mindllm_caption_init.ckpt \
  > jepa_merge.log 2>&1 &
```

The merged checkpoint is an initialization for the original `main.py` caption
stage. Its `checkpoint=` path is distinct from the caption stage's `last.ckpt`
resume path.
