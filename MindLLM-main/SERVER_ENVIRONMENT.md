# Server environment configured on 2026-10-13

Project: `/mnt/data2/yjs2026/HLQ/MindLLM/MindLLM-main`.
Environment: `.venv-repaired`, Python 3.10.21.
The broken `.venv` was preserved as `.venv.before-env-fix-20261013-084020`.
`.venv` now links to `.venv-repaired`; the old `.venv-torch-probe` is untouched.

Installed versions include PyTorch 2.2.2+cu121, torchvision 0.17.2+cu121,
NumPy 1.24.4, Lightning 2.5.0.post0, torchmetrics 1.4.1, Transformers 4.44.2,
and PEFT 0.12.0. RTX 4090 is detected with driver 595.99.02.
Temurin Java 11.0.32.1+1 is available.

The server lacked Git. `nsd_access` was installed from the GitHub archive of
the exact commit pinned in requirements.txt:
`8d9f2ec4fed2516cdda1ad5ebd9fcd7eb99e7b4d`. requirements.txt was not modified.
The two Stanford CoreNLP 3.6.0 JARs needed by SPICE were restored from the
previous environment into the repaired environment.

## Verified

- `uv pip check`: all installed packages compatible.
- CPU 16-31: ten independent torch/torchvision/Lightning imports passed.
- CPU 16-31: five independent `main.py --help` invocations passed.
- CUDA: 100 mixed-precision forward/backward/AdamW steps passed.
- Project PTB tokenization and CIDEr, BLEU, ROUGE, METEOR, SPICE toy scoring passed.
- `outputs/coco-caption/checkpoints/epoch=48.ckpt` and
  `data/nsd0/subject_8.h5` exist. Their presence does not establish checkpoint
  training provenance or verify an end-to-end caption evaluation.

## Remaining runtime limitation

Unrestricted repeated imports produced intermittent segmentation faults.
A CPU 0-15 import trial failed on attempt 9. The tested CPU 16-31 configuration
passed the checks above. This observation does not establish the root cause,
repair CPU hardware, or guarantee long-run training stability.

Use the server-specific `run_mindllm_server.sh` launcher. It sets CPU affinity
to 16-31, limits OMP/MKL/OpenBLAS to one thread, selects `.venv-repaired`, and
passes arguments to `uv run --active --no-sync`. For example:

```bash
nohup env WANDB_MODE=disabled ./run_mindllm_server.sh python -u main.py \
  stage=test data.task=coco-caption 'data.subjects=[8]' \
  data.group_by_coco=true data.num_workers=0 lora=false resume=false \
  lm_checkpointing=off global_infonce.enabled=false contrastive.enabled=false \
  +trainer.precision=bf16-mixed output_dir=outputs/M0_S8_coco-caption_test \
  'checkpoint="outputs/coco-caption/checkpoints/epoch=48.ckpt"' \
  > M0_S8_coco-caption.log 2>&1 &
```

No full caption evaluation or training was launched during this environment
configuration task. Package versions and check metadata are also recorded on
the server in `.venv-repaired/installed-packages.txt` and
`.venv-repaired/environment-report.json`. No SSH password is stored in these
files or in the launcher.
