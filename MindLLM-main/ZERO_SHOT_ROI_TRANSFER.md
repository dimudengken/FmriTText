# Source functional clusters transferred through shared ROI and spatial metadata

Authorized experiment: S2/S5/S7 are source subjects, S1 is an unseen target.
COCO captioning, seed 42, baseline / functional / size-matched random controls.
This is a single-target exploratory experiment, not a general cross-subject claim.

## Data and zero-shot boundary

Only source training images and fMRI responses fit CLIP-to-voxel ridge coefficients.
Source training voxel vectors fit PCA64 then GMM128 (diagonal covariance, reg=1e-3,
two initializations, max_iter=200). GMM must converge before model training.
All source validation image IDs and all source/target test image IDs are excluded from
source training and functional fitting, including cross-subject occurrences.
Target test image identifiers are used only for split exclusion. No target training
or validation samples, paired images, response arrays or answers fit cluster assignments.

Target cluster labels are inferred from target anatomical metadata only. Shared HCP-MMP1
parcel and hemisphere are matched first; Kastner/pRF parcels then hemisphere-only are
fallbacks. Within each matched parcel, center/scale native RAS coordinates separately
for each subject; vote from five nearest source voxels, weighting inverse distance,
normalizing each source's total vote equally. Record fallback rates, distances,
vote confidence and cluster occupancy for EVERY target voxel.
Native-volume midplanes separate hemispheres. This is approximate ROI-local spatial
correspondence, not surface registration, verified homology or MNI alignment.
No voxel-index copying and no nearest-neighbor matching across hemispheres.
Vote confidence measures agreement, not a calibrated probability of functional identity.

## Models and inference

The frozen Vicuna model is shared. Encoder and projector are initialized from scratch
with the same seed in all three arms: no official brain checkpoint that might contain S1
training. Encoder construction accesses the same anatomical buffers as baseline.
Functional bank includes source fitted assignments and transferred target assignments,
so target inference can load the source-trained checkpoint without adding trainable parameters.
Random control independently permutes each subject's labels, preserving per-subject sizes;
it tests the combined functional grouping/correspondence against arbitrary organization,
not an isolated comparison of two registration algorithms.

Source-only train/validation data module never constructs a target training loader.
Target inference uses only its fixed test set; zero gradients/updates/target early stopping.
Train original MindLLM, GMM-transfer clustered encoder, and random clustered control with
lr=1e-4, batch=4, accumulation=8, bf16, no LoRA or auxiliary losses, 128x1024 brain tokens.
Each arm has max 15,000 optimizer steps, max 60 source epochs, source validation patience=5;
actual steps, early stopping, GPU time and memory must be reported. Same maximum budget
is not an assertion of equal wall-clock cost. A 48-hour total active run limit applies.
First run three 8-batch source CUDA smoke tests, then train three arms sequentially;
only after source checkpoint selection evaluate S1 real/zero/shuffled fMRI (1000 stimuli).
Report CIDEr x100, BLEU4, ROUGE-L and paired image bootstrap conditional on this seed.
No target test-driven hyperparameter tuning and no performance promise.

## Entry point and artifacts

`scripts/run_zero_shot_roi_transfer.py --mode campaign --output outputs/zero_shot_roi_s257_to_s1_seed42`

Use the repaired server environment, CPU affinity 16-31, thread counts 1, explicit JAVA_HOME.
Feature bank must cover source images; use the existing S1-S7 CLIP bank, not the S1-only bank.
Deploy as an independent Git release with data/ckpts/.venv-repaired links; preserve paused
adaptive-readout work and all previous clustering runs.

Artifacts: protocol.json (source hashes, teacher-bank hash and source-only initialization),
manifest.json (image-level splits), functional.pt/random.pt, transfer_report.json,
transfer_diagnostics.npz, source training configs/CSV/checkpoints, target predictions,
per-image scores, summary.json, metrics.csv and RESULTS.md.
Process monitoring checks time budget, stage progress and child identity. Known transient
process/CUDA failures may retry twice with the same settings and existing checkpoint;
configuration errors, unconverged GMM and repeated failures stop for attention.
`--mode status --ensure` writes progress.log and can restore a missing supervisor without
starting duplicate workers. A frozen source-file check precedes each phase.

Tests cover parcel/hemisphere matching, fallback behavior, source-label dependence,
global image leakage exclusion and an instrumented target handle that forbids train access.
The first real smoke probe must verify checkpoint loading and target encoder shapes before
any claim of end-to-end zero-shot inference. Multiple target subjects/seeds are future work.
