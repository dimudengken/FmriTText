# Caption dependence on stimulus-matched brain input

Evaluate M0 and the supplied M1/M2/M3 final caption checkpoints. This evaluates the
existing models without changing their weights or training objectives.

Run `python -m scripts.evaluate_neural_dependence` from the repository root.
Supply repeated `--checkpoint M0=PATH` arguments for M0 and at least one of
M1/M2/M3. M0/M1/M2 is supported without an M3 checkpoint.
Use complete caption checkpoints including encoder and projector, not Stage 1
encoder exports. Frozen-LLM checkpoints are required (`lora=false`).

## Fixed protocol

- Subject 1 and all shared1000 images by default; `--subjects 1,2,...` is optional
  only if each checkpoint supports those subjects and the training split matches.
- One query per subject/image; average test fMRI repeats before intervention.
- Real: correct averaged voxel input. Zero: all voxels zero, same shape.
- Shuffle: a reproducible bijection within each subject with no fixed points.
  Image identity, prompt, first-chat text, and reference captions stay attached
  to the recipient. Only the voxel input comes from the donor.
- Same mapping in all supplied models; same greedy decoding, one beam, maximum
  128 new tokens. Real is recomputed under this protocol rather than borrowed
  from earlier test logs.
- Default one Shuffle seed (42); use `--shuffle-seeds 42,43,44` to assess
  sensitivity to the permutation. Report every seed and their mean.
- CIDEr, BLEU-4 and ROUGE-L use the project's PTB tokenization and COCO scorers.
  CIDEr raw and multiplied by 100 are both reported (0.668 corresponds to 66.8).
- Paired subject-stratified bootstrap intervals for Real minus each control,
  plus the difference of those effects from M0. CIDEr reference IDF remains
  fixed; intervals are conditional on checkpoints and permutation seeds.

## Outputs and restart

`manifest.json` records checkpoint hashes, dataset/prompt/voxel fingerprints,
ordered IDs, generation settings and complete donor mappings. Per-condition
JSON files save predictions every 50 images, and on Python exceptions.
Rerunning the identical command resumes generation. Different settings or
weights require a new output directory. Scoring uses the complete matched
sample set and refuses partial or conflicting cached results.

`dependence.csv` gives Real/Zero/Shuffle-mean and both deltas in CIDEr x100.
`summary.json` includes paired intervals and differences from M0. `metrics.csv`
contains all conditions and individual Shuffle seeds. `per_image_cider.npz`
retains individual CIDEr contributions in manifest order.

## Evidence limits

Shared1000 did not enter encoder/caption training, but it previously influenced
early stopping and checkpoint selection. These are exploratory diagnostics.
A Zero drop can reflect distribution shift. A robust Shuffle drop supports
dependence on matched brain input, but does not by itself establish that the
additional brain information is correct or useful. Check the absolute Real
score as well as the deltas; compare deltas to M0 with their paired uncertainty.

No end-to-end run has been verified in the local Windows workspace, which lacks
Python, NSD data and the required caption checkpoints. Protocol helper tests can be
run on the training server with `python -m unittest discover -s tests -p
'test_neural_dependence_protocol.py'`.
