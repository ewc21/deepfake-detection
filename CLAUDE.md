# CLAUDE.md — project context for coding sessions

Read this first in every session. Keep it up to date: tick TODOs, log new decisions and
assumptions (with dates) as work progresses.

## Project goal

Build a clean, reproducible PyTorch pipeline for **binary deepfake detection** on the
DeepFakeFace (DFF) dataset, then study robustness (Phase 3–4) and generalisation (Phase 5).

### Research questions
1. How well do lightweight pretrained CNNs detect face swaps vs. fully AI-generated faces?
2. Does one combined detector match task-specific detectors?
3. How much do realistic image transformations (JPEG, resizing, cropping, blur, brightness,
   noise) degrade detection, and does augmentation training recover robustness, and at what clean-accuracy cost?
4. Do detectors generalise across manipulation types (swap ↔ text2img) and to an unseen
   type (SD inpainting)?

## Dataset

Source: Hugging Face `OpenRL/DeepFakeFace` (pinned commit `eb2a54fb…` in `src/data/download.py`).
Four archives, 30,000 JPGs each, all in the IMDB-WIKI layout `NN/<person_id>_<dob>_<year_taken>.jpg`.

| folder | content | label |
|---|---|---|
| `wiki/` | real celebrity faces from IMDB-WIKI | 0 (real) |
| `insight/` | face swaps made with InsightFace | 1 (fake) |
| `text2img/` | fully generated faces, Stable Diffusion v1.5 | 1 (fake) |
| `inpainting/` | faces edited with SD Inpainting | 1 (fake), eval only |

**Verified on the real data (2026-11-19)** — reports in `data/reports/`:
- 120,000 files, **0 corrupted**. Each fake folder has exactly the same 30,000 relative paths as
  `wiki/` → every fake maps 1:1 by name to a wiki image (0 unmatched).
- All 30,000 wiki filenames follow the IMDB-WIKI pattern; all `person_id`s are unique (no person
  has 2+ images), no conflicting birth dates, subfolder = last two digits of the ID.
- **Name ≠ pixels for face swaps.** The filename of an `insight/` image is the identity DONOR (A);
  the pixels (pose, background, body) come from a different TARGET wiki photo (B). pHash: insight
  vs. its own-named wiki image has median distance 32 (= unrelated), while 22,172/30,000 insight
  images are within distance 6 of some other wiki image (11,590 at distance 0). The remaining ~7.8k
  targets are presumably IMDB-WIKI photos not included in DFF. Verified visually.
  `inpainting/` images DO match their own-named source (median distance 4); `text2img/` images are
  unrelated to any wiki image (nearest usually 12–16).
- **Exact duplicates:** 165 groups / 344 files, all wiki+wiki (same photo under different person_ids).
- **Formats/sizes:** all fakes are 512×512 RGB JPEG (~35 KB). Wiki images have 9,202 distinct
  sizes (68–1300 px wide), median aspect 0.76, only 2.8% square, and include PNG (1,005), GIF (36),
  MPO (7), grayscale `L` (2,840) and RGBA (664) files — all with a `.jpg` extension. The fakes are
  the full wiki frame squashed to 512×512 (see "Geometry" below).

## Planned experiments

| phase | experiment | how |
|---|---|---|
| 2 | baselines: `faceswap`, `text2img`, `combined` | `python -m src.train --task <task>` (3 seeds) |
| 3 | robustness: JPEG, resize, crop, blur, brightness, noise × severities | register perturbations in `src/data/transforms.py`, evaluate with `--condition` |
| 4 | augmentation training | `data.train_perturbations` in config, retrain, compare clean vs. perturbed |
| 5 | cross-task + unseen type | `python -m src.evaluate --run <run> --task text2img|insight-task|inpainting` |
| 6 | poster figures, write-up | from committed metrics JSON/CSV |

## Project structure

```
configs/       default.yaml (all defaults), tasks.yaml (detectors), local.example.yaml
data/          COMMITTED metadata only — never images
  metadata/file_index.csv.gz   per-file sha256, size, format, dims, decode-ok, pHash
  splits/master.csv            one row per wiki source: identity, group, split, fake paths
  splits/tasks/<task>.csv      relpath,label,fake_type,source,split (relative paths)
  reports/                     scan_summary, matching, split_summary, identity_overlap, duplicates
src/
  device.py      ALL hardware logic (CUDA→TPU→MPS→CPU, AMP, XLA loader/optimizer step)
  paths.py       dataset-root resolution (CLI > DFF_ROOT > configs/local.yaml)
  config.py      YAML loading + --set overrides
  seed.py        seeds + deterministic settings
  run_dir.py     outputs/<experiment>/<timestamp>_s<seed>_<user>
  engine.py      device-agnostic train/predict loops, DataLoaders
  train.py       multi-seed training entry point
  evaluate.py    evaluate a checkpoint on any task/split/condition
  aggregate.py   mean ± std over seeds
  metrics.py     metrics @0.5, Youden threshold, mean/std
  plots.py       ROC, confusion matrix, training curves
  models/factory.py   torchvision backbones with 1-logit head
  data/  download, scan, match, split, dedup, verify, dataset, transforms, harmonize
tests/         pytest on synthetic data (no dataset needed)
notebooks/     exploration only
outputs/       run outputs (only small json/csv/png/yaml/log committed)
```

## Commands

```bash
# setup
pip install -r requirements.txt                 # (TPU: requirements-tpu.txt)
python -m src.data.download --dest dataset      # opt-in, skips extracted archives
cp configs/local.example.yaml configs/local.yaml  # set dataset_root (or DFF_ROOT)
python -m src.data.verify [--hashes]            # local copy vs committed manifests

# data pipeline (maintainers; outputs are committed)
python -m src.data.scan && python -m src.data.split && python -m src.data.dedup

# train / evaluate
python -m src.train --task faceswap|text2img|combined [--model resnet18] [--seeds 0 1 2] [--device auto] [--harmonize]
python -m src.evaluate --run outputs/<exp>/<run> [--task inpainting] [--condition "{name: jpeg, quality: 50}"]
python -m src.aggregate --experiment baseline_faceswap

# tests
python -m pytest -q
```

## Reproducibility rules

1. **Never commit**: dataset images, zips, `configs/local.yaml`, checkpoints (`*.pt`), caches.
   Checkpoints are shared through external storage (shared drive / HF Hub), with run folder names.
2. **No absolute paths in committed files.** Manifests store POSIX paths relative to the dataset root.
3. **Committed manifests are the source of truth.** Everyone trains/tests on the same images.
   Changing `split.*` in `configs/default.yaml` or re-running `src.data.split` changes them — only do this deliberately and note it here.
4. Run `python -m src.data.verify` after getting the data; `src.train` checks the task's files exist before training.
5. All seeds set (Python, NumPy, torch, CUDA, DataLoader workers via `seed_worker` + generator);
   deterministic algorithms with `warn_only=True`. Default: 3 seeds, report mean ± std (sample std, ddof=1).
6. Every run saves its resolved `config.yaml`, `device.json`, metrics, figures; output folders are
   unique per timestamp/seed/user so collaborators never overwrite each other.
7. Pinned dependencies (`requirements*.txt`). TPU uses torch 2.9 (torch_xla constraint) — note this when comparing.
8. Core logic in `src/`, not notebooks.

## Implementation decisions and assumptions

- **Labels**: real = 0, fake = 1; model outputs one logit, `sigmoid` = P(fake); BCEWithLogitsLoss.
- **Baseline**: EfficientNet-B0 (ImageNet weights), full fine-tuning, AdamW lr 1e-4, wd 0.01,
  batch 64, ≤10 epochs, early stopping on val AUROC (patience 3), best checkpoint by val AUROC.
- **Preprocessing**: resize to 224×224 (no crop, aspect ratio not preserved), ImageNet normalisation.
  **No augmentation** in baselines (not even flips).
- **Threshold**: headline metrics at fixed 0.5; val-optimal (Youden J) threshold logged as reference only.
- **Fake → source mapping**: same relative path (extension-insensitive) in the fake folder and `wiki/`.
  Unmatched or corrupted fakes would be excluded from all manifests and counted in `data/reports/matching.json`.
- **Identity**: `person_id` from the wiki filename. In DFF every wiki image has a distinct person_id,
  so identity grouping ≈ per-image grouping. Limitation: the same celebrity under a different ID
  cannot be detected from filenames.
- **Leakage groups** (union-find over wiki sources): same identity, OR any two images of ANY type
  within pHash distance ≤ 6 (each image stands for its by-name source). This ties every face swap's
  donor A to its target photo B. Threshold 6 chosen from the data: unrelated text2img faces reach
  ≤ 6 from a wiki image only 6/30,000 times. Result: 18,382 groups (7,285 singletons, 10,786 pairs,
  largest 15 sources). Whole groups are assigned to splits.
- **Master split**: 70/15/15 by image count over shuffled groups (seed `split.seed`). Every fake
  inherits the split of its source wiki image, for ALL fake types including inpainting → cross-task
  evaluation (Phase 5) is leakage-free and test sources are the same across tasks.
- **Balance**: each task uses sources that have all of its fake types; each source contributes one real
  and one fake image → exactly balanced per split. `combined`: sources of each split are partitioned
  50/50 (largest-remainder rounding) between insight and text2img, so a source never appears twice as a fake.
- **Duplicates**: exact via SHA-256, near via 64-bit pHash (band-bucketed exact search, distance ≤ 6).
  Reported overall, across splits, and per task in `data/reports/duplicates.json`.
  Current result: 0 exact and 0 near-duplicate pairs across splits, in every task. (With the first,
  wiki-only grouping there were 16,755 cross-split pairs, mostly insight↔wiki — see "Name ≠ pixels".)

### Split results (seed 20261119, committed in `data/splits/`)
- Master: train 21,001 / val 4,499 / test 4,500 source wiki images.
- Every task is exactly balanced per split (real = fake = master count). `combined` test:
  2,250 insight + 2,250 text2img fakes (train/val split the same way, ±1).
- Identity overlap across splits: 0 (trivially, as every person_id is unique); leakage groups across splits: 0.

### Geometry: fakes are SQUASHED full frames → keep plain resize (decided 2026-11-19)
DFF fakes were made from the FULL wiki photo resized (aspect-distorted) to 512×512 — not from a
centre crop. Mean abs pixel error between a fake and its source prepared three ways (n≈35 each):
insight vs. target photo: squash 3.3 | centre crop 37.5 | pad 56.3;
inpainting vs. own source: squash 11.8 | centre crop 37.6 | pad 52.5.
So the spec's plain resize to 224×224 distorts real and fake images identically. **Do NOT
centre-crop or pad**: that would make real images undistorted while fakes stay squashed, creating
a new geometric shortcut.

### Processing-history shortcut → `--harmonize` option (implemented 2026-11-19)
Fakes = full frame squashed to 512×512 + manipulation + JPEG; reals = untouched originals of many
sizes/formats (PNG/GIF/MPO). Even at 224×224, resampling scale and JPEG traces differ by class.
Measured fake encoding: 450/450 sampled fakes (150 per type) use exactly the IJG quality-75
tables with 4:2:0 subsampling (= Pillow's default `save`). Resize filter: re-creating face swaps
from their target photo, bicubic matches best (median abs error outside the face 1.07 vs.
bilinear 2.17, lanczos 1.89, nearest 3.69); inpainting can't discriminate (SD re-renders the
whole image). Wiki JPEGs vary (mostly q80; 4:4:4, 4:2:2 and 4:2:0 mixed).

`data.harmonize_real` (CLI `--harmonize`, `src/data/harmonize.py`): REAL images only get a
bicubic full-frame resize to 512×512 + in-memory JPEG q75 4:2:0 round trip, before the normal
transform, in training AND evaluation. Fakes are untouched (re-encoding them would add a second
compression only they have). Deterministic → not augmentation. OFF for the spec baselines;
default experiment name gets `_harmonized` when on. Plan: run each baseline both ways; the gap
measures how much the processing shortcut contributes.

### Still open
- Grayscale: 2,840 real images are grayscale (`L`); check how many fakes are also grayscale.
- Optional sanity baseline: tiny classifier on metadata only (size, format, grayscale).
- **Devices**: all in `src/device.py`. AMP: CUDA bf16 if supported else fp16 + GradScaler; TPU bf16;
  MPS/CPU fp32. TPU: `xm.xla_device()`, `MpDeviceLoader`, `xm.optimizer_step()`, training `drop_last=True`
  (eval batches may have a smaller last batch → one extra XLA compile).
- **Perturbations (Phase 3/4)**: registry in `src/data/transforms.py`; applied to the PIL image at
  native resolution BEFORE resizing; classes (not lambdas) so they pickle on Windows.
- **Cross-task naming**: evaluating the face-swap model "on text2img images" = evaluating on the
  `text2img` task's test split (same real test images, text2img fakes).

## TODO

## Phase 1 — Setup and Data
- [x] Write the opt-in download script and the local path config (configs/local.example.yaml).
- [x] Configure local paths to each extracted DeepFakeFace archive.
- [x] Write the dataset-vs-manifest consistency check for teammates.
- [x] Inspect image counts, file types, dimensions, and corrupted files.
- [x] Confirm correspondence between real and fake filenames/identities.
- [x] Create one reproducible master split manifest keyed by source wiki image.
- [x] Validate class balance and absence of duplicate images (exact + perceptual) across splits.
- [x] Report identity overlap across splits.
- [x] Decide geometry handling: keep plain resize (fakes are squashed full frames; no centre crop).
- [x] Add the `--harmonize` switch (real images get the fakes' 512×512 + JPEG q75 processing).
- [ ] Optional: grayscale check + metadata-only shortcut baseline (see "Still open").

## Phase 2 — Baseline Models
- [ ] Train face-swap detector: wiki vs. insight.
- [ ] Train AI-generated-image detector: wiki vs. text2img.
- [ ] Train combined detector: wiki vs. insight + text2img (fakes subsampled 50/50).
- [ ] Run each baseline with 3 seeds and report mean ± std.
- [ ] Re-run each baseline with `--harmonize` and compare (processing-shortcut check).
- [ ] Compare clean test performance across all models.
- [ ] Save checkpoints, metrics, ROC curves, and confusion matrices.

## Phase 3 — Robustness Evaluation
- [ ] Implement JPEG compression at several severity levels.
- [ ] Implement image resizing/downsampling.
- [ ] Implement center/random cropping.
- [ ] Implement Gaussian blur.
- [ ] Implement brightness adjustments.
- [ ] Implement Gaussian noise.
- [ ] Measure performance degradation by transformation and severity.
- [ ] Create accuracy/AUROC-versus-severity plots.

## Phase 4 — Augmentation Training
- [ ] Add realistic transformations to the training pipeline.
- [ ] Retrain baseline models with augmentation.
- [ ] Compare clean-image performance against baseline models.
- [ ] Compare transformed-image performance against baseline models.
- [ ] Document the clean-performance versus robustness tradeoff.

## Phase 5 — Generalization and Analysis
- [ ] Test face-swap detector on text2img images.
- [ ] Test AI-generated-image detector on insight face swaps.
- [ ] Test all models on inpainting images as an unseen manipulation type.
- [ ] Identify and save representative false positives and false negatives.
- [ ] Analyze whether mistakes correlate with transformations or fake type.

## Phase 6 — Final Deliverables
- [ ] Create final visualizations for the poster.
- [ ] Write results and limitations.
- [ ] Document dataset bias and scope: conclusions apply primarily to celebrity face images.
- [ ] Ensure all experiments are reproducible from saved configs and split manifests.

## Work log
- 2026-11-19: Project scaffolded (code, configs, tests). Unit tests + end-to-end CLI run pass on
  synthetic data (CPU, torch 2.14.1). Same-seed runs reproduce identical losses on CPU.
- 2026-11-19: Real-data Phase 1 done: scan (file index), split, dedup; reports committed. Found the
  face-swap donor/target issue and switched to cross-type pHash grouping (threshold 6) → 0 cross-split
  duplicates. Folder listing switched to os.walk (rglob was very slow on the mounted drive); scan
  cache recovers from interrupted writes. Training/eval smoke-tested on 44 real images incl.
  PNG/GIF/grayscale/RGBA wiki files (EfficientNet-B0 and ResNet-18, random init, CPU).
  Not yet tested: ImageNet weight download, CUDA/MPS/TPU code paths (no such hardware/network here).
- 2026-11-19: Decided to keep plain resize (fakes are squashed full frames; centre crop would add a
  shortcut). Measured the fakes' encoding and added the `--harmonize` switch (+3 tests, 19 total
  passing; harmonized smoke run on real images OK).
