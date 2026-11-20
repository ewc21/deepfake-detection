# Deepfake detection on DeepFakeFace

Reproducible PyTorch pipeline for binary deepfake detection on the
[DeepFakeFace](https://github.com/OpenRL-Lab/DeepFakeFace) dataset (real IMDB-WIKI celebrity
faces vs. InsightFace swaps, Stable Diffusion text-to-image faces and SD-inpainted faces).

Three baseline detectors, each just an entry in `configs/tasks.yaml`:

| task | real | fake |
|---|---|---|
| `faceswap` | wiki | insight |
| `text2img` | wiki | text2img |
| `combined` | wiki | insight + text2img (50/50, total = real count) |
| `inpainting` | wiki | inpainting (evaluation only, Phase 5) |

## 1. Setup (once per person)

```bash
git clone <this repo> && cd deepfake-detection
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate

# NVIDIA GPU: install the CUDA build of the pinned torch first (see requirements.txt header), e.g.
#   pip install torch==2.14.1 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/<cuXXX>
pip install -r requirements.txt          # CUDA / Apple MPS / CPU
# pip install -r requirements-tpu.txt    # Google TPU VM / Colab TPU instead
```

### Get the dataset (opt-in, ~4.3 GB download, ~9 GB with extraction)

The images are **never** committed. Each person keeps a local copy:

```bash
python -m src.data.download --dest dataset        # or any folder, e.g. D:/data/dff
```

It downloads `wiki.zip`, `insight.zip`, `text2img.zip`, `inpainting.zip` from the Hugging Face
dataset `OpenRL/DeepFakeFace` (pinned commit) and skips archives that are already extracted.
Nothing else in the code ever downloads data.

### Tell the code where your copy is

```bash
cp configs/local.example.yaml configs/local.yaml    # Windows: copy configs\local.example.yaml configs\local.yaml
# edit dataset_root in configs/local.yaml   (or set the env var DFF_ROOT=<path>)
```

Expected layout under `dataset_root`:

```
<dataset_root>/
  wiki/00/10049200_1891-09-16_1958.jpg   ... 30,000 real images (IMDB-WIKI layout)
  insight/00/10049200_1891-09-16_1958.jpg      30,000 face swaps   (same relative paths)
  text2img/...                                  30,000 SD v1.5 faces
  inpainting/...                                30,000 SD inpainting
```

### Check your copy matches the committed manifests

```bash
python -m src.data.verify            # file existence + counts per folder
python -m src.data.verify --hashes   # + SHA-256 of every file (a few minutes)
```

## 2. Train

```bash
python -m src.train --task faceswap                 # 3 seeds (0,1,2), EfficientNet-B0, auto device
python -m src.train --task text2img
python -m src.train --task combined

# variations
python -m src.train --task faceswap --model resnet18 --seeds 0
python -m src.train --task faceswap --device cpu --set train.epochs=1 data.subset_per_split=512   # quick test
python -m src.train --task faceswap --set train.lr=3e-4 train.batch_size=32 --experiment lr3e-4_faceswap

# harmonized: real images get the fakes' processing first (see below)
python -m src.train --task faceswap --harmonize     # -> outputs/baseline_faceswap_harmonized/
```

**Harmonize (`--harmonize`).** DeepFakeFace fakes were made by squashing the full wiki photo
to 512×512 and saving it with Pillow's default JPEG settings (quality 75, 4:2:0), while the real
images are untouched originals of many sizes and formats. A detector can learn that processing
difference instead of the manipulation. `--harmonize` gives every real image the same
512×512 bicubic resize + JPEG q75 round trip before the normal 224×224 resize (fakes untouched,
also applied at evaluation). Baselines run without it (as specified); run both to measure how
much the processing shortcut contributes.

Device auto-detection order: CUDA → TPU (only if `torch_xla` is installed) → Apple MPS → CPU.
Override with `--device cuda|tpu|mps|cpu|auto`. The detected device is logged and saved in
every run (`device.json`, `config.yaml`, `metrics.json`).

Each seed writes `outputs/<experiment>/<YYYYmmdd-HHMMSS>_s<seed>_<user>/` (set `DFF_USER` to
choose `<user>`), and the command ends with `outputs/<experiment>/summary.json` (mean ± std
over seeds). Re-aggregate later with `python -m src.aggregate --experiment baseline_faceswap`.

## 3. Evaluate

```bash
RUN=outputs/baseline_faceswap/<run_folder>
python -m src.evaluate --run $RUN                      # test split of its own task
python -m src.evaluate --run $RUN --task text2img      # cross-task (Phase 5)
python -m src.evaluate --run $RUN --task inpainting    # unseen manipulation (Phase 5)
```

Results go to `<run>/eval/<task>_<split>_<condition>/`.

Metrics: accuracy, precision, recall, F1 (fixed threshold 0.5), AUROC, confusion matrix and
ROC curve. The validation-optimal threshold is logged for reference only.

## 4. What goes into git

| committed | NOT committed (gitignored) |
|---|---|
| code, configs, `configs/local.example.yaml` | `dataset/` and any images, `*.zip` |
| `data/splits/*.csv`, `data/metadata/file_index.csv.gz`, `data/reports/*` | `configs/local.yaml` |
| small run results: `*.json`, `*.csv`, `*.png`, `config.yaml`, `train.log` under `outputs/` | checkpoints (`*.pt`, `outputs/**/checkpoints/`), `outputs/cache/` |

**Checkpoints** stay under `outputs/` locally. Share them through external storage (a shared
drive, or a private Hugging Face Hub model repo) and keep the run folder name, so
`python -m src.evaluate --run <run> --checkpoint <downloaded best.pt>` can reproduce results.

## 5. Regenerating the data manifests (maintainers only)

The committed manifests define the experiment; do not regenerate them casually.

```bash
python -m src.data.scan      # file index: hashes, sizes, corruption, pHash (resumable)
python -m src.data.split     # master split + task manifests + reports
python -m src.data.dedup     # exact / near-duplicate report across splits
```

## 6. Tests

```bash
python -m pytest -q          # synthetic data; no dataset needed
```

## Scope

Conclusions apply primarily to celebrity face images (IMDB-WIKI); see `CLAUDE.md` for
assumptions, the split design, and the project TODO list.
