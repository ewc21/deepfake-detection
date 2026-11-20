"""Project paths and dataset-root resolution.

The dataset location is personal to each collaborator and is NEVER stored in a committed
file. It is resolved in this order (first hit wins):
  1. explicit argument (e.g. --dataset-root on the command line)
  2. environment variable DFF_ROOT
  3. `dataset_root` in configs/local.yaml (gitignored; copy configs/local.example.yaml)
Relative paths are interpreted relative to the repository root.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
METADATA_DIR = DATA_DIR / "metadata"
REPORTS_DIR = DATA_DIR / "reports"
LOCAL_CONFIG = CONFIG_DIR / "local.yaml"

# Folder names inside the dataset root (one per DeepFakeFace archive).
SOURCES = ("wiki", "insight", "text2img", "inpainting")
FAKE_SOURCES = ("insight", "text2img", "inpainting")
EXPECTED_FILES_PER_SOURCE = 30_000  # per the DeepFakeFace dataset card


def load_local_config() -> dict:
    """Return configs/local.yaml as a dict ({} if the file does not exist)."""
    if LOCAL_CONFIG.exists():
        with open(LOCAL_CONFIG, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _absolute(p: str | os.PathLike) -> Path:
    path = Path(p).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_dataset_root(explicit: str | os.PathLike | None = None, must_exist: bool = True) -> Path:
    """Find the folder that contains wiki/, insight/, text2img/, inpainting/."""
    if explicit:
        root, origin = _absolute(explicit), "command line"
    elif os.environ.get("DFF_ROOT"):
        root, origin = _absolute(os.environ["DFF_ROOT"]), "DFF_ROOT"
    elif load_local_config().get("dataset_root"):
        root, origin = _absolute(load_local_config()["dataset_root"]), "configs/local.yaml"
    else:
        raise FileNotFoundError(
            "Dataset location is not configured. Either copy configs/local.example.yaml to "
            "configs/local.yaml and set dataset_root, or set the DFF_ROOT environment variable. "
            "To download the data: python -m src.data.download --dest <dir>"
        )
    if must_exist and not root.is_dir():
        raise FileNotFoundError(f"Dataset root {root} (from {origin}) does not exist.")
    return root


def to_posix_rel(path: Path, root: Path) -> str:
    """Path relative to `root` with forward slashes (manifests are OS-independent)."""
    return path.relative_to(root).as_posix()
