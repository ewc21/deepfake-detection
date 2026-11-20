"""PyTorch Dataset that reads a task manifest (paths relative to the dataset root)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..paths import SPLITS_DIR


def load_manifest(task: str, split: str | None = None, splits_dir: Path | None = None) -> pd.DataFrame:
    path = Path(splits_dir or SPLITS_DIR) / "tasks" / f"{task}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m src.data.split")
    df = pd.read_csv(path)
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    return df


def balanced_subset(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Debug helper: n rows with equal real/fake counts, reproducible."""
    per_class = max(1, n // 2)
    parts = [
        g.sample(n=min(per_class, len(g)), random_state=seed) for _, g in df.groupby("label", sort=True)
    ]
    return pd.concat(parts).sort_values("relpath").reset_index(drop=True)


class ManifestDataset(Dataset):
    """Returns (image_tensor, label_float, row_index).

    `real_preprocess` (optional) is applied to REAL images only (label 0), before
    `transform` — used by the harmonize option (see src/data/harmonize.py).
    """

    def __init__(self, root: Path, df: pd.DataFrame, transform, real_preprocess=None):
        self.root = Path(root)
        self.relpaths = df["relpath"].tolist()
        self.labels = df["label"].to_numpy(dtype=np.float32)
        self.transform = transform
        self.real_preprocess = real_preprocess

    def __len__(self) -> int:
        return len(self.relpaths)

    def __getitem__(self, i: int):
        with Image.open(self.root / self.relpaths[i]) as im:
            img = im.convert("RGB")
        if self.real_preprocess is not None and self.labels[i] == 0:
            img = self.real_preprocess(img)
        return self.transform(img), torch.tensor(self.labels[i]), i
