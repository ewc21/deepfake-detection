"""Shared fixtures: a tiny synthetic DeepFakeFace-like dataset (no real data needed)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.scan import COLUMNS, inspect_file  # noqa: E402


def _make_image(path: Path, rng: np.random.Generator, size: int = 48) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, quality=95)


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory) -> Path:
    """40 wiki images + one fake of each type per wiki image, IMDB-WIKI-style names."""
    root = tmp_path_factory.mktemp("dff")
    rng = np.random.default_rng(0)
    for i in range(40):
        pid = 100000 + i * 37
        rel = f"{pid % 100:02d}/{pid}_1980-01-{(i % 28) + 1:02d}_2010.jpg"
        for source in ("wiki", "insight", "text2img", "inpainting"):
            _make_image(root / source / rel, rng)
    # one fake with no source (should be reported + excluded)
    _make_image(root / "insight" / "99" / "999999_1970-01-01_2000.jpg", rng)
    return root


@pytest.fixture(scope="session")
def synthetic_index(synthetic_root) -> pd.DataFrame:
    rels = sorted(p.relative_to(synthetic_root).as_posix() for p in synthetic_root.rglob("*.jpg"))
    rows = [inspect_file((str(synthetic_root), r)) for r in rels]
    return pd.DataFrame(rows, columns=COLUMNS).astype(str)
