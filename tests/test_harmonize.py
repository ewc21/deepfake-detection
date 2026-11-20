"""Harmonize option: real images get the fakes' processing; fakes are untouched."""
from __future__ import annotations

import io
import pickle

import numpy as np
import pandas as pd
import torch
from PIL import Image, JpegImagePlugin

from src.config import load_config
from src.data.dataset import ManifestDataset
from src.data.harmonize import HarmonizeReal, build_harmonizer


def test_harmonize_output_matches_fake_format():
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (660, 440, 3), dtype=np.uint8))  # 440x660 portrait
    out = HarmonizeReal()(img)
    assert out.size == (512, 512) and out.mode == "RGB"
    # deterministic (not an augmentation)
    assert np.array_equal(np.asarray(out), np.asarray(HarmonizeReal()(img)))
    # encode settings: quality 75 tables, 4:2:0 subsampling
    buf = io.BytesIO()
    img.resize((512, 512), Image.Resampling.BICUBIC).save(buf, "JPEG", quality=75, subsampling="4:2:0")
    buf.seek(0)
    ref = Image.open(buf)
    assert JpegImagePlugin.get_sampling(ref) == 2
    assert np.array_equal(np.asarray(out), np.asarray(ref.convert("RGB")))


def test_harmonizer_pickles_and_config_switch():
    assert build_harmonizer(None) is None
    assert build_harmonizer({"enabled": False}) is None
    h = build_harmonizer({"enabled": True, "size": 512, "jpeg_quality": 75})
    assert isinstance(pickle.loads(pickle.dumps(h)), HarmonizeReal)
    cfg = load_config(task="faceswap", use_local=False)
    assert cfg["data"]["harmonize_real"]["enabled"] is False  # spec baseline unchanged
    assert cfg["experiment"] == "baseline_faceswap"
    cfg_h = load_config(task="faceswap", use_local=False, overrides=["data.harmonize_real.enabled=true"])
    assert cfg_h["experiment"] == "baseline_faceswap_harmonized"


def test_only_real_images_are_harmonized(tmp_path):
    rng = np.random.default_rng(1)
    for name in ("real.png", "fake.jpg"):
        Image.fromarray(rng.integers(0, 256, (100, 80, 3), dtype=np.uint8)).save(tmp_path / name)
    df = pd.DataFrame({"relpath": ["real.png", "fake.jpg"], "label": [0, 1]})

    class Record:
        """Transform that records the image size it receives."""
        def __init__(self):
            self.sizes = []

        def __call__(self, img):
            self.sizes.append(img.size)
            return torch.zeros(1)

    rec = Record()
    ds = ManifestDataset(tmp_path, df, rec, real_preprocess=HarmonizeReal())
    ds[0]
    ds[1]
    assert rec.sizes == [(512, 512), (80, 100)]
