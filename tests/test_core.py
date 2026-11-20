"""Device selection, config, metrics, models, transforms and an end-to-end training smoke test."""
from __future__ import annotations

import json
import sys

import numpy as np
import pytest
import torch

from src.config import load_config
from src.data.split import build_master, build_task_manifest
from src.data.transforms import build_transform, condition_name
from src.device import resolve_device, xla_installed
from src.metrics import compute_metrics, mean_std, optimal_threshold
from src.models import build_model


def test_torch_xla_not_imported_by_pipeline():
    import src.train  # noqa: F401
    import src.evaluate  # noqa: F401

    assert "torch_xla" not in sys.modules or xla_installed()


def test_device_auto_and_cpu_override():
    ctx = resolve_device("cpu", "auto")
    assert ctx.kind == "cpu" and ctx.amp_dtype is None and ctx.scaler is None
    auto = resolve_device("auto", "auto")
    assert auto.kind in {"cuda", "tpu", "mps", "cpu"}
    assert "device" in auto.describe()
    with pytest.raises(ValueError):
        resolve_device("gpu")


def test_unavailable_device_raises():
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            resolve_device("cuda")
    if not xla_installed():
        with pytest.raises(RuntimeError):
            resolve_device("tpu")


def test_config_overrides():
    cfg = load_config(overrides=["train.lr=3e-4", "model.name=resnet18"], task="combined", use_local=False)
    assert cfg["train"]["lr"] == pytest.approx(3e-4)
    assert cfg["model"]["name"] == "resnet18"
    assert cfg["experiment"] == "baseline_combined"
    assert cfg["task_def"]["fakes"] == {"insight": 0.5, "text2img": 0.5}
    assert "dataset_root" not in cfg  # never stored in run configs


def test_metrics():
    y = [0, 0, 1, 1]
    p = [0.1, 0.6, 0.4, 0.9]
    m = compute_metrics(y, p, 0.5)
    assert m["accuracy"] == 0.5 and m["confusion_matrix"] == [[1, 1], [1, 1]]
    assert m["auroc"] == 0.75
    assert 0 <= optimal_threshold(y, p) <= 1
    s = mean_std([0.9, 0.92, 0.94])
    assert s["mean"] == pytest.approx(0.92) and s["std"] == pytest.approx(0.02)


@pytest.mark.parametrize("name", ["resnet18", "efficientnet_b0"])
def test_models_single_logit(name):
    model = build_model(name, pretrained=False)
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 1)


def test_transform_baseline_no_augmentation():
    from PIL import Image

    tf = build_transform(224, None)
    img = Image.fromarray(np.full((50, 70, 3), 128, dtype=np.uint8))
    a, b = tf(img), tf(img)
    assert a.shape == (3, 224, 224) and torch.equal(a, b)  # deterministic: no random aug
    assert condition_name(None) == "clean"
    assert condition_name({"name": "jpeg", "quality": 50}) == "jpeg_quality-50"


def test_training_smoke(synthetic_root, synthetic_index, tmp_path):
    """Two tiny epochs on CPU end-to-end: loaders, training, checkpoint, metrics, plots."""
    from src.train import train_one_seed

    master, _ = build_master(synthetic_index, seed=1, ratios={"train": 0.7, "val": 0.15, "test": 0.15},
                             phash_max_distance=4)
    cfg = load_config(task="faceswap", use_local=False, overrides=[
        "model.name=resnet18", "model.pretrained=false", "train.epochs=2", "train.batch_size=8",
        "eval.batch_size=8", "data.num_workers=0", "data.image_size=64",
    ])
    splits = tmp_path / "splits"
    (splits / "tasks").mkdir(parents=True)
    build_task_manifest(master, cfg["task_def"], seed=1).to_csv(splits / "tasks" / "faceswap.csv", index=False)
    cfg["splits_dir"] = str(splits)
    cfg["output_root"] = str(tmp_path / "outputs")

    ctx = resolve_device("cpu", "auto")
    run_dir = train_one_seed(cfg, seed=0, root=synthetic_root, ctx=ctx)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["device"]["device"] == "cpu"
    assert set(metrics["test"]) >= {"accuracy", "precision", "recall", "f1", "auroc", "confusion_matrix"}
    for f in ["config.yaml", "device.json", "history.csv", "roc_test.png", "confusion_test.png",
              "predictions_test.csv", "checkpoints/best.pt"]:
        assert (run_dir / f).exists(), f
    assert run_dir.name.split("_")[1] == "s0"
