"""Comparison script on synthetic run folders (no training needed)."""
from __future__ import annotations

import json

import pandas as pd
import pytest
import yaml

from src.compare import main as compare_main


def _fake_run(root, experiment, run, task, seed, harmonized, auroc, acc):
    d = root / experiment / run
    d.mkdir(parents=True)
    test = {"accuracy": acc, "precision": acc, "recall": acc, "f1": acc, "auroc": auroc}
    (d / "metrics.json").write_text(json.dumps({
        "task": task, "model": "efficientnet_b0", "seed": seed, "device": {"device": "cpu"},
        "best_epoch": 3, "val_optimal_threshold": 0.5, "test": test}))
    (d / "config.yaml").write_text(yaml.safe_dump({"data": {"harmonize_real": {"enabled": harmonized}}}))


def test_compare_table_and_figure(tmp_path):
    for task, base in [("faceswap", 0.95), ("text2img", 0.99), ("combined", 0.97)]:
        for harmonized, drop in [(False, 0.0), (True, 0.05)]:
            exp = f"baseline_{task}" + ("_harmonized" if harmonized else "")
            for seed in range(3):
                _fake_run(tmp_path, exp, f"20261120-1200{seed:02d}_s{seed}_me", task, seed, harmonized,
                          base - drop + 0.01 * seed, base - drop)
    # an older run of seed 0 that must be ignored (newest per seed wins)
    _fake_run(tmp_path, "baseline_faceswap", "20261119-090000_s0_me", "faceswap", 0, False, 0.10, 0.10)

    assert compare_main(["--output-root", str(tmp_path)]) == 0
    out = tmp_path / "comparison"
    for f in ["comparison.csv", "comparison.md", "comparison_runs.csv", "comparison.png"]:
        assert (out / f).exists(), f
    table = pd.read_csv(out / "comparison.csv")
    assert len(table) == 6 and set(table["variant"]) == {"plain", "harmonized"}
    row = table[(table["task"] == "faceswap") & (~table["harmonized"])].iloc[0]
    assert row["n_seeds"] == 3
    assert row["auroc_mean"] == pytest.approx(0.96)   # 0.95, 0.96, 0.97 -> old 0.10 run ignored
    assert row["auroc_std"] == pytest.approx(0.01)
    assert list(table["task"].drop_duplicates()) == ["faceswap", "text2img", "combined"]
    assert "± " in (out / "comparison.md").read_text(encoding="utf-8")


def test_compare_no_runs(tmp_path):
    assert compare_main(["--output-root", str(tmp_path)]) == 1
