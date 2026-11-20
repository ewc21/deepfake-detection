"""Aggregate test metrics across seeds/runs of an experiment (mean ± std).

    python -m src.aggregate --experiment baseline_faceswap
    python -m src.aggregate --experiment baseline_faceswap --runs <run_dir> <run_dir> ...

Writes outputs/<experiment>/summary.csv and summary.json (all runs found, newest
run per seed by default) — small files that are committed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from .metrics import HEADLINE_KEYS, mean_std
from .paths import PROJECT_ROOT


def collect(run_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for rd in run_dirs:
        f = rd / "metrics.json"
        if not f.exists():
            continue
        m = json.loads(f.read_text(encoding="utf-8"))
        row = {"run": rd.name, "seed": m["seed"], "device": m["device"]["device"],
               "best_epoch": m["best_epoch"], "val_optimal_threshold": m["val_optimal_threshold"]}
        row.update({f"test_{k}": m["test"][k] for k in HEADLINE_KEYS})
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> dict:
    return {k: mean_std(df[f"test_{k}"].tolist()) for k in HEADLINE_KEYS}


def write_summary(exp_dir: Path, run_dirs: list[Path]) -> dict:
    df = collect(run_dirs)
    if df.empty:
        raise FileNotFoundError(f"No finished runs (metrics.json) found in {exp_dir}")
    summary = {"experiment": exp_dir.name, "runs": df["run"].tolist(), "metrics": summarise(df)}
    df.to_csv(exp_dir / "summary_runs.csv", index=False)
    with open(exp_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    lines = [f"{k:9s} {v['mean']:.4f} ± {v['std']:.4f} (n={v['n']})" for k, v in summary["metrics"].items()]
    print(f"[{exp_dir.name}] test metrics over {len(df)} runs:\n  " + "\n  ".join(lines))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--output-root", default=os.environ.get("DFF_OUTPUT_ROOT", "outputs"),
                    help="Default: $DFF_OUTPUT_ROOT if set, else outputs/")
    ap.add_argument("--runs", nargs="*", help="Specific run dirs (default: newest run per seed)")
    args = ap.parse_args(argv)
    exp_dir = Path(args.output_root)
    exp_dir = (exp_dir if exp_dir.is_absolute() else PROJECT_ROOT / exp_dir) / args.experiment
    if args.runs:
        runs = [Path(r) for r in args.runs]
    else:
        df = collect(sorted(p for p in exp_dir.iterdir() if p.is_dir()))
        latest = df.sort_values("run").groupby("seed").tail(1)  # run names start with a timestamp
        runs = [exp_dir / r for r in latest["run"]]
    write_summary(exp_dir, runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
