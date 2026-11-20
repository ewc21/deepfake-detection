"""Compare finished experiments: one table + one figure across tasks and variants.

    python -m src.compare                         # every experiment under outputs/
    python -m src.compare --experiments baseline_faceswap baseline_faceswap_harmonized
    python -m src.compare --all-runs              # use every run, not just the newest per seed

For each experiment it takes the newest run per seed (like src.aggregate), reads the TEST
metrics at the fixed 0.5 threshold, and reports mean ± sample std over seeds.

Writes to outputs/comparison/ (small files, committed):
  comparison.csv       one row per experiment: task, model, harmonized, n_seeds, <metric>_mean/_std
  comparison.md        the same as a readable Markdown table (mean ± std)
  comparison_runs.csv  one row per run that went into the table (for checking / re-plotting)
  comparison.png       dot plot: one panel per metric, tasks on x, one marker per variant
                       (e.g. plain vs. --harmonize), error bars = ±1 std over seeds
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

from .metrics import HEADLINE_KEYS, mean_std
from .paths import PROJECT_ROOT

OUT_NAME = "comparison"
TASK_ORDER = ["faceswap", "text2img", "combined", "inpainting"]
PLOT_METRICS = ("auroc", "accuracy", "f1")
METRIC_LABELS = {"auroc": "AUROC", "accuracy": "Accuracy @0.5", "f1": "F1 @0.5",
                 "precision": "Precision @0.5", "recall": "Recall @0.5"}

# Validated categorical palette (dataviz reference palette, slots 1-3 pass all-pairs CVD
# checks on the light surface). Assigned in fixed order to variants, never cycled.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
MARKERS = ["o", "s", "D"]  # secondary encoding: identity is never color alone


def _read_run(run_dir: Path) -> dict | None:
    """One row per finished run (metrics.json present)."""
    mfile, cfile = run_dir / "metrics.json", run_dir / "config.yaml"
    if not mfile.exists():
        return None
    m = json.loads(mfile.read_text(encoding="utf-8"))
    cfg = yaml.safe_load(cfile.read_text(encoding="utf-8")) if cfile.exists() else {}
    harmonized = bool(((cfg.get("data") or {}).get("harmonize_real") or {}).get("enabled", False))
    row = {
        "experiment": run_dir.parent.name, "run": run_dir.name, "task": m["task"], "model": m["model"],
        "harmonized": harmonized, "seed": m["seed"], "device": m["device"]["device"],
        "best_epoch": m["best_epoch"], "val_optimal_threshold": m["val_optimal_threshold"],
    }
    row.update({k: m["test"][k] for k in HEADLINE_KEYS})
    return row


def collect_runs(output_root: Path, experiments: list[str] | None, all_runs: bool) -> pd.DataFrame:
    exp_dirs = sorted(p for p in output_root.iterdir() if p.is_dir() and p.name not in (OUT_NAME, "cache", "logs"))
    if not experiments:  # smoke tests are setup checks, not results
        exp_dirs = [p for p in exp_dirs if not p.name.startswith("smoke")]
    if experiments:
        wanted = set(experiments)
        missing = wanted - {p.name for p in exp_dirs}
        if missing:
            raise FileNotFoundError(f"Experiments not found under {output_root}: {sorted(missing)}")
        exp_dirs = [p for p in exp_dirs if p.name in wanted]
    rows = [r for e in exp_dirs for d in sorted(e.iterdir()) if d.is_dir() and (r := _read_run(d))]
    runs = pd.DataFrame(rows)
    if runs.empty:
        return runs
    if not all_runs:  # newest run per (experiment, seed); run names start with a timestamp
        runs = runs.sort_values("run").groupby(["experiment", "seed"]).tail(1)
    return runs.sort_values(["experiment", "seed"]).reset_index(drop=True)


def variant_label(model: str, harmonized: bool, multiple_models: bool) -> str:
    base = model if multiple_models else "plain"
    if harmonized:
        return f"{model} + harmonize" if multiple_models else "harmonized"
    return base


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    multiple_models = runs["model"].nunique() > 1
    rows = []
    for exp, g in runs.groupby("experiment", sort=False):
        first = g.iloc[0]
        row = {"experiment": exp, "task": first["task"], "model": first["model"],
               "harmonized": bool(first["harmonized"]),
               "variant": variant_label(first["model"], bool(first["harmonized"]), multiple_models),
               "n_seeds": int(g["seed"].nunique()), "seeds": " ".join(map(str, sorted(g["seed"])))}
        for k in HEADLINE_KEYS:
            s = mean_std(g[k].tolist())
            row[f"{k}_mean"], row[f"{k}_std"] = s["mean"], s["std"]
        rows.append(row)
    table = pd.DataFrame(rows)
    order = {t: i for i, t in enumerate(TASK_ORDER)}
    table["_t"] = table["task"].map(lambda t: order.get(t, len(order)))
    return table.sort_values(["_t", "harmonized", "model", "experiment"]).drop(columns="_t").reset_index(drop=True)


def to_markdown(table: pd.DataFrame) -> str:
    cols = ["task", "variant", "n_seeds", *HEADLINE_KEYS]
    head = "| " + " | ".join(METRIC_LABELS.get(c, c) for c in cols) + " |"
    lines = [head, "|" + "---|" * len(cols)]
    for r in table.itertuples():
        cells = [r.task, r.variant, str(r.n_seeds)]
        cells += [f"{getattr(r, k + '_mean'):.4f} ± {getattr(r, k + '_std'):.4f}" for k in HEADLINE_KEYS]
        lines.append("| " + " | ".join(cells) + " |")
    note = ("\nTest split, fixed threshold 0.5; mean ± sample std (ddof=1) over seeds. "
            "Std is 0.0000 when only one seed is available.\n")
    return "\n".join(lines) + "\n" + note


def plot(table: pd.DataFrame, path: Path) -> None:
    """Dot plot with ±1 std error bars, one panel per metric (no dual axes)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    variants = list(dict.fromkeys(table["variant"]))  # fixed order of first appearance
    if len(variants) > len(SERIES_COLORS):
        print(f"[compare] {len(variants)} variants; plotting the first {len(SERIES_COLORS)} "
              f"(see comparison.md for all): {variants[:len(SERIES_COLORS)]}")
        variants = variants[:len(SERIES_COLORS)]
    tasks = [t for t in TASK_ORDER if t in set(table["task"])] + sorted(set(table["task"]) - set(TASK_ORDER))
    x = np.arange(len(tasks))
    offsets = np.linspace(-0.15, 0.15, len(variants)) if len(variants) > 1 else [0.0]

    fig, axes = plt.subplots(1, len(PLOT_METRICS), figsize=(4.2 * len(PLOT_METRICS), 3.8))
    fig.patch.set_facecolor(SURFACE)
    for ax, metric in zip(np.atleast_1d(axes), PLOT_METRICS):
        ax.set_facecolor(SURFACE)
        lo, hi = 1.0, 0.0
        for vi, variant in enumerate(variants):
            sub = table[table["variant"] == variant].set_index("task")
            xs, ys, es = [], [], []
            for ti, task in enumerate(tasks):
                if task in sub.index:
                    xs.append(x[ti] + offsets[vi])
                    ys.append(sub.at[task, f"{metric}_mean"])
                    es.append(sub.at[task, f"{metric}_std"])
            if not xs:
                continue
            ax.errorbar(xs, ys, yerr=es, fmt=MARKERS[vi], ms=8, color=SERIES_COLORS[vi], ecolor=SERIES_COLORS[vi],
                        elinewidth=2, capsize=4, capthick=2, markeredgecolor=SURFACE, markeredgewidth=2,
                        label=variant, zorder=3)
            lo = min(lo, min(y - e for y, e in zip(ys, es)))
            hi = max(hi, max(y + e for y, e in zip(ys, es)))
        # Zoomed axis (dots, not bars, so a non-zero baseline is honest); cap at 1.0.
        pad = max(0.01, (hi - lo) * 0.25)
        ax.set_ylim(max(0.0, lo - pad), min(1.005, hi + pad))
        ax.set_xticks(x, tasks)
        ax.set_xlim(-0.5, len(tasks) - 0.5)
        ax.set_title(METRIC_LABELS[metric], color=INK, fontsize=11, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    if len(variants) >= 2:
        fig.legend(handles, labels, loc="upper right", ncol=len(variants), frameon=False,
                   labelcolor=INK_2, fontsize=9, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Test performance by task (mean ± 1 std over seeds)", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-root", default=os.environ.get("DFF_OUTPUT_ROOT", "outputs"),
                    help="Default: $DFF_OUTPUT_ROOT if set, else outputs/")
    ap.add_argument("--experiments", nargs="*", help="Experiment folder names (default: all)")
    ap.add_argument("--all-runs", action="store_true", help="Use every run, not only the newest per seed")
    ap.add_argument("--out", default=None, help=f"Output folder (default: <output-root>/{OUT_NAME})")
    args = ap.parse_args(argv)

    root = Path(args.output_root)
    root = root if root.is_absolute() else PROJECT_ROOT / root
    runs = collect_runs(root, args.experiments, args.all_runs)
    if runs.empty:
        print(f"No finished runs (metrics.json) found under {root}. Train first: python -m src.train --task faceswap")
        return 1
    table = summarise(runs)
    out = Path(args.out) if args.out else root / OUT_NAME
    out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "comparison_runs.csv", index=False)
    table.to_csv(out / "comparison.csv", index=False)
    md = to_markdown(table)
    (out / "comparison.md").write_text(md, encoding="utf-8")
    plot(table, out / "comparison.png")
    print(md)
    small = table[table["n_seeds"] < 2]["experiment"].tolist()
    if small:
        print(f"[compare] fewer than 2 seeds (std not meaningful): {small}")
    print(f"Wrote {out}/comparison.csv, comparison.md, comparison_runs.csv, comparison.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
