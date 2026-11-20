"""Evaluate a trained run's best checkpoint on any task/split/condition.

    # same task, test split (re-evaluation)
    python -m src.evaluate --run outputs/baseline_faceswap/<run_dir>
    # cross-task (Phase 5): face-swap detector on text2img fakes
    python -m src.evaluate --run outputs/baseline_faceswap/<run_dir> --task text2img
    # unseen manipulation (Phase 5)
    python -m src.evaluate --run outputs/baseline_combined/<run_dir> --task inpainting
    # test-time condition (Phase 3, once perturbations are registered)
    python -m src.evaluate --run ... --condition "{name: jpeg, quality: 50}"

The master split guarantees that the test images of every task come from source wiki
images never seen in training by ANY task, so cross-task evaluation is leakage-free.
Results go to <run_dir>/eval/<task>_<split>_<condition>/.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import yaml

from .config import load_tasks
from .data.transforms import condition_name
from .device import resolve_device
from .engine import load_split, make_loader, predict, predictions_frame
from .metrics import compute_metrics
from .models import build_model
from .paths import resolve_dataset_root
from .plots import save_confusion, save_roc
from .seed import set_seed

log = logging.getLogger("evaluate")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="Run directory containing config.yaml and checkpoints/best.pt")
    ap.add_argument("--task", default=None, help="Evaluation task (default: the run's training task)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--condition", default=None, help="YAML dict for a test-time perturbation")
    ap.add_argument("--checkpoint", default=None, help="Default: <run>/checkpoints/best.pt")
    ap.add_argument("--device", default=None, choices=["auto", "cuda", "tpu", "mps", "cpu"])
    ap.add_argument("--dataset-root", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    run = Path(args.run)
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    task = args.task or cfg["task"]
    if task not in load_tasks():
        raise KeyError(f"Unknown task {task}")
    condition = yaml.safe_load(args.condition) if args.condition else None

    root = resolve_dataset_root(args.dataset_root)
    ctx = resolve_device(args.device or cfg["device"], cfg["train"]["amp"])
    set_seed(cfg["seed"], cfg["train"]["deterministic"])
    df = load_split(cfg, task, args.split, cfg["seed"])
    loader = make_loader(root, df, cfg, ctx, train=False, seed=cfg["seed"],
                         perturbations=[condition] if condition else None)

    ckpt = Path(args.checkpoint) if args.checkpoint else run / "checkpoints" / "best.pt"
    state = torch.load(ckpt, map_location="cpu")
    model = build_model(state["model_name"], pretrained=False)
    model.load_state_dict(state["model"])
    model.to(ctx.device)

    thr = cfg["eval"]["threshold"]
    probs, labels, idxs, _ = predict(model, loader, ctx)
    metrics = compute_metrics(labels, probs, thr)
    out = run / "eval" / f"{task}_{args.split}_{condition_name(condition)}"
    out.mkdir(parents=True, exist_ok=True)
    result = {"trained_on": cfg["task"], "evaluated_on": task, "split": args.split,
              "condition": condition, "checkpoint_epoch": state.get("epoch"),
              "device": ctx.describe(), **metrics}
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    predictions_frame(df, probs, labels, idxs, thr).to_csv(out / "predictions.csv", index=False)
    title = f"{cfg['task']} model on {task} ({args.split}, {condition_name(condition)})"
    save_roc(labels, probs, out / "roc.png", title, metrics["auroc"])
    save_confusion(metrics["confusion_matrix"], out / "confusion.png", title)
    log.info("%s -> acc %.4f | f1 %.4f | auroc %.4f  (saved to %s)", title,
             metrics["accuracy"], metrics["f1"], metrics["auroc"], out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
