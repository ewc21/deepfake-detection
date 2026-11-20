"""Train a binary deepfake detector (one config entry per task), for one or more seeds.

    python -m src.train --task faceswap
    python -m src.train --task text2img --model resnet18 --seeds 0
    python -m src.train --task combined --device cpu --set train.epochs=1 data.subset_per_split=256
    python -m src.train --task faceswap --harmonize     # real images get the fakes' processing

Each seed writes outputs/<experiment>/<timestamp>_s<seed>_<user>/ containing:
  config.yaml, device.json, train.log, history.csv, curves.png,
  metrics.json (val + test @0.5, val-optimal threshold for reference),
  predictions_test.csv, roc_test.png, confusion_test.png,
  checkpoints/best.pt (gitignored; share via external storage)
After all seeds: outputs/<experiment>/summary.json + summary_runs.csv (mean ± std).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from .aggregate import write_summary
from .config import load_config, save_config
from .data.verify import check_task_files
from .device import DeviceContext, resolve_device, state_dict_to_cpu
from .engine import load_split, make_loader, predict, predictions_frame, train_one_epoch
from .metrics import compute_metrics, optimal_threshold
from .models import build_model
from .paths import PROJECT_ROOT, resolve_dataset_root
from .plots import save_confusion, save_roc, save_training_curves
from .run_dir import make_run_dir
from .seed import set_seed

log = logging.getLogger("train")


def setup_logging(run_dir: Path | None = None) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S")
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler):
            root.removeHandler(h)
            h.close()
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.addHandler(sh)
    if run_dir is not None:
        fh = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="Base config (default: configs/default.yaml)")
    ap.add_argument("--task", default=None, help="Task from configs/tasks.yaml (faceswap|text2img|combined)")
    ap.add_argument("--device", default=None, choices=["auto", "cuda", "tpu", "mps", "cpu"])
    ap.add_argument("--model", default=None, help="Backbone name, e.g. efficientnet_b0, resnet18")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="e.g. --seeds 0 1 2")
    ap.add_argument("--experiment", default=None, help="Output folder name (default baseline_<task>)")
    ap.add_argument("--dataset-root", default=None, help="Overrides DFF_ROOT / configs/local.yaml")
    ap.add_argument("--harmonize", action="store_true",
                    help="Give real images the fakes' processing (512x512 bicubic + JPEG q75); "
                         "see src/data/harmonize.py. Default experiment name gets '_harmonized'.")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Config overrides")
    ap.add_argument("--skip-check", action="store_true", help="Skip the dataset-vs-manifest check")
    return ap.parse_args(argv)


def build_cfg(args) -> dict:
    overrides = list(args.set)
    if args.harmonize:  # applied before load_config so the default experiment name gets "_harmonized"
        overrides.append("data.harmonize_real.enabled=true")
    cfg = load_config(args.config, overrides, task=args.task)
    if args.device:
        cfg["device"] = args.device
    if args.model:
        cfg["model"]["name"] = args.model
    if args.seeds:
        cfg["seeds"] = args.seeds
    if args.experiment:
        cfg["experiment"] = args.experiment
    if cfg["task_def"].get("eval_only"):
        log.warning("Task '%s' is marked eval_only in configs/tasks.yaml.", cfg["task"])
    return cfg


def train_one_seed(cfg: dict, seed: int, root: Path, ctx: DeviceContext) -> Path:
    run_dir = make_run_dir(cfg["output_root"], cfg["experiment"], seed)
    setup_logging(run_dir)
    device_info = ctx.describe()
    log.info("Run dir: %s", run_dir.relative_to(PROJECT_ROOT) if run_dir.is_relative_to(PROJECT_ROOT) else run_dir)
    log.info("Detected device: %s", device_info)

    set_seed(seed, cfg["train"]["deterministic"])
    ctx.seed(seed)
    run_cfg = {**cfg, "seed": seed, "device_info": device_info, "run_dir": run_dir.name}
    save_config(run_cfg, run_dir / "config.yaml")
    (run_dir / "device.json").write_text(json.dumps(device_info, indent=2), encoding="utf-8")

    task = cfg["task"]
    dfs = {s: load_split(cfg, task, s, seed) for s in ("train", "val", "test")}
    log.info("Images: %s", {s: len(d) for s, d in dfs.items()})
    train_loader = make_loader(root, dfs["train"], cfg, ctx, train=True, seed=seed,
                               perturbations=cfg["data"]["train_perturbations"])
    val_loader = make_loader(root, dfs["val"], cfg, ctx, train=False, seed=seed)
    test_loader = make_loader(root, dfs["test"], cfg, ctx, train=False, seed=seed)

    m = cfg["model"]
    model = build_model(m["name"], m["pretrained"], m["freeze_backbone"]).to(ctx.device)
    params = [p for p in model.parameters() if p.requires_grad]
    t = cfg["train"]
    if t["optimizer"].lower() != "adamw":
        raise ValueError("Only optimizer=adamw is implemented")
    optimizer = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["weight_decay"])
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # ---------------------------------------------------------------- training loop
    ckpt_path = run_dir / "checkpoints" / "best.pt"
    best, best_epoch, bad_epochs, history = -math.inf, 0, 0, []
    patience = t["early_stopping"]["patience"]
    for epoch in range(1, t["epochs"] + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, ctx)
        probs, labels, _, val_loss = predict(model, val_loader, ctx, loss_fn)
        vm = compute_metrics(labels, probs, cfg["eval"]["threshold"])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_auroc": vm["auroc"], "val_accuracy": vm["accuracy"], "val_f1": vm["f1"],
                        "seconds": round(time.time() - t0, 1)})
        log.info("epoch %d | train_loss %.4f | val_loss %.4f | val_auroc %.4f | val_acc %.4f | %.0fs",
                 epoch, train_loss, val_loss, vm["auroc"], vm["accuracy"], time.time() - t0)
        score = vm[t["early_stopping"]["metric"]]
        if score > best:
            best, best_epoch, bad_epochs = score, epoch, 0
            torch.save({"model": state_dict_to_cpu(model), "model_name": m["name"], "epoch": epoch,
                        "seed": seed, "val_auroc": score, "task": task}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log.info("Early stopping: no val %s improvement for %d epochs", t["early_stopping"]["metric"], patience)
                break
    hist = pd.DataFrame(history)
    hist.to_csv(run_dir / "history.csv", index=False)
    save_training_curves(hist, run_dir / "curves.png")

    # ---------------------------------------------------------------- final evaluation
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"])
    thr = cfg["eval"]["threshold"]
    vp, vl, _, _ = predict(model, val_loader, ctx)
    val_metrics = compute_metrics(vl, vp, thr)
    val_opt = optimal_threshold(vl, vp)
    tp, tl, ti, _ = predict(model, test_loader, ctx)
    test_metrics = compute_metrics(tl, tp, thr)
    metrics = {
        "task": task, "experiment": cfg["experiment"], "seed": seed, "model": m["name"],
        "device": device_info, "best_epoch": best_epoch, "epochs_run": len(history),
        "val": val_metrics, "test": test_metrics,
        "val_optimal_threshold": val_opt,
        "test_at_val_optimal_threshold_reference_only": compute_metrics(tl, tp, val_opt),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    predictions_frame(dfs["test"], tp, tl, ti, thr).to_csv(run_dir / "predictions_test.csv", index=False)
    title = f"{task} / {m['name']} / seed {seed}"
    save_roc(tl, tp, run_dir / "roc_test.png", f"ROC (test) — {title}", test_metrics["auroc"])
    save_confusion(test_metrics["confusion_matrix"], run_dir / "confusion_test.png", f"Test @ {thr} — {title}")
    log.info("TEST @%.2f: acc %.4f | prec %.4f | rec %.4f | f1 %.4f | auroc %.4f",
             thr, test_metrics["accuracy"], test_metrics["precision"], test_metrics["recall"],
             test_metrics["f1"], test_metrics["auroc"])
    return run_dir


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging()
    cfg = build_cfg(args)
    root = resolve_dataset_root(args.dataset_root)
    if not args.skip_check:
        missing = check_task_files(root, cfg["task"])
        if missing:
            log.error("%d files of task '%s' are missing under %s (e.g. %s). Run python -m src.data.verify",
                      len(missing), cfg["task"], root, missing[:3])
            return 1
    ctx = resolve_device(cfg["device"], cfg["train"]["amp"])
    run_dirs = [train_one_seed(cfg, seed, root, ctx) for seed in cfg["seeds"]]
    setup_logging()  # detach the last run's file handler
    write_summary(run_dirs[0].parent, run_dirs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
