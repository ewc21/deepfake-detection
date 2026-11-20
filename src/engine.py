"""Device-agnostic training/evaluation loops and data loader construction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data.dataset import ManifestDataset, balanced_subset, load_manifest
from .data.harmonize import build_harmonizer
from .data.transforms import build_transform
from .device import DeviceContext
from .paths import PROJECT_ROOT
from .seed import make_generator, seed_worker


def make_loader(
    root: Path,
    df: pd.DataFrame,
    cfg: dict,
    ctx: DeviceContext,
    *,
    train: bool,
    seed: int,
    perturbations: list[dict] | None = None,
) -> DataLoader:
    transform = build_transform(cfg["data"]["image_size"], perturbations)
    ds = ManifestDataset(root, df, transform, build_harmonizer(cfg["data"].get("harmonize_real")))
    workers = int(cfg["data"]["num_workers"])
    return DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"] if train else cfg["eval"]["batch_size"],
        shuffle=train,
        drop_last=train and (cfg["train"]["drop_last"] or ctx.force_drop_last),
        num_workers=workers,
        pin_memory=ctx.pin_memory,
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
    )


def load_split(cfg: dict, task: str, split: str, seed: int) -> pd.DataFrame:
    splits_dir = Path(cfg["splits_dir"])
    if not splits_dir.is_absolute():
        splits_dir = PROJECT_ROOT / splits_dir
    df = load_manifest(task, split, splits_dir)
    n = cfg["data"].get("subset_per_split")
    if n:
        # Same subset for every training seed so seeds differ only in training randomness.
        df = balanced_subset(df, int(n), seed=0)
    return df


def train_one_epoch(model, loader, optimizer, loss_fn, ctx: DeviceContext) -> float:
    model.train()
    total = torch.zeros((), device=ctx.device)  # accumulate on device: no per-step host sync
    count = 0
    for x, y, _ in ctx.wrap_loader(loader):
        x, y = ctx.to_device(x, y)
        optimizer.zero_grad(set_to_none=True)
        with ctx.autocast():
            logits = model(x).squeeze(1)
        loss = loss_fn(logits.float(), y)
        ctx.backward_and_step(loss, optimizer)
        total += loss.detach() * x.shape[0]
        count += x.shape[0]
    return float(total.item()) / max(count, 1)


@torch.no_grad()
def predict(model, loader, ctx: DeviceContext, loss_fn=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (probabilities, labels, row indices, mean loss)."""
    model.eval()
    probs, labels, idxs = [], [], []
    total, count = 0.0, 0
    for x, y, i in ctx.wrap_loader(loader):
        x, y = ctx.to_device(x, y)
        with ctx.autocast():
            logits = model(x).squeeze(1)
        logits = logits.float()
        if loss_fn is not None:
            total += float(loss_fn(logits, y).item()) * x.shape[0]
        count += x.shape[0]
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.cpu().numpy())
        idxs.append(i.cpu().numpy() if torch.is_tensor(i) else np.asarray(i))
    return (
        np.concatenate(probs),
        np.concatenate(labels).astype(int),
        np.concatenate(idxs),
        total / max(count, 1),
    )


def predictions_frame(df: pd.DataFrame, probs, labels, idxs, threshold: float) -> pd.DataFrame:
    out = df.iloc[idxs][["relpath", "fake_type", "source"]].reset_index(drop=True)
    out["label"] = labels
    out["prob_fake"] = np.round(probs, 6)
    out["pred"] = (probs >= threshold).astype(int)
    return out.sort_values("relpath").reset_index(drop=True)
