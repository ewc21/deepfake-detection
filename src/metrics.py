"""Binary classification metrics.

Headline threshold-dependent metrics (accuracy, precision, recall, F1, confusion matrix)
use a FIXED threshold (0.5). The validation-optimal threshold (Youden's J) is logged for
reference only. "Positive" = fake (label 1).
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

HEADLINE_KEYS = ("accuracy", "precision", "recall", "f1", "auroc")


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    both_classes = len(np.unique(y_true)) == 2
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_prob)) if both_classes else float("nan"),
        # rows = true (real, fake), cols = predicted (real, fake)
        "confusion_matrix": cm.tolist(),
    }


def optimal_threshold(y_true, y_prob) -> float:
    """Threshold maximising Youden's J = TPR - FPR (reference only)."""
    fpr, tpr, thr = roc_curve(np.asarray(y_true).astype(int), np.asarray(y_prob, dtype=float))
    best = int(np.argmax(tpr - fpr))
    return float(min(thr[best], 1.0))


def mean_std(values: list[float]) -> dict:
    """Mean and sample std (ddof=1) across seeds, ignoring NaNs."""
    arr = np.array([v for v in values if v is not None and not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {"mean": float(arr.mean()), "std": std, "n": int(arr.size)}
