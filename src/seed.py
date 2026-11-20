"""Reproducibility helpers: seed every RNG and enable deterministic kernels where practical."""
from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

log = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU + all GPUs).

    With deterministic=True, cuDNN autotuning is disabled and PyTorch is asked to use
    deterministic algorithms. `warn_only=True` keeps runs going when an op has no
    deterministic implementation (a warning is logged instead of an error).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # affects subprocesses (e.g. DataLoader workers)
    if deterministic:
        # Required by cuBLAS for deterministic matmuls on CUDA >= 10.2
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: derive NumPy/random seeds from the torch worker seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Generator for DataLoader shuffling/sampling so the batch order is reproducible."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
