"""Unique run-output folders: outputs/<experiment>/<YYYYmmdd-HHMMSS>_s<seed>_<user>.

<user> comes from the DFF_USER environment variable if set, otherwise the OS login name.
Unique names mean results from different collaborators never overwrite each other.
"""
from __future__ import annotations

import getpass
import os
import re
from datetime import datetime
from pathlib import Path

from .paths import PROJECT_ROOT


def current_user() -> str:
    try:
        name = os.environ.get("DFF_USER") or getpass.getuser()
    except Exception:  # getpass can fail in some containers
        name = "unknown"
    return re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-") or "unknown"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def make_run_dir(output_root: str, experiment: str, seed: int) -> Path:
    base = Path(output_root)
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    exp_dir = base / experiment
    name = f"{timestamp()}_s{seed}_{current_user()}"
    run_dir, i = exp_dir / name, 1
    while run_dir.exists():  # extremely unlikely, but never overwrite
        run_dir, i = exp_dir / f"{name}_{i}", i + 1
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    return run_dir
