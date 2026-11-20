"""Configuration loading: configs/default.yaml + configs/local.yaml + CLI overrides.

Precedence (later wins): default.yaml < local.yaml < --set overrides < dedicated CLI flags.
The task definition from configs/tasks.yaml is attached as cfg["task_def"].
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterable

import yaml

from .paths import CONFIG_DIR, load_local_config

# Keys in local.yaml that are about locating data, not about the experiment.
_LOCAL_ONLY_KEYS = {"dataset_root"}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base`."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_by_dotted_key(cfg: dict, dotted: str, value: Any) -> None:
    """cfg['a']['b'] = value for dotted == 'a.b' (creates intermediate dicts)."""
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def parse_overrides(pairs: Iterable[str]) -> list[tuple[str, Any]]:
    """Parse ["train.lr=3e-4", "model.name=resnet18"]; values are parsed as YAML."""
    parsed = []
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"Override '{pair}' must look like key.subkey=value")
        key, raw = pair.split("=", 1)
        value = yaml.safe_load(raw)
        if isinstance(value, str):
            # YAML 1.1 reads "3e-4" (no dot) as a string; accept it as a number.
            try:
                value = float(value)
            except ValueError:
                pass
        parsed.append((key.strip(), value))
    return parsed


def load_tasks(path: Path | None = None) -> dict:
    with open(path or CONFIG_DIR / "tasks.yaml", "r", encoding="utf-8") as f:
        tasks = yaml.safe_load(f)
    for name, t in tasks.items():
        total = sum(t["fakes"].values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Task '{name}': fake fractions sum to {total}, expected 1.0")
    return tasks


def load_config(
    config_path: str | Path | None = None,
    overrides: Iterable[str] = (),
    task: str | None = None,
    use_local: bool = True,
) -> dict:
    """Build the resolved experiment config."""
    with open(config_path or CONFIG_DIR / "default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if use_local:
        local = {k: v for k, v in load_local_config().items() if k not in _LOCAL_ONLY_KEYS}
        cfg = deep_merge(cfg, local)
        # Per-machine output location (e.g. the TPU group bucket); --set output_root=... still wins.
        if os.environ.get("DFF_OUTPUT_ROOT"):
            cfg["output_root"] = os.environ["DFF_OUTPUT_ROOT"]
    for key, value in parse_overrides(overrides):
        set_by_dotted_key(cfg, key, value)
    if task:
        cfg["task"] = task
    tasks = load_tasks()
    if cfg["task"] not in tasks:
        raise KeyError(f"Unknown task '{cfg['task']}'. Available: {sorted(tasks)}")
    cfg["task_def"] = tasks[cfg["task"]]
    return finalize_experiment_name(cfg)


def finalize_experiment_name(cfg: dict) -> dict:
    """Default experiment name: baseline_<task>[_harmonized]. Call after all overrides."""
    if not cfg.get("experiment"):
        suffix = "_harmonized" if (cfg["data"].get("harmonize_real") or {}).get("enabled") else ""
        cfg["experiment"] = f"baseline_{cfg['task']}{suffix}"
    return cfg


def save_config(cfg: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
