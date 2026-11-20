"""ALL hardware-specific logic lives in this module.

The rest of the pipeline only talks to a `DeviceContext`, so training/evaluation code is
hardware-agnostic. Auto-detection order: CUDA GPU -> Google TPU (only if torch_xla is
installed) -> Apple MPS -> CPU. `torch_xla` is optional and imported lazily.

TPU (PyTorch/XLA) specifics handled here:
  * device from xm.xla_device()
  * data loaders wrapped in MpDeviceLoader (moves batches to the device asynchronously)
  * optimizer steps via xm.optimizer_step()
  * static batch shapes: callers ask `ctx.force_drop_last` for training loaders
Mixed precision is used only where supported (see `_choose_amp`).
"""
from __future__ import annotations

import contextlib
import importlib.util
import logging
import platform
from dataclasses import dataclass
from typing import Any, Iterable

import torch

log = logging.getLogger(__name__)

VALID_DEVICES = ("auto", "cuda", "tpu", "mps", "cpu")
VALID_AMP = ("auto", "off", "bf16", "fp16")


def xla_installed() -> bool:
    """True if torch_xla can be imported (checked without importing it)."""
    return importlib.util.find_spec("torch_xla") is not None


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def _try_tpu():
    """Return (xm, pl, device) if a real TPU is reachable through torch_xla, else None."""
    if not xla_installed():
        return None
    try:
        import torch_xla.core.xla_model as xm  # noqa: WPS433 (optional import by design)
        import torch_xla.distributed.parallel_loader as pl

        dev = xm.xla_device()
        hw = xm.xla_device_hw(dev) if hasattr(xm, "xla_device_hw") else "TPU"
        if hw != "TPU":  # torch_xla can also expose CPU/GPU backends; only accept TPU
            return None
        return xm, pl, dev
    except Exception as exc:  # pragma: no cover - depends on hardware
        log.warning("torch_xla is installed but no TPU could be initialised: %s", exc)
        return None


@dataclass
class DeviceContext:
    kind: str                      # "cuda" | "tpu" | "mps" | "cpu"
    device: torch.device
    amp_dtype: torch.dtype | None  # None = full fp32
    scaler: Any = None             # torch.amp.GradScaler for CUDA fp16 only
    _xm: Any = None                # torch_xla.core.xla_model (TPU only)
    _pl: Any = None                # torch_xla.distributed.parallel_loader (TPU only)

    # ------------------------------------------------------------------ properties
    @property
    def is_tpu(self) -> bool:
        return self.kind == "tpu"

    @property
    def pin_memory(self) -> bool:
        return self.kind == "cuda"

    @property
    def force_drop_last(self) -> bool:
        """TPU needs static batch shapes for training to avoid recompilation."""
        return self.is_tpu

    # ------------------------------------------------------------------ helpers
    def seed(self, seed: int) -> None:
        """Device-specific RNG seeding (CPU/CUDA seeding is done in src.seed)."""
        if self.is_tpu:
            self._xm.set_rng_state(seed)

    def autocast(self):
        """Mixed-precision context, or a no-op where AMP is not used."""
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        device_type = "xla" if self.is_tpu else self.kind
        return torch.autocast(device_type=device_type, dtype=self.amp_dtype)

    def wrap_loader(self, loader: Iterable) -> Iterable:
        """On TPU, wrap with MpDeviceLoader (batches arrive already on the XLA device)."""
        if self.is_tpu:
            return self._pl.MpDeviceLoader(loader, self.device)
        return loader

    def to_device(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.is_tpu:  # MpDeviceLoader already moved the batch
            return tensors
        non_blocking = self.kind == "cuda"
        return tuple(t.to(self.device, non_blocking=non_blocking) for t in tensors)

    def backward_and_step(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
        """loss.backward() + optimizer step, using the right mechanism for each device."""
        if self.scaler is not None:  # CUDA fp16
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.is_tpu:
                self._xm.optimizer_step(optimizer)  # also marks the XLA step
            else:
                optimizer.step()

    def describe(self) -> dict:
        """Information saved with every run (config + metrics)."""
        info = {
            "device": self.kind,
            "torch_device": str(self.device),
            "amp_dtype": str(self.amp_dtype).replace("torch.", "") if self.amp_dtype else "fp32",
            "grad_scaler": self.scaler is not None,
            "torch_version": str(torch.__version__),  # TorchVersion -> plain str (YAML-safe)
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
        if self.kind == "cuda":
            info["gpu_name"] = torch.cuda.get_device_name(self.device)
            info["gpu_count"] = torch.cuda.device_count()
            info["cuda_version"] = str(torch.version.cuda)
            info["cudnn_version"] = int(torch.backends.cudnn.version() or 0)
        if self.is_tpu:
            import torch_xla  # noqa: WPS433

            info["torch_xla_version"] = str(getattr(torch_xla, "__version__", "unknown"))
        return info


def _choose_amp(kind: str, amp: str) -> tuple[torch.dtype | None, bool]:
    """Return (autocast dtype or None, needs_grad_scaler)."""
    if amp not in VALID_AMP:
        raise ValueError(f"train.amp must be one of {VALID_AMP}, got {amp!r}")
    if amp == "off":
        return None, False
    if kind == "cuda":
        bf16_ok = torch.cuda.is_bf16_supported()
        if amp == "bf16" or (amp == "auto" and bf16_ok):
            return torch.bfloat16, False
        return torch.float16, True  # fp16 needs loss scaling
    if kind == "tpu":
        return (torch.bfloat16, False) if amp in ("auto", "bf16") else (None, False)
    # MPS autocast is still incomplete for many ops and CPU autocast rarely speeds up
    # CNN training, so "auto" keeps fp32 there. Explicit requests are honoured on CPU only.
    if kind == "cpu" and amp == "bf16":
        return torch.bfloat16, False
    if amp in ("bf16", "fp16"):
        log.warning("AMP %s requested but not supported on %s; using fp32.", amp, kind)
    return None, False


def resolve_device(requested: str = "auto", amp: str = "auto") -> DeviceContext:
    """Pick the device (auto or explicit) and build its DeviceContext."""
    requested = (requested or "auto").lower()
    if requested not in VALID_DEVICES:
        raise ValueError(f"device must be one of {VALID_DEVICES}, got {requested!r}")

    tpu = None
    if requested == "auto":
        if torch.cuda.is_available():
            kind = "cuda"
        elif (tpu := _try_tpu()) is not None:
            kind = "tpu"
        elif _mps_available():
            kind = "mps"
        else:
            kind = "cpu"
    else:
        kind = requested
        if kind == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available.")
        if kind == "mps" and not _mps_available():
            raise RuntimeError("--device mps requested but Apple MPS is not available.")
        if kind == "tpu":
            tpu = _try_tpu()
            if tpu is None:
                raise RuntimeError(
                    "--device tpu requested but no TPU found (is torch_xla installed? "
                    "see requirements-tpu.txt)."
                )

    amp_dtype, needs_scaler = _choose_amp(kind, amp)
    if kind == "tpu":
        xm, pl, dev = tpu
        ctx = DeviceContext(kind, dev, amp_dtype, None, xm, pl)
    else:
        dev = torch.device("cuda:0" if kind == "cuda" else kind)
        scaler = torch.amp.GradScaler("cuda") if needs_scaler else None
        ctx = DeviceContext(kind, dev, amp_dtype, scaler)
    log.info("Using device: %s", ctx.describe())
    return ctx


def state_dict_to_cpu(model: torch.nn.Module) -> dict:
    """Device-independent copy of a model's weights (works for CUDA, MPS, XLA)."""
    return {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
