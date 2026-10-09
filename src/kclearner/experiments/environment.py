"""Runtime environment captured into a run directory."""

from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any


def _package(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def environment_report(
    requested_device: str | None = None,
    resolved_device: str | None = None,
    *,
    execution_mode: str | None = None,
) -> dict[str, Any]:
    torch_version = None
    cuda = False
    cuda_version = None
    gpu_name = None
    try:
        import torch

        torch_version = torch.__version__
        cuda = bool(torch.cuda.is_available())
        cuda_version = getattr(torch.version, "cuda", None)
        if cuda:
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        torch_version = None
    # An omitted resolved device is not evidence that CUDA ran.
    resolved = "cpu" if resolved_device is None else resolved_device
    report = {
        "requested_device": requested_device,
        "resolved_device": resolved,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": _package("numpy"),
        "torch": torch_version,
        "pykt_toolkit": _package("pykt-toolkit"),
        "cuda_available": cuda,
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
        "device": resolved,
    }
    if execution_mode is not None:
        report["execution_mode"] = execution_mode
    return report


def execution_device(model: str, requested: str) -> dict[str, str]:
    """Requested device and the device that actually applies.

    IRT and AR-KT always execute on CPU. A CUDA request is recorded
    and is not treated as a neural device. Neural ``auto`` stays
    unresolved until training selects CPU or CUDA.
    """

    name = str(requested)
    if model in ("irt", "ar_kt"):
        return {
            "requested_device": name,
            "resolved_device": "cpu",
            "execution_mode": "psychometric_cpu",
        }
    if name == "auto":
        return {
            "requested_device": "auto",
            "resolved_device": "auto",
            "execution_mode": "neural",
        }
    if name in ("cpu", "cuda"):
        return {
            "requested_device": name,
            "resolved_device": name,
            "execution_mode": "neural",
        }
    raise ValueError(f"device must be auto, cpu, or cuda, got {requested!r}")
