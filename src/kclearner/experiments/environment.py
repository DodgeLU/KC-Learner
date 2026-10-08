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
    resolved = resolved_device
    if resolved is None:
        resolved = "cuda" if cuda else "cpu"
    return {
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
