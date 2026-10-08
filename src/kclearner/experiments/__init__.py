"""Corrected-EdNet experiment orchestration."""

from kclearner.experiments.config import resolve_run
from kclearner.experiments.factory import create_model

__all__ = ["create_model", "resolve_run"]
