"""AR-KT: the IRT anchor plus the legacy KC residual.

Formal EdNet reference: ``use_residual=True``, ``eta_r=0.20``,
``use_time=False``. Headline probability is ``p_art``. Theta and b
still move on ``p_global``. EVAL freezes b and keeps updating theta
and ``r_post``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from kclearner.models.state import (
    copy_state_arrays,
    read_legacy_checkpoint,
    read_native_state,
    save_native_state,
)
from kclearner.models.streaming import (
    B_L2,
    B_LR,
    FORMAL_ETA_R,
    THETA_L2,
    THETA_LR,
    LegacyEdNetStreamer,
    StreamingResult,
    StreamingRow,
)


class ARKTModel:
    """Deterministic AR-KT in the formal EdNet hyperparameter set."""

    _model_kind = "ar_kt"

    def __init__(
        self,
        n_students: int,
        n_problems: int,
        n_skills: int,
        *,
        eta_r: float = FORMAL_ETA_R,
        theta_lr: float = THETA_LR,
        b_lr: float = B_LR,
        theta_l2: float = THETA_L2,
        b_l2: float = B_L2,
        learn_b: bool = True,
        freeze_global_in_eval: bool = True,
        residual_update: str = "ednet_token",
    ) -> None:
        self.eta_r = float(eta_r)
        self._core = LegacyEdNetStreamer(
            n_students,
            n_problems,
            n_skills,
            use_residual=True,
            eta_r=self.eta_r,
            headline_art=True,
            theta_lr=theta_lr,
            b_lr=b_lr,
            theta_l2=theta_l2,
            b_l2=b_l2,
            learn_b=learn_b,
            freeze_global_in_eval=freeze_global_in_eval,
            residual_update=residual_update,
        )

    @property
    def theta(self) -> np.ndarray:
        return self._core.theta

    @property
    def b(self) -> np.ndarray:
        return self._core.b

    @property
    def r_post(self) -> np.ndarray:
        return self._core.r_post

    @property
    def r_seen(self) -> np.ndarray:
        return self._core.r_seen

    @property
    def last_r_update_ts(self) -> np.ndarray:
        return self._core.last_r_update_ts

    @property
    def residual_update(self) -> str:
        return self._core.residual_update

    def run_streaming(
        self,
        rows: Sequence[StreamingRow],
        phase: str = "train",
    ) -> StreamingResult:
        return self._core.run_streaming(list(rows), phase)

    def save_state(self, path: str | Path) -> None:
        """Write a native state file. The model mathematics are unchanged."""

        save_native_state(path, self._core, self._model_kind)

    @classmethod
    def load_state(cls, path: str | Path) -> "ARKTModel":
        """Restore a model from a native state file."""

        payload = read_native_state(path)
        if payload["model_kind"] != cls._model_kind:
            raise ValueError(
                f"state file is {payload['model_kind']}, not {cls._model_kind}"
            )
        if not payload["use_residual"]:
            raise ValueError("AR-KT state file has use_residual=False")
        model = cls(
            payload["n_students"],
            payload["n_problems"],
            payload["n_skills"],
            eta_r=payload["eta_r"],
            theta_lr=payload["theta_lr"],
            b_lr=payload["b_lr"],
            theta_l2=payload["theta_l2"],
            b_l2=payload["b_l2"],
            learn_b=payload["learn_b"],
            freeze_global_in_eval=payload["freeze_global_in_eval"],
            residual_update=payload.get("residual_update", "ednet_token"),
        )
        copy_state_arrays(model._core, payload)
        return model

    @classmethod
    def load_legacy_checkpoint(cls, path: str | Path) -> "ARKTModel":
        """Restore an EdNet or ASSISTments MAIN_V1 AR-KT ``.npz`` snapshot.

        The ASSISTments checkpoint selects the single-skill group mean.
        An EdNet checkpoint keeps the token residual.
        """

        payload = read_legacy_checkpoint(path, cls._model_kind)
        model = cls(
            payload["n_students"],
            payload["n_problems"],
            payload["n_skills"],
            eta_r=payload["eta_r"],
            theta_lr=payload["theta_lr"],
            b_lr=payload["b_lr"],
            theta_l2=payload["theta_l2"],
            b_l2=payload["b_l2"],
            learn_b=payload["learn_b"],
            freeze_global_in_eval=payload["freeze_global_in_eval"],
            residual_update=payload.get("residual_update", "ednet_token"),
        )
        copy_state_arrays(model._core, payload)
        return model
