"""IRT anchor: EdNet streaming with the residual layer off.

``z_global = theta - b`` and ``p_global = sigmoid(z_global)``.
TRAIN updates theta and b. EVAL freezes b and still updates theta.
The walker is :class:`kclearner.models.streaming.LegacyEdNetStreamer`.
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
    THETA_L2,
    THETA_LR,
    LegacyEdNetStreamer,
    StreamingResult,
    StreamingRow,
)


class IRTModel:
    """Deterministic IRT anchor in the formal EdNet hyperparameter set."""

    _model_kind = "irt"

    def __init__(
        self,
        n_students: int,
        n_problems: int,
        n_skills: int = 1,
        *,
        theta_lr: float = THETA_LR,
        b_lr: float = B_LR,
        theta_l2: float = THETA_L2,
        b_l2: float = B_L2,
        learn_b: bool = True,
        freeze_global_in_eval: bool = True,
    ) -> None:
        self._core = LegacyEdNetStreamer(
            n_students,
            n_problems,
            n_skills,
            use_residual=False,
            eta_r=0.0,
            headline_art=False,
            theta_lr=theta_lr,
            b_lr=b_lr,
            theta_l2=theta_l2,
            b_l2=b_l2,
            learn_b=learn_b,
            freeze_global_in_eval=freeze_global_in_eval,
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
    def load_state(cls, path: str | Path) -> "IRTModel":
        """Restore a model from a native state file."""

        payload = read_native_state(path)
        if payload["model_kind"] != cls._model_kind:
            raise ValueError(
                f"state file is {payload['model_kind']}, not {cls._model_kind}"
            )
        if payload["use_residual"] or payload["eta_r"] != 0.0:
            raise ValueError("IRT state file has residual updates enabled")
        model = cls(
            payload["n_students"],
            payload["n_problems"],
            payload["n_skills"],
            theta_lr=payload["theta_lr"],
            b_lr=payload["b_lr"],
            theta_l2=payload["theta_l2"],
            b_l2=payload["b_l2"],
            learn_b=payload["learn_b"],
            freeze_global_in_eval=payload["freeze_global_in_eval"],
        )
        copy_state_arrays(model._core, payload)
        return model

    @classmethod
    def load_legacy_checkpoint(cls, path: str | Path) -> "IRTModel":
        """Restore from a ``project_c`` EdNet IRT ``.npz`` snapshot.

        Residual arrays are not in that file. They stay at the unseen
        initialization, which is the IRT anchor's persisted state.
        """

        payload = read_legacy_checkpoint(path, cls._model_kind)
        model = cls(
            payload["n_students"],
            payload["n_problems"],
            payload["n_skills"],
            theta_lr=payload["theta_lr"],
            b_lr=payload["b_lr"],
            theta_l2=payload["theta_l2"],
            b_l2=payload["b_l2"],
            learn_b=payload["learn_b"],
            freeze_global_in_eval=payload["freeze_global_in_eval"],
        )
        copy_state_arrays(model._core, payload)
        return model
