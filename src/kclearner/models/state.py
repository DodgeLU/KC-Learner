"""Deterministic IRT / AR-KT state files.

Native files use ``format_id = kclearner_ednet_streaming_state_v1``.
They store every array the streaming walker needs to continue, plus
the hyperparameters that affect later updates.

``load_legacy_checkpoint`` reads a ``project_c`` ``state_train_only.npz``.
That layout is not the native format. IRT files contain ``theta`` and
``b`` only. AR-KT files also contain ``r_post``, ``r_seen``, and
``last_r_update_ts``. Time decay is rejected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

FORMAT_ID = "kclearner_ednet_streaming_state_v1"
LEGACY_IRT_MODEL_ID = "ednet_irt_anchor_v1"
LEGACY_AR_KT_MODEL_ID = "ednet_ar_kt_v0_3"
ASSISTMENTS_IRT_MODEL_ID = "assist2017_irt_anchor_v1"
ASSISTMENTS_AR_KT_MODEL_ID = "assist2017_ar_kt_v1"
_ACCEPTED_LEGACY_IDS = {
    "irt": frozenset({"", LEGACY_IRT_MODEL_ID, ASSISTMENTS_IRT_MODEL_ID}),
    "ar_kt": frozenset({"", LEGACY_AR_KT_MODEL_ID, ASSISTMENTS_AR_KT_MODEL_ID}),
}

_NATIVE_ARRAYS = (
    "theta",
    "b",
    "r_post",
    "r_seen",
    "last_r_update_ts",
)
_NATIVE_FLOATS = ("eta_r", "theta_lr", "b_lr", "theta_l2", "b_l2")
_NATIVE_INTS = ("n_students", "n_problems", "n_skills")
_NATIVE_BOOLS = ("use_residual", "learn_b", "freeze_global_in_eval")


def _item(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.item()
    return value


def _require(saved: np.lib.npyio.NpzFile, key: str) -> Any:
    if key not in saved.files:
        raise ValueError(f"state file is missing {key}")
    return saved[key]


def save_native_state(path: str | Path, core: Any, model_kind: str) -> None:
    """Write the native state file. ``core`` is a streaming walker."""

    if model_kind not in ("irt", "ar_kt"):
        raise ValueError(f"model_kind must be 'irt' or 'ar_kt', got {model_kind!r}")
    np.savez_compressed(
        str(path),
        format_id=np.str_(FORMAT_ID),
        model_kind=np.str_(model_kind),
        theta=np.asarray(core.theta, dtype=np.float64),
        b=np.asarray(core.b, dtype=np.float64),
        r_post=np.asarray(core.r_post, dtype=np.float64),
        r_seen=np.asarray(core.r_seen, dtype=np.bool_),
        last_r_update_ts=np.asarray(core.last_r_update_ts, dtype=np.int64),
        n_students=np.int64(core.n_students),
        n_problems=np.int64(core.n_problems),
        n_skills=np.int64(core.n_skills),
        eta_r=np.float64(core.eta_r),
        theta_lr=np.float64(core.theta_lr),
        b_lr=np.float64(core.b_lr),
        theta_l2=np.float64(core.theta_l2),
        b_l2=np.float64(core.b_l2),
        use_residual=np.bool_(core.use_residual),
        learn_b=np.bool_(core.learn_b),
        freeze_global_in_eval=np.bool_(core.freeze_global_in_eval),
        residual_update=np.str_(getattr(core, "residual_update", "ednet_token")),
    )


def read_native_state(path: str | Path) -> dict[str, Any]:
    """Load a native state file into plain arrays and scalars."""

    with np.load(str(path), allow_pickle=False) as saved:
        format_id = str(_item(_require(saved, "format_id")))
        if format_id != FORMAT_ID:
            raise ValueError(
                f"unsupported state format {format_id!r}; expected {FORMAT_ID}"
            )
        payload: dict[str, Any] = {
            "format_id": format_id,
            "model_kind": str(_item(_require(saved, "model_kind"))),
        }
        for key in _NATIVE_ARRAYS:
            payload[key] = np.array(_require(saved, key), copy=True)
        for key in _NATIVE_INTS:
            payload[key] = int(_item(_require(saved, key)))
        for key in _NATIVE_FLOATS:
            payload[key] = float(_item(_require(saved, key)))
        for key in _NATIVE_BOOLS:
            payload[key] = bool(_item(_require(saved, key)))
        if "residual_update" in saved.files:
            payload["residual_update"] = str(_item(saved["residual_update"]))
        else:
            payload["residual_update"] = "ednet_token"
    _check_array_shapes(payload)
    return payload


def read_legacy_checkpoint(path: str | Path, model_kind: str) -> dict[str, Any]:
    """Load a ``project_c`` EdNet IRT or AR-KT ``.npz`` snapshot.

    Residual arrays are required only for AR-KT. IRT snapshots leave
    them absent; the caller keeps the zero / unseen initialization.
    """

    if model_kind not in ("irt", "ar_kt"):
        raise ValueError(f"model_kind must be 'irt' or 'ar_kt', got {model_kind!r}")
    with np.load(str(path), allow_pickle=False) as saved:
        keys = set(saved.files)
        required = {"theta", "b", "n_students", "n_problems", "n_skills"}
        if model_kind == "ar_kt":
            required |= {"r_post", "r_seen", "last_r_update_ts"}
        missing = sorted(required - keys)
        if missing:
            raise ValueError(f"legacy checkpoint is missing {missing}")
        if "use_time" in keys and bool(_item(saved["use_time"])):
            raise ValueError("legacy checkpoint has use_time=True; EdNet runtime does not")
        use_residual = (
            bool(_item(saved["use_residual"])) if "use_residual" in keys else model_kind == "ar_kt"
        )
        if model_kind == "irt" and use_residual:
            raise ValueError("IRT checkpoint has use_residual=True")
        if model_kind == "ar_kt" and not use_residual:
            raise ValueError("AR-KT checkpoint has use_residual=False")
        eta_r = float(_item(saved["eta_r"])) if "eta_r" in keys else 0.0
        if model_kind == "irt" and eta_r != 0.0:
            raise ValueError(f"IRT checkpoint eta_r={eta_r}; expected 0")
        model_id = str(_item(saved["model_id"])) if "model_id" in keys else ""
        if model_id not in _ACCEPTED_LEGACY_IDS[model_kind]:
            raise ValueError(
                f"legacy model_id {model_id!r} is not a {model_kind} checkpoint"
            )
        payload: dict[str, Any] = {
            "model_kind": model_kind,
            "model_id": model_id,
            "theta": np.array(saved["theta"], dtype=np.float64, copy=True),
            "b": np.array(saved["b"], dtype=np.float64, copy=True),
            "n_students": int(_item(saved["n_students"])),
            "n_problems": int(_item(saved["n_problems"])),
            "n_skills": int(_item(saved["n_skills"])),
            "eta_r": eta_r,
            "use_residual": use_residual,
            "theta_lr": float(_item(saved["theta_lr"])) if "theta_lr" in keys else 0.05,
            "b_lr": float(_item(saved["b_lr"])) if "b_lr" in keys else 0.02,
            "theta_l2": float(_item(saved["theta_l2"])) if "theta_l2" in keys else 1e-4,
            "b_l2": float(_item(saved["b_l2"])) if "b_l2" in keys else 1e-4,
            "learn_b": bool(_item(saved["learn_b"])) if "learn_b" in keys else True,
            "freeze_global_in_eval": (
                bool(_item(saved["freeze_global_in_eval"]))
                if "freeze_global_in_eval" in keys
                else True
            ),
            "has_residual_arrays": model_kind == "ar_kt",
            "residual_update": (
                "assistments_skill_mean"
                if model_id == ASSISTMENTS_AR_KT_MODEL_ID
                else "ednet_token"
            ),
        }
        if model_kind == "ar_kt":
            payload["r_post"] = np.array(saved["r_post"], dtype=np.float64, copy=True)
            payload["r_seen"] = np.array(saved["r_seen"], dtype=np.bool_, copy=True)
            payload["last_r_update_ts"] = np.array(
                saved["last_r_update_ts"], dtype=np.int64, copy=True
            )
    _check_array_shapes(payload)
    return payload


def copy_state_arrays(core: Any, payload: dict[str, Any]) -> None:
    """Copy persisted arrays onto an already constructed walker."""

    _overwrite(core.theta, payload["theta"], "theta")
    _overwrite(core.b, payload["b"], "b")
    if "r_post" in payload:
        _overwrite(core.r_post, payload["r_post"], "r_post")
        _overwrite(core.r_seen, payload["r_seen"], "r_seen")
        _overwrite(
            core.last_r_update_ts,
            payload["last_r_update_ts"],
            "last_r_update_ts",
        )


def _check_array_shapes(payload: dict[str, Any]) -> None:
    n_students = int(payload["n_students"])
    n_problems = int(payload["n_problems"])
    n_skills = int(payload["n_skills"])
    theta = payload["theta"]
    b = payload["b"]
    if tuple(theta.shape) != (n_students,):
        raise ValueError(f"theta shape {theta.shape} != ({n_students},)")
    if tuple(b.shape) != (n_problems,):
        raise ValueError(f"b shape {b.shape} != ({n_problems},)")
    expected = (n_students, n_skills)
    for key in ("r_post", "r_seen", "last_r_update_ts"):
        if key not in payload:
            continue
        if tuple(payload[key].shape) != expected:
            raise ValueError(f"{key} shape {payload[key].shape} != {expected}")


def _overwrite(dest: np.ndarray, src: np.ndarray, name: str) -> None:
    if dest.shape != src.shape:
        raise ValueError(f"{name} shape {src.shape} != model shape {dest.shape}")
    dest[...] = src
