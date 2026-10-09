"""Versioned experiment freeze identity.

``freeze_id`` is the SHA-256 of the canonical identity object.
Timestamps, paths, hostnames, and CPU/CUDA stay in provenance and do
not enter that hash.

``evaluation_semantics_v1`` is the compatibility gate. A later
evaluator that changes OOV handling, metric definitions, or the
psychometric VALID-replay rule must use a new semantics version.
Package version and git state are recorded and do not by themselves
reject a freeze.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Mapping

from kclearner.data.publish import make_staging_directory
from kclearner.experiments.evaluation_semantics import EVALUATION_SEMANTICS

IDENTITY_VERSION = "experiment_freeze_identity_v1"
MANIFEST_NAME = "freeze_manifest.json"

# Historical VALID-replay absolute tolerances.
# EdNet: project_c/train/_smoke_test_replay_ednet_{irt,ar_kt}_dev5000.py
# ASSISTments: project_c/train/_smoke_test_replay_assist2017_{irt,ar_kt}_main_v1.py
# The ASSISTments scripts document 1e-9 because the replay reloads npz.
# EdNet's executed constant is 1e-12.
REPLAY_TOLERANCE = {
    "ednet_kt1_corrected_v1": 1e-12,
    "assistments2017": 1e-9,
}

_IDENTITY_KEYS = (
    "identity_version",
    "evaluation_semantics",
    "dataset_id",
    "protocol_id",
    "preprocessing_version",
    "dataset_logical_hash",
    "cohort_hash",
    "item_vocabulary_hash",
    "kc_vocabulary_hash",
    "model",
    "family",
    "kc_aware",
    "recipe_hash",
    "seed",
    "seed_role",
    "checkpoint_sha256",
    "checkpoint_format",
    "phases_completed",
    "valid_nll",
    "valid_brier",
    "valid_auc",
    "valid_global_nll",
    "valid_global_brier",
    "valid_global_auc",
    "valid_replay_tolerance",
    "valid_replay_required",
)


class FreezeError(RuntimeError):
    """A freeze manifest cannot be written or accepted."""


def replay_tolerance(dataset_id: str) -> float:
    """Return the historical psychometric VALID-replay tolerance."""

    try:
        return REPLAY_TOLERANCE[dataset_id]
    except KeyError as exc:
        raise FreezeError(f"no VALID-replay tolerance for dataset {dataset_id!r}") from exc


def recipe_hash(recipe: Mapping[str, object]) -> str:
    """SHA-256 of one model recipe. Dataset fields are not included."""

    payload = json.dumps(recipe, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_freeze_json(identity: Mapping[str, object]) -> str:
    """Compact JSON in fixed key order. Extra keys are rejected."""

    unknown = set(identity) - set(_IDENTITY_KEYS)
    if unknown:
        raise FreezeError(f"freeze identity has non-identity fields {sorted(unknown)}")
    missing = [key for key in _IDENTITY_KEYS if key not in identity]
    if missing:
        raise FreezeError(f"freeze identity is missing {missing}")
    body = ",".join(
        f"{json.dumps(key)}:{_token(identity[key])}" for key in _IDENTITY_KEYS
    )
    return "{" + body + "}"


def freeze_id_for(identity: Mapping[str, object]) -> str:
    """SHA-256 of :func:`canonical_freeze_json`."""

    return hashlib.sha256(canonical_freeze_json(identity).encode("utf-8")).hexdigest()


def metrics_within_tolerance(
    actual: Mapping[str, float],
    expected: Mapping[str, float],
    tolerance: float,
) -> bool:
    """Historical absolute gate. NaN makes the comparison false."""

    for key in ("nll", "brier", "auc"):
        delta = abs(float(actual[key]) - float(expected[key]))
        if not (delta <= float(tolerance)):
            return False
    return True


def make_identity(
    *,
    dataset_id: str,
    protocol_id: str,
    preprocessing_version: str,
    dataset_logical_hash: str,
    cohort_hash: str,
    item_vocabulary_hash: str,
    kc_vocabulary_hash: str,
    model: str,
    family: str,
    kc_aware: bool,
    recipe_hash_value: str,
    seed: int,
    seed_role: str,
    checkpoint_sha256: str,
    checkpoint_format: str,
    valid_metrics: Mapping[str, float] | None,
    valid_global_metrics: Mapping[str, float] | None,
    replay_required: bool,
    tolerance: float | None,
) -> dict[str, object]:
    """Assemble the identity object. Provenance is not accepted here."""

    return {
        "identity_version": IDENTITY_VERSION,
        "evaluation_semantics": EVALUATION_SEMANTICS,
        "dataset_id": dataset_id,
        "protocol_id": protocol_id,
        "preprocessing_version": preprocessing_version,
        "dataset_logical_hash": dataset_logical_hash,
        "cohort_hash": cohort_hash,
        "item_vocabulary_hash": item_vocabulary_hash,
        "kc_vocabulary_hash": kc_vocabulary_hash,
        "model": model,
        "family": family,
        "kc_aware": bool(kc_aware),
        "recipe_hash": recipe_hash_value,
        "seed": int(seed),
        "seed_role": seed_role,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_format": checkpoint_format,
        "phases_completed": ["train", "valid"],
        "valid_nll": _metric(valid_metrics, "nll"),
        "valid_brier": _metric(valid_metrics, "brier"),
        "valid_auc": _metric(valid_metrics, "auc"),
        "valid_global_nll": _metric(valid_global_metrics, "nll"),
        "valid_global_brier": _metric(valid_global_metrics, "brier"),
        "valid_global_auc": _metric(valid_global_metrics, "auc"),
        "valid_replay_tolerance": None if tolerance is None else float(tolerance),
        "valid_replay_required": bool(replay_required),
    }


def _metric(metrics: Mapping[str, float] | None, key: str) -> float | None:
    if metrics is None:
        return None
    value = metrics.get(key)
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise FreezeError(f"VALID {key} is not finite")
    return number


def write_freeze_manifest(
    run_dir: str | Path,
    identity: Mapping[str, object],
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Atomically write a complete freeze manifest.

    An existing manifest with the same ``freeze_id`` is left in place.
    A different identity is not overwritten.
    """

    if list(identity.get("phases_completed") or []) != ["train", "valid"]:
        raise FreezeError("freeze requires completed TRAIN and VALID")
    target = Path(run_dir)
    target.mkdir(parents=True, exist_ok=True)
    destination = target / MANIFEST_NAME
    document = {
        "status": "complete",
        "identity": {key: identity[key] for key in _IDENTITY_KEYS},
        "freeze_id": freeze_id_for(identity),
        "provenance": dict(provenance),
    }
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing.get("status") == "complete" and existing.get("freeze_id") == document["freeze_id"]:
            return existing
        raise FreezeError(
            f"{destination} already holds freeze {existing.get('freeze_id')}; "
            "refusing to replace it"
        )
    staging = make_staging_directory(destination)
    path = staging / MANIFEST_NAME
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    try:
        path.rename(destination)
    except OSError as exc:
        if destination.is_file():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if (
                existing.get("status") == "complete"
                and existing.get("freeze_id") == document["freeze_id"]
            ):
                return existing
        raise FreezeError(f"could not publish {destination}") from exc
    finally:
        if staging.exists():
            import shutil
            shutil.rmtree(staging, ignore_errors=True)
    _record_freeze_id(target, document["freeze_id"])
    return document


def load_freeze_manifest(path: str | Path) -> dict[str, object]:
    """Load a complete manifest and recompute its freeze id."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FreezeError(f"freeze manifest does not exist: {path}") from exc
    if payload.get("status") != "complete":
        raise FreezeError("freeze manifest is not complete")
    identity = payload.get("identity")
    if not isinstance(identity, dict):
        raise FreezeError("freeze manifest has no identity")
    expected = freeze_id_for(identity)
    if payload.get("freeze_id") != expected:
        raise FreezeError("freeze_id does not match the canonical identity")
    if identity.get("evaluation_semantics") != EVALUATION_SEMANTICS:
        raise FreezeError(
            "evaluation semantics "
            f"{identity.get('evaluation_semantics')!r} != {EVALUATION_SEMANTICS}"
        )
    return payload


def record_formal_freeze(
    run_dir: str | Path,
    dataset_dir: str | Path,
    config,
    checkpoint: str | Path,
    valid_metrics: Mapping[str, float] | None,
    valid_global_metrics: Mapping[str, float] | None,
    checkpoint_format: str,
    requested_device: str | None = None,
    resolved_device: str | None = None,
) -> dict[str, object]:
    """Write the freeze for one completed TRAIN/VALID run."""

    from kclearner.data.ednet_corrected import file_sha256

    logical = json.loads(
        (Path(dataset_dir) / "logical_identity.json").read_text(encoding="utf-8")
    )
    body = logical["dataset_logical_identity"]
    family = config.protocol["models"][config.model]
    replay = config.model in ("irt", "ar_kt")
    identity = make_identity(
        dataset_id=str(body["dataset_id"]),
        protocol_id=str(body["protocol_id"]),
        preprocessing_version=str(body["preprocessing_version"]),
        dataset_logical_hash=str(logical["dataset_logical_hash"]),
        cohort_hash=str(body["cohort_hash"]),
        item_vocabulary_hash=str(body["item_vocabulary_hash"]),
        kc_vocabulary_hash=str(body["kc_vocabulary_hash"]),
        model=config.model,
        family=str(family["family"]),
        kc_aware=bool(family["kc_aware"]),
        recipe_hash_value=recipe_hash(family["recipe"]),
        seed=int(config.seed),
        seed_role=str(config.to_dict()["seed_role"]),
        checkpoint_sha256=file_sha256(checkpoint),
        checkpoint_format=checkpoint_format,
        valid_metrics=valid_metrics,
        valid_global_metrics=valid_global_metrics if replay else None,
        replay_required=replay,
        tolerance=replay_tolerance(str(body["dataset_id"])) if replay else None,
    )
    provenance = environment_provenance()
    provenance["checkpoint_filename"] = Path(checkpoint).name
    if requested_device is not None:
        provenance["requested_device"] = requested_device
    if resolved_device is not None:
        provenance["resolved_device"] = resolved_device
    if config.model in ("irt", "ar_kt"):
        provenance["execution_mode"] = "psychometric_cpu"
        provenance["resolved_device"] = "cpu"
    return write_freeze_manifest(run_dir, identity, provenance)


def environment_provenance() -> dict[str, object]:
    """Non-identity facts. Dirty git state does not block a freeze."""

    from kclearner import __version__

    commit, dirty = _git_facts()
    return {
        "kclearner_version": __version__,
        "git_commit": commit,
        "git_dirty": dirty,
        "python": sys.version.split()[0],
    }


def _record_freeze_id(run_dir: Path, freeze_id: str) -> None:
    path = run_dir / "metadata.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {"role": "formal_train_valid"}
    payload["freeze_id"] = freeze_id
    payload["phases_completed"] = ["train", "valid"]
    payload["publication_protocol"] = True
    payload["publication"] = True
    payload["role"] = payload.get("role") or "formal_train_valid"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_facts() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None, None
    if commit.returncode != 0 or dirty.returncode != 0:
        return None, None
    text = commit.stdout.strip()
    return (text or None), bool(dirty.stdout.strip())


def _token(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FreezeError("freeze identity rejects non-finite numbers")
        return format(value, ".17g")
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_token(item) for item in value) + "]"
    raise FreezeError(f"freeze identity cannot encode {type(value).__name__}")
