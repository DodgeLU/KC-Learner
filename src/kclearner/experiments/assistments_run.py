"""ASSISTments prepared-data loading and TRAIN/VALID execution.

EdNet's ``dev5000_users.txt`` and ``solving_id`` columns are not read.
The model's bundle-run field still receives the ASSISTments dense
``group_id``, because that is the existing bundle key on ``DKTRow``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kclearner.data.adapters.assistments2017 import (
    AssistmentsModelRow,
    oov_behavior,
    rows_for_neural_state,
    to_streaming_rows,
)
from kclearner.data.assistments_prepare import (
    LOGICAL_NAME,
    SPLIT_FILES,
    VOCAB_NAME,
    read_model_rows,
)
from kclearner.data.logical_identity import (
    DatasetLogicalIdentity,
    SplitLogicalIdentity,
    dataset_logical_hash,
)
from kclearner.experiments.config import NEURAL_MODELS, RunConfig
from kclearner.experiments.environment import environment_report, execution_device
from kclearner.experiments.factory import create_model
from kclearner.models.dkt import DKTRow


def rows_for_execution(rows: tuple[AssistmentsModelRow, ...] | list[AssistmentsModelRow], family: str):
    """Apply the family OOV rule, then map onto the common model rows.

    IRT and AR-KT keep OOV rows in the sequence. DKT and DKVMN drop
    state-transparent OOV rows. Neither path invents a shared OOV policy.
    """

    policy = oov_behavior(family)
    if family in ("irt", "ar_kt"):
        if policy.state_transparent:
            raise ValueError(f"{family} was marked state-transparent")
        return to_streaming_rows(tuple(rows), family=family)
    if family not in NEURAL_MODELS:
        raise ValueError(f"unknown model family {family!r}")
    if not policy.state_transparent:
        raise ValueError(f"{family} is missing the neural OOV rule")
    visible = rows_for_neural_state(tuple(rows), family)
    use_kc = family.endswith("qc")
    return [
        DKTRow(
            student_idx=row.student_idx,
            problem_idx=row.problem_idx,
            correct=row.correct,
            solving_id=row.group_id,
            tag_idxs=(row.skill_idx,) if use_kc else (),
            interaction_id=row.interaction_id,
        )
        for row in visible
    ]


def verify_assistments_prepared(
    dataset_dir: str | Path,
    protocol: dict,
    *,
    read_test: bool = True,
) -> dict[str, Any]:
    """Check a prepared directory against its protocol. Does not score rows."""

    from kclearner.data.ednet_corrected import file_sha256
    from kclearner.data.logical_identity import hash_ordered_ids
    from kclearner.experiments.runner import IdentityError

    dataset = Path(dataset_dir)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    logical_document = json.loads((dataset / LOGICAL_NAME).read_text(encoding="utf-8"))
    vocabulary = json.loads((dataset / VOCAB_NAME).read_text(encoding="utf-8"))
    expected = protocol["dataset"]
    if manifest.get("protocol_id") != protocol["protocol_id"]:
        raise IdentityError("protocol_id mismatch")
    if manifest.get("dataset_id") != expected["dataset_id"]:
        raise IdentityError("dataset_id mismatch")
    if manifest.get("preprocessing_version") != expected["preprocessing_version"]:
        raise IdentityError("preprocessing version mismatch")
    identity = logical_document.get("dataset_logical_identity")
    if not isinstance(identity, dict):
        raise IdentityError("logical identity is missing")
    rebuilt = DatasetLogicalIdentity(
        dataset_id=str(identity["dataset_id"]),
        protocol_id=str(identity["protocol_id"]),
        preprocessing_version=str(identity["preprocessing_version"]),
        cohort_hash=identity.get("cohort_hash"),
        splits=tuple(
            SplitLogicalIdentity(
                str(item["split_id"]),
                str(item["logical_hash"]),
                int(item["row_count"]),
            )
            for item in identity["splits"]
        ),
        item_vocabulary_hash=identity.get("item_vocabulary_hash"),
        kc_vocabulary_hash=identity.get("kc_vocabulary_hash"),
    )
    if dataset_logical_hash(rebuilt) != logical_document.get("dataset_logical_hash"):
        raise IdentityError("stored dataset logical hash does not match its identity")
    if identity.get("protocol_id") != protocol["protocol_id"]:
        raise IdentityError("logical protocol_id mismatch")
    if identity.get("dataset_id") != expected["dataset_id"]:
        raise IdentityError("logical dataset_id mismatch")
    counts = {
        item["split_id"]: int(item["row_count"])
        for item in identity["splits"]
    }
    recorded_counts = manifest.get("row_counts") or {}
    for split, count in counts.items():
        if int(recorded_counts.get(split, -1)) != count:
            raise IdentityError(f"manifest {split} row count mismatch")
    protocol_counts = expected.get("row_counts") or {}
    for split, count in protocol_counts.items():
        if split == "eligible":
            continue
        if counts.get(split) != int(count):
            raise IdentityError(f"{split} row count {counts.get(split)} != {count}")
    if "eligible" in protocol_counts:
        eligible = sum(counts.values())
        if eligible != int(protocol_counts["eligible"]):
            raise IdentityError(f"eligible row count {eligible} != {protocol_counts['eligible']}")
    items = tuple(str(item) for item in vocabulary["item_ids"])
    kcs = tuple(str(item) for item in vocabulary["kc_ids"])
    learners = tuple(str(item) for item in vocabulary["learner_ids"])
    if hash_ordered_ids(items) != identity.get("item_vocabulary_hash"):
        raise IdentityError("item vocabulary hash mismatch")
    if hash_ordered_ids(kcs) != identity.get("kc_vocabulary_hash"):
        raise IdentityError("KC vocabulary hash mismatch")
    if hash_ordered_ids(learners) != identity.get("cohort_hash"):
        raise IdentityError("learner vocabulary hash mismatch")
    for name, expected_count in (
        ("item_count", len(items)),
        ("kc_count", len(kcs)),
        ("learner_count", len(learners)),
    ):
        recorded = protocol.get("vocabulary", {}).get(name)
        if recorded is not None and int(recorded) != expected_count:
            raise IdentityError(f"{name} {expected_count} != {recorded}")
    recorded_hashes = protocol.get("vocabulary", {})
    for name, actual in (
        ("item_ids_sha256", identity.get("item_vocabulary_hash")),
        ("kc_ids_sha256", identity.get("kc_vocabulary_hash")),
        ("learner_ids_sha256", identity.get("cohort_hash")),
    ):
        if name in recorded_hashes and recorded_hashes[name] != actual:
            raise IdentityError(f"{name} mismatch")
    files = manifest.get("files_sha256") or {}
    for split, filename in SPLIT_FILES.items():
        if split == "test" and not read_test:
            continue
        path = dataset / filename
        if not path.is_file():
            raise IdentityError(f"missing {filename}")
        if files.get(split) != file_sha256(path):
            raise IdentityError(f"{filename} byte hash mismatch")
    if (dataset / "dev5000_users.txt").exists():
        raise IdentityError("ASSISTments prepared data must not use dev5000_users.txt")
    return {
        "learner_count": len(learners),
        "item_count": len(items),
        "kc_count": len(kcs),
        "cohort_hash": identity.get("cohort_hash"),
        "item_ids_sha256": identity.get("item_vocabulary_hash"),
        "kc_ids_sha256": identity.get("kc_vocabulary_hash"),
    }


def _write_assistments_metadata(
    run_dir: Path,
    config: RunConfig,
    *,
    phase: str,
    extra: dict[str, Any],
    identity: dict[str, Any],
) -> None:
    from kclearner import __version__
    from kclearner.experiments.runner import NOT_FOR_PUBLICATION, _git_commit

    requested = str(extra.get("requested_device", "cpu"))
    devices = execution_device(config.model, requested)
    resolved = extra.get("resolved_device", devices["resolved_device"])
    env = environment_report(
        requested,
        str(resolved),
        execution_mode=devices["execution_mode"],
    )
    payload = {
        "run_id": run_dir.name,
        "kclearner_version": __version__,
        "git_commit": _git_commit(),
        "dataset_id": config.protocol["dataset"]["dataset_id"],
        "protocol_id": config.protocol["protocol_id"],
        "cohort_hash": identity.get("cohort_hash"),
        "item_ids_sha256": identity.get("item_ids_sha256"),
        "kc_ids_sha256": identity.get("kc_ids_sha256"),
        "model": config.model,
        "model_family": config.protocol["models"][config.model]["family"],
        "kc_aware": config.protocol["models"][config.model]["kc_aware"],
        "seed": config.seed,
        "deterministic": config.deterministic,
        "phase": phase,
        "config_hash": config.config_hash(),
        "publication_protocol": False,
        "publication": False,
        "notice": NOT_FOR_PUBLICATION,
        "environment": env,
    }
    payload.update(extra)
    (run_dir / "metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def dry_run_assistments(
    config: RunConfig,
    *,
    dataset_dir: str | Path,
    runs_dir: str | Path,
    construct_model: bool = True,
    requested_device: str = "cpu",
) -> dict[str, Any]:
    """Validate ASSISTments identity and construct the model. Does not train."""

    from kclearner.experiments.runner import NOT_FOR_PUBLICATION

    sizes = verify_assistments_prepared(dataset_dir, config.protocol)
    model_ready = False
    dependency_error = None
    if construct_model:
        try:
            create_model(config.model, config, sizes)
            model_ready = True
        except Exception as exc:
            dependency_error = f"{type(exc).__name__}: {exc}"
            if config.model not in NEURAL_MODELS:
                raise
    run_dir = Path(runs_dir) / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    devices = execution_device(config.model, requested_device)
    if config.model in NEURAL_MODELS:
        devices = {
            "requested_device": requested_device,
            "resolved_device": "not_executed",
            "execution_mode": "neural",
        }
    report = {
        "status": "dry_run",
        "publication_protocol": False,
        "publication": False,
        "notice": NOT_FOR_PUBLICATION,
        "requested_device": devices["requested_device"],
        "resolved_device": devices["resolved_device"],
        "run_id": config.run_id,
        "config_hash": config.config_hash(),
        "model_ready": model_ready,
        "dependency_error": dependency_error,
        "environment": environment_report(
            devices["requested_device"],
            devices["resolved_device"],
            execution_mode=devices["execution_mode"],
        ),
        "output_dir": str(run_dir),
        "dataset_id": config.protocol["dataset"]["dataset_id"],
        "protocol_id": config.protocol["protocol_id"],
    }
    _write_assistments_metadata(
        run_dir,
        config,
        phase="dry_run",
        extra=report,
        identity=sizes,
    )
    (run_dir / "config.json").write_text(config.canonical_json(), encoding="utf-8")
    (run_dir / "dry_run.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def smoke_assistments(
    config: RunConfig,
    *,
    dataset_dir: str | Path,
    runs_dir: str | Path,
    max_rows: int = 24,
    requested_device: str = "cpu",
) -> dict[str, Any]:
    """Smoke TRAIN rows only. Nominal TEST is not read."""

    from kclearner.experiments.runner import (
        FormalRunBlocked,
        IdentityError,
        NOT_FOR_PUBLICATION,
        masked_metrics,
    )

    if max_rows > 64:
        raise FormalRunBlocked("smoke row cap is 64")
    preflight = dry_run_assistments(
        config,
        dataset_dir=dataset_dir,
        runs_dir=runs_dir,
        construct_model=True,
        requested_device=requested_device,
    )
    if not preflight["model_ready"]:
        raise IdentityError(preflight["dependency_error"] or "model was not constructed")
    rows = read_model_rows(dataset_dir, "train")[:max_rows]
    if any(row.split != "train" for row in rows):
        raise IdentityError("smoke received a non-TRAIN row")
    run_dir = Path(runs_dir) / f"{config.run_id}_smoke"
    run_dir.mkdir(parents=True, exist_ok=True)
    sizes = verify_assistments_prepared(dataset_dir, config.protocol)
    if config.model in ("irt", "ar_kt"):
        model = create_model(config.model, config, sizes)
        traced = model.run_streaming(rows_for_execution(rows, config.model), phase="train")
        metrics = masked_metrics(
            [row.correct for row in traced.rows],
            [row.probability for row in traced.rows],
            [row.metric_mask for row in rows],
        )
        model.save_state(run_dir / "checkpoint.npz")
    else:
        from kclearner.experiments.neural_training import fit_neural

        mapped = rows_for_execution(rows, config.model)
        split_at = max(1, len(mapped) // 2)
        fit_neural(
            create_model(config.model, config, sizes),
            mapped[:split_at],
            mapped[split_at:] or mapped[:1],
            config.protocol["models"][config.model]["recipe"],
            seed=config.seed,
            requested_device=requested_device,
            run_dir=run_dir,
            config_hash=config.config_hash(),
            dataset_id=config.protocol["dataset"]["dataset_id"],
            cohort_hash=str(sizes["cohort_hash"]),
            vocab_hashes={
                "item_ids_sha256": sizes["item_ids_sha256"],
                "kc_ids_sha256": sizes["kc_ids_sha256"],
            },
            max_epochs=1,
            authorize_formal=False,
            publication_protocol=False,
        )
        metrics = None
    _write_assistments_metadata(
        run_dir,
        config,
        phase="smoke",
        extra={
            "publication_protocol": False,
            "publication": False,
            "notice": NOT_FOR_PUBLICATION,
            "rows": len(rows),
            "requested_device": requested_device,
            "resolved_device": execution_device(config.model, requested_device)["resolved_device"],
        },
        identity=sizes,
    )
    return {
        "run_id": run_dir.name,
        "status": "smoke",
        "publication_protocol": False,
        "publication": False,
        "rows": len(rows),
        "metrics": metrics,
    }


def run_assistments_formal(
    config: RunConfig,
    *,
    dataset_dir: str | Path,
    runs_dir: str | Path,
    requested_device: str,
) -> dict[str, Any]:
    """TRAIN then VALID. Nominal TEST rows are not opened."""

    from kclearner.experiments.runner import IdentityError, masked_metrics

    recipe = config.protocol["models"][config.model].get("recipe")
    if not isinstance(recipe, dict) or not recipe:
        raise IdentityError(
            f"{config.model} has no frozen ASSISTments recipe; "
            "formal TRAIN/VALID is blocked"
        )
    if config.model == "ar_kt" and recipe.get("residual_update") != "assistments_skill_mean":
        raise IdentityError("ASSISTments AR-KT residual_update is not assistments_skill_mean")
    sizes = verify_assistments_prepared(dataset_dir, config.protocol)
    train_rows = read_model_rows(dataset_dir, "train")
    valid_rows = read_model_rows(dataset_dir, "valid")
    if any(row.split == "test" for row in (*train_rows, *valid_rows)):
        raise IdentityError("formal TRAIN/VALID received a TEST row")
    model = create_model(config.model, config, sizes)
    run_dir = Path(runs_dir) / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if config.model in NEURAL_MODELS:
        from kclearner.experiments.neural_training import fit_neural

        result = fit_neural(
            model,
            rows_for_execution(train_rows, config.model),
            rows_for_execution(valid_rows, config.model),
            recipe,
            seed=config.seed,
            requested_device=requested_device,
            run_dir=run_dir,
            config_hash=config.config_hash(),
            dataset_id=config.protocol["dataset"]["dataset_id"],
            cohort_hash=str(sizes["cohort_hash"]),
            vocab_hashes={
                "item_ids_sha256": sizes["item_ids_sha256"],
                "kc_ids_sha256": sizes["kc_ids_sha256"],
            },
            authorize_formal=True,
            publication_protocol=bool(config.publication_protocol),
        )
        from kclearner.experiments.runner import _freeze_neural

        _freeze_neural(config, dataset_dir, run_dir, result)
        return result
    model.run_streaming(rows_for_execution(train_rows, config.model), phase="train")
    state_path = run_dir / "state_train_only.npz"
    model.save_state(state_path)
    traced = model.run_streaming(rows_for_execution(valid_rows, config.model), phase="eval")
    masks = [row.metric_mask for row in valid_rows]
    headline = masked_metrics(
        [row.correct for row in traced.rows],
        [row.probability for row in traced.rows],
        masks,
    )
    global_metrics = masked_metrics(
        [row.correct for row in traced.rows],
        [row.p_global for row in traced.rows],
        masks,
    )
    from kclearner.experiments.freeze import record_formal_freeze
    from kclearner.models.state import FORMAT_ID

    devices = execution_device(config.model, requested_device)
    frozen = record_formal_freeze(
        run_dir,
        dataset_dir,
        config,
        state_path,
        headline,
        global_metrics,
        FORMAT_ID,
        requested_device=devices["requested_device"],
        resolved_device=devices["resolved_device"],
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({
            **headline,
            "publication_protocol": True,
            "publication": True,
        })
        + "\n",
        encoding="utf-8",
    )
    return {"best_epoch": 0, "metrics": headline, "freeze_id": frozen["freeze_id"]}
