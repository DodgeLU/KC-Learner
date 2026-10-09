"""Non-formal TRAIN then VALID for one external execution adapter.

The runner selects split roles and applies ``metric_mask``. The adapter
owns traversal, bundle predict-before-update, and state. TEST-role rows
are not passed to the adapter. No freeze manifest is written.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from kclearner.data.canonical_dataset import CanonicalDataset, load_canonical_dataset
from kclearner.data.schema import Interaction
from kclearner.experiments.adapter_contract import (
    NEURAL_BUNDLE_CONTRACT,
    PSYCHOMETRIC_STREAMING_CONTRACT,
    AdapterProvenance,
    adapter_provenance,
    load_adapter,
)
from kclearner.experiments.runner import masked_metrics


class ExternalExecutionError(ValueError):
    """The external non-formal path cannot run this adapter."""


def execute_external_directory(
    dataset_dir: str | Path,
    adapter_spec: str,
    config: Mapping[str, Any],
    *,
    runs_dir: str | Path | None = None,
) -> dict[str, object]:
    """Load a canonical directory and run TRAIN then VALID.

    ``config`` must include the adapter's declared OOV state semantics.
    Publication and TEST stay closed.
    """

    dataset = load_canonical_dataset(dataset_dir)
    adapter = load_adapter(adapter_spec, config)
    provenance = adapter_provenance(adapter, config)
    _validate_before_train(adapter, dataset)
    train_rows = _rows_for_role(dataset, "train")
    valid_rows = _rows_for_role(dataset, "valid")
    _reject_test_rows(train_rows, valid_rows, dataset)
    train_output, valid_output = _run_family(adapter, train_rows, valid_rows)
    train_aligned, train_metrics = _align_and_score(train_rows, train_output)
    valid_aligned, valid_metrics = _align_and_score(valid_rows, valid_output)
    destination = Path(runs_dir) if runs_dir is not None else Path("runs")
    output_dir = destination / (
        f"external_{dataset.dataset_contract_hash}_{provenance.provenance_hash}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "adapter_state.json"
    _save_family_state(adapter, state_path)
    document = {
        "status": "external_nonformal",
        "publication": False,
        "publication_protocol": False,
        "freeze": False,
        "formal_test": False,
        "formal_freeze_eligible": provenance.formal_freeze_eligible,
        "model_execution": True,
        "dataset_logical_hash": dataset.dataset_logical_hash,
        "dataset_contract_hash": dataset.dataset_contract_hash,
        "adapter_contract_version": provenance.identity["adapter_contract_version"],
        "execution_semantics_id": provenance.identity["execution_semantics_id"],
        "traversal_id": adapter.traversal_id,
        "oov_representation_semantics_id": dataset.oov_representation_semantics_id,
        "oov_state_semantics_id": provenance.identity["oov_state_semantics_id"],
        "config_hash": provenance.identity["config_hash"],
        "executed_roles": ["train", "valid"],
        "provenance_hash": provenance.provenance_hash,
        "provenance_identity": provenance.identity,
        "descriptive_provenance": {
            "module_path": provenance.module_path,
            "source_mode": provenance.source_mode,
        },
        "train": {"metrics": train_metrics, "rows": train_aligned},
        "valid": {"metrics": valid_metrics, "rows": valid_aligned},
    }
    (output_dir / "external_execution.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if (output_dir / "freeze_manifest.json").exists():
        raise ExternalExecutionError("external execution must not write a freeze manifest")
    return {
        "status": document["status"],
        "publication": False,
        "freeze": False,
        "formal_test": False,
        "formal_freeze_eligible": provenance.formal_freeze_eligible,
        "model_execution": True,
        "dataset_logical_hash": dataset.dataset_logical_hash,
        "dataset_contract_hash": dataset.dataset_contract_hash,
        "provenance_hash": provenance.provenance_hash,
        "traversal_id": adapter.traversal_id,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "output_dir": str(output_dir),
    }


def _validate_before_train(adapter: Any, dataset: CanonicalDataset) -> None:
    try:
        adapter.validate_dataset_contract(dataset)
    except Exception as exc:
        raise ExternalExecutionError(
            "dataset OOV representation is incompatible with this adapter; "
            f"TRAIN was not started: {exc}"
        ) from exc


def _run_family(adapter: Any, train_rows: Sequence[Interaction], valid_rows: Sequence[Interaction]):
    if adapter.contract_version == PSYCHOMETRIC_STREAMING_CONTRACT:
        return (
            _execute_phase(adapter, train_rows, "train"),
            _execute_phase(adapter, valid_rows, "valid"),
        )
    if adapter.contract_version == NEURAL_BUNDLE_CONTRACT:
        return (
            _as_output(adapter.train(tuple(train_rows)), "train"),
            _as_output(adapter.replay(tuple(valid_rows)), "replay"),
        )
    raise ExternalExecutionError(
        f"unsupported contract {adapter.contract_version!r}"
    )


def _save_family_state(adapter: Any, path: Path) -> None:
    if adapter.contract_version == PSYCHOMETRIC_STREAMING_CONTRACT:
        adapter.save_state(path)
        return
    adapter.save_checkpoint(path)


def _as_output(value: Any, label: str) -> tuple:
    if value is None:
        raise ExternalExecutionError(f"neural adapter {label} returned no probabilities")
    return tuple(value)


def _execute_phase(adapter: Any, rows: Sequence[Interaction], phase: str):
    try:
        output = adapter.execute_phase(tuple(rows), phase=phase)
    except TypeError as exc:
        raise ExternalExecutionError(
            f"adapter execute_phase rejected phase {phase!r}"
        ) from exc
    return tuple(output)


def _align_and_score(rows: Sequence[Interaction], output: Sequence[tuple[str, float]]):
    if len(output) != len(rows):
        raise ExternalExecutionError(
            f"adapter returned {len(output)} probabilities for {len(rows)} rows"
        )
    aligned = []
    correct = []
    probability = []
    mask = []
    for row, item in zip(rows, output):
        if not isinstance(item, tuple) or len(item) != 2:
            raise ExternalExecutionError("adapter probability rows must be (interaction_id, probability)")
        interaction_id, score = item
        if interaction_id != row.interaction_id:
            raise ExternalExecutionError(
                f"adapter probability for {interaction_id!r} is not aligned "
                f"to {row.interaction_id!r}"
            )
        aligned.append({
            "interaction_id": row.interaction_id,
            "bundle_id": row.bundle_id,
            "correct": row.correct,
            "probability": float(score),
            "metric_mask": row.metric_mask,
        })
        correct.append(row.correct)
        probability.append(float(score))
        mask.append(row.metric_mask)
    metrics = masked_metrics(correct, probability, mask)
    if metrics is None:
        metrics = {"nll": None, "brier": None, "auc": None, "n": 0}
    return aligned, metrics


def _rows_for_role(dataset: CanonicalDataset, role: str) -> tuple[Interaction, ...]:
    matches = [
        split_id
        for split_id, split_role in dataset.split_roles
        if split_role == role
    ]
    if len(matches) != 1:
        raise ExternalExecutionError(
            f"external execution requires exactly one split with role {role!r}"
        )
    split_id = matches[0]
    return tuple(row for row in dataset.interactions if row.split == split_id)


def _reject_test_rows(
    train_rows: Sequence[Interaction],
    valid_rows: Sequence[Interaction],
    dataset: CanonicalDataset,
) -> None:
    test_ids = {
        row.interaction_id
        for split_id, role in dataset.split_roles
        if role == "test"
        for row in dataset.interactions
        if row.split == split_id
    }
    seen = {row.interaction_id for row in train_rows} | {row.interaction_id for row in valid_rows}
    leaked = seen & test_ids
    if leaked:
        raise ExternalExecutionError(f"TEST-role rows entered execution: {sorted(leaked)}")


def provenance_for(adapter: Any, config: Mapping[str, Any]) -> AdapterProvenance:
    """Inspect eligibility without running a dataset."""

    return adapter_provenance(adapter, config)
