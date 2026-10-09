"""Non-formal bundle execution for an external canonical directory.

The reference runners are not called. Rows are selected by the declared
split role, then grouped with :func:`replay_bundle_pre_state`, which
reads ``bundle_id``. This module does not write a freeze manifest and
does not open TEST.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from kclearner.data.canonical_dataset import CanonicalDataset, load_canonical_dataset
from kclearner.sequence.contract import BundleStepRecord, replay_bundle_pre_state

SMOKE_ROW_CAP = 24
EXECUTED_ROLE = "train"


class CanonicalRunError(ValueError):
    """The canonical directory cannot take this non-formal execution."""


@dataclass(frozen=True)
class CanonicalBundleExecution:
    """Bundle steps for one declared role. Not a publication artifact."""

    status: str
    dataset_logical_hash: str
    dataset_contract_hash: str
    executed_role: str
    executed_split_id: str
    rows: tuple[BundleStepRecord, ...]
    truncated: bool

    @property
    def bundle_count(self) -> int:
        count = 0
        previous: tuple[str, str, int] | None = None
        for row in self.rows:
            key = (row.learner_id, row.bundle_id, row.pre_bundle_update_count)
            if key != previous:
                count += 1
                previous = key
        return count


def execute_canonical(
    dataset: CanonicalDataset,
    *,
    dry_run: bool = False,
    smoke: bool = False,
) -> CanonicalBundleExecution:
    """Execute the train-role split. Other roles stay on the dataset only."""

    if dry_run and smoke:
        raise CanonicalRunError("dry-run and smoke are mutually exclusive")
    split_id = _split_id_for_role(dataset, EXECUTED_ROLE)
    selected = tuple(row for row in dataset.interactions if row.split == split_id)
    steps = replay_bundle_pre_state(selected)
    status = "bundle_execution"
    truncated = False
    if smoke:
        steps, truncated = _cap_complete_bundles(steps, SMOKE_ROW_CAP)
        status = "smoke"
    elif dry_run:
        status = "dry_run"
    return CanonicalBundleExecution(
        status=status,
        dataset_logical_hash=dataset.dataset_logical_hash,
        dataset_contract_hash=dataset.dataset_contract_hash,
        executed_role=EXECUTED_ROLE,
        executed_split_id=split_id,
        rows=steps,
        truncated=truncated,
    )


def execute_canonical_directory(
    dataset_dir: str | Path,
    *,
    runs_dir: str | Path | None = None,
    dry_run: bool = False,
    smoke: bool = False,
) -> dict[str, object]:
    """Load a canonical directory, execute its train role, and write a report.

    The output directory is ``canonical_<dataset_contract_hash>``, the full
    64-character contract hash. The report is not ``freeze_manifest.json``.
    Formal TEST is not started.
    """

    dataset = load_canonical_dataset(dataset_dir)
    execution = execute_canonical(dataset, dry_run=dry_run, smoke=smoke)
    destination = Path(runs_dir) if runs_dir is not None else Path("runs")
    output_dir = destination / f"canonical_{execution.dataset_contract_hash}"
    output_dir.mkdir(parents=True, exist_ok=True)
    document = execution_document(execution)
    (output_dir / "bundle_execution.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if (output_dir / "freeze_manifest.json").exists():
        raise CanonicalRunError("canonical execution must not write freeze_manifest.json")
    summary = {
        "status": execution.status,
        "publication": False,
        "publication_protocol": False,
        "freeze": False,
        "formal_test": False,
        "grouping": "bundle_id",
        "model_execution": False,
        "inspection_traversal": "learner_id_then_interaction_order",
        "executed_role": execution.executed_role,
        "executed_split_id": execution.executed_split_id,
        "dataset_logical_hash": execution.dataset_logical_hash,
        "dataset_contract_hash": execution.dataset_contract_hash,
        "row_count": len(execution.rows),
        "bundle_count": execution.bundle_count,
        "truncated": execution.truncated,
        "output_dir": str(output_dir),
    }
    return summary


def execution_document(execution: CanonicalBundleExecution) -> dict[str, object]:
    """JSON body of the inspection report. Grouping is ``bundle_id`` only.

    ``model_execution`` and ``inspection_traversal`` describe this replay.
    They are not inputs to dataset logical identity, dataset contract
    identity, or an experiment freeze identity.
    """

    return {
        "status": execution.status,
        "publication": False,
        "publication_protocol": False,
        "freeze": False,
        "formal_test": False,
        "grouping": "bundle_id",
        "model_execution": False,
        "inspection_traversal": "learner_id_then_interaction_order",
        "executed_role": execution.executed_role,
        "executed_split_id": execution.executed_split_id,
        "dataset_logical_hash": execution.dataset_logical_hash,
        "dataset_contract_hash": execution.dataset_contract_hash,
        "row_count": len(execution.rows),
        "bundle_count": execution.bundle_count,
        "truncated": execution.truncated,
        "rows": [
            {
                "interaction_id": row.interaction_id,
                "learner_id": row.learner_id,
                "bundle_id": row.bundle_id,
                "order": row.order,
                "pre_bundle_update_count": row.pre_bundle_update_count,
            }
            for row in execution.rows
        ],
    }


def _split_id_for_role(dataset: CanonicalDataset, role: str) -> str:
    matches = [
        split_id
        for split_id, split_role in dataset.split_roles
        if split_role == role
    ]
    if len(matches) != 1:
        raise CanonicalRunError(
            "current workflow execution requires exactly one split "
            f"whose role is {role!r}; found {matches}"
        )
    return matches[0]


def _cap_complete_bundles(
    steps: tuple[BundleStepRecord, ...],
    max_rows: int,
) -> tuple[tuple[BundleStepRecord, ...], bool]:
    """Keep whole bundles until the next one would pass ``max_rows``.

    The first bundle is kept even when it alone exceeds the cap, so a
    bundle is never split.
    """

    kept: list[BundleStepRecord] = []
    index = 0
    while index < len(steps):
        end = index + 1
        while end < len(steps) and _same_bundle(steps[index], steps[end]):
            end += 1
        group = steps[index:end]
        if kept and len(kept) + len(group) > max_rows:
            return tuple(kept), True
        kept.extend(group)
        index = end
        if len(kept) >= max_rows and index < len(steps):
            return tuple(kept), True
    return tuple(kept), False


def _same_bundle(left: BundleStepRecord, right: BundleStepRecord) -> bool:
    return (
        left.learner_id == right.learner_id
        and left.bundle_id == right.bundle_id
        and left.pre_bundle_update_count == right.pre_bundle_update_count
    )
