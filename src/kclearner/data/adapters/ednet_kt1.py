"""Fixture-scale EdNet-KT1 interaction adapter.

Reads a small interaction CSV plus question metadata and returns
canonical interactions, filter statistics, and source-quality
diagnostics. It does not open the dev5000 cohort or any ``project_c``
artifact.

Source-faithful order:

* sequence order is the eligible source-row order
* ``solving_id`` identifies the bundle ``(learner_id, solving_id)``
* it is not used to sort the learner history
* timestamp is metadata; an inversion is reported, not repaired

``questions.csv`` ``bundle_id`` stays content identity. It is not the
interaction bundle.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterable, Mapping

from kclearner.data.adapters.base import ensure_bundle_id
from kclearner.data.adapters.ednet_metadata import (
    EdNetQuestionMetadata,
    load_ednet_question_metadata,
)
from kclearner.data.adapters.ednet_diagnostics import (
    EdNetDiagnostics,
    build_ednet_diagnostics,
)
from kclearner.data.schema import Interaction
from kclearner.data.validation import DatasetValidator

_ANSWER_LABELS = frozenset({"a", "b", "c", "d"})
_KT1_REQUIRED = (
    "user_id",
    "timestamp",
    "solving_id",
    "question_id",
    "user_answer",
)
_INVALID_ANSWER_REASONS = frozenset(
    {
        "user_answer_minus_one",
        "user_answer_empty",
        "user_answer_unexpected",
    }
)
_REASON_ORDER = (
    "user_answer_minus_one",
    "user_answer_empty",
    "user_answer_unexpected",
    "missing_metadata",
    "missing_solving_id",
    "empty_question_id",
    "empty_user_id",
    "invalid_timestamp",
    "missing_correct_answer",
)


class EdNetAdapterError(ValueError):
    """The fixture CSV cannot be converted to canonical interactions."""


@dataclass(frozen=True)
class DropReasonCount:
    reason: str
    count: int


@dataclass(frozen=True)
class EdNetFilterStats:
    """Visible record of rows removed before they become interactions.

    Invalid answers are counted, not rewritten as ``correct=0``.
    ``dropped_reasons`` lists every non-zero reason in a fixed order.
    """

    raw_rows: int
    eligible_rows: int
    dropped_invalid_user_answer: int
    dropped_missing_metadata: int
    dropped_reasons: tuple[DropReasonCount, ...]


@dataclass(frozen=True)
class EdNetKT1LoadResult:
    interactions: tuple[Interaction, ...]
    stats: EdNetFilterStats
    diagnostics: EdNetDiagnostics


@dataclass(frozen=True)
class _KT1Row:
    learner_id: str
    item_id: str
    solving_id: int | None
    user_answer: str
    timestamp: int | None
    source_index: int
    drop_reason: str | None


def load_ednet_kt1_interactions(
    interactions: str | Path | IO[str],
    questions: str | Path | IO[str],
) -> EdNetKT1LoadResult:
    """Join synthetic KT1 rows to question metadata.

    Rows with ``user_answer="-1"``, an empty answer, an unexpected
    answer, or no question metadata are excluded and counted. They are
    not emitted as interactions. Invalid tags still raise
    :class:`~kclearner.data.adapters.ednet_metadata.EdNetTagParseError`
    while the question file is read.
    """

    catalog = {
        row.question_id: row
        for row in load_ednet_question_metadata(questions)
    }
    raw_rows = _read_kt1(interactions)
    eligible, reason_counts = _select_eligible(raw_rows, catalog)
    built = _to_interactions(eligible, catalog)
    stats = _stats(len(raw_rows), len(built), reason_counts)
    checked = DatasetValidator().validate(
        built,
        check_temporal=False,
        check_bundle_contiguity=False,
    )
    if not checked.ok:
        codes = ", ".join(issue.code for issue in checked.issues)
        raise EdNetAdapterError(f"canonical interactions failed validation: {codes}")
    return EdNetKT1LoadResult(
        interactions=built,
        stats=stats,
        diagnostics=build_ednet_diagnostics(built, stats),
    )


_RAW_USER_REQUIRED = (
    "timestamp",
    "solving_id",
    "question_id",
    "user_answer",
)


def load_raw_user_kt1(
    path: str | Path | IO[str],
    user_id: str,
    questions: str | Path | IO[str] | Mapping[str, EdNetQuestionMetadata],
) -> EdNetKT1LoadResult:
    """Read one original per-user KT1 file in CSV row order.

    The raw file has no ``user_id`` column. ``user_id`` is the file
    stem. ``source_index`` is the 0-based data-row index, including
    rows that are later filtered. Nothing in this function sorts the
    file.
    """

    if not isinstance(user_id, str) or user_id.strip() == "":
        raise EdNetAdapterError("user_id is required for a raw KT1 file")
    if isinstance(questions, Mapping):
        catalog = questions
    else:
        catalog = {
            row.question_id: row
            for row in load_ednet_question_metadata(questions)
        }
    raw_rows = _read_raw_user(path, user_id.strip())
    eligible, reason_counts = _select_eligible(raw_rows, catalog)
    built = _to_interactions(eligible, catalog)
    stats = _stats(len(raw_rows), len(built), reason_counts)
    checked = DatasetValidator().validate(
        built,
        check_temporal=False,
        check_bundle_contiguity=False,
    )
    if not checked.ok:
        codes = ", ".join(issue.code for issue in checked.issues)
        raise EdNetAdapterError(f"canonical interactions failed validation: {codes}")
    return EdNetKT1LoadResult(
        interactions=built,
        stats=stats,
        diagnostics=build_ednet_diagnostics(built, stats),
    )


def _read_raw_user(source: str | Path | IO[str], user_id: str) -> tuple[_KT1Row, ...]:
    if isinstance(source, (str, Path)):
        with Path(source).open("r", encoding="utf-8", newline="") as handle:
            return _parse_raw_user(handle, user_id)
    return _parse_raw_user(source, user_id)


def _parse_raw_user(handle: Iterable[str], user_id: str) -> tuple[_KT1Row, ...]:
    reader = csv.DictReader(handle)
    if reader.fieldnames is None:
        raise EdNetAdapterError("raw KT1 CSV has no header")
    fields = set(reader.fieldnames)
    missing = [name for name in _RAW_USER_REQUIRED if name not in fields]
    if missing:
        raise EdNetAdapterError(
            f"raw KT1 CSV is missing columns {missing}; got {reader.fieldnames}"
        )
    rows: list[_KT1Row] = []
    for source_index, row in enumerate(reader):
        payload = dict(row)
        payload["user_id"] = user_id
        rows.append(_parse_kt1_row(payload, source_index))
    return tuple(rows)


def _read_kt1(source: str | Path | IO[str]) -> tuple[_KT1Row, ...]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        with path.open("r", encoding="utf-8", newline="") as handle:
            return _parse_kt1(handle)
    return _parse_kt1(source)


def _parse_kt1(handle: Iterable[str]) -> tuple[_KT1Row, ...]:
    reader = csv.DictReader(handle)
    if reader.fieldnames is None:
        raise EdNetAdapterError("KT1 CSV has no header")
    fields = set(reader.fieldnames)
    missing = [name for name in _KT1_REQUIRED if name not in fields]
    if missing:
        raise EdNetAdapterError(
            f"KT1 CSV is missing columns {missing}; got {reader.fieldnames}"
        )
    rows: list[_KT1Row] = []
    for source_index, row in enumerate(reader):
        rows.append(_parse_kt1_row(row, source_index))
    return tuple(rows)


def _parse_kt1_row(row: Mapping[str, str | None], source_index: int) -> _KT1Row:
    learner_id = _cell(row, "user_id")
    item_id = _cell(row, "question_id")
    solving_text = _cell(row, "solving_id")
    user_answer = _cell(row, "user_answer")
    timestamp_text = _cell(row, "timestamp")
    drop_reason: str | None = None
    solving_id: int | None = None
    timestamp: int | None = None
    if learner_id == "":
        drop_reason = "empty_user_id"
    elif item_id == "":
        drop_reason = "empty_question_id"
    elif not solving_text.isdigit():
        drop_reason = "missing_solving_id"
    elif timestamp_text != "" and not timestamp_text.isdigit():
        drop_reason = "invalid_timestamp"
    else:
        solving_id = int(solving_text)
        timestamp = int(timestamp_text) if timestamp_text != "" else None
        if user_answer == "-1":
            drop_reason = "user_answer_minus_one"
        elif user_answer == "":
            drop_reason = "user_answer_empty"
        elif user_answer not in _ANSWER_LABELS:
            drop_reason = "user_answer_unexpected"
    return _KT1Row(
        learner_id=learner_id,
        item_id=item_id,
        solving_id=solving_id,
        user_answer=user_answer,
        timestamp=timestamp,
        source_index=source_index,
        drop_reason=drop_reason,
    )


def _select_eligible(
    raw_rows: tuple[_KT1Row, ...],
    catalog: Mapping[str, EdNetQuestionMetadata],
) -> tuple[tuple[_KT1Row, ...], dict[str, int]]:
    counts: dict[str, int] = defaultdict(int)
    eligible: list[_KT1Row] = []
    for row in raw_rows:
        reason = row.drop_reason
        if reason is None and row.item_id not in catalog:
            reason = "missing_metadata"
        elif reason is None:
            meta = catalog[row.item_id]
            if meta.correct_answer not in _ANSWER_LABELS:
                reason = "missing_correct_answer"
        if reason is None:
            eligible.append(row)
        else:
            counts[reason] += 1
    return tuple(eligible), counts


def _to_interactions(
    raw_rows: tuple[_KT1Row, ...],
    catalog: Mapping[str, EdNetQuestionMetadata],
) -> tuple[Interaction, ...]:
    """Emit eligible rows in source-record order.

    ``order`` is the eligible appearance index within one learner.
    ``solving_id`` and timestamp do not change that sequence.
    """

    next_order: dict[str, int] = defaultdict(int)
    interactions: list[Interaction] = []
    for row in sorted(raw_rows, key=lambda item: item.source_index):
        meta = catalog[row.item_id]
        assert row.solving_id is not None
        assert meta.correct_answer is not None
        learner_id = row.learner_id
        order = next_order[learner_id]
        next_order[learner_id] += 1
        interaction_id = f"{learner_id}:{row.source_index}"
        source_bundle = f"{learner_id}|{row.solving_id}"
        interactions.append(
            Interaction(
                interaction_id=interaction_id,
                learner_id=learner_id,
                item_id=row.item_id,
                correct=1 if row.user_answer == meta.correct_answer else 0,
                kc_ids=meta.kc_ids,
                order=order,
                timestamp=row.timestamp,
                bundle_id=ensure_bundle_id(source_bundle, interaction_id),
                split=None,
                metric_mask=True,
            )
        )
    return tuple(interactions)


def _stats(
    raw_rows: int,
    eligible_rows: int,
    reason_counts: Mapping[str, int],
) -> EdNetFilterStats:
    reasons = tuple(
        DropReasonCount(reason=reason, count=reason_counts[reason])
        for reason in _REASON_ORDER
        if reason_counts.get(reason, 0) > 0
    )
    return EdNetFilterStats(
        raw_rows=raw_rows,
        eligible_rows=eligible_rows,
        dropped_invalid_user_answer=sum(
            reason_counts.get(reason, 0) for reason in _INVALID_ANSWER_REASONS
        ),
        dropped_missing_metadata=int(reason_counts.get("missing_metadata", 0)),
        dropped_reasons=reasons,
    )


def _cell(row: Mapping[str, str | None], name: str) -> str:
    value = row.get(name)
    if value is None:
        return ""
    return str(value).strip()
