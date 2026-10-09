"""Source-quality diagnostics for an EdNet interaction load.

The report describes the eligible sequence. It does not drop, reorder,
or rewrite rows.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from kclearner.data.schema import Interaction


@dataclass(frozen=True)
class EdNetDiagnostics:
    """Counts of source irregularities and filter outcomes."""

    raw_rows: int
    eligible_rows: int
    filtered_rows: int
    filtered_by_reason: tuple
    duplicate_kc_rows: int
    duplicate_kc_questions: int
    repeated_question_attempt_rows: int
    questions_repeated_across_bundles: int
    duplicate_timestamp_groups: int
    timestamp_inversions: int
    solving_id_inversions: int
    bundle_count: int
    min_bundle_size: int
    max_bundle_size: int
    multi_kc_rows: int
    multi_kc_rate: float
    missing_kc_rows: int
    missing_kc_rate: float


def build_ednet_diagnostics(interactions, stats) -> EdNetDiagnostics:
    rows = tuple(interactions)
    eligible = len(rows)
    by_learner: dict[str, list[Interaction]] = defaultdict(list)
    for row in rows:
        by_learner[row.learner_id].append(row)

    duplicate_questions: set[str] = set()
    duplicate_kc_rows = 0
    multi_kc_rows = 0
    missing_kc_rows = 0
    bundle_sizes: Counter[tuple[str, str]] = Counter()
    timestamp_inversions = 0
    solving_id_inversions = 0
    duplicate_timestamp_groups = 0
    repeated_question_attempt_rows = 0
    questions_repeated_across_bundles = 0

    for row in rows:
        bundle_sizes[(row.learner_id, row.bundle_id)] += 1
        if len(row.kc_ids) != len(set(row.kc_ids)):
            duplicate_kc_rows += 1
            duplicate_questions.add(row.item_id)
        if len(row.kc_ids) > 1:
            multi_kc_rows += 1
        if len(row.kc_ids) == 0:
            missing_kc_rows += 1

    for learner_rows in by_learner.values():
        ordered = sorted(learner_rows, key=lambda item: item.order)
        previous_timestamp: float | None = None
        previous_solving: int | None = None
        timestamp_counts: Counter[int | float] = Counter()
        attempts: dict[str, set[str]] = defaultdict(set)
        attempt_rows: Counter[str] = Counter()
        for row in ordered:
            attempt_rows[row.item_id] += 1
            attempts[row.item_id].add(row.bundle_id)
            solving = _solving_id(row)
            if (
                previous_solving is not None
                and solving is not None
                and solving < previous_solving
            ):
                solving_id_inversions += 1
            if solving is not None:
                previous_solving = solving
            if row.timestamp is not None:
                timestamp_counts[row.timestamp] += 1
                if previous_timestamp is not None and row.timestamp < previous_timestamp:
                    timestamp_inversions += 1
                previous_timestamp = float(row.timestamp)
        duplicate_timestamp_groups += sum(
            1 for count in timestamp_counts.values() if count > 1
        )
        repeated_question_attempt_rows += sum(
            count - 1 for count in attempt_rows.values() if count > 1
        )
        questions_repeated_across_bundles += sum(
            1 for bundles in attempts.values() if len(bundles) > 1
        )

    sizes = tuple(bundle_sizes.values())
    return EdNetDiagnostics(
        raw_rows=stats.raw_rows,
        eligible_rows=eligible,
        filtered_rows=stats.raw_rows - eligible,
        filtered_by_reason=stats.dropped_reasons,
        duplicate_kc_rows=duplicate_kc_rows,
        duplicate_kc_questions=len(duplicate_questions),
        repeated_question_attempt_rows=repeated_question_attempt_rows,
        questions_repeated_across_bundles=questions_repeated_across_bundles,
        duplicate_timestamp_groups=duplicate_timestamp_groups,
        timestamp_inversions=timestamp_inversions,
        solving_id_inversions=solving_id_inversions,
        bundle_count=len(sizes),
        min_bundle_size=min(sizes) if sizes else 0,
        max_bundle_size=max(sizes) if sizes else 0,
        multi_kc_rows=multi_kc_rows,
        multi_kc_rate=(multi_kc_rows / eligible) if eligible else 0.0,
        missing_kc_rows=missing_kc_rows,
        missing_kc_rate=(missing_kc_rows / eligible) if eligible else 0.0,
    )


def _solving_id(row: Interaction) -> int | None:
    prefix = f"{row.learner_id}|"
    if not row.bundle_id.startswith(prefix):
        return None
    suffix = row.bundle_id[len(prefix):]
    if not suffix.isdigit():
        return None
    return int(suffix)
