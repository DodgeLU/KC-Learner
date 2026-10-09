"""Dataset-level checks for canonical interactions.

OOV eligibility is intentionally absent. ``metric_mask`` is only checked
as a boolean on the record.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from kclearner.data.schema import Interaction


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    interaction_id: str | None = None


@dataclass(frozen=True)
class ValidationResult:
    issues: tuple[ValidationIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues


class DatasetValidator:
    """Validate a sequence of canonical interactions.

    A passing result means the rows can be ordered per learner, bundles
    stay inside one learner and one split, and each row remains one
    target. It does not score model outputs.
    """

    def validate(
        self,
        interactions: Sequence[Interaction],
        *,
        check_temporal: bool = True,
        check_bundle_contiguity: bool = True,
    ) -> ValidationResult:
        issues: list[ValidationIssue] = []
        self._check_rows(interactions, issues)
        self._check_identity_and_order(interactions, issues)
        self._check_bundles(interactions, issues, check_contiguity=check_bundle_contiguity)
        if check_temporal:
            self._check_time(interactions, issues)
        return ValidationResult(issues=tuple(issues))

    def report_duplicate_kc(
        self,
        interactions: Sequence[Interaction],
    ) -> tuple[ValidationIssue, ...]:
        """List rows whose ``kc_ids`` repeat a value.

        This does not change the records and is not part of ``validate``.
        Repeated KC tokens stay in the canonical tuple.
        """

        found: list[ValidationIssue] = []
        for row in interactions:
            if len(row.kc_ids) != len(set(row.kc_ids)):
                found.append(
                    ValidationIssue(
                        "duplicate_kc_id",
                        f"kc_ids repeats a token: {row.kc_ids}",
                        row.interaction_id,
                    )
                )
        return tuple(found)

    def _check_rows(
        self,
        interactions: Sequence[Interaction],
        issues: list[ValidationIssue],
    ) -> None:
        for row in interactions:
            if not isinstance(row, Interaction):
                raise TypeError(
                    "DatasetValidator expects Interaction records, "
                    f"got {type(row).__name__}"
                )
            if row.learner_id.strip() == "":
                issues.append(
                    ValidationIssue(
                        "empty_learner_id",
                        "learner_id is empty",
                        row.interaction_id,
                    )
                )
            if row.item_id.strip() == "":
                issues.append(
                    ValidationIssue(
                        "empty_item_id",
                        "item_id is empty",
                        row.interaction_id,
                    )
                )
            if row.interaction_id.strip() == "":
                issues.append(
                    ValidationIssue(
                        "empty_interaction_id",
                        "interaction_id is empty",
                        row.interaction_id,
                    )
                )
            if row.bundle_id.strip() == "":
                issues.append(
                    ValidationIssue(
                        "empty_bundle_id",
                        "bundle_id is empty",
                        row.interaction_id,
                    )
                )
            if row.correct not in (0, 1):
                issues.append(
                    ValidationIssue(
                        "invalid_correct",
                        f"correct must be 0 or 1, got {row.correct}",
                        row.interaction_id,
                    )
                )
            if row.order < 0:
                issues.append(
                    ValidationIssue(
                        "invalid_order",
                        f"order must be >= 0, got {row.order}",
                        row.interaction_id,
                    )
                )
            if row.split is not None and row.split.strip() == "":
                issues.append(
                    ValidationIssue(
                        "empty_split",
                        "split is an empty string",
                        row.interaction_id,
                    )
                )
            for kc_id in row.kc_ids:
                if kc_id < 0:
                    issues.append(
                        ValidationIssue(
                            "invalid_kc_id",
                            f"kc_ids contains a negative id {kc_id}; "
                            "negative sentinels are not knowledge components",
                            row.interaction_id,
                        )
                    )
            if row.timestamp is not None and (
                isinstance(row.timestamp, float) and math.isnan(row.timestamp)
            ):
                issues.append(
                    ValidationIssue(
                        "invalid_timestamp",
                        "timestamp is NaN",
                        row.interaction_id,
                    )
                )

    def _check_identity_and_order(
        self,
        interactions: Sequence[Interaction],
        issues: list[ValidationIssue],
    ) -> None:
        seen_ids: dict[str, int] = {}
        for row in interactions:
            seen_ids[row.interaction_id] = seen_ids.get(row.interaction_id, 0) + 1
        for interaction_id, count in seen_ids.items():
            if interaction_id.strip() == "":
                continue
            if count > 1:
                issues.append(
                    ValidationIssue(
                        "duplicate_interaction_id",
                        f"interaction_id {interaction_id!r} appears {count} times",
                        interaction_id,
                    )
                )

        by_learner: dict[str, list[Interaction]] = defaultdict(list)
        for row in interactions:
            by_learner[row.learner_id].append(row)
        for learner_id, rows in by_learner.items():
            orders: dict[int, str] = {}
            for row in rows:
                if row.order in orders:
                    issues.append(
                        ValidationIssue(
                            "duplicate_learner_order",
                            f"learner {learner_id!r} has order {row.order} "
                            f"on both {orders[row.order]!r} and {row.interaction_id!r}",
                            row.interaction_id,
                        )
                    )
                else:
                    orders[row.order] = row.interaction_id

    def _check_bundles(
        self,
        interactions: Sequence[Interaction],
        issues: list[ValidationIssue],
        *,
        check_contiguity: bool,
    ) -> None:
        learners_by_bundle: dict[str, set[str]] = defaultdict(set)
        splits_by_bundle: dict[str, set[str | None]] = defaultdict(set)
        for row in interactions:
            learners_by_bundle[row.bundle_id].add(row.learner_id)
            splits_by_bundle[row.bundle_id].add(row.split)
        for bundle_id, learners in learners_by_bundle.items():
            if bundle_id.strip() == "":
                continue
            if len(learners) > 1:
                issues.append(
                    ValidationIssue(
                        "bundle_crosses_learner",
                        f"bundle {bundle_id!r} spans learners {sorted(learners)}",
                    )
                )
        for bundle_id, splits in splits_by_bundle.items():
            if bundle_id.strip() == "":
                continue
            if len(splits) > 1:
                issues.append(
                    ValidationIssue(
                        "bundle_crosses_split",
                        f"bundle {bundle_id!r} spans splits {sorted(splits, key=_split_sort_key)}",
                    )
                )

        if not check_contiguity:
            return
        by_learner: dict[str, list[Interaction]] = defaultdict(list)
        for row in interactions:
            by_learner[row.learner_id].append(row)
        for learner_id, rows in by_learner.items():
            ordered = sorted(rows, key=lambda row: row.order)
            closed: set[str] = set()
            open_bundle: str | None = None
            for row in ordered:
                if row.bundle_id == open_bundle:
                    continue
                if open_bundle is not None:
                    closed.add(open_bundle)
                if row.bundle_id in closed:
                    issues.append(
                        ValidationIssue(
                            "bundle_not_contiguous",
                            f"learner {learner_id!r} revisits bundle {row.bundle_id!r} "
                            f"at order {row.order} after another bundle",
                            row.interaction_id,
                        )
                    )
                open_bundle = row.bundle_id

    def _check_time(
        self,
        interactions: Sequence[Interaction],
        issues: list[ValidationIssue],
    ) -> None:
        by_learner: dict[str, list[Interaction]] = defaultdict(list)
        for row in interactions:
            by_learner[row.learner_id].append(row)
        for learner_id, rows in by_learner.items():
            ordered = sorted(rows, key=lambda row: row.order)
            previous: float | None = None
            for row in ordered:
                if row.timestamp is None:
                    continue
                if isinstance(row.timestamp, float) and math.isnan(row.timestamp):
                    continue
                if previous is not None and row.timestamp < previous:
                    issues.append(
                        ValidationIssue(
                            "temporal_regression",
                            f"learner {learner_id!r} order {row.order} has timestamp "
                            f"{row.timestamp} before previous timestamp {previous}",
                            row.interaction_id,
                        )
                    )
                previous = float(row.timestamp)


def _split_sort_key(value: str | None) -> tuple[int, str]:
    if value is None:
        return (0, "")
    return (1, value)
