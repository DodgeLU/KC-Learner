"""Dummy bundle replay used to lock predict-before-update grouping.

State is an integer count of revealed responses for one learner. Every
row in a bundle is predicted from the same pre-bundle count. The count
increases only after the whole bundle has been predicted.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from kclearner.data.schema import Interaction
from kclearner.data.validation import DatasetValidator


@dataclass(frozen=True)
class BundleStepRecord:
    interaction_id: str
    learner_id: str
    bundle_id: str
    order: int
    pre_bundle_update_count: int


def replay_bundle_pre_state(
    interactions: Sequence[Interaction],
) -> tuple[BundleStepRecord, ...]:
    """Replay bundle grouping on a dummy per-learner counter.

    Raises ``ValueError`` when ``DatasetValidator`` rejects the rows.
    The returned pre-bundle count is shared by every row of a bundle.
    """

    result = DatasetValidator().validate(interactions)
    if not result.ok:
        codes = ", ".join(issue.code for issue in result.issues)
        raise ValueError(f"bundle replay requires a valid dataset: {codes}")

    by_learner: dict[str, list[Interaction]] = defaultdict(list)
    for row in interactions:
        by_learner[row.learner_id].append(row)

    steps: list[BundleStepRecord] = []
    for learner_id in sorted(by_learner):
        ordered = sorted(by_learner[learner_id], key=lambda row: row.order)
        groups: list[list[Interaction]] = []
        for row in ordered:
            if groups and groups[-1][0].bundle_id == row.bundle_id:
                groups[-1].append(row)
            else:
                groups.append([row])
        revealed = 0
        for group in groups:
            pre_bundle = revealed
            for row in group:
                steps.append(
                    BundleStepRecord(
                        interaction_id=row.interaction_id,
                        learner_id=learner_id,
                        bundle_id=row.bundle_id,
                        order=row.order,
                        pre_bundle_update_count=pre_bundle,
                    )
                )
            revealed += len(group)
    return tuple(steps)
