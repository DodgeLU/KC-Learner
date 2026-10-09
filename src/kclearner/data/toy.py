"""Synthetic interactions for semantic tests.

This builder does not read EdNet or ASSISTments.
"""

from __future__ import annotations

from kclearner.data.schema import Interaction
from kclearner.data.validation import DatasetValidator


def make_toy_interactions() -> tuple[Interaction, ...]:
    """Return a small multi-learner dataset.

    Coverage: one KC, several KCs, no KC, several rows in one bundle,
    two learners, and non-decreasing timestamps.
    """

    rows = (
        Interaction(
            interaction_id="L1-0",
            learner_id="L1",
            item_id="Q1",
            correct=1,
            kc_ids=(10,),
            order=0,
            timestamp=1000,
            bundle_id="L1-B1",
            split="train",
            metric_mask=True,
        ),
        Interaction(
            interaction_id="L1-1",
            learner_id="L1",
            item_id="Q2",
            correct=0,
            kc_ids=(10, 20, 30),
            order=1,
            timestamp=1000,
            bundle_id="L1-B1",
            split="train",
            metric_mask=True,
        ),
        Interaction(
            interaction_id="L1-2",
            learner_id="L1",
            item_id="Q3",
            correct=1,
            kc_ids=(),
            order=2,
            timestamp=1500,
            bundle_id="L1-B2",
            split="train",
            metric_mask=True,
        ),
        Interaction(
            interaction_id="L1-3",
            learner_id="L1",
            item_id="Q4",
            correct=1,
            kc_ids=(20,),
            order=3,
            timestamp=2000,
            bundle_id="L1-B3",
            split="valid",
            metric_mask=True,
        ),
        Interaction(
            interaction_id="L2-0",
            learner_id="L2",
            item_id="Q1",
            correct=1,
            kc_ids=(10,),
            order=0,
            timestamp=100,
            bundle_id="L2-B1",
            split="train",
            metric_mask=True,
        ),
        Interaction(
            interaction_id="L2-1",
            learner_id="L2",
            item_id="Q5",
            correct=0,
            kc_ids=(40, 50),
            order=1,
            timestamp=200,
            bundle_id="L2-B2",
            split="train",
            metric_mask=True,
        ),
        Interaction(
            interaction_id="L2-2",
            learner_id="L2",
            item_id="Q6",
            correct=1,
            kc_ids=(),
            order=2,
            timestamp=250,
            bundle_id="L2-B2",
            split="train",
            metric_mask=True,
        ),
    )
    result = DatasetValidator().validate(rows)
    if not result.ok:
        rendered = "; ".join(issue.code for issue in result.issues)
        raise RuntimeError(f"toy dataset failed validation: {rendered}")
    return rows
