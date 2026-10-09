"""Fractional bundle labels for an already ordered interaction sequence.

This helper is for synthetic and custom datasets. It is not the formal
EdNet dev5000 regression split. Formal EdNet regression keeps the
existing split labels and the physical row order inside each split
parquet.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Sequence

from kclearner.data.schema import Interaction


def assign_fractional_bundle_split(
    interactions: Sequence[Interaction],
    *,
    train_fraction: float,
    valid_fraction: float,
) -> tuple[Interaction, ...]:
    """Assign each whole bundle to train, valid, or test.

    Within one learner, bundles are taken in the order of their first
    row in ``interactions``. ``train_fraction`` and ``valid_fraction``
    are applied with truncation toward zero. The remainder is test.
    Every row of a bundle receives the same label. The returned tuple
    has the same interaction order as the input.
    """

    if train_fraction < 0 or valid_fraction < 0:
        raise ValueError("split fractions must be non-negative")
    if train_fraction + valid_fraction > 1:
        raise ValueError("train_fraction + valid_fraction must be <= 1")

    bundle_order: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for row in interactions:
        if row.bundle_id not in seen[row.learner_id]:
            seen[row.learner_id].add(row.bundle_id)
            bundle_order[row.learner_id].append(row.bundle_id)

    label_of: dict[tuple[str, str], str] = {}
    for learner_id, bundles in bundle_order.items():
        n_train, n_valid, _n_test = _bundle_counts(
            len(bundles),
            train_fraction,
            valid_fraction,
        )
        for index, bundle_id in enumerate(bundles):
            if index < n_train:
                label = "train"
            elif index < n_train + n_valid:
                label = "valid"
            else:
                label = "test"
            label_of[(learner_id, bundle_id)] = label

    return tuple(
        replace(row, split=label_of[(row.learner_id, row.bundle_id)])
        for row in interactions
    )


def _bundle_counts(
    n_bundles: int,
    train_fraction: float,
    valid_fraction: float,
) -> tuple[int, int, int]:
    n_train = int(n_bundles * train_fraction)
    n_valid = int(n_bundles * valid_fraction)
    if n_train + n_valid > n_bundles:
        n_valid = n_bundles - n_train
    n_test = n_bundles - n_train - n_valid
    return n_train, n_valid, n_test
