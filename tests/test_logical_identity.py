"""Logical interaction and dataset hashes. No third-party data."""

from __future__ import annotations

import unittest
from dataclasses import replace

from kclearner.data.ednet_corrected import cohort_id_hash
from kclearner.data.logical_identity import (
    SERIALIZATION_VERSION,
    DatasetLogicalIdentity,
    EncodingProvenance,
    LogicalIdentityError,
    SplitLogicalIdentity,
    canonical_dataset_json,
    canonical_interaction_json,
    dataset_logical_hash,
    hash_ordered_ids,
    logical_split_hash,
)
from kclearner.data.schema import Interaction


def _row(**overrides: object) -> Interaction:
    payload = dict(
        interaction_id="L:0",
        learner_id="L",
        item_id="Q",
        correct=1,
        kc_ids=(2, 2, 5),
        order=0,
        timestamp=1000,
        bundle_id="L|7",
        split="train",
        metric_mask=True,
    )
    payload.update(overrides)
    return Interaction(**payload)  # type: ignore[arg-type]


def _identity(**overrides: object) -> DatasetLogicalIdentity:
    left = logical_split_hash((_row(),))
    payload = dict(
        dataset_id="toy",
        protocol_id="toy_v1",
        preprocessing_version="toy_prep_v1",
        cohort_hash=hash_ordered_ids(("L",)),
        splits=(
            SplitLogicalIdentity("train", left, 1),
            SplitLogicalIdentity("valid", logical_split_hash(()), 0),
        ),
        item_vocabulary_hash=hash_ordered_ids(("Q",)),
        kc_vocabulary_hash=hash_ordered_ids(("2", "5")),
    )
    payload.update(overrides)
    return DatasetLogicalIdentity(**payload)  # type: ignore[arg-type]


class LogicalHashTests(unittest.TestCase):
    def test_repeated_execution_is_stable(self) -> None:
        rows = (_row(), _row(interaction_id="L:1", order=1, correct=0))
        self.assertEqual(logical_split_hash(rows), logical_split_hash(rows))
        self.assertIn(SERIALIZATION_VERSION, canonical_dataset_json(_identity()))

    def test_row_order_changes_the_hash(self) -> None:
        first = _row()
        second = _row(interaction_id="L:1", order=1)
        self.assertNotEqual(
            logical_split_hash((first, second)),
            logical_split_hash((second, first)),
        )

    def test_response_kc_order_mask_and_bundle_change_the_hash(self) -> None:
        base = logical_split_hash((_row(),))
        self.assertNotEqual(base, logical_split_hash((_row(correct=0),)))
        self.assertNotEqual(base, logical_split_hash((_row(kc_ids=(5, 2, 2)),)))
        self.assertNotEqual(base, logical_split_hash((_row(metric_mask=False),)))
        self.assertNotEqual(base, logical_split_hash((_row(bundle_id="L|8"),)))

    def test_integral_float_timestamp_matches_int(self) -> None:
        self.assertEqual(
            canonical_interaction_json(_row(timestamp=1000)),
            canonical_interaction_json(_row(timestamp=1000.0)),
        )

    def test_non_finite_timestamp_is_rejected(self) -> None:
        with self.assertRaises(LogicalIdentityError):
            canonical_interaction_json(_row(timestamp=float("nan")))

    def test_ordered_ids_keep_order_and_duplicates(self) -> None:
        self.assertEqual(hash_ordered_ids(("b", "a")), cohort_id_hash(("b", "a")))
        self.assertNotEqual(hash_ordered_ids(("a", "b")), hash_ordered_ids(("b", "a")))
        self.assertNotEqual(hash_ordered_ids(("a",)), hash_ordered_ids(("a", "a")))
        self.assertNotEqual(hash_ordered_ids(("a", "b")), hash_ordered_ids(("a", "c")))

    def test_experiment_fields_are_absent_from_dataset_hash(self) -> None:
        identity = _identity()
        text = canonical_dataset_json(identity)
        for token in (
            "irt",
            "cuda",
            "learning_rate",
            "adam",
            "seed",
            "pyarrow",
            "parquet",
        ):
            self.assertNotIn(token, text)
        encoding = EncodingProvenance(
            parquet_sha256="a" * 64,
            pyarrow_version="25.0.1",
            serialization_format="parquet",
        )
        self.assertEqual(
            dataset_logical_hash(identity),
            dataset_logical_hash(identity),
        )
        self.assertNotIn(encoding.parquet_sha256, text)
        swapped = replace(
            identity,
            splits=(identity.splits[1], identity.splits[0]),
        )
        self.assertNotEqual(
            dataset_logical_hash(identity),
            dataset_logical_hash(swapped),
        )
