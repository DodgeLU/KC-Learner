"""DatasetValidator semantic invariants."""

from __future__ import annotations

import unittest
from dataclasses import replace

from kclearner.data.schema import Interaction
from kclearner.data.toy import make_toy_interactions
from kclearner.data.validation import DatasetValidator


def _codes(rows: tuple[Interaction, ...]) -> set[str]:
    result = DatasetValidator().validate(rows)
    return {issue.code for issue in result.issues}


class ValidatorTests(unittest.TestCase):
    def test_toy_dataset_passes(self) -> None:
        result = DatasetValidator().validate(make_toy_interactions())
        self.assertTrue(result.ok)
        self.assertEqual(result.issues, ())

    def test_empty_learner_and_item(self) -> None:
        row = replace(make_toy_interactions()[0], learner_id="  ", item_id="")
        self.assertEqual(
            _codes((row,)),
            {"empty_learner_id", "empty_item_id"},
        )

    def test_correct_domain(self) -> None:
        row = replace(make_toy_interactions()[0], correct=2)
        self.assertIn("invalid_correct", _codes((row,)))

    def test_duplicate_interaction_id(self) -> None:
        rows = make_toy_interactions()
        broken = (rows[0], replace(rows[1], interaction_id=rows[0].interaction_id))
        self.assertIn("duplicate_interaction_id", _codes(broken))

    def test_duplicate_order_inside_learner(self) -> None:
        rows = make_toy_interactions()
        broken = (rows[0], replace(rows[1], order=rows[0].order))
        self.assertIn("duplicate_learner_order", _codes(broken))

    def test_duplicate_kc_ids_are_preserved(self) -> None:
        row = replace(make_toy_interactions()[0], kc_ids=(10, 10))
        self.assertTrue(DatasetValidator().validate((row,)).ok)
        self.assertEqual(row.kc_ids, (10, 10))
        reported = DatasetValidator().report_duplicate_kc((row,))
        self.assertEqual(len(reported), 1)
        self.assertEqual(reported[0].interaction_id, row.interaction_id)
        self.assertEqual(row.kc_ids, (10, 10))

    def test_negative_kc_is_not_a_component(self) -> None:
        row = replace(make_toy_interactions()[0], kc_ids=(-1,))
        self.assertIn("invalid_kc_id", _codes((row,)))

    def test_bundle_cannot_cross_learner(self) -> None:
        rows = make_toy_interactions()
        broken = (
            rows[0],
            replace(rows[4], bundle_id=rows[0].bundle_id),
        )
        self.assertIn("bundle_crosses_learner", _codes(broken))

    def test_bundle_cannot_cross_split(self) -> None:
        rows = make_toy_interactions()
        broken = (
            rows[0],
            replace(rows[1], split="valid"),
        )
        self.assertIn("bundle_crosses_split", _codes(broken))

    def test_bundle_must_stay_contiguous(self) -> None:
        rows = make_toy_interactions()
        interleaved = (
            rows[0],
            replace(rows[2], order=1, bundle_id="L1-B2"),
            replace(rows[1], order=2, bundle_id="L1-B1"),
        )
        self.assertIn("bundle_not_contiguous", _codes(interleaved))

    def test_temporal_order_must_not_go_backwards(self) -> None:
        rows = make_toy_interactions()
        broken = (
            rows[0],
            replace(rows[2], timestamp=999),
        )
        self.assertIn("temporal_regression", _codes(broken))

    def test_equal_timestamps_inside_a_bundle_are_allowed(self) -> None:
        result = DatasetValidator().validate(make_toy_interactions())
        self.assertTrue(result.ok)
        same_time = [
            row.timestamp
            for row in make_toy_interactions()
            if row.bundle_id == "L1-B1"
        ]
        self.assertEqual(same_time, [1000, 1000])


if __name__ == "__main__":
    unittest.main()
