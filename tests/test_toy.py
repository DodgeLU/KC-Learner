"""Toy dataset coverage. No real EdNet or ASSISTments rows."""

from __future__ import annotations

import unittest

from kclearner.data.schema import prediction_target_count
from kclearner.data.toy import make_toy_interactions
from kclearner.data.validation import DatasetValidator


class ToyDatasetTests(unittest.TestCase):
    def test_coverage(self) -> None:
        rows = make_toy_interactions()
        learners = {row.learner_id for row in rows}
        self.assertEqual(learners, {"L1", "L2"})
        self.assertTrue(any(len(row.kc_ids) == 1 for row in rows))
        self.assertTrue(any(len(row.kc_ids) > 1 for row in rows))
        self.assertTrue(any(row.kc_ids == () for row in rows))
        bundle_sizes: dict[str, int] = {}
        for row in rows:
            bundle_sizes[row.bundle_id] = bundle_sizes.get(row.bundle_id, 0) + 1
        self.assertTrue(any(size > 1 for size in bundle_sizes.values()))
        self.assertEqual(prediction_target_count(rows), 7)
        self.assertTrue(DatasetValidator().validate(rows).ok)

    def test_orders_match_timestamps(self) -> None:
        by_learner: dict[str, list] = {}
        for row in make_toy_interactions():
            by_learner.setdefault(row.learner_id, []).append(row)
        for rows in by_learner.values():
            ordered = sorted(rows, key=lambda row: row.order)
            timestamps = [row.timestamp for row in ordered]
            self.assertEqual(timestamps, sorted(timestamps))


if __name__ == "__main__":
    unittest.main()
