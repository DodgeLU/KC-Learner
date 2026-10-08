"""Schema type, one-target, and serialization tests."""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from kclearner.data.schema import (
    Interaction,
    interactions_from_json,
    interactions_to_json,
    prediction_target_count,
)
from kclearner.data.toy import make_toy_interactions


class SchemaTests(unittest.TestCase):
    def test_multi_kc_remains_one_target(self) -> None:
        rows = make_toy_interactions()
        multi = [row for row in rows if row.interaction_id == "L1-1"]
        self.assertEqual(len(multi), 1)
        self.assertEqual(multi[0].kc_ids, (10, 20, 30))
        self.assertEqual(prediction_target_count(multi), 1)
        self.assertEqual(prediction_target_count(rows), len(rows))

    def test_missing_kc_is_empty_tuple(self) -> None:
        rows = make_toy_interactions()
        missing = [row for row in rows if row.interaction_id == "L1-2"]
        self.assertEqual(missing[0].kc_ids, ())
        self.assertIsInstance(missing[0].kc_ids, tuple)

    def test_kc_ids_reject_list(self) -> None:
        with self.assertRaises(TypeError):
            Interaction(
                interaction_id="x",
                learner_id="L",
                item_id="Q",
                correct=1,
                kc_ids=[1, 2],  # type: ignore[arg-type]
                order=0,
                timestamp=None,
                bundle_id="B",
                split=None,
                metric_mask=True,
            )

    def test_record_is_frozen(self) -> None:
        row = make_toy_interactions()[0]
        with self.assertRaises(FrozenInstanceError):
            row.correct = 0  # type: ignore[misc]

    def test_json_roundtrip(self) -> None:
        rows = make_toy_interactions()
        restored = interactions_from_json(interactions_to_json(rows))
        self.assertEqual(restored, rows)
        self.assertIsInstance(restored[1].kc_ids, tuple)
        self.assertEqual(restored[1].kc_ids, (10, 20, 30))
        self.assertEqual(restored[2].kc_ids, ())


if __name__ == "__main__":
    unittest.main()
