"""Bundle-safe split on synthetic interactions. No EdNet parquet."""

from __future__ import annotations

import unittest
from collections import defaultdict

from kclearner.data.schema import Interaction
from kclearner.data.splitting import assign_fractional_bundle_split


def _row(index: int, bundle: str, order: int, learner: str = "L") -> Interaction:
    return Interaction(
        interaction_id=f"{learner}:{index}",
        learner_id=learner,
        item_id=f"Q{index}",
        correct=1,
        kc_ids=(1,),
        order=order,
        timestamp=index,
        bundle_id=bundle,
        split=None,
        metric_mask=True,
    )


class BundleSafeSplitTests(unittest.TestCase):
    def test_labels_whole_bundles_without_reordering(self) -> None:
        rows = tuple(_row(index, f"B{index}", index) for index in range(10))
        labeled = assign_fractional_bundle_split(
            rows,
            train_fraction=0.7,
            valid_fraction=0.1,
        )
        self.assertEqual(
            [row.interaction_id for row in labeled],
            [row.interaction_id for row in rows],
        )
        self.assertEqual(
            [row.split for row in labeled],
            ["train"] * 7 + ["valid"] + ["test"] * 2,
        )
        by_bundle: dict[str, set[str | None]] = defaultdict(set)
        for row in labeled:
            by_bundle[row.bundle_id].add(row.split)
        self.assertTrue(all(len(labels) == 1 for labels in by_bundle.values()))

    def test_interleaved_bundle_keeps_one_label_and_source_order(self) -> None:
        rows = (
            _row(0, "B1", 0),
            _row(1, "B2", 1),
            _row(2, "B1", 2),
        )
        labeled = assign_fractional_bundle_split(
            rows,
            train_fraction=0.5,
            valid_fraction=0.0,
        )
        self.assertEqual([row.interaction_id for row in labeled], ["L:0", "L:1", "L:2"])
        self.assertEqual([row.split for row in labeled], ["train", "test", "train"])
        self.assertEqual(labeled[0].split, labeled[2].split)


if __name__ == "__main__":
    unittest.main()
