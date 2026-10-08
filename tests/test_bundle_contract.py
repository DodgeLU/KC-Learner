"""Bundle predict-before-update contract on a dummy counter."""

from __future__ import annotations

import unittest
from collections import defaultdict

from kclearner.data.toy import make_toy_interactions
from kclearner.sequence.contract import replay_bundle_pre_state


class BundleContractTests(unittest.TestCase):
    def test_rows_in_a_bundle_share_pre_bundle_state(self) -> None:
        rows = make_toy_interactions()
        steps = replay_bundle_pre_state(rows)
        grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
        for step in steps:
            grouped[(step.learner_id, step.bundle_id)].append(
                step.pre_bundle_update_count
            )
        for key, counts in grouped.items():
            self.assertEqual(
                len(set(counts)),
                1,
                msg=f"{key} saw more than one pre-bundle state: {counts}",
            )

    def test_state_advances_only_after_the_bundle(self) -> None:
        steps = replay_bundle_pre_state(make_toy_interactions())
        by_id = {step.interaction_id: step for step in steps}
        self.assertEqual(by_id["L1-0"].pre_bundle_update_count, 0)
        self.assertEqual(by_id["L1-1"].pre_bundle_update_count, 0)
        self.assertEqual(by_id["L1-2"].pre_bundle_update_count, 2)
        self.assertEqual(by_id["L1-3"].pre_bundle_update_count, 3)
        self.assertEqual(by_id["L2-1"].pre_bundle_update_count, 1)
        self.assertEqual(by_id["L2-2"].pre_bundle_update_count, 1)

    def test_learners_do_not_share_state(self) -> None:
        steps = replay_bundle_pre_state(make_toy_interactions())
        first = [
            step.pre_bundle_update_count
            for step in steps
            if step.order == 0
        ]
        self.assertEqual(first, [0, 0])


if __name__ == "__main__":
    unittest.main()
