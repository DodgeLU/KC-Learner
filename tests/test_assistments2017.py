"""Public ASSISTments MAIN_V1 checks. No primary.csv and no checkpoints."""

from __future__ import annotations

import io
import unittest

import numpy as np

from kclearner.data.adapters.assistments2017 import (
    AssistmentsModelRow,
    bundle_key,
    kc_index_tuple,
    load_assistments2017_main_v1,
    oov_behavior,
    rows_for_neural_state,
)
from kclearner.data.schema import Interaction, interactions_from_json, interactions_to_json, prediction_target_count
from kclearner.models import ARKTModel, IRTModel, StreamingRow

_HEADER = (
    "student_id,problem_id,skill_id,correct,timestamp,attempt_count\n"
)


def _csv(body: str) -> io.StringIO:
    return io.StringIO(_HEADER + body)


class AssistmentsAdapterTests(unittest.TestCase):
    def test_eligibility_order_bundle_split_and_vocab(self) -> None:
        # File order is intentionally not time order. The last data row
        # is a non-first attempt and must not become an interaction.
        body = """1,2,area,1,9,1
1,2,area,1,1,1
1,10,ratio,0,2,1
1,2,area,1,3,1
1,2,area,0,4,1
1,2,area,1,5,1
1,2,area,1,6,1
1,2,area,0,7,1
1,2,area,1,7,1
1,99,area,0,8,1
1,2,area,1,11,2
1,2,,0,12,1
,2,area,1,13,1
1,,area,1,14,1
1,2,area,3,15,1
"""
        loaded = load_assistments2017_main_v1(_csv(body))
        stats = loaded.diagnostics
        self.assertEqual(stats.raw_rows, 15)
        self.assertEqual(stats.eligible_rows, 10)
        self.assertEqual(stats.filtered_non_first_attempt, 1)
        self.assertEqual(stats.missing_kc_rows, 1)
        self.assertEqual(stats.missing_learner_rows, 1)
        self.assertEqual(stats.missing_item_rows, 1)
        self.assertEqual(stats.invalid_response_rows, 1)
        self.assertEqual(
            [row.timestamp for row in loaded.interactions],
            [1, 2, 3, 4, 5, 6, 7, 7, 8, 9],
        )
        self.assertEqual(stats.promoted_to_valid, 1)
        self.assertEqual(stats.nominal_cross_boundary_bundles, 1)
        self.assertEqual(stats.derived_cross_boundary_bundles, 0)
        self.assertEqual(stats.train_rows, 6)
        self.assertEqual(stats.valid_rows, 2)
        self.assertEqual(stats.test_rows, 2)
        promoted = [row for row in loaded.interactions if row.timestamp == 7]
        self.assertEqual([row.split for row in promoted], ["valid", "valid"])
        self.assertEqual(promoted[0].bundle_id, promoted[1].bundle_id)
        self.assertEqual(promoted[0].bundle_id, bundle_key("1", 7))
        self.assertEqual(loaded.vocabulary.problem_index("10"), 0)
        self.assertEqual(loaded.vocabulary.problem_index("2"), 1)
        self.assertFalse(loaded.vocabulary.has_problem("99"))
        self.assertEqual(loaded.vocabulary.skill_index("area"), 0)
        self.assertEqual(loaded.vocabulary.skill_index("ratio"), 1)
        ratio = [row for row in loaded.interactions if row.item_id == "10"]
        self.assertEqual(len(ratio), 1)
        self.assertEqual(ratio[0].kc_ids, (1,))
        oov = [row for row in loaded.interactions if row.item_id == "99"]
        self.assertEqual(len(oov), 1)
        self.assertFalse(oov[0].metric_mask)
        self.assertEqual(oov[0].split, "test")
        self.assertEqual(stats.oov_rows, 1)
        self.assertEqual(stats.multi_kc_rows, 0)
        self.assertEqual(prediction_target_count(loaded.interactions), stats.eligible_rows)
        self.assertEqual(loaded.train_rows[1].problem_idx, 0)
        self.assertEqual(loaded.train_rows[1].skill_idx, 1)
        self.assertEqual(loaded.valid_rows[0].group_id, loaded.valid_rows[1].group_id)

    def test_student_index_is_lexicographic(self) -> None:
        body = """10,1,a,1,1,1
10,1,a,1,2,1
10,1,a,0,3,1
2,1,a,1,1,1
2,1,a,0,2,1
2,1,a,1,3,1
"""
        loaded = load_assistments2017_main_v1(_csv(body))
        self.assertEqual(loaded.vocabulary.students_sorted, ("10", "2"))
        self.assertEqual(loaded.vocabulary.student_index("10"), 0)
        self.assertEqual(loaded.vocabulary.student_index("2"), 1)

    def test_kc_tuple_stays_one_interaction(self) -> None:
        indexes = kc_index_tuple(("ratio", "area", "ratio"), {"area": 0, "ratio": 1})
        self.assertEqual(indexes, (1, 0, 1))
        row = Interaction(
            interaction_id="m",
            learner_id="1",
            item_id="2",
            correct=1,
            kc_ids=indexes,
            order=0,
            timestamp=1,
            bundle_id=bundle_key("1", 1),
            split="train",
            metric_mask=True,
        )
        self.assertEqual(prediction_target_count((row,)), 1)
        with self.assertRaises(KeyError):
            kc_index_tuple(("missing",), {"area": 0})

    def test_canonical_json_roundtrip(self) -> None:
        loaded = load_assistments2017_main_v1(
            _csv("1,2,area,1,1,1\n1,2,area,0,2,1\n1,2,area,1,3,1\n")
        )
        restored = interactions_from_json(interactions_to_json(loaded.interactions))
        self.assertEqual(restored, loaded.interactions)

    def test_oov_policies_differ_by_family(self) -> None:
        irt = oov_behavior("irt")
        ar = oov_behavior("ar_kt")
        dkt = oov_behavior("dkt_q")
        dkvmn = oov_behavior("dkvmn_qc")
        self.assertFalse(irt.eligible_for_metrics)
        self.assertFalse(irt.state_transparent)
        self.assertTrue(irt.updates_learner_state)
        self.assertFalse(irt.updates_item_state)
        self.assertEqual(irt.fallback_problem_index, 0)
        self.assertTrue(ar.updates_kc_state)
        self.assertFalse(irt.updates_kc_state)
        self.assertTrue(dkt.state_transparent)
        self.assertFalse(dkt.receives_prediction)
        self.assertFalse(dkt.updates_learner_state)
        self.assertTrue(dkvmn.state_transparent)
        self.assertIsNone(dkvmn.fallback_problem_index)
        middle = AssistmentsModelRow(
            student_idx=0,
            problem_idx=0,
            skill_idx=0,
            group_id=2,
            correct=0,
            timestamp=2,
            split="test",
            metric_mask=False,
            oov_problem=True,
            interaction_id="oov",
            learner_id="1",
            item_id="99",
            skill_id="area",
        )
        kept = AssistmentsModelRow(
            student_idx=0,
            problem_idx=1,
            skill_idx=0,
            group_id=1,
            correct=1,
            timestamp=1,
            split="test",
            metric_mask=True,
            oov_problem=False,
            interaction_id="keep",
            learner_id="1",
            item_id="2",
            skill_id="area",
        )
        self.assertEqual(
            [row.interaction_id for row in rows_for_neural_state((kept, middle, kept), "dkt_qc")],
            ["keep", "keep"],
        )
        self.assertEqual(len(rows_for_neural_state((kept, middle), "irt")), 2)

    def test_irt_eval_oov_updates_theta_and_freezes_b(self) -> None:
        model = IRTModel(1, 2)
        model.run_streaming(
            [StreamingRow(0, 1, 1, 1, interaction_id="seen", timestamp_ms=1, group_last_timestamp_ms=1)],
            phase="train",
        )
        b_before = model.b.copy()
        theta_before = float(model.theta[0])
        replay = model.run_streaming(
            [StreamingRow(0, 0, 2, 0, interaction_id="oov", timestamp_ms=2, group_last_timestamp_ms=2)],
            phase="eval",
        )
        self.assertEqual(len(replay.rows), 1)
        self.assertTrue(np.array_equal(model.b, b_before))
        self.assertNotEqual(float(model.theta[0]), theta_before)
        self.assertGreater(replay.rows[0].p_global, 0.0)
        self.assertLess(replay.rows[0].p_global, 1.0)

    def test_assistments_skill_mean_is_not_the_ednet_token_update(self) -> None:
        mixed = [
            StreamingRow(0, 0, 1, 1, (0,), "a", 1, 1),
            StreamingRow(0, 1, 1, 0, (1,), "b", 1, 1),
        ]
        ednet = ARKTModel(1, 2, 2, eta_r=0.20)
        assist = ARKTModel(1, 2, 2, eta_r=0.20, residual_update="assistments_skill_mean")
        ednet.run_streaming(mixed, phase="train")
        assist.run_streaming(mixed, phase="train")
        self.assertTrue(np.allclose(ednet.r_post, 0.0))
        self.assertAlmostEqual(float(assist.r_post[0, 0]), 0.1)
        self.assertAlmostEqual(float(assist.r_post[0, 1]), -0.1)
        repeated = [
            StreamingRow(0, 0, 1, 1, (0,), "a", 1, 1),
            StreamingRow(0, 1, 1, 1, (0,), "b", 1, 1),
        ]
        ednet_same = ARKTModel(1, 2, 2, eta_r=0.20)
        assist_same = ARKTModel(1, 2, 2, eta_r=0.20, residual_update="assistments_skill_mean")
        ednet_same.run_streaming(repeated, phase="train")
        assist_same.run_streaming(repeated, phase="train")
        self.assertAlmostEqual(float(ednet_same.r_post[0, 0]), 0.05)
        self.assertAlmostEqual(float(assist_same.r_post[0, 0]), 0.1)

    def test_assistments_checkpoint_selects_skill_mean(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "assist.npz"
            np.savez_compressed(
                path,
                theta=np.zeros(1),
                b=np.zeros(1),
                r_post=np.zeros((1, 1)),
                r_seen=np.zeros((1, 1), dtype=bool),
                last_r_update_ts=np.full((1, 1), -1),
                n_students=np.int64(1),
                n_problems=np.int64(1),
                n_skills=np.int64(1),
                eta_r=np.float64(0.20),
                use_residual=np.bool_(True),
                use_time=np.bool_(False),
                model_id=np.str_("assist2017_ar_kt_v1"),
            )
            model = ARKTModel.load_legacy_checkpoint(path)
        self.assertEqual(model.residual_update, "assistments_skill_mean")
        self.assertEqual(model.eta_r, 0.20)


if __name__ == "__main__":
    unittest.main()
