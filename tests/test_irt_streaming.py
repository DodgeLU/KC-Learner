"""Synthetic IRT streaming semantics. No EdNet data."""

from __future__ import annotations

import unittest

import numpy as np

from kclearner.models import IRTModel, StreamingRow, metric_auc, metric_brier, metric_nll
from kclearner.models.streaming import sigmoid


def _row(
    student: int,
    problem: int,
    group: int,
    correct: int,
    interaction_id: str,
) -> StreamingRow:
    return StreamingRow(
        student_idx=student,
        problem_idx=problem,
        group_id=group,
        correct=correct,
        interaction_id=interaction_id,
    )


def _physical_fixture() -> list[StreamingRow]:
    # Physical order is the model order. Group 1 is interrupted by another
    # learner and later continues as a new run.
    return [
        _row(0, 0, 1, 1, "r0"),
        _row(0, 1, 1, 0, "r1"),
        _row(1, 0, 1, 1, "r2"),
        _row(0, 0, 3, 0, "r3"),
        _row(0, 1, 1, 1, "r4"),
        _row(0, 1, 1, 0, "r5"),
    ]


class IRTStreamingTests(unittest.TestCase):
    def test_physical_scan_flushes_on_learner_and_later_group(self) -> None:
        result = IRTModel(2, 3).run_streaming(_physical_fixture(), phase="train")
        self.assertEqual(
            [row.interaction_id for row in result.rows],
            ["r0", "r1", "r2", "r3", "r4", "r5"],
        )
        self.assertEqual(result.n_runs, 4)
        self.assertEqual(
            result.run_spans(),
            (
                (0, 0, 1, 0, 1),
                (1, 2, 2, 1, 1),
                (2, 3, 3, 0, 3),
                (3, 4, 5, 0, 1),
            ),
        )
        run0 = result.rows[0]
        self.assertEqual(run0.theta_pre, result.rows[1].theta_pre)
        self.assertEqual(run0.theta_post, result.rows[1].theta_post)
        self.assertEqual(result.rows[2].run_id, 1)
        self.assertEqual(result.rows[2].theta_pre, 0.0)
        self.assertEqual(result.rows[4].run_id, result.rows[5].run_id)
        self.assertNotEqual(result.rows[4].run_id, result.rows[0].run_id)
        self.assertEqual(result.rows[3].theta_pre, result.rows[1].theta_post)

    def test_first_run_matches_anchor_update(self) -> None:
        model = IRTModel(2, 3)
        result = model.run_streaming(_physical_fixture(), phase="train")
        self.assertEqual(result.rows[0].p_global, 0.5)
        self.assertEqual(result.rows[1].p_global, 0.5)
        self.assertEqual(result.rows[0].probability, result.rows[0].p_global)
        self.assertEqual(result.rows[0].theta_post, 0.0)
        self.assertEqual(result.rows[0].b_post, 0.02 * -0.5)
        self.assertEqual(result.rows[1].b_post, 0.02 * 0.5)
        self.assertEqual(result.rows[3].theta_pre, 0.0)
        self.assertNotEqual(result.rows[3].b_pre, result.rows[0].b_post)
        z = result.rows[3].theta_pre - result.rows[3].b_pre
        expected = float(sigmoid(np.asarray([z], dtype=np.float64))[0])
        self.assertEqual(result.rows[3].p_global, expected)
        self.assertNotEqual(result.rows[3].p_global, 0.5)

    def test_duplicate_item_inside_run_keeps_vectorized_last_write(self) -> None:
        rows = [
            _row(0, 0, 7, 1, "a"),
            _row(0, 0, 7, 0, "b"),
        ]
        model = IRTModel(1, 1)
        result = model.run_streaming(rows, phase="train")
        index = np.array([0, 0], dtype=np.int64)
        b = np.zeros(1, dtype=np.float64)
        p = np.array([0.5, 0.5], dtype=np.float64)
        y = np.array([1.0, 0.0], dtype=np.float64)
        b_grad = -(y - p) - 1e-4 * b[index]
        b[index] = b[index] + 0.02 * b_grad
        self.assertEqual(model.b[0], b[0])
        self.assertEqual(result.rows[0].b_post, float(b[0]))
        self.assertEqual(result.rows[1].b_post, float(b[0]))
        self.assertEqual(model.theta[0], 0.0)

    def test_eval_freezes_b_and_updates_theta(self) -> None:
        model = IRTModel(1, 1)
        train_row = [_row(0, 0, 1, 1, "train")]
        model.run_streaming(train_row, phase="train")
        b_after_train = model.b.copy()
        theta_after_train = float(model.theta[0])
        self.assertNotEqual(b_after_train[0], 0.0)
        result = model.run_streaming(train_row, phase="eval")
        self.assertEqual(model.b[0], b_after_train[0])
        self.assertEqual(result.rows[0].b_pre, result.rows[0].b_post)
        self.assertNotEqual(model.theta[0], theta_after_train)
        self.assertGreater(model.theta[0], theta_after_train)

    def test_tags_do_not_move_the_irt_anchor(self) -> None:
        plain = [_row(0, 0, 1, 1, "a"), _row(0, 1, 2, 0, "b")]
        tagged = [
            StreamingRow(0, 0, 1, 1, (0, 1, 0), "a"),
            StreamingRow(0, 1, 2, 0, (1,), "b"),
        ]
        plain_model = IRTModel(1, 2, n_skills=4)
        tagged_model = IRTModel(1, 2, n_skills=4)
        plain_out = plain_model.run_streaming(plain, phase="train")
        tagged_out = tagged_model.run_streaming(tagged, phase="train")
        self.assertEqual(
            [row.p_global for row in plain_out.rows],
            [row.p_global for row in tagged_out.rows],
        )
        self.assertTrue(np.array_equal(plain_model.theta, tagged_model.theta))
        self.assertTrue(np.array_equal(plain_model.b, tagged_model.b))
        self.assertTrue(np.all(tagged_model.r_post == 0.0))
        self.assertEqual(tagged_out.rows[0].p_art, tagged_out.rows[0].p_global)

    def test_metrics_follow_legacy_clip_and_single_class(self) -> None:
        y = np.array([1.0, 0.0], dtype=np.float64)
        p = np.array([0.0, 1.0], dtype=np.float64)
        clipped = np.clip(p, 1e-12, 1.0 - 1e-12)
        expected_nll = float(
            -(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)).mean()
        )
        self.assertEqual(metric_nll(y, p), expected_nll)
        self.assertNotEqual(metric_nll(np.array([1.0]), np.array([0.0])), 0.0)
        self.assertEqual(metric_brier(y, p), 1.0)
        self.assertEqual(metric_auc(np.array([1.0, 1.0]), np.array([0.2, 0.9])), 0.5)
        tied = metric_auc(
            np.array([0.0, 1.0, 1.0]),
            np.array([0.2, 0.4, 0.4]),
        )
        self.assertEqual(tied, 1.0)

    def test_phase_name_is_train_or_eval(self) -> None:
        with self.assertRaises(ValueError):
            IRTModel(1, 1).run_streaming([_row(0, 0, 1, 1, "a")], phase="test")
