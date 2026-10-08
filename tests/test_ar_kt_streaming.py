"""Synthetic AR-KT streaming semantics. No EdNet data."""

from __future__ import annotations

import unittest

import numpy as np

from kclearner.models import ARKTModel, IRTModel, StreamingRow
from kclearner.models.streaming import FORMAL_ETA_R


def _row(
    student: int,
    problem: int,
    group: int,
    correct: int,
    tags: tuple[int, ...],
    interaction_id: str,
) -> StreamingRow:
    return StreamingRow(
        student_idx=student,
        problem_idx=problem,
        group_id=group,
        correct=correct,
        tag_idxs=tags,
        interaction_id=interaction_id,
    )


def _fixture() -> list[StreamingRow]:
    return [
        _row(0, 0, 1, 1, (0, 1, 0), "a"),
        _row(0, 1, 1, 1, (2,), "b"),
        _row(0, 0, 2, 1, (0, 1, 0), "c"),
        _row(1, 2, 1, 0, (2,), "d"),
        _row(0, 1, 1, 0, (1,), "e"),
        _row(0, 3, 1, 1, (0,), "f"),
    ]


class ARKTStreamingTests(unittest.TestCase):
    def test_runs_cover_multi_kc_duplicate_token_and_reexposure(self) -> None:
        result = ARKTModel(2, 4, n_skills=3).run_streaming(_fixture(), phase="train")
        self.assertEqual(result.n_runs, 4)
        self.assertEqual(result.run_spans()[0], (0, 0, 1, 0, 1))
        self.assertEqual(result.run_spans()[1], (1, 2, 2, 0, 2))
        self.assertEqual(result.run_spans()[2], (2, 3, 3, 1, 1))
        self.assertEqual(result.run_spans()[3], (3, 4, 5, 0, 1))
        self.assertEqual(result.rows[0].k_c, ((0, 2), (1, 1), (2, 1)))
        self.assertEqual(result.rows[0].kc_ids, (0, 1, 0))
        self.assertEqual(result.rows[0].r_post_pre, (0.0, 0.0, 0.0))
        self.assertEqual(result.rows[0].r_bar, 0.0)
        self.assertEqual(result.rows[0].e_group, 0.5)
        self.assertEqual(result.rows[0].probability, result.rows[0].p_art)
        self.assertEqual(result.rows[1].theta_pre, result.rows[0].theta_pre)
        self.assertEqual(result.rows[1].r_post_pre, (0.0,))

    def test_duplicate_token_changes_mean_and_k_c(self) -> None:
        result = ARKTModel(2, 4, n_skills=3).run_streaming(_fixture(), phase="train")
        eta = FORMAL_ETA_R
        e_group = 0.5
        r0 = eta * (e_group / 2.0)
        r1 = eta * (e_group / 1.0)
        r2 = eta * (e_group / 1.0)
        self.assertEqual(result.rows[0].r_post_post, (r0, r1, r0))
        self.assertEqual(result.rows[1].r_post_post, (r2,))
        self.assertNotEqual(r0, r1)
        exposed = result.rows[2]
        self.assertEqual(exposed.r_post_pre, (r0, r1, r0))
        pre = np.asarray(exposed.r_post_pre, dtype=np.float64)
        duplicated = float(np.mean(pre))
        deduplicated = float(np.mean(np.unique(pre)))
        self.assertEqual(exposed.r_bar, duplicated)
        self.assertNotEqual(exposed.r_bar, deduplicated)
        self.assertNotEqual(exposed.p_art, exposed.p_global)
        self.assertEqual(exposed.k_c, ((0, 2), (1, 1)))
        self.assertEqual(result.rows[5].kc_ids, (0,))
        self.assertNotEqual(result.rows[5].r_post_pre[0], 0.0)

    def test_eval_freezes_b_and_still_updates_residual(self) -> None:
        model = ARKTModel(1, 1, n_skills=1)
        row = [_row(0, 0, 1, 1, (0,), "a")]
        model.run_streaming(row, phase="train")
        b_after_train = model.b.copy()
        r_after_train = float(model.r_post[0, 0])
        theta_after_train = float(model.theta[0])
        self.assertNotEqual(r_after_train, 0.0)
        model.run_streaming(row, phase="eval")
        self.assertEqual(model.b[0], b_after_train[0])
        self.assertNotEqual(model.r_post[0, 0], r_after_train)
        self.assertNotEqual(model.theta[0], theta_after_train)

    def test_tag_free_anchor_matches_irt(self) -> None:
        rows = [
            StreamingRow(0, 0, 1, 1, (), "a"),
            StreamingRow(0, 1, 1, 0, (), "b"),
            StreamingRow(0, 0, 2, 1, (), "c"),
        ]
        irt = IRTModel(1, 2, n_skills=1)
        ar = ARKTModel(1, 2, n_skills=1)
        irt_out = irt.run_streaming(rows, phase="train")
        ar_out = ar.run_streaming(rows, phase="train")
        self.assertEqual(
            [row.p_global for row in irt_out.rows],
            [row.p_global for row in ar_out.rows],
        )
        self.assertEqual(
            [row.p_global for row in ar_out.rows],
            [row.p_art for row in ar_out.rows],
        )
        self.assertTrue(np.array_equal(irt.theta, ar.theta))
        self.assertTrue(np.array_equal(irt.b, ar.b))
        self.assertTrue(np.all(ar.r_post == 0.0))
