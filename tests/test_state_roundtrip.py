"""Native save/load round trips. No EdNet data."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from kclearner.models import ARKTModel, IRTModel, StreamingRow


def _irt_rows() -> tuple[list[StreamingRow], list[StreamingRow]]:
    prefix = [
        StreamingRow(0, 0, 1, 1, interaction_id="p0"),
        StreamingRow(0, 1, 1, 0, interaction_id="p1"),
        StreamingRow(1, 0, 1, 1, interaction_id="p2"),
    ]
    suffix = [
        StreamingRow(0, 0, 3, 0, interaction_id="s0"),
        StreamingRow(0, 1, 1, 1, interaction_id="s1"),
        StreamingRow(0, 1, 1, 0, interaction_id="s2"),
    ]
    return prefix, suffix


def _ar_rows() -> tuple[list[StreamingRow], list[StreamingRow]]:
    prefix = [
        StreamingRow(
            0, 0, 1, 1, (0, 1, 0), "p0", timestamp_ms=5, group_last_timestamp_ms=5
        ),
        StreamingRow(
            0, 1, 1, 1, (2,), "p1", timestamp_ms=8, group_last_timestamp_ms=9
        ),
    ]
    suffix = [
        StreamingRow(
            0, 0, 2, 1, (0, 0), "s0", timestamp_ms=12, group_last_timestamp_ms=12
        ),
        StreamingRow(
            1, 2, 4, 0, (2,), "s1", timestamp_ms=3, group_last_timestamp_ms=3
        ),
    ]
    return prefix, suffix


class StateRoundTripTests(unittest.TestCase):
    def test_irt_suffix_matches_uninterrupted_model(self) -> None:
        prefix, suffix = _irt_rows()
        running = IRTModel(2, 3, theta_lr=0.2, b_lr=0.03)
        running.run_streaming(prefix, phase="train")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "irt.npz"
            running.save_state(path)
            restored = IRTModel.load_state(path)
        continued = running.run_streaming(suffix, phase="train")
        resumed = restored.run_streaming(suffix, phase="train")
        self.assertEqual(
            [row.p_global for row in continued.rows],
            [row.p_global for row in resumed.rows],
        )
        self.assertTrue(np.array_equal(running.theta, restored.theta))
        self.assertTrue(np.array_equal(running.b, restored.b))
        self.assertTrue(np.array_equal(running.r_post, restored.r_post))

    def test_irt_eval_suffix_keeps_frozen_b(self) -> None:
        prefix, suffix = _irt_rows()
        running = IRTModel(2, 3)
        running.run_streaming(prefix, phase="train")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "irt.npz"
            running.save_state(path)
            restored = IRTModel.load_state(path)
        running.run_streaming(suffix, phase="eval")
        restored.run_streaming(suffix, phase="eval")
        self.assertTrue(np.array_equal(running.b, restored.b))
        self.assertTrue(np.array_equal(running.theta, restored.theta))

    def test_ar_kt_suffix_restores_residual_arrays(self) -> None:
        prefix, suffix = _ar_rows()
        running = ARKTModel(2, 4, n_skills=3, eta_r=0.20)
        running.run_streaming(prefix, phase="train")
        self.assertFalse(np.all(running.r_post == 0.0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ar.npz"
            running.save_state(path)
            restored = ARKTModel.load_state(path)
        self.assertTrue(np.array_equal(running.r_post, restored.r_post))
        self.assertTrue(np.array_equal(running.r_seen, restored.r_seen))
        self.assertTrue(
            np.array_equal(running.last_r_update_ts, restored.last_r_update_ts)
        )
        continued = running.run_streaming(suffix, phase="train")
        resumed = restored.run_streaming(suffix, phase="train")
        self.assertEqual(
            [row.p_art for row in continued.rows],
            [row.p_art for row in resumed.rows],
        )
        self.assertEqual(
            [row.p_global for row in continued.rows],
            [row.p_global for row in resumed.rows],
        )
        self.assertTrue(np.array_equal(running.theta, restored.theta))
        self.assertTrue(np.array_equal(running.b, restored.b))
        self.assertTrue(np.array_equal(running.r_post, restored.r_post))
        self.assertTrue(np.array_equal(running.r_seen, restored.r_seen))
        self.assertTrue(
            np.array_equal(running.last_r_update_ts, restored.last_r_update_ts)
        )

    def test_native_file_rejects_the_other_model(self) -> None:
        model = IRTModel(1, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "irt.npz"
            model.save_state(path)
            with self.assertRaises(ValueError):
                ARKTModel.load_state(path)

    def test_legacy_irt_checkpoint_shape_restores_without_residual_arrays(self) -> None:
        theta = np.array([0.25, -0.5], dtype=np.float64)
        b = np.array([0.125], dtype=np.float64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy_irt.npz"
            np.savez_compressed(
                path,
                theta=theta,
                b=b,
                n_students=np.int64(2),
                n_problems=np.int64(1),
                n_skills=np.int64(3),
                eta_r=np.float64(0.0),
                use_residual=np.bool_(False),
                use_time=np.bool_(False),
                theta_lr=np.float64(0.05),
                b_lr=np.float64(0.02),
                theta_l2=np.float64(1e-4),
                b_l2=np.float64(1e-4),
                learn_b=np.bool_(True),
                freeze_global_in_eval=np.bool_(True),
                model_id=np.str_("ednet_irt_anchor_v1"),
            )
            model = IRTModel.load_legacy_checkpoint(path)
        self.assertTrue(np.array_equal(model.theta, theta))
        self.assertTrue(np.array_equal(model.b, b))
        self.assertTrue(np.all(model.r_post == 0.0))
        self.assertTrue(np.all(model.last_r_update_ts == -1))
        self.assertFalse(np.any(model.r_seen))

    def test_legacy_ar_kt_checkpoint_restores_residual_arrays(self) -> None:
        theta = np.zeros(1, dtype=np.float64)
        b = np.zeros(1, dtype=np.float64)
        r_post = np.array([[0.2, 0.0, -0.1]], dtype=np.float64)
        r_seen = np.array([[True, False, True]])
        last_ts = np.array([[15, -1, 9]], dtype=np.int64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy_ar.npz"
            np.savez_compressed(
                path,
                theta=theta,
                b=b,
                r_post=r_post,
                r_seen=r_seen,
                last_r_update_ts=last_ts,
                n_students=np.int64(1),
                n_problems=np.int64(1),
                n_skills=np.int64(3),
                eta_r=np.float64(0.20),
                use_residual=np.bool_(True),
                use_time=np.bool_(False),
                model_id=np.str_("ednet_ar_kt_v0_3"),
            )
            model = ARKTModel.load_legacy_checkpoint(path)
        self.assertEqual(model.eta_r, 0.20)
        self.assertTrue(np.array_equal(model.r_post, r_post))
        self.assertTrue(np.array_equal(model.r_seen, r_seen))
        self.assertTrue(np.array_equal(model.last_r_update_ts, last_ts))

    def test_legacy_time_decay_checkpoint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timed.npz"
            np.savez_compressed(
                path,
                theta=np.zeros(1),
                b=np.zeros(1),
                n_students=np.int64(1),
                n_problems=np.int64(1),
                n_skills=np.int64(1),
                eta_r=np.float64(0.0),
                use_residual=np.bool_(False),
                use_time=np.bool_(True),
                model_id=np.str_("ednet_irt_anchor_v1"),
            )
            with self.assertRaises(ValueError):
                IRTModel.load_legacy_checkpoint(path)
