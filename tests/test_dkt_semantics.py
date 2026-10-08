"""DKT segmentation and, when torch/pyKT exist, causal wrapper checks."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kclearner.models.dkt import DKTQModel, DKTQCModel, DKTRow
from kclearner.models.neural_runs import learner_solving_runs

try:
    import torch
    from pykt.models.dkt import DKT as _DKT  # noqa: F401

    HAS_BACKEND = True
except Exception:
    HAS_BACKEND = False


def _row(student, problem, correct, solving, tags=(), name=""):
    return DKTRow(student, problem, correct, solving, tags, name)


class NeuralRunTests(unittest.TestCase):
    def test_other_learners_do_not_split_a_run(self) -> None:
        students = [1, 0, 0, 1]
        solving = [5, 1, 1, 5]
        runs = learner_solving_runs(students, solving)
        by_learner = {}
        for run in runs:
            by_learner.setdefault(run.learner_index, []).append(run)
        self.assertEqual(len(by_learner[1]), 1)
        self.assertEqual(by_learner[1][0].row_positions, (0, 3))
        self.assertEqual(by_learner[0][0].row_positions, (1, 2))

    def test_reappearing_solving_id_is_a_new_run(self) -> None:
        runs = learner_solving_runs([0, 0, 0], [1, 2, 1])
        self.assertEqual(len(runs), 3)
        self.assertEqual(runs[0].solving_id, 1)
        self.assertEqual(runs[2].solving_id, 1)
        self.assertEqual(runs[0].row_positions, (0,))
        self.assertEqual(runs[2].row_positions, (2,))


@unittest.skipUnless(HAS_BACKEND, "torch and pykt-toolkit are not importable")
class DKTBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def _sequence(self) -> list[DKTRow]:
        return [
            _row(1, 0, 1, 5, (0, 1, 0), "a"),
            _row(0, 0, 1, 1, (1,), "b"),
            _row(0, 1, 0, 1, (1, 1), "c"),
            _row(1, 1, 0, 5, (), "d"),
            _row(0, 0, 1, 2, (2,), "e"),
            _row(0, 2, 1, 1, (0,), "f"),
        ]

    def test_runs_and_shared_pre_state(self) -> None:
        model = DKTQModel(4, emb_size=8, dropout=0.1)
        result = model.replay(self._sequence())
        learner0 = [row for row in result.rows if row.learner_index == 0]
        self.assertEqual(learner0[0].hidden_pre, learner0[1].hidden_pre)
        self.assertNotEqual(learner0[0].run_id, learner0[2].run_id)
        self.assertEqual(learner0[0].solving_id, learner0[-1].solving_id)
        self.assertNotEqual(learner0[0].run_id, learner0[-1].run_id)
        learner1 = [row for row in result.rows if row.learner_index == 1]
        self.assertEqual(len({row.run_id for row in learner1}), 1)
        self.assertEqual(learner1[0].hidden_pre, learner1[1].hidden_pre)

    def test_q_ignores_kc_and_current_label_does_not_change_its_score(self) -> None:
        base = [
            _row(0, 0, 1, 1, (0,), "p"),
            _row(0, 1, 0, 2, (1, 1), "q"),
        ]
        flipped_tags = [
            _row(0, 0, 1, 1, (3, 3, 1), "p"),
            _row(0, 1, 0, 2, (), "q"),
        ]
        flipped_label = [
            _row(0, 0, 0, 1, (0,), "p"),
            _row(0, 1, 0, 2, (1, 1), "q"),
        ]
        torch.manual_seed(0)
        model = DKTQModel(4, emb_size=8)
        original = model.replay(base)
        tagged = model.replay(flipped_tags)
        relabeled = model.replay(flipped_label)
        self.assertEqual(
            [row.probability for row in original.rows],
            [row.probability for row in tagged.rows],
        )
        self.assertEqual(original.rows[0].probability, relabeled.rows[0].probability)
        self.assertNotEqual(original.rows[1].probability, relabeled.rows[1].probability)

    def test_qc_keeps_duplicate_tokens_and_zero_missing_kc(self) -> None:
        torch.manual_seed(0)
        model = DKTQCModel(4, n_concepts=4, emb_size=8)
        with torch.no_grad():
            weight = model.backend.concept_interaction_emb.weight
            weight.copy_(torch.arange(weight.numel(), dtype=torch.float32).reshape_as(weight))
        duplicated = model.replay([_row(0, 0, 1, 1, (0, 1, 0), "d")])
        unique = model.replay([_row(0, 0, 1, 1, (0, 1), "u")])
        missing = model.replay([_row(0, 0, 1, 1, (), "m")])
        q_only = DKTQModel(4, emb_size=8)
        q_only.backend.load_state_dict(
            {
                key: value
                for key, value in model.backend.state_dict().items()
                if key in q_only.backend.state_dict()
            },
            strict=False,
        )
        q_trace = q_only.replay([_row(0, 0, 1, 1, (), "m")])
        self.assertNotEqual(
            duplicated.rows[0].kc_contribution_checksum,
            unique.rows[0].kc_contribution_checksum,
        )
        self.assertEqual(missing.rows[0].kc_contribution_checksum, 0.0)
        self.assertEqual(missing.rows[0].probability, q_trace.rows[0].probability)
        self.assertEqual(len(duplicated.rows), 1)

    def test_checkpoint_continuation_matches_uninterrupted_replay(self) -> None:
        torch.manual_seed(0)
        model = DKTQCModel(4, n_concepts=3, emb_size=8)
        rows = self._sequence()
        full = model.replay(rows)
        prefix = model.replay(rows[:4])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dkt.pt"
            model.save_runtime(path, prefix.final_state)
            restored, hidden = DKTQCModel.load_runtime(path)
        suffix = restored.replay(rows[4:], initial_state=hidden)
        self.assertEqual(
            [row.probability for row in full.rows[4:]],
            [row.probability for row in suffix.rows],
        )
        self.assertEqual(
            [row.logit for row in full.rows[4:]],
            [row.logit for row in suffix.rows],
        )
