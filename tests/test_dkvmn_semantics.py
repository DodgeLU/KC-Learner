"""DKVMN run semantics. Torch checks run only when pyKT is importable."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from kclearner.models.dkvmn import DKVMNQModel, DKVMNQCModel
from kclearner.models.dkt import DKTRow

try:
    import torch

    HAS_BACKEND = True
except Exception:
    HAS_BACKEND = False


def _row(student, problem, correct, solving, tags=()):
    return DKTRow(student, problem, correct, solving, tags, f"{student}-{solving}")


@unittest.skipUnless(HAS_BACKEND, "torch and pykt-toolkit are not importable")
class DKVMNBackendTests(unittest.TestCase):
    def _rows(self) -> list[DKTRow]:
        return [
            _row(1, 0, 1, 7, (0,)),
            _row(0, 0, 1, 1, (0, 1, 0)),
            _row(0, 1, 0, 1, (1,)),
            _row(1, 1, 0, 7, ()),
            _row(0, 0, 1, 2, ()),
            _row(0, 2, 1, 1, (0, 0)),
        ]

    def test_same_pre_run_memory_and_qc_write_invariance(self) -> None:
        torch.manual_seed(1)
        question = DKVMNQModel(5, dim_s=8, size_m=4, dropout=0.2)
        torch.manual_seed(1)
        concept = DKVMNQCModel(5, n_concepts=3, dim_s=8, size_m=4, dropout=0.2)
        concept.backend.load_state_dict(question.backend.state_dict(), strict=False)
        with torch.no_grad():
            concept.backend.human_concept_emb.weight.normal_()
        q_out = question.replay(self._rows())
        c_out = concept.replay(self._rows())
        first_run = [row for row in q_out.rows if row.run_id == q_out.rows[1].run_id]
        self.assertGreater(len(first_run), 1)
        self.assertEqual(first_run[0].memory_checksum, first_run[1].memory_checksum)
        self.assertEqual(
            [row.attention_checksum for row in q_out.rows],
            [row.attention_checksum for row in c_out.rows],
        )
        for learner in q_out.final_memory:
            self.assertTrue(
                torch.equal(q_out.final_memory[learner], c_out.final_memory[learner])
            )
        self.assertFalse(
            np.allclose(
                [row.probability for row in q_out.rows],
                [row.probability for row in c_out.rows],
            )
        )
        self.assertEqual(c_out.rows[4].concept_checksum, 0.0)
        duplicated = [row for row in c_out.rows if row.kc_ids == (0, 1, 0)][0]
        self.assertNotEqual(duplicated.concept_checksum, 0.0)

    def test_runtime_continuation(self) -> None:
        torch.manual_seed(2)
        model = DKVMNQCModel(5, n_concepts=3, dim_s=8, size_m=4)
        rows = self._rows()
        full = model.replay(rows)
        prefix = model.replay(rows[:4])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dkvmn.pt"
            model.save_runtime(path, prefix.final_memory)
            restored, memory = DKVMNQCModel.load_runtime(path)
        suffix = restored.replay(rows[4:], initial_memory=memory)
        self.assertEqual(
            [row.probability for row in full.rows[4:]],
            [row.probability for row in suffix.rows],
        )
