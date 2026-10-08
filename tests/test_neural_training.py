"""Tiny neural training checks. No full corrected EdNet run."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from kclearner.experiments.neural_training import (
    DeviceError,
    EarlyStopState,
    fit_neural,
    improved,
    load_checkpoint,
    resolve_device,
)
from kclearner.experiments.runner import FormalRunBlocked
from kclearner.models.dkt import DKTRow

try:
    import torch

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


def _rows() -> tuple[list[DKTRow], list[DKTRow]]:
    train = [
        DKTRow(0, 0, 1, 1, (0, 0), "t0"),
        DKTRow(0, 1, 0, 1, (1,), "t1"),
        DKTRow(0, 0, 1, 2, (0,), "t2"),
        DKTRow(1, 1, 0, 3, (1, 1), "t3"),
    ]
    valid = [
        DKTRow(0, 1, 1, 4, (0,), "v0"),
        DKTRow(1, 0, 0, 5, (1,), "v1"),
    ]
    return train, valid


def _recipe(kind: str) -> dict:
    shared = {
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "effective_chunk_size": 256,
        "tbptt_target_rows": 200,
        "max_epochs": 1,
        "early_stop_patience_epochs": 10,
        "checkpoint_min_delta": 0.001,
    }
    if kind == "dkt":
        shared.update({"emb_size": 8, "hidden_size": 8, "dropout": 0.0})
    else:
        shared.update({"dim_s": 8, "size_m": 4, "dropout": 0.0})
    return shared


@unittest.skipUnless(HAS_TORCH, "torch/pyKT are not importable")
class NeuralTrainingTests(unittest.TestCase):
    def test_early_stopping_uses_legacy_min_delta(self) -> None:
        self.assertTrue(improved(0.50, -1.0, 0.001))
        self.assertFalse(improved(0.5005, 0.50, 0.001))
        self.assertTrue(improved(0.5011, 0.50, 0.001))
        state = EarlyStopState()
        self.assertFalse(state.observe(0, 0.40, 0.7, 0.2, 0.001, 2))
        self.assertEqual(state.patience_counter, 0)
        self.assertFalse(state.observe(1, 0.4005, 0.7, 0.2, 0.001, 2))
        self.assertEqual(state.patience_counter, 1)
        self.assertTrue(state.observe(2, 0.4005, 0.7, 0.2, 0.001, 2))

    def test_cuda_request_fails_when_unavailable(self) -> None:
        if torch.cuda.is_available():
            self.assertEqual(resolve_device("cuda").type, "cuda")
            return
        with self.assertRaises(DeviceError):
            resolve_device("cuda")
        self.assertEqual(resolve_device("cpu").type, "cpu")
        self.assertEqual(resolve_device("auto").type, "cpu")

    def test_cpu_dkt_and_dkvmn_smoke(self) -> None:
        from kclearner.models.dkt import DKTQModel
        from kclearner.models.dkvmn import DKVMNQModel

        train, valid = _rows()
        for name, model, kind in (
            ("dkt_q", DKTQModel(4, emb_size=8, dropout=0.0), "dkt"),
            ("dkvmn_q", DKVMNQModel(4, dim_s=8, size_m=4, dropout=0.0), "dkvmn"),
        ):
            before = [tensor.detach().clone() for tensor in model.backend.parameters()]
            with tempfile.TemporaryDirectory() as directory:
                result = fit_neural(
                    model,
                    train,
                    valid,
                    _recipe(kind),
                    seed=42,
                    requested_device="cpu",
                    run_dir=directory,
                    config_hash="smoke",
                    dataset_id="synthetic",
                    cohort_hash="none",
                    vocab_hashes={},
                    max_epochs=1,
                )
                self.assertEqual(result["history"][0]["epoch"], 0)
                self.assertTrue((Path(directory) / "training_log.csv").is_file())
                checkpoints = list(Path(directory).glob("best_checkpoint_epoch_*.pt"))
                self.assertEqual(len(checkpoints), 1)
                fresh = type(model)(4, emb_size=8, dropout=0.0) if name == "dkt_q" else type(model)(4, dim_s=8, size_m=4, dropout=0.0)
                payload = load_checkpoint(checkpoints[0], fresh)
                self.assertEqual(payload["epoch"], 0)
                self.assertEqual(payload["seed"], 42)
            changed = any(not torch.equal(old, new.detach().cpu()) for old, new in zip(before, model.backend.parameters()))
            self.assertTrue(changed, name)

    def test_formal_row_cap_is_locked(self) -> None:
        from kclearner.models.dkt import DKTQModel

        model = DKTQModel(2, emb_size=4, dropout=0.0)
        rows = [DKTRow(0, 0, 1, 1, (), str(index)) for index in range(65)]
        with self.assertRaises(FormalRunBlocked):
            fit_neural(
                model,
                rows,
                rows[:2],
                _recipe("dkt"),
                seed=42,
                requested_device="cpu",
                run_dir=Path("unused"),
                config_hash="x",
                dataset_id="synthetic",
                cohort_hash="none",
                vocab_hashes={},
            )

    def test_dkt_step_matches_target_logit_update(self) -> None:
        from kclearner.experiments.neural_training import dkt_slab_loss
        from kclearner.models.dkt import DKTQModel

        torch.manual_seed(42)
        left = DKTQModel(4, emb_size=8, dropout=0.0)
        right = DKTQModel(4, emb_size=8, dropout=0.0)
        right.backend.load_state_dict(left.backend.state_dict())
        train, _valid = _rows()
        left_opt = torch.optim.Adam(left.backend.parameters(), lr=0.001, weight_decay=0.0)
        right_opt = torch.optim.Adam(right.backend.parameters(), lr=0.001, weight_decay=0.0)
        left.backend.train()
        loss, _state = dkt_slab_loss(left, [train], {})
        loss.backward()
        left_opt.step()
        right.backend.train()
        loss_right = _legacy_dkt_loss(right, train)
        loss_right.backward()
        right_opt.step()
        self.assertAlmostEqual(float(loss.detach()), float(loss_right.detach()), places=6)
        for a, b in zip(left.backend.parameters(), right.backend.parameters()):
            self.assertTrue(torch.allclose(a, b, atol=1e-6, rtol=1e-6))

    def test_dkvmn_step_matches_bundle_bce(self) -> None:
        from kclearner.experiments.neural_training import dkvmn_chunk_loss
        from kclearner.models.dkvmn import DKVMNQModel, predict_run

        torch.manual_seed(7)
        left = DKVMNQModel(4, dim_s=8, size_m=4, dropout=0.0)
        right = DKVMNQModel(4, dim_s=8, size_m=4, dropout=0.0)
        right.backend.load_state_dict(left.backend.state_dict())
        train, _valid = _rows()
        left_opt = torch.optim.Adam(left.backend.parameters(), lr=0.001)
        right_opt = torch.optim.Adam(right.backend.parameters(), lr=0.001)
        loss, _memory = dkvmn_chunk_loss(left, train, left.backend.Mv0)
        loss.backward()
        left_opt.step()
        loss_right = _legacy_dkvmn_loss(right, train, predict_run)
        loss_right.backward()
        right_opt.step()
        self.assertAlmostEqual(float(loss.detach()), float(loss_right.detach()), places=6)
        for a, b in zip(left.backend.parameters(), right.backend.parameters()):
            self.assertTrue(torch.allclose(a, b, atol=1e-6, rtol=1e-6))

    def test_qc_gradients(self) -> None:
        from kclearner.experiments.neural_training import dkt_slab_loss, dkvmn_chunk_loss
        from kclearner.models.dkt import DKTQCModel
        from kclearner.models.dkvmn import DKVMNQCModel

        dkt = DKTQCModel(4, 3, emb_size=8, dropout=0.0)
        rows = [DKTRow(0, 1, 1, 1, (0, 0), "a"), DKTRow(0, 2, 0, 2, (1,), "b")]
        loss, _state = dkt_slab_loss(dkt, [rows], {})
        loss.backward()
        self.assertGreater(float(dkt.backend.concept_interaction_emb.weight.grad.abs().sum()), 0.0)
        dkvmn = DKVMNQCModel(4, 3, dim_s=8, size_m=4, dropout=0.0)
        loss, memory = dkvmn_chunk_loss(dkvmn, rows, dkvmn.backend.Mv0)
        loss.backward()
        self.assertGreater(float(dkvmn.backend.human_concept_emb.weight.grad.abs().sum()), 0.0)
        self.assertTrue(memory.requires_grad)


def _legacy_dkt_loss(model, rows):
    import torch

    model.backend.train()
    problems = torch.tensor([row.problem_idx for row in rows], dtype=torch.long)
    corrects = torch.tensor([row.correct for row in rows], dtype=torch.long)
    inputs = model.backend.interaction_emb(problems + model.n_questions * corrects)
    h0 = torch.zeros(1, 1, model.backend.hidden_size)
    c0 = torch.zeros(1, 1, model.backend.hidden_size)
    outputs, _state = model.backend.lstm_layer(inputs.unsqueeze(0), (h0, c0))
    hidden = outputs.squeeze(0)
    losses = []
    starts = [0]
    for index, row in enumerate(rows[1:], start=1):
        if row.solving_id != rows[index - 1].solving_id:
            starts.append(index)
    starts.append(len(rows))
    for start, end in zip(starts, starts[1:]):
        h_pre = h0.squeeze(0).squeeze(0) if start == 0 else hidden[start - 1]
        h_drop = model.backend.dropout_layer(h_pre.unsqueeze(0)).squeeze(0)
        for index in range(start, end):
            target = rows[index].problem_idx
            logit = (model.backend.out_layer.weight[target] * h_drop).sum() + model.backend.out_layer.bias[target]
            label = torch.tensor(float(rows[index].correct))
            losses.append(torch.nn.functional.binary_cross_entropy_with_logits(logit, label))
    return torch.stack(losses).mean()


def _legacy_dkvmn_loss(model, rows, predict):
    import torch

    memory = model.backend.Mv0
    losses = []
    count = 0
    previous = None
    bundle = []
    def flush():
        nonlocal memory, count
        if not bundle:
            return
        problems = torch.tensor([row.problem_idx for row in bundle], dtype=torch.long)
        corrects = torch.tensor([row.correct for row in bundle], dtype=torch.long)
        probability, memory, *_rest = predict(torch, model.backend, memory, problems, corrects, None, model.n_questions)
        target = corrects.to(dtype=probability.dtype)
        losses.append(torch.nn.functional.binary_cross_entropy(probability, target, reduction="sum"))
        count += len(bundle)
        bundle.clear()
    for row in rows:
        if previous is not None and row.solving_id != previous:
            flush()
        bundle.append(row)
        previous = row.solving_id
    flush()
    return torch.stack(losses).sum() / float(count)


if __name__ == "__main__":
    unittest.main()
