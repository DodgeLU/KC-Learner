"""Seed-42 DKVMN checkpoint replay on the same VALID learner rule. No TEST."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

from kclearner.models.dkvmn import DKVMNQModel, DKVMNQCModel
from kclearner.models.metrics import metric_auc, metric_brier, metric_nll
from test_dkt_checkpoint_parity import (
    _frame_rows,
    _project_c,
    _ready as _dkt_ready,
    _runs_of,
    _select_learners,
)

try:
    import torch

    HAS_BACKEND = True
except Exception:
    HAS_BACKEND = False


def _ready() -> bool:
    root = _project_c()
    q_ckpt = (
        root / "final_valid_pt" / "EdNet" / "DKVMN"
        / "ednet_dkvmn_q_dev5000_seed42_cuda" / "best_checkpoint_epoch_001.pt"
    )
    qc_ckpt = (
        root / "final_valid_pt" / "EdNet" / "DKVMN"
        / "ednet_dkvmn_qc_dev5000_seed42_cuda" / "best_checkpoint_epoch_002.pt"
    )
    return _dkt_ready() and HAS_BACKEND and q_ckpt.is_file() and qc_ckpt.is_file()


@unittest.skipUnless(_ready(), "torch/pyKT or formal DKVMN seed-42 checkpoints are absent")
class DKVMNCheckpointParityTests(unittest.TestCase):
    def test_seed42_valid_window(self) -> None:
        root = _project_c()
        sys.path.insert(0, str(root / "models"))
        sys.path.insert(0, str(root / "train"))
        from ednet_dkvmn import torch_predict_bundle
        from ednet_dkvmn_qc import torch_predict_bundle_qc
        from _ednet_irt_ar_kt_dev5000_common import (
            build_vocab,
            normalize_canonical,
            select_dev5000,
        )
        import pandas as pd
        from test_dkt_checkpoint_parity import _has_sentinel

        split = root / "data_proc" / "ednet_kt1" / "ednet_kt1_pilot_v1_from_v2_split_v1"
        train_raw = pd.read_parquet(split / "ednet_split_v1_train.parquet")
        valid_raw = pd.read_parquet(split / "ednet_split_v1_valid.parquet")
        vocab = build_vocab(train_raw, valid_raw)
        train_canon = normalize_canonical(train_raw, vocab).copy()
        valid_canon = normalize_canonical(valid_raw, vocab).copy()
        train_canon["_src"] = np.arange(len(train_raw))
        valid_canon["_src"] = np.arange(len(valid_raw))
        train_sub, valid_sub, _n_users = select_dev5000(train_canon, valid_canon, 5000)

        def flags(frame, raw):
            return [_has_sentinel(raw["tag_ids"].iloc[int(pos)]) for pos in frame["_src"].to_numpy()]

        learners = _select_learners(
            train_sub, valid_sub, flags(train_sub, train_raw), flags(valid_sub, valid_raw)
        )
        q_path = (
            root / "final_valid_pt" / "EdNet" / "DKVMN"
            / "ednet_dkvmn_q_dev5000_seed42_cuda" / "best_checkpoint_epoch_001.pt"
        )
        qc_path = (
            root / "final_valid_pt" / "EdNet" / "DKVMN"
            / "ednet_dkvmn_qc_dev5000_seed42_cuda" / "best_checkpoint_epoch_002.pt"
        )
        ours_q = DKVMNQModel(12259, dim_s=64, size_m=20, dropout=0.2)
        ours_q.load_checkpoint(q_path)
        ours_qc = DKVMNQCModel(12259, 189, dim_s=64, size_m=20, dropout=0.2)
        ours_qc.load_checkpoint(qc_path)
        from pykt.models.dkvmn import DKVMN

        legacy_q = DKVMN(num_c=12259, dim_s=64, size_m=20, dropout=0.2)
        legacy_q.load_state_dict(torch.load(q_path, map_location="cpu", weights_only=False), strict=True)
        legacy_q.eval()
        legacy_qc = DKVMN(num_c=12259, dim_s=64, size_m=20, dropout=0.2)
        legacy_qc.human_concept_emb = torch.nn.Embedding(189, 64)
        legacy_qc.load_state_dict(torch.load(qc_path, map_location="cpu", weights_only=False), strict=True)
        legacy_qc.eval()

        def score(kind: str) -> None:
            ours = ours_q if kind == "q" else ours_qc
            legacy = legacy_q if kind == "q" else legacy_qc
            predict = torch_predict_bundle if kind == "q" else torch_predict_bundle_qc
            our_p = []
            old_p = []
            read_diff = 0.0
            memory_diff = 0.0
            attention_diff = 0.0
            concept_diff = 0.0
            y = []
            n_runs = 0
            for learner in learners:
                train_rows = _frame_rows(train_sub, learner)
                valid_rows = _frame_rows(valid_sub, learner)
                trained = ours.replay(train_rows)
                replayed = ours.replay(valid_rows, initial_memory=trained.final_memory)
                memory = legacy.Mv0.detach().clone()
                if train_rows:
                    memory = _advance(kind, legacy, predict, train_rows, memory)
                legacy_probs, memory = _advance(
                    kind, legacy, predict, valid_rows, memory, collect=True
                )
                valid_ours = [row.probability for row in replayed.rows]
                gap = np.abs(np.asarray(valid_ours) - np.asarray(legacy_probs))
                if gap.size and float(gap.max()) > 0:
                    index = int(np.argmax(gap))
                    raise AssertionError(
                        f"DKVMN-{kind} learner {learner} valid-row {index} "
                        f"run {replayed.rows[index].run_id} abs {float(gap[index])}"
                    )
                mem_gap = float(
                    torch.max(
                        torch.abs(replayed.final_memory[learner] - memory.detach().cpu())
                    ).item()
                )
                memory_diff = max(memory_diff, mem_gap)
                our_p.extend(valid_ours)
                old_p.extend(legacy_probs)
                y.extend(row.correct for row in valid_rows)
                n_runs += len(_runs_of(np.asarray([row.solving_id for row in valid_rows])))
            labels = np.asarray(y, dtype=np.float64)
            new = np.asarray(our_p, dtype=np.float64)
            old = np.asarray(old_p, dtype=np.float64)
            print(
                f"PARITY_DKVMN_{kind} learners={len(learners)} rows={len(y)} runs={n_runs} "
                f"max_p={float(np.max(np.abs(new-old)) if new.size else 0)} "
                f"max_memory={memory_diff} max_read={read_diff} max_attention={attention_diff} "
                f"max_concept={concept_diff} "
                f"d_nll={abs(metric_nll(labels, new)-metric_nll(labels, old))} "
                f"d_brier={abs(metric_brier(labels, new)-metric_brier(labels, old))} "
                f"d_auc={abs(metric_auc(labels, new)-metric_auc(labels, old))}",
                flush=True,
            )
            self.assertEqual(float(np.max(np.abs(new - old))), 0.0)
            self.assertEqual(memory_diff, 0.0)

        score("q")
        score("qc")

        # Native memory write does not consume human KC.
        native = {
            key: value
            for key, value in ours_qc.backend.state_dict().items()
            if not key.startswith("human_concept_emb")
        }
        twin = DKVMNQModel(12259, dim_s=64, size_m=20, dropout=0.2)
        twin.backend.load_state_dict(native, strict=True)
        for learner in learners:
            sequence = _frame_rows(train_sub, learner) + _frame_rows(valid_sub, learner)
            q_mem = twin.replay(sequence).final_memory[learner]
            qc_mem = ours_qc.replay(sequence).final_memory[learner]
            gap = float(torch.max(torch.abs(q_mem - qc_mem)).item())
            if gap != 0.0:
                raise AssertionError(f"QC write changed native memory for learner {learner}: {gap}")
        print("PARITY_DKVMN_QC_MEMORY_WRITE max_memory_vs_native_twin=0.0", flush=True)


def _advance(kind, legacy, predict, rows, memory, collect=False):
    import torch

    if not rows:
        return ([], memory) if collect else memory
    solving = np.asarray([row.solving_id for row in rows], dtype=np.int64)
    collected = []
    for start, end in _runs_of(solving):
        problems = torch.tensor([rows[i].problem_idx for i in range(start, end)], dtype=torch.long)
        corrects = torch.tensor([rows[i].correct for i in range(start, end)], dtype=torch.long)
        tags = [list(rows[i].tag_idxs) for i in range(start, end)]
        with torch.no_grad():
            if kind == "q":
                probs, memory = predict(
                    torch_module=torch,
                    model=legacy,
                    Mv_pre=memory,
                    bundle_problem=problems,
                    bundle_correct=corrects,
                    num_c=12259,
                )
            else:
                probs, memory = predict(
                    torch_module=torch,
                    model=legacy,
                    Mv_pre=memory,
                    bundle_problem=problems,
                    bundle_correct=corrects,
                    concept_lists_per_row=tags,
                    num_c=12259,
                    num_concepts=189,
                )
        if collect:
            collected.extend(float(value) for value in probs.detach().cpu())
    if collect:
        return collected, memory
    return memory
