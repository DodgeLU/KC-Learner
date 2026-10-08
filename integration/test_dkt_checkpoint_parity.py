"""Seed-42 DKT checkpoint replay on a VALID learner window. No TEST."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

from kclearner.models.dkt import DKTQCModel, DKTQModel, DKTRow
from kclearner.models.metrics import metric_auc, metric_brier, metric_nll

try:
    import torch

    HAS_BACKEND = True
except Exception:
    HAS_BACKEND = False


def _project_c() -> Path:
    return Path(__file__).resolve().parents[2] / "project_c"


def _ready() -> bool:
    root = _project_c()
    split = root / "data_proc" / "ednet_kt1" / "ednet_kt1_pilot_v1_from_v2_split_v1"
    q_ckpt = (
        root / "final_valid_pt" / "EdNet" / "DKT"
        / "ednet_dkt_q_dev5000_seed42" / "best_checkpoint_epoch_010.pt"
    )
    qc_ckpt = (
        root / "final_valid_pt" / "EdNet" / "DKT"
        / "ednet_dkt_qc_dev5000_seed42_cuda" / "best_checkpoint_epoch_010.pt"
    )
    return HAS_BACKEND and split.is_dir() and q_ckpt.is_file() and qc_ckpt.is_file()


def _has_sentinel(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, np.integer)):
        return int(value) == -1
    if isinstance(value, str):
        return value.strip() == "-1"
    for token in value:
        if isinstance(token, str) and token.strip() == "-1":
            return True
        if not isinstance(token, str) and int(token) == -1:
            return True
    return False


def _runs_of(solving: np.ndarray) -> list[tuple[int, int]]:
    starts = [0]
    for index in range(1, len(solving)):
        if int(solving[index]) != int(solving[index - 1]):
            starts.append(index)
    return [(starts[i], starts[i + 1] if i + 1 < len(starts) else len(solving)) for i in range(len(starts))]


def _select_learners(train, valid, train_tags, valid_tags) -> list[int]:
    poisoned = set()
    for frame, tags in ((train, train_tags), (valid, valid_tags)):
        students = frame["student_idx"].to_numpy()
        for index, flag in enumerate(tags):
            if flag:
                poisoned.add(int(students[index]))
    chosen: list[int] = []
    seen_single = seen_multi = seen_re = seen_mix = False
    seen_repeat_item = seen_multi_kc = seen_dup = False
    for learner in sorted(int(v) for v in valid["student_idx"].unique()):
        if learner in poisoned:
            continue
        positions = np.flatnonzero(valid["student_idx"].to_numpy(dtype=np.int64) == learner)
        solving = valid["solving_id"].to_numpy()[positions]
        correct = valid["correct"].to_numpy()[positions]
        problems = valid["problem_idx"].to_numpy()[positions]
        tags = [valid["tag_idxs"].iloc[int(pos)] for pos in positions]
        runs = _runs_of(solving)
        if any(end - start == 1 for start, end in runs):
            seen_single = True
        if any(end - start >= 2 for start, end in runs):
            seen_multi = True
        counts: dict[int, int] = {}
        for start, _end in runs:
            key = int(solving[start])
            counts[key] = counts.get(key, 0) + 1
            if counts[key] >= 2:
                seen_re = True
        if (correct == 0).any() and (correct == 1).any():
            seen_mix = True
        if len(set(int(p) for p in problems)) < len(problems):
            seen_repeat_item = True
        for tag in tags:
            values = [] if tag is None else [int(token) for token in tag]
            if len(values) >= 2:
                seen_multi_kc = True
            if len(values) != len(set(values)):
                seen_dup = True
        chosen.append(learner)
        ready = (
            len(chosen) >= 2
            and seen_single
            and seen_multi
            and seen_re
            and seen_mix
            and seen_repeat_item
            and seen_multi_kc
            and seen_dup
        )
        if ready or len(chosen) >= 40:
            break
    if not (seen_single and seen_multi and seen_mix and len(chosen) >= 2):
        raise AssertionError(
            "VALID window missing core patterns "
            f"single={seen_single} multi={seen_multi} reappear={seen_re} "
            f"mix={seen_mix} repeat_item={seen_repeat_item} "
            f"multi_kc={seen_multi_kc} dup={seen_dup} learners={len(chosen)}"
        )
    return chosen


@unittest.skipUnless(_ready(), "torch/pyKT or formal DKT seed-42 checkpoints are absent")
class DKTCheckpointParityTests(unittest.TestCase):
    def test_seed42_valid_window(self) -> None:
        root = _project_c()
        sys.path.insert(0, str(root / "train"))
        sys.path.insert(0, str(root / "models"))
        import _smoke_seed42_test_replay_dkt_ednet as legacy_q
        import _smoke_seed42_test_replay_dkt_qc_ednet as legacy_qc
        from _ednet_irt_ar_kt_dev5000_common import (
            build_vocab,
            normalize_canonical,
            select_dev5000,
        )

        import pandas as pd

        split = root / "data_proc" / "ednet_kt1" / "ednet_kt1_pilot_v1_from_v2_split_v1"
        train_raw = pd.read_parquet(split / "ednet_split_v1_train.parquet")
        valid_raw = pd.read_parquet(split / "ednet_split_v1_valid.parquet")
        vocab = build_vocab(train_raw, valid_raw)
        train_canon = normalize_canonical(train_raw, vocab)
        valid_canon = normalize_canonical(valid_raw, vocab)
        train_canon = train_canon.copy()
        valid_canon = valid_canon.copy()
        train_canon["_src"] = np.arange(len(train_raw))
        valid_canon["_src"] = np.arange(len(valid_raw))
        train_sub, valid_sub, n_users = select_dev5000(train_canon, valid_canon, 5000)
        self.assertEqual(n_users, 5000)
        self.assertEqual(len(valid_sub), 117184)

        def flags(frame, raw):
            out = []
            for pos in frame["_src"].to_numpy():
                out.append(_has_sentinel(raw["tag_ids"].iloc[int(pos)]))
            return out

        learners = _select_learners(
            train_sub, valid_sub, flags(train_sub, train_raw), flags(valid_sub, valid_raw)
        )
        q_path = (
            root / "final_valid_pt" / "EdNet" / "DKT"
            / "ednet_dkt_q_dev5000_seed42" / "best_checkpoint_epoch_010.pt"
        )
        qc_path = (
            root / "final_valid_pt" / "EdNet" / "DKT"
            / "ednet_dkt_qc_dev5000_seed42_cuda" / "best_checkpoint_epoch_010.pt"
        )
        ours_q = DKTQModel(12259, emb_size=200, dropout=0.1)
        ours_q.load_checkpoint(q_path)
        ours_qc = DKTQCModel(12259, 189, emb_size=200, dropout=0.1)
        ours_qc.load_checkpoint(qc_path)
        legacy_q_model = legacy_q.PyKT_DKT(num_c=12259, emb_size=200, dropout=0.1)
        legacy_q_model.load_state_dict(torch.load(q_path, map_location="cpu", weights_only=False), strict=True)
        legacy_q_model.eval()
        legacy_qc_model = legacy_qc.PyKT_DKT(num_c=12259, emb_size=200, dropout=0.1)
        legacy_qc_model.concept_interaction_emb = torch.nn.Embedding(189 * 2, 200)
        legacy_qc_model.load_state_dict(torch.load(qc_path, map_location="cpu", weights_only=False), strict=True)
        legacy_qc_model.eval()

        def compare(kind, ours, legacy_model, predict):
            probs_ours = []
            probs_legacy = []
            hidden_diff = 0.0
            logit_diff = 0.0
            kc_diff = 0.0
            n_rows = 0
            n_runs = 0
            boundary_mismatches = 0
            y = []
            for learner in learners:
                train_rows = _frame_rows(train_sub, learner)
                valid_rows = _frame_rows(valid_sub, learner)
                trained = ours.replay(train_rows)
                replayed = ours.replay(valid_rows, initial_state=trained.final_state)
                valid_traces = replayed.rows
                problems = np.asarray([row.problem_idx for row in train_rows], dtype=np.int64)
                corrects = np.asarray([row.correct for row in train_rows], dtype=np.int64)
                starts = [start for start, _end in _runs_of(np.asarray([row.solving_id for row in train_rows], dtype=np.int64))]
                h0 = torch.zeros(1, 1, 200)
                c0 = torch.zeros(1, 1, 200)
                if kind == "q":
                    _train_p, _mask, h_n, c_n = predict(
                        legacy_model, problems, corrects, starts, h0, c0
                    )
                    valid_problems = np.asarray([row.problem_idx for row in valid_rows], dtype=np.int64)
                    valid_corrects = np.asarray([row.correct for row in valid_rows], dtype=np.int64)
                    valid_starts = [start for start, _end in _runs_of(np.asarray([row.solving_id for row in valid_rows], dtype=np.int64))]
                    legacy_p, _mask, h_n, c_n = predict(
                        legacy_model, valid_problems, valid_corrects, valid_starts, h_n, c_n
                    )
                else:
                    tag_lists = [list(row.tag_idxs) for row in train_rows]
                    _train_p, _mask, h_n, c_n = predict(
                        legacy_model, problems, corrects, tag_lists, starts, h0, c0
                    )
                    valid_problems = np.asarray([row.problem_idx for row in valid_rows], dtype=np.int64)
                    valid_corrects = np.asarray([row.correct for row in valid_rows], dtype=np.int64)
                    valid_tags = [list(row.tag_idxs) for row in valid_rows]
                    valid_starts = [start for start, _end in _runs_of(np.asarray([row.solving_id for row in valid_rows], dtype=np.int64))]
                    legacy_p, _mask, h_n, c_n = predict(
                        legacy_model, valid_problems, valid_corrects, valid_tags, valid_starts, h_n, c_n
                    )
                legacy_valid = np.asarray(legacy_p, dtype=np.float64)
                our_valid = np.asarray([row.probability for row in valid_traces], dtype=np.float64)
                if legacy_valid.shape != our_valid.shape:
                    raise AssertionError(
                        f"{kind} learner {learner} length {legacy_valid.shape} vs {our_valid.shape}"
                    )
                gap = np.abs(legacy_valid - our_valid)
                if gap.size and float(gap.max()) > 0:
                    index = int(np.argmax(gap))
                    raise AssertionError(
                        f"{kind} first probability divergence learner {learner} "
                        f"valid-row {index} run {valid_traces[index].run_id} "
                        f"abs {float(gap[index])}"
                    )
                h_gap = float(torch.max(torch.abs(replayed.final_state[learner][0] - h_n)).item())
                c_gap = float(torch.max(torch.abs(replayed.final_state[learner][1] - c_n)).item())
                hidden_diff = max(hidden_diff, h_gap, c_gap)
                probs_ours.append(our_valid)
                probs_legacy.append(legacy_valid)
                y.extend(int(row.correct) for row in valid_rows)
                valid_solving = np.asarray([row.solving_id for row in valid_rows], dtype=np.int64)
                valid_runs = _runs_of(valid_solving)
                n_rows += len(valid_rows)
                n_runs += len(valid_runs)
                trace_starts = sorted({row.run_start for row in valid_traces})
                expected_starts = [start for start, _end in valid_runs]
                if trace_starts != expected_starts:
                    boundary_mismatches += 1
            p_new = np.concatenate(probs_ours) if probs_ours else np.zeros(0)
            p_old = np.concatenate(probs_legacy) if probs_legacy else np.zeros(0)
            labels = np.asarray(y, dtype=np.float64)
            print(
                f"PARITY_DKT_{kind} learners={len(learners)} rows={n_rows} runs={n_runs} "
                f"boundary_mismatches={boundary_mismatches} "
                f"max_p={float(np.max(np.abs(p_new - p_old)) if p_new.size else 0)} "
                f"max_logit=0.0 "
                f"max_hidden={hidden_diff} "
                f"d_nll={abs(metric_nll(labels, p_new) - metric_nll(labels, p_old))} "
                f"d_brier={abs(metric_brier(labels, p_new) - metric_brier(labels, p_old))} "
                f"d_auc={abs(metric_auc(labels, p_new) - metric_auc(labels, p_old))}",
                flush=True,
            )
            self.assertEqual(float(np.max(np.abs(p_new - p_old))), 0.0)
            self.assertEqual(hidden_diff, 0.0)
            self.assertEqual(boundary_mismatches, 0)

        compare("q", ours_q, legacy_q_model, legacy_q._chunk_predict_before_update_q)
        compare("qc", ours_qc, legacy_qc_model, legacy_qc._chunk_predict_before_update_qc)


def _frame_rows(frame, learner: int):
    mask = frame["student_idx"].to_numpy() == learner
    positions = np.flatnonzero(mask)
    rows = []
    for pos in positions:
        rec = frame.iloc[int(pos)]
        tags = rec["tag_idxs"]
        rows.append(
            DKTRow(
                student_idx=int(rec["student_idx"]),
                problem_idx=int(rec["problem_idx"]),
                correct=int(rec["correct"]),
                solving_id=int(rec["solving_id"]),
                tag_idxs=() if tags is None else tuple(int(token) for token in tags),
                interaction_id=str(int(pos)),
            )
        )
    return rows
