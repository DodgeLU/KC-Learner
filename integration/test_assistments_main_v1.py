"""ASSISTments MAIN_V1 index identity and VALID checkpoint parity.

Reads ``primary.csv`` and TRAIN-only checkpoints. It does not score TEST
and it does not open ``final_test_results``.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

from kclearner.data.adapters.assistments2017 import (
    FROZEN_MAIN_V1_COUNTS,
    load_assistments2017_main_v1,
    rows_for_neural_state,
    to_streaming_rows,
)
from kclearner.models import ARKTModel, IRTModel
from kclearner.models.metrics import metric_auc, metric_brier, metric_nll

try:
    import torch

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


def _project_c() -> Path:
    return Path(__file__).resolve().parents[2] / "project_c"


def _primary() -> Path:
    return _project_c() / "data_proc" / "assistments2017" / "primary.csv"


def _irt_state() -> Path:
    return _project_c() / "final_valid_pt" / "assistments2017" / "IRT" / "main_v1" / "state_train_only.npz"


def _ar_state() -> Path:
    return _project_c() / "final_valid_pt" / "assistments2017" / "AR-KT" / "main_v1" / "state_train_only.npz"


def _ready() -> bool:
    return _primary().is_file() and _irt_state().is_file() and _ar_state().is_file()


def _max_abs(left, right) -> float:
    gap = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    if gap.size == 0:
        return 0.0
    return float(np.max(np.abs(gap)))


def _legacy_frame(rows):
    import pandas as pd

    return pd.DataFrame(
        {
            "student_idx": [row.student_idx for row in rows],
            "problem_idx": [row.problem_idx for row in rows],
            "skill_idx": [row.skill_idx for row in rows],
            "correct": [row.correct for row in rows],
            "timestamp": [row.timestamp for row in rows],
            "group_id": [row.group_id for row in rows],
            "dt_same_skill_diff_problem_sec": np.zeros(len(rows), dtype=np.int64),
        }
    )


def _put_legacy(legacy, ours, kind: str) -> None:
    legacy.theta[:] = ours.theta
    legacy.b[:] = ours.b
    if kind == "ar_kt":
        legacy.r_post[:] = ours.r_post
        legacy.r_seen[:] = ours.r_seen
        legacy.last_r_update_ts[:] = ours.last_r_update_ts


@unittest.skipUnless(_ready(), "ASSISTments primary.csv or MAIN_V1 checkpoints are absent")
class AssistmentsMainV1ParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        project = _project_c()
        sys.path.insert(0, str(project))
        sys.path.insert(0, str(project / "models"))
        from assist_dkt import AssistDKTQ

        cls.loaded = load_assistments2017_main_v1(_primary())
        legacy = AssistDKTQ(str(_primary()), seed=42, verbose=False)
        cls.legacy_train = legacy.train_df
        cls.legacy_valid = legacy.valid_df
        cls.legacy_students = int(legacy.n_students)
        cls.legacy_questions = int(legacy.n_questions_train_valid)
        cls.legacy_skills = int(legacy.n_skills_train_valid)

    def test_frozen_counts_and_index_identity(self) -> None:
        stats = self.loaded.diagnostics
        for key, expected in FROZEN_MAIN_V1_COUNTS.items():
            self.assertEqual(getattr(stats, key), expected, key)
        self.assertEqual(self.legacy_students, 1709)
        self.assertEqual(self.legacy_questions, 2909)
        self.assertEqual(self.legacy_skills, 102)
        self._assert_frame("train", self.loaded.train_rows, self.legacy_train)
        self._assert_frame("valid", self.loaded.valid_rows, self.legacy_valid)
        print(
            "ASSISTMENTS_INDEX "
            f"train_rows={stats.train_rows} valid_rows={stats.valid_rows} "
            f"test_rows={stats.test_rows} oov_rows={stats.oov_rows} "
            f"questions={stats.n_questions_vocab} skills={stats.n_skills_vocab} "
            f"students={stats.n_students_vocab} promoted={stats.promoted_to_valid}",
            flush=True,
        )

    def test_irt_valid_parity(self) -> None:
        summary = self._streaming_parity("irt", _irt_state(), IRTModel)
        recorded = json.loads(
            (_project_c() / "final_valid_pt" / "assistments2017" / "IRT" / "main_v1" / "result.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(summary["max_p_global"], 0.0)
        self.assertEqual(summary["max_theta"], 0.0)
        self.assertEqual(summary["max_b"], 0.0)
        self.assertEqual(summary["boundary_mismatches"], 0)
        self.assertAlmostEqual(summary["nll"], recorded["valid_nll"], places=12)
        self.assertAlmostEqual(summary["brier"], recorded["valid_brier"], places=12)
        self.assertAlmostEqual(summary["auc"], recorded["valid_auc"], places=12)

    def test_ar_kt_valid_parity(self) -> None:
        summary = self._streaming_parity("ar_kt", _ar_state(), ARKTModel)
        recorded = json.loads(
            (
                _project_c() / "final_valid_pt" / "assistments2017" / "AR-KT" / "main_v1" / "result.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(summary["max_p_global"], 0.0)
        self.assertEqual(summary["max_p_art"], 0.0)
        self.assertEqual(summary["max_r"], 0.0)
        self.assertEqual(summary["max_theta"], 0.0)
        self.assertEqual(summary["max_b"], 0.0)
        self.assertEqual(summary["r_seen_mismatches"], 0)
        self.assertEqual(summary["boundary_mismatches"], 0)
        self.assertEqual(summary["model"].residual_update, "assistments_skill_mean")
        self.assertAlmostEqual(summary["nll"], recorded["valid_nll"], places=12)
        self.assertAlmostEqual(summary["brier"], recorded["valid_brier"], places=12)
        self.assertAlmostEqual(summary["auc"], recorded["valid_auc"], places=12)
        self.assertAlmostEqual(summary["nll_global"], recorded["valid_nll_p_global"], places=12)

    def test_neural_seed42_valid_window(self) -> None:
        if not HAS_TORCH:
            self.skipTest("torch/pyKT are not importable")
        from kclearner.models.dkt import DKTQCModel, DKTQModel, DKTRow
        from kclearner.models.dkvmn import DKVMNQModel, DKVMNQCModel

        root = _project_c() / "final_valid_pt" / "assistments2017"
        paths = {
            "dkt_q": root / "DKT" / "seed_42" / "q" / "best_checkpoint_epoch_015.pt",
            "dkt_qc": root / "DKT" / "seed_42" / "qc" / "best_checkpoint_epoch_010.pt",
            "dkvmn_q": root / "DKVMN" / "seed_42" / "q" / "best_checkpoint_epoch_003.pt",
            "dkvmn_qc": root / "DKVMN" / "seed_42" / "qc" / "best_checkpoint_epoch_004.pt",
        }
        if any(not path.is_file() for path in paths.values()):
            self.skipTest("ASSISTments seed-42 neural checkpoints are absent")
        learners = _window_learners(self.loaded.valid_rows)
        train = _rows_for(self.loaded.train_rows, learners)
        valid = _rows_for(self.loaded.valid_rows, learners)
        self._dkt_parity("dkt_q", DKTQModel(2909, emb_size=200, dropout=0.1), paths["dkt_q"], train, valid, DKTRow)
        self._dkt_parity(
            "dkt_qc",
            DKTQCModel(2909, 102, emb_size=200, dropout=0.1),
            paths["dkt_qc"],
            train,
            valid,
            DKTRow,
        )
        self._dkvmn_parity(
            "dkvmn_q",
            DKVMNQModel(2909, dim_s=64, size_m=20, dropout=0.2),
            paths["dkvmn_q"],
            train,
            valid,
            DKTRow,
        )
        self._dkvmn_parity(
            "dkvmn_qc",
            DKVMNQCModel(2909, 102, dim_s=64, size_m=20, dropout=0.2),
            paths["dkvmn_qc"],
            train,
            valid,
            DKTRow,
        )

    def _assert_frame(self, name: str, rows, frame) -> None:
        self.assertEqual(len(rows), len(frame), name)
        for column, getter in (
            ("student_idx", lambda row: row.student_idx),
            ("problem_idx", lambda row: row.problem_idx),
            ("skill_idx", lambda row: row.skill_idx),
            ("correct", lambda row: row.correct),
            ("group_id", lambda row: row.group_id),
            ("timestamp", lambda row: row.timestamp),
        ):
            got = np.asarray([getter(row) for row in rows], dtype=np.int64)
            expected = frame[column].to_numpy(dtype=np.int64)
            mismatch = np.flatnonzero(got != expected)
            if mismatch.size:
                index = int(mismatch[0])
                self.fail(
                    f"{name} {column} mismatch at {index}: "
                    f"{int(got[index])} != {int(expected[index])}"
                )

    def _streaming_parity(self, kind: str, path: Path, model_cls):
        from art_kt_viability.models.art_kt import ARTKTConfig, ARTKTModel as LegacyARTKT

        ours = model_cls.load_legacy_checkpoint(path)
        cfg = ARTKTConfig(
            n_students=int(ours.theta.shape[0]),
            n_problems=int(ours.b.shape[0]),
            n_skills=int(ours.r_post.shape[1]),
            eta_r=0.0 if kind == "irt" else 0.20,
            use_residual=kind == "ar_kt",
            use_time=False,
            tau_sec=float("inf"),
        )
        legacy = LegacyARTKT(cfg)
        _put_legacy(legacy, ours, kind)
        rows = self.loaded.valid_rows
        streaming = to_streaming_rows(rows, family=kind)
        new = ours.run_streaming(streaming, phase="eval")
        old = legacy.run_streaming(_legacy_frame(rows), phase="eval")
        p_global = np.asarray([row.p_global for row in new.rows], dtype=np.float64)
        p_art = np.asarray([row.p_art for row in new.rows], dtype=np.float64)
        y = np.asarray([row.correct for row in rows], dtype=np.float64)
        boundaries = 0
        previous = None
        expected_runs = 0
        for row in rows:
            key = (row.student_idx, row.group_id)
            if key != previous:
                expected_runs += 1
                previous = key
        if new.n_runs != expected_runs:
            boundaries = abs(new.n_runs - expected_runs)
        headline = p_art if kind == "ar_kt" else p_global
        summary = {
            "model": ours,
            "rows": len(rows),
            "runs": new.n_runs,
            "max_p_global": _max_abs(old["p_global"], p_global),
            "max_p_art": _max_abs(old["p_art"], p_art),
            "max_r": _max_abs(old["r_pre"], [row.r_bar for row in new.rows]),
            "max_theta": _max_abs(legacy.theta, ours.theta),
            "max_b": _max_abs(legacy.b, ours.b),
            "r_seen_mismatches": int(np.sum(legacy.r_seen != ours.r_seen)),
            "boundary_mismatches": boundaries,
            "nll": float(metric_nll(y, headline)),
            "brier": float(metric_brier(y, headline)),
            "auc": float(metric_auc(y, headline)),
            "nll_global": float(metric_nll(y, p_global)),
        }
        print(
            f"PARITY_ASSIST_{kind} rows={summary['rows']} runs={summary['runs']} "
            f"max_p_global={summary['max_p_global']} max_p_art={summary['max_p_art']} "
            f"max_r={summary['max_r']} max_theta={summary['max_theta']} "
            f"max_b={summary['max_b']} r_seen_mismatches={summary['r_seen_mismatches']} "
            f"boundary_mismatches={summary['boundary_mismatches']} "
            f"nll={summary['nll']} brier={summary['brier']} auc={summary['auc']}",
            flush=True,
        )
        return summary

    def _dkt_parity(self, kind, ours, path, train, valid, row_cls) -> None:
        from pykt.models.dkt import DKT

        ours.load_checkpoint(path)
        legacy = DKT(num_c=2909, emb_size=200, dropout=0.1)
        if kind == "dkt_qc":
            legacy.concept_interaction_emb = torch.nn.Embedding(204, 200)
        state = torch.load(str(path), map_location="cpu", weights_only=False)
        legacy.load_state_dict(state, strict=True)
        legacy.eval()
        max_p = 0.0
        max_h = 0.0
        max_kc = 0.0
        n_rows = 0
        n_runs = 0
        y = []
        p_new = []
        p_old = []
        for learner in sorted({row.student_idx for row in valid}):
            train_rows = _dkt_rows(train, learner, kind, row_cls)
            valid_rows = _dkt_rows(valid, learner, kind, row_cls)
            trained = ours.replay(train_rows)
            replayed = ours.replay(valid_rows, initial_state=trained.final_state)
            _train_p, _train_h, h_n, c_n = _legacy_dkt(
                legacy,
                train_rows,
                kind,
                torch.zeros(1, 1, 200),
                torch.zeros(1, 1, 200),
            )
            old_p, old_h, h_n, c_n = _legacy_dkt(legacy, valid_rows, kind, h_n, c_n)
            our_p = np.asarray([row.probability for row in replayed.rows], dtype=np.float64)
            max_p = max(max_p, _max_abs(old_p, our_p))
            our_h = np.asarray([row.hidden_pre for row in replayed.rows], dtype=np.float64)
            max_h = max(max_h, _max_abs(old_h, our_h))
            if kind == "dkt_qc":
                max_kc = max(max_kc, _max_abs(_kc_checksum(legacy, valid_rows), [row.kc_contribution_checksum for row in replayed.rows]))
            h_gap = float(torch.max(torch.abs(replayed.final_state[learner][0] - h_n)).item())
            c_gap = float(torch.max(torch.abs(replayed.final_state[learner][1] - c_n)).item())
            max_h = max(max_h, h_gap, c_gap)
            n_rows += len(valid_rows)
            n_runs += replayed.n_runs
            y.extend(row.correct for row in valid_rows)
            p_new.append(our_p)
            p_old.append(np.asarray(old_p, dtype=np.float64))
        labels = np.asarray(y, dtype=np.float64)
        joined_new = np.concatenate(p_new)
        joined_old = np.concatenate(p_old)
        print(
            f"PARITY_ASSIST_{kind} learners={len(set(row.student_idx for row in valid))} "
            f"rows={n_rows} runs={n_runs} max_p={max_p} max_hidden={max_h} "
            f"max_kc={max_kc} d_nll={abs(metric_nll(labels, joined_new) - metric_nll(labels, joined_old))} "
            f"d_brier={abs(metric_brier(labels, joined_new) - metric_brier(labels, joined_old))} "
            f"d_auc={abs(metric_auc(labels, joined_new) - metric_auc(labels, joined_old))}",
            flush=True,
        )
        self.assertEqual(max_p, 0.0)
        self.assertEqual(max_h, 0.0)
        self.assertEqual(max_kc, 0.0)

    def _dkvmn_parity(self, kind, ours, path, train, valid, row_cls) -> None:
        from pykt.models.dkvmn import DKVMN
        from ednet_dkvmn import torch_predict_bundle
        from ednet_dkvmn_qc import torch_predict_bundle_qc

        ours.load_checkpoint(path)
        legacy = DKVMN(num_c=2909, dim_s=64, size_m=20, dropout=0.2)
        if kind == "dkvmn_qc":
            legacy.human_concept_emb = torch.nn.Embedding(102, 64)
        state = torch.load(str(path), map_location="cpu", weights_only=False)
        legacy.load_state_dict(state, strict=True)
        legacy.eval()
        if kind == "dkvmn_qc":
            self._assert_write_invariance(legacy, torch_predict_bundle_qc)
        max_p = 0.0
        max_memory = 0.0
        max_kc = 0.0
        n_rows = 0
        n_runs = 0
        y = []
        p_new = []
        p_old = []
        for learner in sorted({row.student_idx for row in valid}):
            train_rows = _dkt_rows(train, learner, kind, row_cls)
            valid_rows = _dkt_rows(valid, learner, kind, row_cls)
            trained = ours.replay(train_rows)
            replayed = ours.replay(valid_rows, initial_memory=trained.final_memory)
            memory = legacy.Mv0.detach().clone()
            memory = _legacy_dkvmn(legacy, train_rows, kind, memory, torch_predict_bundle, torch_predict_bundle_qc)
            old_p, memory = _legacy_dkvmn_probs(
                legacy, valid_rows, kind, memory, torch_predict_bundle, torch_predict_bundle_qc
            )
            our_p = np.asarray([row.probability for row in replayed.rows], dtype=np.float64)
            max_p = max(max_p, _max_abs(old_p, our_p))
            max_memory = max(
                max_memory,
                float(torch.max(torch.abs(replayed.final_memory[learner] - memory)).item()),
            )
            if kind == "dkvmn_qc":
                expected = [_concept_checksum(legacy, row.tag_idxs) for row in valid_rows]
                got = [row.concept_checksum for row in replayed.rows]
                max_kc = max(max_kc, _max_abs(expected, got))
            n_rows += len(valid_rows)
            n_runs += replayed.n_runs
            y.extend(row.correct for row in valid_rows)
            p_new.append(our_p)
            p_old.append(np.asarray(old_p, dtype=np.float64))
        labels = np.asarray(y, dtype=np.float64)
        joined_new = np.concatenate(p_new)
        joined_old = np.concatenate(p_old)
        print(
            f"PARITY_ASSIST_{kind} learners={len(set(row.student_idx for row in valid))} "
            f"rows={n_rows} runs={n_runs} max_p={max_p} max_memory={max_memory} "
            f"max_kc={max_kc} d_nll={abs(metric_nll(labels, joined_new) - metric_nll(labels, joined_old))} "
            f"d_brier={abs(metric_brier(labels, joined_new) - metric_brier(labels, joined_old))} "
            f"d_auc={abs(metric_auc(labels, joined_new) - metric_auc(labels, joined_old))}",
            flush=True,
        )
        self.assertEqual(max_p, 0.0)
        self.assertEqual(max_memory, 0.0)
        self.assertEqual(max_kc, 0.0)

    def _assert_write_invariance(self, model, predict_qc) -> None:
        problems = torch.tensor([0, 1], dtype=torch.long)
        corrects = torch.tensor([1, 0], dtype=torch.long)
        memory = model.Mv0.detach().clone()
        first, memory_a = predict_qc(
            torch_module=torch,
            model=model,
            Mv_pre=memory,
            bundle_problem=problems,
            bundle_correct=corrects,
            concept_lists_per_row=[[3], [4]],
            num_c=2909,
            num_concepts=102,
        )
        _second, memory_b = predict_qc(
            torch_module=torch,
            model=model,
            Mv_pre=memory,
            bundle_problem=problems,
            bundle_correct=corrects,
            concept_lists_per_row=[[8], [9]],
            num_c=2909,
            num_concepts=102,
        )
        gap = float(torch.max(torch.abs(memory_a - memory_b)).item())
        self.assertEqual(gap, 0.0)
        self.assertGreater(float(torch.max(torch.abs(first - _second)).item()), 0.0)


def _window_learners(rows) -> list[int]:
    counts: dict[int, int] = {}
    for row in rows:
        counts[row.group_id] = counts.get(row.group_id, 0) + 1
    chosen: list[int] = []
    seen: set[int] = set()
    for row in rows:
        if counts[row.group_id] < 2 or row.student_idx in seen:
            continue
        seen.add(row.student_idx)
        chosen.append(row.student_idx)
        if len(chosen) == 4:
            break
    if len(chosen) < 2:
        raise AssertionError("VALID window has fewer than 2 multi-row bundles")
    return chosen


def _rows_for(rows, learners: list[int]):
    keep = set(learners)
    return tuple(row for row in rows if row.student_idx in keep)


def _dkt_rows(rows, learner: int, kind: str, row_cls):
    use_skill = kind.endswith("qc")
    built = []
    for row in rows_for_neural_state(rows, kind):
        if row.student_idx != learner:
            continue
        built.append(
            row_cls(
                student_idx=row.student_idx,
                problem_idx=row.problem_idx,
                correct=row.correct,
                solving_id=row.group_id,
                tag_idxs=(row.skill_idx,) if use_skill else (),
                interaction_id=row.interaction_id,
            )
        )
    return built


def _bundle_starts(rows) -> list[int]:
    starts = [0]
    previous = rows[0].solving_id
    for index, row in enumerate(rows[1:], start=1):
        if row.solving_id != previous:
            starts.append(index)
            previous = row.solving_id
    return starts


def _legacy_dkt(model, rows, kind: str, h, c):
    if not rows:
        return np.zeros(0), np.zeros((0, 200)), h, c
    problems = torch.tensor([row.problem_idx for row in rows], dtype=torch.long)
    corrects = torch.tensor([row.correct for row in rows], dtype=torch.long)
    if kind == "dkt_qc":
        from ednet_dkt_qc import build_qc_replay_input

        inputs = build_qc_replay_input(
            torch,
            model,
            problems,
            corrects,
            [list(row.tag_idxs) for row in rows],
            2909,
            102,
        )
    else:
        inputs = model.interaction_emb(problems + 2909 * corrects)
    with torch.no_grad():
        outputs, (h_n, c_n) = model.lstm_layer(inputs.unsqueeze(0), (h, c))
    hidden = outputs.squeeze(0)
    starts = set(_bundle_starts(rows))
    incoming = h.squeeze(0).squeeze(0)
    probs = []
    vectors = []
    h_pre = incoming
    for index, row in enumerate(rows):
        if index in starts:
            h_pre = incoming if index == 0 else hidden[index - 1]
        logit = torch.nn.functional.linear(h_pre.unsqueeze(0), model.out_layer.weight, model.out_layer.bias).squeeze(0)
        probs.append(float(torch.sigmoid(logit[int(row.problem_idx)]).detach().cpu()))
        vectors.append([float(value) for value in h_pre.detach().float().cpu().tolist()])
    return np.asarray(probs), np.asarray(vectors), h_n.detach().cpu().clone(), c_n.detach().cpu().clone()


def _kc_checksum(model, rows) -> list[float]:
    """Legacy concept-mean checksum, taken before it is added to ``q_emb``."""

    if not rows:
        return []
    problems = torch.tensor([row.problem_idx for row in rows], dtype=torch.long)
    corrects = torch.tensor([row.correct for row in rows], dtype=torch.long)
    tags = [list(row.tag_idxs) for row in rows]
    max_k = max(len(item) for item in tags)
    padded = torch.zeros((len(rows), max_k), dtype=torch.long)
    mask = torch.zeros((len(rows), max_k), dtype=torch.bool)
    for index, item in enumerate(tags):
        if not item:
            continue
        values = torch.tensor(item, dtype=torch.long)
        padded[index, : values.numel()] = values
        mask[index, : values.numel()] = True
    concept_id = padded + 102 * corrects.unsqueeze(1)
    gathered = model.concept_interaction_emb(concept_id) * mask.unsqueeze(-1)
    count = mask.sum(dim=1, keepdim=True).to(gathered.dtype).clamp(min=1.0)
    concept = gathered.sum(dim=1) / count
    empty = ~mask.any(dim=1)
    concept = concept.clone()
    concept[empty] = 0
    return [float(value) for value in concept.detach().float().sum(dim=1).cpu().tolist()]


def _legacy_dkvmn(model, rows, kind, memory, predict_q, predict_qc):
    _probs, memory = _legacy_dkvmn_probs(model, rows, kind, memory, predict_q, predict_qc)
    return memory


def _legacy_dkvmn_probs(model, rows, kind, memory, predict_q, predict_qc):
    if not rows:
        return np.zeros(0), memory
    probs = []
    cursor = 0
    starts = _bundle_starts(rows)
    ends = starts[1:] + [len(rows)]
    for start, end in zip(starts, ends):
        chosen = rows[start:end]
        problems = torch.tensor([row.problem_idx for row in chosen], dtype=torch.long)
        corrects = torch.tensor([row.correct for row in chosen], dtype=torch.long)
        with torch.no_grad():
            if kind == "dkvmn_qc":
                probability, memory = predict_qc(
                    torch_module=torch,
                    model=model,
                    Mv_pre=memory,
                    bundle_problem=problems,
                    bundle_correct=corrects,
                    concept_lists_per_row=[list(row.tag_idxs) for row in chosen],
                    num_c=2909,
                    num_concepts=102,
                )
            else:
                probability, memory = predict_q(
                    torch_module=torch,
                    model=model,
                    Mv_pre=memory,
                    bundle_problem=problems,
                    bundle_correct=corrects,
                    num_c=2909,
                )
        probs.extend(float(value) for value in probability.detach().cpu().tolist())
        cursor = end
    return np.asarray(probs, dtype=np.float64), memory.detach().cpu().clone()


def _concept_checksum(model, tags) -> float:
    if not tags:
        return 0.0
    ids = torch.tensor(list(tags), dtype=torch.long)
    return float(model.human_concept_emb(ids).mean(dim=0).sum().detach().cpu())


if __name__ == "__main__":
    unittest.main()
