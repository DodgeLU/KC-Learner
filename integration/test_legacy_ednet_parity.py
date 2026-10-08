"""Local legacy parity for IRT and AR-KT.

Skipped when ``project_c`` or the formal EdNet dev5000 train/valid
parquet is absent. See ``FIXTURE_MANIFEST.md``. This module does not
open the TEST parquet and does not modify ``project_c``.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

from kclearner.models import ARKTModel, IRTModel, StreamingRow
from kclearner.models.metrics import metric_auc, metric_brier, metric_nll

ATOL = 1e-12


def _project_c() -> Path:
    override = os.environ.get("KCLEARNER_PROJECT_C")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "project_c"


def _split_dir() -> Path:
    return (
        _project_c()
        / "data_proc"
        / "ednet_kt1"
        / "ednet_kt1_pilot_v1_from_v2_split_v1"
    )


def _artifacts_ready() -> bool:
    root = _project_c()
    split = _split_dir()
    needed = [
        root / "models" / "ednet_art_kt.py",
        root / "train" / "_ednet_irt_ar_kt_dev5000_common.py",
        split / "ednet_split_v1_train.parquet",
        split / "ednet_split_v1_valid.parquet",
    ]
    return all(path.is_file() for path in needed)


def _import_legacy():
    root = _project_c()
    for sub in (
        root / "models",
        root / "art_kt_viability" / "models",
        root / "train",
    ):
        text = str(sub)
        if text not in sys.path:
            sys.path.insert(0, text)
    from _ednet_irt_ar_kt_dev5000_common import (  # type: ignore
        FROZEN_5K_UNIVERSE,
        build_vocab,
        metric_auc as legacy_auc,
        metric_brier as legacy_brier,
        metric_nll as legacy_nll,
        normalize_canonical,
        select_dev5000,
    )
    from ednet_art_kt import EdNetARTKTConfig, EdNetARTKTModel  # type: ignore

    return {
        "FROZEN_5K_UNIVERSE": FROZEN_5K_UNIVERSE,
        "build_vocab": build_vocab,
        "legacy_auc": legacy_auc,
        "legacy_brier": legacy_brier,
        "legacy_nll": legacy_nll,
        "normalize_canonical": normalize_canonical,
        "select_dev5000": select_dev5000,
        "EdNetARTKTConfig": EdNetARTKTConfig,
        "EdNetARTKTModel": EdNetARTKTModel,
    }


def _guard_parquet_reader():
    import pandas as pd

    original = pd.read_parquet

    def guarded(path, *args, **kwargs):
        name = Path(str(path)).name.lower()
        if "test" in name and "train" not in name and "valid" not in name:
            raise AssertionError(f"TEST parquet must not be read: {path}")
        return original(path, *args, **kwargs)

    pd.read_parquet = guarded
    return pd, original


def _legacy_frame(rows: list[StreamingRow]):
    import pandas as pd

    return pd.DataFrame(
        {
            "student_idx": np.asarray([row.student_idx for row in rows], dtype=np.int64),
            "problem_idx": np.asarray([row.problem_idx for row in rows], dtype=np.int64),
            "group_id": np.asarray([row.group_id for row in rows], dtype=np.int64),
            "correct": np.asarray([row.correct for row in rows], dtype=np.int64),
            "timestamp_ms": np.asarray(
                [row.timestamp_ms for row in rows], dtype=np.int64
            ),
            "group_last_timestamp_ms": np.asarray(
                [row.group_last_timestamp_ms for row in rows], dtype=np.int64
            ),
            "tag_idxs": [list(row.tag_idxs) for row in rows],
        }
    )


def _key_spans(rows: list[StreamingRow]) -> tuple[tuple[int, int, int, int], ...]:
    spans: list[tuple[int, int, int, int]] = []
    index = 0
    while index < len(rows):
        end = index + 1
        while (
            end < len(rows)
            and rows[end].student_idx == rows[index].student_idx
            and rows[end].group_id == rows[index].group_id
        ):
            end += 1
        spans.append(
            (index, end - 1, rows[index].student_idx, rows[index].group_id)
        )
        index = end
    return tuple(spans)


def _boundary_mismatches(result, rows: list[StreamingRow]) -> int:
    observed = tuple(
        (start, end, student, group)
        for _run_id, start, end, student, group in result.run_spans()
    )
    expected = _key_spans(rows)
    shared = min(len(observed), len(expected))
    mismatched = sum(
        1 for index in range(shared) if observed[index] != expected[index]
    )
    return mismatched + abs(len(observed) - len(expected))


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        raise AssertionError(f"shape {left.shape} != {right.shape}")
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left - right)))


def _first_row(legacy: np.ndarray, new: np.ndarray, result) -> str | None:
    diff = np.abs(np.asarray(legacy, dtype=np.float64) - np.asarray(new, dtype=np.float64))
    hits = np.flatnonzero(diff > ATOL)
    if hits.size == 0:
        return None
    index = int(hits[0])
    row = result.rows[index]
    return (
        f"row {index} run {row.run_id} "
        f"[{row.run_start}, {row.run_end}] "
        f"student {row.student_idx} group {row.group_id} "
        f"legacy={float(legacy[index])!r} new={float(new[index])!r} "
        f"abs={float(diff[index])!r}"
    )


def _new_model(kind: str, n_students: int, n_problems: int, n_skills: int):
    if kind == "irt":
        return IRTModel(
            n_students,
            n_problems,
            n_skills,
            theta_lr=0.05,
            b_lr=0.02,
            theta_l2=1e-4,
            b_l2=1e-4,
            learn_b=True,
            freeze_global_in_eval=True,
        )
    return ARKTModel(
        n_students,
        n_problems,
        n_skills,
        eta_r=0.20,
        theta_lr=0.05,
        b_lr=0.02,
        theta_l2=1e-4,
        b_l2=1e-4,
        learn_b=True,
        freeze_global_in_eval=True,
    )


def _legacy_model(legacy, kind: str, n_students: int, n_problems: int, n_skills: int):
    cfg = legacy["EdNetARTKTConfig"](
        n_students=int(n_students),
        n_problems=int(n_problems),
        n_skills=int(n_skills),
        eta_r=0.0 if kind == "irt" else 0.20,
        use_residual=kind != "irt",
        use_time=False,
        tau_sec=float("inf"),
        theta_lr=0.05,
        b_lr=0.02,
        theta_l2=1e-4,
        b_l2=1e-4,
        learn_b=True,
        freeze_global_in_eval=True,
    )
    return legacy["EdNetARTKTModel"](cfg)


def _compare(kind: str, rows: list[StreamingRow], new_model, legacy_model, legacy) -> dict:
    frame = _legacy_frame(rows)
    y = np.asarray([row.correct for row in rows], dtype=np.float64)
    summary = {
        "rows": len(rows),
        "runs": 0,
        "max_p_global": 0.0,
        "max_p_art": 0.0,
        "max_r_bar": 0.0,
        "max_theta": 0.0,
        "max_b": 0.0,
        "max_r": 0.0,
        "boundary_mismatches": 0,
        "d_nll": 0.0,
        "d_brier": 0.0,
        "d_auc": 0.0,
    }
    for phase in ("train", "eval"):
        new_out = new_model.run_streaming(rows, phase=phase)
        old_out = legacy_model.run_streaming(frame, phase=phase)
        summary["runs"] = new_out.n_runs
        summary["boundary_mismatches"] = max(
            summary["boundary_mismatches"],
            _boundary_mismatches(new_out, rows),
        )
        p_global = np.asarray([row.p_global for row in new_out.rows], dtype=np.float64)
        p_art = np.asarray([row.p_art for row in new_out.rows], dtype=np.float64)
        r_bar = np.asarray([row.r_bar for row in new_out.rows], dtype=np.float64)
        summary["max_p_global"] = max(
            summary["max_p_global"], _max_abs(old_out["p_global"], p_global)
        )
        summary["max_p_art"] = max(
            summary["max_p_art"], _max_abs(old_out["p_art"], p_art)
        )
        summary["max_r_bar"] = max(
            summary["max_r_bar"], _max_abs(old_out["r_bar"], r_bar)
        )
        summary["max_theta"] = max(
            summary["max_theta"], _max_abs(legacy_model.theta, new_model.theta)
        )
        summary["max_b"] = max(summary["max_b"], _max_abs(legacy_model.b, new_model.b))
        summary["max_r"] = max(
            summary["max_r"], _max_abs(legacy_model.r_post, new_model.r_post)
        )
        if not np.array_equal(legacy_model.r_seen, new_model.r_seen):
            raise AssertionError(f"{kind} {phase} r_seen mismatch")
        if not np.array_equal(
            legacy_model.last_r_update_ts, new_model.last_r_update_ts
        ):
            raise AssertionError(f"{kind} {phase} last_r_update_ts mismatch")
        headline_old = old_out["p_global"] if kind == "irt" else old_out["p_art"]
        headline_new = p_global if kind == "irt" else p_art
        for name, fn_new, fn_old, values in (
            ("nll", metric_nll, legacy["legacy_nll"], headline_new),
            ("brier", metric_brier, legacy["legacy_brier"], headline_new),
            ("auc", metric_auc, legacy["legacy_auc"], headline_new),
        ):
            delta = abs(float(fn_new(y, values)) - float(fn_old(y, headline_old)))
            summary[f"d_{name}"] = max(summary[f"d_{name}"], delta)
            port = abs(float(fn_new(y, headline_old)) - float(fn_old(y, headline_old)))
            if port > ATOL:
                raise AssertionError(f"{kind} {phase} metric port {name} delta {port}")
        problems = []
        if summary["boundary_mismatches"] != 0:
            problems.append(
                f"run-boundary mismatches={summary['boundary_mismatches']}"
            )
        for label, legacy_values, new_values in (
            ("p_global", old_out["p_global"], p_global),
            ("p_art", old_out["p_art"], p_art),
            ("r_bar", old_out["r_bar"], r_bar),
        ):
            where = _first_row(legacy_values, new_values, new_out)
            if where is not None:
                problems.append(f"{phase} {label} {where}")
        if _max_abs(legacy_model.theta, new_model.theta) > ATOL:
            problems.append(f"{phase} theta max abs {summary['max_theta']}")
        if _max_abs(legacy_model.b, new_model.b) > ATOL:
            problems.append(f"{phase} b max abs {summary['max_b']}")
        if _max_abs(legacy_model.r_post, new_model.r_post) > ATOL:
            problems.append(f"{phase} r_post max abs {summary['max_r']}")
        for name in ("d_nll", "d_brier", "d_auc"):
            if summary[name] > ATOL:
                problems.append(f"{phase} {name}={summary[name]}")
        if problems:
            raise AssertionError(
                f"{kind} parity failed; first divergences: " + "; ".join(problems)
            )
    return summary


def _format_summary(kind: str, summary: dict, label: str) -> str:
    return (
        f"PARITY_{kind} label={label} rows={summary['rows']} runs={summary['runs']} "
        f"max_p_global={summary['max_p_global']} max_p_art={summary['max_p_art']} "
        f"max_r_bar={summary['max_r_bar']} max_theta={summary['max_theta']} "
        f"max_b={summary['max_b']} max_r={summary['max_r']} "
        f"boundary_mismatches={summary['boundary_mismatches']} "
        f"d_nll={summary['d_nll']} d_brier={summary['d_brier']} d_auc={summary['d_auc']}"
    )


def _irt_rows() -> list[StreamingRow]:
    def row(student, problem, group, correct, interaction_id):
        return StreamingRow(
            student_idx=student,
            problem_idx=problem,
            group_id=group,
            correct=correct,
            interaction_id=interaction_id,
            timestamp_ms=10,
            group_last_timestamp_ms=10,
        )

    return [
        row(0, 0, 1, 1, "r0"),
        row(0, 1, 1, 0, "r1"),
        row(1, 0, 1, 1, "r2"),
        row(0, 0, 3, 0, "r3"),
        row(0, 1, 1, 1, "r4"),
        row(0, 1, 1, 0, "r5"),
    ]


def _ar_rows() -> list[StreamingRow]:
    def row(student, problem, group, correct, tags, interaction_id, stamp):
        return StreamingRow(
            student_idx=student,
            problem_idx=problem,
            group_id=group,
            correct=correct,
            tag_idxs=tags,
            interaction_id=interaction_id,
            timestamp_ms=stamp,
            group_last_timestamp_ms=stamp,
        )

    return [
        row(0, 0, 1, 1, (0, 1, 0), "a", 5),
        row(0, 1, 1, 1, (2,), "b", 9),
        row(0, 0, 2, 1, (0, 1, 0), "c", 12),
        row(1, 2, 1, 0, (2,), "d", 4),
        row(0, 1, 1, 0, (1,), "e", 15),
        row(0, 3, 1, 1, (0,), "f", 16),
    ]


def _token_is_sentinel(token: object) -> bool:
    if isinstance(token, str):
        text = token.strip()
        if text == "":
            return False
        return int(text) == -1
    return int(token) == -1


def _raw_has_sentinel(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return False
        return int(value) == -1
    if isinstance(value, (int, np.integer)):
        return int(value) == -1
    if isinstance(value, str):
        return _token_is_sentinel(value)
    for token in value:
        if _token_is_sentinel(token):
            return True
    return False


@unittest.skipUnless(
    _artifacts_ready(),
    "local project_c EdNet dev5000 train/valid artifacts are not present",
)
class LegacyEdNetParityTests(unittest.TestCase):
    synthetic_ok = False

    def test_01_synthetic_matches_legacy(self) -> None:
        legacy = _import_legacy()
        irt_rows = _irt_rows()
        irt_summary = _compare(
            "irt",
            irt_rows,
            _new_model("irt", 2, 3, 1),
            _legacy_model(legacy, "irt", 2, 3, 1),
            legacy,
        )
        print(_format_summary("IRT", irt_summary, "synthetic"), flush=True)
        ar_rows = _ar_rows()
        ar_summary = _compare(
            "ar_kt",
            ar_rows,
            _new_model("ar_kt", 2, 4, 3),
            _legacy_model(legacy, "ar_kt", 2, 4, 3),
            legacy,
        )
        print(_format_summary("AR-KT", ar_summary, "synthetic"), flush=True)
        type(self).synthetic_ok = True

    def test_02_formal_train_window(self) -> None:
        if not type(self).synthetic_ok:
            self.test_01_synthetic_matches_legacy()
        from window_selection import find_window

        legacy = _import_legacy()
        pandas, original = _guard_parquet_reader()
        try:
            split = _split_dir()
            train_raw = pandas.read_parquet(split / "ednet_split_v1_train.parquet")
            valid_raw = pandas.read_parquet(split / "ednet_split_v1_valid.parquet")
            vocab = legacy["build_vocab"](train_raw, valid_raw)
            n_students = int(len(vocab.sid_to_idx))
            n_problems = int(len(vocab.pid_to_idx))
            n_skills = int(len(vocab.kid_to_idx))
            frozen = legacy["FROZEN_5K_UNIVERSE"]
            if n_skills != int(frozen["n_concepts"]):
                raise AssertionError(
                    f"concept vocab {n_skills} != {frozen['n_concepts']}"
                )
            if (n_students, n_problems, n_skills) != (50000, 12259, 189):
                raise AssertionError(
                    "formal union vocab "
                    f"{(n_students, n_problems, n_skills)} != (50000, 12259, 189)"
                )
            train_canon = legacy["normalize_canonical"](train_raw, vocab)
            if len(train_canon) != len(train_raw):
                raise AssertionError("normalize_canonical changed the TRAIN row count")
            train_canon = train_canon.copy()
            train_canon["_src_pos"] = np.arange(len(train_raw), dtype=np.int64)
            valid_canon = legacy["normalize_canonical"](valid_raw, vocab)
            train_sub, _valid_sub, n_users = legacy["select_dev5000"](
                train_canon, valid_canon, int(frozen["n_train_users"])
            )
            del valid_raw, valid_canon, _valid_sub
            if n_users != 5000 or len(train_sub) != int(frozen["n_train_rows"]):
                raise AssertionError(
                    f"dev5000 TRAIN identity n_users={n_users} rows={len(train_sub)}"
                )
            src_pos = train_sub["_src_pos"].to_numpy(dtype=np.int64)
            raw_tags = train_raw["tag_ids"].to_numpy()
            dense_tags = train_sub["tag_idxs"].tolist()
            sentinel = [
                _raw_has_sentinel(raw_tags[int(pos)]) for pos in src_pos
            ]
            tags = [
                () if value is None else tuple(int(token) for token in value)
                for value in dense_tags
            ]
            start, end = find_window(
                train_sub["student_idx"].to_numpy(dtype=np.int64),
                train_sub["group_id"].to_numpy(dtype=np.int64),
                train_sub["problem_idx"].to_numpy(dtype=np.int64),
                train_sub["correct"].to_numpy(dtype=np.int64),
                tags,
                sentinel,
            )
            needed = [
                "student_idx",
                "problem_idx",
                "group_id",
                "correct",
                "tag_idxs",
                "user_id",
                "solving_id",
                "timestamp_ms",
                "group_last_timestamp_ms",
            ]
            window = train_sub.iloc[start:end].loc[:, needed]
            rows: list[StreamingRow] = []
            for offset, rec in enumerate(window.itertuples(index=False)):
                tag_value = rec.tag_idxs
                rows.append(
                    StreamingRow(
                        student_idx=int(rec.student_idx),
                        problem_idx=int(rec.problem_idx),
                        group_id=int(rec.group_id),
                        correct=int(rec.correct),
                        tag_idxs=(
                            ()
                            if tag_value is None
                            else tuple(int(token) for token in tag_value)
                        ),
                        interaction_id=f"{rec.user_id}|{int(rec.solving_id)}|{offset}",
                        timestamp_ms=int(rec.timestamp_ms),
                        group_last_timestamp_ms=int(rec.group_last_timestamp_ms),
                    )
                )
            if any(sentinel[index] for index in range(start, end)):
                raise AssertionError("selected window contains a raw -1 tag")
            label = f"train[{start}:{end}]"
            irt_summary = _compare(
                "irt",
                rows,
                _new_model("irt", n_students, n_problems, n_skills),
                _legacy_model(legacy, "irt", n_students, n_problems, n_skills),
                legacy,
            )
            print(_format_summary("IRT", irt_summary, label), flush=True)
            ar_summary = _compare(
                "ar_kt",
                rows,
                _new_model("ar_kt", n_students, n_problems, n_skills),
                _legacy_model(legacy, "ar_kt", n_students, n_problems, n_skills),
                legacy,
            )
            print(_format_summary("AR-KT", ar_summary, label), flush=True)
        finally:
            pandas.read_parquet = original
