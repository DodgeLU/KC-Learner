"""Restore formal TRAIN-only checkpoints and replay a VALID window.

Does not open the TEST parquet or the post-VALID ``state.npz``.
See ``FIXTURE_MANIFEST.md``.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from kclearner.models import ARKTModel, IRTModel, StreamingRow
from kclearner.models.metrics import metric_auc, metric_brier, metric_nll
from test_legacy_ednet_parity import (
    ATOL,
    _artifacts_ready,
    _boundary_mismatches,
    _first_row,
    _guard_parquet_reader,
    _import_legacy,
    _legacy_frame,
    _max_abs,
    _project_c,
    _raw_has_sentinel,
    _split_dir,
)
from window_selection import find_window


def _checkpoint(kind: str) -> Path:
    family = "IRT" if kind == "irt" else "AR-KT"
    name = "ednet_irt_dev5000" if kind == "irt" else "ednet_ar_kt_dev5000"
    return (
        _project_c()
        / "final_valid_pt"
        / "EdNet"
        / family
        / name
        / "state_train_only.npz"
    )


def _checkpoints_ready() -> bool:
    return _artifacts_ready() and _checkpoint("irt").is_file() and _checkpoint("ar_kt").is_file()


def _item(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.item()
    return value


def _legacy_from_checkpoint(legacy: dict, path: Path, kind: str):
    with np.load(str(path), allow_pickle=False) as saved:
        cfg = legacy["EdNetARTKTConfig"](
            n_students=int(_item(saved["n_students"])),
            n_problems=int(_item(saved["n_problems"])),
            n_skills=int(_item(saved["n_skills"])),
            eta_r=float(_item(saved["eta_r"])),
            use_residual=bool(_item(saved["use_residual"])),
            use_time=False,
            tau_sec=float("inf"),
            theta_lr=float(_item(saved["theta_lr"])),
            b_lr=float(_item(saved["b_lr"])),
            theta_l2=float(_item(saved["theta_l2"])),
            b_l2=float(_item(saved["b_l2"])),
            learn_b=bool(_item(saved["learn_b"])),
            freeze_global_in_eval=bool(_item(saved["freeze_global_in_eval"])),
        )
        theta = np.array(saved["theta"], dtype=np.float64, copy=True)
        b = np.array(saved["b"], dtype=np.float64, copy=True)
        residual = None
        if kind == "ar_kt":
            residual = (
                np.array(saved["r_post"], dtype=np.float64, copy=True),
                np.array(saved["r_seen"], dtype=np.bool_, copy=True),
                np.array(saved["last_r_update_ts"], dtype=np.int64, copy=True),
            )
    model = legacy["EdNetARTKTModel"](cfg)
    model.theta = theta
    model.b = b
    if residual is not None:
        model.r_post, model.r_seen, model.last_r_update_ts = residual
    return model


def _rows_from_frame(frame) -> list[StreamingRow]:
    rows: list[StreamingRow] = []
    for offset, rec in enumerate(frame.itertuples(index=False)):
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
    return rows


def _metric_delta(fn_new, fn_old, y, new_p, old_p) -> float:
    return abs(float(fn_new(y, new_p)) - float(fn_old(y, old_p)))


def _replay(kind: str, rows: list[StreamingRow], legacy: dict, path: Path) -> dict:
    if kind == "irt":
        restored = IRTModel.load_legacy_checkpoint(path)
    else:
        restored = ARKTModel.load_legacy_checkpoint(path)
    oracle = _legacy_from_checkpoint(legacy, path, kind)
    frame = _legacy_frame(rows)
    new_out = restored.run_streaming(rows, phase="eval")
    old_out = oracle.run_streaming(frame, phase="eval")
    y = np.asarray([row.correct for row in rows], dtype=np.float64)
    p_global = np.asarray([row.p_global for row in new_out.rows], dtype=np.float64)
    p_art = np.asarray([row.p_art for row in new_out.rows], dtype=np.float64)
    r_bar = np.asarray([row.r_bar for row in new_out.rows], dtype=np.float64)
    summary = {
        "rows": len(rows),
        "runs": new_out.n_runs,
        "max_p_global": _max_abs(old_out["p_global"], p_global),
        "max_p_art": _max_abs(old_out["p_art"], p_art),
        "max_r_bar": _max_abs(old_out["r_bar"], r_bar),
        "max_theta": _max_abs(oracle.theta, restored.theta),
        "max_b": _max_abs(oracle.b, restored.b),
        "max_r": _max_abs(oracle.r_post, restored.r_post),
        "r_seen_mismatches": int(np.sum(oracle.r_seen != restored.r_seen)),
        "ts_mismatches": int(
            np.sum(oracle.last_r_update_ts != restored.last_r_update_ts)
        ),
        "boundary_mismatches": _boundary_mismatches(new_out, rows),
        "d_nll_global": _metric_delta(
            metric_nll, legacy["legacy_nll"], y, p_global, old_out["p_global"]
        ),
        "d_brier_global": _metric_delta(
            metric_brier, legacy["legacy_brier"], y, p_global, old_out["p_global"]
        ),
        "d_auc_global": _metric_delta(
            metric_auc, legacy["legacy_auc"], y, p_global, old_out["p_global"]
        ),
        "d_nll_art": _metric_delta(
            metric_nll, legacy["legacy_nll"], y, p_art, old_out["p_art"]
        ),
        "d_brier_art": _metric_delta(
            metric_brier, legacy["legacy_brier"], y, p_art, old_out["p_art"]
        ),
        "d_auc_art": _metric_delta(
            metric_auc, legacy["legacy_auc"], y, p_art, old_out["p_art"]
        ),
    }
    problems: list[str] = []
    if summary["boundary_mismatches"] != 0:
        problems.append(f"run-boundary mismatches={summary['boundary_mismatches']}")
    for label, legacy_values, new_values in (
        ("p_global", old_out["p_global"], p_global),
        ("p_art", old_out["p_art"], p_art),
        ("r_bar", old_out["r_bar"], r_bar),
    ):
        where = _first_row(legacy_values, new_values, new_out)
        if where is not None:
            problems.append(f"{label} {where}")
            break
    if not problems and summary["max_theta"] > ATOL:
        problems.append(f"theta max abs {summary['max_theta']}")
    if not problems and summary["max_b"] > ATOL:
        problems.append(f"b max abs {summary['max_b']}")
    if not problems and summary["max_r"] > ATOL:
        problems.append(f"r_post max abs {summary['max_r']}")
    if summary["r_seen_mismatches"] or summary["ts_mismatches"]:
        problems.append(
            f"r_seen mismatches={summary['r_seen_mismatches']} "
            f"last_r_update_ts mismatches={summary['ts_mismatches']}"
        )
    for name in (
        "d_nll_global",
        "d_brier_global",
        "d_auc_global",
        "d_nll_art",
        "d_brier_art",
        "d_auc_art",
    ):
        if summary[name] > ATOL:
            problems.append(f"{name}={summary[name]}")
    if problems:
        raise AssertionError(
            f"{kind} checkpoint parity failed; first divergence: " + "; ".join(problems)
        )
    return summary


def _format(kind: str, summary: dict, label: str) -> str:
    return (
        f"PARITY_{kind}_CKPT label={label} rows={summary['rows']} "
        f"runs={summary['runs']} max_p_global={summary['max_p_global']} "
        f"max_p_art={summary['max_p_art']} max_r_bar={summary['max_r_bar']} "
        f"max_theta={summary['max_theta']} max_b={summary['max_b']} "
        f"max_r={summary['max_r']} r_seen_mismatches={summary['r_seen_mismatches']} "
        f"ts_mismatches={summary['ts_mismatches']} "
        f"boundary_mismatches={summary['boundary_mismatches']} "
        f"d_nll_global={summary['d_nll_global']} "
        f"d_brier_global={summary['d_brier_global']} "
        f"d_auc_global={summary['d_auc_global']} "
        f"d_nll_art={summary['d_nll_art']} "
        f"d_brier_art={summary['d_brier_art']} "
        f"d_auc_art={summary['d_auc_art']}"
    )


@unittest.skipUnless(
    _checkpoints_ready(),
    "formal EdNet state_train_only checkpoints or dev5000 parquet are absent",
)
class LegacyCheckpointParityTests(unittest.TestCase):
    def test_valid_window_from_train_only_checkpoints(self) -> None:
        legacy = _import_legacy()
        pandas, original = _guard_parquet_reader()
        try:
            split = _split_dir()
            train_raw = pandas.read_parquet(split / "ednet_split_v1_train.parquet")
            valid_raw = pandas.read_parquet(split / "ednet_split_v1_valid.parquet")
            vocab = legacy["build_vocab"](train_raw, valid_raw)
            frozen = legacy["FROZEN_5K_UNIVERSE"]
            train_canon = legacy["normalize_canonical"](train_raw, vocab)
            valid_canon = legacy["normalize_canonical"](valid_raw, vocab)
            if len(valid_canon) != len(valid_raw):
                raise AssertionError("normalize_canonical changed the VALID row count")
            valid_canon = valid_canon.copy()
            valid_canon["_src_pos"] = np.arange(len(valid_raw), dtype=np.int64)
            _train_sub, valid_sub, n_users = legacy["select_dev5000"](
                train_canon, valid_canon, int(frozen["n_train_users"])
            )
            del train_raw, train_canon, valid_canon, _train_sub
            if n_users != 5000 or len(valid_sub) != int(frozen["n_valid_rows"]):
                raise AssertionError(
                    f"dev5000 VALID identity n_users={n_users} rows={len(valid_sub)}"
                )
            src_pos = valid_sub["_src_pos"].to_numpy(dtype=np.int64)
            raw_tags = valid_raw["tag_ids"].to_numpy()
            sentinel = [_raw_has_sentinel(raw_tags[int(pos)]) for pos in src_pos]
            tags = [
                () if value is None else tuple(int(token) for token in value)
                for value in valid_sub["tag_idxs"].tolist()
            ]
            start, end = find_window(
                valid_sub["student_idx"].to_numpy(dtype=np.int64),
                valid_sub["group_id"].to_numpy(dtype=np.int64),
                valid_sub["problem_idx"].to_numpy(dtype=np.int64),
                valid_sub["correct"].to_numpy(dtype=np.int64),
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
            window = valid_sub.iloc[start:end].loc[:, needed]
            rows = _rows_from_frame(window)
            if any(sentinel[index] for index in range(start, end)):
                raise AssertionError("selected VALID window contains a raw -1 tag")
            label = f"valid[{start}:{end}]"
            irt_summary = _replay("irt", rows, legacy, _checkpoint("irt"))
            print(_format("IRT", irt_summary, label), flush=True)
            ar_summary = _replay("ar_kt", rows, legacy, _checkpoint("ar_kt"))
            print(_format("AR-KT", ar_summary, label), flush=True)
        finally:
            pandas.read_parquet = original
