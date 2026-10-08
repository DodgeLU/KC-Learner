"""Physical-order streaming port of EdNet ``EdNetARTKTModel``.

This is the legacy regression walker. It is not a generic sequence
engine, and it does not call ``replay_bundle_pre_state``.

Preserved from ``project_c/models/ednet_art_kt.py``:

* scan rows in the order given;
* flush when ``(student_idx, group_id)`` changes;
* predict every row in the run from the pre-run state;
* then update;
* ``z_global = theta - b``, ``p_global = sigmoid(z_global)``;
* item update is the legacy vectorized write;
* theta update is the mean gradient of the run;
* optional residual uses ``r_bar = mean(r_post[u, c] for c in tags)``
  with duplicate KC ids kept in the mean and in ``k_c``.

``sigmoid`` is the numerically stable helper from ``project_c/models/irt.py``.
Time decay is not implemented. EdNet runtime forces ``use_time=False``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Frozen EdNet dev5000 hyperparameters. IRT uses eta_r = 0.
# AR-KT formal aligned value is 0.20, not the older 0.50 protocol.
THETA_LR = 0.05
B_LR = 0.02
THETA_L2 = 1e-4
B_L2 = 1e-4
FORMAL_ETA_R = 0.20


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically safe sigmoid from ``models/irt.py``."""

    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _as_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be int, got {type(value).__name__}")
    return int(value)


@dataclass(frozen=True)
class StreamingRow:
    """One physical row in legacy index space.

    ``tag_idxs`` keeps source multiplicity. An empty tuple is an item
    with no KC ids. This record does not reinterpret EdNet ``tags="-1"``.
    """

    student_idx: int
    problem_idx: int
    group_id: int
    correct: int
    tag_idxs: tuple[int, ...] = ()
    interaction_id: str = ""
    timestamp_ms: int = 0
    group_last_timestamp_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "student_idx", _as_int("student_idx", self.student_idx))
        object.__setattr__(self, "problem_idx", _as_int("problem_idx", self.problem_idx))
        object.__setattr__(self, "group_id", _as_int("group_id", self.group_id))
        object.__setattr__(self, "correct", _as_int("correct", self.correct))
        object.__setattr__(
            self, "timestamp_ms", _as_int("timestamp_ms", self.timestamp_ms)
        )
        object.__setattr__(
            self,
            "group_last_timestamp_ms",
            _as_int("group_last_timestamp_ms", self.group_last_timestamp_ms),
        )
        if not isinstance(self.interaction_id, str):
            raise TypeError(
                "interaction_id must be str, "
                f"got {type(self.interaction_id).__name__}"
            )
        if not isinstance(self.tag_idxs, tuple):
            raise TypeError(
                "tag_idxs must be a tuple of int, "
                f"got {type(self.tag_idxs).__name__}"
            )
        object.__setattr__(
            self,
            "tag_idxs",
            tuple(_as_int("tag_idxs entry", tag) for tag in self.tag_idxs),
        )


@dataclass(frozen=True)
class StreamingRowTrace:
    """Regression trace for one physical row.

    ``run_start`` and ``run_end`` are inclusive indexes into the input.
    ``probability`` is ``p_global`` for IRT and ``p_art`` for AR-KT.
    ``k_c`` is the run-level token count in first-seen order, including
    duplicate KC ids. ``r_post_pre`` / ``r_post_post`` follow ``kc_ids``.
    """

    row_index: int
    interaction_id: str
    student_idx: int
    problem_idx: int
    group_id: int
    run_id: int
    run_start: int
    run_end: int
    theta_pre: float
    b_pre: float
    probability: float
    correct: int
    theta_post: float
    b_post: float
    p_global: float
    p_art: float
    kc_ids: tuple[int, ...]
    r_post_pre: tuple[float, ...]
    r_bar: float
    e_group: float
    k_c: tuple[tuple[int, int], ...]
    r_post_post: tuple[float, ...]


@dataclass(frozen=True)
class StreamingResult:
    rows: tuple[StreamingRowTrace, ...]

    @property
    def n_runs(self) -> int:
        if not self.rows:
            return 0
        return self.rows[-1].run_id + 1

    def run_spans(self) -> tuple[tuple[int, int, int, int, int], ...]:
        """Return ``(run_id, start, end, student_idx, group_id)`` spans.

        ``start`` and ``end`` are inclusive.
        """

        spans: list[tuple[int, int, int, int, int]] = []
        seen: set[int] = set()
        for row in self.rows:
            if row.run_id in seen:
                continue
            seen.add(row.run_id)
            spans.append(
                (
                    row.run_id,
                    row.run_start,
                    row.run_end,
                    row.student_idx,
                    row.group_id,
                )
            )
        return tuple(spans)


def _row_mean_r_pre(
    r_post: np.ndarray,
    student_idx: np.ndarray,
    tag_lists: list[list[int]],
) -> np.ndarray:
    """``r_bar = mean(r_post[u, c] for c in tag_idxs)``, duplicates kept."""

    out = np.zeros(len(student_idx), dtype=np.float64)
    for i in range(len(student_idx)):
        cs = tag_lists[i]
        if not cs:
            out[i] = 0.0
            continue
        vals = r_post[int(student_idx[i]), cs]
        out[i] = float(np.mean(vals))
    return out


def _kc_values(r_post: np.ndarray, student: int, tags: list[int]) -> tuple[float, ...]:
    if not tags:
        return ()
    vals = r_post[student, tags]
    return tuple(float(v) for v in np.asarray(vals, dtype=np.float64))


class LegacyEdNetStreamer:
    """Mutable theta / b / residual state and one streaming pass."""

    def __init__(
        self,
        n_students: int,
        n_problems: int,
        n_skills: int,
        *,
        use_residual: bool,
        eta_r: float,
        headline_art: bool,
        theta_lr: float = THETA_LR,
        b_lr: float = B_LR,
        theta_l2: float = THETA_L2,
        b_l2: float = B_L2,
        learn_b: bool = True,
        freeze_global_in_eval: bool = True,
        residual_update: str = "ednet_token",
    ) -> None:
        self.n_students = _as_int("n_students", n_students)
        self.n_problems = _as_int("n_problems", n_problems)
        self.n_skills = _as_int("n_skills", n_skills)
        self.use_residual = bool(use_residual)
        self.eta_r = float(eta_r)
        self.headline_art = bool(headline_art)
        self.theta_lr = float(theta_lr)
        self.b_lr = float(b_lr)
        self.theta_l2 = float(theta_l2)
        self.b_l2 = float(b_l2)
        self.learn_b = bool(learn_b)
        self.freeze_global_in_eval = bool(freeze_global_in_eval)
        if residual_update not in ("ednet_token", "assistments_skill_mean"):
            raise ValueError(
                "residual_update must be 'ednet_token' or "
                f"'assistments_skill_mean', got {residual_update!r}"
            )
        self.residual_update = residual_update
        self.theta = np.zeros(self.n_students, dtype=np.float64)
        self.b = np.zeros(self.n_problems, dtype=np.float64)
        self.r_post = np.zeros((self.n_students, self.n_skills), dtype=np.float64)
        self.r_seen = np.zeros((self.n_students, self.n_skills), dtype=bool)
        self.last_r_update_ts = np.full(
            (self.n_students, self.n_skills), -1, dtype=np.int64
        )

    def run_streaming(
        self,
        rows: tuple[StreamingRow, ...] | list[StreamingRow],
        phase: str = "train",
    ) -> StreamingResult:
        if phase not in ("train", "eval"):
            raise ValueError(f"phase must be 'train' or 'eval', got {phase!r}")
        eval_mode = (phase == "eval") and self.freeze_global_in_eval

        n = len(rows)
        traces: list[StreamingRowTrace] = []
        cur_sid = -1
        cur_gid = -1
        run_id = 0
        buf_idx: list[int] = []
        buf_s: list[int] = []
        buf_p: list[int] = []
        buf_y: list[int] = []
        buf_glt: list[int] = []
        buf_tags: list[list[int]] = []
        buf_ids: list[str] = []
        buf_g: list[int] = []

        def flush() -> None:
            nonlocal run_id
            if not buf_idx:
                return
            s = np.asarray(buf_s, dtype=np.int64)
            p = np.asarray(buf_p, dtype=np.int64)
            y = np.asarray(buf_y, dtype=np.int64)
            glt = np.asarray(buf_glt, dtype=np.int64)
            if not bool((s == s[0]).all()):
                raise ValueError("EdNet group must be single-student")

            current_ts = int(glt.max())
            theta_pre = float(self.theta[int(s[0])])
            b_pre = self.b[p].copy()
            r_pre = [
                _kc_values(self.r_post, int(s[j]), buf_tags[j])
                for j in range(len(buf_idx))
            ]
            r_bar = _row_mean_r_pre(self.r_post, s, buf_tags)
            z_g = self.theta[s] - self.b[p]
            p_g = sigmoid(z_g)
            z_a = z_g + r_bar
            p_a = sigmoid(z_a)

            if self.learn_b and not eval_mode:
                residual_g = y.astype(np.float64) - p_g
                b_grad = -residual_g - self.b_l2 * self.b[p]
                self.b[p] = self.b[p] + self.b_lr * b_grad

            residual_g = y.astype(np.float64) - p_g
            theta_grad = residual_g - self.theta_l2 * self.theta[s]
            avg_grad = float(theta_grad.mean())
            self.theta[int(s[0])] = (
                self.theta[int(s[0])] + self.theta_lr * avg_grad
            )

            e_group = float((y.astype(np.float64) - p_g).mean())
            k_c_counter: dict[int, int] = {}
            for cs in buf_tags:
                for c in cs:
                    k_c_counter[c] = k_c_counter.get(c, 0) + 1
            if self.use_residual and self.eta_r > 0.0:
                eta = float(self.eta_r)
                uu = int(s[0])
                if self.residual_update == "ednet_token":
                    for c, k_c in k_c_counter.items():
                        r_prev = float(self.r_post[uu, c])
                        self.r_post[uu, c] = (1.0 - eta) * r_prev + eta * (
                            e_group / float(k_c)
                        )
                        self.r_seen[uu, c] = True
                        self.last_r_update_ts[uu, c] = int(current_ts)
                else:
                    # ASSISTments MAIN_V1: one skill per row, and the
                    # group innovation for that skill is the mean of
                    # (y - p_global) over rows carrying that skill.
                    sums: dict[int, float] = {}
                    counts: dict[int, int] = {}
                    row_error = y.astype(np.float64) - p_g
                    for j, tags in enumerate(buf_tags):
                        if len(tags) != 1:
                            raise ValueError(
                                "assistments_skill_mean requires exactly "
                                "one KC id per row"
                            )
                        concept = int(tags[0])
                        sums[concept] = sums.get(concept, 0.0) + float(row_error[j])
                        counts[concept] = counts.get(concept, 0) + 1
                    clock = int(buf_glt[0])
                    for concept, count in counts.items():
                        e_skill = sums[concept] / float(count)
                        r_prev = float(self.r_post[uu, concept])
                        self.r_post[uu, concept] = (
                            (1.0 - eta) * r_prev + eta * e_skill
                        )
                        self.r_seen[uu, concept] = True
                        self.last_r_update_ts[uu, concept] = clock

            theta_post = float(self.theta[int(s[0])])
            b_post = self.b[p].copy()
            k_c_pairs = tuple(k_c_counter.items())
            run_start = buf_idx[0]
            run_end = buf_idx[-1]
            for j, idx in enumerate(buf_idx):
                tags = tuple(buf_tags[j])
                headline = float(p_a[j] if self.headline_art else p_g[j])
                traces.append(
                    StreamingRowTrace(
                        row_index=idx,
                        interaction_id=buf_ids[j],
                        student_idx=int(s[j]),
                        problem_idx=int(p[j]),
                        group_id=buf_g[0],
                        run_id=run_id,
                        run_start=run_start,
                        run_end=run_end,
                        theta_pre=theta_pre,
                        b_pre=float(b_pre[j]),
                        probability=headline,
                        correct=int(y[j]),
                        theta_post=theta_post,
                        b_post=float(b_post[j]),
                        p_global=float(p_g[j]),
                        p_art=float(p_a[j]),
                        kc_ids=tags,
                        r_post_pre=r_pre[j],
                        r_bar=float(r_bar[j]),
                        e_group=e_group,
                        k_c=k_c_pairs,
                        r_post_post=_kc_values(self.r_post, int(s[j]), buf_tags[j]),
                    )
                )
            run_id += 1
            buf_idx.clear()
            buf_s.clear()
            buf_p.clear()
            buf_y.clear()
            buf_glt.clear()
            buf_tags.clear()
            buf_ids.clear()
            buf_g.clear()

        for i, row in enumerate(rows):
            sid = row.student_idx
            gid = row.group_id
            if sid != cur_sid or gid != cur_gid:
                flush()
                cur_sid, cur_gid = sid, gid
            buf_idx.append(i)
            buf_s.append(sid)
            buf_p.append(row.problem_idx)
            buf_y.append(row.correct)
            buf_glt.append(row.group_last_timestamp_ms)
            buf_tags.append(list(row.tag_idxs))
            buf_ids.append(row.interaction_id)
            buf_g.append(gid)
        flush()
        if len(traces) != n:
            raise RuntimeError(
                f"trace row count {len(traces)} != input row count {n}"
            )
        return StreamingResult(tuple(traces))
