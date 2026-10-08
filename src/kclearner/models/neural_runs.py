"""Learner-wise contiguous solving-id runs for legacy DKT / DKVMN.

This is not a public sequence engine. It matches
``project_c/models/ednet_dkt.py:list_user_bundles``:

* group rows by learner;
* keep each learner's physical appearance order;
* split when ``solving_id`` changes inside that subsequence;
* other learners between two of this learner's rows do not split a run;
* the same ``solving_id`` later in the subsequence is a new run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class NeuralRun:
    learner_index: int
    solving_id: int
    row_positions: tuple[int, ...]

    @property
    def length(self) -> int:
        return len(self.row_positions)


def learner_solving_runs(
    student_idx: Sequence[int],
    solving_id: Sequence[int],
) -> tuple[NeuralRun, ...]:
    """Return runs in ascending learner index, then physical order."""

    n = len(student_idx)
    if n != len(solving_id):
        raise ValueError("student_idx and solving_id must have the same length")
    positions: dict[int, list[int]] = {}
    for index in range(n):
        learner = int(student_idx[index])
        positions.setdefault(learner, []).append(index)
    runs: list[NeuralRun] = []
    for learner in sorted(positions):
        rows = positions[learner]
        start = 0
        previous = int(solving_id[rows[0]])
        for offset in range(1, len(rows)):
            current = int(solving_id[rows[offset]])
            if current != previous:
                chosen = tuple(rows[start:offset])
                runs.append(NeuralRun(learner, previous, chosen))
                start = offset
                previous = current
        runs.append(
            NeuralRun(learner, previous, tuple(rows[start:]))
        )
    return tuple(runs)
