"""Select a deterministic dev5000 TRAIN window.

The rule is documented in ``FIXTURE_MANIFEST.md``. This module does not
read parquet files and does not import ``project_c``.
"""

from __future__ import annotations

from typing import Sequence


class _Acc:
    """Predicate state for a growing span of complete execution runs.

    Adding a finished run never clears a predicate. A run is already a
    maximal contiguous ``(student, group)`` block, so it cannot turn a
    one-row run into a multi-row run.
    """

    def __init__(self) -> None:
        self.single_runs = 0
        self.multi_runs = 0
        self.group_runs: dict[tuple[int, int], int] = {}
        self.reappear = False
        self.problem_count: dict[int, int] = {}
        self.repeated_problem = False
        self.has0 = False
        self.has1 = False
        self.students: set[int] = set()
        self.multi_kc = False
        self.dup_token = False
        self.repeat_exposure = False
        self.tag_runs: dict[int, int] = {}

    def add_run(
        self,
        student: int,
        group: int,
        n_rows: int,
        has0: bool,
        has1: bool,
        problems: dict[int, int],
        concepts: dict[int, int],
        multi_kc: bool,
        dup_token: bool,
    ) -> None:
        if n_rows <= 0:
            return
        if n_rows == 1:
            self.single_runs += 1
        else:
            self.multi_runs += 1
        key = (student, group)
        self.group_runs[key] = self.group_runs.get(key, 0) + 1
        if self.group_runs[key] >= 2:
            self.reappear = True
        self.students.add(student)
        self.has0 = self.has0 or has0
        self.has1 = self.has1 or has1
        self.multi_kc = self.multi_kc or multi_kc
        self.dup_token = self.dup_token or dup_token
        for problem, count in problems.items():
            seen = self.problem_count.get(problem, 0) + count
            self.problem_count[problem] = seen
            if seen >= 2:
                self.repeated_problem = True
        for concept in concepts:
            seen = self.tag_runs.get(concept, 0) + 1
            self.tag_runs[concept] = seen
            if seen >= 2:
                self.repeat_exposure = True

    def ok(self) -> bool:
        return (
            self.single_runs >= 1
            and self.multi_runs >= 1
            and self.reappear
            and self.repeated_problem
            and self.has0
            and self.has1
            and len(self.students) >= 2
            and self.multi_kc
            and self.dup_token
            and self.repeat_exposure
        )


def find_window(
    student: Sequence[int],
    group: Sequence[int],
    problem: Sequence[int],
    correct: Sequence[int],
    tags: Sequence[Sequence[int]],
    sentinel: Sequence[bool],
    max_len: int | None = None,
) -> tuple[int, int]:
    """Return ``[start, end)`` for the earliest qualifying slice.

    The slice starts and ends on execution-run boundaries. A run that
    contains a raw ``-1`` tag is a barrier and is not included. ``end``
    is exclusive.
    """

    n = len(student)
    if not (
        n == len(group) == len(problem) == len(correct) == len(tags) == len(sentinel)
    ):
        raise ValueError("window columns must have the same length")

    def finish(acc: _Acc, start: int, end: int) -> tuple[int, int] | None:
        if not acc.ok():
            return None
        if max_len is not None and (end - start) > max_len:
            return None
        return start, end

    acc = _Acc()
    segment_start = 0
    run_start = 0
    poisoned = False
    has0 = False
    has1 = False
    multi_kc = False
    dup_token = False
    problems: dict[int, int] = {}
    concepts: dict[int, int] = {}

    def close_run(run_end: int) -> tuple[int, int] | None:
        nonlocal acc, segment_start, poisoned, has0, has1, multi_kc, dup_token
        nonlocal problems, concepts
        if run_end <= run_start:
            return None
        if poisoned:
            found = finish(acc, segment_start, run_start)
            acc = _Acc()
            segment_start = run_end
        else:
            acc.add_run(
                int(student[run_start]),
                int(group[run_start]),
                run_end - run_start,
                has0,
                has1,
                problems,
                concepts,
                multi_kc,
                dup_token,
            )
            found = finish(acc, segment_start, run_end)
        poisoned = False
        has0 = False
        has1 = False
        multi_kc = False
        dup_token = False
        problems = {}
        concepts = {}
        return found

    for index in range(n):
        if (
            index > run_start
            and (
                int(student[index]) != int(student[run_start])
                or int(group[index]) != int(group[run_start])
            )
        ):
            found = close_run(index)
            if found is not None:
                return found
            run_start = index
        tag_list = tags[index]
        if sentinel[index]:
            poisoned = True
        if int(correct[index]) == 0:
            has0 = True
        elif int(correct[index]) == 1:
            has1 = True
        problem_id = int(problem[index])
        problems[problem_id] = problems.get(problem_id, 0) + 1
        if len(tag_list) >= 2:
            multi_kc = True
        if len(tag_list) != len(set(tag_list)):
            dup_token = True
        for concept in tag_list:
            concepts[int(concept)] = concepts.get(int(concept), 0) + 1
    found = close_run(n)
    if found is not None:
        return found
    raise LookupError(
        "no dev5000 TRAIN execution-run span met the parity-window predicates"
    )
