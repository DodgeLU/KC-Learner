"""ASSISTments 2017 formal MAIN_V1 adapter.

This is the regression path for ``ASSISTMENTS2017_MAIN_V1``. It is not
the generic fractional splitter, and it does not read ``strict.csv``.

Formal source: ``primary.csv`` (``assist17_primary_v1``), already
restricted to ``attempt_count == 1`` by the export. This loader still
counts that rule, plus invalid responses and missing learner, item, or
skill. Blank skills are dropped and counted. They are not given an
invented KC id.

After ordering by ``(student_id, timestamp, source_row)``:

* nominal ``B_temporal_v1`` is a per-learner 70/10/20 cut
  (``round``, with the legacy short-history adjustment);
* a ``(student_id, timestamp)`` bundle that touches both nominal TRAIN
  and nominal VALID is moved entirely into VALID
  (``assist17_b_temporal_bundle_safe_v1``);
* the VALID/TEST boundary is not closed. Neural TEST replay treats
  those crossing bundles as context. This adapter only labels them.

Vocabulary is the sorted string identity of the derived TRAIN+VALID
union: problems, skills, and students. Index ``"10"`` sorts before
``"2"``. Model ``group_id`` is a separate numeric dense id over
``(student_id, timestamp)`` on that same union.

One source row stays one :class:`~kclearner.data.schema.Interaction`.
``kc_ids`` holds dense skill indexes. MAIN_V1's ``skill_id`` column is
one label, so the tuple has length 1. A caller can still place several
indexes in one tuple; this module does not split the row.

Problem OOV exists on the nominal TEST tail (949 rows in the frozen
universe). TRAIN and VALID are in-vocabulary by construction.

OOV families differ. See :func:`oov_behavior`.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterable, Mapping, Sequence, TextIO

from kclearner.data.schema import Interaction
from kclearner.data.validation import DatasetValidator
from kclearner.models.streaming import StreamingRow

PROTOCOL_ID = "ASSISTMENTS2017_MAIN_V1"
DATASET_VERSION = "assist17_primary_v1"
SPLIT_BASE = "B_temporal_v1"
SPLIT_POLICY = "assist17_b_temporal_bundle_safe_v1"
QUESTION_MAPPING_POLICY = "assist17_train_valid_problem_dense_v1"
KC_MAPPING_POLICY = "assist17_train_valid_skill_dense_v1"
IRT_OOV_PROBLEM_INDEX = 0

_REQUIRED = (
    "student_id",
    "problem_id",
    "skill_id",
    "correct",
    "timestamp",
    "attempt_count",
)

# Frozen AssistDKTQ / IRT provenance. Used by the integration check,
# not by synthetic loads.
# ``eligible_rows`` is the nominal accepted canonical total:
# TRAIN + VALID + nominal TEST = 428495.
# It is not ``metric_eligible_rows`` (427546) and not the
# metric-eligible TEST count (85741 - 949 = 84792).
FROZEN_MAIN_V1_COUNTS = {
    "eligible_rows": 428495,
    "train_rows": 299907,
    "valid_rows": 42847,
    "test_rows": 85741,
    "oov_rows": 949,
    "metric_eligible_rows": 427546,
    "n_students_vocab": 1709,
    "n_questions_vocab": 2909,
    "n_skills_vocab": 102,
    "n_train_users": 1709,
    "n_valid_users": 1707,
    "n_questions_full": 3162,
    "n_skills_full": 102,
    "promoted_to_valid": 3,
    "nominal_cross_boundary_bundles": 3,
    "train_bundles": 299345,
    "valid_bundles": 42702,
}


class AssistmentsAdapterError(ValueError):
    """The CSV cannot be converted under MAIN_V1."""


@dataclass(frozen=True)
class OOVBehavior:
    """What an out-of-vocabulary problem row does in one model family.

    ``eligible_for_metrics`` is false for every family. Whether the row
    still moves learner state is family-specific. The item fallback is
    the eval-phase rule: TEST freezes ``b``, so the sentinel difficulty
    is read and not written.
    """

    family: str
    receives_prediction: bool
    updates_learner_state: bool
    updates_item_state: bool
    updates_kc_state: bool
    eligible_for_metrics: bool
    state_transparent: bool
    fallback_problem_index: int | None


def oov_behavior(family: str) -> OOVBehavior:
    """Return the formal MAIN_V1 problem-OOV rule for ``family``."""

    if family == "irt":
        return OOVBehavior(
            family=family,
            receives_prediction=True,
            updates_learner_state=True,
            updates_item_state=False,
            updates_kc_state=False,
            eligible_for_metrics=False,
            state_transparent=False,
            fallback_problem_index=IRT_OOV_PROBLEM_INDEX,
        )
    if family == "ar_kt":
        return OOVBehavior(
            family=family,
            receives_prediction=True,
            updates_learner_state=True,
            updates_item_state=False,
            updates_kc_state=True,
            eligible_for_metrics=False,
            state_transparent=False,
            fallback_problem_index=IRT_OOV_PROBLEM_INDEX,
        )
    if family in ("dkt_q", "dkt_qc", "dkvmn_q", "dkvmn_qc"):
        return OOVBehavior(
            family=family,
            receives_prediction=False,
            updates_learner_state=False,
            updates_item_state=False,
            updates_kc_state=False,
            eligible_for_metrics=False,
            state_transparent=True,
            fallback_problem_index=None,
        )
    raise ValueError(f"unknown model family {family!r}")


@dataclass(frozen=True)
class AssistmentsDiagnostics:
    """Counts for one MAIN_V1 load. Nothing here is a predictive metric."""

    raw_rows: int
    eligible_rows: int
    filtered_non_first_attempt: int
    invalid_response_rows: int
    missing_learner_rows: int
    missing_item_rows: int
    missing_kc_rows: int
    learners: int
    items: int
    kcs: int
    bundles: int
    train_rows: int
    valid_rows: int
    test_rows: int
    train_bundles: int
    valid_bundles: int
    metric_eligible_rows: int
    oov_rows: int
    multi_kc_rows: int
    multi_kc_rate: float
    promoted_to_valid: int
    nominal_cross_boundary_bundles: int
    derived_cross_boundary_bundles: int
    n_students_vocab: int
    n_questions_vocab: int
    n_skills_vocab: int
    n_train_users: int
    n_valid_users: int
    n_questions_full: int
    n_skills_full: int


@dataclass(frozen=True)
class MainV1Vocabulary:
    """Dense TRAIN+VALID maps. Order is sorted ``str`` identity."""

    students_sorted: tuple[str, ...]
    problems_sorted: tuple[str, ...]
    skills_sorted: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_students",
            {value: index for index, value in enumerate(self.students_sorted)},
        )
        object.__setattr__(
            self,
            "_problems",
            {value: index for index, value in enumerate(self.problems_sorted)},
        )
        object.__setattr__(
            self,
            "_skills",
            {value: index for index, value in enumerate(self.skills_sorted)},
        )

    def student_index(self, learner_id: str) -> int:
        return self._students[learner_id]

    def problem_index(self, item_id: str) -> int:
        return self._problems[item_id]

    def skill_index(self, skill_id: str) -> int:
        return self._skills[skill_id]

    def has_student(self, learner_id: str) -> bool:
        return learner_id in self._students

    def has_problem(self, item_id: str) -> bool:
        return item_id in self._problems

    def has_skill(self, skill_id: str) -> bool:
        return skill_id in self._skills


@dataclass(frozen=True)
class AssistmentsModelRow:
    """One TRAIN or VALID row in the formal dense index space."""

    student_idx: int
    problem_idx: int
    skill_idx: int
    group_id: int
    correct: int
    timestamp: int
    split: str
    metric_mask: bool
    oov_problem: bool
    interaction_id: str
    learner_id: str
    item_id: str
    skill_id: str


@dataclass(frozen=True)
class AssistmentsMainV1:
    interactions: tuple[Interaction, ...]
    diagnostics: AssistmentsDiagnostics
    vocabulary: MainV1Vocabulary
    train_rows: tuple[AssistmentsModelRow, ...]
    valid_rows: tuple[AssistmentsModelRow, ...]
    source_indexes: tuple[int, ...]


@dataclass(frozen=True)
class _Parsed:
    student_id: int
    problem_id: int
    skill_id: str
    correct: int
    timestamp: int
    source_index: int
    drop_reason: str | None


def bundle_key(learner_id: str, timestamp: int) -> str:
    """Formal bundle identity ``(student_id, timestamp)``."""

    return f"{learner_id}|{int(timestamp)}"


def kc_index_tuple(
    skill_labels: Sequence[str],
    skill_to_index: Mapping[str, int],
) -> tuple[int, ...]:
    """Map skill labels onto dense indexes without splitting the row.

    Order and duplicates are kept. An unknown label is an error, not a
    new KC.
    """

    out: list[int] = []
    for label in skill_labels:
        if label not in skill_to_index:
            raise KeyError(f"skill label {label!r} is not in the vocabulary")
        out.append(int(skill_to_index[label]))
    return tuple(out)


def problem_index_for_streaming(
    item_id: str,
    vocabulary: MainV1Vocabulary,
) -> tuple[int, bool]:
    """Return ``(problem_idx, metric_eligible)``.

    An unknown problem uses the in-bounds sentinel ``0`` and is not
    eligible for metrics. That is the IRT / AR-KT eval rule.
    """

    if vocabulary.has_problem(item_id):
        return vocabulary.problem_index(item_id), True
    return IRT_OOV_PROBLEM_INDEX, False


def rows_for_neural_state(
    rows: Sequence[AssistmentsModelRow],
    family: str,
) -> tuple[AssistmentsModelRow, ...]:
    """Drop state-transparent problem-OOV rows for a neural family."""

    policy = oov_behavior(family)
    if not policy.state_transparent:
        return tuple(rows)
    return tuple(row for row in rows if not row.oov_problem)


def execution_group_ids(
    keys: Sequence[object],
    existing: Sequence[int | None],
) -> tuple[int, ...]:
    """Fill missing execution bundle ids.

    TRAIN and VALID already store a dense id. The frozen protocol
    bundles ``(student_id, timestamp)``. A missing id must not collapse
    every TEST row of one learner into one bundle. The numeric value is
    only an equality key for the streaming scan. It is not part of
    dataset logical identity. The historical TEST script left
    ``group_id`` unset, and that walker then used the row index, so
    same-timestamp TEST rows updated one at a time. This function
    follows the protocol bundle.
    """

    if len(keys) != len(existing):
        raise ValueError("group keys and existing ids have different lengths")
    assigned: dict[tuple[str, int], int] = {}
    known = [int(current) for current in existing if current is not None and int(current) >= 0]
    next_id = (max(known) + 1) if known else 0
    filled: list[int] = []
    for key, current in zip(keys, existing):
        if current is not None and int(current) >= 0:
            filled.append(int(current))
            continue
        if key not in assigned:
            assigned[key] = next_id
            next_id += 1
        filled.append(assigned[key])
    return tuple(filled)


def to_streaming_rows(
    rows: Sequence[AssistmentsModelRow],
    *,
    family: str,
) -> list[StreamingRow]:
    """Map dense MAIN_V1 rows onto the existing streaming walker.

    AR-KT carries the single skill index. IRT ignores KC ids. Problem
    OOV rows, if present, already store the sentinel problem index.
    """

    if family not in ("irt", "ar_kt"):
        raise ValueError(f"streaming family must be irt or ar_kt, got {family!r}")
    use_skill = family == "ar_kt"
    built: list[StreamingRow] = []
    for row in rows:
        built.append(
            StreamingRow(
                student_idx=row.student_idx,
                problem_idx=row.problem_idx,
                group_id=row.group_id,
                correct=row.correct,
                tag_idxs=(row.skill_idx,) if use_skill else (),
                interaction_id=row.interaction_id,
                timestamp_ms=row.timestamp,
                group_last_timestamp_ms=row.timestamp,
            )
        )
    return built


def load_assistments2017_main_v1(
    source: str | Path | IO[str],
) -> AssistmentsMainV1:
    """Load a MAIN_V1 CSV into canonical interactions and dense rows.

    TRAIN and VALID model rows are returned separately. Nominal TEST
    rows stay on ``interactions`` with ``split="test"`` so the loader
    can count them. This function does not score TEST.
    """

    parsed = _read_rows(source)
    eligible = [row for row in parsed if row.drop_reason is None]
    eligible.sort(key=lambda row: (row.student_id, row.timestamp, row.source_index))
    labels, promoted, nominal_cross = _assign_splits(eligible)
    if _derived_crossings(eligible, labels) != 0:
        raise AssistmentsAdapterError(
            "bundle closure left a TRAIN/VALID crossing bundle"
        )
    vocabulary = _vocabulary(eligible, labels)
    group_of = _group_ids(eligible, labels)
    interactions, train_rows, valid_rows = _materialize(
        eligible, labels, vocabulary, group_of
    )
    train_valid = tuple(
        row for row in interactions if row.split in ("train", "valid")
    )
    checked = DatasetValidator().validate(train_valid)
    if not checked.ok:
        codes = ", ".join(issue.code for issue in checked.issues[:8])
        raise AssistmentsAdapterError(
            f"TRAIN/VALID interactions failed validation: {codes}"
        )
    diagnostics = _diagnostics(
        parsed,
        interactions,
        vocabulary,
        train_rows,
        valid_rows,
        promoted,
        nominal_cross,
    )
    source_indexes = tuple(row.source_index for row in eligible)
    if len(source_indexes) != len(interactions):
        raise AssistmentsAdapterError("source_row provenance is not aligned")
    return AssistmentsMainV1(
        interactions=interactions,
        diagnostics=diagnostics,
        vocabulary=vocabulary,
        train_rows=tuple(train_rows),
        valid_rows=tuple(valid_rows),
        source_indexes=source_indexes,
    )


def _read_rows(source: str | Path | IO[str]) -> tuple[_Parsed, ...]:
    if isinstance(source, (str, Path)):
        with Path(source).open("r", encoding="utf-8", newline="") as handle:
            return _parse_csv(handle)
    return _parse_csv(source)


def _parse_csv(handle: Iterable[str] | TextIO) -> tuple[_Parsed, ...]:
    reader = csv.DictReader(handle)
    if reader.fieldnames is None:
        raise AssistmentsAdapterError("ASSISTments CSV has no header")
    fields = set(reader.fieldnames)
    missing = [name for name in _REQUIRED if name not in fields]
    if missing:
        raise AssistmentsAdapterError(
            f"ASSISTments CSV is missing columns {missing}"
        )
    rows: list[_Parsed] = []
    for source_index, record in enumerate(reader):
        rows.append(_parse_record(record, source_index))
    return tuple(rows)


def _cell(record: Mapping[str, str | None], name: str) -> str:
    value = record.get(name)
    if value is None:
        return ""
    return str(value).strip()


def _parse_record(record: Mapping[str, str | None], source_index: int) -> _Parsed:
    learner = _cell(record, "student_id")
    item = _cell(record, "problem_id")
    skill = _cell(record, "skill_id")
    correct_text = _cell(record, "correct")
    timestamp_text = _cell(record, "timestamp")
    attempt_text = _cell(record, "attempt_count")
    reason: str | None = None
    student_id = 0
    problem_id = 0
    correct = 0
    timestamp = 0
    if not _is_int_token(learner):
        reason = "missing_learner"
    elif not _is_int_token(item):
        reason = "missing_item"
    elif correct_text not in ("0", "1"):
        reason = "invalid_response"
    elif not _is_int_token(timestamp_text):
        reason = "invalid_response"
    elif not _is_int_token(attempt_text):
        reason = "invalid_response"
    elif int(attempt_text) != 1:
        reason = "non_first_attempt"
    elif skill == "":
        reason = "missing_kc"
    else:
        student_id = int(learner)
        problem_id = int(item)
        correct = int(correct_text)
        timestamp = int(timestamp_text)
    return _Parsed(
        student_id=student_id,
        problem_id=problem_id,
        skill_id=skill,
        correct=correct,
        timestamp=timestamp,
        source_index=source_index,
        drop_reason=reason,
    )


def _is_int_token(value: str) -> bool:
    if value == "":
        return False
    if value[0] == "-":
        return value[1:].isdigit()
    return value.isdigit()


def _assign_splits(
    rows: Sequence[_Parsed],
) -> tuple[list[str], int, int]:
    by_student: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_student[row.student_id].append(index)
    nominal = [""] * len(rows)
    for student_id in sorted(by_student):
        indexes = by_student[student_id]
        n_i = len(indexes)
        n_train = max(1, int(round(n_i * 0.70)))
        n_valid = max(1, int(round(n_i * 0.10)))
        if n_train + n_valid >= n_i:
            n_valid = max(0, n_i - n_train - 1)
        for index in indexes[:n_train]:
            nominal[index] = "train"
        for index in indexes[n_train:n_train + n_valid]:
            nominal[index] = "valid"
        for index in indexes[n_train + n_valid:]:
            nominal[index] = "test"
    splits_of: dict[tuple[int, int], set[str]] = defaultdict(set)
    for index, row in enumerate(rows):
        if nominal[index] in ("train", "valid"):
            splits_of[(row.student_id, row.timestamp)].add(nominal[index])
    crossing = {
        key for key, seen in splits_of.items() if seen == {"train", "valid"}
    }
    labels = list(nominal)
    promoted = 0
    for index, row in enumerate(rows):
        key = (row.student_id, row.timestamp)
        if key in crossing and nominal[index] == "train":
            labels[index] = "valid"
            promoted += 1
    return labels, promoted, len(crossing)


def _derived_crossings(rows: Sequence[_Parsed], labels: Sequence[str]) -> int:
    splits_of: dict[tuple[int, int], set[str]] = defaultdict(set)
    for index, row in enumerate(rows):
        if labels[index] in ("train", "valid"):
            splits_of[(row.student_id, row.timestamp)].add(labels[index])
    return sum(len(seen) > 1 for seen in splits_of.values())


def _vocabulary(
    rows: Sequence[_Parsed],
    labels: Sequence[str],
) -> MainV1Vocabulary:
    students: set[str] = set()
    problems: set[str] = set()
    skills: set[str] = set()
    for index, row in enumerate(rows):
        if labels[index] not in ("train", "valid"):
            continue
        students.add(str(row.student_id))
        problems.add(str(row.problem_id))
        skills.add(row.skill_id)
    return MainV1Vocabulary(
        students_sorted=tuple(sorted(students)),
        problems_sorted=tuple(sorted(problems)),
        skills_sorted=tuple(sorted(skills)),
    )


def _group_ids(
    rows: Sequence[_Parsed],
    labels: Sequence[str],
) -> dict[tuple[int, int], int]:
    keys = sorted(
        {
            (rows[index].student_id, rows[index].timestamp)
            for index, label in enumerate(labels)
            if label in ("train", "valid")
        }
    )
    return {key: index for index, key in enumerate(keys)}


def _materialize(
    rows: Sequence[_Parsed],
    labels: Sequence[str],
    vocabulary: MainV1Vocabulary,
    group_of: Mapping[tuple[int, int], int],
) -> tuple[tuple[Interaction, ...], list[AssistmentsModelRow], list[AssistmentsModelRow]]:
    interactions: list[Interaction] = []
    train_rows: list[AssistmentsModelRow] = []
    valid_rows: list[AssistmentsModelRow] = []
    for index, row in enumerate(rows):
        learner_id = str(row.student_id)
        item_id = str(row.problem_id)
        label = labels[index]
        in_problem = vocabulary.has_problem(item_id)
        in_skill = vocabulary.has_skill(row.skill_id)
        in_student = vocabulary.has_student(learner_id)
        if label in ("train", "valid"):
            if not in_problem or not in_skill or not in_student:
                raise AssistmentsAdapterError(
                    "TRAIN/VALID row fell outside the TRAIN+VALID vocabulary"
                )
            skill_idx = vocabulary.skill_index(row.skill_id)
            problem_idx = vocabulary.problem_index(item_id)
            student_idx = vocabulary.student_index(learner_id)
            group_id = group_of[(row.student_id, row.timestamp)]
            metric_mask = True
            oov_problem = False
        else:
            metric_mask = in_problem and in_skill and in_student
        kc_ids: tuple[int, ...]
        if in_skill:
            kc_ids = (vocabulary.skill_index(row.skill_id),)
        else:
            kc_ids = ()
        interaction_id = f"assist17:{index}"
        interaction = Interaction(
            interaction_id=interaction_id,
            learner_id=learner_id,
            item_id=item_id,
            correct=row.correct,
            kc_ids=kc_ids,
            order=index,
            timestamp=float(row.timestamp),
            bundle_id=bundle_key(learner_id, row.timestamp),
            split=label,
            metric_mask=metric_mask,
        )
        interactions.append(interaction)
        if label not in ("train", "valid"):
            continue
        model_row = AssistmentsModelRow(
            student_idx=student_idx,
            problem_idx=problem_idx,
            skill_idx=skill_idx,
            group_id=group_id,
            correct=row.correct,
            timestamp=row.timestamp,
            split=label,
            metric_mask=True,
            oov_problem=oov_problem,
            interaction_id=interaction_id,
            learner_id=learner_id,
            item_id=item_id,
            skill_id=row.skill_id,
        )
        if label == "train":
            train_rows.append(model_row)
        else:
            valid_rows.append(model_row)
    return tuple(interactions), train_rows, valid_rows


def _diagnostics(
    parsed: Sequence[_Parsed],
    interactions: Sequence[Interaction],
    vocabulary: MainV1Vocabulary,
    train_rows: Sequence[AssistmentsModelRow],
    valid_rows: Sequence[AssistmentsModelRow],
    promoted: int,
    nominal_cross: int,
) -> AssistmentsDiagnostics:
    reasons = [row.drop_reason for row in parsed if row.drop_reason is not None]
    learners = {row.learner_id for row in interactions}
    items = {row.item_id for row in interactions}
    skills = {row.skill_id for row in parsed if row.drop_reason is None}
    problems = {str(row.problem_id) for row in parsed if row.drop_reason is None}
    bundles = {row.bundle_id for row in interactions}
    train_users = {row.student_idx for row in train_rows}
    valid_users = {row.student_idx for row in valid_rows}
    multi = sum(len(row.kc_ids) > 1 for row in interactions)
    eligible = len(interactions)
    oov_rows = sum(not row.metric_mask for row in interactions)
    return AssistmentsDiagnostics(
        raw_rows=len(parsed),
        eligible_rows=eligible,
        filtered_non_first_attempt=reasons.count("non_first_attempt"),
        invalid_response_rows=reasons.count("invalid_response"),
        missing_learner_rows=reasons.count("missing_learner"),
        missing_item_rows=reasons.count("missing_item"),
        missing_kc_rows=reasons.count("missing_kc"),
        learners=len(learners),
        items=len(items),
        kcs=len(skills),
        bundles=len(bundles),
        train_rows=len(train_rows),
        valid_rows=len(valid_rows),
        test_rows=sum(row.split == "test" for row in interactions),
        train_bundles=len({row.group_id for row in train_rows}),
        valid_bundles=len({row.group_id for row in valid_rows}),
        metric_eligible_rows=sum(row.metric_mask for row in interactions),
        oov_rows=oov_rows,
        multi_kc_rows=multi,
        multi_kc_rate=(float(multi) / float(eligible)) if eligible else 0.0,
        promoted_to_valid=promoted,
        nominal_cross_boundary_bundles=nominal_cross,
        derived_cross_boundary_bundles=0,
        n_students_vocab=len(vocabulary.students_sorted),
        n_questions_vocab=len(vocabulary.problems_sorted),
        n_skills_vocab=len(vocabulary.skills_sorted),
        n_train_users=len(train_users),
        n_valid_users=len(valid_users),
        n_questions_full=len(problems),
        n_skills_full=len(skills),
    )
