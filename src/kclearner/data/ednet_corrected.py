"""Corrected public EdNet-KT1 preprocessing, ``ednet_kt1_corrected_v1``.

Reads original per-user KT1 files in CSV order. It does not read the
legacy row-hash shards, and it does not sort by timestamp or
``solving_id``.

The formal dev5000 population is the first 5,000 user ids in
lexicographic order over the unique users of the historical
``ednet_split_v1`` TRAIN ∪ VALID tables. That is the same cut as
``select_dev5000`` after ``build_vocab``.

Split labels use the historical bundle rule: per learner, unique
``solving_id`` values in numeric order, then 70/10/20. A bundle is not
divided. KC normalization does not move a row between splits.

User-id lists are a local generated artifact. The committed record is
the SHA-256 of the ordered ids.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from kclearner.data.adapters.ednet_kt1 import (
    EdNetFilterStats,
    load_raw_user_kt1,
)
from kclearner.data.adapters.ednet_metadata import (
    EdNetQuestionMetadata,
    load_ednet_question_metadata,
)
from kclearner.data.schema import Interaction

PREPROCESSING_VERSION = "ednet_kt1_corrected_v1"
SCHEMA_VERSION = "kclearner_interaction_v1"
COHORT_ID = "ednet_kt1_dev5000"
SPLIT_ID = "ednet_split_v1_solving_id_70_10_20"
COHORT_SIZE = 5000


class CorrectedEdNetError(RuntimeError):
    """The corrected build violated a frozen source invariant."""


def cohort_id_hash(user_ids: Sequence[str]) -> str:
    """SHA-256 of the ordered ids joined by newlines, with no trailing newline."""

    payload = "\n".join(user_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_first_sorted_users(user_ids: Iterable[str], n_users: int) -> tuple[str, ...]:
    """Return the first ``n_users`` distinct ids in lexicographic order.

    This matches ``numpy.sort`` on the ASCII EdNet user ids used by the
    historical dense ``student_idx``.
    """

    if n_users < 1:
        raise ValueError(f"n_users must be >= 1, got {n_users}")
    ordered = tuple(sorted(set(user_ids)))
    if n_users > len(ordered):
        raise ValueError(
            f"requested {n_users} users but only {len(ordered)} are available"
        )
    return ordered[:n_users]


def split_bundle_counts(n_bundles: int) -> tuple[int, int, int]:
    """Historical per-learner 70/10/20 bundle counts.

    ``n_train = max(1, round(n * 0.70))`` and the same adjustment the
    ``ednet_split_v1`` builder records for short histories.
    """

    if n_bundles < 1:
        raise ValueError("n_bundles must be >= 1")
    n_train = max(1, int(round(n_bundles * 0.70)))
    n_valid = max(1, int(round(n_bundles * 0.10)))
    if n_train + n_valid >= n_bundles:
        n_valid = max(0, n_bundles - n_train - 1)
    n_test = n_bundles - n_train - n_valid
    return n_train, n_valid, n_test


def assign_solving_splits(
    interactions: Sequence[Interaction],
    solving_ids: Sequence[int],
) -> tuple[Interaction, ...]:
    """Label every row of a ``solving_id`` with the same split.

    Bundle order for the cut is numeric ``solving_id``, not source
    order. The returned tuple keeps the input row order.
    """

    if len(interactions) != len(solving_ids):
        raise ValueError("interactions and solving_ids must have the same length")
    by_learner: dict[str, list[int]] = defaultdict(list)
    for index, (row, solving_id) in enumerate(zip(interactions, solving_ids)):
        by_learner[row.learner_id].append(index)
    labels = [""] * len(interactions)
    for indexes in by_learner.values():
        ordered_ids = sorted({int(solving_ids[index]) for index in indexes})
        n_train, n_valid, _n_test = split_bundle_counts(len(ordered_ids))
        label_of = {}
        for position, solving_id in enumerate(ordered_ids):
            if position < n_train:
                label_of[solving_id] = "train"
            elif position < n_train + n_valid:
                label_of[solving_id] = "valid"
            else:
                label_of[solving_id] = "test"
        for index in indexes:
            labels[index] = label_of[int(solving_ids[index])]
    return tuple(
        replace(row, split=labels[index])
        for index, row in enumerate(interactions)
    )


def row_identity(
    user_id: str,
    source_row: int,
    question_id: str,
    solving_id: int,
    timestamp: int,
) -> tuple[str, int, str, int, int]:
    """Stable identity. KC text is not part of it."""

    return (user_id, int(source_row), question_id, int(solving_id), int(timestamp))


@dataclass(frozen=True)
class CorrectedEdNetRow:
    interaction: Interaction
    source_row: int
    solving_id: int

    @property
    def identity(self) -> tuple[str, int, str, int, int]:
        timestamp = self.interaction.timestamp
        if timestamp is None:
            raise CorrectedEdNetError(
                f"{self.interaction.interaction_id} has no timestamp"
            )
        return row_identity(
            self.interaction.learner_id,
            self.source_row,
            self.interaction.item_id,
            self.solving_id,
            int(timestamp),
        )


@dataclass(frozen=True)
class SequenceInvariants:
    noncontiguous_bundles: int
    solving_id_inversions: int
    timestamp_inversions: int


def sequence_invariants(rows: Sequence[CorrectedEdNetRow]) -> SequenceInvariants:
    """Count order defects inside each learner's source order."""

    by_learner: dict[str, list[CorrectedEdNetRow]] = defaultdict(list)
    for row in rows:
        by_learner[row.interaction.learner_id].append(row)
    noncontiguous = 0
    solving_inversions = 0
    timestamp_inversions = 0
    for learner_rows in by_learner.values():
        ordered = sorted(learner_rows, key=lambda item: item.interaction.order)
        closed: set[int] = set()
        open_solving: int | None = None
        previous_timestamp: int | None = None
        for row in ordered:
            solving_id = row.solving_id
            if open_solving is None or solving_id != open_solving:
                if open_solving is not None:
                    closed.add(open_solving)
                if solving_id in closed:
                    noncontiguous += 1
                    closed.remove(solving_id)
                if open_solving is not None and solving_id < open_solving:
                    solving_inversions += 1
                open_solving = solving_id
            timestamp = row.interaction.timestamp
            if timestamp is not None and previous_timestamp is not None:
                if int(timestamp) < previous_timestamp:
                    timestamp_inversions += 1
            if timestamp is not None:
                previous_timestamp = int(timestamp)
    return SequenceInvariants(
        noncontiguous_bundles=noncontiguous,
        solving_id_inversions=solving_inversions,
        timestamp_inversions=timestamp_inversions,
    )


@dataclass(frozen=True)
class CorrectedVocabulary:
    scope: str
    learner_count: int
    item_count: int
    kc_count: int
    learner_ids_sha256: str
    item_ids_sha256: str
    kc_ids_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "scope": self.scope,
            "learner_vocabulary_size": self.learner_count,
            "item_vocabulary_size": self.item_count,
            "kc_vocabulary_size": self.kc_count,
            "learner_ids_sha256": self.learner_ids_sha256,
            "item_ids_sha256": self.item_ids_sha256,
            "kc_ids_sha256": self.kc_ids_sha256,
        }


def vocabulary_from_rows(rows: Sequence[CorrectedEdNetRow]) -> CorrectedVocabulary:
    """Source-id vocabulary over TRAIN+VALID. Rows stay in source ids."""

    chosen = [
        row for row in rows
        if row.interaction.split in ("train", "valid")
    ]
    learners = sorted({row.interaction.learner_id for row in chosen})
    items = sorted({row.interaction.item_id for row in chosen})
    kcs = sorted({kc for row in chosen for kc in row.interaction.kc_ids})
    return CorrectedVocabulary(
        scope="train+valid",
        learner_count=len(learners),
        item_count=len(items),
        kc_count=len(kcs),
        learner_ids_sha256=cohort_id_hash(learners),
        item_ids_sha256=cohort_id_hash(items),
        kc_ids_sha256=cohort_id_hash(tuple(str(kc) for kc in kcs)),
    )


@dataclass(frozen=True)
class CorrectedDiagnostics:
    raw_rows: int
    eligible_rows: int
    filtered_rows: int
    filter_reasons: tuple[tuple[str, int], ...]
    learner_count: int
    item_count: int
    real_kc_count: int
    missing_kc_rows: int
    multi_kc_rows: int
    duplicate_kc_rows: int
    bundle_count: int
    min_bundle_size: int
    mean_bundle_size: float
    max_bundle_size: int
    noncontiguous_bundles: int
    solving_id_inversions: int
    timestamp_inversions: int
    train_rows: int
    valid_rows: int
    test_rows: int

    def to_dict(self) -> dict[str, object]:
        return {
            "raw_rows": self.raw_rows,
            "eligible_rows": self.eligible_rows,
            "filtered_rows": self.filtered_rows,
            "filter_reasons": [
                {"reason": reason, "count": count}
                for reason, count in self.filter_reasons
            ],
            "learner_count": self.learner_count,
            "item_count": self.item_count,
            "real_kc_count": self.real_kc_count,
            "missing_kc_rows": self.missing_kc_rows,
            "multi_kc_rows": self.multi_kc_rows,
            "duplicate_kc_rows": self.duplicate_kc_rows,
            "bundle_count": self.bundle_count,
            "min_bundle_size": self.min_bundle_size,
            "mean_bundle_size": self.mean_bundle_size,
            "max_bundle_size": self.max_bundle_size,
            "noncontiguous_bundles": self.noncontiguous_bundles,
            "solving_id_inversions": self.solving_id_inversions,
            "timestamp_inversions": self.timestamp_inversions,
            "train_rows": self.train_rows,
            "valid_rows": self.valid_rows,
            "test_rows": self.test_rows,
        }


def diagnostics_from_build(
    rows: Sequence[CorrectedEdNetRow],
    stats: Sequence[EdNetFilterStats],
) -> CorrectedDiagnostics:
    reasons: Counter[str] = Counter()
    raw_rows = 0
    for item in stats:
        raw_rows += item.raw_rows
        for reason in item.dropped_reasons:
            reasons[reason.reason] += reason.count
    interactions = [row.interaction for row in rows]
    bundle_sizes: Counter[str] = Counter(row.interaction.bundle_id for row in rows)
    sizes = tuple(bundle_sizes.values())
    kcs = {kc for row in interactions for kc in row.kc_ids}
    invariants = sequence_invariants(rows)
    eligible = len(rows)
    return CorrectedDiagnostics(
        raw_rows=raw_rows,
        eligible_rows=eligible,
        filtered_rows=raw_rows - eligible,
        filter_reasons=tuple(sorted(reasons.items())),
        learner_count=len({row.interaction.learner_id for row in rows}),
        item_count=len({row.interaction.item_id for row in rows}),
        real_kc_count=len(kcs),
        missing_kc_rows=sum(len(row.interaction.kc_ids) == 0 for row in rows),
        multi_kc_rows=sum(len(row.interaction.kc_ids) > 1 for row in rows),
        duplicate_kc_rows=sum(
            len(row.interaction.kc_ids) != len(set(row.interaction.kc_ids))
            for row in rows
        ),
        bundle_count=len(bundle_sizes),
        min_bundle_size=min(sizes) if sizes else 0,
        mean_bundle_size=(float(eligible) / float(len(sizes))) if sizes else 0.0,
        max_bundle_size=max(sizes) if sizes else 0,
        noncontiguous_bundles=invariants.noncontiguous_bundles,
        solving_id_inversions=invariants.solving_id_inversions,
        timestamp_inversions=invariants.timestamp_inversions,
        train_rows=sum(row.interaction.split == "train" for row in rows),
        valid_rows=sum(row.interaction.split == "valid" for row in rows),
        test_rows=sum(row.interaction.split == "test" for row in rows),
    )


def reconstruct_dev5000_user_ids(
    train_user_ids: Iterable[str],
    valid_user_ids: Iterable[str],
    n_users: int = COHORT_SIZE,
) -> tuple[str, ...]:
    """Formal dev5000 cut from the historical TRAIN ∪ VALID population."""

    return select_first_sorted_users(
        list(train_user_ids) + list(valid_user_ids),
        n_users,
    )


def user_ids_from_parquet(path: str | Path, column: str = "user_id") -> tuple[str, ...]:
    """Read one string column. This does not score responses."""

    import pyarrow.parquet as pq

    table = pq.read_table(str(path), columns=[column])
    return tuple(str(value) for value in table.column(column).to_pylist())


def build_corrected_rows(
    raw_root: str | Path,
    questions: str | Path,
    user_ids: Sequence[str],
    *,
    catalog: Mapping[str, EdNetQuestionMetadata] | None = None,
) -> tuple[tuple[CorrectedEdNetRow, ...], tuple[EdNetFilterStats, ...]]:
    """Convert the named raw user files in cohort order and source order."""

    root = Path(raw_root)
    if catalog is None:
        catalog = {
            row.question_id: row
            for row in load_ednet_question_metadata(questions)
        }
    built: list[CorrectedEdNetRow] = []
    stats: list[EdNetFilterStats] = []
    for user_id in user_ids:
        path = root / f"{user_id}.csv"
        if not path.is_file():
            raise CorrectedEdNetError(f"missing raw KT1 file {path}")
        loaded = load_raw_user_kt1(path, user_id, catalog)
        stats.append(loaded.stats)
        solving = [
            int(row.bundle_id.split("|", 1)[1])
            for row in loaded.interactions
        ]
        source_rows = [
            int(row.interaction_id.rsplit(":", 1)[1])
            for row in loaded.interactions
        ]
        labeled = assign_solving_splits(loaded.interactions, solving)
        for interaction, source_row, solving_id in zip(labeled, source_rows, solving):
            built.append(
                CorrectedEdNetRow(
                    interaction=interaction,
                    source_row=source_row,
                    solving_id=solving_id,
                )
            )
    invariants = sequence_invariants(built)
    if invariants.noncontiguous_bundles != 0:
        raise CorrectedEdNetError(
            "non-contiguous bundles "
            f"{invariants.noncontiguous_bundles} != 0"
        )
    if invariants.solving_id_inversions != 0:
        raise CorrectedEdNetError(
            f"solving_id inversions {invariants.solving_id_inversions} != 0"
        )
    if invariants.timestamp_inversions != 0:
        raise CorrectedEdNetError(
            f"timestamp inversions {invariants.timestamp_inversions} != 0"
        )
    return tuple(built), tuple(stats)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vocabulary_record(
    written_rows: CorrectedVocabulary,
    publication_vocabulary: Mapping[str, object] | None,
    cohort_hash: str,
    learner_count: int,
) -> dict[str, object]:
    if publication_vocabulary is None:
        return written_rows.to_dict()
    return {
        "scope": "ednet_split_v1_train_valid_50k",
        "item_vocabulary_size": int(publication_vocabulary["item_count"]),
        "kc_vocabulary_size": int(publication_vocabulary["kc_count"]),
        "item_ids_sha256": publication_vocabulary["item_ids_sha256"],
        "kc_ids_sha256": publication_vocabulary["kc_ids_sha256"],
        "learner_vocabulary_size": learner_count,
        "learner_ids_sha256": cohort_hash,
    }


def write_corrected_dataset(
    rows: Sequence[CorrectedEdNetRow],
    stats: Sequence[EdNetFilterStats],
    output_dir: str | Path,
    *,
    cohort_user_ids: Sequence[str],
    questions_path: str | Path,
    raw_root: str | Path,
    publication_vocabulary: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Write gitignored parquet files plus the two manifests.

    ``publication_vocabulary`` is the 50k TRAIN+VALID vocabulary when
    the caller has built it. The dev5000 rows alone are not that
    vocabulary. When it is omitted, the manifest records the vocabulary
    of the written rows, which is the historical helper behavior for
    synthetic tests.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    diagnostics = diagnostics_from_build(rows, stats)
    vocabulary = vocabulary_from_rows(rows)
    cohort_hash = cohort_id_hash(tuple(cohort_user_ids))
    questions_hash = file_sha256(questions_path)
    _write_users(output / "dev5000_users.txt", cohort_user_ids)
    cohort_manifest = {
        "cohort_id": COHORT_ID,
        "preprocessing_version": PREPROCESSING_VERSION,
        "selection_rule": (
            "first 5000 user_ids in lexicographic order over the unique "
            "user_id values of ednet_split_v1 TRAIN union VALID"
        ),
        "learner_count": len(cohort_user_ids),
        "ordered_learner_ids_sha256": cohort_hash,
        "source_identity": {
            "raw_root": str(raw_root),
            "questions_path": str(questions_path),
            "questions_sha256": questions_hash,
            "legacy_split": "ednet_split_v1",
        },
        "hash_encoding": "sha256 of newline-joined ordered ids, no trailing newline",
        "user_id_list": "dev5000_users.txt",
        "redistribution": (
            "user ids stay in the local generated directory and are not "
            "part of the source package"
        ),
    }
    manifest = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "schema_version": SCHEMA_VERSION,
        "cohort_id": COHORT_ID,
        "split_id": SPLIT_ID,
        "ordered_learner_ids_sha256": cohort_hash,
        "questions_sha256": questions_hash,
        "row_counts": {
            "train": diagnostics.train_rows,
            "valid": diagnostics.valid_rows,
            "test": diagnostics.test_rows,
            "eligible": diagnostics.eligible_rows,
            "raw": diagnostics.raw_rows,
            "filtered": diagnostics.filtered_rows,
        },
        "vocabulary": _vocabulary_record(
            vocabulary,
            publication_vocabulary,
            cohort_hash,
            len(cohort_user_ids),
        ),
        "diagnostics": diagnostics.to_dict(),
        "files": {
            "train": "train.parquet",
            "valid": "valid.parquet",
            "test": "test.parquet",
            "cohort_manifest": "cohort_manifest.json",
            "users": "dev5000_users.txt",
            **(
                {"publication_vocab": "publication_vocab.json"}
                if publication_vocabulary is not None
                else {}
            ),
        },
    }
    _write_split(output / "train.parquet", rows, "train")
    _write_split(output / "valid.parquet", rows, "valid")
    _write_split(output / "test.parquet", rows, "test")
    manifest["files_sha256"] = {
        "train": file_sha256(output / "train.parquet"),
        "valid": file_sha256(output / "valid.parquet"),
        "test": file_sha256(output / "test.parquet"),
    }
    if publication_vocabulary is not None:
        _write_json(output / "publication_vocab.json", publication_vocabulary)
    _write_json(output / "cohort_manifest.json", cohort_manifest)
    _write_json(output / "manifest.json", manifest)
    return manifest


def _write_users(path: Path, user_ids: Sequence[str]) -> None:
    path.write_text("\n".join(user_ids) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_split(path: Path, rows: Sequence[CorrectedEdNetRow], split: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    chosen = [row for row in rows if row.interaction.split == split]
    schema = pa.schema(
        [
            ("interaction_id", pa.string()),
            ("learner_id", pa.string()),
            ("item_id", pa.string()),
            ("correct", pa.int64()),
            ("kc_ids", pa.list_(pa.int64())),
            ("order", pa.int64()),
            ("timestamp", pa.int64()),
            ("bundle_id", pa.string()),
            ("split", pa.string()),
            ("metric_mask", pa.bool_()),
            ("source_row", pa.int64()),
            ("solving_id", pa.int64()),
        ]
    )
    columns = {
        "interaction_id": [row.interaction.interaction_id for row in chosen],
        "learner_id": [row.interaction.learner_id for row in chosen],
        "item_id": [row.interaction.item_id for row in chosen],
        "correct": [row.interaction.correct for row in chosen],
        "kc_ids": [list(row.interaction.kc_ids) for row in chosen],
        "order": [row.interaction.order for row in chosen],
        "timestamp": [int(row.interaction.timestamp) for row in chosen],
        "bundle_id": [row.interaction.bundle_id for row in chosen],
        "split": [row.interaction.split for row in chosen],
        "metric_mask": [row.interaction.metric_mask for row in chosen],
        "source_row": [row.source_row for row in chosen],
        "solving_id": [row.solving_id for row in chosen],
    }
    if not chosen:
        columns = {
            field.name: pa.array([], type=field.type) for field in schema
        }
    table = pa.table(columns, schema=schema)
    pq.write_table(table, str(path))
