"""Public ASSISTments 2017 MAIN_V1 preparation.

The scientific decisions stay in
:func:`kclearner.data.adapters.assistments2017.load_assistments2017_main_v1`.
This module writes the canonical cache, logical identity, and provenance,
then publishes them through :mod:`kclearner.data.publish`.

``source_row`` is preparation provenance. It is not a field of
``dataset_logical_identity_v1``. ``solving_id`` and ``dev5000_users.txt``
are not ASSISTments artifacts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from kclearner.data.adapters.assistments2017 import (
    DATASET_VERSION,
    FROZEN_MAIN_V1_COUNTS,
    PROTOCOL_ID,
    AssistmentsMainV1,
    AssistmentsModelRow,
    MainV1Vocabulary,
    execution_group_ids,
    load_assistments2017_main_v1,
)
from kclearner.data.ednet_corrected import file_sha256
from kclearner.data.logical_identity import (
    DATASET_IDENTITY_VERSION,
    DatasetLogicalIdentity,
    SplitLogicalIdentity,
    canonical_dataset_json,
    dataset_logical_hash,
    hash_ordered_ids,
)
from kclearner.data.publish import (
    PublishError,
    discard_staging,
    make_staging_directory,
    publish_prepared_directory,
)
from kclearner.data.schema import Interaction

DATASET_ID = "assistments2017"
RAW_PROVENANCE_VERSION = "assistments_raw_source_provenance_v1"
COMPLETE_NAME = "prepare_status.json"
LOGICAL_NAME = "logical_identity.json"
SOURCE_NAME = "source_provenance.json"
VOCAB_NAME = "vocabulary.json"
LEARNER_NAME = "learner_ids.txt"
SPLIT_FILES = {
    "train": "train.parquet",
    "valid": "valid.parquet",
    "test": "test.parquet",
}


class AssistmentsPrepareError(RuntimeError):
    """Public ASSISTments preparation cannot finish under this protocol."""


@dataclass(frozen=True)
class AssistmentsPrepareReport:
    output_dir: Path
    reused: bool
    dataset_logical_hash: str
    csv_sha256: str
    cohort_hash: str
    item_ids_sha256: str
    kc_ids_sha256: str
    row_counts: dict[str, int]


def logical_identity_for(
    loaded: AssistmentsMainV1,
    protocol: Mapping[str, object],
) -> DatasetLogicalIdentity:
    """Build dataset identity from canonical rows and vocabulary only.

    Model name, Q/QC, hyperparameters, seed, and device are not read.
    """

    dataset = protocol["dataset"]
    if not isinstance(dataset, Mapping):
        raise AssistmentsPrepareError("protocol dataset block is missing")
    cohort_hash, item_hash, kc_hash = vocabulary_hashes(loaded.vocabulary)
    splits: list[SplitLogicalIdentity] = []
    for split in ("train", "valid", "test"):
        chosen = tuple(row for row in loaded.interactions if row.split == split)
        splits.append(
            SplitLogicalIdentity(split, _split_hash(chosen), len(chosen))
        )
    return DatasetLogicalIdentity(
        dataset_id=str(dataset["dataset_id"]),
        protocol_id=str(protocol["protocol_id"]),
        preprocessing_version=str(dataset["preprocessing_version"]),
        cohort_hash=cohort_hash,
        splits=tuple(splits),
        item_vocabulary_hash=item_hash,
        kc_vocabulary_hash=kc_hash,
    )


def vocabulary_hashes(
    vocabulary: MainV1Vocabulary,
) -> tuple[str, str, str]:
    """Cohort, item, and KC hashes in the Part 2 ordered-id encoding."""

    return (
        hash_ordered_ids(vocabulary.students_sorted),
        hash_ordered_ids(vocabulary.problems_sorted),
        hash_ordered_ids(vocabulary.skills_sorted),
    )


def assess_assistments_output(
    output_dir: str | Path,
    *,
    protocol_id: str,
    csv_sha256: str,
    dataset_logical_hash_value: str | None = None,
) -> str:
    """Return ``missing``, ``reuse``, or ``incompatible``."""

    output = Path(output_dir)
    if not output.exists():
        return "missing"
    required = (
        COMPLETE_NAME,
        LOGICAL_NAME,
        SOURCE_NAME,
        VOCAB_NAME,
        LEARNER_NAME,
        *SPLIT_FILES.values(),
        "manifest.json",
    )
    if any(not (output / name).is_file() for name in required):
        return "incompatible"
    status = _read_json(output / COMPLETE_NAME)
    source = _read_json(output / SOURCE_NAME)
    logical = _read_json(output / LOGICAL_NAME)
    if status.get("status") != "complete" or status.get("protocol_id") != protocol_id:
        return "incompatible"
    primary = source.get("primary_csv") or {}
    if primary.get("sha256") != csv_sha256:
        return "incompatible"
    if dataset_logical_hash_value is not None and (
        logical.get("dataset_logical_hash") != dataset_logical_hash_value
    ):
        return "incompatible"
    identity = logical.get("dataset_logical_identity") or {}
    if identity.get("protocol_id") != protocol_id:
        return "incompatible"
    return "reuse"


def prepare_assistments2017(
    raw_file: str | Path,
    output_dir: str | Path,
    protocol: Mapping[str, object],
    *,
    rebuild: bool = False,
) -> AssistmentsPrepareReport:
    """Prepare ``primary.csv`` into ``output_dir``.

    The formal protocol fails closed when the frozen MAIN_V1 counts do
    not match. A protocol with ``enforce_frozen_counts`` false keeps the
    same loader and is the synthetic path.
    """

    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise AssistmentsPrepareError(
            f"protocol_id {protocol.get('protocol_id')!r} != {PROTOCOL_ID}"
        )
    dataset = protocol.get("dataset")
    if not isinstance(dataset, Mapping):
        raise AssistmentsPrepareError("protocol dataset block is missing")
    if str(dataset.get("dataset_id")) != DATASET_ID:
        raise AssistmentsPrepareError(
            f"dataset_id {dataset.get('dataset_id')!r} != {DATASET_ID}"
        )
    if str(dataset.get("preprocessing_version")) != DATASET_VERSION:
        raise AssistmentsPrepareError(
            "preprocessing_version "
            f"{dataset.get('preprocessing_version')!r} != {DATASET_VERSION}"
        )
    source = Path(raw_file)
    if not source.is_file():
        raise AssistmentsPrepareError(f"primary CSV does not exist: {source}")
    destination = Path(output_dir)
    csv_hash = file_sha256(source)
    if not rebuild and destination.exists():
        decision = assess_assistments_output(
            destination,
            protocol_id=PROTOCOL_ID,
            csv_sha256=csv_hash,
        )
        if decision == "reuse":
            return _report_from_existing(destination, csv_hash)
        if decision == "incompatible":
            raise AssistmentsPrepareError(
                f"{destination} exists but does not match this protocol and "
                "CSV. Refusing to overwrite it. Pass --rebuild to replace it."
            )
    loaded = load_assistments2017_main_v1(source)
    if len(loaded.source_indexes) != len(loaded.interactions):
        raise AssistmentsPrepareError("source_row provenance is not aligned")
    _enforce_formal_counts(protocol, loaded)
    _enforce_recorded_hashes(protocol, loaded)
    identity = logical_identity_for(loaded, protocol)
    logical_hash = dataset_logical_hash(identity)
    staging = make_staging_directory(destination)
    try:
        _write_output(staging, loaded, protocol, identity, logical_hash, source, csv_hash)
    except Exception:
        discard_staging(staging)
        raise

    def assess() -> str:
        return assess_assistments_output(
            destination,
            protocol_id=PROTOCOL_ID,
            csv_sha256=csv_hash,
            dataset_logical_hash_value=logical_hash,
        )

    try:
        outcome = publish_prepared_directory(
            staging,
            destination,
            assess,
            rebuild=rebuild,
        )
    except PublishError:
        discard_staging(staging)
        raise
    cohort_hash, item_hash, kc_hash = vocabulary_hashes(loaded.vocabulary)
    counts = _row_counts(loaded)
    return AssistmentsPrepareReport(
        output_dir=destination,
        reused=outcome == "reused",
        dataset_logical_hash=logical_hash,
        csv_sha256=csv_hash,
        cohort_hash=cohort_hash,
        item_ids_sha256=item_hash,
        kc_ids_sha256=kc_hash,
        row_counts=counts,
    )


def read_model_rows(
    dataset_dir: str | Path,
    split: str,
) -> tuple[AssistmentsModelRow, ...]:
    """Read one prepared split into dense model rows.

    The parquet schema has ``group_id`` and ``source_row``. It does not
    have an EdNet ``solving_id`` column.
    """

    if split not in SPLIT_FILES:
        raise AssistmentsPrepareError(f"unknown split {split!r}")
    root = Path(dataset_dir)
    vocabulary = _vocabulary_from_disk(root)
    import pyarrow.parquet as pq

    path = root / SPLIT_FILES[split]
    table = pq.read_table(path)
    names = set(table.schema.names)
    if "solving_id" in names:
        raise AssistmentsPrepareError(
            f"{path.name} contains EdNet solving_id; ASSISTments does not"
        )
    required = {
        "interaction_id",
        "learner_id",
        "item_id",
        "correct",
        "kc_ids",
        "timestamp",
        "split",
        "metric_mask",
        "bundle_id",
        "group_id",
        "source_row",
    }
    missing = sorted(required - names)
    if missing:
        raise AssistmentsPrepareError(f"{path.name} is missing {missing}")
    data = table.to_pydict()
    rows: list[AssistmentsModelRow] = []
    bundles: list[str] = []
    for index in range(table.num_rows):
        if str(data["split"][index]) != split:
            raise AssistmentsPrepareError(f"{path.name} contains another split")
        learner_id = str(data["learner_id"][index])
        item_id = str(data["item_id"][index])
        kc_ids = tuple(int(kc) for kc in data["kc_ids"][index])
        in_vocab = (
            vocabulary.has_student(learner_id)
            and vocabulary.has_problem(item_id)
            and len(kc_ids) == 1
            and 0 <= kc_ids[0] < len(vocabulary.skills_sorted)
        )
        if split in ("train", "valid") and not in_vocab:
            raise AssistmentsPrepareError(
                f"{data['interaction_id'][index]} fell outside the TRAIN+VALID vocabulary"
            )
        if in_vocab:
            student_idx = vocabulary.student_index(learner_id)
            problem_idx = vocabulary.problem_index(item_id)
            skill_idx = kc_ids[0]
            skill_id = vocabulary.skills_sorted[skill_idx]
            oov_problem = False
        else:
            student_idx = (
                vocabulary.student_index(learner_id)
                if vocabulary.has_student(learner_id)
                else -1
            )
            if vocabulary.has_problem(item_id):
                problem_idx = vocabulary.problem_index(item_id)
                oov_problem = False
            else:
                problem_idx = 0
                oov_problem = True
            if kc_ids and 0 <= kc_ids[0] < len(vocabulary.skills_sorted):
                skill_idx = kc_ids[0]
                skill_id = vocabulary.skills_sorted[skill_idx]
            else:
                skill_idx = -1
                skill_id = ""
        group_value = data["group_id"][index]
        if split in ("train", "valid") and group_value is None:
            raise AssistmentsPrepareError(
                f"{data['interaction_id'][index]} is missing group_id"
            )
        group_id = -1 if group_value is None else int(group_value)
        bundle = data["bundle_id"][index]
        if not isinstance(bundle, str) or bundle == "":
            raise AssistmentsPrepareError(
                f"{data['interaction_id'][index]} is missing bundle_id"
            )
        bundles.append(bundle)
        rows.append(
            AssistmentsModelRow(
                student_idx=student_idx,
                problem_idx=problem_idx,
                skill_idx=skill_idx,
                group_id=group_id,
                correct=int(data["correct"][index]),
                timestamp=int(data["timestamp"][index]),
                split=split,
                metric_mask=bool(data["metric_mask"][index]),
                oov_problem=oov_problem,
                interaction_id=str(data["interaction_id"][index]),
                learner_id=learner_id,
                item_id=item_id,
                skill_id=skill_id,
            )
        )
    if any(row.group_id < 0 for row in rows):
        filled = execution_group_ids(
            bundles,
            [None if row.group_id < 0 else row.group_id for row in rows],
        )
        rows = [
            AssistmentsModelRow(
                student_idx=row.student_idx,
                problem_idx=row.problem_idx,
                skill_idx=row.skill_idx,
                group_id=group_id,
                correct=row.correct,
                timestamp=row.timestamp,
                split=row.split,
                metric_mask=row.metric_mask,
                oov_problem=row.oov_problem,
                interaction_id=row.interaction_id,
                learner_id=row.learner_id,
                item_id=row.item_id,
                skill_id=row.skill_id,
            )
            for row, group_id in zip(rows, filled)
        ]
    return tuple(rows)


def _enforce_formal_counts(
    protocol: Mapping[str, object],
    loaded: AssistmentsMainV1,
) -> None:
    dataset = protocol["dataset"]
    if not isinstance(dataset, Mapping) or not dataset.get("enforce_frozen_counts"):
        return
    frozen = FROZEN_MAIN_V1_COUNTS
    diagnostics = loaded.diagnostics
    row_counts = dataset.get("row_counts")
    if not isinstance(row_counts, Mapping):
        raise AssistmentsPrepareError("formal protocol is missing row_counts")
    compared = (
        ("train", "train_rows", diagnostics.train_rows),
        ("valid", "valid_rows", diagnostics.valid_rows),
        ("test", "test_rows", diagnostics.test_rows),
        ("eligible", "eligible_rows", diagnostics.eligible_rows),
    )
    for name, frozen_name, actual in compared:
        if int(row_counts[name]) != int(frozen[frozen_name]):
            raise AssistmentsPrepareError(
                f"protocol {name} count does not match the frozen adapter constant"
            )
        if int(actual) != int(frozen[frozen_name]):
            raise AssistmentsPrepareError(
                f"{name} row count {actual} != {frozen[frozen_name]}"
            )
    scalars = (
        ("learners", "n_students_vocab", diagnostics.n_students_vocab),
        ("train_valid_problems", "n_questions_vocab", diagnostics.n_questions_vocab),
        ("kcs", "n_skills_vocab", diagnostics.n_skills_vocab),
        ("nominal_test_item_oov", "oov_rows", diagnostics.oov_rows),
    )
    for name, frozen_name, actual in scalars:
        if int(dataset[name]) != int(frozen[frozen_name]):
            raise AssistmentsPrepareError(
                f"protocol {name} does not match the frozen adapter constant"
            )
        if int(actual) != int(frozen[frozen_name]):
            raise AssistmentsPrepareError(f"{name} {actual} != {frozen[frozen_name]}")
    expected_test_metric = int(frozen["test_rows"]) - int(frozen["oov_rows"])
    if int(dataset["metric_eligible_test"]) != expected_test_metric:
        raise AssistmentsPrepareError(
            "protocol metric_eligible_test does not match frozen TEST rows minus OOV rows"
        )
    test_metric = sum(
        1 for row in loaded.interactions if row.split == "test" and row.metric_mask
    )
    if test_metric != expected_test_metric:
        raise AssistmentsPrepareError(
            f"metric-eligible TEST rows {test_metric} != {expected_test_metric}"
        )
    for name in (
        "metric_eligible_rows",
        "n_train_users",
        "n_valid_users",
        "n_questions_full",
        "n_skills_full",
        "promoted_to_valid",
        "nominal_cross_boundary_bundles",
        "train_bundles",
        "valid_bundles",
    ):
        actual = getattr(diagnostics, name)
        if int(actual) != int(frozen[name]):
            raise AssistmentsPrepareError(f"{name} {actual} != {frozen[name]}")


def _enforce_recorded_hashes(
    protocol: Mapping[str, object],
    loaded: AssistmentsMainV1,
) -> None:
    vocabulary = protocol.get("vocabulary")
    if not isinstance(vocabulary, Mapping):
        return
    cohort_hash, item_hash, kc_hash = vocabulary_hashes(loaded.vocabulary)
    expected = (
        ("learner_ids_sha256", cohort_hash),
        ("item_ids_sha256", item_hash),
        ("kc_ids_sha256", kc_hash),
    )
    for name, actual in expected:
        recorded = vocabulary.get(name)
        if recorded is not None and str(recorded) != actual:
            raise AssistmentsPrepareError(
                f"{name} {actual} does not match the protocol"
            )


def _write_output(
    staging: Path,
    loaded: AssistmentsMainV1,
    protocol: Mapping[str, object],
    identity: DatasetLogicalIdentity,
    logical_hash: str,
    source: Path,
    csv_hash: str,
) -> None:
    _require_pyarrow()
    cohort_hash, item_hash, kc_hash = vocabulary_hashes(loaded.vocabulary)
    vocabulary = loaded.vocabulary
    group_of = _group_by_interaction(loaded)
    for split, filename in SPLIT_FILES.items():
        _write_split(staging / filename, loaded, group_of, split)
    vocab_document = {
        "vocab_id": "assist17_train_valid_problem_skill_v1",
        "source_partitions": ["train", "valid"],
        "test_included": False,
        "item_count": len(vocabulary.problems_sorted),
        "kc_count": len(vocabulary.skills_sorted),
        "learner_count": len(vocabulary.students_sorted),
        "item_ids_sha256": item_hash,
        "kc_ids_sha256": kc_hash,
        "learner_ids_sha256": cohort_hash,
        "item_ids": list(vocabulary.problems_sorted),
        "kc_ids": list(vocabulary.skills_sorted),
        "learner_ids": list(vocabulary.students_sorted),
    }
    _write_json(staging / VOCAB_NAME, vocab_document)
    (staging / LEARNER_NAME).write_text(
        "\n".join(vocabulary.students_sorted) + ("\n" if vocabulary.students_sorted else ""),
        encoding="utf-8",
    )
    logical_document = {
        "dataset_logical_identity": json.loads(canonical_dataset_json(identity)),
        "dataset_logical_hash": logical_hash,
        "identity_version": DATASET_IDENTITY_VERSION,
    }
    _write_json(staging / LOGICAL_NAME, logical_document)
    _write_json(
        staging / SOURCE_NAME,
        {
            "version": RAW_PROVENANCE_VERSION,
            "primary_csv": {
                "sha256": csv_hash,
                "filename": source.name,
            },
        },
    )
    counts = _row_counts(loaded)
    manifest = {
        "dataset_id": DATASET_ID,
        "protocol_id": PROTOCOL_ID,
        "preprocessing_version": DATASET_VERSION,
        "cohort_id": "assistments2017_main_v1_train_valid_learners",
        "ordered_learner_ids_sha256": cohort_hash,
        "row_counts": counts,
        "dataset_logical_identity": logical_document["dataset_logical_identity"],
        "dataset_logical_hash": logical_hash,
        "files_sha256": {
            split: file_sha256(staging / filename)
            for split, filename in SPLIT_FILES.items()
        },
    }
    _write_json(staging / "manifest.json", manifest)
    _write_json(
        staging / COMPLETE_NAME,
        {
            "status": "complete",
            "protocol_id": PROTOCOL_ID,
            "dataset_id": DATASET_ID,
            "preprocessing_version": DATASET_VERSION,
        },
    )


def _write_split(
    path: Path,
    loaded: AssistmentsMainV1,
    group_of: Mapping[str, int],
    split: str,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    chosen = [
        (interaction, source_row)
        for interaction, source_row in zip(loaded.interactions, loaded.source_indexes)
        if interaction.split == split
    ]
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
            ("group_id", pa.int64()),
        ]
    )
    columns = {
        "interaction_id": [row.interaction_id for row, _source in chosen],
        "learner_id": [row.learner_id for row, _source in chosen],
        "item_id": [row.item_id for row, _source in chosen],
        "correct": [row.correct for row, _source in chosen],
        "kc_ids": [list(row.kc_ids) for row, _source in chosen],
        "order": [row.order for row, _source in chosen],
        "timestamp": [int(row.timestamp) for row, _source in chosen],
        "bundle_id": [row.bundle_id for row, _source in chosen],
        "split": [row.split for row, _source in chosen],
        "metric_mask": [row.metric_mask for row, _source in chosen],
        "source_row": [source for _row, source in chosen],
        "group_id": list(
            execution_group_ids(
                [row.bundle_id for row, _source in chosen],
                [group_of.get(row.interaction_id) for row, _source in chosen],
            )
        ),
    }
    if not chosen:
        columns = {field.name: pa.array([], type=field.type) for field in schema}
    table = pa.table(columns, schema=schema)
    pq.write_table(table, str(path))


def _group_by_interaction(loaded: AssistmentsMainV1) -> dict[str, int]:
    grouped: dict[str, int] = {}
    for row in (*loaded.train_rows, *loaded.valid_rows):
        grouped[row.interaction_id] = row.group_id
    return grouped


def _row_counts(loaded: AssistmentsMainV1) -> dict[str, int]:
    diagnostics = loaded.diagnostics
    return {
        "train": diagnostics.train_rows,
        "valid": diagnostics.valid_rows,
        "test": diagnostics.test_rows,
        "eligible": diagnostics.eligible_rows,
    }


def _report_from_existing(destination: Path, csv_hash: str) -> AssistmentsPrepareReport:
    logical = _read_json(destination / LOGICAL_NAME)
    identity = logical["dataset_logical_identity"]
    counts = {
        split["split_id"]: int(split["row_count"])
        for split in identity["splits"]
    }
    counts["eligible"] = sum(counts.values())
    return AssistmentsPrepareReport(
        output_dir=destination,
        reused=True,
        dataset_logical_hash=str(logical["dataset_logical_hash"]),
        csv_sha256=csv_hash,
        cohort_hash=str(identity["cohort_hash"]),
        item_ids_sha256=str(identity["item_vocabulary_hash"]),
        kc_ids_sha256=str(identity["kc_vocabulary_hash"]),
        row_counts=counts,
    )


def _vocabulary_from_disk(root: Path) -> MainV1Vocabulary:
    payload = _read_json(root / VOCAB_NAME)
    return MainV1Vocabulary(
        students_sorted=tuple(str(item) for item in payload["learner_ids"]),
        problems_sorted=tuple(str(item) for item in payload["item_ids"]),
        skills_sorted=tuple(str(item) for item in payload["kc_ids"]),
    )


def _split_hash(rows: Sequence[Interaction]) -> str:
    from kclearner.data.logical_identity import logical_split_hash

    return logical_split_hash(rows)


def _require_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise AssistmentsPrepareError(
            'Parquet preparation requires pyarrow. Install it with '
            'pip install -e ".[parquet]".'
        ) from exc


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
