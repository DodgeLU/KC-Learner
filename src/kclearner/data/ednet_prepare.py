"""Public EdNet-KT1 corrected preparation.

Reads provider KT1 files and ``questions.csv`` only. It does not open
``project_c`` or a historical split Parquet. The 50k cohort comes from
``ednet_pilot50k_sampler_v1``. Publication items and KCs are the frozen
50k TRAIN+VALID population, not the dev5000 subset.

``source_row`` and ``solving_id`` stay off the generic interaction
schema. ``ednet_source_provenance_hash_v1`` hashes them beside the
generic logical identity.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from kclearner.data.adapters.ednet_kt1 import load_raw_user_kt1
from kclearner.data.adapters.ednet_metadata import load_ednet_question_metadata
from kclearner.data.ednet_corrected import (
    COHORT_ID,
    CorrectedEdNetError,
    CorrectedEdNetRow,
    EdNetFilterStats,
    PREPROCESSING_VERSION,
    SPLIT_ID,
    cohort_id_hash,
    diagnostics_from_build,
    file_sha256,
    sequence_invariants,
    write_corrected_dataset,
)
from kclearner.data.ednet_corrected import assign_solving_splits
from kclearner.data.ednet_pilot_sample import (
    HISTORICAL_SAMPLE_SHA256,
    SAMPLER_ID,
    dev5000_from_pilot,
    select_ednet_pilot50k,
)
from kclearner.data.logical_identity import (
    DATASET_IDENTITY_VERSION,
    DatasetLogicalIdentity,
    SplitLogicalIdentity,
    canonical_dataset_json,
    dataset_logical_hash,
    hash_ordered_ids,
    logical_split_hash,
)
from kclearner.data.publish import (
    discard_staging,
    make_staging_directory,
    publish_prepared_directory,
)
from kclearner.data.schema import Interaction

DATASET_ID = "ednet_kt1"
PROTOCOL_ID = "ednet_corrected_publication_v1"
PROVENANCE_VERSION = "ednet_source_provenance_hash_v1"
RAW_PROVENANCE_VERSION = "ednet_raw_source_provenance_v1"
FILE_AGGREGATE_VERSION = "user_id_tab_file_sha256_lines_v1"
POPULATION_HASH_VERSION = "ordered_user_id_sha256_v1"
COMPLETE_NAME = "prepare_status.json"
LOGICAL_NAME = "logical_identity.json"
SOURCE_NAME = "source_provenance.json"
BLANK_ANSWER_ROWS = 222


class PrepareError(RuntimeError):
    """Public EdNet preparation cannot finish under the frozen protocol."""


@dataclass(frozen=True)
class PrepareReport:
    output_dir: Path
    reused: bool
    dataset_logical_hash: str
    pilot_sha256: str
    dev5000_sha256: str
    item_ids_sha256: str
    kc_ids_sha256: str
    row_counts: dict[str, int]


def list_kt1_user_ids(raw_dir: str | Path) -> tuple[str, ...]:
    """Sorted stems of ``u*.csv`` files. Directory order is ignored."""

    root = Path(raw_dir)
    if not root.is_dir():
        raise PrepareError(f"KT1 directory does not exist: {root}")
    names: list[str] = []
    with os.scandir(root) as entries:
        for entry in entries:
            filename = entry.name
            if filename.startswith("u") and filename.endswith(".csv"):
                names.append(filename[:-4])
    names.sort()
    if not names:
        raise PrepareError(f"KT1 directory has no user CSV files: {root}")
    return tuple(names)


def publication_ids_from_rows(
    rows: Sequence[tuple[str, str, tuple[int, ...]]],
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Sorted TRAIN+VALID item ids and KC ids.

    Each tuple is ``(split, item_id, kc_ids)``. TEST rows do not enter
    either set. A ``-1`` KC is a protocol failure, not a vocabulary entry.
    """

    items: set[str] = set()
    kcs: set[int] = set()
    for split, item_id, kc_ids in rows:
        if split not in ("train", "valid"):
            continue
        items.add(item_id)
        for kc in kc_ids:
            if kc < 0:
                raise PrepareError(f"publication vocabulary received KC {kc}")
            kcs.add(kc)
    return tuple(sorted(items)), tuple(sorted(kcs))


def provenance_hash(rows: Sequence[CorrectedEdNetRow]) -> str:
    """SHA-256 of EdNet source fields in the given row order.

    Each line is
    ``learner_id,source_row,item_id,solving_id,timestamp,split``
    with no spaces, followed by LF. This is
    ``ednet_source_provenance_hash_v1``.
    """

    digest = hashlib.sha256()
    for row in rows:
        timestamp = row.interaction.timestamp
        if timestamp is None:
            raise PrepareError(f"{row.interaction.interaction_id} has no timestamp")
        line = (
            f"{row.interaction.learner_id},{row.source_row},"
            f"{row.interaction.item_id},{row.solving_id},"
            f"{int(timestamp)},{row.interaction.split}\n"
        )
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def file_aggregate_hash(entries: Sequence[tuple[str, str]]) -> str:
    """SHA-256 of ``user_id<TAB>file_sha256`` lines in the given order."""

    digest = hashlib.sha256()
    for user_id, file_hash in entries:
        digest.update(f"{user_id}\t{file_hash}\n".encode("utf-8"))
    return digest.hexdigest()


def assess_existing_output(
    output_dir: str | Path,
    *,
    protocol_id: str,
    questions_sha256: str,
    population_sha256: str,
    pilot_file_sha256: str,
    cohort_hash: str,
    item_vocabulary_hash: str,
    kc_vocabulary_hash: str,
    dataset_logical_hash_value: str | None = None,
) -> str:
    """Return ``missing``, ``reuse``, or ``incompatible``.

    ``dataset_logical_hash_value`` is optional. When it is omitted, reuse
    requires a complete status and matching raw-source provenance only.
    Callers that already know the logical hash should pass it.
    """

    output = Path(output_dir)
    if not output.exists():
        return "missing"
    status_path = output / COMPLETE_NAME
    source_path = output / SOURCE_NAME
    logical_path = output / LOGICAL_NAME
    if not status_path.is_file() or not source_path.is_file() or not logical_path.is_file():
        return "incompatible"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    source = json.loads(source_path.read_text(encoding="utf-8"))
    logical = json.loads(logical_path.read_text(encoding="utf-8"))
    if status.get("status") != "complete" or status.get("protocol_id") != protocol_id:
        return "incompatible"
    questions = source.get("questions") or {}
    population = source.get("population") or {}
    files = source.get("selected_pilot_files") or {}
    if questions.get("sha256") != questions_sha256:
        return "incompatible"
    if population.get("ordered_ids_sha256") != population_sha256:
        return "incompatible"
    if files.get("aggregate_sha256") != pilot_file_sha256:
        return "incompatible"
    if files.get("cohort_sha256") != HISTORICAL_SAMPLE_SHA256:
        return "incompatible"
    identity = logical.get("dataset_logical_identity") or {}
    if identity.get("cohort_hash") != cohort_hash:
        return "incompatible"
    if identity.get("item_vocabulary_hash") != item_vocabulary_hash:
        return "incompatible"
    if identity.get("kc_vocabulary_hash") != kc_vocabulary_hash:
        return "incompatible"
    if dataset_logical_hash_value is not None and (
        logical.get("dataset_logical_hash") != dataset_logical_hash_value
    ):
        return "incompatible"
    required = (
        "train.parquet",
        "valid.parquet",
        "test.parquet",
        "manifest.json",
        "cohort_manifest.json",
        "dev5000_users.txt",
        "publication_vocab.json",
    )
    if any(not (output / name).is_file() for name in required):
        return "incompatible"
    return "reuse"


def prepare_ednet_kt1(
    raw_dir: str | Path,
    questions: str | Path,
    output_dir: str | Path,
    protocol: Mapping[str, object],
    *,
    rebuild: bool = False,
) -> PrepareReport:
    """Build the corrected dev5000 directory from provider files."""

    _require_pyarrow()
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise PrepareError(
            f"protocol_id {protocol.get('protocol_id')!r} != {PROTOCOL_ID}"
        )
    raw_root = Path(raw_dir)
    questions_path = Path(questions)
    if not questions_path.is_file():
        raise PrepareError(f"questions CSV does not exist: {questions_path}")
    destination = Path(output_dir)
    population = list_kt1_user_ids(raw_root)
    pilot = select_ednet_pilot50k(population)
    cohort = dev5000_from_pilot(pilot)
    questions_hash = file_sha256(questions_path)
    population_hash = hash_ordered_ids(population)
    if not rebuild and destination.exists():
        decision = _reuse_or_reject(
            destination,
            questions_hash=questions_hash,
            population_hash=population_hash,
            pilot=pilot,
            cohort=cohort,
            protocol=protocol,
            raw_root=raw_root,
        )
        if decision is not None:
            return decision
    if destination.exists() and not rebuild:
        raise PrepareError(
            f"{destination} exists but does not match this protocol and raw "
            "input. Refusing to overwrite it. Pass --rebuild to replace it."
        )
    staging = make_staging_directory(destination)
    try:
        report = _build_into(
            staging,
            raw_root,
            questions_path,
            protocol,
            population,
            pilot,
            cohort,
            questions_hash,
            population_hash,
        )
    except Exception:
        discard_staging(staging)
        raise
    source_document = json.loads((staging / SOURCE_NAME).read_text(encoding="utf-8"))
    pilot_files = source_document["selected_pilot_files"]

    def assess() -> str:
        return assess_existing_output(
            destination,
            protocol_id=PROTOCOL_ID,
            questions_sha256=questions_hash,
            population_sha256=population_hash,
            pilot_file_sha256=str(pilot_files["aggregate_sha256"]),
            cohort_hash=cohort_id_hash(tuple(cohort)),
            item_vocabulary_hash=str(protocol["vocabulary"]["item_ids_sha256"]),
            kc_vocabulary_hash=str(protocol["vocabulary"]["kc_ids_sha256"]),
            dataset_logical_hash_value=report.dataset_logical_hash,
        )

    try:
        outcome = publish_prepared_directory(
            staging,
            destination,
            assess,
            rebuild=rebuild,
        )
    except Exception:
        discard_staging(staging)
        raise
    return PrepareReport(
        output_dir=destination,
        reused=outcome == "reused",
        dataset_logical_hash=report.dataset_logical_hash,
        pilot_sha256=report.pilot_sha256,
        dev5000_sha256=report.dev5000_sha256,
        item_ids_sha256=report.item_ids_sha256,
        kc_ids_sha256=report.kc_ids_sha256,
        row_counts=report.row_counts,
    )


def _require_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise PrepareError(
            'Parquet preparation requires pyarrow. Install it with '
            'pip install -e ".[parquet]".'
        ) from exc


def _reuse_or_reject(
    destination: Path,
    *,
    questions_hash: str,
    population_hash: str,
    pilot: Sequence[str],
    cohort: Sequence[str],
    protocol: Mapping[str, object],
    raw_root: Path,
) -> PrepareReport | None:
    entries = tuple(
        (user_id, file_sha256(raw_root / f"{user_id}.csv"))
        for user_id in pilot
    )
    aggregate = file_aggregate_hash(entries)
    vocabulary = protocol["vocabulary"]
    decision = assess_existing_output(
        destination,
        protocol_id=PROTOCOL_ID,
        questions_sha256=questions_hash,
        population_sha256=population_hash,
        pilot_file_sha256=aggregate,
        cohort_hash=cohort_id_hash(tuple(cohort)),
        item_vocabulary_hash=str(vocabulary["item_ids_sha256"]),
        kc_vocabulary_hash=str(vocabulary["kc_ids_sha256"]),
    )
    if decision == "missing":
        return None
    if decision != "reuse":
        return None
    logical = json.loads((destination / LOGICAL_NAME).read_text(encoding="utf-8"))
    identity = logical["dataset_logical_identity"]
    counts = {
        split["split_id"]: int(split["row_count"])
        for split in identity["splits"]
    }
    counts["eligible"] = sum(counts.values())
    return PrepareReport(
        output_dir=destination,
        reused=True,
        dataset_logical_hash=str(logical["dataset_logical_hash"]),
        pilot_sha256=HISTORICAL_SAMPLE_SHA256,
        dev5000_sha256=str(identity["cohort_hash"]),
        item_ids_sha256=str(identity["item_vocabulary_hash"]),
        kc_ids_sha256=str(identity["kc_vocabulary_hash"]),
        row_counts=counts,
    )


def _build_into(
    staging: Path,
    raw_root: Path,
    questions_path: Path,
    protocol: Mapping[str, object],
    population: Sequence[str],
    pilot: tuple[str, ...],
    cohort: tuple[str, ...],
    questions_hash: str,
    population_hash: str,
) -> PrepareReport:
    catalog = {
        row.question_id: row
        for row in load_ednet_question_metadata(questions_path)
    }
    dev_members = set(cohort)
    kept: dict[str, tuple[CorrectedEdNetRow, ...]] = {}
    kept_stats: dict[str, EdNetFilterStats] = {}
    items: set[str] = set()
    kcs: set[int] = set()
    file_entries: list[tuple[str, str]] = []
    for index, user_id in enumerate(pilot, start=1):
        path = raw_root / f"{user_id}.csv"
        if not path.is_file():
            raise PrepareError(f"missing raw KT1 file {path}")
        file_entries.append((user_id, file_sha256(path)))
        loaded = load_raw_user_kt1(path, user_id, catalog)
        labeled, solving, source_rows = _label_user(loaded.interactions)
        for interaction, solving_id in zip(labeled, solving):
            if interaction.split not in ("train", "valid"):
                continue
            items.add(interaction.item_id)
            for kc in interaction.kc_ids:
                if kc < 0:
                    raise PrepareError(f"{interaction.interaction_id} has KC {kc}")
                kcs.add(int(kc))
        if user_id in dev_members:
            kept[user_id] = tuple(
                CorrectedEdNetRow(
                    interaction=interaction,
                    source_row=source_row,
                    solving_id=solving_id,
                )
                for interaction, source_row, solving_id in zip(
                    labeled, source_rows, solving
                )
            )
            kept_stats[user_id] = loaded.stats
        if index % 2000 == 0:
            print(f"[prepare] scanned {index} / {len(pilot)} pilot users", flush=True)
    missing = [user_id for user_id in cohort if user_id not in kept]
    if missing:
        raise PrepareError(f"dev5000 users missing from the pilot scan: {len(missing)}")
    rows: list[CorrectedEdNetRow] = []
    stats: list[EdNetFilterStats] = []
    for user_id in cohort:
        rows.extend(kept[user_id])
        stats.append(kept_stats[user_id])
    ordered = tuple(rows)
    invariants = sequence_invariants(ordered)
    if invariants.noncontiguous_bundles != 0:
        raise CorrectedEdNetError(
            f"non-contiguous bundles {invariants.noncontiguous_bundles} != 0"
        )
    if invariants.solving_id_inversions != 0:
        raise CorrectedEdNetError(
            f"solving_id inversions {invariants.solving_id_inversions} != 0"
        )
    if invariants.timestamp_inversions != 0:
        raise CorrectedEdNetError(
            f"timestamp inversions {invariants.timestamp_inversions} != 0"
        )
    item_tuple = tuple(sorted(items))
    kc_tuple = tuple(sorted(kcs))
    oov_rows, oov_types = _oov_count(ordered, set(item_tuple))
    publication = _publication_document(item_tuple, kc_tuple, protocol, oov_rows, oov_types)
    diagnostics = diagnostics_from_build(ordered, tuple(stats))
    _check_counts(diagnostics, protocol)
    identity = _dataset_identity(ordered, publication, protocol)
    logical_hash = dataset_logical_hash(identity)
    provenance = _provenance_document(ordered)
    aggregate = file_aggregate_hash(tuple(file_entries))
    write_corrected_dataset(
        ordered,
        tuple(stats),
        staging,
        cohort_user_ids=cohort,
        questions_path=questions_path,
        raw_root=raw_root,
        publication_vocabulary=publication,
    )
    logical_document = {
        "dataset_logical_identity": json.loads(canonical_dataset_json(identity)),
        "dataset_logical_hash": logical_hash,
        "identity_version": DATASET_IDENTITY_VERSION,
        "ednet_source_provenance": provenance,
    }
    _write(staging / LOGICAL_NAME, logical_document)
    manifest_path = staging / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset_logical_identity"] = logical_document["dataset_logical_identity"]
    manifest["dataset_logical_hash"] = logical_hash
    _write(manifest_path, manifest)
    _write(
        staging / SOURCE_NAME,
        {
            "version": RAW_PROVENANCE_VERSION,
            "population": {
                "version": POPULATION_HASH_VERSION,
                "count": len(population),
                "ordered_ids_sha256": population_hash,
            },
            "selected_pilot_files": {
                "version": FILE_AGGREGATE_VERSION,
                "count": len(pilot),
                "aggregate_sha256": aggregate,
                "cohort_sha256": HISTORICAL_SAMPLE_SHA256,
                "sampler_id": SAMPLER_ID,
            },
            "questions": {
                "sha256": questions_hash,
                "filename": questions_path.name,
            },
        },
    )
    _write(
        staging / COMPLETE_NAME,
        {
            "status": "complete",
            "protocol_id": PROTOCOL_ID,
            "dataset_id": protocol["dataset"]["dataset_id"],
            "preprocessing_version": PREPROCESSING_VERSION,
            "cohort_id": COHORT_ID,
            "split_id": SPLIT_ID,
        },
    )
    return PrepareReport(
        output_dir=staging,
        reused=False,
        dataset_logical_hash=logical_hash,
        pilot_sha256=HISTORICAL_SAMPLE_SHA256,
        dev5000_sha256=cohort_id_hash(cohort),
        item_ids_sha256=str(publication["item_ids_sha256"]),
        kc_ids_sha256=str(publication["kc_ids_sha256"]),
        row_counts={
            "train": diagnostics.train_rows,
            "valid": diagnostics.valid_rows,
            "test": diagnostics.test_rows,
            "eligible": diagnostics.eligible_rows,
        },
    )


def _label_user(
    interactions: Sequence[Interaction],
) -> tuple[tuple[Interaction, ...], tuple[int, ...], tuple[int, ...]]:
    solving = tuple(int(row.bundle_id.split("|", 1)[1]) for row in interactions)
    source_rows = tuple(int(row.interaction_id.rsplit(":", 1)[1]) for row in interactions)
    labeled = assign_solving_splits(interactions, solving)
    return labeled, solving, source_rows


def _publication_document(
    items: tuple[str, ...],
    kcs: tuple[int, ...],
    protocol: Mapping[str, object],
    oov_rows: int,
    oov_types: int,
) -> dict[str, object]:
    expected = protocol["vocabulary"]
    item_hash = hash_ordered_ids(items)
    kc_hash = hash_ordered_ids(tuple(str(kc) for kc in kcs))
    if len(items) != int(expected["item_count"]) or item_hash != expected["item_ids_sha256"]:
        raise PrepareError(
            "publication item vocabulary "
            f"count={len(items)} hash={item_hash} does not match the frozen protocol"
        )
    if len(kcs) != int(expected["kc_count"]) or kc_hash != expected["kc_ids_sha256"]:
        raise PrepareError(
            "publication KC vocabulary "
            f"count={len(kcs)} hash={kc_hash} does not match the frozen protocol"
        )
    if -1 in kcs:
        raise PrepareError("publication KC vocabulary still contains -1")
    if oov_rows != int(expected["test_item_oov_rows"]) or oov_types != int(expected["test_item_oov_types"]):
        raise PrepareError(
            f"TEST item OOV rows={oov_rows} types={oov_types} "
            "does not match the frozen publication vocabulary"
        )
    return {
        "vocab_id": expected["vocab_id"],
        "source_population": expected["source_population"],
        "source_partitions": list(expected["source_partitions"]),
        "test_included": False,
        "sentinel_removed": -1,
        "sorting": {
            "items": "lexicographic question_id",
            "kcs": "ascending integer, sentinel -1 removed",
            "learners": "lexicographic dev5000 order",
        },
        "item_count": len(items),
        "kc_count": len(kcs),
        "item_ids_sha256": item_hash,
        "kc_ids_sha256": kc_hash,
        "item_ids": list(items),
        "kc_ids": list(kcs),
        "test_item_oov_rows": oov_rows,
        "test_item_oov_types": oov_types,
    }


def _check_counts(diagnostics: object, protocol: Mapping[str, object]) -> None:
    expected = protocol["dataset"]["row_counts"]
    actual = {
        "train": diagnostics.train_rows,
        "valid": diagnostics.valid_rows,
        "test": diagnostics.test_rows,
        "eligible": diagnostics.eligible_rows,
    }
    for name, count in expected.items():
        if int(actual[name]) != int(count):
            raise PrepareError(f"{name} row count {actual[name]} != {count}")
    if diagnostics.filtered_rows != BLANK_ANSWER_ROWS:
        raise PrepareError(
            f"filtered rows {diagnostics.filtered_rows} != {BLANK_ANSWER_ROWS}"
        )
    reasons = dict(diagnostics.filter_reasons)
    if reasons.get("user_answer_empty") != BLANK_ANSWER_ROWS:
        raise PrepareError("blank user_answer filter does not match the frozen 222 rows")


def _dataset_identity(
    rows: Sequence[CorrectedEdNetRow],
    publication: Mapping[str, object],
    protocol: Mapping[str, object],
) -> DatasetLogicalIdentity:
    dataset = protocol["dataset"]
    splits: list[SplitLogicalIdentity] = []
    for split in ("train", "valid", "test"):
        chosen = tuple(
            row.interaction for row in rows if row.interaction.split == split
        )
        splits.append(
            SplitLogicalIdentity(
                split,
                logical_split_hash(chosen),
                len(chosen),
            )
        )
    return DatasetLogicalIdentity(
        dataset_id=str(dataset["dataset_id"]),
        protocol_id=PROTOCOL_ID,
        preprocessing_version=PREPROCESSING_VERSION,
        cohort_hash=str(dataset["cohort_sha256"]),
        splits=tuple(splits),
        item_vocabulary_hash=str(publication["item_ids_sha256"]),
        kc_vocabulary_hash=str(publication["kc_ids_sha256"]),
    )


def _provenance_document(rows: Sequence[CorrectedEdNetRow]) -> dict[str, object]:
    splits: dict[str, object] = {}
    for split in ("train", "valid", "test"):
        chosen = tuple(row for row in rows if row.interaction.split == split)
        splits[split] = {
            "rows": len(chosen),
            "hash": provenance_hash(chosen),
        }
    return {
        "version": PROVENANCE_VERSION,
        "fields": [
            "learner_id",
            "source_row",
            "item_id",
            "solving_id",
            "timestamp",
            "split",
        ],
        "splits": splits,
        "all_rows_hash": provenance_hash(rows),
    }


def _write(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _oov_count(rows: Sequence[CorrectedEdNetRow], items: set[str]) -> tuple[int, int]:
    test_rows = [row for row in rows if row.interaction.split == "test"]
    missing = [row.interaction.item_id for row in test_rows if row.interaction.item_id not in items]
    return len(missing), len(set(missing))
