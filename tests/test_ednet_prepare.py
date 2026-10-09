"""Synthetic checks for public EdNet preparation helpers."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from kclearner.data.ednet_corrected import CorrectedEdNetRow
from kclearner.data.ednet_prepare import (
    PROVENANCE_VERSION,
    PrepareError,
    assess_existing_output,
    file_aggregate_hash,
    provenance_hash,
    publication_ids_from_rows,
)
from kclearner.data.schema import Interaction
from kclearner.experiments.config import load_protocol
from kclearner.experiments.runner import IdentityError, verify_dataset_manifest


def _interaction(split: str, item: str, kc: tuple[int, ...] = (1,)) -> Interaction:
    return Interaction(
        interaction_id=f"u:{item}",
        learner_id="u",
        item_id=item,
        correct=1,
        kc_ids=kc,
        order=0,
        timestamp=5,
        bundle_id="u|1",
        split=split,
        metric_mask=True,
    )


def _corrected(split: str, source_row: int) -> CorrectedEdNetRow:
    return CorrectedEdNetRow(
        interaction=_interaction(split, "q1"),
        source_row=source_row,
        solving_id=1,
    )


class PublicationScopeTests(unittest.TestCase):
    def test_test_rows_do_not_enter_the_vocabulary(self) -> None:
        items, kcs = publication_ids_from_rows((
            ("train", "b", (2, 2)),
            ("valid", "a", (9,)),
            ("test", "zzz", (4,)),
        ))
        self.assertEqual(items, ("a", "b"))
        self.assertEqual(kcs, (2, 9))

    def test_negative_kc_is_rejected(self) -> None:
        with self.assertRaises(PrepareError):
            publication_ids_from_rows((("train", "a", (-1,)),))


class ProvenanceTests(unittest.TestCase):
    def test_source_row_changes_the_hash(self) -> None:
        self.assertEqual(PROVENANCE_VERSION, "ednet_source_provenance_hash_v1")
        self.assertNotEqual(
            provenance_hash((_corrected("train", 0),)),
            provenance_hash((_corrected("train", 1),)),
        )
        self.assertEqual(
            file_aggregate_hash((("u1", "aa"), ("u2", "bb"))),
            file_aggregate_hash((("u1", "aa"), ("u2", "bb"))),
        )
        self.assertNotEqual(
            file_aggregate_hash((("u1", "aa"), ("u2", "bb"))),
            file_aggregate_hash((("u2", "bb"), ("u1", "aa"))),
        )


class ReuseTests(unittest.TestCase):
    def _write_complete(self, root: Path, cohort: str, item_hash: str, kc_hash: str) -> None:
        logical = {
            "dataset_logical_hash": "d" * 64,
            "dataset_logical_identity": {
                "cohort_hash": cohort,
                "item_vocabulary_hash": item_hash,
                "kc_vocabulary_hash": kc_hash,
                "splits": [],
            },
        }
        source = {
            "questions": {"sha256": "q" * 64},
            "population": {"ordered_ids_sha256": "p" * 64},
            "selected_pilot_files": {
                "aggregate_sha256": "f" * 64,
                "cohort_sha256": "20b29ac2ebf6e8cca9e763f11a44d850c773cf04e27464c8c5b0e8e9c7566b96",
            },
        }
        for name in (
            "train.parquet",
            "valid.parquet",
            "test.parquet",
            "manifest.json",
            "cohort_manifest.json",
            "dev5000_users.txt",
            "publication_vocab.json",
        ):
            (root / name).write_text("x", encoding="utf-8")
        (root / "logical_identity.json").write_text(json.dumps(logical), encoding="utf-8")
        (root / "source_provenance.json").write_text(json.dumps(source), encoding="utf-8")
        (root / "prepare_status.json").write_text(
            json.dumps({"status": "complete", "protocol_id": "ednet_corrected_publication_v1"}),
            encoding="utf-8",
        )

    def test_complete_matching_directory_is_reused(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_complete(root, "c" * 64, "i" * 64, "k" * 64)
            self.assertEqual(
                assess_existing_output(
                    root,
                    protocol_id="ednet_corrected_publication_v1",
                    questions_sha256="q" * 64,
                    population_sha256="p" * 64,
                    pilot_file_sha256="f" * 64,
                    cohort_hash="c" * 64,
                    item_vocabulary_hash="i" * 64,
                    kc_vocabulary_hash="k" * 64,
                ),
                "reuse",
            )

    def test_changed_questions_hash_is_incompatible(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_complete(root, "c" * 64, "i" * 64, "k" * 64)
            self.assertEqual(
                assess_existing_output(
                    root,
                    protocol_id="ednet_corrected_publication_v1",
                    questions_sha256="z" * 64,
                    population_sha256="p" * 64,
                    pilot_file_sha256="f" * 64,
                    cohort_hash="c" * 64,
                    item_vocabulary_hash="i" * 64,
                    kc_vocabulary_hash="k" * 64,
                ),
                "incompatible",
            )

    def test_missing_directory(self) -> None:
        with TemporaryDirectory() as directory:
            self.assertEqual(
                assess_existing_output(
                    Path(directory) / "absent",
                    protocol_id="ednet_corrected_publication_v1",
                    questions_sha256="q" * 64,
                    population_sha256="p" * 64,
                    pilot_file_sha256="f" * 64,
                    cohort_hash="c" * 64,
                    item_vocabulary_hash="i" * 64,
                    kc_vocabulary_hash="k" * 64,
                ),
                "missing",
            )


class LogicalManifestTests(unittest.TestCase):
    def test_byte_mismatch_without_logical_identity_fails(self) -> None:
        protocol = load_protocol(
            Path(__file__).resolve().parents[1] / "configs" / "ednet_corrected" / "protocol.json"
        )
        manifest = {
            "preprocessing_version": "ednet_kt1_corrected_v1",
            "cohort_id": "ednet_kt1_dev5000",
            "ordered_learner_ids_sha256": protocol["dataset"]["cohort_sha256"],
            "row_counts": protocol["dataset"]["row_counts"],
            "files_sha256": {"train": "0" * 64, "valid": "0" * 64, "test": "0" * 64},
        }
        with self.assertRaises(IdentityError):
            verify_dataset_manifest(manifest, protocol)

    def test_logical_identity_accepts_a_different_parquet_encoding(self) -> None:
        protocol = load_protocol(
            Path(__file__).resolve().parents[1] / "configs" / "ednet_corrected" / "protocol.json"
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"not-the-historical-parquet"
            import hashlib
            digest = hashlib.sha256(payload).hexdigest()
            for name in ("train", "valid", "test"):
                (root / f"{name}.parquet").write_bytes(payload)
            manifest = {
                "preprocessing_version": "ednet_kt1_corrected_v1",
                "cohort_id": "ednet_kt1_dev5000",
                "ordered_learner_ids_sha256": protocol["dataset"]["cohort_sha256"],
                "row_counts": protocol["dataset"]["row_counts"],
                "files_sha256": {"train": digest, "valid": digest, "test": digest},
                "dataset_logical_identity": {
                    "cohort_hash": protocol["dataset"]["cohort_sha256"],
                    "item_vocabulary_hash": protocol["vocabulary"]["item_ids_sha256"],
                    "kc_vocabulary_hash": protocol["vocabulary"]["kc_ids_sha256"],
                    "splits": [
                        {"split_id": "train", "row_count": 806148},
                        {"split_id": "valid", "row_count": 117184},
                        {"split_id": "test", "row_count": 235031},
                    ],
                },
            }
            verify_dataset_manifest(manifest, protocol, dataset_dir=root)
