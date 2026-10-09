"""External canonical-directory loader. No EdNet or ASSISTments rows."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from kclearner.data.canonical_dataset import (
    CANONICAL_SCHEMA_VERSION,
    CONTRACT_IDENTITY_VERSION,
    DECLARATION_NAME,
    INTERACTIONS_NAME,
    ITEM_IDS_NAME,
    KC_IDS_NAME,
    LEARNER_IDS_NAME,
    CanonicalDatasetError,
    canonical_contract_json,
    load_canonical_dataset,
)
from kclearner.data.ednet_corrected import SCHEMA_VERSION
from kclearner.data.logical_identity import (
    DATASET_IDENTITY_VERSION,
    SERIALIZATION_VERSION,
)


def _row(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "interaction_id": "L1:0",
        "learner_id": "L1",
        "item_id": "Q1",
        "correct": 1,
        "kc_ids": [3],
        "order": 0,
        "timestamp": 10,
        "bundle_id": "L1|b0",
        "split": "fit",
        "metric_mask": True,
    }
    payload.update(overrides)
    return payload


def _declaration(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "dataset_id": "synthetic_external_v1",
        "protocol_id": "synthetic_protocol_v1",
        "preprocessing_version": "synthetic_prep_v1",
        "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
        "interaction_identity_version": SERIALIZATION_VERSION,
        "dataset_identity_version": DATASET_IDENTITY_VERSION,
        "split_order": ["fit"],
        "split_roles": {"fit": "train"},
        "bundle_semantics_id": "caller_bundle_v1",
        "oov_representation_semantics_id": "caller_oov_representation_v1",
        "metric_mask_semantics": "caller_metric_mask_v1",
        "provenance": {"note": "synthetic note", "url": "https://example.invalid/data"},
    }
    payload.update(overrides)
    return payload


def _write(
    root: Path,
    *,
    declaration: dict[str, object] | None = None,
    rows: list[dict[str, object]] | None = None,
    learners: list[str] | None = None,
    items: list[str] | None = None,
    kcs: list[str] | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / DECLARATION_NAME).write_text(
        json.dumps(declaration if declaration is not None else _declaration()),
        encoding="utf-8",
    )
    chosen = rows if rows is not None else [_row()]
    (root / INTERACTIONS_NAME).write_text(
        "".join(json.dumps(row) + "\n" for row in chosen),
        encoding="utf-8",
    )
    _write_ids(root / LEARNER_IDS_NAME, learners if learners is not None else ["L1"])
    _write_ids(root / ITEM_IDS_NAME, items if items is not None else ["Q1"])
    _write_ids(root / KC_IDS_NAME, kcs if kcs is not None else ["3"])


def _write_ids(path: Path, ids: list[str]) -> None:
    path.write_text("".join(item + "\n" for item in ids), encoding="utf-8")


class CanonicalDatasetTests(unittest.TestCase):
    def test_directory_loads_and_hash_is_stable(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root)
            first = load_canonical_dataset(root)
            second = load_canonical_dataset(root)
        self.assertEqual(first.dataset_logical_hash, second.dataset_logical_hash)
        self.assertEqual(first.dataset_contract_hash, second.dataset_contract_hash)
        self.assertEqual(first.learner_ids, ("L1",))
        self.assertEqual(first.interactions[0].bundle_id, "L1|b0")
        self.assertEqual(first.logical_identity.identity_version, DATASET_IDENTITY_VERSION)
        self.assertEqual(
            first.logical_identity.serialization_version, SERIALIZATION_VERSION
        )
        self.assertEqual(len(first.dataset_logical_hash), 64)
        self.assertEqual(len(first.dataset_contract_hash), 64)

    def test_invalid_row_schema_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = _row()
            del row["bundle_id"]
            _write(root, rows=[row])
            with self.assertRaises(CanonicalDatasetError):
                load_canonical_dataset(root)

    def test_unknown_execution_column_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, rows=[_row(solving_id=7)])
            with self.assertRaises(CanonicalDatasetError):
                load_canonical_dataset(root)

    def test_bundle_id_change_changes_logical_identity(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root)
            original = load_canonical_dataset(root)
            _write(root, rows=[_row(bundle_id="L1|other")])
            changed = load_canonical_dataset(root)
        self.assertNotEqual(original.dataset_logical_hash, changed.dataset_logical_hash)
        self.assertNotEqual(original.dataset_contract_hash, changed.dataset_contract_hash)

    def test_identity_bearing_semantics_change_only_the_contract_hash(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root)
            original = load_canonical_dataset(root)
            _write(root, declaration=_declaration(bundle_semantics_id="caller_bundle_v2"))
            changed = load_canonical_dataset(root)
        self.assertEqual(original.dataset_logical_hash, changed.dataset_logical_hash)
        self.assertNotEqual(original.dataset_contract_hash, changed.dataset_contract_hash)
        self.assertNotIn("caller_bundle_v1", canonical_contract_json(
            dataset_logical_hash_value=changed.dataset_logical_hash,
            canonical_schema_version=changed.canonical_schema_version,
            bundle_semantics_id=changed.bundle_semantics_id,
            oov_representation_semantics_id=changed.oov_representation_semantics_id,
            metric_mask_semantics=changed.metric_mask_semantics,
            split_roles=changed.split_roles,
        ))

    def test_split_role_change_keeps_logical_hash(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root)
            original = load_canonical_dataset(root)
            _write(
                root,
                declaration=_declaration(split_roles={"fit": "valid"}),
            )
            changed = load_canonical_dataset(root)
        self.assertEqual(original.dataset_logical_hash, changed.dataset_logical_hash)
        self.assertNotEqual(original.dataset_contract_hash, changed.dataset_contract_hash)
        self.assertEqual(changed.split_roles, (("fit", "valid"),))

    def test_descriptive_provenance_does_not_change_contract_hash(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root)
            original = load_canonical_dataset(root)
            _write(
                root,
                declaration=_declaration(
                    provenance={"note": "edited note", "prepared_at": "2026-10-09"},
                    source_fingerprint={"raw_sha256": "abc", "url": "https://example.invalid/raw"},
                ),
            )
            changed = load_canonical_dataset(root)
        self.assertEqual(original.dataset_logical_hash, changed.dataset_logical_hash)
        self.assertEqual(original.dataset_contract_hash, changed.dataset_contract_hash)
        self.assertTrue(changed.has_source_fingerprint)
        contract = canonical_contract_json(
            dataset_logical_hash_value=changed.dataset_logical_hash,
            canonical_schema_version=changed.canonical_schema_version,
            bundle_semantics_id=changed.bundle_semantics_id,
            oov_representation_semantics_id=changed.oov_representation_semantics_id,
            metric_mask_semantics=changed.metric_mask_semantics,
            split_roles=changed.split_roles,
        )
        self.assertNotIn("edited note", contract)
        self.assertNotIn("raw_sha256", contract)
        self.assertNotIn("https://example.invalid/raw", contract)

    def test_missing_split_role_is_rejected(self) -> None:
        declaration = _declaration(
            split_order=["fit", "holdout", "later"],
            split_roles={"fit": "train", "holdout": "valid"},
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, declaration=declaration)
            with self.assertRaises(CanonicalDatasetError) as caught:
                load_canonical_dataset(root)
        self.assertIn("later", str(caught.exception))

    def test_duplicate_workflow_role_is_rejected(self) -> None:
        declaration = _declaration(
            split_order=["fit", "check", "later"],
            split_roles={"fit": "train", "check": "valid", "later": "valid"},
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, declaration=declaration)
            with self.assertRaises(CanonicalDatasetError) as caught:
                load_canonical_dataset(root)
        self.assertIn("valid", str(caught.exception))

    def test_list_position_does_not_assign_test(self) -> None:
        declaration = _declaration(
            split_order=["a", "b", "c"],
            split_roles={"a": "train", "b": "valid", "c": "not_a_role"},
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, declaration=declaration)
            with self.assertRaises(CanonicalDatasetError) as caught:
                load_canonical_dataset(root)
        self.assertIn("List position does not assign a role", str(caught.exception))

    def test_oov_representation_is_required_and_not_interpreted(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = _declaration()
            del missing["oov_representation_semantics_id"]
            _write(root, declaration=missing)
            with self.assertRaises(CanonicalDatasetError):
                load_canonical_dataset(root)
            _write(
                root,
                declaration=_declaration(
                    oov_representation_semantics_id="family_dependent_representation_v1"
                ),
                rows=[_row(), _row(interaction_id="L1:1", order=1, item_id="Q9", bundle_id="L1|b1")],
            )
            loaded = load_canonical_dataset(root)
        self.assertEqual(
            loaded.oov_representation_semantics_id,
            "family_dependent_representation_v1",
        )
        self.assertEqual(len(loaded.interactions), 2)
        self.assertFalse(hasattr(loaded, "state_transparent"))
        self.assertFalse(hasattr(loaded, "updates_learner_state"))
        names = set(loaded.__dataclass_fields__)
        self.assertNotIn("oov_policy_id", names)

    def test_vocabulary_order_is_not_sorted(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, learners=["b", "a"], items=["q2", "q1"], kcs=["10", "2"])
            descending = load_canonical_dataset(root)
            _write(root, learners=["a", "b"], items=["q1", "q2"], kcs=["2", "10"])
            ascending = load_canonical_dataset(root)
        self.assertEqual(descending.learner_ids, ("b", "a"))
        self.assertNotEqual(
            descending.dataset_logical_hash, ascending.dataset_logical_hash
        )

    def test_hash_version_and_ednet_schema_label_are_not_accepted_as_schema(self) -> None:
        self.assertEqual(SCHEMA_VERSION, "kclearner_interaction_v1")
        self.assertNotEqual(SCHEMA_VERSION, CANONICAL_SCHEMA_VERSION)
        self.assertNotEqual(SERIALIZATION_VERSION, CANONICAL_SCHEMA_VERSION)
        self.assertEqual(CONTRACT_IDENTITY_VERSION, "dataset_contract_identity_v1")
        for version in (SERIALIZATION_VERSION, SCHEMA_VERSION):
            with TemporaryDirectory() as tmp:
                root = Path(tmp)
                _write(root, declaration=_declaration(canonical_schema_version=version))
                with self.assertRaises(CanonicalDatasetError):
                    load_canonical_dataset(root)

    def test_duplicate_vocabulary_id_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, learners=["L1", "L1"])
            with self.assertRaises(CanonicalDatasetError):
                load_canonical_dataset(root)

    def test_noncanonical_kc_token_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root, kcs=["01"])
            with self.assertRaises(CanonicalDatasetError):
                load_canonical_dataset(root)


if __name__ == "__main__":
    unittest.main()
