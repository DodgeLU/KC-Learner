"""Protocol, factory, dry-run, and artifact tests. No full training."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from kclearner.experiments.cli import matrix_ids
from kclearner.experiments.config import formal_run_matrix, load_protocol, resolve_run
from kclearner.experiments.environment import environment_report
from kclearner.experiments.factory import create_model
from kclearner.experiments.runner import (
    FormalRunBlocked,
    IdentityError,
    execute_formal,
    masked_metrics,
    verify_dataset_manifest,
    verify_vocab_file,
)
from kclearner.data.ednet_corrected import cohort_id_hash


def _protocol() -> dict:
    path = Path(__file__).resolve().parents[1] / "configs" / "ednet_corrected" / "protocol.json"
    return load_protocol(path)


class ProtocolTests(unittest.TestCase):
    def test_frozen_identity_and_fourteen_runs(self) -> None:
        protocol = _protocol()
        self.assertEqual(
            protocol["dataset"]["cohort_sha256"],
            "d9bb296cef94ba4ca0fa80ac8da98b7179b537dbce14bcb7555d021423bcaf9b",
        )
        self.assertEqual(protocol["dataset"]["row_counts"]["train"], 806148)
        self.assertEqual(protocol["vocabulary"]["item_count"], 12259)
        self.assertEqual(protocol["vocabulary"]["kc_count"], 188)
        self.assertEqual(protocol["vocabulary"]["test_item_oov_rows"], 0)
        runs = formal_run_matrix(protocol)
        self.assertEqual(len(runs), 14)
        self.assertEqual(runs[0].run_id, "ednet_corrected_irt")
        dkt = [run for run in runs if run.model == "dkt_q"]
        self.assertEqual([run.seed for run in dkt], [42, 43, 44])
        self.assertEqual(dkt[0].config_hash(), resolve_run(protocol, "dkt_q", 42).config_hash())
        self.assertNotEqual(dkt[0].config_hash(), dkt[1].config_hash())
        self.assertEqual(matrix_ids(), [run.run_id for run in runs])

    def test_deterministic_model_rejects_another_seed(self) -> None:
        with self.assertRaises(Exception):
            resolve_run(_protocol(), "irt", 43)

    def test_factory_uses_validated_irt(self) -> None:
        config = resolve_run(_protocol(), "irt")
        model = create_model(
            "irt",
            config,
            {"learner_count": 4, "item_count": 3, "kc_count": 2},
        )
        self.assertEqual(model.theta.shape, (4,))
        self.assertEqual(model.b.shape, (3,))

    def test_manifest_and_vocab_hashes(self) -> None:
        protocol = _protocol()
        manifest = {
            "preprocessing_version": "ednet_kt1_corrected_v1",
            "cohort_id": "ednet_kt1_dev5000",
            "ordered_learner_ids_sha256": protocol["dataset"]["cohort_sha256"],
            "row_counts": protocol["dataset"]["row_counts"],
            "files_sha256": protocol["dataset"]["files_sha256"],
        }
        verify_dataset_manifest(manifest, protocol)
        items = ("q1", "q2")
        kcs = (1, 5)
        payload = {"item_ids": list(items), "kc_ids": list(kcs)}
        mini = json.loads(json.dumps(protocol))
        mini["vocabulary"]["item_count"] = 2
        mini["vocabulary"]["kc_count"] = 2
        mini["vocabulary"]["item_ids_sha256"] = cohort_id_hash(items)
        mini["vocabulary"]["kc_ids_sha256"] = cohort_id_hash(("1", "5"))
        vocab = verify_vocab_file(payload, mini)
        self.assertEqual(vocab.item_ids, items)
        bad = dict(payload)
        bad["kc_ids"] = [-1, 1]
        mini["vocabulary"]["kc_count"] = 2
        mini["vocabulary"]["kc_ids_sha256"] = cohort_id_hash(("-1", "1"))
        with self.assertRaises(IdentityError):
            verify_vocab_file(bad, mini)

    def test_metric_mask_drops_rows(self) -> None:
        full = masked_metrics([0, 1], [0.2, 0.8], [True, True])
        masked = masked_metrics([0, 1], [0.2, 0.8], [True, False])
        self.assertEqual(full["n"], 2)
        self.assertEqual(masked["n"], 1)
        self.assertIsNone(masked_metrics([1], [0.5], [False]))

    def test_environment_and_formal_block(self) -> None:
        report = environment_report()
        self.assertIn("python", report)
        self.assertIn("numpy", report)
        self.assertIn(report["device"], ("cpu", "cuda"))
        with self.assertRaises(FormalRunBlocked):
            execute_formal()

    def test_dry_run_writes_directory(self) -> None:
        from kclearner.experiments.runner import dry_run

        protocol = _protocol()
        config = resolve_run(protocol, "irt")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "data"
            dataset.mkdir()
            items = ("q1",)
            kcs = (2,)
            learners = ("u1",)
            mini = json.loads(json.dumps(protocol))
            mini["vocabulary"]["item_count"] = 1
            mini["vocabulary"]["kc_count"] = 1
            mini["vocabulary"]["learner_count"] = 1
            mini["vocabulary"]["item_ids_sha256"] = cohort_id_hash(items)
            mini["vocabulary"]["kc_ids_sha256"] = cohort_id_hash(("2",))
            mini["vocabulary"]["learner_ids_sha256"] = cohort_id_hash(learners)
            config.protocol.clear()
            config.protocol.update(mini)
            (dataset / "manifest.json").write_text(json.dumps({
                "preprocessing_version": "ednet_kt1_corrected_v1",
                "cohort_id": "ednet_kt1_dev5000",
                "ordered_learner_ids_sha256": protocol["dataset"]["cohort_sha256"],
                "row_counts": protocol["dataset"]["row_counts"],
                "files_sha256": protocol["dataset"]["files_sha256"],
            }), encoding="utf-8")
            (dataset / "publication_vocab.json").write_text(json.dumps({
                "item_ids": ["q1"],
                "kc_ids": [2],
            }), encoding="utf-8")
            (dataset / "dev5000_users.txt").write_text("u1\n", encoding="utf-8")
            for name in ("train.parquet", "valid.parquet", "test.parquet"):
                (dataset / name).write_bytes(b"")
            report = dry_run(config, dataset_dir=dataset, runs_dir=root / "runs")
            self.assertTrue(report["model_ready"])
            self.assertTrue((root / "runs" / config.run_id / "metadata.json").is_file())
            self.assertTrue((root / "runs" / config.run_id / "environment.json").is_file())
            saved = json.loads((root / "runs" / config.run_id / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["phase"], "dry_run")
            self.assertEqual(saved["cohort_hash"], protocol["dataset"]["cohort_sha256"])
            self.assertFalse(saved["publication"])
