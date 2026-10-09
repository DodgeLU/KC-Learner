"""Synthetic canonical bundle execution. No EdNet or ASSISTments rows."""

from __future__ import annotations

import json
import unittest
from io import StringIO
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from kclearner.data.canonical_dataset import canonical_contract_json, load_canonical_dataset
from kclearner.experiments.canonical_run import (
    _cap_complete_bundles,
    execute_canonical,
    execute_canonical_directory,
)
from kclearner.experiments.cli import main
from test_canonical_dataset import _declaration, _row, _write


def _directory(root: Path) -> None:
    _write(
        root,
        declaration=_declaration(
            split_order=["later", "fit"],
            split_roles={"later": "test", "fit": "train"},
        ),
        rows=[
            _row(
                interaction_id="L2:0",
                learner_id="L2",
                item_id="Q2",
                order=0,
                bundle_id="L2|only",
                split="later",
            ),
            _row(interaction_id="L1:0", order=0, bundle_id="L1|same", split="fit"),
            _row(interaction_id="L1:1", order=1, bundle_id="L1|same", split="fit"),
            _row(interaction_id="L1:2", order=2, bundle_id="L1|next", split="fit"),
        ],
        learners=["L2", "L1"],
        items=["Q2", "Q1"],
        kcs=["3"],
    )


class CanonicalRunTests(unittest.TestCase):
    def test_contract_hash_separates_output_directories(self) -> None:
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            rows = [_row()]
            left = base / "left"
            right = base / "right"
            runs = base / "runs"
            _write(left, rows=rows, declaration=_declaration(bundle_semantics_id="caller_bundle_v1"))
            _write(right, rows=rows, declaration=_declaration(bundle_semantics_id="caller_bundle_v2"))
            first = execute_canonical_directory(left, runs_dir=runs)
            second = execute_canonical_directory(right, runs_dir=runs)
        self.assertEqual(first["dataset_logical_hash"], second["dataset_logical_hash"])
        self.assertNotEqual(first["dataset_contract_hash"], second["dataset_contract_hash"])
        self.assertNotEqual(first["output_dir"], second["output_dir"])
        self.assertEqual(
            Path(first["output_dir"]).name,
            "canonical_" + str(first["dataset_contract_hash"]),
        )
        self.assertEqual(
            Path(second["output_dir"]).name,
            "canonical_" + str(second["dataset_contract_hash"]),
        )
        self.assertNotEqual(
            Path(first["output_dir"]).name,
            "canonical_" + str(first["dataset_logical_hash"])[:16],
        )
    def test_train_role_groups_by_bundle_id_and_skips_test(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _directory(root)
            loaded = load_canonical_dataset(root)
            execution = execute_canonical(loaded)
        self.assertEqual(execution.executed_role, "train")
        self.assertEqual(execution.executed_split_id, "fit")
        self.assertEqual(
            [row.interaction_id for row in execution.rows],
            ["L1:0", "L1:1", "L1:2"],
        )
        self.assertEqual(
            [row.pre_bundle_update_count for row in execution.rows],
            [0, 0, 2],
        )
        self.assertEqual(execution.rows[0].bundle_id, "L1|same")
        self.assertEqual(execution.rows[2].bundle_id, "L1|next")
        self.assertEqual(execution.bundle_count, 2)
        text = json.dumps(execution.rows[0].__dict__)
        self.assertNotIn("solving_id", text)
        self.assertNotIn("group_id", text)

    def test_cli_writes_a_non_freeze_report(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            runs = Path(tmp) / "runs"
            _directory(root)
            loaded = load_canonical_dataset(root)
            buffer = StringIO()
            with redirect_stdout(buffer):
                code = main([
                    "run-canonical",
                    "--dataset-dir",
                    str(root),
                    "--runs-dir",
                    str(runs),
                ])
            payload = json.loads(buffer.getvalue())
            report_path = Path(payload["output_dir"]) / "bundle_execution.json"
            document = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertFalse((Path(payload["output_dir"]) / "freeze_manifest.json").exists())
        self.assertEqual(code, 0)
        self.assertFalse(payload["publication"])
        self.assertFalse(payload["freeze"])
        self.assertFalse(payload["formal_test"])
        self.assertEqual(payload["grouping"], "bundle_id")
        self.assertEqual(payload["dataset_logical_hash"], loaded.dataset_logical_hash)
        self.assertEqual(payload["dataset_contract_hash"], loaded.dataset_contract_hash)
        self.assertEqual(payload["executed_split_id"], "fit")
        self.assertEqual(document["rows"][0]["bundle_id"], "L1|same")
        self.assertFalse(document["model_execution"])
        self.assertEqual(document["inspection_traversal"], "learner_id_then_interaction_order")
        contract = canonical_contract_json(
            dataset_logical_hash_value=loaded.dataset_logical_hash,
            canonical_schema_version=loaded.canonical_schema_version,
            bundle_semantics_id=loaded.bundle_semantics_id,
            oov_representation_semantics_id=loaded.oov_representation_semantics_id,
            metric_mask_semantics=loaded.metric_mask_semantics,
            split_roles=loaded.split_roles,
        )
        self.assertNotIn("learner_id_then_interaction_order", contract)
        self.assertNotIn("model_execution", contract)
        self.assertNotIn("solving_id", json.dumps(document))
        self.assertNotIn("group_id", json.dumps(document))
        self.assertNotIn("source_fingerprint", json.dumps(document))

    def test_smoke_does_not_split_a_bundle(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _directory(root)
            execution = execute_canonical(load_canonical_dataset(root), smoke=True)
        self.assertEqual(execution.status, "smoke")
        self.assertFalse(execution.truncated)
        self.assertEqual(len(execution.rows), 3)
        capped, truncated = _cap_complete_bundles(execution.rows, 1)
        self.assertEqual([row.interaction_id for row in capped], ["L1:0", "L1:1"])
        self.assertTrue(truncated)
        self.assertEqual(capped[0].pre_bundle_update_count, capped[1].pre_bundle_update_count)

    def test_formal_and_reference_commands_stay_separate(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            _directory(root)
            stderr = StringIO()
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as caught:
                    main([
                        "run-canonical",
                        "--dataset-dir",
                        str(root),
                        "--authorize-formal",
                    ])
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("unrecognized arguments", stderr.getvalue())
        help_run = StringIO()
        with redirect_stdout(help_run):
            with self.assertRaises(SystemExit):
                main(["run", "--help"])
        text = help_run.getvalue()
        self.assertIn("{ednet_kt1,assistments2017}", text)
        self.assertNotIn("run-canonical", text)
        help_eval = StringIO()
        with redirect_stdout(help_eval):
            with self.assertRaises(SystemExit):
                main(["evaluate", "--help"])
        self.assertIn("{ednet_kt1,assistments2017}", help_eval.getvalue())
        self.assertNotIn("canonical", help_eval.getvalue())

    def test_reference_dry_run_does_not_call_the_canonical_loader(self) -> None:
        with TemporaryDirectory() as tmp:
            empty = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                main([
                    "run",
                    "--dataset",
                    "ednet_kt1",
                    "--model",
                    "irt",
                    "--dry-run",
                    "--dataset-dir",
                    str(empty),
                ])


if __name__ == "__main__":
    unittest.main()
