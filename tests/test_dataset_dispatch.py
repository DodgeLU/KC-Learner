"""Dataset selection, AR-KT recipe dispatch, and ASSISTments OOV mapping."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from kclearner.data.adapters.assistments2017 import AssistmentsModelRow
from kclearner.experiments.assistments_run import rows_for_execution, run_assistments_formal
from kclearner.experiments.cli import main
from kclearner.experiments.config import (
    ConfigError,
    formal_run_matrix,
    load_protocol,
    resolve_run,
)
from kclearner.experiments.factory import create_model
from kclearner.experiments.runner import IdentityError
from kclearner.models.dkt import DKTRow
from kclearner.models.streaming import StreamingRow

try:
    import pyarrow
except Exception:
    pyarrow = None


def _protocol(name: str, dataset: str) -> dict:
    path = Path(__file__).resolve().parents[1] / "configs" / name / "protocol.json"
    return load_protocol(path, dataset=dataset)


class ProtocolSelectionTests(unittest.TestCase):
    def test_dataset_and_protocol_mismatches_are_rejected(self) -> None:
        root = Path(__file__).resolve().parents[1] / "configs"
        with self.assertRaises(ConfigError):
            load_protocol(root / "ednet_corrected" / "protocol.json", dataset="assistments2017")
        with self.assertRaises(ConfigError):
            load_protocol(root / "assistments2017" / "protocol.json", dataset="ednet_kt1")
        with self.assertRaises(ConfigError):
            load_protocol(root / "assistments2017" / "protocol.json")
        loaded = load_protocol(
            root / "assistments2017" / "protocol.json",
            dataset="assistments2017",
        )
        self.assertEqual(loaded["protocol_id"], "ASSISTMENTS2017_MAIN_V1")

    def test_publication_seeds_stay_dataset_specific(self) -> None:
        ednet = _protocol("ednet_corrected", "ednet_kt1")
        assist = _protocol("assistments2017", "assistments2017")
        self.assertEqual(len(formal_run_matrix(ednet)), 14)
        self.assertEqual(formal_run_matrix(ednet)[0].run_id, "ednet_corrected_irt")
        with self.assertRaises(ConfigError):
            resolve_run(ednet, "dkt_q", 45)
        self.assertEqual(resolve_run(assist, "dkt_q", 45).run_id, "assistments2017_dkt_q_seed45")
        self.assertEqual(resolve_run(assist, "irt").run_id, "assistments2017_irt")
        with self.assertRaises(ConfigError):
            resolve_run(assist, "dkvmn_qc", 47)
        self.assertEqual(len(formal_run_matrix(assist)), 22)

    def test_prepare_rejects_the_other_datasets_inputs(self) -> None:
        with self.assertRaises(SystemExit) as missing:
            main(["prepare", "--dataset", "assistments2017"])
        self.assertEqual(missing.exception.code, 2)
        with self.assertRaises(SystemExit) as ednet_file:
            main(["prepare", "--dataset", "ednet_kt1", "--raw-file", "primary.csv"])
        self.assertEqual(ednet_file.exception.code, 2)
        with self.assertRaises(SystemExit) as assist_questions:
            main([
                "prepare",
                "--dataset",
                "assistments2017",
                "--raw-file",
                "primary.csv",
                "--questions",
                "questions.csv",
            ])
        self.assertEqual(assist_questions.exception.code, 2)


class ResidualAndOOVTests(unittest.TestCase):
    def test_ar_kt_residual_update_follows_the_dataset_recipe(self) -> None:
        sizes = {"learner_count": 2, "item_count": 3, "kc_count": 2}
        ednet = create_model("ar_kt", resolve_run(_protocol("ednet_corrected", "ednet_kt1"), "ar_kt"), sizes)
        assist = create_model(
            "ar_kt",
            resolve_run(_protocol("assistments2017", "assistments2017"), "ar_kt"),
            sizes,
        )
        self.assertEqual(ednet.residual_update, "ednet_token")
        self.assertEqual(assist.residual_update, "assistments_skill_mean")

    def test_execution_rows_keep_the_family_oov_difference(self) -> None:
        kept = AssistmentsModelRow(
            student_idx=0,
            problem_idx=1,
            skill_idx=0,
            group_id=4,
            correct=1,
            timestamp=1,
            split="test",
            metric_mask=True,
            oov_problem=False,
            interaction_id="keep",
            learner_id="1",
            item_id="2",
            skill_id="area",
        )
        oov = AssistmentsModelRow(
            student_idx=0,
            problem_idx=0,
            skill_idx=0,
            group_id=5,
            correct=0,
            timestamp=2,
            split="test",
            metric_mask=False,
            oov_problem=True,
            interaction_id="oov",
            learner_id="1",
            item_id="99",
            skill_id="area",
        )
        psychometric = rows_for_execution((kept, oov), "ar_kt")
        neural = rows_for_execution((kept, oov), "dkvmn_q")
        self.assertEqual([row.interaction_id for row in psychometric], ["keep", "oov"])
        self.assertIsInstance(psychometric[0], StreamingRow)
        self.assertEqual([row.interaction_id for row in neural], ["keep"])
        self.assertIsInstance(neural[0], DKTRow)
        self.assertEqual(neural[0].solving_id, 4)
        self.assertEqual(neural[0].tag_idxs, ())
        qc = rows_for_execution((kept,), "dkt_qc")
        self.assertEqual(qc[0].tag_idxs, (0,))

    def test_formal_run_blocks_a_missing_recipe_before_reading_rows(self) -> None:
        protocol = _protocol("assistments2017", "assistments2017")
        protocol["models"]["irt"]["recipe"] = {}
        config = resolve_run(protocol, "irt")
        with self.assertRaises(IdentityError):
            run_assistments_formal(
                config,
                dataset_dir=".",
                runs_dir=".",
                requested_device="cpu",
            )

    def test_formal_run_rejects_the_ednet_residual_policy(self) -> None:
        protocol = _protocol("assistments2017", "assistments2017")
        protocol["models"]["ar_kt"]["recipe"]["residual_update"] = "ednet_token"
        config = resolve_run(protocol, "ar_kt")
        with self.assertRaises(IdentityError):
            run_assistments_formal(
                config,
                dataset_dir=".",
                runs_dir=".",
                requested_device="cpu",
            )


@unittest.skipUnless(pyarrow is not None, "pyarrow is not installed")
class PreparedDispatchTests(unittest.TestCase):
    def test_dry_run_uses_the_assistments_reader(self) -> None:
        from kclearner.data.assistments_prepare import prepare_assistments2017
        from kclearner.experiments.runner import dry_run

        protocol = {
            "protocol_id": "ASSISTMENTS2017_MAIN_V1",
            "run_id_prefix": "assistments2017",
            "dataset": {
                "dataset_id": "assistments2017",
                "preprocessing_version": "assist17_primary_v1",
                "enforce_frozen_counts": False,
            },
            "vocabulary": {},
            "sequence": {"bundle": "(student_id, timestamp)"},
            "metrics": ["nll", "brier", "auc"],
            "phases": ["train", "valid"],
            "models": {
                "irt": {
                    "family": "irt",
                    "kc_aware": False,
                    "deterministic": True,
                    "provenance_seed": 42,
                    "recipe": {
                        "theta_lr": 0.05,
                        "b_lr": 0.02,
                        "theta_l2": 0.0001,
                        "b_l2": 0.0001,
                        "learn_b": True,
                        "freeze_global_in_eval": True,
                    },
                }
            },
        }
        body = "student_id,problem_id,skill_id,correct,timestamp,attempt_count\n"
        body += "\n".join(
            f"1,{index + 1},skill,{index % 2},{index + 1},1" for index in range(10)
        ) + "\n"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "primary.csv"
            source.write_text(body, encoding="utf-8")
            prepared = root / "prepared"
            prepare_assistments2017(source, prepared, protocol)
            report = dry_run(
                resolve_run(protocol, "irt"),
                dataset_dir=prepared,
                runs_dir=root / "runs",
            )
            self.assertEqual(report["status"], "dry_run")
            self.assertTrue(report["model_ready"])
            self.assertEqual(report["protocol_id"], "ASSISTMENTS2017_MAIN_V1")
            self.assertFalse((prepared / "dev5000_users.txt").exists())
            self.assertTrue((root / "runs" / "assistments2017_irt" / "dry_run.json").is_file())
