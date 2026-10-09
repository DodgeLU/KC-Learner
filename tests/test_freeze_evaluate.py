"""Freeze identity and TEST authorization. No third-party datasets."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from kclearner.data.adapters.assistments2017 import FROZEN_MAIN_V1_COUNTS
from kclearner.experiments.config import load_protocol, resolve_run
from kclearner.experiments.evaluate import EvaluateError, run_evaluation
from kclearner.experiments.freeze import (
    EVALUATION_SEMANTICS,
    MANIFEST_NAME,
    REPLAY_TOLERANCE,
    FreezeError,
    freeze_id_for,
    make_identity,
    metrics_within_tolerance,
    recipe_hash,
    write_freeze_manifest,
)

try:
    import pyarrow
except Exception:
    pyarrow = None


def _identity(**overrides):
    base = make_identity(
        dataset_id="ednet_kt1_corrected_v1",
        protocol_id="ednet_corrected_publication_v1",
        preprocessing_version="ednet_kt1_corrected_v1",
        dataset_logical_hash="a" * 64,
        cohort_hash="b" * 64,
        item_vocabulary_hash="c" * 64,
        kc_vocabulary_hash="d" * 64,
        model="irt",
        family="irt",
        kc_aware=False,
        recipe_hash_value="e" * 64,
        seed=42,
        seed_role="recorded_for_provenance_only",
        checkpoint_sha256="f" * 64,
        checkpoint_format="kclearner_ednet_streaming_state_v1",
        valid_metrics={"nll": 0.1, "brier": 0.2, "auc": 0.7},
        valid_global_metrics={"nll": 0.1, "brier": 0.2, "auc": 0.7},
        replay_required=True,
        tolerance=1e-12,
    )
    base.update(overrides)
    return base


def _checkpoint(directory: Path, text: str = "state") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "state_train_only.npz"
    path.write_bytes(text.encode("utf-8"))
    return path


def _write(directory: Path, identity: dict) -> dict:
    return write_freeze_manifest(directory, identity, {"device": "cpu", "path": "D:/tmp"})


class FreezeIdentityTests(unittest.TestCase):
    def test_same_identity_has_one_freeze_id(self) -> None:
        self.assertEqual(freeze_id_for(_identity()), freeze_id_for(_identity()))

    def test_scientific_changes_alter_the_freeze_id(self) -> None:
        original = freeze_id_for(_identity())
        self.assertNotEqual(original, freeze_id_for(_identity(dataset_logical_hash="9" * 64)))
        self.assertNotEqual(original, freeze_id_for(_identity(recipe_hash="1" * 64)))
        self.assertNotEqual(original, freeze_id_for(_identity(seed=43, seed_role="publication_seed")))
        self.assertNotEqual(original, freeze_id_for(_identity(checkpoint_sha256="2" * 64)))
        self.assertNotEqual(
            original,
            freeze_id_for(_identity(evaluation_semantics="evaluation_semantics_v2")),
        )

    def test_device_path_and_timestamp_do_not_alter_the_freeze_id(self) -> None:
        identity = _identity()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = write_freeze_manifest(
                root / "a",
                identity,
                {"device": "cpu", "path": "C:/one", "created_at": "t1"},
            )
            second = write_freeze_manifest(
                root / "b",
                identity,
                {"device": "cuda", "path": "D:/two", "created_at": "t2"},
            )
        self.assertEqual(first["freeze_id"], second["freeze_id"])
        self.assertNotIn("device", first["identity"])
        self.assertEqual(first["provenance"]["device"], "cpu")
        self.assertEqual(second["provenance"]["device"], "cuda")

    def test_incomplete_training_cannot_freeze(self) -> None:
        identity = _identity()
        identity["phases_completed"] = ["train"]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FreezeError):
                write_freeze_manifest(root, identity, {})
            self.assertFalse((root / MANIFEST_NAME).is_file())

    def test_existing_freeze_is_not_replaced(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            first = _write(root, _identity())
            again = _write(root, _identity())
            self.assertEqual(first["freeze_id"], again["freeze_id"])
            with self.assertRaises(FreezeError):
                _write(root, _identity(seed=43, seed_role="publication_seed"))
            stored = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
            self.assertEqual(stored["freeze_id"], first["freeze_id"])
            self.assertEqual(
                json.loads((root / "metadata.json").read_text(encoding="utf-8"))["freeze_id"],
                first["freeze_id"],
            )


class EvaluateGateTests(unittest.TestCase):
    def _material(self, root: Path, **overrides):
        checkpoint = _checkpoint(root)
        from kclearner.data.ednet_corrected import file_sha256

        identity = _identity(checkpoint_sha256=file_sha256(checkpoint), **overrides)
        manifest = _write(root, identity)
        return checkpoint, manifest

    def _run(self, root, manifest, checkpoint, *, enable=True, logical="a" * 64, recipe="e" * 64, seed=42, replay=None, test_rows=None, neural=None, valid_override=None):
        opened = []
        self.opened = opened

        def valid_rows():
            opened.append("valid")
            return ["valid"], [True]

        if valid_override is not None:
            valid_rows = valid_override

        def opened_test():
            opened.append("test")
            return ["test"], [True]

        def default_replay(rows, masks):
            def score(test_rows_value, test_masks):
                return {"nll": 0.2, "brier": 0.2, "auc": 0.6, "n": 1}
            score.headline = {"nll": 0.1, "brier": 0.2, "auc": 0.7}
            score.global_metrics = score.headline
            return score

        if neural is None:
            def neural_context():
                raise AssertionError("neural context is not used for this psychometric freeze")
        else:
            neural_context = neural
        return opened, run_evaluation(
            root / MANIFEST_NAME,
            enable_test=enable,
            dataset_logical_hash=logical,
            checkpoint_path=checkpoint,
            recipe_hash_value=recipe,
            seed=seed,
            output_dir=root / "test-out",
            valid_rows=valid_rows,
            test_rows=opened_test if test_rows is None else test_rows,
            replay=default_replay if replay is None else replay,
            neural_context=neural_context,
        )

    def test_missing_authorization_does_not_open_test(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, _manifest = self._material(root)
            with self.assertRaises(EvaluateError):
                self._run(root, _manifest, checkpoint, enable=False)
            self.assertEqual(self.opened, [])

    def test_missing_freeze_fails(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(EvaluateError):
                run_evaluation(
                    root / "missing.json",
                    enable_test=True,
                    dataset_logical_hash="a" * 64,
                    checkpoint_path=root / "missing.npz",
                    recipe_hash_value="e" * 64,
                    seed=42,
                    output_dir=root / "out",
                    valid_rows=lambda: ([], []),
                    test_rows=lambda: ([], []),
                    replay=lambda rows, masks: None,
                    neural_context=lambda: None,
                )

    def test_dataset_recipe_and_checkpoint_mismatches_fail_before_test(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, manifest = self._material(root)
            for kwargs in (
                {"logical": "b" * 64},
                {"recipe": "c" * 64},
                {"seed": 43},
            ):
                with self.assertRaises(EvaluateError):
                    self._run(root, manifest, checkpoint, **kwargs)
                self.assertEqual(self.opened, [])
            other = _checkpoint(root / "other", "different")
            with self.assertRaises(EvaluateError):
                self._run(root, manifest, other)
            self.assertEqual(self.opened, [])

    def test_replay_failure_blocks_test(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, manifest = self._material(root)

            def replay(rows, masks):
                def score(test_rows_value, test_masks):
                    raise AssertionError("TEST was scored after a failed replay")
                score.headline = {"nll": 0.1 + 1e-9, "brier": 0.2, "auc": 0.7}
                score.global_metrics = score.headline
                return score

            with self.assertRaises(EvaluateError):
                self._run(root, manifest, checkpoint, replay=replay)
            self.assertEqual(self.opened, ["valid"])

    def test_successful_replay_then_test_and_no_overwrite(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, manifest = self._material(root)
            opened, report = self._run(root, manifest, checkpoint)
            self.assertEqual(opened, ["valid", "test"])
            self.assertEqual(report["phase"], "test")
            self.assertEqual(report["role"], "evaluation")
            self.assertTrue(report["derived_from_frozen_experiment"])
            target = root / "test-out" / report["freeze_id"] / "test_result.json"
            original = target.read_text(encoding="utf-8")
            with self.assertRaises(EvaluateError):
                self._run(root, manifest, checkpoint)
            self.assertEqual(target.read_text(encoding="utf-8"), original)

    def test_neural_gate_has_no_valid_replay(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, manifest = self._material(
                root,
                model="dkt_q",
                family="dkt",
                kc_aware=False,
                seed=42,
                seed_role="publication_seed",
                valid_global_nll=None,
                valid_global_brier=None,
                valid_global_auc=None,
                valid_replay_tolerance=None,
                valid_replay_required=False,
                checkpoint_format="kclearner_neural_checkpoint_v1",
            )
            seen = []

            def reject_valid():
                raise AssertionError("neural TEST must not replay VALID metrics")

            def context():
                seen.append("context")

                def score(rows, masks):
                    return {"nll": 0.3, "brier": 0.2, "auc": 0.8, "n": 2}
                return score

            opened, report = self._run(
                root,
                manifest,
                checkpoint,
                neural=context,
                valid_override=reject_valid,
            )
            del opened
            self.assertEqual(seen, ["context"])
            self.assertEqual(report["phase"], "test")

    def test_ednet_tolerance_blocks_a_difference_above_1e_12(self) -> None:
        self.assertEqual(REPLAY_TOLERANCE["ednet_kt1_corrected_v1"], 1e-12)
        self.assertEqual(REPLAY_TOLERANCE["assistments2017"], 1e-9)
        expected = {"nll": 0.1, "brier": 0.2, "auc": 0.7}
        self.assertTrue(metrics_within_tolerance(expected, expected, 1e-12))
        shifted = {"nll": 0.1 + 1e-9, "brier": 0.2, "auc": 0.7}
        self.assertFalse(metrics_within_tolerance(shifted, expected, 1e-12))
        self.assertTrue(metrics_within_tolerance(shifted, expected, 1e-9))
        nan = {"nll": float("nan"), "brier": 0.2, "auc": 0.7}
        self.assertFalse(metrics_within_tolerance(nan, expected, 1e-9))


class DatasetSemanticsTests(unittest.TestCase):
    def test_nominal_count_is_not_the_metric_eligible_count(self) -> None:
        protocol = json.loads(
            (Path(__file__).resolve().parents[1] / "configs" / "assistments2017" / "protocol.json").read_text(
                encoding="utf-8"
            )
        )
        note = protocol["nominal_accepted_rows"]
        self.assertEqual(note["count"], 428495)
        self.assertEqual(protocol["dataset"]["row_counts"]["eligible"], FROZEN_MAIN_V1_COUNTS["eligible_rows"])
        self.assertEqual(FROZEN_MAIN_V1_COUNTS["metric_eligible_rows"], 427546)
        self.assertNotEqual(note["count"], FROZEN_MAIN_V1_COUNTS["metric_eligible_rows"])
        ednet = load_protocol(
            Path(__file__).resolve().parents[1] / "configs" / "ednet_corrected" / "protocol.json"
        )
        self.assertEqual(ednet["vocabulary"]["test_item_oov_rows"], 0)
        self.assertEqual(ednet["vocabulary"]["item_count"], 12259)

    def test_assistments_recipes_are_literals(self) -> None:
        protocol = load_protocol(
            Path(__file__).resolve().parents[1] / "configs" / "assistments2017" / "protocol.json",
            dataset="assistments2017",
        )
        self.assertEqual(protocol["publication_seed_policy"], "closed_set")
        self.assertEqual(protocol["models"]["dkt_q"]["recipe"]["emb_size"], 200)
        self.assertEqual(protocol["models"]["dkt_q"]["recipe"]["learning_rate"], 0.001)
        self.assertEqual(protocol["models"]["dkvmn_q"]["recipe"]["dim_s"], 64)
        self.assertEqual(protocol["models"]["dkvmn_q"]["recipe"]["size_m"], 20)
        self.assertEqual(protocol["models"]["ar_kt"]["recipe"]["residual_update"], "assistments_skill_mean")
        self.assertEqual(recipe_hash(protocol["models"]["dkt_q"]["recipe"]), recipe_hash(dict(protocol["models"]["dkt_q"]["recipe"])))
        for path in (Path(__file__).resolve().parents[1] / "src" / "kclearner").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("EdNetDKTConfig", text)
            self.assertNotIn("EdNetDKVMNConfig", text)
        with self.assertRaises(Exception):
            resolve_run(protocol, "dkt_q", 41)

    def test_family_oov_mapping_is_unchanged(self) -> None:
        from kclearner.data.adapters.assistments2017 import AssistmentsModelRow
        from kclearner.experiments.assistments_run import rows_for_execution

        kept = AssistmentsModelRow(0, 1, 0, 1, 1, 1, "test", True, False, "keep", "1", "2", "s")
        oov = AssistmentsModelRow(0, 0, 0, 2, 0, 2, "test", False, True, "oov", "1", "99", "s")
        self.assertEqual(len(rows_for_execution((kept, oov), "irt")), 2)
        self.assertEqual(len(rows_for_execution((kept, oov), "dkt_q")), 1)


@unittest.skipUnless(pyarrow is not None, "pyarrow is not installed")
class SyntheticFormalTests(unittest.TestCase):
    def test_dry_run_and_smoke_do_not_freeze_but_formal_does(self) -> None:
        from kclearner.data.assistments_prepare import prepare_assistments2017
        from kclearner.experiments.assistments_run import run_assistments_formal
        from kclearner.experiments.evaluate import evaluate_prepared
        from kclearner.experiments.runner import dry_run, smoke_run

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
        problems = [1, 2, 3, 4, 5, 6, 7, 8, 1, 1]
        body += "\n".join(
            f"1,{problems[index]},skill,{index % 2},{index + 1},1" for index in range(10)
        ) + "\n"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "primary.csv"
            source.write_text(body, encoding="utf-8")
            prepared = root / "prepared"
            prepare_assistments2017(source, prepared, protocol)
            config = resolve_run(protocol, "irt")
            dry_run(config, dataset_dir=prepared, runs_dir=root / "runs")
            self.assertFalse((root / "runs" / config.run_id / MANIFEST_NAME).exists())
            smoke_run(config, dataset_dir=prepared, runs_dir=root / "runs")
            self.assertFalse((root / "runs" / f"{config.run_id}_smoke" / MANIFEST_NAME).exists())
            report = run_assistments_formal(
                config,
                dataset_dir=prepared,
                runs_dir=root / "runs",
                requested_device="cpu",
            )
            freeze = root / "runs" / config.run_id / MANIFEST_NAME
            self.assertTrue(freeze.is_file())
            self.assertEqual(report["freeze_id"], json.loads(freeze.read_text(encoding="utf-8"))["freeze_id"])
            self.assertEqual(
                json.loads(freeze.read_text(encoding="utf-8"))["identity"]["evaluation_semantics"],
                EVALUATION_SEMANTICS,
            )
            with self.assertRaises(EvaluateError):
                evaluate_prepared(
                    freeze,
                    prepared,
                    root / "runs" / config.run_id / "state_train_only.npz",
                    root / "evaluated",
                    enable_test=False,
                    protocol=protocol,
                    model="irt",
                    seed=42,
                )
            self.assertFalse((root / "evaluated").exists())
            evaluated = evaluate_prepared(
                freeze,
                prepared,
                root / "runs" / config.run_id / "state_train_only.npz",
                root / "evaluated",
                enable_test=True,
                protocol=protocol,
                model="irt",
                seed=42,
            )
            stored = root / "evaluated" / evaluated["freeze_id"] / "test_result.json"
            text = stored.read_text(encoding="utf-8")
            with self.assertRaises(EvaluateError):
                evaluate_prepared(
                    freeze,
                    prepared,
                    root / "runs" / config.run_id / "state_train_only.npz",
                    root / "evaluated",
                    enable_test=True,
                    protocol=protocol,
                    model="irt",
                    seed=42,
                )
            self.assertEqual(stored.read_text(encoding="utf-8"), text)
