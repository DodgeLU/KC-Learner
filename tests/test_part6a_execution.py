"""Part 6A execution-layer checks. No TRAIN/VALID/TEST data is read."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from kclearner.experiments.cli import main
from kclearner.experiments.config import ConfigError, load_protocol, resolve_run
from kclearner.experiments.environment import environment_report, execution_device
from kclearner.experiments.evaluation_semantics import EVALUATION_SEMANTICS
from kclearner.experiments.freeze import EVALUATION_SEMANTICS as FREEZE_SEMANTICS
from kclearner.experiments.neural_training import DeviceError, resolve_device


def _protocol(name: str, dataset: str) -> dict:
    path = Path(__file__).resolve().parents[1] / "configs" / name / "protocol.json"
    return load_protocol(path, dataset=dataset)


class SeedAndModeTests(unittest.TestCase):
    def test_formal_publication_seeds_are_accepted(self) -> None:
        ednet = _protocol("ednet_corrected", "ednet_kt1")
        assist = _protocol("assistments2017", "assistments2017")
        for seed in (42, 43, 44):
            run = resolve_run(ednet, "dkt_q", seed, publication_protocol=True)
            self.assertTrue(run.publication_protocol)
            self.assertEqual(run.seed_role, "publication_seed")
        for seed in (42, 43, 44, 45, 46):
            run = resolve_run(assist, "dkvmn_qc", seed, publication_protocol=True)
            self.assertTrue(run.publication_protocol)
            self.assertEqual(run.seed, seed)

    def test_exploratory_seed_cannot_claim_publication_protocol(self) -> None:
        assist = _protocol("assistments2017", "assistments2017")
        run = resolve_run(assist, "dkt_qc", 99, publication_protocol=False)
        self.assertEqual(run.seed, 99)
        self.assertEqual(run.seed_role, "exploratory_seed")
        self.assertFalse(run.publication_protocol)
        self.assertNotIn("publication_protocol\":true", run.canonical_json())
        with self.assertRaises(ConfigError):
            resolve_run(assist, "dkt_qc", 99, publication_protocol=True)
        ednet = _protocol("ednet_corrected", "ednet_kt1")
        with self.assertRaises(ConfigError):
            resolve_run(ednet, "dkt_q", 45, publication_protocol=True)

    def test_nonpublication_use_of_a_publication_seed_stays_unmarked(self) -> None:
        ednet = _protocol("ednet_corrected", "ednet_kt1")
        run = resolve_run(ednet, "dkt_q", 42, publication_protocol=False)
        self.assertEqual(run.seed_role, "publication_seed")
        self.assertFalse(run.publication_protocol)

    def test_conflicting_modes_and_removed_phase_are_rejected(self) -> None:
        with self.assertRaises(SystemExit) as conflict:
            main(["run", "--model", "irt", "--authorize-formal", "--dry-run"])
        self.assertEqual(conflict.exception.code, 2)
        with self.assertRaises(SystemExit) as phase:
            main(["run", "--model", "irt", "--phase", "test", "--dry-run"])
        self.assertEqual(phase.exception.code, 2)
        help_text = _help(["run", "--help"])
        self.assertNotIn("--phase", help_text)
        self.assertIn("--enable-test-execution", _help(["evaluate", "--help"]))
        self.assertNotIn("--authorize-formal", _help(["evaluate", "--help"]))

    def test_psychometric_cuda_request_stays_on_cpu(self) -> None:
        recorded = execution_device("irt", "cuda")
        self.assertEqual(recorded["requested_device"], "cuda")
        self.assertEqual(recorded["resolved_device"], "cpu")
        self.assertEqual(recorded["execution_mode"], "psychometric_cpu")
        same = execution_device("ar_kt", "cuda")
        self.assertEqual(same["resolved_device"], "cpu")

    def test_neural_cuda_request_fails_when_cuda_is_unavailable(self) -> None:
        try:
            import torch
        except Exception:
            torch = None
        if torch is not None and torch.cuda.is_available():
            self.skipTest("CUDA is available in this interpreter")
        with self.assertRaises(DeviceError):
            resolve_device("cuda")
        code = main(["run", "--model", "dkt_q", "--seed", "42", "--device", "cuda", "--dry-run"])
        self.assertEqual(code, 2)

    def test_unresolved_environment_does_not_claim_cuda(self) -> None:
        report = environment_report()
        self.assertEqual(report["resolved_device"], "cpu")
        self.assertEqual(report["device"], "cpu")
        explicit = environment_report("cuda", "cpu", execution_mode="psychometric_cpu")
        self.assertEqual(explicit["requested_device"], "cuda")
        self.assertEqual(explicit["resolved_device"], "cpu")
        self.assertNotEqual(explicit["device"], "cuda")

    def test_evaluation_semantics_version_is_central(self) -> None:
        self.assertEqual(EVALUATION_SEMANTICS, "evaluation_semantics_v1")
        self.assertEqual(FREEZE_SEMANTICS, EVALUATION_SEMANTICS)
        note = (
            Path(__file__).resolve().parents[1] / "docs" / "evaluation_semantics.md"
        ).read_text(encoding="utf-8")
        self.assertIn("evaluation_semantics_v1", note)
        self.assertIn("eligibility", note)


class DatasetIdentityRegressionTests(unittest.TestCase):
    def test_prepared_logical_identities_when_present(self) -> None:
        root = Path(__file__).resolve().parents[1] / "generated"
        expected = {
            "ednet_kt1_corrected_v1_public": (
                "a60371dab222a51e1ec5911edc916618fd18315d87a962224600fce3a1af6da2",
                (806148, 117184, 235031),
            ),
            "assistments2017_main_v1": (
                "de35b91ee7e80cae678da446626ec0c55b432287221b3c9deed53eac3374683d",
                (299907, 42847, 85741),
            ),
        }
        seen = False
        for folder, (digest, counts) in expected.items():
            path = root / folder / "logical_identity.json"
            if not path.is_file():
                continue
            seen = True
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["dataset_logical_hash"], digest)
            rows = {
                item["split_id"]: item["row_count"]
                for item in payload["dataset_logical_identity"]["splits"]
            }
            self.assertEqual(
                (rows["train"], rows["valid"], rows["test"]),
                counts,
            )
        if not seen:
            self.skipTest("prepared datasets are not in generated/")


def _help(argv: list[str]) -> str:
    from io import StringIO
    from contextlib import redirect_stdout

    buffer = StringIO()
    with self_raises_exit():
        with redirect_stdout(buffer):
            main(argv)
    return buffer.getvalue()


class _Exit(Exception):
    pass


def self_raises_exit():
    class _Catch:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return exc_type is SystemExit

    return _Catch()


if __name__ == "__main__":
    unittest.main()
