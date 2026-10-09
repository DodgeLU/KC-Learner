"""Formal external freeze and authorized TEST. No reference datasets."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from kclearner.experiments.external_evidence import (
    metric_token,
    probability_token,
    valid_prediction_trace_hash,
)
from kclearner.experiments.external_formal import (
    PHASE_INDEPENDENT,
    PHASE_REPLAY_VALID,
    TEST_FILE,
    TEST_RESULT_NAME,
    ExternalFormalError,
    FileRead,
    authorize_external_test,
    materialize_formal_directory,
    rebuild_identity_from_role_files,
    run_external_formal,
    write_external_freeze,
)
from kclearner.experiments.freeze import IDENTITY_VERSION
from test_external_execution import EXAMPLES, OOV_REPRESENTATION, _dataset


class RecordingRead(FileRead):
    def __init__(self) -> None:
        super().__init__()
        self.test_reads = 0

    def read_text(self, path: Path) -> str:
        if path.name == TEST_FILE:
            self.test_reads += 1
        return super().read_text(path)


def _enable() -> None:
    location = str(EXAMPLES)
    if location not in sys.path:
        sys.path.insert(0, location)


def _formal_tree(root: Path) -> Path:
    combined = root / "combined"
    formal = root / "formal"
    _dataset(
        combined,
        extra_rows=[
            {
                "interaction_id": "L1:4",
                "learner_id": "L1",
                "item_id": "Q1",
                "correct": 0,
                "kc_ids": [3],
                "order": 5,
                "timestamp": 10,
                "bundle_id": "L1|C",
                "split": "check",
                "metric_mask": False,
            },
            {
                "interaction_id": "L1:5",
                "learner_id": "L1",
                "item_id": "Q1",
                "correct": 0,
                "kc_ids": [3],
                "order": 6,
                "timestamp": 11,
                "bundle_id": "L1|D",
                "split": "check",
                "metric_mask": True,
            },
        ],
        representation=OOV_REPRESENTATION,
    )
    from kclearner.data.canonical_dataset import load_canonical_dataset

    loaded = load_canonical_dataset(combined)
    materialize_formal_directory(combined, formal)
    rebuilt_logical, rebuilt_contract = rebuild_identity_from_role_files(formal)
    if rebuilt_logical != loaded.dataset_logical_hash or rebuilt_contract != loaded.dataset_contract_hash:
        raise AssertionError("role files changed dataset identity")
    return formal


class ExternalFormalTests(unittest.TestCase):
    def test_reference_freeze_version_is_unchanged(self) -> None:
        self.assertEqual(IDENTITY_VERSION, "experiment_freeze_identity_v1")

    def test_trace_includes_masked_rows_and_rejects_nonfinite(self) -> None:
        full = (("L1:3", 0.25), ("L1:4", 0.25), ("L1:5", 1.0 / 6.0))
        omitted = (("L1:3", 0.25), ("L1:5", 1.0 / 6.0))
        self.assertNotEqual(valid_prediction_trace_hash(full), valid_prediction_trace_hash(omitted))
        self.assertEqual(metric_token(float("nan")), "null")
        with self.assertRaises(Exception):
            probability_token(float("nan"))

    def test_role_files_do_not_change_dataset_hashes(self) -> None:
        with TemporaryDirectory() as tmp:
            _formal_tree(Path(tmp))

    def test_formal_lifecycle_isolates_test_and_checkpoint(self) -> None:
        _enable()
        cases = (
            ("toy_streaming_adapter:ToyStreamingAdapter", {"oov_state_semantics_id": "retain_and_score_v1"}, "revealed"),
            ("toy_neural_adapter:ToyNeuralAdapter", {"oov_state_semantics_id": "retain_and_replay_v1", "seed": 7}, "trained"),
        )
        for spec, config, state_key in cases:
            for phase, expected in (
                (PHASE_INDEPENDENT, 0.25),
                (PHASE_REPLAY_VALID, 1.0 / 7.0),
            ):
                with self.subTest(spec=spec, phase=phase):
                    self._run_case(spec, config, state_key, phase, expected)

    def _run_case(self, spec, config, state_key, phase, expected) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            formal = _formal_tree(root)
            access = RecordingRead()
            run = run_external_formal(
                formal,
                spec,
                config,
                phase_state_semantics_id=phase,
                runs_dir=root / "runs",
                access=access,
            )
            self.assertNotIn(TEST_FILE, run.files_read)
            self.assertEqual(access.test_reads, 0)
            state = json.loads(run.checkpoint_path.read_text(encoding="utf-8"))
            self.assertEqual(state[state_key]["L1"], 3)
            frozen = write_external_freeze(run, formal, access=access)
            self.assertEqual(access.test_reads, 0)
            self.assertNotIn("module_path", json.dumps({
                key: value for key, value in frozen.items() if key != "descriptive_provenance"
            }))
            with self.assertRaises(Exception):
                authorize_external_test(
                    run.run_dir / "external_freeze_manifest.json",
                    formal,
                    spec,
                    config,
                    enable_test=False,
                    access=access,
                )
            self.assertEqual(access.test_reads, 0)
            tampered = dict(frozen)
            tampered["valid_prediction_trace_hash"] = "0" * 64
            tampered_path = run.run_dir / "tampered_freeze.json"
            tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaises(Exception):
                authorize_external_test(
                    tampered_path,
                    formal,
                    spec,
                    config,
                    enable_test=True,
                    access=access,
                )
            self.assertEqual(access.test_reads, 0)
            with self.assertRaises(Exception):
                authorize_external_test(
                    run.run_dir / "external_freeze_manifest.json",
                    formal,
                    "toy_streaming_adapter:MissingAdapter",
                    config,
                    enable_test=True,
                    access=access,
                )
            self.assertEqual(access.test_reads, 0)
            result = authorize_external_test(
                run.run_dir / "external_freeze_manifest.json",
                formal,
                spec,
                config,
                enable_test=True,
                access=access,
            )
            self.assertEqual(access.test_reads, 1)
            self.assertFalse(result["publication"])
            self.assertAlmostEqual(result["first_probability"], expected)
            self.assertEqual(json.loads(run.checkpoint_path.read_text(encoding="utf-8"))[state_key]["L1"], 3)

    def test_modified_test_file_fails_before_scoring(self) -> None:
        _enable()
        spec = "toy_streaming_adapter:ToyStreamingAdapter"
        config = {"oov_state_semantics_id": "retain_and_score_v1"}
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            formal = _formal_tree(root)
            run = run_external_formal(
                formal,
                spec,
                config,
                phase_state_semantics_id=PHASE_INDEPENDENT,
                runs_dir=root / "runs",
            )
            write_external_freeze(run, formal)
            test_path = formal / TEST_FILE
            original = test_path.read_text(encoding="utf-8")
            self.assertIn("L1:9", original)
            test_path.write_text(original.replace("L1:9", "L1:8", 1), encoding="utf-8")
            access = RecordingRead()
            with self.assertRaises(ExternalFormalError) as caught:
                authorize_external_test(
                    run.run_dir / "external_freeze_manifest.json",
                    formal,
                    spec,
                    config,
                    enable_test=True,
                    access=access,
                )
            message = str(caught.exception)
            self.assertIn("TEST split content mismatch", message)
            self.assertIn("expected logical_hash=", message)
            self.assertIn("observed logical_hash=", message)
            self.assertEqual(access.test_reads, 1)
            self.assertFalse((run.run_dir / TEST_RESULT_NAME).exists())

    def test_checkpoint_mutation_during_valid_replay_blocks_test(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            module_path = root / "checkpoint_mutating_adapter.py"
            module_path.write_text(_MUTATING_ADAPTER, encoding="utf-8")
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            formal = _formal_tree(root)
            spec = "checkpoint_mutating_adapter:CheckpointMutatingAdapter"
            config = {"oov_state_semantics_id": "retain_and_score_v1"}
            try:
                for phase in (PHASE_INDEPENDENT, PHASE_REPLAY_VALID):
                    with self.subTest(phase=phase):
                        run = run_external_formal(
                            formal,
                            spec,
                            config,
                            phase_state_semantics_id=phase,
                            runs_dir=root / "runs" / phase,
                        )
                        write_external_freeze(run, formal)
                        module_path.with_suffix(".mutate").write_text("1", encoding="utf-8")
                        access = RecordingRead()
                        with self.assertRaises(ExternalFormalError) as caught:
                            authorize_external_test(
                                run.run_dir / "external_freeze_manifest.json",
                                formal,
                                spec,
                                config,
                                enable_test=True,
                                access=access,
                            )
                        self.assertIn("TRAIN-only checkpoint", str(caught.exception))
                        self.assertEqual(access.test_reads, 0)
                        self.assertFalse((run.run_dir / TEST_RESULT_NAME).exists())
                        module_path.with_suffix(".mutate").unlink()
            finally:
                sys.modules.pop("checkpoint_mutating_adapter", None)
                if sys.path and sys.path[0] == str(root):
                    sys.path.pop(0)


_MUTATING_ADAPTER = '''
import json
from pathlib import Path

from kclearner.experiments.adapter_contract import (
    FINGERPRINT_SCOPE_SINGLE_MODULE,
    PSYCHOMETRIC_STREAMING_CONTRACT,
)

class CheckpointMutatingAdapter:
    """VALID replay appends one checkpoint byte when a sidecar flag exists."""

    contract_version = PSYCHOMETRIC_STREAMING_CONTRACT
    implementation_id = "checkpoint_mutating_v1"
    checkpoint_format_id = "checkpoint_mutating_state_v1"
    traversal_id = "recorded_split_order"
    execution_semantics_id = "checkpoint_mutating_v1"
    fingerprint_coverage = FINGERPRINT_SCOPE_SINGLE_MODULE
    supported_oov_representations = ("row_retained_in_canonical_sequence_v1",)
    supported_phase_state_semantics = (
        "independent_test_state_v1",
        "replay_valid_before_test_v1",
    )

    def __init__(self, config):
        oov = config.get("oov_state_semantics_id")
        if oov != "retain_and_score_v1":
            raise ValueError(oov)
        self.oov_state_semantics_id = oov
        self._checkpoint_path = None

    def validate_dataset_contract(self, dataset):
        if dataset.oov_representation_semantics_id not in self.supported_oov_representations:
            raise ValueError(dataset.oov_representation_semantics_id)

    def execute_phase(self, rows, *, phase):
        if phase not in ("train", "valid"):
            raise ValueError(phase)
        if phase == "valid" and Path(__file__).with_suffix(".mutate").is_file():
            target = self._checkpoint_path
            if target is not None:
                file = Path(target)
                file.write_bytes(file.read_bytes() + b"x")
        return tuple((row.interaction_id, 0.5) for row in rows)

    def save_state(self, path):
        Path(path).write_text(
            json.dumps({"format_id": self.checkpoint_format_id}),
            encoding="utf-8",
        )

    def load_state(self, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_id") != self.checkpoint_format_id:
            raise ValueError("checkpoint format_id does not match")
        self._checkpoint_path = str(path)
'''


if __name__ == "__main__":
    unittest.main()
