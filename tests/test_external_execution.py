"""External adapter execution on a synthetic canonical directory."""

from __future__ import annotations

import inspect
import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from kclearner.experiments.adapter_contract import (
    NEURAL_BUNDLE_CONTRACT,
    freeze_eligible,
    load_adapter,
    require_neural_bundle_adapter,
    require_psychometric_streaming_adapter,
    source_mode_from_direct_url,
)
from kclearner.experiments.config import ALL_MODELS
from kclearner.experiments.external_execution import execute_external_directory
from kclearner.experiments.factory import create_model
from test_canonical_dataset import _declaration, _row, _write

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
OOV_STATE = "retain_and_score_v1"
OOV_REPRESENTATION = "row_retained_in_canonical_sequence_v1"


def _config(**extra: str) -> dict[str, str]:
    payload = {"oov_state_semantics_id": OOV_STATE}
    payload.update(extra)
    return payload


def _dataset(root: Path, *, extra_rows: list[dict[str, object]] | None = None, representation: str = OOV_REPRESENTATION) -> None:
    rows = [
            _row(
                interaction_id="L1:9",
                order=0,
                bundle_id="L1|test",
                split="later",
                correct=0,
            ),
            _row(interaction_id="L1:0", order=1, bundle_id="L1|A", split="fit", correct=1),
            _row(
                interaction_id="L1:1",
                order=2,
                bundle_id="L1|A",
                split="fit",
                correct=0,
                metric_mask=False,
            ),
            _row(interaction_id="L1:2", order=3, bundle_id="L1|B", split="fit", correct=0),
            _row(interaction_id="L1:3", order=4, bundle_id="L1|C", split="check", correct=1),
        ]
    if extra_rows:
        rows.extend(extra_rows)
    _write(
        root,
        declaration=_declaration(
            split_order=["later", "fit", "check"],
            split_roles={"later": "test", "fit": "train", "check": "valid"},
            oov_representation_semantics_id=representation,
        ),
        rows=rows,
        learners=["L1"],
        items=["Q1"],
        kcs=["3"],
    )


def _enable_example_import() -> None:
    location = str(EXAMPLES)
    if location not in sys.path:
        sys.path.insert(0, location)


class ExternalExecutionTests(unittest.TestCase):
    def test_external_adapter_runs_train_and_valid_without_the_registry(self) -> None:
        self.assertNotIn("ToyStreamingAdapter", ALL_MODELS)
        self.assertNotIn("toy_streaming_count_v1", ALL_MODELS)
        self.assertNotIn("ToyStreaming", inspect.getsource(create_model))
        _enable_example_import()
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            runs = Path(tmp) / "runs"
            _dataset(root)
            summary = execute_external_directory(
                root,
                "toy_streaming_adapter:ToyStreamingAdapter",
                _config(),
                runs_dir=runs,
            )
            document = json.loads(
                (Path(summary["output_dir"]) / "external_execution.json").read_text(encoding="utf-8")
            )
            state = json.loads(
                (Path(summary["output_dir"]) / "adapter_state.json").read_text(encoding="utf-8")
            )
        self.assertFalse(summary["publication"])
        self.assertFalse(summary["freeze"])
        self.assertFalse(summary["formal_test"])
        self.assertTrue(summary["formal_freeze_eligible"])
        self.assertTrue(summary["model_execution"])
        self.assertEqual(summary["traversal_id"], "recorded_split_order")
        self.assertFalse((Path(summary["output_dir"]) / "freeze_manifest.json").exists())
        train = document["train"]["rows"]
        self.assertEqual([row["interaction_id"] for row in train], ["L1:0", "L1:1", "L1:2"])
        self.assertEqual(train[0]["probability"], train[1]["probability"])
        self.assertNotEqual(train[0]["probability"], train[2]["probability"])
        self.assertEqual(train[0]["bundle_id"], "L1|A")
        self.assertEqual(document["train"]["metrics"]["n"], 2)
        self.assertTrue(any(row["interaction_id"] == "L1:1" and row["metric_mask"] is False for row in train))
        self.assertEqual(document["valid"]["rows"][0]["interaction_id"], "L1:3")
        self.assertEqual(document["valid"]["metrics"]["n"], 1)
        self.assertNotIn("L1:9", json.dumps(document["train"]))
        self.assertNotIn("L1:9", json.dumps(document["valid"]))
        self.assertEqual(state["format_id"], "toy_streaming_count_state_v1")
        self.assertNotIn(str(EXAMPLES), json.dumps(document["provenance_identity"]))
        self.assertIn(str(EXAMPLES), document["descriptive_provenance"]["module_path"])
        self.assertEqual(document["oov_representation_semantics_id"], OOV_REPRESENTATION)
        self.assertEqual(document["execution_semantics_id"], "toy_streaming_learner_count_v1")
        self.assertEqual(document["descriptive_provenance"]["source_mode"], "local")
        self.assertEqual(
            document["provenance_identity"]["implementation_fingerprint_scope"],
            "single_module",
        )
        self.assertIsNotNone(document["provenance_identity"]["implementation_fingerprint_value"])
        self.assertEqual(document["config_hash"], document["provenance_identity"]["config_hash"])
        self.assertEqual(
            document["provenance_identity"]["oov_state_semantics_id"],
            OOV_STATE,
        )

    def test_config_identity_changes_provenance_only(self) -> None:
        _enable_example_import()
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            left = base / "left"
            right = base / "right"
            _dataset(left)
            _dataset(right)
            first = execute_external_directory(
                left, "toy_streaming_adapter:ToyStreamingAdapter", _config(marker="a"), runs_dir=base / "runs"
            )
            second = execute_external_directory(
                right, "toy_streaming_adapter:ToyStreamingAdapter", _config(marker="b"), runs_dir=base / "runs"
            )
        self.assertEqual(first["dataset_logical_hash"], second["dataset_logical_hash"])
        self.assertEqual(first["dataset_contract_hash"], second["dataset_contract_hash"])
        self.assertNotEqual(first["provenance_hash"], second["provenance_hash"])
        self.assertNotEqual(first["output_dir"], second["output_dir"])

    def test_unversioned_adapter_without_fingerprint_is_ineligible(self) -> None:
        module = types.ModuleType("ephemeral_toy_adapter")
        module.__file__ = None
        exec(_EPHEMERAL_SOURCE, module.__dict__)
        sys.modules["ephemeral_toy_adapter"] = module
        try:
            with TemporaryDirectory() as tmp:
                root = Path(tmp) / "data"
                _dataset(root)
                summary = execute_external_directory(
                    root,
                    "ephemeral_toy_adapter:ToyStreamingAdapter",
                    _config(),
                    runs_dir=Path(tmp) / "runs",
                )
        finally:
            sys.modules.pop("ephemeral_toy_adapter", None)
        self.assertFalse(summary["publication"])
        self.assertFalse(summary["formal_freeze_eligible"])
        self.assertIsNotNone(summary["train_metrics"])

    def test_execution_semantics_id_is_required_and_changes_provenance(self) -> None:
        _enable_example_import()
        from toy_streaming_adapter import ToyStreamingAdapter
        from kclearner.experiments.adapter_contract import adapter_provenance

        config = _config()
        original = ToyStreamingAdapter(config)
        changed = ToyStreamingAdapter(config)
        changed.execution_semantics_id = "other_phase_semantics_v1"
        first = adapter_provenance(original, config)
        second = adapter_provenance(changed, config)
        self.assertNotEqual(first.provenance_hash, second.provenance_hash)
        self.assertNotIn("source_mode", first.identity)
        blank = ToyStreamingAdapter(config)
        blank.execution_semantics_id = ""
        with self.assertRaises(Exception):
            require_psychometric_streaming_adapter(blank)

    def test_incompatible_oov_representation_fails_before_train(self) -> None:
        _enable_example_import()
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            runs = Path(tmp) / "runs"
            _dataset(root, representation="unknown_representation_v1")
            with self.assertRaises(Exception) as caught:
                execute_external_directory(
                    root,
                    "toy_streaming_adapter:ToyStreamingAdapter",
                    _config(),
                    runs_dir=runs,
                )
            self.assertIn("TRAIN was not started", str(caught.exception))
            self.assertEqual(list(runs.glob("**/external_execution.json")), [])

    def test_install_mode_eligibility(self) -> None:
        self.assertEqual(
            source_mode_from_direct_url(
                None, has_distribution=True, has_version=True
            ),
            "versioned_non_editable",
        )
        self.assertEqual(
            source_mode_from_direct_url(
                {"dir_info": {"editable": True}},
                has_distribution=True,
                has_version=True,
            ),
            "editable",
        )
        self.assertTrue(freeze_eligible(
            source_mode="versioned_non_editable",
            distribution_version="1.2.3",
            fingerprint_kind="installed_distribution",
            fingerprint_scope="distribution_version",
            fingerprint_value=None,
        ))
        self.assertFalse(freeze_eligible(
            source_mode="editable",
            distribution_version="1.2.3",
            fingerprint_kind="module_source_sha256",
            fingerprint_scope="single_module",
            fingerprint_value=None,
        ))
        self.assertTrue(freeze_eligible(
            source_mode="editable",
            distribution_version="1.2.3",
            fingerprint_kind="module_source_sha256",
            fingerprint_scope="single_module",
            fingerprint_value="abc",
        ))
        self.assertFalse(freeze_eligible(
            source_mode="local",
            distribution_version=None,
            fingerprint_kind=None,
            fingerprint_scope=None,
            fingerprint_value="abc",
        ))
        self.assertFalse(freeze_eligible(
            source_mode="unversioned",
            distribution_version=None,
            fingerprint_kind=None,
            fingerprint_scope=None,
            fingerprint_value=None,
        ))

    def test_file_fingerprint_makes_local_adapter_eligible(self) -> None:
        source = (EXAMPLES / "toy_streaming_adapter.py").read_text(encoding="utf-8")
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            module_path = folder / "local_fingerprinted_adapter.py"
            module_path.write_text(source, encoding="utf-8")
            sys.path.insert(0, str(folder))
            try:
                from kclearner.experiments.adapter_contract import adapter_provenance
                from local_fingerprinted_adapter import ToyStreamingAdapter

                provenance = adapter_provenance(ToyStreamingAdapter(_config()), _config())
            finally:
                sys.path.remove(str(folder))
                sys.modules.pop("local_fingerprinted_adapter", None)
        self.assertEqual(provenance.source_mode, "local")
        self.assertEqual(provenance.identity["implementation_fingerprint_scope"], "single_module")
        self.assertIsNotNone(provenance.identity["implementation_fingerprint_value"])
        self.assertTrue(provenance.formal_freeze_eligible)
        self.assertIsNone(provenance.identity["distribution_version"])

    def test_neural_adapter_replays_valid_bundles_and_skips_test(self) -> None:
        self.assertNotIn("ToyNeuralAdapter", ALL_MODELS)
        self.assertNotIn("toy_neural_bundle_count_v1", inspect.getsource(create_model))
        self.assertNotIn("lstm_layer", inspect.getsource(
            __import__("kclearner.experiments.adapter_contract", fromlist=["require_neural_bundle_adapter"]).require_neural_bundle_adapter
        ))
        _enable_example_import()
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            runs = Path(tmp) / "runs"
            _dataset(
                root,
                extra_rows=[
                    _row(
                        interaction_id="L1:4",
                        order=5,
                        bundle_id="L1|C",
                        split="check",
                        correct=0,
                        metric_mask=False,
                    ),
                    _row(interaction_id="L1:5", order=6, bundle_id="L1|D", split="check", correct=1),
                ],
            )
            summary = execute_external_directory(
                root,
                "toy_neural_adapter:ToyNeuralAdapter",
                {"oov_state_semantics_id": "retain_and_replay_v1", "seed": 7},
                runs_dir=runs,
            )
            document = json.loads(
                (Path(summary["output_dir"]) / "external_execution.json").read_text(encoding="utf-8")
            )
            state = json.loads(
                (Path(summary["output_dir"]) / "adapter_state.json").read_text(encoding="utf-8")
            )
        self.assertEqual(document["adapter_contract_version"], NEURAL_BUNDLE_CONTRACT)
        self.assertEqual(document["execution_semantics_id"], "toy_neural_bundle_replay_v1")
        self.assertFalse(document["publication"])
        self.assertFalse(document["freeze"])
        self.assertFalse(document["formal_test"])
        valid = document["valid"]["rows"]
        self.assertEqual([row["interaction_id"] for row in valid], ["L1:3", "L1:4", "L1:5"])
        self.assertEqual(valid[0]["probability"], valid[1]["probability"])
        self.assertNotEqual(valid[0]["probability"], valid[2]["probability"])
        self.assertEqual(document["valid"]["metrics"]["n"], 2)
        self.assertNotIn("L1:9", json.dumps(document["train"]))
        self.assertNotIn("L1:9", json.dumps(document["valid"]))
        self.assertEqual(state["trained"], {"L1": 3})
        self.assertEqual(state["seed"], 7)
        self.assertFalse(hasattr(
            __import__("toy_neural_adapter", fromlist=["ToyNeuralAdapter"]).ToyNeuralAdapter,
            "lstm_layer",
        ))
        self.assertFalse(hasattr(
            __import__("toy_neural_adapter", fromlist=["ToyNeuralAdapter"]).ToyNeuralAdapter,
            "Mv0",
        ))

    def test_neural_contract_does_not_require_pykt_fields(self) -> None:
        require_neural_bundle_adapter(_NeuralShape())
        self.assertFalse(hasattr(_NeuralShape, "lstm_layer"))
        self.assertFalse(hasattr(_NeuralShape, "Mv0"))
        incomplete = _NeuralShape()
        incomplete.replay = None  # type: ignore[method-assign]
        with self.assertRaises(Exception):
            require_neural_bundle_adapter(incomplete)

    def test_loader_rejects_a_missing_oov_declaration(self) -> None:
        _enable_example_import()
        with self.assertRaises(Exception):
            load_adapter("toy_streaming_adapter:ToyStreamingAdapter", {})


class _NeuralShape:
    contract_version = NEURAL_BUNDLE_CONTRACT
    implementation_id = "shape_v1"
    checkpoint_format_id = "shape_state_v1"
    traversal_id = "adapter_defined_v1"
    execution_semantics_id = "shape_replay_v1"
    oov_state_semantics_id = OOV_STATE

    def train(self, rows):
        return ()

    def replay(self, rows):
        return ()

    def save_checkpoint(self, path):
        return None

    def load_checkpoint(self, path):
        return None

    def validate_dataset_contract(self, dataset):
        return None


_EPHEMERAL_SOURCE = """
from kclearner.experiments.adapter_contract import PSYCHOMETRIC_STREAMING_CONTRACT

class ToyStreamingAdapter:
    contract_version = PSYCHOMETRIC_STREAMING_CONTRACT
    implementation_id = "ephemeral_count_v1"
    checkpoint_format_id = "ephemeral_state_v1"
    traversal_id = "recorded_split_order"
    execution_semantics_id = "ephemeral_count_v1"
    supported_oov_representations = ("row_retained_in_canonical_sequence_v1",)

    def __init__(self, config):
        oov = config.get("oov_state_semantics_id")
        if oov != "retain_and_score_v1":
            raise ValueError(oov)
        self.oov_state_semantics_id = oov
        self._revealed = {}

    def validate_dataset_contract(self, dataset):
        if dataset.oov_representation_semantics_id not in self.supported_oov_representations:
            raise ValueError(dataset.oov_representation_semantics_id)

    def execute_phase(self, rows, *, phase):
        if phase not in ("train", "valid"):
            raise ValueError(phase)
        output = []
        for row in rows:
            pre = self._revealed.get(row.learner_id, 0)
            output.append((row.interaction_id, 1.0 / (1.0 + pre)))
            self._revealed[row.learner_id] = pre + 1
        return tuple(output)

    def save_state(self, path):
        from pathlib import Path
        Path(path).write_text("{}", encoding="utf-8")

    def load_state(self, path):
        return None
"""


if __name__ == "__main__":
    unittest.main()
