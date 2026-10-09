"""Synthetic ASSISTments preparation. The publication CSV is not read."""

from __future__ import annotations

import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from kclearner.data.adapters.assistments2017 import (
    FROZEN_MAIN_V1_COUNTS,
    load_assistments2017_main_v1,
)
from kclearner.data.assistments_prepare import (
    AssistmentsPrepareError,
    logical_identity_for,
    prepare_assistments2017,
)
from kclearner.data.logical_identity import dataset_logical_hash
from kclearner.experiments.config import load_protocol

try:
    import pyarrow
except Exception:
    pyarrow = None


_HEADER = "student_id,problem_id,skill_id,correct,timestamp,attempt_count\n"


def _body(count: int = 10) -> str:
    lines = [
        f"1,{index + 1},skill,{index % 2},{index + 1},1"
        for index in range(count)
    ]
    return _HEADER + "\n".join(lines) + "\n"


def _synthetic_protocol() -> dict:
    return {
        "protocol_id": "ASSISTMENTS2017_MAIN_V1",
        "dataset": {
            "dataset_id": "assistments2017",
            "preprocessing_version": "assist17_primary_v1",
            "enforce_frozen_counts": False,
        },
    }


def _formal_protocol() -> dict:
    path = Path(__file__).resolve().parents[1] / "configs" / "assistments2017" / "protocol.json"
    return load_protocol(path, dataset="assistments2017")


class FormalProtocolTests(unittest.TestCase):
    def test_frozen_counts_match_the_adapter_constant(self) -> None:
        protocol = _formal_protocol()
        counts = protocol["dataset"]["row_counts"]
        frozen = FROZEN_MAIN_V1_COUNTS
        self.assertEqual(counts["train"], frozen["train_rows"])
        self.assertEqual(counts["valid"], frozen["valid_rows"])
        self.assertEqual(counts["test"], frozen["test_rows"])
        self.assertEqual(counts["eligible"], frozen["eligible_rows"])
        self.assertEqual(protocol["dataset"]["learners"], frozen["n_students_vocab"])
        self.assertEqual(protocol["dataset"]["train_valid_problems"], frozen["n_questions_vocab"])
        self.assertEqual(protocol["dataset"]["kcs"], frozen["n_skills_vocab"])
        self.assertEqual(protocol["dataset"]["nominal_test_item_oov"], frozen["oov_rows"])
        self.assertEqual(
            protocol["dataset"]["metric_eligible_test"],
            frozen["test_rows"] - frozen["oov_rows"],
        )
        self.assertTrue(protocol["dataset"]["enforce_frozen_counts"])
        self.assertEqual(protocol["phases"], ["train", "valid"])
        self.assertEqual(
            protocol["vocabulary"]["item_ids_sha256"],
            "3717448c8a379d4608e821ca4a78b5ff59c9e5af709ef0c4cb4d98e5de52969a",
        )
        self.assertEqual(
            protocol["vocabulary"]["kc_ids_sha256"],
            "230fa310cf1f1cb351f99b26a6f338337f228db4ababf908820cad343734edb8",
        )
        self.assertEqual(
            protocol["vocabulary"]["learner_ids_sha256"],
            "58a3aafd1f8895d0f506c8baa6bdaa17cfad6ddd67cb0dd4a2bbe813ada0ebfa",
        )

    def test_small_csv_fails_the_formal_count_gate(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "primary.csv"
            source.write_text(_body(), encoding="utf-8")
            with self.assertRaises(AssistmentsPrepareError):
                prepare_assistments2017(source, root / "out", _formal_protocol())
            self.assertFalse((root / "out").exists())


class IdentityTests(unittest.TestCase):
    def test_model_settings_do_not_change_the_dataset_hash(self) -> None:
        loaded = load_assistments2017_main_v1(io.StringIO(_body()))
        left_protocol = _synthetic_protocol()
        left_protocol.update({
            "device": "cpu",
            "models": {"irt": {"recipe": {"theta_lr": 0.05, "seed": 42}}},
        })
        right_protocol = _synthetic_protocol()
        right_protocol.update({
            "device": "cuda",
            "models": {"dkt_qc": {"recipe": {"learning_rate": 99, "seed": 46, "epochs": 3}}},
        })
        left = dataset_logical_hash(logical_identity_for(loaded, left_protocol))
        right = dataset_logical_hash(logical_identity_for(loaded, right_protocol))
        self.assertEqual(left, right)
        changed = _body().replace("1,1,skill,0,1,1", "1,1,skill,1,1,1", 1)
        other = load_assistments2017_main_v1(io.StringIO(changed))
        self.assertNotEqual(
            left,
            dataset_logical_hash(logical_identity_for(other, left_protocol)),
        )


@unittest.skipUnless(pyarrow is not None, "pyarrow is not installed")
class SyntheticPrepareTests(unittest.TestCase):
    def test_prepare_reuses_and_does_not_require_ednet_fields(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "primary.csv"
            source.write_text(_body(), encoding="utf-8")
            output = root / "prepared"
            first = prepare_assistments2017(source, output, _synthetic_protocol())
            second = prepare_assistments2017(source, output, _synthetic_protocol())
            self.assertFalse(first.reused)
            self.assertTrue(second.reused)
            self.assertEqual(first.dataset_logical_hash, second.dataset_logical_hash)
            self.assertFalse((output / "dev5000_users.txt").exists())
            import pyarrow.parquet as pq

            schema = pq.read_schema(output / "train.parquet")
            self.assertNotIn("solving_id", schema.names)
            self.assertIn("group_id", schema.names)
            self.assertIn("source_row", schema.names)
            self.assertTrue((output / "logical_identity.json").is_file())
            self.assertTrue((output / "vocabulary.json").is_file())
            logical = json.loads((output / "logical_identity.json").read_text(encoding="utf-8"))
            self.assertEqual(logical["dataset_logical_hash"], first.dataset_logical_hash)
            source.write_text(
                _body().replace("1,1,skill,0,1,1", "1,1,skill,1,1,1", 1),
                encoding="utf-8",
            )
            with self.assertRaises(AssistmentsPrepareError):
                prepare_assistments2017(source, output, _synthetic_protocol())
            self.assertEqual(
                json.loads((output / "logical_identity.json").read_text(encoding="utf-8"))[
                    "dataset_logical_hash"
                ],
                first.dataset_logical_hash,
            )
