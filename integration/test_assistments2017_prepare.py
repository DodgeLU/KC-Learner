"""Optional ASSISTments MAIN_V1 prepare check.

Skipped unless the caller already has provider ``primary.csv``. The file
is read only and is not copied into the repository.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from kclearner.data.adapters.assistments2017 import FROZEN_MAIN_V1_COUNTS
from kclearner.data.assistments_prepare import prepare_assistments2017
from kclearner.experiments.config import load_protocol


def _primary() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "project_c"
        / "data_proc"
        / "assistments2017"
        / "primary.csv"
    )


def _ready() -> bool:
    return _primary().is_file()


@unittest.skipUnless(_ready(), "provider primary.csv is not available")
class AssistmentsFormalPrepareTests(unittest.TestCase):
    def test_frozen_counts_and_reuse(self) -> None:
        protocol = load_protocol(
            Path(__file__).resolve().parents[1] / "configs" / "assistments2017" / "protocol.json",
            dataset="assistments2017",
        )
        with TemporaryDirectory() as directory:
            output = Path(directory) / "prepared"
            first = prepare_assistments2017(_primary(), output, protocol)
            second = prepare_assistments2017(_primary(), output, protocol)
        frozen = FROZEN_MAIN_V1_COUNTS
        self.assertFalse(first.reused)
        self.assertTrue(second.reused)
        self.assertEqual(first.dataset_logical_hash, second.dataset_logical_hash)
        self.assertEqual(first.row_counts["train"], frozen["train_rows"])
        self.assertEqual(first.row_counts["valid"], frozen["valid_rows"])
        self.assertEqual(first.row_counts["test"], frozen["test_rows"])
        self.assertEqual(first.row_counts["eligible"], frozen["eligible_rows"])
