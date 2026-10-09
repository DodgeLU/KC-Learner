"""Confirm the vendored 50k sampler against local KT1 filenames.

Reads file stems only. It does not read interaction rows and it does
not write a user list.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from kclearner.data.ednet_pilot_sample import (
    HISTORICAL_DEV5000_SHA256,
    HISTORICAL_SAMPLE_SHA256,
    dev5000_from_pilot,
    select_ednet_pilot50k,
)
from kclearner.data.logical_identity import hash_ordered_ids


def _raw_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "KT1"


def _stems(root: Path) -> tuple[str, ...]:
    names: list[str] = []
    with os.scandir(root) as entries:
        for entry in entries:
            name = entry.name
            if name.startswith("u") and name.endswith(".csv"):
                names.append(name[:-4])
    names.sort()
    return tuple(names)


@unittest.skipUnless(_raw_root().is_dir(), "local KT1 filename directory is absent")
class EdNetPilotFilenameTests(unittest.TestCase):
    def test_historical_hashes(self) -> None:
        selected = select_ednet_pilot50k(_stems(_raw_root()))
        prefix = dev5000_from_pilot(selected)
        self.assertEqual(len(selected), 50000)
        self.assertEqual(len(prefix), 5000)
        self.assertEqual(hash_ordered_ids(selected), HISTORICAL_SAMPLE_SHA256)
        self.assertEqual(hash_ordered_ids(prefix), HISTORICAL_DEV5000_SHA256)
        self.assertEqual(prefix[0], "u100004")
        self.assertEqual(prefix[-1], "u171592")
