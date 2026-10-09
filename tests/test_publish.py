"""Synthetic publication races. No EdNet raw data."""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from kclearner.data.publish import (
    PublishError,
    discard_staging,
    make_staging_directory,
    publish_prepared_directory,
)


class PublishTests(unittest.TestCase):
    def test_staging_directories_are_unique_and_independent(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            destination = parent / "prepared"
            first = make_staging_directory(destination)
            second = make_staging_directory(destination)
            self.assertNotEqual(first, second)
            self.assertNotEqual(first, destination.with_name(destination.name + ".incomplete"))
            self.assertEqual(first.parent, destination.parent)
            (first / "only-first").write_text("a", encoding="utf-8")
            discard_staging(second)
            self.assertTrue((first / "only-first").is_file())
            self.assertFalse(second.exists())
            discard_staging(first)

    def test_missing_destination_is_published(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            destination = parent / "prepared"
            staging = make_staging_directory(destination)
            (staging / "marker.txt").write_text("ready", encoding="utf-8")
            outcome = publish_prepared_directory(
                staging,
                destination,
                lambda: "missing",
            )
            self.assertEqual(outcome, "prepared")
            self.assertEqual((destination / "marker.txt").read_text(encoding="utf-8"), "ready")
            self.assertFalse(staging.exists())

    def test_same_identity_is_reused_without_replacing_destination(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            destination = parent / "prepared"
            destination.mkdir()
            (destination / "keep.txt").write_text("winner", encoding="utf-8")
            staging = make_staging_directory(destination)
            (staging / "local.txt").write_text("loser", encoding="utf-8")
            sibling = make_staging_directory(destination)
            (sibling / "other.txt").write_text("other", encoding="utf-8")
            outcome = publish_prepared_directory(
                staging,
                destination,
                lambda: "reuse",
            )
            self.assertEqual(outcome, "reused")
            self.assertEqual((destination / "keep.txt").read_text(encoding="utf-8"), "winner")
            self.assertFalse((destination / "local.txt").exists())
            self.assertFalse(staging.exists())
            self.assertTrue((sibling / "other.txt").is_file())
            discard_staging(sibling)

    def test_incompatible_identity_does_not_overwrite(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            destination = parent / "prepared"
            destination.mkdir()
            (destination / "keep.txt").write_text("keep", encoding="utf-8")
            staging = make_staging_directory(destination)
            (staging / "local.txt").write_text("new", encoding="utf-8")
            with self.assertRaises(PublishError):
                publish_prepared_directory(
                    staging,
                    destination,
                    lambda: "incompatible",
                )
            self.assertEqual((destination / "keep.txt").read_text(encoding="utf-8"), "keep")
            self.assertFalse(staging.exists())
            self.assertFalse((destination / "local.txt").exists())

    def test_failed_rename_reuses_only_when_identity_matches(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            destination = parent / "prepared"
            destination.mkdir()
            (destination / "keep.txt").write_text("winner", encoding="utf-8")
            staging = make_staging_directory(destination)
            calls = {"n": 0}

            def assess() -> str:
                calls["n"] += 1
                if calls["n"] == 1:
                    return "missing"
                return "reuse"

            def explode(path: Path, target: Path) -> None:
                raise FileNotFoundError(2, "The system cannot find the file specified", str(path))

            with patch.object(Path, "rename", explode):
                outcome = publish_prepared_directory(staging, destination, assess)
            self.assertEqual(outcome, "reused")
            self.assertEqual((destination / "keep.txt").read_text(encoding="utf-8"), "winner")
            self.assertFalse(staging.exists())

    def test_failed_rename_does_not_treat_a_conflict_as_success(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            destination = parent / "prepared"
            destination.mkdir()
            (destination / "keep.txt").write_text("keep", encoding="utf-8")
            staging = make_staging_directory(destination)
            calls = {"n": 0}

            def assess() -> str:
                calls["n"] += 1
                if calls["n"] == 1:
                    return "missing"
                return "incompatible"

            def explode(path: Path, target: Path) -> None:
                raise FileNotFoundError(2, "The system cannot find the file specified", str(path))

            with patch.object(Path, "rename", explode):
                with self.assertRaises(PublishError):
                    publish_prepared_directory(staging, destination, assess)
            self.assertEqual((destination / "keep.txt").read_text(encoding="utf-8"), "keep")
            self.assertFalse(staging.exists())

    def test_discard_refuses_the_destination_name(self) -> None:
        with TemporaryDirectory() as directory:
            target = Path(directory) / "prepared"
            target.mkdir()
            with self.assertRaises(PublishError):
                discard_staging(target)
            self.assertTrue(target.is_dir())
            shutil.rmtree(target)
