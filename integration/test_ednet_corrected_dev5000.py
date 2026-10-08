"""Build corrected dev5000 from raw KT1 and compare split membership.

Reads identity columns from the legacy split parquets. It does not
compute predictive metrics and it does not read response labels.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from kclearner.data.ednet_corrected import (
    build_corrected_rows,
    cohort_id_hash,
    diagnostics_from_build,
    reconstruct_dev5000_user_ids,
    user_ids_from_parquet,
    write_corrected_dataset,
)


def _project() -> Path:
    return Path(__file__).resolve().parents[2]


def _split_dir() -> Path:
    return (
        _project()
        / "project_c"
        / "data_proc"
        / "ednet_kt1"
        / "ednet_kt1_pilot_v1_from_v2_split_v1"
    )


def _ready() -> bool:
    root = _project() / "data"
    return (
        (root / "KT1").is_dir()
        and (root / "EdNet-Contents" / "contents" / "questions.csv").is_file()
        and (_split_dir() / "ednet_split_v1_train.parquet").is_file()
        and (_split_dir() / "ednet_split_v1_valid.parquet").is_file()
        and (_split_dir() / "ednet_split_v1_test.parquet").is_file()
    )


@unittest.skipUnless(_ready(), "raw KT1, questions.csv, or ednet_split_v1 parquet is absent")
class CorrectedDev5000Tests(unittest.TestCase):
    def test_membership_matches_legacy_identity(self) -> None:
        split_dir = _split_dir()
        users = reconstruct_dev5000_user_ids(
            user_ids_from_parquet(split_dir / "ednet_split_v1_train.parquet"),
            user_ids_from_parquet(split_dir / "ednet_split_v1_valid.parquet"),
        )
        self.assertEqual(len(users), 5000)
        self.assertEqual(users[0], "u100004")
        self.assertEqual(users[4999], "u171592")
        digest = cohort_id_hash(users)
        raw_root = _project() / "data" / "KT1"
        questions = _project() / "data" / "EdNet-Contents" / "contents" / "questions.csv"
        rows, stats = build_corrected_rows(raw_root, questions, users)
        report = diagnostics_from_build(rows, stats)
        self.assertEqual(report.noncontiguous_bundles, 0)
        self.assertEqual(report.solving_id_inversions, 0)
        self.assertEqual(report.timestamp_inversions, 0)
        output = _project() / "KC-Learner" / "generated" / "ednet_kt1_corrected_v1"
        write_corrected_dataset(
            rows,
            stats,
            output,
            cohort_user_ids=users,
            questions_path=questions,
            raw_root=raw_root,
        )
        mismatches = {}
        for split, filename in (
            ("train", "ednet_split_v1_train.parquet"),
            ("valid", "ednet_split_v1_valid.parquet"),
            ("test", "ednet_split_v1_test.parquet"),
        ):
            legacy = _legacy_identity(split_dir / filename, set(users))
            corrected = {
                (row.interaction.learner_id, row.source_row): (
                    row.interaction.item_id,
                    row.solving_id,
                    int(row.interaction.timestamp),
                )
                for row in rows
                if row.interaction.split == split
            }
            missing = sorted(set(legacy) - set(corrected))
            extra = sorted(set(corrected) - set(legacy))
            field = [
                key for key in set(legacy) & set(corrected)
                if legacy[key] != corrected[key]
            ]
            mismatches[split] = {
                "legacy": len(legacy),
                "corrected": len(corrected),
                "missing": len(missing),
                "extra": len(extra),
                "field": len(field),
                "missing_example": missing[:3],
                "extra_example": extra[:3],
            }
        print(
            "CORRECTED_EDNET "
            f"hash={digest} raw={report.raw_rows} eligible={report.eligible_rows} "
            f"filtered={report.filtered_rows} reasons={report.filter_reasons} "
            f"train={report.train_rows} valid={report.valid_rows} test={report.test_rows} "
            f"learners={report.learner_count} items={report.item_count} "
            f"kcs={report.real_kc_count} missing_kc={report.missing_kc_rows} "
            f"multi_kc={report.multi_kc_rows} duplicate_kc={report.duplicate_kc_rows} "
            f"bundles={report.bundle_count} "
            f"bundle_size={report.min_bundle_size}/{report.mean_bundle_size}/{report.max_bundle_size} "
            f"mismatches={mismatches}",
            flush=True,
        )
        for split, item in mismatches.items():
            self.assertEqual(item["missing"], 0, split)
            self.assertEqual(item["extra"], 0, split)
            self.assertEqual(item["field"], 0, split)


def _legacy_identity(path: Path, users: set[str]) -> dict[tuple[str, int], tuple[str, int, int]]:
    import pyarrow.parquet as pq

    table = pq.read_table(
        str(path),
        columns=["user_id", "source_row", "question_id", "solving_id", "timestamp_ms"],
    )
    found: dict[tuple[str, int], tuple[str, int, int]] = {}
    users_col = table.column("user_id").to_pylist()
    source_col = table.column("source_row").to_pylist()
    question_col = table.column("question_id").to_pylist()
    solving_col = table.column("solving_id").to_pylist()
    timestamp_col = table.column("timestamp_ms").to_pylist()
    for user_id, source_row, question_id, solving_id, timestamp in zip(
        users_col, source_col, question_col, solving_col, timestamp_col
    ):
        if user_id not in users:
            continue
        found[(str(user_id), int(source_row))] = (
            str(question_id),
            int(solving_id),
            int(timestamp),
        )
    return found


if __name__ == "__main__":
    unittest.main()
