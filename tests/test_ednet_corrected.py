"""Public corrected-EdNet checks. No raw KT1 files."""

from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

from kclearner.data.adapters.ednet_kt1 import load_raw_user_kt1
from kclearner.data.ednet_corrected import (
    PREPROCESSING_VERSION,
    assign_solving_splits,
    cohort_id_hash,
    diagnostics_from_build,
    row_identity,
    select_first_sorted_users,
    sequence_invariants,
    split_bundle_counts,
    write_corrected_dataset,
)
from kclearner.data.schema import Interaction

_QUESTIONS = """question_id,bundle_id,correct_answer,tags
q1,b1,a,5;2;5
q2,b2,b,-1
q3,b3,c,9
"""


def _user(body: str) -> io.StringIO:
    header = "timestamp,solving_id,question_id,user_answer,elapsed_time\n"
    return io.StringIO(header + body)


def _interaction(
    learner: str,
    source_row: int,
    item: str,
    solving: int,
    timestamp: int,
    order: int,
    kc: tuple[int, ...] = (),
) -> Interaction:
    return Interaction(
        interaction_id=f"{learner}:{source_row}",
        learner_id=learner,
        item_id=item,
        correct=1,
        kc_ids=kc,
        order=order,
        timestamp=timestamp,
        bundle_id=f"{learner}|{solving}",
        split=None,
        metric_mask=True,
    )


class CorrectedEdNetTests(unittest.TestCase):
    def test_source_order_bundle_and_filters(self) -> None:
        # Later solving_id is written first. Eligible order must follow the file.
        body = """30,3,q1,a,1
10,1,q2,b,1
20,2,q3,-1,1
40,2,q3,c,1
50,4,q9,a,1
"""
        loaded = load_raw_user_kt1(_user(body), "u1", io.StringIO(_QUESTIONS))
        self.assertEqual(
            [row.interaction_id for row in loaded.interactions],
            ["u1:0", "u1:1", "u1:3"],
        )
        self.assertEqual(
            [row.bundle_id for row in loaded.interactions],
            ["u1|3", "u1|1", "u1|2"],
        )
        self.assertEqual(loaded.interactions[0].kc_ids, (5, 2, 5))
        self.assertEqual(loaded.interactions[1].kc_ids, ())
        self.assertEqual(loaded.stats.raw_rows, 5)
        self.assertEqual(loaded.stats.eligible_rows, 3)
        reasons = {item.reason: item.count for item in loaded.stats.dropped_reasons}
        self.assertEqual(reasons["user_answer_minus_one"], 1)
        self.assertEqual(reasons["missing_metadata"], 1)
        self.assertEqual(loaded.interactions[2].correct, 1)

    def test_same_solving_id_stays_one_bundle_in_source_order(self) -> None:
        body = """1,7,q1,a,1
2,7,q3,c,1
3,8,q2,b,1
"""
        loaded = load_raw_user_kt1(_user(body), "u2", io.StringIO(_QUESTIONS))
        self.assertEqual(
            [row.bundle_id for row in loaded.interactions],
            ["u2|7", "u2|7", "u2|8"],
        )
        self.assertEqual(loaded.interactions[0].order, 0)
        self.assertEqual(loaded.interactions[1].order, 1)

    def test_split_follows_numeric_solving_id_not_source_order(self) -> None:
        self.assertEqual(split_bundle_counts(1), (1, 0, 0))
        self.assertEqual(split_bundle_counts(2), (1, 0, 1))
        self.assertEqual(split_bundle_counts(6), (4, 1, 1))
        rows = (
            _interaction("u", 0, "q1", 9, 1, 0),
            _interaction("u", 1, "q1", 1, 2, 1),
            _interaction("u", 2, "q1", 2, 3, 2),
            _interaction("u", 3, "q1", 3, 4, 3),
            _interaction("u", 4, "q1", 4, 5, 4),
            _interaction("u", 5, "q1", 5, 6, 5),
        )
        labeled = assign_solving_splits(rows, [9, 1, 2, 3, 4, 5])
        by_solving = {
            int(row.bundle_id.split("|", 1)[1]): row.split for row in labeled
        }
        self.assertEqual(
            [row.interaction_id for row in labeled],
            [row.interaction_id for row in rows],
        )
        self.assertEqual(by_solving[1], "train")
        self.assertEqual(by_solving[4], "train")
        self.assertEqual(by_solving[5], "valid")
        self.assertEqual(by_solving[9], "test")

    def test_cohort_hash_and_selection_are_stable(self) -> None:
        selected = select_first_sorted_users(["u2", "u10", "u1", "u2"], 2)
        self.assertEqual(selected, ("u1", "u10"))
        first = cohort_id_hash(selected)
        self.assertEqual(first, cohort_id_hash(("u1", "u10")))
        self.assertNotEqual(first, cohort_id_hash(("u10", "u1")))
        self.assertEqual(len(first), 64)

    def test_identity_ignores_kc_text(self) -> None:
        left = row_identity("u1", 4, "q1", 7, 10)
        right = row_identity("u1", 4, "q1", 7, 10)
        self.assertEqual(left, right)
        self.assertNotEqual(left, row_identity("u1", 5, "q1", 7, 10))

    def test_invariants_and_manifest(self) -> None:
        from kclearner.data.ednet_corrected import CorrectedEdNetRow

        body = """1,1,q1,a,1
2,1,q3,b,1
3,2,q2,a,1
"""
        loaded = load_raw_user_kt1(_user(body), "u1", io.StringIO(_QUESTIONS))
        solving = [int(row.bundle_id.split("|", 1)[1]) for row in loaded.interactions]
        labeled = assign_solving_splits(loaded.interactions, solving)
        rows = tuple(
            CorrectedEdNetRow(
                interaction=row,
                source_row=int(row.interaction_id.rsplit(":", 1)[1]),
                solving_id=solving_id,
            )
            for row, solving_id in zip(labeled, solving)
        )
        invariants = sequence_invariants(rows)
        self.assertEqual(invariants.noncontiguous_bundles, 0)
        self.assertEqual(invariants.solving_id_inversions, 0)
        self.assertEqual(invariants.timestamp_inversions, 0)
        report = diagnostics_from_build(rows, (loaded.stats,))
        self.assertEqual(report.duplicate_kc_rows, 1)
        self.assertEqual(report.multi_kc_rows, 1)
        self.assertEqual(report.missing_kc_rows, 1)
        self.assertEqual(report.real_kc_count, 3)
        if importlib.util.find_spec("pyarrow") is None:
            self.skipTest("pyarrow is required to write corrected parquet")
        with tempfile.TemporaryDirectory() as directory:
            questions = Path(directory) / "questions.csv"
            questions.write_text(_QUESTIONS, encoding="utf-8")
            manifest = write_corrected_dataset(
                rows,
                (loaded.stats,),
                Path(directory) / "out",
                cohort_user_ids=("u1",),
                questions_path=questions,
                raw_root=Path(directory),
            )
            saved = json.loads((Path(directory) / "out" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["preprocessing_version"], PREPROCESSING_VERSION)
            self.assertEqual(saved["diagnostics"]["eligible_rows"], manifest["diagnostics"]["eligible_rows"])
            self.assertTrue((Path(directory) / "out" / "train.parquet").is_file())
