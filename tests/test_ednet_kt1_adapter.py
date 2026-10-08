"""Synthetic EdNet-KT1 join. No dev5000 files."""

from __future__ import annotations

import io
import unittest

from kclearner.data.adapters.base import ensure_bundle_id
from kclearner.data.adapters.ednet_kt1 import (
    EdNetAdapterError,
    EdNetKT1LoadResult,
    load_ednet_kt1_interactions,
)
from kclearner.data.adapters.ednet_metadata import EdNetTagParseError
from kclearner.data.schema import Interaction, prediction_target_count
from kclearner.data.validation import DatasetValidator

_QUESTIONS = """\
question_id,bundle_id,explanation_id,correct_answer,part,tags
q1,content-b,e1,a,1,5;2;182
q2,content-b,e2,b,1,-1
q3,content-c,e3,c,2,7
"""

_INTERACTIONS = """\
user_id,timestamp,solving_id,question_id,user_answer,elapsed_time
u1,1000,10,q1,a,500
u1,1000,10,q2,a,400
u1,2000,11,q3,b,300
u2,1500,10,q1,c,100
"""


def _load(
    interactions: str = _INTERACTIONS,
    questions: str = _QUESTIONS,
) -> EdNetKT1LoadResult:
    return load_ednet_kt1_interactions(
        io.StringIO(interactions),
        io.StringIO(questions),
    )


class BundleAssignmentTests(unittest.TestCase):
    def test_missing_source_bundle_uses_interaction_id(self) -> None:
        self.assertEqual(ensure_bundle_id(None, "u1:0"), "u1:0")
        self.assertEqual(ensure_bundle_id("  ", "u1:0"), "u1:0")
        self.assertEqual(ensure_bundle_id("u1|10", "u1:0"), "u1|10")


class EdNetKT1AdapterTests(unittest.TestCase):
    def test_join_builds_canonical_interactions(self) -> None:
        loaded = _load()
        self.assertEqual(loaded.stats.raw_rows, 4)
        self.assertEqual(loaded.stats.eligible_rows, 4)
        self.assertEqual(loaded.stats.dropped_invalid_user_answer, 0)
        self.assertEqual(loaded.stats.dropped_missing_metadata, 0)
        rows = loaded.interactions
        self.assertEqual(len(rows), 4)
        self.assertEqual(prediction_target_count(rows), 4)
        self.assertTrue(all(isinstance(row, Interaction) for row in rows))
        self.assertTrue(DatasetValidator().validate(rows).ok)

        by_id = {row.interaction_id: row for row in rows}
        first = by_id["u1:0"]
        second = by_id["u1:1"]
        third = by_id["u1:2"]
        other = by_id["u2:3"]

        self.assertEqual(first.learner_id, "u1")
        self.assertEqual(first.item_id, "q1")
        self.assertEqual(first.correct, 1)
        self.assertEqual(first.timestamp, 1000)
        self.assertEqual(first.order, 0)
        self.assertEqual(first.kc_ids, (5, 2, 182))
        self.assertNotEqual(first.bundle_id, "content-b")

        self.assertEqual(second.kc_ids, ())
        self.assertEqual(second.correct, 0)
        self.assertEqual(second.order, 1)
        self.assertEqual(first.bundle_id, second.bundle_id)
        self.assertEqual(first.bundle_id, "u1|10")

        self.assertEqual(third.bundle_id, "u1|11")
        self.assertNotEqual(third.bundle_id, first.bundle_id)
        self.assertEqual(third.item_id, "q3")
        self.assertEqual(third.kc_ids, (7,))
        self.assertEqual(third.correct, 0)
        self.assertEqual(third.order, 2)
        self.assertEqual(third.timestamp, 2000)

        self.assertEqual(other.learner_id, "u2")
        self.assertEqual(other.bundle_id, "u2|10")
        self.assertNotEqual(other.bundle_id, first.bundle_id)
        self.assertEqual(other.correct, 0)
        self.assertEqual(other.order, 0)

    def test_multi_kc_stays_one_interaction(self) -> None:
        rows = [row for row in _load().interactions if row.item_id == "q1"]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row.kc_ids == (5, 2, 182) for row in rows))

    def test_missing_question_metadata_is_counted_not_emitted(self) -> None:
        interactions = _INTERACTIONS + "u1,3000,12,q404,a,10\n"
        loaded = _load(interactions=interactions)
        self.assertEqual(loaded.stats.raw_rows, 5)
        self.assertEqual(loaded.stats.eligible_rows, 4)
        self.assertEqual(loaded.stats.dropped_missing_metadata, 1)
        self.assertTrue(
            all(row.item_id != "q404" for row in loaded.interactions)
        )
        reasons = {item.reason: item.count for item in loaded.stats.dropped_reasons}
        self.assertEqual(reasons["missing_metadata"], 1)

    def test_invalid_kc_metadata_raises(self) -> None:
        questions = _QUESTIONS + "q4,content-d,e4,a,1,5;foo\n"
        with self.assertRaises(EdNetTagParseError) as caught:
            _load(questions=questions)
        message = str(caught.exception)
        self.assertIn("question_id='q4'", message)
        self.assertIn("raw_tags='5;foo'", message)
        self.assertIn("invalid_token='foo'", message)

    def test_unanswered_marker_is_filtered_not_scored_incorrect(self) -> None:
        interactions = """\
user_id,timestamp,solving_id,question_id,user_answer
u1,1000,10,q1,a
u1,1100,10,q2,-1
u1,1200,11,q404,b
"""
        loaded = _load(interactions=interactions)
        self.assertEqual(loaded.stats.raw_rows, 3)
        self.assertEqual(loaded.stats.eligible_rows, 1)
        self.assertEqual(loaded.stats.dropped_invalid_user_answer, 1)
        self.assertEqual(loaded.stats.dropped_missing_metadata, 1)
        self.assertEqual(
            [(item.reason, item.count) for item in loaded.stats.dropped_reasons],
            [("user_answer_minus_one", 1), ("missing_metadata", 1)],
        )
        self.assertEqual(len(loaded.interactions), 1)
        self.assertEqual(loaded.interactions[0].item_id, "q1")
        self.assertEqual(loaded.interactions[0].correct, 1)
        self.assertTrue(all(row.correct in (0, 1) for row in loaded.interactions))
        self.assertNotIn("q2", {row.item_id for row in loaded.interactions})

    def test_source_order_is_preserved_when_solving_id_and_timestamp_invert(self) -> None:
        interactions = """\
user_id,timestamp,solving_id,question_id,user_answer
u2,50,9,q1,a
u1,500,2,q1,a
u1,100,1,q2,b
u1,400,2,q3,c
"""
        loaded = _load(interactions=interactions)
        rows = loaded.interactions
        self.assertEqual(
            [(row.learner_id, row.item_id, row.timestamp, row.bundle_id, row.order) for row in rows],
            [
                ("u2", "q1", 50, "u2|9", 0),
                ("u1", "q1", 500, "u1|2", 0),
                ("u1", "q2", 100, "u1|1", 1),
                ("u1", "q3", 400, "u1|2", 2),
            ],
        )
        self.assertEqual(rows[1].bundle_id, rows[3].bundle_id)
        self.assertEqual(loaded.diagnostics.solving_id_inversions, 1)
        self.assertEqual(loaded.diagnostics.timestamp_inversions, 1)
        self.assertEqual(loaded.diagnostics.eligible_rows, 4)

    def test_same_solving_id_keeps_source_row_order(self) -> None:
        interactions = """\
user_id,timestamp,solving_id,question_id,user_answer
u1,500,1,q1,a
u1,100,1,q2,b
"""
        rows = _load(interactions=interactions).interactions
        self.assertEqual([row.order for row in rows], [0, 1])
        self.assertEqual([row.timestamp for row in rows], [500, 100])
        self.assertEqual(rows[0].bundle_id, rows[1].bundle_id)

    def test_timestamp_inversion_is_reported_not_fatal(self) -> None:
        interactions = """\
user_id,timestamp,solving_id,question_id,user_answer
u1,500,1,q1,a
u1,100,2,q2,b
"""
        loaded = _load(interactions=interactions)
        self.assertEqual(
            [(row.timestamp, row.bundle_id) for row in loaded.interactions],
            [(500, "u1|1"), (100, "u1|2")],
        )
        self.assertEqual(loaded.diagnostics.timestamp_inversions, 1)
        self.assertEqual(loaded.diagnostics.filtered_rows, 0)

    def test_repeated_question_id_is_allowed_across_bundles(self) -> None:
        interactions = """\
user_id,timestamp,solving_id,question_id,user_answer
u1,100,1,q1,a
u1,200,1,q1,b
u1,300,2,q1,c
"""
        loaded = _load(interactions=interactions)
        self.assertEqual(len(loaded.interactions), 3)
        self.assertEqual(loaded.diagnostics.repeated_question_attempt_rows, 2)
        self.assertEqual(loaded.diagnostics.questions_repeated_across_bundles, 1)
        self.assertEqual(
            [row.bundle_id for row in loaded.interactions],
            ["u1|1", "u1|1", "u1|2"],
        )


if __name__ == "__main__":
    unittest.main()
