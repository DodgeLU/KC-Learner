"""Corrected EdNet tags parser, including the -1 sentinel."""

from __future__ import annotations

import io
import unittest

from kclearner.data.adapters.ednet_metadata import (
    CORRECTED_PREPROCESSING_ID,
    LEGACY_PREPROCESSING_ID,
    EdNetTagParseError,
    load_ednet_question_metadata,
    parse_ednet_tags,
)


class EdNetTagTests(unittest.TestCase):
    def test_sentinel_minus_one_is_missing_kc(self) -> None:
        parsed = parse_ednet_tags("-1")
        self.assertEqual(parsed, ())
        self.assertNotIn(-1, parsed)

    def test_blank_and_missing_are_empty(self) -> None:
        self.assertEqual(parse_ednet_tags(None), ())
        self.assertEqual(parse_ednet_tags(""), ())
        self.assertEqual(parse_ednet_tags("   "), ())
        self.assertEqual(parse_ednet_tags(" -1 "), ())

    def test_valid_tags_keep_source_order(self) -> None:
        self.assertEqual(parse_ednet_tags("5;2;182"), (5, 2, 182))
        self.assertEqual(parse_ednet_tags(" 182 ; 5 ; 2 "), (182, 5, 2))

    def test_multi_kc_is_not_exploded(self) -> None:
        text = (
            "question_id,bundle_id,explanation_id,correct_answer,part,tags\n"
            "q1,b9,e1,a,1,5;2;182\n"
            "q2,b9,e2,b,1,-1\n"
            "q3,b8,e3,c,2,\n"
        )
        rows = load_ednet_question_metadata(io.StringIO(text))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0].kc_ids, (5, 2, 182))
        self.assertEqual(rows[0].content_bundle_id, "b9")
        self.assertEqual(rows[1].kc_ids, ())
        self.assertEqual(rows[1].tags_raw, "-1")
        self.assertEqual(rows[2].kc_ids, ())
        self.assertTrue(
            all(row.preprocessing_id == CORRECTED_PREPROCESSING_ID for row in rows)
        )
        self.assertNotEqual(CORRECTED_PREPROCESSING_ID, LEGACY_PREPROCESSING_ID)

    def test_mixed_sentinel_is_an_error(self) -> None:
        with self.assertRaises(EdNetTagParseError) as caught:
            parse_ednet_tags("5;-1;2", question_id="q9")
        message = str(caught.exception)
        self.assertIn("q9", message)
        self.assertIn("5;-1;2", message)
        self.assertIn("-1", message)
        self.assertEqual(caught.exception.token, "-1")

    def test_mixed_sentinel_from_question_csv_names_the_question(self) -> None:
        text = (
            "question_id,bundle_id,correct_answer,tags\n"
            "q9,b1,a,5;-1;2\n"
        )
        with self.assertRaises(EdNetTagParseError) as caught:
            load_ednet_question_metadata(io.StringIO(text))
        message = str(caught.exception)
        self.assertIn("question_id='q9'", message)
        self.assertIn("raw_tags='5;-1;2'", message)
        self.assertIn("invalid_token='-1'", message)

    def test_duplicate_tokens_keep_source_multiplicity(self) -> None:
        self.assertEqual(parse_ednet_tags("5;2;5"), (5, 2, 5))
        self.assertEqual(
            parse_ednet_tags("146;158;163;179;178;149;178"),
            (146, 158, 163, 179, 178, 149, 178),
        )

    def test_non_integer_token_is_rejected(self) -> None:
        text = (
            "question_id,bundle_id,correct_answer,tags\n"
            "q4,b1,c,5;foo\n"
        )
        with self.assertRaises(EdNetTagParseError) as caught:
            load_ednet_question_metadata(io.StringIO(text))
        message = str(caught.exception)
        self.assertIn("question_id='q4'", message)
        self.assertIn("raw_tags='5;foo'", message)
        self.assertIn("invalid_token='foo'", message)
        self.assertEqual(caught.exception.question_id, "q4")
        self.assertEqual(caught.exception.tags_raw, "5;foo")
        self.assertEqual(caught.exception.token, "foo")

    def test_does_not_invent_an_unknown_kc(self) -> None:
        parsed = parse_ednet_tags("-1")
        self.assertEqual(parsed, ())
        self.assertNotIn(0, parsed)


if __name__ == "__main__":
    unittest.main()
