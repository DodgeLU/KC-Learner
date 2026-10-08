"""Synthetic checks for the dev5000 window rule. No EdNet rows."""

from __future__ import annotations

import unittest

from window_selection import find_window


class WindowRuleTests(unittest.TestCase):
    def test_earliest_slice_skips_a_sentinel_prefix(self) -> None:
        student = [9, 0, 0, 1, 0, 0]
        group = [7, 1, 1, 9, 2, 1]
        problem = [0, 0, 1, 0, 2, 1]
        correct = [1, 1, 0, 1, 0, 1]
        tags = [(0, 0), (0, 1, 0), (2,), (1,), (0,), (1,)]
        sentinel = [True, False, False, False, False, False]
        self.assertEqual(
            find_window(student, group, problem, correct, tags, sentinel),
            (1, 6),
        )

    def test_missing_pattern_raises(self) -> None:
        with self.assertRaises(LookupError):
            find_window(
                [0, 0],
                [1, 1],
                [0, 1],
                [1, 0],
                [(0,), (1,)],
                [False, False],
                max_len=8,
            )
