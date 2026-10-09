"""Synthetic checks for the vendored EdNet pilot sampler."""

from __future__ import annotations

import random
import unittest

from kclearner.data.ednet_pilot_sample import (
    HISTORICAL_SAMPLE_SHA256,
    SAMPLER_ID,
    select_sorted_sample,
)
from kclearner.data.logical_identity import hash_ordered_ids


class PilotSamplerTests(unittest.TestCase):
    def test_pool_and_set_branches_are_fixed(self) -> None:
        self.assertEqual(SAMPLER_ID, "ednet_pilot50k_sampler_v1")
        small = tuple(f"u{index}" for index in range(10))
        self.assertEqual(
            select_sorted_sample(small, k=3, seed=42),
            ("u0", "u1", "u4"),
        )
        wide = tuple(f"u{index:03d}" for index in range(100))
        selected = select_sorted_sample(wide, k=6, seed=42)
        self.assertEqual(
            selected,
            ("u003", "u014", "u031", "u035", "u081", "u094"),
        )
        self.assertEqual(
            hash_ordered_ids(selected),
            "aaa2ec5d4f42bbadd90239bfe421f0afd7caf7519d5dbec9a8164a1c0b7a043f",
        )

    def test_sampler_does_not_call_random_sample(self) -> None:
        original = random.Random.sample

        def _forbidden(self, population, k, *, counts=None):  # type: ignore[no-untyped-def]
            raise AssertionError("random.sample must not be used")

        random.Random.sample = _forbidden  # type: ignore[method-assign]
        try:
            selected = select_sorted_sample(
                tuple(f"u{index}" for index in range(10)),
                k=3,
                seed=42,
            )
        finally:
            random.Random.sample = original  # type: ignore[method-assign]
        self.assertEqual(selected, ("u0", "u1", "u4"))

    def test_unsorted_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_sorted_sample(("u2", "u1"), k=1, seed=42)

    def test_frozen_hash_constant_is_the_historical_value(self) -> None:
        self.assertEqual(
            HISTORICAL_SAMPLE_SHA256,
            "20b29ac2ebf6e8cca9e763f11a44d850c773cf04e27464c8c5b0e8e9c7566b96",
        )
