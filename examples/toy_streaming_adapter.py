"""Synthetic psychometric streaming adapter outside the six-model factory.

This file is an architecture demonstration. It is not a benchmark model,
not a scientific performance result, and not a replacement for IRT or AR-KT.
Probabilities are a learner-local count, not an IRT or AR-KT result.
Rows in one contiguous ``bundle_id`` share the pre-bundle count. The
count advances only after that bundle. ``metric_mask`` is ignored here;
KC-Learner applies it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from kclearner.data.schema import Interaction
from kclearner.experiments.adapter_contract import (
    FINGERPRINT_SCOPE_SINGLE_MODULE,
    PSYCHOMETRIC_STREAMING_CONTRACT,
)

SUPPORTED_OOV_STATE = "retain_and_score_v1"
SUPPORTED_OOV_REPRESENTATION = "row_retained_in_canonical_sequence_v1"
EXECUTION_SEMANTICS_ID = "toy_streaming_learner_count_v1"


class ToyStreamingAdapter:
    """Count adapter. It does not read TEST and it does not own splits.

    ``toy_streaming_learner_count_v1`` updates the learner-local count
    after each bundle in both TRAIN and VALID. There is no shared
    parameter to freeze. Rows inside one bundle share the pre-bundle count.
    """

    contract_version = PSYCHOMETRIC_STREAMING_CONTRACT
    implementation_id = "toy_streaming_count_v1"
    checkpoint_format_id = "toy_streaming_count_state_v1"
    traversal_id = "recorded_split_order"
    execution_semantics_id = EXECUTION_SEMANTICS_ID
    fingerprint_coverage = FINGERPRINT_SCOPE_SINGLE_MODULE
    supported_oov_representations = (SUPPORTED_OOV_REPRESENTATION,)
    supported_phase_state_semantics = (
        "independent_test_state_v1",
        "replay_valid_before_test_v1",
    )

    def __init__(self, config: Mapping[str, object]) -> None:
        oov = config.get("oov_state_semantics_id")
        if oov != SUPPORTED_OOV_STATE:
            raise ValueError(
                "ToyStreamingAdapter implements only oov state semantics "
                f"{SUPPORTED_OOV_STATE!r}, got {oov!r}"
            )
        self.oov_state_semantics_id = str(oov)
        self._revealed: dict[str, int] = {}

    def validate_dataset_contract(self, dataset: object) -> None:
        representation = getattr(dataset, "oov_representation_semantics_id", None)
        if representation not in self.supported_oov_representations:
            raise ValueError(
                "unsupported dataset OOV representation "
                f"{representation!r}; this adapter supports "
                f"{self.supported_oov_representations}"
            )

    def execute_phase(
        self,
        rows: Sequence[Interaction],
        *,
        phase: str,
    ) -> tuple[tuple[str, float], ...]:
        if phase not in ("train", "valid"):
            raise ValueError(f"phase must be 'train' or 'valid', got {phase!r}")
        output: list[tuple[str, float]] = []
        for group in _contiguous_bundles(rows):
            learner = group[0].learner_id
            pre = self._revealed.get(learner, 0)
            probability = 1.0 / (1.0 + pre)
            for row in group:
                output.append((row.interaction_id, probability))
            self._revealed[learner] = pre + len(group)
        return tuple(output)

    def save_state(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "format_id": self.checkpoint_format_id,
                    "revealed": self._revealed,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def load_state(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_id") != self.checkpoint_format_id:
            raise ValueError("toy state format_id does not match this adapter")
        self._revealed = {str(key): int(value) for key, value in payload["revealed"].items()}


def _contiguous_bundles(rows: Sequence[Interaction]) -> tuple[tuple[Interaction, ...], ...]:
    groups: list[list[Interaction]] = []
    for row in rows:
        if (
            groups
            and groups[-1][0].learner_id == row.learner_id
            and groups[-1][0].bundle_id == row.bundle_id
        ):
            groups[-1].append(row)
        else:
            groups.append([row])
    return tuple(tuple(group) for group in groups)
