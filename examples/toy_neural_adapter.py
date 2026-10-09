"""Synthetic neural-bundle adapter. It does not import torch or pyKT.

This file is an architecture demonstration. It is not a benchmark model,
not a scientific performance result, and not a replacement for DKT or DKVMN.

TRAIN walks canonical bundles and stores a learner-local count.
VALID replay starts from that count, scores each bundle from the
pre-bundle cursor, and does not write the cursor back into the
checkpoint. ``metric_mask`` is ignored here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from kclearner.data.schema import Interaction
from kclearner.experiments.adapter_contract import (
    FINGERPRINT_SCOPE_SINGLE_MODULE,
    NEURAL_BUNDLE_CONTRACT,
)

SUPPORTED_OOV_STATE = "retain_and_replay_v1"
SUPPORTED_OOV_REPRESENTATION = "row_retained_in_canonical_sequence_v1"
EXECUTION_SEMANTICS_ID = "toy_neural_bundle_replay_v1"


class ToyNeuralAdapter:
    """Bundle count with a frozen checkpoint during replay."""

    contract_version = NEURAL_BUNDLE_CONTRACT
    implementation_id = "toy_neural_bundle_count_v1"
    checkpoint_format_id = "toy_neural_bundle_count_state_v1"
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
                "ToyNeuralAdapter implements only oov state semantics "
                f"{SUPPORTED_OOV_STATE!r}, got {oov!r}"
            )
        self.oov_state_semantics_id = str(oov)
        seed = config.get("seed")
        if seed is not None and not isinstance(seed, int):
            raise ValueError("seed must be an int when it is supplied")
        self.seed = seed
        self._trained: dict[str, int] = {}
        self._replay_cursor: dict[str, int] | None = None

    def validate_dataset_contract(self, dataset: object) -> None:
        representation = getattr(dataset, "oov_representation_semantics_id", None)
        if representation not in self.supported_oov_representations:
            raise ValueError(
                "unsupported dataset OOV representation "
                f"{representation!r}; this adapter supports "
                f"{self.supported_oov_representations}"
            )

    def train(self, rows: Sequence[Interaction]) -> tuple[tuple[str, float], ...]:
        """Update the learned count. Return the probabilities used on the way."""

        output, trained = _score_bundles(rows, self._trained)
        self._trained = trained
        self._replay_cursor = None
        return output

    def replay(self, rows: Sequence[Interaction]) -> tuple[tuple[str, float], ...]:
        """Score from the trained count. The checkpoint count stays put."""

        output, cursor = _score_bundles(rows, self._trained)
        self._replay_cursor = cursor
        return output

    def continue_replay(self, rows: Sequence[Interaction]) -> tuple[tuple[str, float], ...]:
        """Continue the learner-local cursor. Shared counts stay frozen."""

        if self._replay_cursor is None:
            raise ValueError("neural replay has no cursor to continue")
        output, cursor = _score_bundles(rows, self._replay_cursor)
        self._replay_cursor = cursor
        return output

    def save_checkpoint(self, path: str | Path) -> None:
        payload = {
            "format_id": self.checkpoint_format_id,
            "trained": self._trained,
            "seed": self.seed,
        }
        Path(path).write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    def load_checkpoint(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_id") != self.checkpoint_format_id:
            raise ValueError("neural toy checkpoint format_id does not match")
        self._trained = {str(key): int(value) for key, value in payload["trained"].items()}
        self.seed = payload.get("seed")
        self._replay_cursor = None


def _score_bundles(
    rows: Sequence[Interaction],
    start: Mapping[str, int],
) -> tuple[tuple[tuple[str, float], ...], dict[str, int]]:
    cursor = dict(start)
    output: list[tuple[str, float]] = []
    for group in _contiguous_bundles(rows):
        learner = group[0].learner_id
        pre = cursor.get(learner, 0)
        probability = 1.0 / (1.0 + pre)
        for row in group:
            output.append((row.interaction_id, probability))
        cursor[learner] = pre + len(group)
    return tuple(output), cursor


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
