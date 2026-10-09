"""Canonical interaction schema.

One :class:`Interaction` is one prediction target. ``kc_ids`` may contain
zero, one, or many knowledge components and is never a reason to split
the row.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


def _require_str(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be str, got {type(value).__name__}")
    return value


def _require_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {type(value).__name__}")
    return value


def _require_kc_ids(value: Any) -> tuple[int, ...]:
    if isinstance(value, tuple):
        items = value
    else:
        raise TypeError(
            "kc_ids must be a tuple of int, "
            f"got {type(value).__name__}"
        )
    out: list[int] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(
                "kc_ids entries must be int, "
                f"got {type(item).__name__}"
            )
        out.append(item)
    return tuple(out)


@dataclass(frozen=True)
class Interaction:
    """One learner-item response and its KC annotations.

    Raw datasets do not need to contain every field. Adapters fill this
    record. After standardization ``bundle_id`` is required. When the
    source has no bundle, the adapter sets ``bundle_id`` to
    ``interaction_id``.
    """

    interaction_id: str
    learner_id: str
    item_id: str
    correct: int
    kc_ids: tuple[int, ...]
    order: int
    timestamp: float | None
    bundle_id: str
    split: str | None
    metric_mask: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "interaction_id", _require_str("interaction_id", self.interaction_id))
        object.__setattr__(self, "learner_id", _require_str("learner_id", self.learner_id))
        object.__setattr__(self, "item_id", _require_str("item_id", self.item_id))
        object.__setattr__(self, "bundle_id", _require_str("bundle_id", self.bundle_id))
        object.__setattr__(self, "correct", _require_int("correct", self.correct))
        object.__setattr__(self, "order", _require_int("order", self.order))
        object.__setattr__(self, "kc_ids", _require_kc_ids(self.kc_ids))
        if self.timestamp is not None and (
            isinstance(self.timestamp, bool)
            or not isinstance(self.timestamp, (int, float))
        ):
            raise TypeError(
                "timestamp must be int, float, or None, "
                f"got {type(self.timestamp).__name__}"
            )
        if self.split is not None and not isinstance(self.split, str):
            raise TypeError(
                f"split must be str or None, got {type(self.split).__name__}"
            )
        if not isinstance(self.metric_mask, bool):
            raise TypeError(
                f"metric_mask must be bool, got {type(self.metric_mask).__name__}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "interaction_id": self.interaction_id,
            "learner_id": self.learner_id,
            "item_id": self.item_id,
            "correct": self.correct,
            "kc_ids": list(self.kc_ids),
            "order": self.order,
            "timestamp": self.timestamp,
            "bundle_id": self.bundle_id,
            "split": self.split,
            "metric_mask": self.metric_mask,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Interaction":
        required = (
            "interaction_id",
            "learner_id",
            "item_id",
            "correct",
            "kc_ids",
            "order",
            "timestamp",
            "bundle_id",
            "split",
            "metric_mask",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"missing interaction fields: {missing}")
        extra = [key for key in payload if key not in required]
        if extra:
            raise ValueError(f"unknown interaction fields: {extra}")
        kc_ids = payload["kc_ids"]
        if not isinstance(kc_ids, (list, tuple)):
            raise TypeError(
                "kc_ids must be a list or tuple in serialized form, "
                f"got {type(kc_ids).__name__}"
            )
        return cls(
            interaction_id=payload["interaction_id"],
            learner_id=payload["learner_id"],
            item_id=payload["item_id"],
            correct=payload["correct"],
            kc_ids=tuple(kc_ids),
            order=payload["order"],
            timestamp=payload["timestamp"],
            bundle_id=payload["bundle_id"],
            split=payload["split"],
            metric_mask=payload["metric_mask"],
        )


def prediction_target_count(interactions: Sequence[Interaction]) -> int:
    """Return the number of prediction targets.

    KC cardinality does not change the count. ``len(kc_ids) == 3`` is
    still one target.
    """

    return len(interactions)


def interactions_to_json(interactions: Sequence[Interaction]) -> str:
    return json.dumps(
        [row.to_dict() for row in interactions],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def interactions_from_json(text: str) -> tuple[Interaction, ...]:
    payload = json.loads(text)
    if not isinstance(payload, list):
        raise TypeError("interaction JSON must be a list")
    return tuple(Interaction.from_dict(item) for item in payload)
