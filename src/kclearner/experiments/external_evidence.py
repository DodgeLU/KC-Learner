"""Canonical VALID evidence for external formal experiments.

Trace serialization uses one compact JSON object per row, in execution
order, with keys ``interaction_id`` then ``probability``. Each record is
followed by one LF byte. Probabilities are finite numbers in ``[0, 1]``.
An integer-valued finite float inside the exact integer range is written
as a decimal integer. Every other finite value uses 17-significant-digit
scientific notation. ``NaN`` and infinities are rejected.

Masked rows stay in the trace. ``metric_mask`` is not part of the trace
hash; it is applied only when metrics are aggregated.

Metric evidence uses the fixed key order ``nll``, ``brier``, ``auc``,
``n``. Undefined metric values are JSON null. This module does not
substitute a fallback AUC.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Sequence

TRACE_VERSION = "external_valid_prediction_trace_v1"
METRIC_EVIDENCE_VERSION = "external_valid_metric_evidence_v1"
_EXACT_FLOAT_INT_LIMIT = 2**53


class EvidenceError(ValueError):
    """A probability or metric value cannot enter formal evidence."""


def probability_token(value: object) -> str:
    """Return the canonical probability token, or reject the value."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(
            f"probability must be int or float, got {type(value).__name__}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise EvidenceError("probability must be finite")
    if number < 0.0 or number > 1.0:
        raise EvidenceError(f"probability {number} is outside [0, 1]")
    if isinstance(value, int) or (
        isinstance(value, float)
        and number.is_integer()
        and abs(number) <= _EXACT_FLOAT_INT_LIMIT
    ):
        return str(int(number))
    return format(number, ".16e")


def canonical_trace_json(rows: Sequence[tuple[str, float]]) -> str:
    """Serialize the ordered prediction trace. Masked rows are included."""

    lines = []
    for interaction_id, probability in rows:
        if not isinstance(interaction_id, str) or interaction_id == "":
            raise EvidenceError("interaction_id must be a non-empty string")
        token = probability_token(probability)
        lines.append(
            "{"
            + f"{_json_string('interaction_id')}:{_json_string(interaction_id)},"
            + f"{_json_string('probability')}:{token}"
            + "}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def valid_prediction_trace_hash(rows: Sequence[tuple[str, float]]) -> str:
    """SHA-256 of :func:`canonical_trace_json`."""

    return hashlib.sha256(canonical_trace_json(rows).encode("utf-8")).hexdigest()


def metric_token(value: object) -> str:
    """Serialize one metric. Undefined and non-finite values are null."""

    if value is None:
        return "null"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"metric must be numeric or null, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        return "null"
    if isinstance(value, int) or (
        isinstance(value, float)
        and number.is_integer()
        and abs(number) <= _EXACT_FLOAT_INT_LIMIT
    ):
        return str(int(number))
    return format(number, ".16e")


def canonical_metric_json(metrics: Mapping[str, object] | None) -> str:
    """Compact metric evidence. Key order is fixed."""

    if metrics is None:
        body = {"nll": None, "brier": None, "auc": None, "n": 0}
    else:
        body = {
            "nll": metrics.get("nll"),
            "brier": metrics.get("brier"),
            "auc": metrics.get("auc"),
            "n": metrics.get("n", 0),
        }
    count = body["n"]
    if isinstance(count, bool) or not isinstance(count, int):
        raise EvidenceError("metric n must be an int")
    fields = (
        ("evidence_version", _json_string(METRIC_EVIDENCE_VERSION)),
        ("nll", metric_token(body["nll"])),
        ("brier", metric_token(body["brier"])),
        ("auc", metric_token(body["auc"])),
        ("n", str(count)),
    )
    return "{" + ",".join(f"{_json_string(key)}:{value}" for key, value in fields) + "}"


def metric_evidence_hash(metrics: Mapping[str, object] | None) -> str:
    """SHA-256 of :func:`canonical_metric_json`."""

    return hashlib.sha256(canonical_metric_json(metrics).encode("utf-8")).hexdigest()


def _json_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
