"""Versioned logical identity for canonical interactions and datasets.

Serialization version ``canonical_interaction_hash_v1`` hashes ordered
canonical rows. It does not sort them, and it does not read Parquet
bytes, paths, clocks, or experiment settings.

Dataset version ``dataset_logical_identity_v1`` hashes the protocol
identity, cohort hash, per-split logical hashes, row counts, and
vocabulary hashes. Model, seed, device, and encoding provenance are
not part of that value.

EdNet ``source_row`` and ``solving_id`` are not fields of
:class:`~kclearner.data.schema.Interaction`. The current EdNet adapter
copies them into ``interaction_id`` and ``bundle_id``. Part 3 should
still hash those two columns on the EdNet row extension so the dataset
identity does not depend on that encoding remaining in place.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Sequence

from kclearner.data.schema import Interaction

SERIALIZATION_VERSION = "canonical_interaction_hash_v1"
DATASET_IDENTITY_VERSION = "dataset_logical_identity_v1"

# Fixed object key order. json.dumps below is only a string encoder for
# individual values; the record layout is written explicitly.
_INTERACTION_KEYS = (
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

_EXACT_FLOAT_INT_LIMIT = 2**53


class LogicalIdentityError(ValueError):
    """A value cannot be serialized under the logical-identity rules."""


def _json_string(value: str) -> str:
    if not isinstance(value, str):
        raise LogicalIdentityError(f"expected str, got {type(value).__name__}")
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_int(value: int) -> str:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LogicalIdentityError(f"expected int, got {type(value).__name__}")
    return str(value)


def _timestamp_token(value: int | float | None) -> str:
    """Serialize a timestamp without using ``repr``.

    ``None`` is JSON null. Integers use decimal form. A finite float
    whose value is an integer inside the IEEE exact-integer range uses
    the same decimal form, so ``1000`` and ``1000.0`` are one timestamp.
    Any other finite float uses scientific notation with 17 significant
    digits. NaN and infinities are rejected.
    """

    if value is None:
        return "null"
    if isinstance(value, bool):
        raise LogicalIdentityError("timestamp must not be bool")
    if isinstance(value, int):
        return _json_int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LogicalIdentityError("timestamp must be finite")
        if value.is_integer() and abs(value) <= _EXACT_FLOAT_INT_LIMIT:
            return str(int(value))
        return format(value, ".16e")
    raise LogicalIdentityError(
        f"timestamp must be int, float, or None, got {type(value).__name__}"
    )


def canonical_interaction_json(row: Interaction) -> str:
    """Return one compact JSON object for ``row``.

    The result is UTF-8 text with no insignificant whitespace. KC ids
    stay in the tuple order stored on the interaction.
    """

    if not isinstance(row, Interaction):
        raise LogicalIdentityError(
            f"expected Interaction, got {type(row).__name__}"
        )
    kc = "[" + ",".join(_json_int(kc) for kc in row.kc_ids) + "]"
    split = "null" if row.split is None else _json_string(row.split)
    mask = "true" if row.metric_mask else "false"
    body = ",".join(
        (
            f"{_json_string(_INTERACTION_KEYS[0])}:{_json_string(row.interaction_id)}",
            f"{_json_string(_INTERACTION_KEYS[1])}:{_json_string(row.learner_id)}",
            f"{_json_string(_INTERACTION_KEYS[2])}:{_json_string(row.item_id)}",
            f"{_json_string(_INTERACTION_KEYS[3])}:{_json_int(row.correct)}",
            f"{_json_string(_INTERACTION_KEYS[4])}:{kc}",
            f"{_json_string(_INTERACTION_KEYS[5])}:{_json_int(row.order)}",
            f"{_json_string(_INTERACTION_KEYS[6])}:{_timestamp_token(row.timestamp)}",
            f"{_json_string(_INTERACTION_KEYS[7])}:{_json_string(row.bundle_id)}",
            f"{_json_string(_INTERACTION_KEYS[8])}:{split}",
            f"{_json_string(_INTERACTION_KEYS[9])}:{mask}",
        )
    )
    return "{" + body + "}"


def logical_split_hash(rows: Sequence[Interaction]) -> str:
    """SHA-256 of the ordered canonical records.

    Each record is followed by one LF byte. The function does not sort
    or deduplicate. An empty sequence hashes the empty byte string.
    """

    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_interaction_json(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def hash_ordered_ids(ids: Sequence[str]) -> str:
    """SHA-256 of ordered string ids joined by LF, with no trailing LF.

    This is the historical cohort and vocabulary encoding: the caller
    supplies the canonical order. The helper does not sort, and it
    keeps duplicates. It is not a hash of an unordered set, and it is
    not the per-interaction KC-order hash.
    """

    if any(not isinstance(item, str) for item in ids):
        raise LogicalIdentityError("ordered ids must be strings")
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SplitLogicalIdentity:
    """Logical identity of one split in protocol order."""

    split_id: str
    logical_hash: str
    row_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.split_id, str) or self.split_id == "":
            raise LogicalIdentityError("split_id must be a non-empty str")
        if not isinstance(self.logical_hash, str) or len(self.logical_hash) != 64:
            raise LogicalIdentityError("logical_hash must be 64 hex characters")
        if isinstance(self.row_count, bool) or not isinstance(self.row_count, int):
            raise LogicalIdentityError("row_count must be int")
        if self.row_count < 0:
            raise LogicalIdentityError("row_count must be >= 0")


@dataclass(frozen=True)
class DatasetLogicalIdentity:
    """Scientific identity of one prepared dataset.

    ``splits`` stay in the order supplied by the protocol. ``None``
    vocabulary or cohort hashes are JSON null and still participate in
    the dataset hash, so missing and present values do not collide.
    """

    dataset_id: str
    protocol_id: str
    preprocessing_version: str
    cohort_hash: str | None
    splits: tuple[SplitLogicalIdentity, ...]
    item_vocabulary_hash: str | None
    kc_vocabulary_hash: str | None
    identity_version: str = DATASET_IDENTITY_VERSION
    serialization_version: str = SERIALIZATION_VERSION

    def __post_init__(self) -> None:
        for name in (
            "dataset_id",
            "protocol_id",
            "preprocessing_version",
            "identity_version",
            "serialization_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or value == "":
                raise LogicalIdentityError(f"{name} must be a non-empty str")
        for name in ("cohort_hash", "item_vocabulary_hash", "kc_vocabulary_hash"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or len(value) != 64
            ):
                raise LogicalIdentityError(f"{name} must be 64 hex characters or None")
        if not isinstance(self.splits, tuple):
            raise LogicalIdentityError("splits must be a tuple")
        for split in self.splits:
            if not isinstance(split, SplitLogicalIdentity):
                raise LogicalIdentityError("splits must contain SplitLogicalIdentity")


@dataclass(frozen=True)
class EncodingProvenance:
    """Cache encoding facts. These do not enter ``dataset_logical_hash``."""

    parquet_sha256: str | None = None
    pyarrow_version: str | None = None
    serialization_format: str | None = None


def _hash_token(value: str | None) -> str:
    if value is None:
        return "null"
    return _json_string(value)


def canonical_dataset_json(identity: DatasetLogicalIdentity) -> str:
    """Compact JSON for a dataset identity, in fixed key order."""

    splits = "[" + ",".join(
        "{"
        + f"{_json_string('split_id')}:{_json_string(split.split_id)},"
        + f"{_json_string('logical_hash')}:{_json_string(split.logical_hash)},"
        + f"{_json_string('row_count')}:{_json_int(split.row_count)}"
        + "}"
        for split in identity.splits
    ) + "]"
    fields = (
        ("identity_version", _json_string(identity.identity_version)),
        ("serialization_version", _json_string(identity.serialization_version)),
        ("dataset_id", _json_string(identity.dataset_id)),
        ("protocol_id", _json_string(identity.protocol_id)),
        ("preprocessing_version", _json_string(identity.preprocessing_version)),
        ("cohort_hash", _hash_token(identity.cohort_hash)),
        ("splits", splits),
        ("item_vocabulary_hash", _hash_token(identity.item_vocabulary_hash)),
        ("kc_vocabulary_hash", _hash_token(identity.kc_vocabulary_hash)),
    )
    return "{" + ",".join(f"{_json_string(key)}:{value}" for key, value in fields) + "}"


def dataset_logical_hash(identity: DatasetLogicalIdentity) -> str:
    """SHA-256 of :func:`canonical_dataset_json` encoded as UTF-8.

    Paths, timestamps, git state, Python version, model settings,
    training seed, device, and Parquet byte hashes are not accepted
    by this function.
    """

    return hashlib.sha256(
        canonical_dataset_json(identity).encode("utf-8")
    ).hexdigest()
