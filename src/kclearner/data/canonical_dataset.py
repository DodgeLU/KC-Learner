"""External canonical-dataset directory.

This loader does not prepare EdNet or ASSISTments, and it does not start
training, freeze, or TEST. Reference executions keep their own files and
their ``solving_id`` / ``group_id`` columns.

Version strings already defined elsewhere:

* ``canonical_interaction_hash_v1`` is the interaction-hash serialization
  (``SERIALIZATION_VERSION``). It is not a schema id.
* ``dataset_logical_identity_v1`` is the dataset-content identity. Its
  hashed fields are unchanged.
* ``kclearner_interaction_v1`` is the EdNet corrected manifest schema
  label. It is not this directory's schema id, because those files also
  carry EdNet columns.
* ``experiment_freeze_identity_v1`` is not read or written here.

This module adds two identifiers that did not previously exist:

* ``canonical_interaction_schema_v1``: the ten ``Interaction`` fields in
  this directory.
* ``dataset_contract_identity_v1``: content hash plus the interpretation
  fields that ``dataset_logical_identity_v1`` does not contain.

``source_fingerprint``, when supplied, is provenance only. It is stored
and is not an input to either hash. The contract identity is the
canonical scientific semantics, not raw-source file identity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from kclearner.data.logical_identity import (
    DATASET_IDENTITY_VERSION,
    SERIALIZATION_VERSION,
    DatasetLogicalIdentity,
    SplitLogicalIdentity,
    dataset_logical_hash,
    hash_ordered_ids,
    logical_split_hash,
)
from kclearner.data.schema import Interaction
from kclearner.data.validation import DatasetValidator

CANONICAL_SCHEMA_VERSION = "canonical_interaction_schema_v1"
CONTRACT_IDENTITY_VERSION = "dataset_contract_identity_v1"

DECLARATION_NAME = "declaration.json"
INTERACTIONS_NAME = "interactions.jsonl"
LEARNER_IDS_NAME = "learner_ids.txt"
ITEM_IDS_NAME = "item_ids.txt"
KC_IDS_NAME = "kc_ids.txt"

# Roles the current TRAIN/VALID/TEST workflow accepts. Execution selects
# a split by this role string. The tuple is not a permanent schema limit:
# a later workflow can extend the gate, and list position still assigns
# nothing.
WORKFLOW_ROLES = ("train", "valid", "test")

_REQUIRED_DECLARATION_KEYS = (
    "dataset_id",
    "protocol_id",
    "preprocessing_version",
    "canonical_schema_version",
    "interaction_identity_version",
    "dataset_identity_version",
    "split_order",
    "split_roles",
    "bundle_semantics_id",
    "oov_representation_semantics_id",
    "metric_mask_semantics",
    "provenance",
)
_OPTIONAL_DECLARATION_KEYS = frozenset({"source_fingerprint"})


class CanonicalDatasetError(ValueError):
    """An external canonical directory cannot be loaded."""


@dataclass(frozen=True)
class CanonicalDataset:
    """One loaded external directory and its two identities.

    ``dataset_logical_hash`` follows ``dataset_logical_identity_v1``.
    ``dataset_contract_hash`` follows ``dataset_contract_identity_v1``.
    Provenance and ``source_fingerprint`` are not inputs to either hash.
    No model-family state-update rule is stored on this object.
    """

    dataset_id: str
    protocol_id: str
    preprocessing_version: str
    canonical_schema_version: str
    interaction_identity_version: str
    dataset_identity_version: str
    split_order: tuple[str, ...]
    split_roles: tuple[tuple[str, str], ...]
    bundle_semantics_id: str
    oov_representation_semantics_id: str
    metric_mask_semantics: str
    learner_ids: tuple[str, ...]
    item_ids: tuple[str, ...]
    kc_ids: tuple[str, ...]
    interactions: tuple[Interaction, ...]
    logical_identity: DatasetLogicalIdentity
    dataset_logical_hash: str
    dataset_contract_hash: str
    provenance: Mapping[str, Any]
    source_fingerprint: Any
    has_source_fingerprint: bool


def load_canonical_dataset(directory: str | Path) -> CanonicalDataset:
    """Load ``directory``. Missing identity-bearing fields fail the load.

    Vocabulary files keep the caller's line order. This function does not
    sort them, and it does not interpret OOV representation identifiers.
    """

    root = Path(directory)
    if not root.is_dir():
        raise CanonicalDatasetError(f"canonical dataset directory is missing: {root}")
    declaration = _read_declaration(root / DECLARATION_NAME)
    learner_ids = _read_ordered_ids(root / LEARNER_IDS_NAME, "learner_ids")
    item_ids = _read_ordered_ids(root / ITEM_IDS_NAME, "item_ids")
    kc_ids = tuple(
        _canonical_kc_token(token)
        for token in _read_ordered_ids(root / KC_IDS_NAME, "kc_ids")
    )
    interactions = _read_interactions(root / INTERACTIONS_NAME)
    split_order = declaration["split_order"]
    _require_split_membership(interactions, split_order)
    validation = DatasetValidator().validate(interactions)
    if not validation.ok:
        codes = ", ".join(issue.code for issue in validation.issues)
        raise CanonicalDatasetError(f"canonical interactions failed validation: {codes}")
    identity = _logical_identity(
        declaration,
        interactions,
        learner_ids,
        item_ids,
        kc_ids,
    )
    logical_hash = dataset_logical_hash(identity)
    contract_hash = dataset_contract_hash(
        dataset_logical_hash_value=logical_hash,
        canonical_schema_version=declaration["canonical_schema_version"],
        bundle_semantics_id=declaration["bundle_semantics_id"],
        oov_representation_semantics_id=declaration["oov_representation_semantics_id"],
        metric_mask_semantics=declaration["metric_mask_semantics"],
        split_roles=declaration["split_roles"],
    )
    return CanonicalDataset(
        dataset_id=declaration["dataset_id"],
        protocol_id=declaration["protocol_id"],
        preprocessing_version=declaration["preprocessing_version"],
        canonical_schema_version=declaration["canonical_schema_version"],
        interaction_identity_version=declaration["interaction_identity_version"],
        dataset_identity_version=declaration["dataset_identity_version"],
        split_order=split_order,
        split_roles=declaration["split_roles"],
        bundle_semantics_id=declaration["bundle_semantics_id"],
        oov_representation_semantics_id=declaration["oov_representation_semantics_id"],
        metric_mask_semantics=declaration["metric_mask_semantics"],
        learner_ids=learner_ids,
        item_ids=item_ids,
        kc_ids=kc_ids,
        interactions=interactions,
        logical_identity=identity,
        dataset_logical_hash=logical_hash,
        dataset_contract_hash=contract_hash,
        provenance=declaration["provenance"],
        source_fingerprint=declaration["source_fingerprint"],
        has_source_fingerprint=declaration["has_source_fingerprint"],
    )


def canonical_contract_json(
    *,
    dataset_logical_hash_value: str,
    canonical_schema_version: str,
    bundle_semantics_id: str,
    oov_representation_semantics_id: str,
    metric_mask_semantics: str,
    split_roles: Sequence[tuple[str, str]],
) -> str:
    """Compact contract JSON. Provenance is not accepted here."""

    roles = "[" + ",".join(
        "{"
        + f"{_json_string('split_id')}:{_json_string(split_id)},"
        + f"{_json_string('role')}:{_json_string(role)}"
        + "}"
        for split_id, role in split_roles
    ) + "]"
    fields = (
        ("contract_identity_version", _json_string(CONTRACT_IDENTITY_VERSION)),
        ("dataset_logical_hash", _json_string(dataset_logical_hash_value)),
        ("canonical_schema_version", _json_string(canonical_schema_version)),
        ("bundle_semantics_id", _json_string(bundle_semantics_id)),
        ("oov_representation_semantics_id", _json_string(oov_representation_semantics_id)),
        ("metric_mask_semantics", _json_string(metric_mask_semantics)),
        ("split_roles", roles),
    )
    return "{" + ",".join(f"{_json_string(key)}:{value}" for key, value in fields) + "}"


def dataset_contract_hash(
    *,
    dataset_logical_hash_value: str,
    canonical_schema_version: str,
    bundle_semantics_id: str,
    oov_representation_semantics_id: str,
    metric_mask_semantics: str,
    split_roles: Sequence[tuple[str, str]],
) -> str:
    """SHA-256 of :func:`canonical_contract_json`."""

    text = canonical_contract_json(
        dataset_logical_hash_value=dataset_logical_hash_value,
        canonical_schema_version=canonical_schema_version,
        bundle_semantics_id=bundle_semantics_id,
        oov_representation_semantics_id=oov_representation_semantics_id,
        metric_mask_semantics=metric_mask_semantics,
        split_roles=split_roles,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _logical_identity(
    declaration: Mapping[str, Any],
    interactions: Sequence[Interaction],
    learner_ids: Sequence[str],
    item_ids: Sequence[str],
    kc_ids: Sequence[str],
) -> DatasetLogicalIdentity:
    grouped = {split_id: [] for split_id in declaration["split_order"]}
    for row in interactions:
        grouped[row.split].append(row)
    splits = tuple(
        SplitLogicalIdentity(
            split_id,
            logical_split_hash(tuple(grouped[split_id])),
            len(grouped[split_id]),
        )
        for split_id in declaration["split_order"]
    )
    return DatasetLogicalIdentity(
        dataset_id=declaration["dataset_id"],
        protocol_id=declaration["protocol_id"],
        preprocessing_version=declaration["preprocessing_version"],
        cohort_hash=hash_ordered_ids(learner_ids),
        splits=splits,
        item_vocabulary_hash=hash_ordered_ids(item_ids),
        kc_vocabulary_hash=hash_ordered_ids(kc_ids),
    )


def _read_declaration(path: Path) -> dict[str, Any]:
    payload = _read_json_object(path)
    unknown = [key for key in payload if key not in _REQUIRED_DECLARATION_KEYS and key not in _OPTIONAL_DECLARATION_KEYS]
    if unknown:
        raise CanonicalDatasetError(f"unknown declaration fields: {unknown}")
    missing = [key for key in _REQUIRED_DECLARATION_KEYS if key not in payload]
    if missing:
        raise CanonicalDatasetError(f"declaration is missing {missing}")
    dataset_id = _require_token("dataset_id", payload["dataset_id"])
    protocol_id = _require_token("protocol_id", payload["protocol_id"])
    preprocessing_version = _require_token(
        "preprocessing_version", payload["preprocessing_version"]
    )
    schema_version = _require_token(
        "canonical_schema_version", payload["canonical_schema_version"]
    )
    if schema_version != CANONICAL_SCHEMA_VERSION:
        raise CanonicalDatasetError(
            "canonical_schema_version "
            f"{schema_version!r} != {CANONICAL_SCHEMA_VERSION}. "
            f"{SERIALIZATION_VERSION} is the interaction-hash version, "
            "not the schema version."
        )
    interaction_version = _require_token(
        "interaction_identity_version", payload["interaction_identity_version"]
    )
    if interaction_version != SERIALIZATION_VERSION:
        raise CanonicalDatasetError(
            "interaction_identity_version "
            f"{interaction_version!r} != {SERIALIZATION_VERSION}"
        )
    dataset_version = _require_token(
        "dataset_identity_version", payload["dataset_identity_version"]
    )
    if dataset_version != DATASET_IDENTITY_VERSION:
        raise CanonicalDatasetError(
            "dataset_identity_version "
            f"{dataset_version!r} != {DATASET_IDENTITY_VERSION}"
        )
    split_order = _split_order(payload["split_order"])
    split_roles = _split_roles(payload["split_roles"], split_order)
    if not isinstance(payload["provenance"], dict):
        raise CanonicalDatasetError("provenance must be a JSON object")
    has_fingerprint = "source_fingerprint" in payload
    return {
        "dataset_id": dataset_id,
        "protocol_id": protocol_id,
        "preprocessing_version": preprocessing_version,
        "canonical_schema_version": schema_version,
        "interaction_identity_version": interaction_version,
        "dataset_identity_version": dataset_version,
        "split_order": split_order,
        "split_roles": split_roles,
        "bundle_semantics_id": _require_token(
            "bundle_semantics_id", payload["bundle_semantics_id"]
        ),
        "oov_representation_semantics_id": _require_token(
            "oov_representation_semantics_id",
            payload["oov_representation_semantics_id"],
        ),
        "metric_mask_semantics": _require_token(
            "metric_mask_semantics", payload["metric_mask_semantics"]
        ),
        "provenance": payload["provenance"],
        "source_fingerprint": payload.get("source_fingerprint"),
        "has_source_fingerprint": has_fingerprint,
    }


def _split_order(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise CanonicalDatasetError("split_order must be a non-empty list")
    order: list[str] = []
    for item in value:
        token = _require_token("split_order entry", item)
        if token in order:
            raise CanonicalDatasetError(f"split_order repeats {token!r}")
        order.append(token)
    return tuple(order)


def _split_roles(value: Any, split_order: Sequence[str]) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, dict):
        raise CanonicalDatasetError("split_roles must be a JSON object")
    missing = [split_id for split_id in split_order if split_id not in value]
    extra = [split_id for split_id in value if split_id not in split_order]
    if missing or extra:
        raise CanonicalDatasetError(
            "split_roles must name each split_order id once; "
            f"missing={missing} extra={extra}"
        )
    assigned: dict[str, str] = {}
    roles: list[tuple[str, str]] = []
    for split_id in split_order:
        role = value[split_id]
        if role not in WORKFLOW_ROLES:
            raise CanonicalDatasetError(
                f"split {split_id!r} has role {role!r}; "
                f"role must be one of {WORKFLOW_ROLES}. "
                "List position does not assign a role."
            )
        if role in assigned:
            raise CanonicalDatasetError(
                f"workflow role {role!r} is assigned to both "
                f"{assigned[role]!r} and {split_id!r}"
            )
        assigned[role] = split_id
        roles.append((split_id, role))
    return tuple(roles)


def _require_split_membership(
    interactions: Sequence[Interaction],
    split_order: Sequence[str],
) -> None:
    allowed = set(split_order)
    for row in interactions:
        if row.split not in allowed:
            raise CanonicalDatasetError(
                f"{row.interaction_id!r} has split {row.split!r}, "
                f"which is not in split_order {list(split_order)}"
            )


def _read_interactions(path: Path) -> tuple[Interaction, ...]:
    if not path.is_file():
        raise CanonicalDatasetError(f"missing {path.name}")
    text = path.read_text(encoding="utf-8")
    if text == "":
        return ()
    rows: list[Interaction] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line == "":
            raise CanonicalDatasetError(
                f"{path.name} line {line_number} is blank"
            )
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CanonicalDatasetError(
                f"{path.name} line {line_number} is not JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise CanonicalDatasetError(
                f"{path.name} line {line_number} must be one JSON object"
            )
        try:
            rows.append(Interaction.from_dict(payload))
        except (TypeError, ValueError) as exc:
            raise CanonicalDatasetError(
                f"{path.name} line {line_number}: {exc}"
            ) from exc
    return tuple(rows)


def _read_ordered_ids(path: Path, label: str) -> tuple[str, ...]:
    """Read one id per line. Line order is the identity order."""

    if not path.is_file():
        raise CanonicalDatasetError(f"missing {path.name}")
    text = path.read_text(encoding="utf-8")
    if text == "":
        return ()
    ids: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line == "":
            raise CanonicalDatasetError(f"{path.name} line {line_number} is blank")
        if line in seen:
            raise CanonicalDatasetError(
                f"{label} repeats {line!r}; the file was not rewritten or sorted"
            )
        seen.add(line)
        ids.append(line)
    return tuple(ids)


def _canonical_kc_token(token: str) -> str:
    """Accept one canonical decimal integer token. Do not renumber it."""

    if token == "0":
        return token
    sign = ""
    body = token
    if token.startswith("-"):
        sign = "-"
        body = token[1:]
    if body == "" or not body.isdigit() or body[0] == "0":
        raise CanonicalDatasetError(
            f"kc id {token!r} is not a canonical decimal integer"
        )
    return sign + body


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CanonicalDatasetError(f"missing {path.name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CanonicalDatasetError(f"{path.name} is not JSON") from exc
    if not isinstance(payload, dict):
        raise CanonicalDatasetError(f"{path.name} must be a JSON object")
    return payload


def _require_token(name: str, value: Any) -> str:
    if not isinstance(value, str) or value == "":
        raise CanonicalDatasetError(f"{name} must be a non-empty string")
    return value


def _json_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
