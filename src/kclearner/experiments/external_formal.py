"""Formal external TRAIN, freeze, and authorized TEST.

This module does not call the reference freeze writer and does not use
``experiment_freeze_identity_v1`` or ``evaluation_semantics_v1``.

Role files are an access profile. Split hashes still use
``dataset_logical_identity_v1``. Formal TRAIN and freeze never build a
TEST path. ``authorize_external_test`` reads that file only after
identity and exact VALID checks succeed, then rejects the file unless
its split hash and row count match the sealed TEST commitment.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from kclearner.data.canonical_dataset import (
    CanonicalDataset,
    dataset_contract_hash,
    load_canonical_dataset,
)
from kclearner.data.logical_identity import (
    DatasetLogicalIdentity,
    SplitLogicalIdentity,
    canonical_dataset_json,
    dataset_logical_hash,
    logical_split_hash,
)
from kclearner.data.schema import Interaction
from kclearner.experiments.adapter_contract import (
    NEURAL_BUNDLE_CONTRACT,
    PSYCHOMETRIC_STREAMING_CONTRACT,
    AdapterProvenance,
    adapter_provenance,
    load_adapter,
)
from kclearner.experiments.external_evidence import (
    canonical_metric_json,
    valid_prediction_trace_hash,
)
from kclearner.experiments.external_execution import _align_and_score

EXTERNAL_EXPERIMENT_IDENTITY_VERSION = "external_experiment_identity_v1"
EXTERNAL_METRIC_SEMANTICS = "external_metric_semantics_v1"
PHASE_INDEPENDENT = "independent_test_state_v1"
PHASE_REPLAY_VALID = "replay_valid_before_test_v1"
VERIFICATION_EXACT = "exact_replay_v1"
_PHASE_POLICIES = (PHASE_INDEPENDENT, PHASE_REPLAY_VALID)

TRAIN_FILE = "interactions_train.jsonl"
VALID_FILE = "interactions_valid.jsonl"
TEST_FILE = "interactions_test.jsonl"
_ROLE_FILES = {"train": TRAIN_FILE, "valid": VALID_FILE, "test": TEST_FILE}
LOGICAL_NAME = "logical_identity.json"
DECLARATION_NAME = "declaration.json"
CHECKPOINT_NAME = "checkpoint_train_only.json"
RUN_NAME = "external_formal_run.json"
FREEZE_NAME = "external_freeze_manifest.json"
TEST_RESULT_NAME = "external_test_result.json"
_IMPLEMENTATION_KEYS = (
    "adapter_contract_version",
    "adapter_family",
    "module_name",
    "class_name",
    "distribution_name",
    "distribution_version",
    "implementation_fingerprint_kind",
    "implementation_fingerprint_scope",
    "implementation_fingerprint_value",
    "implementation_id",
    "checkpoint_format_id",
    "execution_semantics_id",
    "traversal_id",
    "oov_state_semantics_id",
)


class ExternalFormalError(ValueError):
    """An external formal freeze or TEST authorization cannot proceed."""


class FileRead:
    """Dataset reads used by the formal path. Tests can subclass this."""

    def __init__(self) -> None:
        self.names: list[str] = []

    def read_text(self, path: Path) -> str:
        self.names.append(path.name)
        return path.read_text(encoding="utf-8")

    def read_bytes(self, path: Path) -> bytes:
        self.names.append(path.name)
        return path.read_bytes()


@dataclass
class FormalRun:
    run_dir: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    dataset_logical_hash: str
    dataset_contract_hash: str
    valid_trace_hash: str
    valid_metrics: dict[str, object]
    provenance: AdapterProvenance
    phase_state_semantics_id: str
    adapter_spec: str
    config: dict[str, Any]
    files_read: tuple[str, ...]


def materialize_formal_directory(source_dir: str | Path, dest_dir: str | Path) -> CanonicalDataset:
    """Copy one canonical directory into role files.

    The source read can see every split. Formal TRAIN does not call this.
    """

    source = Path(source_dir)
    dataset = load_canonical_dataset(source)
    destination = Path(dest_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name in (DECLARATION_NAME, "learner_ids.txt", "item_ids.txt", "kc_ids.txt"):
        (destination / name).write_bytes((source / name).read_bytes())
    for split_id, role in dataset.split_roles:
        chosen = tuple(row for row in dataset.interactions if row.split == split_id)
        _write_jsonl(destination / _ROLE_FILES[role], chosen)
    document = {
        "dataset_logical_hash": dataset.dataset_logical_hash,
        "dataset_contract_hash": dataset.dataset_contract_hash,
        "dataset_logical_identity": json.loads(canonical_dataset_json(dataset.logical_identity)),
    }
    (destination / LOGICAL_NAME).write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return dataset


def rebuild_identity_from_role_files(directory: str | Path) -> tuple[str, str]:
    """Recompute both hashes from all three role files.

    Equivalence tests use this. Formal TRAIN does not.
    """

    root = Path(directory)
    access = FileRead()
    declaration = json.loads(access.read_text(root / DECLARATION_NAME))
    rows: list[Interaction] = []
    for role in ("train", "valid", "test"):
        rows.extend(_read_jsonl(access.read_text(root / _ROLE_FILES[role])))
    identity = _identity_for_rows(declaration, tuple(rows), root, access)
    logical = dataset_logical_hash(identity)
    return logical, _contract_for(declaration, logical)


def run_external_formal(
    directory: str | Path,
    adapter_spec: str,
    config: Mapping[str, Any],
    *,
    phase_state_semantics_id: str,
    runs_dir: str | Path,
    access: FileRead | None = None,
) -> FormalRun:
    """TRAIN, persist TRAIN-only bytes, then score VALID from a new restore."""

    if phase_state_semantics_id not in _PHASE_POLICIES:
        raise ExternalFormalError(
            f"unsupported phase_state_semantics_id {phase_state_semantics_id!r}"
        )
    reader = access or FileRead()
    root = Path(directory)
    declaration, _identity, logical_hash, contract_hash = _verify_observed(
        root, reader, ("train", "valid")
    )
    train_rows = _rows_for_role(root, declaration, "train", reader)
    valid_rows = _rows_for_role(root, declaration, "valid", reader)
    adapter = load_adapter(adapter_spec, config)
    _require_phase_support(adapter, phase_state_semantics_id)
    _validate_contract(adapter, declaration)
    provenance = adapter_provenance(adapter, config)
    if not provenance.formal_freeze_eligible:
        raise ExternalFormalError("implementation is not eligible for formal freeze")
    if adapter.contract_version == PSYCHOMETRIC_STREAMING_CONTRACT:
        adapter.execute_phase(train_rows, phase="train")
    elif adapter.contract_version == NEURAL_BUNDLE_CONTRACT:
        adapter.train(train_rows)
    else:
        raise ExternalFormalError(f"unsupported contract {adapter.contract_version!r}")
    locator = hashlib.sha256(
        f"{contract_hash}:{_implementation_hash(provenance)}".encode("utf-8")
    ).hexdigest()[:16]
    run_dir = Path(runs_dir) / f"formal_{locator}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / CHECKPOINT_NAME
    _save_family_state(adapter, checkpoint)
    snapshot = checkpoint.read_bytes()
    digest = hashlib.sha256(snapshot).hexdigest()
    fresh = load_adapter(adapter_spec, config)
    _restore_family_state(fresh, checkpoint)
    valid_output = _replay_valid(fresh, valid_rows)
    if checkpoint.read_bytes() != snapshot:
        raise ExternalFormalError("VALID replay changed the TRAIN-only checkpoint")
    aligned, metrics = _align_and_score(valid_rows, valid_output)
    trace_hash = valid_prediction_trace_hash(
        tuple((row["interaction_id"], float(row["probability"])) for row in aligned)
    )
    record = {
        "adapter_spec": adapter_spec,
        "config": dict(config),
        "phase_state_semantics_id": phase_state_semantics_id,
        "dataset_logical_hash": logical_hash,
        "dataset_contract_hash": contract_hash,
        "checkpoint_sha256": digest,
        "valid_prediction_trace_hash": trace_hash,
        "valid_metric_evidence": canonical_metric_json(metrics),
        "implementation_hash": _implementation_hash(provenance),
        "config_hash": provenance.identity["config_hash"],
    }
    (run_dir / RUN_NAME).write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return FormalRun(
        run_dir=run_dir,
        checkpoint_path=checkpoint,
        checkpoint_sha256=digest,
        dataset_logical_hash=logical_hash,
        dataset_contract_hash=contract_hash,
        valid_trace_hash=trace_hash,
        valid_metrics=dict(metrics),
        provenance=provenance,
        phase_state_semantics_id=phase_state_semantics_id,
        adapter_spec=adapter_spec,
        config=dict(config),
        files_read=_files_read(reader),
    )


def write_external_freeze(
    run: FormalRun,
    directory: str | Path,
    *,
    access: FileRead | None = None,
) -> dict[str, object]:
    """Revalidate implementation and evidence, then write one manifest."""

    reader = access or FileRead()
    destination = run.run_dir / FREEZE_NAME
    if destination.exists():
        raise ExternalFormalError(f"external freeze already exists: {destination}")
    root = Path(directory)
    current = adapter_provenance(load_adapter(run.adapter_spec, run.config), run.config)
    if not current.formal_freeze_eligible:
        raise ExternalFormalError("implementation is not eligible for formal freeze")
    if _implementation_hash(current) != _implementation_hash(run.provenance):
        raise ExternalFormalError("implementation identity changed before freeze")
    declaration, _identity, logical_hash, contract_hash = _verify_observed(
        root, reader, ("train", "valid")
    )
    if logical_hash != run.dataset_logical_hash or contract_hash != run.dataset_contract_hash:
        raise ExternalFormalError("dataset identity does not match the formal run")
    snapshot = reader.read_bytes(run.checkpoint_path)
    if hashlib.sha256(snapshot).hexdigest() != run.checkpoint_sha256:
        raise ExternalFormalError("TRAIN-only checkpoint hash changed before freeze")
    fresh = load_adapter(run.adapter_spec, run.config)
    _require_phase_support(fresh, run.phase_state_semantics_id)
    _restore_family_state(fresh, run.checkpoint_path)
    valid_rows = _rows_for_role(root, declaration, "valid", reader)
    aligned, metrics = _align_and_score(valid_rows, _replay_valid(fresh, valid_rows))
    trace_hash = valid_prediction_trace_hash(
        tuple((row["interaction_id"], float(row["probability"])) for row in aligned)
    )
    metric_json = canonical_metric_json(metrics)
    if trace_hash != run.valid_trace_hash or metric_json != canonical_metric_json(run.valid_metrics):
        raise ExternalFormalError("freeze-time VALID evidence does not match the formal run")
    if reader.read_bytes(run.checkpoint_path) != snapshot:
        raise ExternalFormalError("freeze-time VALID replay changed the checkpoint")
    manifest = _manifest(run, current, trace_hash=trace_hash, metric_json=metric_json)
    _create_new(destination, manifest)
    return manifest


def authorize_external_test(
    freeze_path: str | Path,
    directory: str | Path,
    adapter_spec: str,
    config: Mapping[str, Any],
    *,
    enable_test: bool,
    access: FileRead | None = None,
) -> dict[str, object]:
    """Open TEST only after freeze, exact VALID, and TRAIN-only bytes match.

    The TEST file is then checked against the sealed split commitment
    before any TEST metric is computed.
    """

    if enable_test is not True:
        raise ExternalFormalError("TEST execution requires enable_test=True")
    reader = access or FileRead()
    root = Path(directory)
    freeze = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    if freeze.get("identity_version") != EXTERNAL_EXPERIMENT_IDENTITY_VERSION:
        raise ExternalFormalError("freeze is not an external experiment identity")
    declaration, identity, logical_hash, contract_hash = _verify_observed(
        root, reader, ("train", "valid")
    )
    if (
        logical_hash != freeze["dataset_logical_hash"]
        or contract_hash != freeze["dataset_contract_hash"]
    ):
        raise ExternalFormalError("dataset identity does not match the freeze")
    adapter = load_adapter(adapter_spec, config)
    provenance = adapter_provenance(adapter, config)
    if _implementation_hash(provenance) != freeze["implementation_hash"]:
        raise ExternalFormalError("implementation identity does not match the freeze")
    if not provenance.formal_freeze_eligible:
        raise ExternalFormalError("implementation is no longer freeze-eligible")
    if provenance.identity["config_hash"] != freeze["config_hash"]:
        raise ExternalFormalError("recipe identity does not match the freeze")
    if freeze["validation_verification_semantics_id"] != VERIFICATION_EXACT:
        raise ExternalFormalError("v1 verifies only exact_replay_v1")
    if freeze["external_metric_semantics"] != EXTERNAL_METRIC_SEMANTICS:
        raise ExternalFormalError("metric semantics do not match this evaluator")
    checkpoint = Path(freeze["checkpoint_path"])
    snapshot = reader.read_bytes(checkpoint)
    checkpoint_digest = hashlib.sha256(snapshot).hexdigest()
    if checkpoint_digest != freeze["checkpoint_sha256"]:
        raise ExternalFormalError("checkpoint bytes do not match the freeze")
    _restore_family_state(adapter, checkpoint)
    valid_rows = _rows_for_role(root, declaration, "valid", reader)
    valid_output = _replay_valid(adapter, valid_rows)
    if hashlib.sha256(reader.read_bytes(checkpoint)).hexdigest() != checkpoint_digest:
        raise ExternalFormalError("VALID replay changed the TRAIN-only checkpoint")
    aligned, metrics = _align_and_score(valid_rows, valid_output)
    trace_hash = valid_prediction_trace_hash(
        tuple((item["interaction_id"], float(item["probability"])) for item in aligned)
    )
    if trace_hash != freeze["valid_prediction_trace_hash"]:
        raise ExternalFormalError("VALID prediction trace does not match the freeze")
    if canonical_metric_json(metrics) != freeze["valid_metric_evidence"]:
        raise ExternalFormalError("VALID metric evidence does not match the freeze")
    phase = freeze["phase_state_semantics_id"]
    if phase == PHASE_INDEPENDENT:
        adapter = load_adapter(adapter_spec, config)
        _restore_family_state(adapter, checkpoint)
    elif phase == PHASE_REPLAY_VALID:
        adapter._continue_verified_replay = True
    else:
        raise ExternalFormalError(f"unsupported phase_state_semantics_id {phase!r}")
    test_rows = _rows_for_role(root, declaration, "test", reader)
    _require_test_commitment(identity, declaration, test_rows)
    _aligned, test_metrics = _align_and_score(test_rows, _score_test(adapter, test_rows))
    artifact = {
        "identity_version": EXTERNAL_EXPERIMENT_IDENTITY_VERSION,
        "external_freeze_id": freeze["external_freeze_id"],
        "phase_state_semantics_id": phase,
        "publication": False,
        "formal_test": True,
        "metrics": test_metrics,
        "prediction_count": len(test_rows),
        "first_probability": _aligned[0]["probability"] if _aligned else None,
    }
    _create_new(Path(freeze_path).parent / TEST_RESULT_NAME, artifact)
    return artifact


def _replay_valid(adapter: Any, rows: tuple[Interaction, ...]):
    if adapter.contract_version == PSYCHOMETRIC_STREAMING_CONTRACT:
        return tuple(adapter.execute_phase(rows, phase="valid"))
    if adapter.contract_version == NEURAL_BUNDLE_CONTRACT:
        return tuple(adapter.replay(rows))
    raise ExternalFormalError(f"unsupported contract {adapter.contract_version!r}")


def _score_test(adapter: Any, rows: tuple[Interaction, ...]):
    if adapter.contract_version == PSYCHOMETRIC_STREAMING_CONTRACT:
        return tuple(adapter.execute_phase(rows, phase="valid"))
    if adapter.contract_version == NEURAL_BUNDLE_CONTRACT:
        if getattr(adapter, "_continue_verified_replay", False):
            return tuple(adapter.continue_replay(rows))
        return tuple(adapter.replay(rows))
    raise ExternalFormalError(f"unsupported contract {adapter.contract_version!r}")


def _manifest(
    run: FormalRun,
    provenance: AdapterProvenance,
    *,
    trace_hash: str,
    metric_json: str,
) -> dict[str, object]:
    implementation = _implementation_identity(provenance)
    identity = {
        "identity_version": EXTERNAL_EXPERIMENT_IDENTITY_VERSION,
        "dataset_logical_hash": run.dataset_logical_hash,
        "dataset_contract_hash": run.dataset_contract_hash,
        "implementation_hash": _implementation_hash(provenance),
        "implementation_identity": implementation,
        "config_hash": run.provenance.identity["config_hash"],
        "external_metric_semantics": EXTERNAL_METRIC_SEMANTICS,
        "phase_state_semantics_id": run.phase_state_semantics_id,
        "validation_verification_semantics_id": VERIFICATION_EXACT,
        "checkpoint_sha256": run.checkpoint_sha256,
        "checkpoint_format_id": provenance.identity["checkpoint_format_id"],
        "checkpoint_path": str(run.checkpoint_path),
        "valid_prediction_trace_hash": trace_hash,
        "valid_metric_evidence": metric_json,
    }
    identity["external_freeze_id"] = hashlib.sha256(
        _canonical_identity_json(identity).encode("utf-8")
    ).hexdigest()
    identity["descriptive_provenance"] = {
        "module_path": provenance.module_path,
        "source_mode": provenance.source_mode,
        "observed_module_sha256": provenance.observed_module_sha256,
        "formal_freeze_eligible": provenance.formal_freeze_eligible,
        "seed": run.config.get("seed"),
    }
    return identity


def _canonical_identity_json(identity: Mapping[str, object]) -> str:
    """Hash-bearing fields only. Paths and source mode are excluded."""

    implementation = _canonical_implementation_json(identity["implementation_identity"])  # type: ignore[arg-type]
    fields = (
        ("identity_version", _json_string(str(identity["identity_version"]))),
        ("dataset_logical_hash", _json_string(str(identity["dataset_logical_hash"]))),
        ("dataset_contract_hash", _json_string(str(identity["dataset_contract_hash"]))),
        ("implementation_hash", _json_string(str(identity["implementation_hash"]))),
        ("implementation_identity", implementation),
        ("config_hash", _json_string(str(identity["config_hash"]))),
        ("external_metric_semantics", _json_string(str(identity["external_metric_semantics"]))),
        ("phase_state_semantics_id", _json_string(str(identity["phase_state_semantics_id"]))),
        (
            "validation_verification_semantics_id",
            _json_string(str(identity["validation_verification_semantics_id"])),
        ),
        ("checkpoint_sha256", _json_string(str(identity["checkpoint_sha256"]))),
        ("checkpoint_format_id", _json_string(str(identity["checkpoint_format_id"]))),
        ("valid_prediction_trace_hash", _json_string(str(identity["valid_prediction_trace_hash"]))),
        ("valid_metric_evidence", str(identity["valid_metric_evidence"])),
    )
    return "{" + ",".join(f"{_json_string(key)}:{value}" for key, value in fields) + "}"


def _implementation_identity(provenance: AdapterProvenance) -> dict[str, Any]:
    family = _family(provenance.identity["adapter_contract_version"])
    body = {
        "adapter_contract_version": provenance.identity["adapter_contract_version"],
        "adapter_family": family,
        "module_name": provenance.identity["module_name"],
        "class_name": provenance.identity["class_name"],
        "distribution_name": provenance.identity["distribution_name"],
        "distribution_version": provenance.identity["distribution_version"],
        "implementation_fingerprint_kind": provenance.identity["implementation_fingerprint_kind"],
        "implementation_fingerprint_scope": provenance.identity["implementation_fingerprint_scope"],
        "implementation_fingerprint_value": provenance.identity["implementation_fingerprint_value"],
        "implementation_id": provenance.identity["implementation_id"],
        "checkpoint_format_id": provenance.identity["checkpoint_format_id"],
        "execution_semantics_id": provenance.identity["execution_semantics_id"],
        "traversal_id": provenance.identity["traversal_id"],
        "oov_state_semantics_id": provenance.identity["oov_state_semantics_id"],
    }
    return body


def _canonical_implementation_json(identity: Mapping[str, Any]) -> str:
    fields = []
    for key in _IMPLEMENTATION_KEYS:
        fields.append(f"{_json_string(key)}:{_json_optional(identity[key])}")
    return "{" + ",".join(fields) + "}"


def _implementation_hash(provenance: AdapterProvenance) -> str:
    return hashlib.sha256(
        _canonical_implementation_json(_implementation_identity(provenance)).encode("utf-8")
    ).hexdigest()


def _family(contract_version: str) -> str:
    if contract_version == PSYCHOMETRIC_STREAMING_CONTRACT:
        return "psychometric_streaming"
    if contract_version == NEURAL_BUNDLE_CONTRACT:
        return "neural_bundle"
    raise ExternalFormalError(f"unsupported contract {contract_version!r}")


def _verify_observed(root: Path, access: FileRead, roles: Sequence[str]):
    declaration = json.loads(access.read_text(root / DECLARATION_NAME))
    sealed = json.loads(access.read_text(root / LOGICAL_NAME))
    identity = _identity_from_document(sealed["dataset_logical_identity"])
    logical_hash = dataset_logical_hash(identity)
    if logical_hash != sealed["dataset_logical_hash"]:
        raise ExternalFormalError("sealed dataset logical hash does not match its document")
    contract_hash = _contract_for(declaration, logical_hash)
    if contract_hash != sealed["dataset_contract_hash"]:
        raise ExternalFormalError("sealed dataset contract hash does not match its document")
    roles_of = {str(split_id): str(role) for split_id, role in _pairs(declaration["split_roles"])}
    split_of = {role: split_id for split_id, role in roles_of.items()}
    stored = {split.split_id: split.logical_hash for split in identity.splits}
    for role in roles:
        split_id = split_of[role]
        rows = _read_jsonl(access.read_text(root / _ROLE_FILES[role]))
        if logical_split_hash(rows) != stored[split_id]:
            raise ExternalFormalError(f"{role} rows do not match the sealed split hash")
    return declaration, identity, logical_hash, contract_hash


def _require_test_commitment(identity: DatasetLogicalIdentity, declaration: Mapping[str, Any], rows) -> None:
    """Reject TEST rows whose hash or count differs from the sealed split."""

    split_of = {str(role): str(split_id) for split_id, role in _pairs(declaration["split_roles"])}
    split_id = split_of["test"]
    expected = next((split for split in identity.splits if split.split_id == split_id), None)
    observed_hash = logical_split_hash(rows)
    observed_count = len(rows)
    expected_hash = None if expected is None else expected.logical_hash
    expected_count = None if expected is None else expected.row_count
    if observed_hash != expected_hash or observed_count != expected_count:
        raise ExternalFormalError(
            "TEST split content mismatch; "
            f"expected logical_hash={expected_hash} row_count={expected_count}; "
            f"observed logical_hash={observed_hash} row_count={observed_count}"
        )


def _rows_for_role(root: Path, declaration: Mapping[str, Any], role: str, access: FileRead):
    split_of = {str(item_role): str(split_id) for split_id, item_role in _pairs(declaration["split_roles"])}
    split_id = split_of[role]
    rows = _read_jsonl(access.read_text(root / _ROLE_FILES[role]))
    foreign = [row.interaction_id for row in rows if row.split != split_id]
    if foreign:
        raise ExternalFormalError(f"{role} file contains another split: {foreign}")
    return rows


def _identity_for_rows(declaration, rows: tuple[Interaction, ...], root: Path, access: FileRead):
    from kclearner.data.logical_identity import hash_ordered_ids

    learners = _read_ids(access.read_text(root / "learner_ids.txt"))
    items = _read_ids(access.read_text(root / "item_ids.txt"))
    kcs = _read_ids(access.read_text(root / "kc_ids.txt"))
    ordered = _ordered_split_roles(declaration)
    grouped = {split_id: [] for split_id, _role in ordered}
    for row in rows:
        grouped[row.split].append(row)
    splits = tuple(
        SplitLogicalIdentity(split_id, logical_split_hash(tuple(grouped[split_id])), len(grouped[split_id]))
        for split_id, _role in ordered
    )
    return DatasetLogicalIdentity(
        dataset_id=str(declaration["dataset_id"]),
        protocol_id=str(declaration["protocol_id"]),
        preprocessing_version=str(declaration["preprocessing_version"]),
        cohort_hash=hash_ordered_ids(learners),
        splits=splits,
        item_vocabulary_hash=hash_ordered_ids(items),
        kc_vocabulary_hash=hash_ordered_ids(kcs),
    )


def _identity_from_document(payload: Mapping[str, Any]) -> DatasetLogicalIdentity:
    return DatasetLogicalIdentity(
        dataset_id=str(payload["dataset_id"]),
        protocol_id=str(payload["protocol_id"]),
        preprocessing_version=str(payload["preprocessing_version"]),
        cohort_hash=payload.get("cohort_hash"),
        splits=tuple(
            SplitLogicalIdentity(str(item["split_id"]), str(item["logical_hash"]), int(item["row_count"]))
            for item in payload["splits"]
        ),
        item_vocabulary_hash=payload.get("item_vocabulary_hash"),
        kc_vocabulary_hash=payload.get("kc_vocabulary_hash"),
    )


def _contract_for(declaration: Mapping[str, Any], logical_hash: str) -> str:
    return dataset_contract_hash(
        dataset_logical_hash_value=logical_hash,
        canonical_schema_version=str(declaration["canonical_schema_version"]),
        bundle_semantics_id=str(declaration["bundle_semantics_id"]),
        oov_representation_semantics_id=str(declaration["oov_representation_semantics_id"]),
        metric_mask_semantics=str(declaration["metric_mask_semantics"]),
        split_roles=_ordered_split_roles(declaration),
    )


def _ordered_split_roles(declaration: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    roles = {str(split_id): str(role) for split_id, role in _pairs(declaration["split_roles"])}
    return tuple((str(split_id), roles[str(split_id)]) for split_id in declaration["split_order"])


def _pairs(value: Any) -> tuple[tuple[str, str], ...]:
    if isinstance(value, dict):
        return tuple((str(key), str(item)) for key, item in value.items())
    return tuple((str(split_id), str(role)) for split_id, role in value)


def _require_phase_support(adapter: Any, phase_state_semantics_id: str) -> None:
    supported = tuple(getattr(adapter, "supported_phase_state_semantics", ()))
    if phase_state_semantics_id not in supported:
        raise ExternalFormalError(
            f"adapter does not declare compatibility with {phase_state_semantics_id!r}"
        )


def _validate_contract(adapter: Any, declaration: Mapping[str, Any]) -> None:
    class _View:
        oov_representation_semantics_id = declaration["oov_representation_semantics_id"]

    try:
        adapter.validate_dataset_contract(_View())
    except Exception as exc:
        raise ExternalFormalError(
            f"dataset OOV representation is incompatible; TRAIN was not started: {exc}"
        ) from exc


def _save_family_state(adapter: Any, path: Path) -> None:
    if adapter.contract_version == PSYCHOMETRIC_STREAMING_CONTRACT:
        adapter.save_state(path)
        return
    adapter.save_checkpoint(path)


def _restore_family_state(adapter: Any, path: Path) -> None:
    if adapter.contract_version == PSYCHOMETRIC_STREAMING_CONTRACT:
        adapter.load_state(path)
        return
    adapter.load_checkpoint(path)


def _files_read(access: FileRead) -> tuple[str, ...]:
    recorded = getattr(access, "names", None)
    if recorded is None:
        return ()
    return tuple(recorded)


def _read_jsonl(text: str) -> tuple[Interaction, ...]:
    if text == "":
        return ()
    rows = []
    for line in text.splitlines():
        if line == "":
            continue
        rows.append(Interaction.from_dict(json.loads(line)))
    return tuple(rows)


def _read_ids(text: str) -> tuple[str, ...]:
    if text == "":
        return ()
    return tuple(line for line in text.splitlines() if line != "")


def _write_jsonl(path: Path, rows: Sequence[Interaction]) -> None:
    path.write_text(
        "".join(json.dumps(row.to_dict(), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _json_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_optional(value: Any) -> str:
    if value is None:
        return "null"
    return _json_string(str(value))


def _create_new(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.incomplete.{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        temporary.unlink(missing_ok=True)
        raise ExternalFormalError(f"external artifact already exists: {path}") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(temporary.read_text(encoding="utf-8"))
    finally:
        temporary.unlink(missing_ok=True)
