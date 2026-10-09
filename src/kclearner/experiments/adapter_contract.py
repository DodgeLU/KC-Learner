"""Two execution-family contracts for adapters outside the six references.

The bundled IRT, AR-KT, DKT, and DKVMN classes are not migrated here.
Their mathematics and reference runners stay on the existing paths.
Seeded learner shuffle, TBPTT chunks, and pyKT layer names belong to
that bundled neural recipe, not to this contract.

A concrete implementation owns parameters and low-level transitions.
An adapter owns how that implementation is traversed, how a bundle is
predicted before it updates state, and which probabilities KC-Learner
scores. The generic runner owns split roles, metric masks, and the
TRAIN-then-VALID order. An adapter does not select splits or open TEST.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata as importlib_metadata
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROVENANCE_IDENTITY_VERSION = "external_adapter_provenance_v1"
PSYCHOMETRIC_STREAMING_CONTRACT = "psychometric_streaming_adapter_v1"
NEURAL_BUNDLE_CONTRACT = "neural_bundle_adapter_v1"

_COMMON_STRINGS = (
    "implementation_id",
    "checkpoint_format_id",
    "traversal_id",
    "oov_state_semantics_id",
    "execution_semantics_id",
)
_PSYCHOMETRIC_METHODS = (
    "execute_phase",
    "save_state",
    "load_state",
    "validate_dataset_contract",
)
_NEURAL_METHODS = (
    "train",
    "replay",
    "save_checkpoint",
    "load_checkpoint",
    "validate_dataset_contract",
)

# Ordinary installed distributions. An editable checkout is not in this set.
_VERSIONED_NON_EDITABLE = "versioned_non_editable"
_EDITABLE = "editable"
_LOCAL = "local"
_UNVERSIONED = "unversioned"
FINGERPRINT_KIND_DISTRIBUTION = "installed_distribution"
FINGERPRINT_SCOPE_DISTRIBUTION = "distribution_version"
FINGERPRINT_KIND_MODULE = "module_source_sha256"
FINGERPRINT_SCOPE_SINGLE_MODULE = "single_module"
FINGERPRINT_KIND_SUPPLIED = "adapter_supplied"
FINGERPRINT_SCOPE_COMPLETE = "complete_implementation"
_SUFFICIENT_MUTABLE_FINGERPRINTS = (
    (FINGERPRINT_KIND_MODULE, FINGERPRINT_SCOPE_SINGLE_MODULE),
    (FINGERPRINT_KIND_SUPPLIED, FINGERPRINT_SCOPE_COMPLETE),
)


class AdapterContractError(ValueError):
    """An object does not satisfy an execution-family contract."""


@dataclass(frozen=True)
class AdapterProvenance:
    """Identity-bearing adapter record plus descriptive fields.

    ``formal_freeze_eligible`` is a gate derived from the identity
    fields. It is not itself an input to ``provenance_hash``. Absolute
    module paths stay in ``module_path`` and are not hashed.
    """

    identity: dict[str, Any]
    module_path: str | None
    source_mode: str
    observed_module_sha256: str | None
    provenance_hash: str
    formal_freeze_eligible: bool

    def identity_json(self) -> str:
        return canonical_provenance_json(self.identity)


def load_adapter(spec: str, config: Mapping[str, Any]):
    """Construct ``module:Class`` with ``config``.

    The class is not looked up in ``ALL_MODELS`` or ``create_model``.
    ``config`` must declare ``oov_state_semantics_id``. This loader does
    not choose one.
    """

    module_name, class_name = _split_spec(spec)
    module = importlib.import_module(module_name)
    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise AdapterContractError(f"{spec} does not name a class") from exc
    if not isinstance(config, Mapping):
        raise AdapterContractError("adapter config must be a mapping")
    if "oov_state_semantics_id" not in config:
        raise AdapterContractError(
            "adapter config must declare oov_state_semantics_id; "
            "the loader does not select a state-update policy"
        )
    adapter = cls(dict(config))
    if getattr(adapter, "contract_version", None) == PSYCHOMETRIC_STREAMING_CONTRACT:
        require_psychometric_streaming_adapter(adapter)
    elif getattr(adapter, "contract_version", None) == NEURAL_BUNDLE_CONTRACT:
        require_neural_bundle_adapter(adapter)
    else:
        raise AdapterContractError(
            f"{spec} contract_version {getattr(adapter, 'contract_version', None)!r} "
            "is not a KC-Learner execution-family contract"
        )
    if adapter.oov_state_semantics_id != config["oov_state_semantics_id"]:
        raise AdapterContractError(
            "adapter oov_state_semantics_id does not match the declared config"
        )
    return adapter


def require_psychometric_streaming_adapter(adapter: Any) -> None:
    """Check the operations IRT/AR-KT already expose, in canonical form.

    Required behavior is construction from config, ``execute_phase`` for
    ``train`` and ``valid``, bundle predict-before-update inside that
    call, probability rows aligned to ``interaction_id``, and
    ``save_state`` / ``load_state``. Shared and learner-local arrays stay
    inside the adapter. pyKT backend fields are not required.
    """

    _require_common(adapter, PSYCHOMETRIC_STREAMING_CONTRACT)
    _require_methods(adapter, _PSYCHOMETRIC_METHODS)


def require_neural_bundle_adapter(adapter: Any) -> None:
    """Check bundle replay operations without a pyKT backend.

    ``train`` and ``replay`` belong to the adapter. Seeded shuffle and
    TBPTT remain inside the bundled neural recipe, not this interface.
    """

    _require_common(adapter, NEURAL_BUNDLE_CONTRACT)
    _require_methods(adapter, _NEURAL_METHODS)


def adapter_provenance(adapter: Any, config: Mapping[str, Any]) -> AdapterProvenance:
    """Build provenance for one constructed adapter.

    A non-editable installed distribution version can identify the
    implementation. An editable or local source also needs a source
    fingerprint, because the version string can stay still while the
    files change. ``source_mode`` is descriptive and is not hashed.
    This function does not write a freeze manifest.
    """

    module_name, class_name = _type_name(adapter)
    module = sys.modules.get(module_name)
    distribution_name, distribution_version = _distribution(module_name)
    observed_module_sha256 = _source_fingerprint(module)
    source_mode = _source_mode(
        distribution_name, distribution_version, observed_module_sha256
    )
    fingerprint_kind, fingerprint_scope, fingerprint_value = _accepted_fingerprint(
        adapter,
        source_mode=source_mode,
        distribution_version=distribution_version,
        observed_module_sha256=observed_module_sha256,
    )
    identity = {
        "provenance_identity_version": PROVENANCE_IDENTITY_VERSION,
        "adapter_contract_version": adapter.contract_version,
        "module_name": module_name,
        "class_name": class_name,
        "distribution_name": distribution_name,
        "distribution_version": distribution_version,
        "implementation_id": adapter.implementation_id,
        "implementation_fingerprint_kind": fingerprint_kind,
        "implementation_fingerprint_scope": fingerprint_scope,
        "implementation_fingerprint_value": fingerprint_value,
        "config_hash": config_hash(config),
        "checkpoint_format_id": adapter.checkpoint_format_id,
        "oov_state_semantics_id": adapter.oov_state_semantics_id,
        "traversal_id": adapter.traversal_id,
        "execution_semantics_id": adapter.execution_semantics_id,
    }
    path = None
    file_name = getattr(module, "__file__", None) if module is not None else None
    if isinstance(file_name, str) and file_name != "":
        path = str(Path(file_name))
    return AdapterProvenance(
        identity=identity,
        module_path=path,
        source_mode=source_mode,
        observed_module_sha256=observed_module_sha256,
        provenance_hash=provenance_hash(identity),
        formal_freeze_eligible=freeze_eligible(
            source_mode=source_mode,
            distribution_version=distribution_version,
            fingerprint_kind=fingerprint_kind,
            fingerprint_scope=fingerprint_scope,
            fingerprint_value=fingerprint_value,
        ),
    )


def freeze_eligible(
    *,
    source_mode: str,
    distribution_version: str | None,
    fingerprint_kind: str | None,
    fingerprint_scope: str | None,
    fingerprint_value: str | None,
) -> bool:
    """Return whether a future formal freeze may bind this implementation.

    A non-editable installed distribution is identified by its version.
    Editable, local, and unversioned code need a fingerprint whose kind
    and scope are explicitly sufficient. A single module hash is enough
    only when the adapter declares ``single_module`` coverage. It is not
    treated as the fingerprint of a multi-module implementation.
    """

    if source_mode == _VERSIONED_NON_EDITABLE:
        return (
            distribution_version is not None
            and fingerprint_kind == FINGERPRINT_KIND_DISTRIBUTION
            and fingerprint_scope == FINGERPRINT_SCOPE_DISTRIBUTION
        )
    if source_mode not in (_EDITABLE, _LOCAL, _UNVERSIONED):
        return False
    if not fingerprint_value or not fingerprint_kind or not fingerprint_scope:
        return False
    return (fingerprint_kind, fingerprint_scope) in _SUFFICIENT_MUTABLE_FINGERPRINTS


def source_mode_from_direct_url(
    direct_url: Mapping[str, Any] | None,
    *,
    has_distribution: bool,
    has_version: bool,
) -> str:
    """Classify PEP 610 ``direct_url.json`` without inventing a version."""

    if not has_distribution or not has_version:
        return _UNVERSIONED
    if not isinstance(direct_url, Mapping):
        return _VERSIONED_NON_EDITABLE
    dir_info = direct_url.get("dir_info")
    if isinstance(dir_info, Mapping) and dir_info.get("editable") is True:
        return _EDITABLE
    return _VERSIONED_NON_EDITABLE


def config_hash(config: Mapping[str, Any]) -> str:
    """SHA-256 of the supplied config. Key order is sorted for stability."""

    payload = json.dumps(dict(config), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_provenance_json(identity: Mapping[str, Any]) -> str:
    """Compact provenance identity. Paths are not accepted."""

    keys = (
        "provenance_identity_version",
        "adapter_contract_version",
        "module_name",
        "class_name",
        "distribution_name",
        "distribution_version",
        "implementation_id",
        "implementation_fingerprint_kind",
        "implementation_fingerprint_scope",
        "implementation_fingerprint_value",
        "config_hash",
        "checkpoint_format_id",
        "oov_state_semantics_id",
        "traversal_id",
        "execution_semantics_id",
    )
    missing = [key for key in keys if key not in identity]
    if missing:
        raise AdapterContractError(f"provenance identity is missing {missing}")
    extra = [key for key in identity if key not in keys]
    if extra:
        raise AdapterContractError(f"provenance identity has non-identity fields {extra}")
    return "{" + ",".join(
        f"{_json_string(key)}:{_json_value(identity[key])}" for key in keys
    ) + "}"


def provenance_hash(identity: Mapping[str, Any]) -> str:
    """SHA-256 of :func:`canonical_provenance_json`."""

    return hashlib.sha256(canonical_provenance_json(identity).encode("utf-8")).hexdigest()


def _require_common(adapter: Any, contract_version: str) -> None:
    if getattr(adapter, "contract_version", None) != contract_version:
        raise AdapterContractError(
            f"contract_version {getattr(adapter, 'contract_version', None)!r} "
            f"!= {contract_version}"
        )
    for name in _COMMON_STRINGS:
        value = getattr(adapter, name, None)
        if not isinstance(value, str) or value == "":
            raise AdapterContractError(f"{name} must be a non-empty string")


def _require_methods(adapter: Any, names: Sequence[str]) -> None:
    for name in names:
        if not callable(getattr(adapter, name, None)):
            raise AdapterContractError(f"adapter is missing callable {name}")


def _split_spec(spec: str) -> tuple[str, str]:
    if spec.count(":") != 1:
        raise AdapterContractError(
            "adapter spec must be module:Class, "
            f"got {spec!r}"
        )
    module_name, class_name = spec.split(":")
    if module_name == "" or class_name == "":
        raise AdapterContractError(f"adapter spec {spec!r} is incomplete")
    return module_name, class_name


def _type_name(adapter: Any) -> tuple[str, str]:
    module_name = type(adapter).__module__
    class_name = type(adapter).__qualname__
    if not module_name or not class_name:
        raise AdapterContractError("adapter class has no module or class name")
    return module_name, class_name


def _distribution(module_name: str) -> tuple[str | None, str | None]:
    top = module_name.split(".", 1)[0]
    names = _package_distributions().get(top) or []
    if not names:
        return None, None
    distribution_name = str(names[0])
    try:
        version = importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return distribution_name, None
    if not isinstance(version, str) or version == "":
        return distribution_name, None
    return distribution_name, version


def _package_distributions() -> Mapping[str, list[str]]:
    global _PACKAGE_DISTRIBUTIONS
    if _PACKAGE_DISTRIBUTIONS is None:
        try:
            _PACKAGE_DISTRIBUTIONS = importlib_metadata.packages_distributions()
        except Exception:
            _PACKAGE_DISTRIBUTIONS = {}
    return _PACKAGE_DISTRIBUTIONS


def _source_mode(
    distribution_name: str | None,
    distribution_version: str | None,
    fingerprint: str | None,
) -> str:
    if distribution_name is None or distribution_version is None:
        return _LOCAL if fingerprint is not None else _UNVERSIONED
    return source_mode_from_direct_url(
        _direct_url(distribution_name),
        has_distribution=True,
        has_version=True,
    )


def _direct_url(distribution_name: str) -> dict[str, Any] | None:
    try:
        dist = importlib_metadata.distribution(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return None
    text = dist.read_text("direct_url.json")
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


_PACKAGE_DISTRIBUTIONS: Mapping[str, list[str]] | None = None


def _accepted_fingerprint(
    adapter: Any,
    *,
    source_mode: str,
    distribution_version: str | None,
    observed_module_sha256: str | None,
) -> tuple[str | None, str | None, str | None]:
    """Return kind, scope, and value that may enter implementation identity.

    A module hash is accepted only when the adapter declares
    ``fingerprint_coverage = "single_module"``. Otherwise a local
    wrapper file is not described as the whole implementation.
    """

    if (
        source_mode == _VERSIONED_NON_EDITABLE
        and distribution_version is not None
    ):
        return (
            FINGERPRINT_KIND_DISTRIBUTION,
            FINGERPRINT_SCOPE_DISTRIBUTION,
            None,
        )
    coverage = getattr(adapter, "fingerprint_coverage", None)
    if coverage == FINGERPRINT_SCOPE_SINGLE_MODULE and observed_module_sha256:
        return (
            FINGERPRINT_KIND_MODULE,
            FINGERPRINT_SCOPE_SINGLE_MODULE,
            observed_module_sha256,
        )
    if coverage == FINGERPRINT_SCOPE_COMPLETE:
        supplied = getattr(adapter, "implementation_fingerprint", None)
        value = supplied() if callable(supplied) else supplied
        if isinstance(value, str) and value != "":
            return FINGERPRINT_KIND_SUPPLIED, FINGERPRINT_SCOPE_COMPLETE, value
    return None, None, None


def _source_fingerprint(module: Any) -> str | None:
    if module is None:
        return None
    file_name = getattr(module, "__file__", None)
    if not isinstance(file_name, str) or file_name == "":
        return None
    path = Path(file_name)
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_value(value: Any) -> str:
    if value is None:
        return "null"
    if not isinstance(value, str):
        raise AdapterContractError(
            f"provenance identity values must be str or null, got {type(value).__name__}"
        )
    return _json_string(value)
