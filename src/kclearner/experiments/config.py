"""Frozen corrected-EdNet run configs.

One protocol file holds the shared identity. A run is that protocol
plus a model name and, for neural models, a seed. Psychometric models
record seed 42 as provenance only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

PROTOCOL_NAME = "ednet_corrected_publication_v1"
DATASET_PROTOCOL_IDS = {
    "ednet_kt1": PROTOCOL_NAME,
    "assistments2017": "ASSISTMENTS2017_MAIN_V1",
}
NEURAL_MODELS = ("dkt_q", "dkt_qc", "dkvmn_q", "dkvmn_qc")
PSYCHOMETRIC_MODELS = ("irt", "ar_kt")
ALL_MODELS = PSYCHOMETRIC_MODELS + NEURAL_MODELS
PUBLICATION_SEEDS = (42, 43, 44)


class ConfigError(ValueError):
    """The run config does not match the frozen protocol."""


@dataclass(frozen=True)
class RunConfig:
    model: str
    seed: int
    deterministic: bool
    run_id: str
    protocol: dict[str, Any]
    publication_protocol: bool = True
    seed_role: str = "publication_seed"

    def to_dict(self) -> dict[str, Any]:
        family = self.protocol["models"][self.model]
        return {
            "protocol_id": self.protocol["protocol_id"],
            "run_id": self.run_id,
            "model": self.model,
            "family": family["family"],
            "kc_aware": bool(family["kc_aware"]),
            "seed": self.seed,
            "deterministic": self.deterministic,
            "publication_protocol": bool(self.publication_protocol),
            "seed_role": self.seed_role,
            "recipe": family["recipe"],
            "dataset": self.protocol["dataset"],
            "vocabulary": self.protocol["vocabulary"],
            "sequence": self.protocol["sequence"],
            "metrics": self.protocol["metrics"],
            "phases": self.protocol["phases"],
        }

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def load_protocol(path: str | Path, *, dataset: str | None = None) -> dict[str, Any]:
    """Load one protocol file for ``dataset``.

    Omitting ``dataset`` keeps the historical EdNet caller behavior.
    A file for the other dataset is rejected rather than inferred.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    protocol_id = payload.get("protocol_id")
    if dataset is None:
        expected = PROTOCOL_NAME
        owner = "ednet_kt1"
    else:
        if dataset not in DATASET_PROTOCOL_IDS:
            raise ConfigError(f"unknown dataset {dataset!r}")
        expected = DATASET_PROTOCOL_IDS[dataset]
        owner = dataset
    if protocol_id != expected:
        raise ConfigError(
            f"protocol_id {protocol_id!r} != {expected} for dataset {owner}"
        )
    return payload


def publication_seeds(protocol: Mapping[str, Any]) -> tuple[int, ...]:
    """Closed publication-seed set.

    ASSISTments neural publication seeds are ``[42, 43, 44, 45, 46]``.
    EdNet neural publication seeds are ``[42, 43, 44]``.
    A formal run rejects every seed outside that set. A nonpublication
    execution may record another integer seed, and that run has
    ``publication_protocol`` false.
    """

    recorded = protocol.get("publication_seeds")
    if recorded is None:
        return PUBLICATION_SEEDS
    return tuple(int(seed) for seed in recorded)


def resolve_run(
    protocol: Mapping[str, Any],
    model: str,
    seed: int | None = None,
    *,
    publication_protocol: bool = True,
) -> RunConfig:
    """Resolve one run.

    ``publication_protocol`` is true only for the frozen formal recipe.
    Dry-run, smoke, and an exploratory seed pass false. Psychometric
    models stay on their provenance seed in every mode.
    """

    if model not in protocol["models"]:
        raise ConfigError(f"unknown model {model!r}")
    family = protocol["models"][model]
    deterministic = bool(family["deterministic"])
    formal = bool(publication_protocol)
    if deterministic:
        chosen = int(family["provenance_seed"])
        if seed is not None and int(seed) != chosen:
            raise ConfigError(
                f"{model} is deterministic; provenance seed is {chosen}"
            )
        role = "recorded_for_provenance_only"
    else:
        if seed is None:
            raise ConfigError(f"{model} requires a seed")
        chosen = int(seed)
        allowed = publication_seeds(protocol)
        if formal and chosen not in allowed:
            raise ConfigError(f"seed {chosen} is outside {allowed}")
        role = "publication_seed" if chosen in allowed else "exploratory_seed"
    prefix = str(protocol.get("run_id_prefix") or "ednet_corrected")
    run_id = f"{prefix}_{model}" if deterministic else f"{prefix}_{model}_seed{chosen}"
    return RunConfig(
        model=model,
        seed=chosen,
        deterministic=deterministic,
        run_id=run_id,
        protocol=dict(protocol),
        publication_protocol=formal,
        seed_role=role,
    )


def formal_run_matrix(protocol: Mapping[str, Any]) -> tuple[RunConfig, ...]:
    runs: list[RunConfig] = []
    for model in ALL_MODELS:
        if protocol["models"][model]["deterministic"]:
            runs.append(resolve_run(protocol, model))
        else:
            for seed in publication_seeds(protocol):
                runs.append(resolve_run(protocol, model, seed))
    return tuple(runs)
