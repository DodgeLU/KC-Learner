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
            "seed_role": (
                "recorded_for_provenance_only"
                if self.deterministic
                else "publication_seed"
            ),
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


def load_protocol(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("protocol_id") != PROTOCOL_NAME:
        raise ConfigError(
            f"protocol_id {payload.get('protocol_id')!r} != {PROTOCOL_NAME}"
        )
    return payload


def resolve_run(protocol: Mapping[str, Any], model: str, seed: int | None = None) -> RunConfig:
    if model not in protocol["models"]:
        raise ConfigError(f"unknown model {model!r}")
    family = protocol["models"][model]
    deterministic = bool(family["deterministic"])
    if deterministic:
        chosen = int(family["provenance_seed"])
        if seed is not None and int(seed) != chosen:
            raise ConfigError(
                f"{model} is deterministic; provenance seed is {chosen}"
            )
    else:
        if seed is None:
            raise ConfigError(f"{model} requires a publication seed")
        chosen = int(seed)
        if chosen not in PUBLICATION_SEEDS:
            raise ConfigError(f"seed {chosen} is outside {PUBLICATION_SEEDS}")
    run_id = f"ednet_corrected_{model}" if deterministic else f"ednet_corrected_{model}_seed{chosen}"
    return RunConfig(
        model=model,
        seed=chosen,
        deterministic=deterministic,
        run_id=run_id,
        protocol=dict(protocol),
    )


def formal_run_matrix(protocol: Mapping[str, Any]) -> tuple[RunConfig, ...]:
    runs: list[RunConfig] = []
    for model in ALL_MODELS:
        if protocol["models"][model]["deterministic"]:
            runs.append(resolve_run(protocol, model))
        else:
            for seed in PUBLICATION_SEEDS:
                runs.append(resolve_run(protocol, model, seed))
    return tuple(runs)
