"""Public TRAIN/VALID orchestration.

EdNet and ASSISTments share this entry point. Each dataset keeps its
own prepared-data reader. Dry-run checks identity and can construct a
model. Smoke writes a marked-non-publication artifact. ``--authorize-formal``
is TRAIN then VALID. TEST scoring is not started here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from kclearner import __version__
from kclearner.data.ednet_corrected import cohort_id_hash, file_sha256
from kclearner.experiments.config import NEURAL_MODELS, RunConfig
from kclearner.experiments.environment import environment_report, execution_device
from kclearner.experiments.factory import create_model
from kclearner.models.dkt import DKTRow
from kclearner.models.metrics import metric_auc, metric_brier, metric_nll
from kclearner.models.streaming import StreamingRow

NOT_FOR_PUBLICATION = "NOT FOR PUBLICATION"


class FormalRunBlocked(RuntimeError):
    """Full corrected-EdNet training or TEST scoring is not authorized."""


class IdentityError(RuntimeError):
    """A dataset or vocabulary manifest does not match the frozen protocol."""


@dataclass(frozen=True)
class PublicationVocab:
    item_ids: tuple[str, ...]
    kc_ids: tuple[int, ...]
    learner_ids: tuple[str, ...]

    def item_index(self) -> dict[str, int]:
        return {item: index for index, item in enumerate(self.item_ids)}

    def kc_index(self) -> dict[int, int]:
        return {kc: index for index, kc in enumerate(self.kc_ids)}

    def learner_index(self) -> dict[str, int]:
        return {learner: index for index, learner in enumerate(self.learner_ids)}


def masked_metrics(
    correct: Sequence[int],
    probability: Sequence[float],
    metric_mask: Sequence[bool],
) -> dict[str, float] | None:
    """Score rows with ``metric_mask`` true using the existing metric functions."""

    y = np.asarray(
        [value for value, keep in zip(correct, metric_mask) if keep],
        dtype=np.float64,
    )
    p = np.asarray(
        [value for value, keep in zip(probability, metric_mask) if keep],
        dtype=np.float64,
    )
    if y.size == 0:
        return None
    return {
        "nll": metric_nll(y, p),
        "brier": metric_brier(y, p),
        "auc": metric_auc(y, p),
        "n": int(y.size),
    }


def verify_dataset_manifest(
    manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    dataset_dir: str | Path | None = None,
) -> None:
    expected = protocol["dataset"]
    if manifest.get("preprocessing_version") != expected["dataset_id"]:
        raise IdentityError("dataset preprocessing version mismatch")
    if manifest.get("cohort_id") != expected["cohort_id"]:
        raise IdentityError("cohort_id mismatch")
    if manifest.get("ordered_learner_ids_sha256") != expected["cohort_sha256"]:
        raise IdentityError("cohort hash mismatch")
    counts = manifest.get("row_counts") or {}
    for split, count in expected["row_counts"].items():
        if int(counts.get(split, -1)) != int(count):
            raise IdentityError(f"{split} row count {counts.get(split)} != {count}")
    recorded = manifest.get("files_sha256") or {}
    expected_files = expected.get("files_sha256") or {}
    byte_match = all(recorded.get(name) == digest for name, digest in expected_files.items())
    if byte_match:
        return
    logical = manifest.get("dataset_logical_identity")
    if not isinstance(logical, dict):
        name = next(iter(expected_files), "parquet")
        raise IdentityError(f"{name} manifest hash mismatch")
    if logical.get("cohort_hash") != expected["cohort_sha256"]:
        raise IdentityError("logical cohort hash mismatch")
    vocabulary = protocol["vocabulary"]
    if logical.get("item_vocabulary_hash") != vocabulary["item_ids_sha256"]:
        raise IdentityError("logical item vocabulary hash mismatch")
    if logical.get("kc_vocabulary_hash") != vocabulary["kc_ids_sha256"]:
        raise IdentityError("logical KC vocabulary hash mismatch")
    splits = {
        item.get("split_id"): item
        for item in logical.get("splits") or []
        if isinstance(item, dict)
    }
    for split, count in expected["row_counts"].items():
        if split == "eligible":
            continue
        got = splits.get(split) or {}
        if int(got.get("row_count", -1)) != int(count):
            raise IdentityError(f"logical {split} row count mismatch")
    if dataset_dir is None:
        raise IdentityError("logical identity requires the prepared directory")
    for name, digest in expected_files.items():
        del digest
        actual = file_sha256(Path(dataset_dir) / f"{name}.parquet")
        if recorded.get(name) != actual:
            raise IdentityError(f"{name}.parquet byte hash mismatch")


def verify_vocab_file(payload: Mapping[str, Any], protocol: Mapping[str, Any]) -> PublicationVocab:
    expected = protocol["vocabulary"]
    items = tuple(payload["item_ids"])
    kcs = tuple(int(value) for value in payload["kc_ids"])
    if len(items) != int(expected["item_count"]):
        raise IdentityError("item vocabulary size mismatch")
    if len(kcs) != int(expected["kc_count"]):
        raise IdentityError("KC vocabulary size mismatch")
    if cohort_id_hash(items) != expected["item_ids_sha256"]:
        raise IdentityError("item vocabulary hash mismatch")
    if cohort_id_hash(tuple(str(kc) for kc in kcs)) != expected["kc_ids_sha256"]:
        raise IdentityError("KC vocabulary hash mismatch")
    if -1 in kcs:
        raise IdentityError("publication KC vocabulary still contains -1")
    return PublicationVocab(items, kcs, ())


def attach_learners(vocab: PublicationVocab, learner_ids: Sequence[str], protocol: Mapping[str, Any]) -> PublicationVocab:
    ordered = tuple(learner_ids)
    expected = protocol["vocabulary"]
    if len(ordered) != int(expected["learner_count"]):
        raise IdentityError("learner vocabulary size mismatch")
    if cohort_id_hash(ordered) != expected["learner_ids_sha256"]:
        raise IdentityError("learner vocabulary hash mismatch")
    return PublicationVocab(vocab.item_ids, vocab.kc_ids, ordered)


def dry_run(
    config: RunConfig,
    *,
    dataset_dir: str | Path,
    runs_dir: str | Path,
    construct_model: bool = True,
    requested_device: str = "cpu",
) -> dict[str, Any]:
    """Validate identity and construct the model. Does not train."""

    if config.protocol.get("protocol_id") == "ASSISTMENTS2017_MAIN_V1":
        from kclearner.experiments.assistments_run import dry_run_assistments

        return dry_run_assistments(
            config,
            dataset_dir=dataset_dir,
            runs_dir=runs_dir,
            construct_model=construct_model,
            requested_device=requested_device,
        )
    dataset = Path(dataset_dir)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    verify_dataset_manifest(manifest, config.protocol, dataset_dir=dataset)
    vocab_payload = json.loads((dataset / "publication_vocab.json").read_text(encoding="utf-8"))
    vocab = verify_vocab_file(vocab_payload, config.protocol)
    learners = (dataset / "dev5000_users.txt").read_text(encoding="utf-8").splitlines()
    vocab = attach_learners(vocab, learners, config.protocol)
    for name in ("train.parquet", "valid.parquet", "test.parquet"):
        if not (dataset / name).is_file():
            raise IdentityError(f"missing {name}")
    devices = execution_device(config.model, requested_device)
    if config.model in NEURAL_MODELS:
        devices = {
            "requested_device": requested_device,
            "resolved_device": "not_executed",
            "execution_mode": "neural",
        }
    env = environment_report(
        devices["requested_device"],
        devices["resolved_device"],
        execution_mode=devices["execution_mode"],
    )
    model_ready = False
    dependency_error = None
    if construct_model:
        try:
            create_model(
                config.model,
                config,
                {
                    "learner_count": len(vocab.learner_ids),
                    "item_count": len(vocab.item_ids),
                    "kc_count": len(vocab.kc_ids),
                },
            )
            model_ready = True
        except Exception as exc:
            dependency_error = f"{type(exc).__name__}: {exc}"
            if config.model not in NEURAL_MODELS:
                raise
    run_dir = Path(runs_dir) / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "dry_run",
        "publication_protocol": False,
        "publication": False,
        "notice": NOT_FOR_PUBLICATION,
        "requested_device": devices["requested_device"],
        "resolved_device": devices["resolved_device"],
        "run_id": config.run_id,
        "config_hash": config.config_hash(),
        "model_ready": model_ready,
        "dependency_error": dependency_error,
        "environment": env,
        "output_dir": str(run_dir),
    }
    _write_metadata(run_dir, config, phase="dry_run", extra=report)
    (run_dir / "config.json").write_text(config.canonical_json(), encoding="utf-8")
    (run_dir / "dry_run.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def smoke_run(
    config: RunConfig,
    *,
    dataset_dir: str | Path,
    runs_dir: str | Path,
    max_rows: int = 24,
    requested_device: str = "cpu",
) -> dict[str, Any]:
    """Run a few source-order bundles through the validated model path."""

    if max_rows > 64:
        raise FormalRunBlocked("smoke row cap is 64")
    if config.protocol.get("protocol_id") == "ASSISTMENTS2017_MAIN_V1":
        from kclearner.experiments.assistments_run import smoke_assistments

        return smoke_assistments(
            config,
            dataset_dir=dataset_dir,
            runs_dir=runs_dir,
            max_rows=max_rows,
            requested_device=requested_device,
        )
    dataset = Path(dataset_dir)
    preflight = dry_run(
        config,
        dataset_dir=dataset,
        runs_dir=runs_dir,
        construct_model=True,
        requested_device=requested_device,
    )
    if not preflight["model_ready"]:
        raise IdentityError(preflight["dependency_error"] or "model was not constructed")
    vocab_payload = json.loads((dataset / "publication_vocab.json").read_text(encoding="utf-8"))
    vocab = attach_learners(
        verify_vocab_file(vocab_payload, config.protocol),
        (dataset / "dev5000_users.txt").read_text(encoding="utf-8").splitlines(),
        config.protocol,
    )
    rows = _read_smoke_rows(dataset / "train.parquet", max_rows)
    run_dir = Path(runs_dir) / f"{config.run_id}_smoke"
    run_dir.mkdir(parents=True, exist_ok=True)
    if config.model in ("irt", "ar_kt"):
        predictions, metrics = _smoke_streaming(config, vocab, rows)
        model = create_model(
            config.model,
            config,
            {
                "learner_count": len(vocab.learner_ids),
                "item_count": len(vocab.item_ids),
                "kc_count": len(vocab.kc_ids),
            },
        )
        model.save_state(run_dir / "checkpoint.npz")
    else:
        predictions, metrics = _smoke_neural(
            config,
            vocab,
            rows,
            run_dir,
            requested_device=requested_device,
        )
    _write_predictions(run_dir / "predictions.parquet", predictions, config)
    (run_dir / "metrics.json").write_text(
        json.dumps({
            "publication_protocol": False,
            "publication": False,
            "notice": NOT_FOR_PUBLICATION,
            "metrics": metrics,
        }, indent=2)
        + "\n",
        encoding="utf-8",
    )
    if config.model in ("irt", "ar_kt"):
        (run_dir / "training_log.csv").write_text(
            "step,notice\n1,NOT FOR PUBLICATION\n",
            encoding="utf-8",
        )
    _write_metadata(
        run_dir,
        config,
        phase="smoke",
        extra={
            "publication_protocol": False,
            "publication": False,
            "notice": NOT_FOR_PUBLICATION,
            "rows": len(rows),
            "requested_device": execution_device(config.model, requested_device)["requested_device"],
            "resolved_device": execution_device(config.model, requested_device)["resolved_device"],
        },
    )
    (run_dir / "config.json").write_text(config.canonical_json(), encoding="utf-8")
    (run_dir / "dataset_manifest.json").write_text(
        (dataset / "manifest.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (run_dir / "vocab_manifest.json").write_text(
        json.dumps(config.protocol["vocabulary"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if config.model in ("irt", "ar_kt"):
        smoke_devices = execution_device(config.model, requested_device)
        (run_dir / "environment.json").write_text(
            json.dumps(
                environment_report(
                    smoke_devices["requested_device"],
                    smoke_devices["resolved_device"],
                    execution_mode=smoke_devices["execution_mode"],
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return {
        "run_id": run_dir.name,
        "rows": len(rows),
        "metrics": metrics,
        "publication_protocol": False,
        "publication": False,
    }


def execute_formal(*_args, **_kwargs):
    raise FormalRunBlocked(
        "formal TRAIN/VALID requires --authorize-formal. "
        "TEST evaluation is not part of this command."
    )


def run_authorized(config: RunConfig, *, dataset_dir: str | Path, runs_dir: str | Path, requested_device: str):
    """Full TRAIN then VALID. The CLI reaches this only with --authorize-formal.

    TEST is not read. This phase does not call this function.
    """

    if config.protocol.get("protocol_id") == "ASSISTMENTS2017_MAIN_V1":
        from kclearner.experiments.assistments_run import run_assistments_formal

        return run_assistments_formal(
            config,
            dataset_dir=dataset_dir,
            runs_dir=runs_dir,
            requested_device=requested_device,
        )

    from kclearner.experiments.neural_training import fit_neural

    dataset = Path(dataset_dir)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    verify_dataset_manifest(manifest, config.protocol, dataset_dir=dataset)
    vocab_payload = json.loads((dataset / "publication_vocab.json").read_text(encoding="utf-8"))
    vocab = attach_learners(
        verify_vocab_file(vocab_payload, config.protocol),
        (dataset / "dev5000_users.txt").read_text(encoding="utf-8").splitlines(),
        config.protocol,
    )
    if config.model not in NEURAL_MODELS:
        return _run_streaming_formal(
            config,
            dataset,
            vocab,
            runs_dir,
            requested_device=requested_device,
        )
    train_rows = _mapped_rows(dataset / "train.parquet", vocab, config)
    valid_rows = _mapped_rows(dataset / "valid.parquet", vocab, config)
    run_dir = Path(runs_dir) / config.run_id
    result = fit_neural(
        create_model(
            config.model,
            config,
            {
                "learner_count": len(vocab.learner_ids),
                "item_count": len(vocab.item_ids),
                "kc_count": len(vocab.kc_ids),
            },
        ),
        train_rows,
        valid_rows,
        config.protocol["models"][config.model]["recipe"],
        seed=config.seed,
        requested_device=requested_device,
        run_dir=run_dir,
        config_hash=config.config_hash(),
        dataset_id=config.protocol["dataset"]["dataset_id"],
        cohort_hash=config.protocol["dataset"]["cohort_sha256"],
        vocab_hashes={
            "item_ids_sha256": config.protocol["vocabulary"]["item_ids_sha256"],
            "kc_ids_sha256": config.protocol["vocabulary"]["kc_ids_sha256"],
        },
        authorize_formal=True,
        publication_protocol=bool(config.publication_protocol),
    )
    _freeze_neural(config, dataset, run_dir, result)
    return result


def _freeze_neural(config, dataset, run_dir, result):
    from kclearner.experiments.freeze import FreezeError, record_formal_freeze
    from kclearner.experiments.neural_training import CHECKPOINT_FORMAT

    checkpoint = result.get("checkpoint")
    metrics = result.get("valid_metrics")
    if not checkpoint or not isinstance(metrics, dict):
        raise FreezeError("formal neural run has no frozen checkpoint")
    if any(metrics.get(name) is None for name in ("nll", "brier", "auc")):
        raise FreezeError("formal neural run has no frozen VALID metrics")
    frozen = record_formal_freeze(
        run_dir,
        dataset,
        config,
        checkpoint,
        metrics,
        None,
        CHECKPOINT_FORMAT,
        requested_device=result.get("requested_device"),
        resolved_device=result.get("resolved_device"),
    )
    result["freeze_id"] = frozen["freeze_id"]


def _smoke_streaming(config: RunConfig, vocab: PublicationVocab, rows: Sequence[Mapping[str, Any]]):
    model = create_model(
        config.model,
        config,
        {
            "learner_count": len(vocab.learner_ids),
            "item_count": len(vocab.item_ids),
            "kc_count": len(vocab.kc_ids),
        },
    )
    learners = vocab.learner_index()
    items = vocab.item_index()
    kcs = vocab.kc_index()
    streaming: list[StreamingRow] = []
    kept: list[Mapping[str, Any]] = []
    for row in rows:
        item = items.get(str(row["item_id"]))
        learner = learners.get(str(row["learner_id"]))
        if item is None or learner is None:
            continue
        tags = tuple(kcs[int(kc)] for kc in row["kc_ids"] if int(kc) in kcs)
        streaming.append(
            StreamingRow(
                student_idx=learner,
                problem_idx=item,
                group_id=int(row["solving_id"]),
                correct=int(row["correct"]),
                tag_idxs=tags if config.model == "ar_kt" else (),
                interaction_id=str(row["interaction_id"]),
                timestamp_ms=int(row["timestamp"]),
                group_last_timestamp_ms=int(row["timestamp"]),
            )
        )
        kept.append(row)
    result = model.run_streaming(streaming, phase="train")
    predictions = []
    for trace, row in zip(result.rows, kept):
        predictions.append(_prediction(row, trace.probability, config))
    metrics = masked_metrics(
        [item["correct"] for item in predictions],
        [item["probability"] for item in predictions],
        [item["metric_mask"] for item in predictions],
    )
    return predictions, metrics


def _smoke_neural(
    config: RunConfig,
    vocab: PublicationVocab,
    rows: Sequence[Mapping[str, Any]],
    run_dir: Path,
    requested_device: str = "cpu",
):
    from kclearner.experiments.neural_training import fit_neural

    model = create_model(
        config.model,
        config,
        {
            "learner_count": len(vocab.learner_ids),
            "item_count": len(vocab.item_ids),
            "kc_count": len(vocab.kc_ids),
        },
    )
    learners = vocab.learner_index()
    items = vocab.item_index()
    kcs = vocab.kc_index()
    use_kc = config.model.endswith("qc")
    mapped: list[DKTRow] = []
    kept: list[Mapping[str, Any]] = []
    for row in rows:
        item = items.get(str(row["item_id"]))
        learner = learners.get(str(row["learner_id"]))
        if item is None or learner is None:
            continue
        tags = tuple(kcs[int(kc)] for kc in row["kc_ids"] if int(kc) in kcs)
        mapped.append(
            DKTRow(
                student_idx=learner,
                problem_idx=item,
                correct=int(row["correct"]),
                solving_id=int(row["solving_id"]),
                tag_idxs=tags if use_kc else (),
                interaction_id=str(row["interaction_id"]),
            )
        )
        kept.append(row)
    split_at = max(1, len(mapped) // 2)
    train_rows = mapped[:split_at]
    valid_rows = mapped[split_at:] or mapped[:1]
    fit_neural(
        model,
        train_rows,
        valid_rows,
        config.protocol["models"][config.model]["recipe"],
        seed=config.seed,
        requested_device=requested_device,
        run_dir=run_dir,
        config_hash=config.config_hash(),
        dataset_id=config.protocol["dataset"]["dataset_id"],
        cohort_hash=config.protocol["dataset"]["cohort_sha256"],
        vocab_hashes={
            "item_ids_sha256": config.protocol["vocabulary"]["item_ids_sha256"],
            "kc_ids_sha256": config.protocol["vocabulary"]["kc_ids_sha256"],
        },
        max_epochs=1,
        authorize_formal=False,
        publication_protocol=False,
    )
    model.backend.eval()
    replayed = model.replay(mapped)
    predictions = [
        _prediction(row, trace.probability, config)
        for trace, row in zip(replayed.rows, kept)
    ]
    metrics = masked_metrics(
        [item["correct"] for item in predictions],
        [item["probability"] for item in predictions],
        [item["metric_mask"] for item in predictions],
    )
    return predictions, metrics


def _prediction(row: Mapping[str, Any], probability: float, config: RunConfig) -> dict[str, Any]:
    return {
        "interaction_id": str(row["interaction_id"]),
        "learner_id": str(row["learner_id"]),
        "item_id": str(row["item_id"]),
        "bundle_id": str(row["bundle_id"]),
        "order": int(row["order"]),
        "split": "train",
        "correct": int(row["correct"]),
        "probability": float(probability),
        "metric_mask": bool(row["metric_mask"]),
        "model": config.model,
        "seed": config.seed,
        "kc_ids": [int(kc) for kc in row["kc_ids"]],
    }


def _run_streaming_formal(
    config: RunConfig,
    dataset: Path,
    vocab: PublicationVocab,
    runs_dir: str | Path,
    requested_device: str = "cpu",
) -> dict[str, Any]:
    model = create_model(
        config.model,
        config,
        {
            "learner_count": len(vocab.learner_ids),
            "item_count": len(vocab.item_ids),
            "kc_count": len(vocab.kc_ids),
        },
    )
    train = _streaming_rows(dataset / "train.parquet", vocab, config)
    valid = _streaming_rows(dataset / "valid.parquet", vocab, config)
    model.run_streaming(train, phase="train")
    run_dir = Path(runs_dir) / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "state_train_only.npz"
    model.save_state(state_path)
    traced = model.run_streaming(valid, phase="eval")
    masks = [True] * len(traced.rows)
    headline = masked_metrics(
        [row.correct for row in traced.rows],
        [row.probability for row in traced.rows],
        masks,
    )
    global_metrics = masked_metrics(
        [row.correct for row in traced.rows],
        [row.p_global for row in traced.rows],
        masks,
    )
    from kclearner.experiments.freeze import record_formal_freeze
    from kclearner.models.state import FORMAT_ID

    devices = execution_device(config.model, requested_device)
    frozen = record_formal_freeze(
        run_dir,
        dataset,
        config,
        state_path,
        headline,
        global_metrics,
        FORMAT_ID,
        requested_device=devices["requested_device"],
        resolved_device=devices["resolved_device"],
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({
            **headline,
            "publication_protocol": True,
            "publication": True,
        })
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "environment.json").write_text(
        json.dumps(
            environment_report(
                devices["requested_device"],
                devices["resolved_device"],
                execution_mode=devices["execution_mode"],
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"best_epoch": 0, "metrics": headline, "freeze_id": frozen["freeze_id"]}


def _streaming_rows(path: Path, vocab: PublicationVocab, config: RunConfig) -> list[StreamingRow]:
    import pyarrow.parquet as pq

    table = pq.read_table(
        str(path),
        columns=["interaction_id", "learner_id", "item_id", "correct", "kc_ids", "solving_id", "timestamp", "metric_mask"],
    )
    data = table.to_pydict()
    learners = vocab.learner_index()
    items = vocab.item_index()
    kcs = vocab.kc_index()
    rows: list[StreamingRow] = []
    for index in range(table.num_rows):
        if not bool(data["metric_mask"][index]):
            continue
        item = items.get(str(data["item_id"][index]))
        learner = learners.get(str(data["learner_id"][index]))
        if item is None or learner is None:
            continue
        tags = tuple(kcs[int(kc)] for kc in data["kc_ids"][index] if int(kc) in kcs)
        rows.append(
            StreamingRow(
                student_idx=learner,
                problem_idx=item,
                group_id=int(data["solving_id"][index]),
                correct=int(data["correct"][index]),
                tag_idxs=tags if config.model == "ar_kt" else (),
                interaction_id=str(data["interaction_id"][index]),
                timestamp_ms=int(data["timestamp"][index]),
                group_last_timestamp_ms=int(data["timestamp"][index]),
            )
        )
    return rows


def _mapped_rows(path: Path, vocab: PublicationVocab, config: RunConfig) -> list[DKTRow]:
    import pyarrow.parquet as pq

    table = pq.read_table(
        str(path),
        columns=["interaction_id", "learner_id", "item_id", "correct", "kc_ids", "solving_id", "metric_mask"],
    )
    data = table.to_pydict()
    learners = vocab.learner_index()
    items = vocab.item_index()
    kcs = vocab.kc_index()
    use_kc = config.model.endswith("qc")
    rows: list[DKTRow] = []
    for index in range(table.num_rows):
        if not bool(data["metric_mask"][index]):
            continue
        item = items.get(str(data["item_id"][index]))
        learner = learners.get(str(data["learner_id"][index]))
        if item is None or learner is None:
            continue
        tags = tuple(kcs[int(kc)] for kc in data["kc_ids"][index] if int(kc) in kcs)
        rows.append(
            DKTRow(
                student_idx=learner,
                problem_idx=item,
                correct=int(data["correct"][index]),
                solving_id=int(data["solving_id"][index]),
                tag_idxs=tags if use_kc else (),
                interaction_id=str(data["interaction_id"][index]),
            )
        )
    return rows


def _read_smoke_rows(path: Path, max_rows: int) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(str(path)).slice(0, max_rows)
    columns = table.to_pydict()
    rows = []
    for index in range(table.num_rows):
        rows.append({name: values[index] for name, values in columns.items()})
    return rows


def _write_predictions(path: Path, rows: Sequence[Mapping[str, Any]], config: RunConfig) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            ("interaction_id", pa.string()),
            ("learner_id", pa.string()),
            ("item_id", pa.string()),
            ("bundle_id", pa.string()),
            ("order", pa.int64()),
            ("split", pa.string()),
            ("correct", pa.int64()),
            ("probability", pa.float64()),
            ("metric_mask", pa.bool_()),
            ("model", pa.string()),
            ("seed", pa.int64()),
            ("kc_ids", pa.list_(pa.int64())),
        ]
    )
    columns = {
        field.name: [row[field.name] for row in rows] if rows else pa.array([], type=field.type)
        for field in schema
    }
    pq.write_table(pa.table(columns, schema=schema), str(path))
    del config


def _write_metadata(run_dir: Path, config: RunConfig, *, phase: str, extra: Mapping[str, Any]) -> None:
    requested = str(extra.get("requested_device", "cpu"))
    devices = execution_device(config.model, requested)
    resolved = extra.get("resolved_device", devices["resolved_device"])
    env = environment_report(
        requested,
        str(resolved),
        execution_mode=devices["execution_mode"],
    )
    git_commit = _git_commit()
    payload = {
        "run_id": run_dir.name,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kclearner_version": __version__,
        "git_commit": git_commit,
        "dataset_id": config.protocol["dataset"]["dataset_id"],
        "cohort_hash": config.protocol["dataset"]["cohort_sha256"],
        "split_identity": config.protocol["dataset"]["split_id"],
        "vocab_id": config.protocol["vocabulary"]["vocab_id"],
        "item_ids_sha256": config.protocol["vocabulary"]["item_ids_sha256"],
        "kc_ids_sha256": config.protocol["vocabulary"]["kc_ids_sha256"],
        "model": config.model,
        "model_family": config.protocol["models"][config.model]["family"],
        "kc_aware": config.protocol["models"][config.model]["kc_aware"],
        "seed": config.seed,
        "deterministic": config.deterministic,
        "device": env["device"],
        "python": env["python"],
        "torch": env["torch"],
        "pykt_toolkit": env["pykt_toolkit"],
        "phase": phase,
        "checkpoint_identity": None,
        "config_hash": config.config_hash(),
        "requested_device": env["requested_device"],
        "resolved_device": env["resolved_device"],
        "publication_protocol": False,
        "publication": False,
        "notice": NOT_FOR_PUBLICATION,
    }
    payload.update(extra)
    (run_dir / "metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "environment.json").write_text(
        json.dumps(env, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_commit() -> str | None:
    try:
        import subprocess

        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def verify_dataset_bytes(dataset_dir: Path, protocol: Mapping[str, Any]) -> None:
    expected = protocol["dataset"]["files_sha256"]
    for name, digest in expected.items():
        actual = file_sha256(dataset_dir / f"{name}.parquet")
        if actual != digest:
            raise IdentityError(f"{name}.parquet byte hash mismatch")
