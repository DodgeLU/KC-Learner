"""Explicit TEST evaluation. It does not train and it does not use
``--authorize-formal``.

Psychometric models replay VALID from the TRAIN-only state and compare
the historical absolute tolerance before TEST is opened. Neural models
check freeze identity only, then carry TRAIN and VALID state into TEST.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Callable, Mapping, Sequence

from kclearner.data.ednet_corrected import file_sha256
from kclearner.experiments.freeze import (
    EVALUATION_SEMANTICS,
    FreezeError,
    load_freeze_manifest,
    metrics_within_tolerance,
)

class EvaluateError(RuntimeError):
    """TEST evaluation cannot start or cannot be stored."""


def run_evaluation(
    freeze_path: str | Path,
    *,
    enable_test: bool,
    dataset_logical_hash: str,
    checkpoint_path: str | Path,
    recipe_hash_value: str,
    seed: int,
    output_dir: str | Path,
    valid_rows: Callable[[], tuple[Sequence, Sequence[bool]]],
    test_rows: Callable[[], tuple[Sequence, Sequence[bool]]],
    replay: Callable[[Sequence, Sequence[bool]], Callable],
    neural_context: Callable[[], Callable],
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Run one frozen TEST evaluation.

    ``replay`` is used only when the freeze requires VALID replay. It
    must raise before returning when the replay does not match.
    ``test_rows`` is called only after that gate, or after the neural
    identity checks and context load.
    """

    if not enable_test:
        raise EvaluateError("TEST execution requires --enable-test-execution")
    try:
        freeze = load_freeze_manifest(freeze_path)
    except FreezeError as exc:
        raise EvaluateError(str(exc)) from exc
    identity = freeze["identity"]
    if not isinstance(identity, dict):
        raise EvaluateError("freeze identity is missing")
    result_path = Path(output_dir) / str(freeze["freeze_id"]) / "test_result.json"
    if result_path.exists():
        raise EvaluateError(
            f"final TEST output already exists: {result_path}. "
            "It will not be overwritten."
        )
    if identity.get("evaluation_semantics") != EVALUATION_SEMANTICS:
        raise EvaluateError("evaluation semantics are incompatible with this evaluator")
    if identity.get("dataset_logical_hash") != dataset_logical_hash:
        raise EvaluateError("dataset logical hash does not match the freeze")
    if identity.get("recipe_hash") != recipe_hash_value:
        raise EvaluateError("model recipe hash does not match the freeze")
    if int(identity.get("seed")) != int(seed):
        raise EvaluateError("seed does not match the freeze")
    checkpoint = Path(checkpoint_path)
    if file_sha256(checkpoint) != identity.get("checkpoint_sha256"):
        raise EvaluateError("checkpoint hash does not match the freeze")
    if list(identity.get("phases_completed") or []) != ["train", "valid"]:
        raise EvaluateError("freeze does not record completed TRAIN and VALID")
    if identity.get("valid_replay_required"):
        rows, masks = valid_rows()
        score = replay(rows, masks)
        psychometric_gate(score.headline, score.global_metrics, identity)
    else:
        score = neural_context()
    rows, masks = test_rows()
    metrics = score(rows, masks)
    if metrics is None:
        metrics = {"nll": None, "brier": None, "auc": None, "n": 0}
    document = {
        "phase": "test",
        "role": "evaluation",
        "derived_from_frozen_experiment": True,
        "freeze_id": freeze["freeze_id"],
        "dataset_logical_hash": identity["dataset_logical_hash"],
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "evaluation_semantics": EVALUATION_SEMANTICS,
        "metrics": {
            "nll": metrics["nll"],
            "brier": metrics["brier"],
            "auc": metrics["auc"],
        },
        "metric_eligible_count": int(metrics["n"]),
        "provenance": dict(provenance or {}),
    }
    global_metrics = metrics.get("global_metrics") if isinstance(metrics, dict) else None
    if isinstance(global_metrics, dict):
        document["global_metrics"] = {
            "nll": global_metrics["nll"],
            "brier": global_metrics["brier"],
            "auc": global_metrics["auc"],
        }
    _create_new(result_path, document)
    return document


def evaluate_prepared(
    freeze_path: str | Path,
    dataset_dir: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    enable_test: bool,
    protocol: Mapping[str, object],
    model: str,
    seed: int,
) -> dict[str, object]:
    """Evaluate one prepared dataset without reading TEST early."""

    from kclearner.experiments.freeze import environment_provenance, recipe_hash

    dataset = Path(dataset_dir)
    logical = json.loads((dataset / "logical_identity.json").read_text(encoding="utf-8"))
    identity = logical["dataset_logical_identity"]
    recipe = protocol["models"][model]["recipe"]
    sizes = _sizes(dataset, protocol, identity)

    def valid_call():
        return _phase_rows(dataset, protocol, model, seed, sizes, "valid")

    def test_call():
        return _phase_rows(dataset, protocol, model, seed, sizes, "test")

    def replay(rows, masks):
        return _psychometric_replay(
            protocol,
            model,
            seed,
            sizes,
            checkpoint_path,
            rows,
            masks,
        )

    def context():
        train_rows, _train_masks = _phase_rows(dataset, protocol, model, seed, sizes, "train")
        valid_rows, _valid_masks = _phase_rows(dataset, protocol, model, seed, sizes, "valid")
        return _neural_scorer(
            protocol,
            model,
            seed,
            sizes,
            checkpoint_path,
            train_rows,
            valid_rows,
        )

    return run_evaluation(
        freeze_path,
        enable_test=enable_test,
        dataset_logical_hash=str(logical["dataset_logical_hash"]),
        checkpoint_path=checkpoint_path,
        recipe_hash_value=recipe_hash(recipe),
        seed=int(seed),
        output_dir=output_dir,
        valid_rows=valid_call,
        test_rows=test_call,
        replay=replay,
        neural_context=context,
        provenance=environment_provenance(),
    )


def _psychometric_replay(protocol, model, seed, sizes, checkpoint, rows, masks):
    from kclearner.experiments.runner import masked_metrics
    from kclearner.models.ar_kt import ARKTModel
    from kclearner.models.irt import IRTModel

    del protocol, seed, sizes
    built = IRTModel.load_state(checkpoint) if model == "irt" else ARKTModel.load_state(checkpoint)
    traced = built.run_streaming(list(rows), phase="eval")
    headline, global_metrics = _trace_metrics(traced, masks, masked_metrics)

    def score(test_rows, test_masks):
        scored = built.run_streaming(list(test_rows), phase="eval")
        headline_metrics, global_metrics = _trace_metrics(scored, test_masks, masked_metrics)
        if headline_metrics is None:
            return None
        scored_metrics = dict(headline_metrics)
        scored_metrics["global_metrics"] = global_metrics
        return scored_metrics

    score.headline = headline
    score.global_metrics = global_metrics
    return score


def psychometric_gate(headline, global_metrics, identity: Mapping[str, object]) -> None:
    """Raise when the historical VALID replay does not match."""

    tolerance = identity.get("valid_replay_tolerance")
    expected = {
        "nll": identity["valid_nll"],
        "brier": identity["valid_brier"],
        "auc": identity["valid_auc"],
    }
    if not metrics_within_tolerance(headline, expected, float(tolerance)):
        raise EvaluateError("VALID replay does not match the frozen metrics")
    if identity.get("model") == "ar_kt":
        expected_global = {
            "nll": identity["valid_global_nll"],
            "brier": identity["valid_global_brier"],
            "auc": identity["valid_global_auc"],
        }
        if not metrics_within_tolerance(global_metrics, expected_global, float(tolerance)):
            raise EvaluateError("VALID p_global replay does not match the frozen metrics")


def _trace_metrics(traced, masks, masked_metrics):
    correct = [row.correct for row in traced.rows]
    headline = masked_metrics(
        correct,
        [row.probability for row in traced.rows],
        masks,
    )
    global_metrics = masked_metrics(
        correct,
        [row.p_global for row in traced.rows],
        masks,
    )
    return headline, global_metrics


def _neural_scorer(protocol, model, seed, sizes, checkpoint, train_rows, valid_rows):
    from kclearner.experiments.factory import create_model
    from kclearner.experiments.neural_training import load_checkpoint
    from kclearner.experiments.runner import masked_metrics

    built = create_model(model, _config_view(protocol, model, seed), sizes)
    load_checkpoint(checkpoint, built)
    if str(getattr(built, "model_kind", "")).startswith("dkt"):
        state = built.replay(list(train_rows)).final_state
        state = built.replay(list(valid_rows), initial_state=state).final_state

        def score(test_rows, _masks):
            traced = built.replay(list(test_rows), initial_state=state)
            return _neural_metrics(traced, masked_metrics)
    else:
        memory = built.replay(list(train_rows)).final_memory
        memory = built.replay(list(valid_rows), initial_memory=memory).final_memory

        def score(test_rows, _masks):
            traced = built.replay(list(test_rows), initial_memory=memory)
            return _neural_metrics(traced, masked_metrics)

    return score


def _neural_metrics(traced, masked_metrics):
    correct = [row.correct for row in traced.rows]
    probability = [row.probability for row in traced.rows]
    return masked_metrics(correct, probability, [True] * len(correct))


def _phase_rows(dataset, protocol, model, seed, sizes, split: str):
    dataset_id = protocol["dataset"]["dataset_id"]
    if dataset_id == "assistments2017":
        from kclearner.data.assistments_prepare import read_model_rows
        from kclearner.experiments.assistments_run import rows_for_execution

        model_rows = read_model_rows(dataset, split)
        executed = rows_for_execution(model_rows, model)
        if model in ("irt", "ar_kt"):
            return executed, [row.metric_mask for row in model_rows]
        return executed, [True] * len(executed)
    from kclearner.experiments.runner import _mapped_rows, _streaming_rows

    vocab = sizes["vocab"]
    path = dataset / f"{split}.parquet"
    config = _config_view(protocol, model, seed)
    if model in ("irt", "ar_kt"):
        if split == "valid":
            rows = _streaming_rows(path, vocab, config)
            return rows, [True] * len(rows)
        return _ednet_psychometric_test_rows(path, vocab, config)
    rows = _mapped_rows(path, vocab, config)
    return rows, [True] * len(rows)


def _ednet_psychometric_test_rows(path, vocab, config):
    """Keep a masked EdNet row in the sequence. Corrected TEST OOV is 0."""

    import pyarrow.parquet as pq
    from kclearner.models.streaming import StreamingRow

    table = pq.read_table(
        str(path),
        columns=["interaction_id", "learner_id", "item_id", "correct", "kc_ids", "solving_id", "timestamp", "metric_mask"],
    )
    data = table.to_pydict()
    learners = vocab.learner_index()
    items = vocab.item_index()
    kcs = vocab.kc_index()
    rows = []
    masks = []
    for index in range(table.num_rows):
        learner = learners.get(str(data["learner_id"][index]))
        if learner is None:
            continue
        item = items.get(str(data["item_id"][index]))
        masked = bool(data["metric_mask"][index]) and item is not None
        if item is None:
            item = 0
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
        masks.append(masked)
    return rows, masks


def _sizes(dataset: Path, protocol: Mapping[str, object], identity: Mapping[str, object]) -> dict:
    dataset_id = protocol["dataset"]["dataset_id"]
    if identity.get("dataset_id") != dataset_id:
        raise EvaluateError("prepared dataset_id does not match the protocol")
    if dataset_id == "assistments2017":
        from kclearner.experiments.assistments_run import verify_assistments_prepared

        sizes = verify_assistments_prepared(dataset, protocol, read_test=False)
        return sizes
    from kclearner.experiments.runner import attach_learners, verify_vocab_file

    payload = json.loads((dataset / "publication_vocab.json").read_text(encoding="utf-8"))
    learners = (dataset / "dev5000_users.txt").read_text(encoding="utf-8").splitlines()
    vocab = attach_learners(verify_vocab_file(payload, protocol), learners, protocol)
    if identity.get("item_vocabulary_hash") != protocol["vocabulary"]["item_ids_sha256"]:
        raise EvaluateError("item vocabulary hash does not match the protocol")
    return {
        "learner_count": len(vocab.learner_ids),
        "item_count": len(vocab.item_ids),
        "kc_count": len(vocab.kc_ids),
        "vocab": vocab,
    }


def _config_view(protocol: Mapping[str, object], model: str, seed: int):
    from kclearner.experiments.config import resolve_run

    if protocol["models"][model]["deterministic"]:
        return resolve_run(protocol, model)
    return resolve_run(protocol, model, int(seed))


def _create_new(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise EvaluateError(f"final TEST output already exists: {path}")
    temporary = path.with_name(f"{path.name}.incomplete.{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        temporary.unlink(missing_ok=True)
        raise EvaluateError(f"final TEST output already exists: {path}") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(temporary.read_text(encoding="utf-8"))
    finally:
        temporary.unlink(missing_ok=True)
