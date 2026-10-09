"""Formal DKT and DKVMN TRAIN → VALID loops.

The scoring step is the already tested predict-before-update path.
DKT uses one Adam step per learner slab and ``BCEWithLogits``.
DKVMN uses one Adam step per TBPTT chunk and ``BCE`` on probabilities.
Early stopping follows the legacy rule
``valid_auc > best_auc + min_delta``.

Full corrected-EdNet execution stays behind ``authorize_formal``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from kclearner.experiments.runner import FormalRunBlocked, masked_metrics
from kclearner.models.dkt import DKTRow, qc_input
from kclearner.models.dkvmn import predict_run
from kclearner.models.neural_runs import learner_solving_runs

CHECKPOINT_FORMAT = "kclearner_neural_checkpoint_v1"
SMOKE_ROW_CAP = 64


class DeviceError(RuntimeError):
    """The requested device cannot be used."""


def resolve_device(requested: str):
    name = requested.strip().lower()
    if name not in ("auto", "cpu", "cuda"):
        raise DeviceError(f"device must be auto, cpu, or cuda, got {requested!r}")
    try:
        import torch
    except Exception as exc:
        if name == "cuda":
            raise DeviceError(
                "device=cuda was requested but CUDA is not available"
            ) from exc
        raise
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise DeviceError("device=cuda was requested but CUDA is not available")
    return torch.device("cuda")


def seed_everything(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def improved(valid_auc: float, best_auc: float, min_delta: float) -> bool:
    """Legacy checkpoint rule: ``valid_auc > best_auc + min_delta``."""

    return float(valid_auc) > float(best_auc) + float(min_delta)


@dataclass
class EarlyStopState:
    best_auc: float = -1.0
    best_epoch: int = -1
    patience_counter: int = 0
    best_nll: float | None = None
    best_brier: float | None = None

    def observe(self, epoch: int, valid_auc: float, valid_nll: float, valid_brier: float, min_delta: float, patience: int) -> bool:
        """Return whether training should stop after this epoch."""

        if improved(valid_auc, self.best_auc, min_delta):
            self.best_auc = float(valid_auc)
            self.best_nll = float(valid_nll)
            self.best_brier = float(valid_brier)
            self.best_epoch = int(epoch)
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        return self.patience_counter >= int(patience)


def _runs_by_learner(rows: Sequence[DKTRow]):
    runs = learner_solving_runs(
        [row.student_idx for row in rows],
        [row.solving_id for row in rows],
    )
    grouped: dict[int, list] = {}
    for run in runs:
        grouped.setdefault(run.learner_index, []).append(run)
    return grouped


def _chunks(runs, target_rows: int) -> list[list]:
    chunks: list[list] = []
    current: list = []
    count = 0
    for run in runs:
        if current and count + run.length > target_rows:
            chunks.append(current)
            current = []
            count = 0
        current.append(run)
        count += run.length
    if current:
        chunks.append(current)
    return chunks


def _rows_of(rows: Sequence[DKTRow], run) -> list[DKTRow]:
    return [rows[index] for index in run.row_positions]


def dkt_slab_loss(model, row_groups: Sequence[Sequence[DKTRow]], carried: dict):
    """Mean BCE-with-logits over one DKT slab. Hidden state is not detached."""

    import torch

    losses = []
    new_state = {}
    model.backend.train()
    for rows in row_groups:
        if not rows:
            continue
        learner = int(rows[0].student_idx)
        problems = torch.tensor([row.problem_idx for row in rows], dtype=torch.long, device=_device_of(model))
        corrects = torch.tensor([row.correct for row in rows], dtype=torch.long, device=_device_of(model))
        hidden_size = int(model.backend.hidden_size)
        if learner in carried:
            h0, c0 = carried[learner]
        else:
            h0 = torch.zeros(1, 1, hidden_size, device=problems.device)
            c0 = torch.zeros(1, 1, hidden_size, device=problems.device)
        if model.use_kc:
            inputs, _checksum = qc_input(
                torch,
                model.backend,
                problems,
                corrects,
                [row.tag_idxs for row in rows],
                model.n_questions,
                model.n_concepts,
            )
        else:
            inputs = model.backend.interaction_emb(problems + model.n_questions * corrects)
        outputs, (h_n, c_n) = model.backend.lstm_layer(inputs.unsqueeze(0), (h0, c0))
        hidden = outputs.squeeze(0)
        incoming = h0.squeeze(0).squeeze(0)
        cursor = 0
        runs = learner_solving_runs([row.student_idx for row in rows], [row.solving_id for row in rows])
        for run in runs:
            h_pre = incoming if cursor == 0 else hidden[cursor - 1]
            h_drop = model.backend.dropout_layer(h_pre.unsqueeze(0)).squeeze(0)
            for offset in range(run.length):
                target = int(rows[cursor + offset].problem_idx)
                logit = (model.backend.out_layer.weight[target] * h_drop).sum() + model.backend.out_layer.bias[target]
                label = torch.tensor(float(rows[cursor + offset].correct), dtype=logit.dtype, device=logit.device)
                losses.append(torch.nn.functional.binary_cross_entropy_with_logits(logit, label))
            cursor += run.length
        new_state[learner] = (h_n, c_n)
    if not losses:
        raise RuntimeError("DKT slab has no target rows")
    return torch.stack(losses).mean(), new_state


def dkvmn_chunk_loss(model, rows: Sequence[DKTRow], memory):
    """Mean BCE over one DKVMN chunk. Returns the loss and the new memory."""

    import torch

    model.backend.train()
    device = _device_of(model)
    losses = []
    runs = learner_solving_runs([row.student_idx for row in rows], [row.solving_id for row in rows])
    for run in runs:
        chosen = [rows[index] for index in run.row_positions]
        problems = torch.tensor([row.problem_idx for row in chosen], dtype=torch.long, device=device)
        corrects = torch.tensor([row.correct for row in chosen], dtype=torch.long, device=device)
        tags = [row.tag_idxs for row in chosen] if model.use_kc else None
        probability, memory, _attention, _read, _concept, _logit = predict_run(
            torch,
            model.backend,
            memory,
            problems,
            corrects,
            tags,
            model.n_questions,
        )
        target = corrects.to(dtype=probability.dtype)
        losses.append(torch.nn.functional.binary_cross_entropy(probability, target, reduction="sum"))
    total_rows = sum(run.length for run in runs)
    return torch.stack(losses).sum() / float(total_rows), memory


def train_one_epoch(model, rows: Sequence[DKTRow], optimizer, recipe: dict, rng: np.random.Generator) -> tuple[float, int]:
    family = "dkvmn" if model.model_kind.startswith("dkvmn") else "dkt"
    if family == "dkt":
        return _train_dkt_epoch(model, rows, optimizer, recipe, rng)
    return _train_dkvmn_epoch(model, rows, optimizer, recipe, rng)


def _train_dkt_epoch(model, rows, optimizer, recipe, rng) -> tuple[float, int]:
    grouped = _runs_by_learner(rows)
    learners = list(grouped)
    rng.shuffle(learners)
    target = int(recipe["tbptt_target_rows"])
    width = int(recipe["effective_chunk_size"])
    chunks = {learner: _chunks(grouped[learner], target) for learner in learners}
    index = {learner: 0 for learner in learners}
    carried: dict = {}
    loss_sum = 0.0
    row_count = 0
    pool: list[int] = []
    next_user = 0
    while True:
        while len(pool) < width and next_user < len(learners):
            pool.append(learners[next_user])
            next_user += 1
        pool = [learner for learner in pool if index[learner] < len(chunks[learner])]
        if not pool:
            break
        groups = []
        for learner in pool:
            chunk = chunks[learner][index[learner]]
            groups.append([row for run in chunk for row in _rows_of(rows, run)])
            index[learner] += 1
        optimizer.zero_grad()
        loss, new_state = dkt_slab_loss(model, groups, carried)
        loss.backward()
        optimizer.step()
        for learner, state in new_state.items():
            carried[learner] = (state[0].detach(), state[1].detach())
        n_rows = sum(len(group) for group in groups)
        loss_sum += float(loss.detach().cpu()) * n_rows
        row_count += n_rows
    return loss_sum / max(row_count, 1), row_count


def _train_dkvmn_epoch(model, rows, optimizer, recipe, rng) -> tuple[float, int]:
    grouped = _runs_by_learner(rows)
    learners = list(grouped)
    rng.shuffle(learners)
    target = int(recipe["tbptt_target_rows"])
    loss_sum = 0.0
    row_count = 0
    for learner in learners:
        memory = model.backend.Mv0
        for chunk in _chunks(grouped[learner], target):
            chosen = [row for run in chunk for row in _rows_of(rows, run)]
            optimizer.zero_grad()
            loss, memory = dkvmn_chunk_loss(model, chosen, memory)
            loss.backward()
            optimizer.step()
            memory = memory.detach()
            loss_sum += float(loss.detach().cpu()) * len(chosen)
            row_count += len(chosen)
    return loss_sum / max(row_count, 1), row_count


def valid_metrics_from_replay(model, train_rows: Sequence[DKTRow], valid_rows: Sequence[DKTRow]) -> dict[str, float]:
    model.backend.eval()
    if model.model_kind.startswith("dkt"):
        trained = model.replay(list(train_rows))
        scored = model.replay(list(valid_rows), initial_state=trained.final_state)
    else:
        trained = model.replay(list(train_rows))
        scored = model.replay(list(valid_rows), initial_memory=trained.final_memory)
    return masked_metrics(
        [row.correct for row in valid_rows],
        [trace.probability for trace in scored.rows],
        [True for _ in valid_rows],
    ) or {"nll": float("nan"), "brier": float("nan"), "auc": float("nan"), "n": 0}


def fit_neural(
    model,
    train_rows: Sequence[DKTRow],
    valid_rows: Sequence[DKTRow],
    recipe: dict,
    *,
    seed: int,
    requested_device: str,
    run_dir: str | Path,
    config_hash: str,
    dataset_id: str,
    cohort_hash: str,
    vocab_hashes: dict,
    max_epochs: int | None = None,
    authorize_formal: bool = False,
    publication_protocol: bool = False,
) -> dict:
    """Train until early stopping. Refuses a full-size formal run unless authorized."""

    import torch

    if not authorize_formal and len(train_rows) > SMOKE_ROW_CAP:
        raise FormalRunBlocked("formal neural TRAIN is not authorized")
    device = resolve_device(requested_device)
    resolved_name = "cuda" if device.type == "cuda" else "cpu"
    formal_protocol = bool(publication_protocol) and bool(authorize_formal)
    seed_everything(seed)
    model.backend.to(device)
    epochs = int(recipe["max_epochs"] if max_epochs is None else max_epochs)
    if not authorize_formal:
        epochs = min(epochs, 1)
    optimizer = torch.optim.Adam(
        model.backend.parameters(),
        lr=float(recipe["learning_rate"]),
        weight_decay=float(recipe["weight_decay"]),
    )
    stopper = EarlyStopState()
    rng = np.random.default_rng(seed)
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    from kclearner.experiments.environment import environment_report

    (run_path / "environment.json").write_text(
        json.dumps(
            environment_report(
                requested_device,
                resolved_name,
                execution_mode="neural",
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    log_path = run_path / "training_log.csv"
    log_path.write_text(
        "epoch,train_loss,valid_nll,valid_brier,valid_auc,best_valid_auc,improved,patience_counter,elapsed_seconds\n",
        encoding="utf-8",
    )
    saved_checkpoint = None
    history = []
    for epoch in range(epochs):
        started = time.perf_counter()
        train_loss, _rows = train_one_epoch(model, train_rows, optimizer, recipe, rng)
        metrics = valid_metrics_from_replay(model, train_rows, valid_rows)
        before = stopper.best_auc
        stop = stopper.observe(
            epoch,
            metrics["auc"],
            metrics["nll"],
            metrics["brier"],
            float(recipe["checkpoint_min_delta"]),
            int(recipe["early_stop_patience_epochs"]),
        )
        did_improve = stopper.best_auc != before or stopper.best_epoch == epoch and stopper.patience_counter == 0
        if stopper.best_epoch == epoch and stopper.patience_counter == 0:
            saved_checkpoint = run_path / f"best_checkpoint_epoch_{epoch:03d}.pt"
            _save_checkpoint(
                saved_checkpoint,
                model,
                optimizer,
                epoch,
                stopper,
                seed,
                config_hash,
                dataset_id,
                cohort_hash,
                vocab_hashes,
                recipe,
                publication_protocol=formal_protocol,
            )
        elapsed = time.perf_counter() - started
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_nll": metrics["nll"],
            "valid_brier": metrics["brier"],
            "valid_auc": metrics["auc"],
            "best_valid_auc": stopper.best_auc,
            "improved": bool(did_improve and stopper.patience_counter == 0),
            "patience_counter": stopper.patience_counter,
            "elapsed_seconds": elapsed,
        }
        history.append(row)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{epoch},{train_loss},{metrics['nll']},{metrics['brier']},{metrics['auc']},"
                f"{stopper.best_auc},{str(row['improved']).lower()},{stopper.patience_counter},{elapsed}\n"
            )
        if stop:
            break
    return {
        "history": history,
        "best_epoch": stopper.best_epoch,
        "best_valid_auc": stopper.best_auc,
        "checkpoint": None if saved_checkpoint is None else str(saved_checkpoint),
        "valid_metrics": None if stopper.best_epoch < 0 else {
            "nll": stopper.best_nll,
            "brier": stopper.best_brier,
            "auc": stopper.best_auc,
        },
        "requested_device": requested_device,
        "resolved_device": resolved_name,
        "publication_protocol": formal_protocol,
    }


def _save_checkpoint(path, model, optimizer, epoch, stopper, seed, config_hash, dataset_id, cohort_hash, vocab_hashes, recipe, publication_protocol: bool = False) -> None:
    import torch

    torch.save(
        {
            "format_id": CHECKPOINT_FORMAT,
            "model_kind": model.model_kind,
            "state_dict": model.backend.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_valid_auc": stopper.best_auc,
            "best_valid_nll": stopper.best_nll,
            "best_valid_brier": stopper.best_brier,
            "patience_counter": stopper.patience_counter,
            "best_epoch": stopper.best_epoch,
            "seed": seed,
            "config_hash": config_hash,
            "dataset_id": dataset_id,
            "cohort_hash": cohort_hash,
            "vocab_hashes": vocab_hashes,
            "dimensions": _dimensions(model, recipe),
            "publication_protocol": bool(publication_protocol),
            "publication": bool(publication_protocol),
            "notice": None if publication_protocol else "NOT FOR PUBLICATION",
        },
        str(path),
    )


def load_checkpoint(path: str | Path, model, optimizer=None) -> dict:
    """Restore a best checkpoint. Optimizer state is restored when supplied."""

    import torch

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if payload.get("format_id") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported checkpoint {payload.get('format_id')!r}")
    model.backend.load_state_dict(payload["state_dict"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def _dimensions(model, recipe: dict) -> dict:
    dims = {
        "n_questions": model.n_questions,
        "n_concepts": model.n_concepts,
        "model_kind": model.model_kind,
    }
    if model.model_kind.startswith("dkt"):
        dims["emb_size"] = int(recipe["emb_size"])
        dims["hidden_size"] = int(recipe["hidden_size"])
    else:
        dims["dim_s"] = int(recipe["dim_s"])
        dims["size_m"] = int(recipe["size_m"])
    return dims


def _device_of(model):
    return next(model.backend.parameters()).device
