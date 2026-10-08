"""Thin DKVMN-Q / DKVMN-QC wrappers over pinned pyKT ``DKVMN`` modules.

Prediction reads one pre-run value memory for every row in the run.
Writes then follow canonical row order. QC adds the mean human-KC
embedding to the read vector before ``f_layer``. Attention and the
erase/add write do not use human KC ids.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from kclearner.models.dkt import DKTRow
from kclearner.models.neural_runs import NeuralRun, learner_solving_runs

FORMAT_ID = "kclearner_dkvmn_runtime_v1"


def _torch():
    import torch

    return torch


def _pykt_dkvmn():
    from pykt.models.dkvmn import DKVMN

    return DKVMN


def mean_human_kc(torch_module, embedding, tag_lists: Sequence[Sequence[int]]):
    weight = embedding.weight
    rows = []
    for tags in tag_lists:
        if not tags:
            rows.append(weight.new_zeros(weight.shape[1]))
            continue
        ids = torch_module.tensor(
            [int(tag) for tag in tags],
            dtype=torch_module.long,
            device=weight.device,
        )
        rows.append(embedding(ids).mean(dim=0))
    return torch_module.stack(rows, dim=0)


def predict_run(
    torch_module,
    model,
    memory,
    problems,
    corrects,
    tag_lists: Sequence[Sequence[int]] | None,
    n_questions: int,
):
    """One legacy bundle step. Returns probabilities, new memory, attention, read."""

    k = model.k_emb_layer(problems)
    logits_w = torch_module.matmul(k, model.Mk.transpose(0, 1))
    logits_w = logits_w - logits_w.max(dim=-1, keepdim=True).values
    attention = torch_module.softmax(logits_w, dim=-1)
    read = torch_module.einsum("tm,md->td", attention, memory)
    if tag_lists is None:
        fused = read
        concept = torch_module.zeros_like(read)
    else:
        concept = mean_human_kc(torch_module, model.human_concept_emb, tag_lists)
        fused = read + concept
    joined = torch_module.cat([fused, k], dim=-1)
    features = torch_module.tanh(model.f_layer(joined))
    logit = model.p_layer(model.dropout_layer(features)).squeeze(-1)
    probability = torch_module.sigmoid(logit)
    interaction = problems.long() + int(n_questions) * corrects.long()
    values = model.v_emb_layer(interaction)
    erase_gate = torch_module.sigmoid(model.e_layer(values))
    add_gate = torch_module.tanh(model.a_layer(values))
    updated = memory
    for index in range(int(problems.shape[0])):
        weight = attention[index]
        erase = weight[:, None] * erase_gate[index][None, :]
        add = weight[:, None] * add_gate[index][None, :]
        updated = updated * (1.0 - erase) + add
    return probability, updated, attention, read, concept, logit


@dataclass(frozen=True)
class DKVMNRowTrace:
    row_index: int
    learner_index: int
    solving_id: int
    run_id: int
    run_start: int
    run_end: int
    problem_idx: int
    correct: int
    kc_ids: tuple[int, ...]
    probability: float
    logit: float
    attention_checksum: float
    read_checksum: float
    concept_checksum: float
    memory_checksum: float


@dataclass(frozen=True)
class DKVMNReplay:
    rows: tuple[DKVMNRowTrace, ...]
    runs: tuple[NeuralRun, ...]
    final_memory: dict

    @property
    def n_runs(self) -> int:
        return len(self.runs)


class _DKVMNBase:
    model_kind = "dkvmn_q"
    use_kc = False

    def __init__(
        self,
        n_questions: int,
        n_concepts: int = 1,
        *,
        dim_s: int = 64,
        size_m: int = 20,
        dropout: float = 0.2,
    ) -> None:
        torch = _torch()
        dkvmn_cls = _pykt_dkvmn()
        self.n_questions = int(n_questions)
        self.n_concepts = int(n_concepts)
        self.dim_s = int(dim_s)
        self.size_m = int(size_m)
        self.dropout = float(dropout)
        self.backend = dkvmn_cls(
            num_c=self.n_questions,
            dim_s=self.dim_s,
            size_m=self.size_m,
            dropout=self.dropout,
        )
        if self.use_kc:
            self.backend.human_concept_emb = torch.nn.Embedding(
                self.n_concepts, self.dim_s
            )
        self.backend.eval()

    def load_checkpoint(self, path: str | Path) -> None:
        torch = _torch()
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        self.backend.load_state_dict(state, strict=True)
        self.backend.eval()

    def save_runtime(self, path: str | Path, memory: dict | None = None) -> None:
        torch = _torch()
        torch.save(
            {
                "format_id": FORMAT_ID,
                "model_kind": self.model_kind,
                "n_questions": self.n_questions,
                "n_concepts": self.n_concepts,
                "dim_s": self.dim_s,
                "size_m": self.size_m,
                "dropout": self.dropout,
                "state_dict": self.backend.state_dict(),
                "memory": memory or {},
            },
            str(path),
        )

    @classmethod
    def load_runtime(cls, path: str | Path):
        torch = _torch()
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        if payload.get("format_id") != FORMAT_ID or payload.get("model_kind") != cls.model_kind:
            raise ValueError("DKVMN runtime file does not match this model")
        model = cls(
            int(payload["n_questions"]),
            int(payload["n_concepts"]),
            dim_s=int(payload["dim_s"]),
            size_m=int(payload["size_m"]),
            dropout=float(payload["dropout"]),
        )
        model.backend.load_state_dict(payload["state_dict"], strict=True)
        model.backend.eval()
        return model, payload.get("memory") or {}

    def replay(self, rows: Sequence[DKTRow], initial_memory: dict | None = None) -> DKVMNReplay:
        torch = _torch()
        self.backend.eval()
        runs = learner_solving_runs(
            [row.student_idx for row in rows],
            [row.solving_id for row in rows],
        )
        by_learner: dict[int, list[NeuralRun]] = {}
        for run in runs:
            by_learner.setdefault(run.learner_index, []).append(run)
        traces: list[DKVMNRowTrace] = []
        final_memory: dict = {}
        carried = initial_memory or {}
        run_id = 0
        for learner in sorted(by_learner):
            memory = (
                carried[learner].detach().clone()
                if learner in carried
                else self.backend.Mv0.detach().clone()
            )
            for run in by_learner[learner]:
                chosen = [rows[index] for index in run.row_positions]
                problems = torch.tensor([row.problem_idx for row in chosen], dtype=torch.long)
                corrects = torch.tensor([row.correct for row in chosen], dtype=torch.long)
                tags = [row.tag_idxs for row in chosen]
                with torch.no_grad():
                    memory_checksum = float(memory.detach().sum().cpu())
                    probability, memory, attention, read, concept, logit = predict_run(
                        torch,
                        self.backend,
                        memory,
                        problems,
                        corrects,
                        tags if self.use_kc else None,
                        self.n_questions,
                    )
                for offset, row in enumerate(chosen):
                    traces.append(
                        DKVMNRowTrace(
                            row_index=run.row_positions[offset],
                            learner_index=learner,
                            solving_id=run.solving_id,
                            run_id=run_id,
                            run_start=run.row_positions[0],
                            run_end=run.row_positions[-1],
                            problem_idx=int(row.problem_idx),
                            correct=int(row.correct),
                            kc_ids=row.tag_idxs,
                            probability=float(probability[offset].detach().cpu()),
                            logit=float(logit[offset].detach().cpu()),
                            attention_checksum=float(attention[offset].detach().sum().cpu()),
                            read_checksum=float(read[offset].detach().sum().cpu()),
                            concept_checksum=float(concept[offset].detach().sum().cpu()),
                            memory_checksum=memory_checksum,
                        )
                    )
                run_id += 1
            final_memory[learner] = memory.detach().cpu().clone()
        traces.sort(key=lambda item: item.row_index)
        return DKVMNReplay(tuple(traces), runs, final_memory)


class DKVMNQModel(_DKVMNBase):
    model_kind = "dkvmn_q"
    use_kc = False


class DKVMNQCModel(_DKVMNBase):
    model_kind = "dkvmn_qc"
    use_kc = True
