"""Thin DKT-Q / DKT-QC wrappers over pinned pyKT ``DKT`` modules.

The LSTM, interaction embedding, and output layer stay inside
``pykt.models.dkt.DKT``. This module does not call ``DKT.forward``,
because that call drops the hidden state. Scoring uses the pre-run
hidden state. The current response enters the LSTM only after that
run's predictions exist.

QC adds the legacy input ``q_emb + mean(concept_interaction_emb)``.
Empty KC lists contribute zeros. Duplicate KC ids stay in the mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from kclearner.models.neural_runs import NeuralRun, learner_solving_runs

FORMAT_ID = "kclearner_dkt_runtime_v1"


def _torch():
    import torch

    return torch


def _pykt_dkt():
    from pykt.models.dkt import DKT

    return DKT


@dataclass(frozen=True)
class DKTRow:
    student_idx: int
    problem_idx: int
    correct: int
    solving_id: int
    tag_idxs: tuple[int, ...] = ()
    interaction_id: str = ""


@dataclass(frozen=True)
class DKTRowTrace:
    row_index: int
    interaction_id: str
    learner_index: int
    learner_row: int
    solving_id: int
    run_id: int
    run_start: int
    run_end: int
    problem_idx: int
    correct: int
    kc_ids: tuple[int, ...]
    logit: float
    probability: float
    hidden_pre: tuple[float, ...]
    kc_contribution_checksum: float


@dataclass(frozen=True)
class DKTReplay:
    rows: tuple[DKTRowTrace, ...]
    runs: tuple[NeuralRun, ...]
    final_state: dict

    @property
    def n_runs(self) -> int:
        return len(self.runs)


def _require_backend():
    torch = _torch()
    dkt_cls = _pykt_dkt()
    return torch, dkt_cls


def qc_input(
    torch_module,
    model,
    problem: "torch.Tensor",
    correct: "torch.Tensor",
    tag_lists: Sequence[Sequence[int]],
    n_questions: int,
    n_concepts: int,
):
    """Legacy ``q_emb + mean(concept interaction)`` input. Duplicates stay."""

    q_int = problem.long() + int(n_questions) * correct.long()
    q_emb = model.interaction_emb(q_int)
    length = int(problem.shape[0])
    device = problem.device
    if n_concepts <= 0 or length == 0:
        return q_emb, torch_module.zeros(length, dtype=q_emb.dtype, device=device)
    max_k = max((len(tags) for tags in tag_lists), default=0)
    if max_k == 0:
        return q_emb, torch_module.zeros(length, dtype=q_emb.dtype, device=device)
    padded = torch_module.zeros((length, max_k), dtype=torch_module.int64, device=device)
    mask = torch_module.zeros((length, max_k), dtype=torch_module.bool, device=device)
    for index, tags in enumerate(tag_lists):
        if not tags:
            continue
        values = torch_module.tensor([int(tag) for tag in tags], dtype=torch_module.int64, device=device)
        padded[index, : values.numel()] = values
        mask[index, : values.numel()] = True
    concept_id = padded + int(n_concepts) * correct.long().unsqueeze(1)
    gathered = model.concept_interaction_emb(concept_id) * mask.unsqueeze(-1)
    count = mask.sum(dim=1, keepdim=True).to(gathered.dtype).clamp(min=1.0)
    concept = gathered.sum(dim=1) / count
    empty = ~mask.any(dim=1)
    concept = concept.clone()
    concept[empty] = 0
    checksum = concept.detach().float().sum(dim=1)
    return q_emb + concept, checksum


class _DKTBase:
    model_kind = "dkt_q"
    use_kc = False

    def __init__(
        self,
        n_questions: int,
        n_concepts: int = 1,
        *,
        emb_size: int = 200,
        dropout: float = 0.1,
    ) -> None:
        torch, dkt_cls = _require_backend()
        self.n_questions = int(n_questions)
        self.n_concepts = int(n_concepts)
        self.emb_size = int(emb_size)
        self.dropout = float(dropout)
        self.backend = dkt_cls(
            num_c=self.n_questions,
            emb_size=self.emb_size,
            dropout=self.dropout,
        )
        if self.use_kc:
            self.backend.concept_interaction_emb = torch.nn.Embedding(
                int(2 * self.n_concepts),
                self.emb_size,
            )
        self.backend.eval()

    def load_checkpoint(self, path: str | Path) -> None:
        torch = _torch()
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        missing, unexpected = self.backend.load_state_dict(state, strict=True)
        if list(missing) or list(unexpected):
            raise ValueError(f"checkpoint mismatch missing={missing} unexpected={unexpected}")
        self.backend.eval()

    def save_runtime(self, path: str | Path, hidden: dict | None = None) -> None:
        torch = _torch()
        torch.save(
            {
                "format_id": FORMAT_ID,
                "model_kind": self.model_kind,
                "n_questions": self.n_questions,
                "n_concepts": self.n_concepts,
                "emb_size": self.emb_size,
                "dropout": self.dropout,
                "state_dict": self.backend.state_dict(),
                "hidden": hidden or {},
            },
            str(path),
        )

    @classmethod
    def load_runtime(cls, path: str | Path):
        torch = _torch()
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        if payload.get("format_id") != FORMAT_ID:
            raise ValueError(f"unsupported DKT runtime {payload.get('format_id')!r}")
        if payload.get("model_kind") != cls.model_kind:
            raise ValueError(
                f"runtime is {payload.get('model_kind')}, not {cls.model_kind}"
            )
        model = cls(
            int(payload["n_questions"]),
            int(payload["n_concepts"]),
            emb_size=int(payload["emb_size"]),
            dropout=float(payload["dropout"]),
        )
        model.backend.load_state_dict(payload["state_dict"], strict=True)
        model.backend.eval()
        return model, payload.get("hidden") or {}

    def replay(
        self,
        rows: Sequence[DKTRow],
        initial_state: dict | None = None,
    ) -> DKTReplay:
        """Score each run from its pre-run hidden state, then update.

        ``initial_state`` maps a learner index to ``(h, c)`` tensors of
        shape ``(1, 1, hidden)``. Missing learners start at zero.
        """

        torch = _torch()
        self.backend.eval()
        runs = learner_solving_runs(
            [row.student_idx for row in rows],
            [row.solving_id for row in rows],
        )
        by_learner: dict[int, list[NeuralRun]] = {}
        for run in runs:
            by_learner.setdefault(run.learner_index, []).append(run)
        traces: list[DKTRowTrace] = []
        final_state: dict = {}
        run_id = 0
        carried = initial_state or {}
        for learner in sorted(by_learner):
            learner_runs = by_learner[learner]
            order = [index for run in learner_runs for index in run.row_positions]
            hidden_size = int(self.backend.hidden_size)
            if learner in carried:
                h = carried[learner][0].detach().clone()
                c = carried[learner][1].detach().clone()
            else:
                h = torch.zeros(1, 1, hidden_size)
                c = torch.zeros(1, 1, hidden_size)
            problems = torch.tensor(
                [rows[index].problem_idx for index in order], dtype=torch.int64
            )
            corrects = torch.tensor(
                [rows[index].correct for index in order], dtype=torch.int64
            )
            tag_lists = [rows[index].tag_idxs for index in order]
            if self.use_kc:
                inputs, checksum = qc_input(
                    torch,
                    self.backend,
                    problems,
                    corrects,
                    tag_lists,
                    self.n_questions,
                    self.n_concepts,
                )
            else:
                interaction = problems.long() + self.n_questions * corrects.long()
                inputs = self.backend.interaction_emb(interaction)
                checksum = torch.zeros(len(order))
            with torch.no_grad():
                outputs, (h_n, c_n) = self.backend.lstm_layer(
                    inputs.unsqueeze(0), (h, c)
                )
            hidden_seq = outputs.squeeze(0)
            incoming = h.squeeze(0).squeeze(0)
            cursor = 0
            for run in learner_runs:
                start = cursor
                h_pre = incoming if start == 0 else hidden_seq[start - 1]
                vector = tuple(
                    float(value) for value in h_pre.detach().float().cpu().tolist()
                )
                with torch.no_grad():
                    logits = torch.nn.functional.linear(
                        h_pre.unsqueeze(0),
                        self.backend.out_layer.weight,
                        self.backend.out_layer.bias,
                    ).squeeze(0)
                for offset, row_index in enumerate(run.row_positions):
                    target = int(rows[row_index].problem_idx)
                    traces.append(
                        DKTRowTrace(
                            row_index=row_index,
                            interaction_id=rows[row_index].interaction_id,
                            learner_index=learner,
                            learner_row=start + offset,
                            solving_id=run.solving_id,
                            run_id=run_id,
                            run_start=run.row_positions[0],
                            run_end=run.row_positions[-1],
                            problem_idx=target,
                            correct=int(rows[row_index].correct),
                            kc_ids=rows[row_index].tag_idxs,
                            logit=float(logits[target].detach().cpu()),
                            probability=float(torch.sigmoid(logits[target]).detach().cpu()),
                            hidden_pre=vector,
                            kc_contribution_checksum=float(
                                checksum[start + offset].detach().cpu()
                            ),
                        )
                    )
                cursor += run.length
                run_id += 1
            final_state[learner] = (
                h_n.detach().cpu().clone(),
                c_n.detach().cpu().clone(),
            )
        traces.sort(key=lambda item: item.row_index)
        return DKTReplay(tuple(traces), runs, final_state)


class DKTQModel(_DKTBase):
    """Question-only DKT. KC ids are ignored."""

    model_kind = "dkt_q"
    use_kc = False


class DKTQCModel(_DKTBase):
    """DKT-Q plus the legacy mean KC-response input."""

    model_kind = "dkt_qc"
    use_kc = True
