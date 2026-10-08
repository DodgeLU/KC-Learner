"""Map a frozen run config onto an already validated model class."""

from __future__ import annotations

from typing import Any

from kclearner.experiments.config import NEURAL_MODELS, RunConfig
from kclearner.models.ar_kt import ARKTModel
from kclearner.models.irt import IRTModel


def create_model(name: str, config: RunConfig, vocab: Mapping[str, int]):
    """Construct one model. ``vocab`` supplies learner, item, and KC sizes.

    Neural constructors import torch/pyKT only when that family is requested.
    """

    if name != config.model:
        raise ValueError(f"factory name {name!r} != config model {config.model!r}")
    recipe = config.protocol["models"][name]["recipe"]
    n_students = int(vocab["learner_count"])
    n_items = int(vocab["item_count"])
    n_kcs = int(vocab["kc_count"])
    if name == "irt":
        return IRTModel(
            n_students,
            n_items,
            n_kcs,
            theta_lr=float(recipe["theta_lr"]),
            b_lr=float(recipe["b_lr"]),
            theta_l2=float(recipe["theta_l2"]),
            b_l2=float(recipe["b_l2"]),
            learn_b=bool(recipe["learn_b"]),
            freeze_global_in_eval=bool(recipe["freeze_global_in_eval"]),
        )
    if name == "ar_kt":
        return ARKTModel(
            n_students,
            n_items,
            n_kcs,
            eta_r=float(recipe["eta_r"]),
            theta_lr=float(recipe["theta_lr"]),
            b_lr=float(recipe["b_lr"]),
            theta_l2=float(recipe["theta_l2"]),
            b_l2=float(recipe["b_l2"]),
            learn_b=bool(recipe["learn_b"]),
            freeze_global_in_eval=bool(recipe["freeze_global_in_eval"]),
            residual_update="ednet_token",
        )
    if name in NEURAL_MODELS:
        return _create_neural(name, recipe, n_items, n_kcs)
    raise ValueError(f"unknown model {name!r}")


def _create_neural(name: str, recipe: Mapping[str, Any], n_items: int, n_kcs: int):
    if name == "dkt_q":
        from kclearner.models.dkt import DKTQModel

        return DKTQModel(n_items, emb_size=int(recipe["emb_size"]), dropout=float(recipe["dropout"]))
    if name == "dkt_qc":
        from kclearner.models.dkt import DKTQCModel

        return DKTQCModel(
            n_items,
            n_kcs,
            emb_size=int(recipe["emb_size"]),
            dropout=float(recipe["dropout"]),
        )
    if name == "dkvmn_q":
        from kclearner.models.dkvmn import DKVMNQModel

        return DKVMNQModel(
            n_items,
            dim_s=int(recipe["dim_s"]),
            size_m=int(recipe["size_m"]),
            dropout=float(recipe["dropout"]),
        )
    from kclearner.models.dkvmn import DKVMNQCModel

    return DKVMNQCModel(
        n_items,
        n_kcs,
        dim_s=int(recipe["dim_s"]),
        size_m=int(recipe["size_m"]),
        dropout=float(recipe["dropout"]),
    )
