# Corrected EdNet publication protocol

This file freezes the next execution matrix. It does not authorize running it.

Dataset: `ednet_kt1_corrected_v1`  
Cohort: `ednet_kt1_dev5000`  
Cohort SHA-256: `d9bb296cef94ba4ca0fa80ac8da98b7179b537dbce14bcb7555d021423bcaf9b`

Split membership is the verified historical `ednet_split_v1` bundle cut:

```text
TRAIN  806,148
VALID  117,184
TEST   235,031
```

Sequence order is the raw eligible KT1 order. A bundle is `(learner_id, solving_id)`. Every row in that bundle is scored from the pre-bundle state, then the state updates. The historical shuffled-parquet contiguous-run rule stays legacy-regression-only.

KC tokens keep source order and multiplicity. `tags="-1"` is a missing KC (`kc_ids=()`), not an extra skill. The 222 blank `user_answer` rows stay filtered; that filter is the existing eligibility rule.

## Vocabulary

Publication indexes use the formal 50k `ednet_split_v1` TRAIN+VALID scope. TEST did not contribute.

```text
items  12,259   sha256 6ed21bad1c6212819044fc71433cd4b824612b6a4fd8b4f3c97b2a07ed5fe4ac
KCs       188   sha256 f17aefb8d1f0902a083b1ca6b95bec2d858efa223d3242923d6a54f71ca20287
learners 5,000  sha256 d9bb296cef94ba4ca0fa80ac8da98b7179b537dbce14bcb7555d021423bcaf9b
```

The historical 189-entry concept list included sentinel `-1`. Removing it leaves the same 188 real KCs. TEST item OOV under this vocabulary is 0 rows. The smaller dev5000 TRAIN+VALID item list (11,650) would mark 172 TEST rows as item OOV, so it is not the publication item vocabulary.

Learner index is the position in the ordered dev5000 list. Those positions are historical `student_idx` 0..4999.

## Differences from the legacy parquet

Intentional corrections:

- raw source order
- semantic `(user_id, solving_id)` bundles
- `tags="-1"` as missing KC

Preserved eligibility:

- blank or invalid `user_answer` rows are filtered and counted

Representation only:

- canonical columns
- source IDs in the parquet; dense indexes are applied by this protocol at run time

## Model recipes

IRT and AR-KT are deterministic. Seed 42 is provenance only.

```text
theta_lr 0.05
b_lr 0.02
theta_l2 1e-4
b_l2 1e-4
learn_b true
freeze_global_in_eval true
AR-KT eta_r 0.20
AR-KT use_time false
AR-KT residual_update ednet_token
```

DKT-Q and DKT-QC, seeds 42, 43, 44:

```text
emb_size 200
hidden_size 200
num_lstm_layers 1
dropout 0.1
Adam lr 0.001
weight_decay 0
effective_chunk_size 256
tbptt_target_rows 200
max_epochs 200
early_stop_patience_epochs 10
checkpoint_metric valid_auc
checkpoint_min_delta 0.001
gradient_clipping NONE
```

DKVMN-Q and DKVMN-QC, seeds 42, 43, 44:

```text
dim_s 64
size_m 20
dropout 0.2
```

The remaining optimizer, epoch, and checkpoint fields match DKT. QC variants consume KC ids. Q variants do not.

VALID selects the checkpoint. TEST is a later phase and is not used to choose hyperparameters, epochs, vocabulary, or architecture.

## Device

`device` is `auto`, `cpu`, or `cuda`. `auto` uses CUDA when it is present and CPU otherwise. `cuda` fails before training if CUDA is missing; it does not fall back. CPU is a supported publication device. Every run records the requested device, the resolved device, Python, torch, pyKT, CUDA availability, the CUDA version, and the GPU name when one is present.

Neural resume from `best_checkpoint_epoch_*.pt` restores the model, Adam state, epoch, and early-stopping counters. The CLI still refuses a formal run.

## Supported matrix and a possible smaller demonstration

The software supports all 14 corrected EdNet runs. Executing every seed is not a software-correctness requirement. A later paper may demonstrate the package with 6 runs: IRT, AR-KT, and the four neural models at seed 42. Seeds 43 and 44 estimate robustness. That choice is not settled here.

ASSISTments already has six-model MAIN_V1 parity. This protocol does not require retraining it.

## Formal matrix

14 runs, not started:

```text
ednet_corrected_irt
ednet_corrected_ar_kt
ednet_corrected_dkt_q_seed42
ednet_corrected_dkt_q_seed43
ednet_corrected_dkt_q_seed44
ednet_corrected_dkt_qc_seed42
ednet_corrected_dkt_qc_seed43
ednet_corrected_dkt_qc_seed44
ednet_corrected_dkvmn_q_seed42
ednet_corrected_dkvmn_q_seed43
ednet_corrected_dkvmn_q_seed44
ednet_corrected_dkvmn_qc_seed42
ednet_corrected_dkvmn_qc_seed43
ednet_corrected_dkvmn_qc_seed44
```

Command shape:

```text
python -m kclearner.experiments.cli run --model dkt_q --seed 42 --dry-run
```

A command without `--dry-run` or `--smoke` is refused. `--smoke` writes `NOT FOR PUBLICATION` artifacts only.
