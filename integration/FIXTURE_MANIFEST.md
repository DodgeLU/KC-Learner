# Formal dev5000 parity window

This file is the selection rule for `integration/test_legacy_ednet_parity.py`.
The test does not copy EdNet rows into KC-Learner.

Normal checks stay on `python -m unittest discover -s tests -v`.
This window is only built when that integration command is run and the local artifacts exist.

## What is read

* `project_c/models/ednet_art_kt.py` as the oracle. It is imported, not modified.
* `ednet_split_v1_train.parquet`
* `ednet_split_v1_valid.parquet`

VALID is read only so the test can call the formal `build_vocab` and `select_dev5000`. The compared rows are a TRAIN slice. `ednet_split_v1_test.parquet` is not opened.

The oracle and KC-Learner both start from zero state on that slice. This checks the streaming equations. It does not replay the full dev5000 checkpoint and it does not reproduce the published VALID metrics.

## Cohort

Same path as `train_ednet_irt_dev5000.py`:

1. Build the TRAIN+VALID union vocabulary.
2. `normalize_canonical` rebuilds `group_id` from `(user_id, solving_id)` and keeps physical row order.
3. Keep the first 5000 dense `student_idx` values in numeric order over the TRAIN+VALID union.
4. Expected TRAIN size is 806,148 rows. A mismatch stops the test.

## Window

Search only that TRAIN frame, in physical order.

Execution runs are contiguous `(student_idx, group_id)` blocks. A run that contains raw `tag_ids` `-1` is a barrier and is left out. The chosen slice is the earliest span of complete runs after such a barrier. It is not cut in the middle of a run. There is no fixed row cap: the span stops at the first run that makes every predicate below true.

* one run of length 1
* one run of length at least 2
* the same `(student_idx, group_id)` starts a later run after being interrupted
* at least two students, so a learner switch flushes a run
* both `correct=0` and `correct=1`
* one `problem_idx` appears at least twice
* one row has at least two KC ids
* one row's KC list contains a repeated id
* one KC id occurs in at least two runs

KC ids in this check are the formal dense `tag_idxs`, so repeated source tokens stay repeated. No row text is written back into this repository.

## Comparison

For IRT (`use_residual=False`, `eta_r=0`) and then AR-KT (`use_residual=True`, `eta_r=0.20`):

1. Run `phase="train"` from zero.
2. Continue `phase="eval"` on the same slice.
3. Compare row count, run spans, `p_global`, `p_art`, `r_bar`, `theta`, `b`, and `r_post`.
4. Float tolerance is `1e-12`, the formal runner's own repeatability bound.
5. Run-span mismatch fails the gate even if the metrics match.

IRT is compared first. AR-KT is not run if that comparison fails.

## Checkpoint restore window

`integration/test_legacy_checkpoint_parity.py` uses the same dev5000 cohort and the same run-aligned predicates, on the VALID frame only.

Inputs:

* `final_valid_pt/EdNet/IRT/ednet_irt_dev5000/state_train_only.npz`
* `final_valid_pt/EdNet/AR-KT/ednet_ar_kt_dev5000/state_train_only.npz`
* train parquet, only to rebuild the formal vocabulary and the 5,000-user membership
* valid parquet, the row source

TEST parquet is not opened. `state.npz` (the snapshot after VALID) is not loaded.

Both the legacy model and KC-Learner import that TRAIN-only snapshot, then run `phase="eval"` on the selected VALID slice. Float tolerance remains `1e-12`. IRT is compared before AR-KT.
