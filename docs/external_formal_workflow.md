# External formal workflow

The external formal lifecycle is separate from the reference freeze. It uses `external_experiment_identity_v1` and does not write `experiment_freeze_identity_v1` or `evaluation_semantics_v1`. Reference EdNet and ASSISTments TEST stays on `evaluate --enable-test-execution`.

There is no CLI command for this workflow. Call the functions in `kclearner.experiments.external_formal`.

```text
TRAIN
  |
  v
TRAIN-only checkpoint
  |
  v
VALID exact replay
  |
  v
external freeze
  |
  v
explicit TEST authorization
  |
  v
TEST evaluation
```

## Before TRAIN

Build a canonical directory as described in [the dataset contract](external_dataset_contract.md). `materialize_formal_directory` copies it into role files:

- `interactions_train.jsonl`
- `interactions_valid.jsonl`
- `interactions_test.jsonl`
- `logical_identity.json`

Materialization reads every split, including TEST, so it can seal the TEST split hash and row count inside `logical_identity.json`. That sealed split is the TEST commitment. Formal TRAIN and freeze do not open `interactions_test.jsonl`. Storage layout does not replace the dataset identity: the logical hash and the contract hash are still the hashes from [the dataset contract](external_dataset_contract.md).

## TRAIN and the TRAIN-only checkpoint

`run_external_formal` checks the sealed document, recomputes the TRAIN and VALID split hashes, and does not read the TEST file. It loads the adapter, checks that the adapter declares the requested `phase_state_semantics_id`, and checks that the implementation is eligible for formal freeze. See [the adapter contract](external_model_adapter.md).

It then runs TRAIN only. Immediately after TRAIN it writes `checkpoint_train_only.json`, reads those bytes, and stores their SHA-256. VALID runs on a newly loaded adapter restored from those bytes. If VALID changes the checkpoint file, the run stops.

The persisted checkpoint remains the TRAIN-only snapshot. VALID may advance learner-local state on the temporary restored object. That object is not the frozen checkpoint.

## VALID exact replay

v1 verification is `exact_replay_v1`. Reference VALID tolerances are not reused.

VALID evidence has two parts:

- aggregate metrics after `metric_mask` is applied;
- the SHA-256 of the ordered VALID prediction trace.

The trace has one entry per VALID row the runner presented, in that order, including rows with `metric_mask` false. Masked rows stay in the evidence because they can still change later state. A missing, extra, duplicate, or reordered probability is rejected before the hash is formed.

`write_external_freeze` reloads the implementation, checks that its identity still matches the run, checks the TRAIN and VALID files again, and replays VALID from the checkpoint. The trace and the metric evidence must match the run. The checkpoint bytes must still match the TRAIN-only digest. The manifest is `external_freeze_manifest.json`. Its `external_freeze_id` hashes the scientific fields. Paths and source mode sit in descriptive provenance and do not enter that id.

## TEST authorization

`authorize_external_test` does not score TEST unless `enable_test=True`.

Before it opens `interactions_test.jsonl` it checks the freeze identity, the dataset logical and contract hashes against the sealed TRAIN/VALID files, the implementation fingerprint, the config hash, the TRAIN-only checkpoint digest, and the VALID exact replay. It hashes the checkpoint before the VALID replay and again after that replay. If the bytes differ, authorization stops. The TEST file is not read, and `external_test_result.json` is not written.

Only after those checks does it open TEST. It then recomputes the TEST split hash and the TEST row count and compares them with the sealed commitment from materialization. A mismatch raises an error that includes the expected and observed hash and row count. No TEST metrics are computed, and no TEST result file is written.

If the commitment matches, TEST is scored from the phase-state policy below. The result file is marked `formal_test: true` and `publication: false`.

## Phase-state semantics

`phase_state_semantics_id` is an external policy. It is not `evaluation_semantics_v1`. The adapter must list the chosen id in `supported_phase_state_semantics`. The persisted checkpoint stays TRAIN-only under both policies.

### `independent_test_state_v1`

After the authorized VALID replay succeeds, KC-Learner loads a fresh adapter and restores the TRAIN-only checkpoint again. TEST is scored from that restored state. TEST does not inherit learner-local state from VALID.

### `replay_valid_before_test_v1`

After the authorized VALID replay succeeds, TEST continues on that same freshly verified object. A psychometric adapter receives the TEST rows through `execute_phase(..., phase="valid")`. A neural adapter continues with `continue_replay`. TEST inherits the learner-local state produced by that VALID replay. It does not load a post-VALID checkpoint from disk.

## What this workflow does not do

It does not preprocess raw data. It does not accept an adapter that has only an `implementation_id` and no sufficient fingerprint. It does not treat the non-formal `run-external` state file as a freeze. It does not claim that the resulting TEST score will match another dataset or another model.
