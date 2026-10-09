# External dataset contract

KC-Learner does not automatically process arbitrary raw educational datasets.

The user preprocesses an external dataset into the canonical interaction representation. Preprocessing choices, including which rows are kept, how bundles are defined, and how out-of-vocabulary items are represented, remain the user's responsibility. KC-Learner then checks that directory and runs the declared experiment.

```text
raw external dataset
        |
        v
user-defined preprocessing
        |
        v
canonical interaction dataset
        |
        v
KC-Learner execution
```

This contract is separate from the EdNet-KT1 and ASSISTments 2017 reference paths. Those paths keep their own preparation code and their own execution columns (`solving_id` for EdNet, `group_id` for ASSISTments). An external directory does not use those columns.

Load a combined directory with `kclearner.data.canonical_dataset.load_canonical_dataset`. The loader does not train, freeze, or open a TEST evaluation.

## Directory

A combined canonical directory contains:

| File | Role |
| --- | --- |
| `declaration.json` | Identity-bearing declarations and descriptive provenance |
| `interactions.jsonl` | One canonical interaction per line, in stored order |
| `learner_ids.txt` | Ordered learner vocabulary, one id per line |
| `item_ids.txt` | Ordered item vocabulary, one id per line |
| `kc_ids.txt` | Ordered KC vocabulary, one canonical decimal token per line |

`declaration.json` must include `dataset_id`, `protocol_id`, `preprocessing_version`, `canonical_schema_version`, `interaction_identity_version`, `dataset_identity_version`, `split_order`, `split_roles`, `bundle_semantics_id`, `oov_representation_semantics_id`, `metric_mask_semantics`, and `provenance`. `source_fingerprint` is optional.

Accepted version strings are `canonical_interaction_schema_v1`, `canonical_interaction_hash_v1`, and `dataset_logical_identity_v1`. The schema version is not the interaction-hash version, and it is not the EdNet manifest label `kclearner_interaction_v1`.

`provenance` and `source_fingerprint` are stored. They do not enter either identity hash below. Free-form paths and notes therefore do not silently change scientific identity.

## Interaction schema

One row is one prediction target. The fields are the ten fields of `kclearner.data.schema.Interaction`:

`interaction_id`, `learner_id`, `item_id`, `correct`, `kc_ids`, `order`, `timestamp`, `bundle_id`, `split`, `metric_mask`.

`kc_ids` stays in the stored order. It may be empty or contain repeats. The loader does not sort it and does not turn it into a set. `correct` and `order` are integers. `timestamp` may be absent. `split` must name an id in `split_order`.

`bundle_id` is required on every external row. It is the bundle key used by external execution. Rows that share a learner and a `bundle_id` in a contiguous run are one bundle for predict-before-update. The question-table bundle id from EdNet is not this field. Reference EdNet execution continues to use `solving_id`, and reference ASSISTments execution continues to use `group_id`.

## Vocabularies

Line order in the three id files is the identity order. The loader does not sort the files. A repeated id is rejected rather than rewritten. KC tokens must be canonical decimal integers, including `0`; a token such as `01` is rejected.

The cohort hash and the item and KC vocabulary hashes are SHA-256 values of those ordered lines. They describe the declared vocabulary order. They are not a hash of an unordered set.

## Split roles

`split_order` lists split ids. `split_roles` maps each of those ids to exactly one of `train`, `valid`, or `test`. Each of those three roles is used once. Position in `split_order` does not assign a role. A missing role, a repeated role, or a role outside those three names fails the load.

Execution selects rows by the role string. The external runner does not infer a test split from the last split in the file.

## Metric mask

`metric_mask` is a boolean on the interaction. Rows with `metric_mask` false stay in the sequence. They can still update later state. Metric aggregation uses only rows whose mask is true. The formal VALID prediction trace includes masked rows for that reason. The trace hash does not store the mask bit; the mask is applied when metrics are computed.

`metric_mask_semantics` in the declaration is a required non-empty string. It names the masking rule the user claims to have applied. The loader does not interpret that string and does not invent a mask.

## OOV representation

`oov_representation_semantics_id` is required. It enters the dataset contract identity. The loader does not choose a representation and does not rewrite rows to match one.

This identifier describes how the canonical rows represent out-of-vocabulary items or KCs. It is not the model's state-update rule. The adapter config separately declares `oov_state_semantics_id`. Before TRAIN, the adapter's `validate_dataset_contract` must accept the dataset representation. A mismatch stops the run before training starts.

## Dataset logical identity

`dataset_logical_identity_v1` is the content identity. `dataset_logical_hash` is the SHA-256 of that object.

It binds:

- dataset, protocol, and preprocessing version;
- the ordered cohort hash;
- each split's ordered-row hash and row count, in `split_order`;
- the ordered item and KC vocabulary hashes.

It does not bind the model, seed, optimizer, device, filesystem path, clock, or Parquet bytes. Changing a row, the row order, a split membership, or a vocabulary line changes this hash.

## Dataset contract identity

`dataset_contract_identity_v1` is the execution-relevant semantic declaration. `dataset_contract_hash` is the SHA-256 of that object.

It binds:

- the dataset logical hash;
- `canonical_schema_version`;
- `bundle_semantics_id`;
- `oov_representation_semantics_id`;
- `metric_mask_semantics`;
- the ordered `(split_id, role)` pairs.

Logical identity answers "which rows, in which order." Contract identity answers "which scientific interpretation those rows carry for execution." Changing bundle, OOV-representation, metric-mask, or split-role declarations changes the contract hash and leaves the logical hash unchanged when the rows themselves are unchanged. Changing only `provenance` or `source_fingerprint` changes neither hash.

The reference freeze identity `experiment_freeze_identity_v1` is a different object. This dataset contract does not write it.
