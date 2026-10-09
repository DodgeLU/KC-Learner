# External model adapter

An external learner model is not accepted only because it is an importable Python class.

The class must implement one KC-Learner adapter contract. KC-Learner loads it as `module:Class` through `kclearner.experiments.adapter_contract.load_adapter`. That loader does not consult `ALL_MODELS` or `create_model`. The six reference models stay on their own runners and are not migrated here.

The adapter owns traversal, predict-before-update inside a bundle, and which probabilities are returned. The runner owns split roles, `metric_mask`, and the TRAIN-then-VALID order. The adapter does not select splits and does not open TEST.

Two families exist. They are not one shared `predict` / `update` state API.

| Family | `contract_version` |
| --- | --- |
| Psychometric streaming | `psychometric_streaming_adapter_v1` |
| Neural bundle | `neural_bundle_adapter_v1` |

The construction config must declare `oov_state_semantics_id`. The loader does not choose one. The attribute on the adapter must equal that config value.

## Psychometric streaming adapter

This family follows the same execution shape as IRT and AR-KT: a stream of bundles, with prediction taken from the state that exists before the bundle, then an update after the bundle.

Required methods are `execute_phase`, `save_state`, `load_state`, and `validate_dataset_contract`.

`execute_phase(rows, phase=...)` accepts `train` and `valid`. It returns one `(interaction_id, probability)` pair per input row, in that input order. Learner-local state and any shared item or global state stay inside the adapter. Because shared state can depend on cross-learner order, the adapter's `traversal_id` must say which order it uses. The dataset row order is not itself a universal execution order.

`save_state` and `load_state` read and write the checkpoint format named by `checkpoint_format_id`. Formal execution saves this state immediately after TRAIN. VALID replay must not replace those bytes.

`examples/toy_streaming_adapter.py` is an architecture demonstration of this family. Its probabilities are a learner-local count. It is not a benchmark, not an IRT or AR-KT result, and not a replacement for those models.

## Neural bundle adapter

This family follows the same execution shape as the bundled DKT and DKVMN runners: TRAIN updates shared parameters, and a later phase replays learner sequences from a saved checkpoint. Seeded learner shuffle, TBPTT chunks, and pyKT layer names belong to the bundled neural recipe. They are not fields of this contract.

Required methods are `train`, `replay`, `save_checkpoint`, `load_checkpoint`, and `validate_dataset_contract`.

`train` walks the TRAIN rows in the adapter's declared traversal, updates the shared parameters, and returns one probability pair per input row. `save_checkpoint` then writes the TRAIN-only checkpoint named by `checkpoint_format_id`. `replay` scores a later split from that checkpoint. Learner-local cursor state may advance during replay. The checkpoint bytes must stay the TRAIN-only snapshot.

The formal carry-over phase calls `continue_replay` on the same object after a verified VALID replay. An adapter that declares `replay_valid_before_test_v1` must provide that method. `independent_test_state_v1` loads a fresh adapter and calls `replay` from the TRAIN-only checkpoint instead. The streaming contract has no `phase="test"`. Formal TEST rows are passed through `execute_phase(..., phase="valid")` after the runner has selected the TEST rows.

`examples/toy_neural_adapter.py` is an architecture demonstration of this family. It does not import torch or pyKT. It is not a benchmark, not a DKT or DKVMN result, and not a replacement for those models.

## Returned probabilities

Both families return `(interaction_id, probability)` in the same order as the rows the runner passed in. The runner rejects a different length, a reordered id, an unknown id, or a duplicate that does not match the next input row. Probabilities used as formal evidence must be finite and inside `[0, 1]`. `metric_mask` is applied by KC-Learner, not by the adapter.

## Required metadata

Every adapter exposes these non-empty strings:

| Attribute | Meaning |
| --- | --- |
| `contract_version` | One of the two family ids above |
| `implementation_id` | Name the author gives this implementation |
| `checkpoint_format_id` | Format written by `save_state` or `save_checkpoint` |
| `traversal_id` | Order in which the adapter walks learners, bundles, and rows |
| `execution_semantics_id` | Predict-before-update and state-update rule |
| `oov_state_semantics_id` | How OOV rows affect model state; must match the config |

`validate_dataset_contract` accepts or rejects the dataset's `oov_representation_semantics_id`. Representation semantics live on the dataset. State semantics live on the adapter. The loader does not equate them.

`supported_phase_state_semantics` lists which formal TEST policies the adapter implements: `independent_test_state_v1`, `replay_valid_before_test_v1`, or both. A formal run rejects a policy that is not in that tuple.

## Identity and fingerprints

`implementation_id` is a label. A self-declared label alone is not sufficient for a formal reproducible experiment.

Formal freeze requires `formal_freeze_eligible` from `kclearner.experiments.adapter_contract`. That gate distinguishes source mode:

- A non-editable installed distribution is identified by its distribution version (`installed_distribution` / `distribution_version`). The version string is the fingerprint. Editing installed files without changing the version is outside what this gate can see.
- An editable install, a local source file, or an unversioned import cannot rely on a version string. The version can stay still while the files change. One of the following fingerprints is required.
- `fingerprint_coverage = "single_module"` accepts the SHA-256 of that one adapter module file (`module_source_sha256` / `single_module`). This is the hash of the wrapper file only. It is not a hash of every module the adapter imports, and it is not a hash of KC-Learner itself.
- `fingerprint_coverage = "complete_implementation"` accepts the string returned by `implementation_fingerprint` (`adapter_supplied` / `complete_implementation`). KC-Learner stores that string. It does not walk the package and compute the hash itself. The adapter must make the string cover the implementation it claims.

Freeze and authorized TEST both recompute this implementation identity. A change that alters the accepted fingerprint between TRAIN and freeze, or between freeze and TEST, stops the run.

Absolute module paths and `source_mode` are descriptive. They are not part of the implementation hash.

Non-formal TRAIN/VALID (`execute_external_directory`, or the `run-external` CLI) can run an adapter that is not freeze-eligible. The CLI passes only `oov_state_semantics_id`. Other config, such as a seed, has to be supplied to `execute_external_directory` in Python. The state file written there is saved after VALID. It is not a formal TRAIN-only checkpoint and it is not an external freeze.
