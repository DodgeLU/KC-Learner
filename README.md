# KC-Learner

Reproducible framework for knowledge-component-aware sequential learner modeling across heterogeneous model representations.

The Python package name is `kclearner`. The version recorded in `pyproject.toml` and `kclearner.__version__` is `0.2.0`. The Zenodo archive for this release is [10.5281/zenodo.23265408](https://doi.org/10.5281/zenodo.23265408).

## Overview

KC-Learner is research software for running learner–item–response–knowledge-component logs through a shared experiment path. It standardizes interaction records, keeps dataset-specific knowledge-component encodings explicit, and runs several existing learner-model families without forcing them to share one internal mathematical implementation.

It is not a learning-management system, a student-facing tutor, a recommendation system, or a new state-of-the-art knowledge-tracing algorithm.

## Key features

- One interaction remains one prediction target, including items with several knowledge components.
- Canonical interaction checks for identity, order, bundles, and time.
- EdNet-KT1 and ASSISTments 2017 adapters, plus a synthetic dataset that does not read those sources.
- Source order and repeated knowledge-component tokens are preserved. A missing EdNet tag is an empty `kc_ids` tuple, not an invented `UNKNOWN_KC`.
- Bundle-aware sequential execution, including predict-before-update behavior.
- Six model variants across three families, selected from one corrected-EdNet protocol file.
- Run artifacts that record the config hash, environment, and dataset identity.
- Unit and semantic tests on synthetic data. Local regression checks stay outside the default test command.

## Supported learner-model families

The families do not share one internal update rule.

- Psychometric / online latent-state: `irt`, `ar_kt`
- Recurrent: `dkt_q`, `dkt_qc`
- Memory-augmented: `dkvmn_q`, `dkvmn_qc`

`Q` variants do not consume knowledge-component ids. `QC` variants do. IRT and AR-KT are deterministic; seed 42 is stored as provenance only. Formal neural runs accept only the frozen publication seeds: EdNet `42, 43, 44` and ASSISTments `42, 43, 44, 45, 46`. Dry-run and smoke may use another integer seed, and those runs set `publication_protocol` to false. LSTM and DKVMN modules come from `pykt-toolkit==0.0.38` when the neural extra is installed.

## Supported datasets

- EdNet-KT1, through the corrected adapter `ednet_kt1_corrected_v1`
- ASSISTments 2017, through `load_assistments2017_main_v1`
- Synthetic interactions from `make_toy_interactions()`

## Extending KC-Learner

KC-Learner can run two kinds of extension beside the reference datasets and the six reference models:

1. External canonical datasets.
2. External learner-model adapters.

An external dataset enters only after the user has turned raw records into the canonical interaction contract. KC-Learner does not read an arbitrary raw archive and infer that contract. An external model enters only by implementing one of the two adapter contracts. A Python class that can be imported is not, by itself, a formal experiment.

The software guarantees controlled execution and comparison under the contracts that the dataset and the adapter declare. It does not guarantee that every model works well on every dataset, that every external dataset is automatically compatible, or that an external implementation generalizes.

Reference EdNet and ASSISTments runs stay on their existing commands. They are not migrated onto the external adapter path.

- [External dataset contract](docs/external_dataset_contract.md)
- [External model adapter](docs/external_model_adapter.md)
- [External formal workflow](docs/external_formal_workflow.md)

`examples/toy_streaming_adapter.py` and `examples/toy_neural_adapter.py` show the two adapter shapes. They are architecture demonstrations. They are not benchmark models, scientific performance results, or replacements for IRT, AR-KT, DKT, or DKVMN.

Non-formal external TRAIN then VALID is available as `run-external`. The formal external freeze and TEST workflow is a Python API, documented in the formal-workflow note. It has no CLI command.

## Data availability and third-party datasets

KC-Learner does not redistribute EdNet-KT1 or ASSISTments 2017. Raw archives, per-user interaction dumps, processed cohorts, and locally generated publication files are not part of this repository. `generated/` and `runs/` are gitignored.

These datasets are third-party resources. The KC-Learner software license does not cover them. Users obtain each dataset from its provider and comply with that provider's access rules and terms. Do not commit dataset files, access tokens, or provider credentials into this repository.

### EdNet-KT1

The provider repository is [riiid/ednet](https://github.com/riiid/ednet). Its README publishes the KT1 archive link `https://bit.ly/ednet_kt1` and the contents archive link `https://bit.ly/ednet-content`. That README states that the dataset is released under Creative Commons Attribution-NonCommercial 4.0 International for research purposes. KC-Learner does not restate that statement as its own license grant.

KT1 is a directory of per-user files named `{user_id}.csv`. The contents archive supplies the question table with columns `question_id`, `bundle_id`, `explanation_id`, `correct_answer`, `part`, `tags`, and `deployed_at`. Pass that questions CSV to the adapter. Required interaction columns are `user_id`, `timestamp`, `solving_id`, `question_id`, and `user_answer`. Preparation reads a directory of `{user_id}.csv` files plus the questions CSV. It does not ship a user-id list.

```bash
python -m kclearner.experiments.cli prepare --dataset ednet_kt1 --raw-dir /path/to/KT1 --questions /path/to/questions.csv
```

`prepare` requires the parquet extra (`pyarrow`). The command writes `generated/ednet_kt1_corrected_v1/` unless `--output-dir` is set. It applies the historical frozen 50k sampler, then the dev5000 cohort. The publication vocabulary is the 50k TRAIN+VALID item and KC sets. The dataset logical hash does not include the model, seed, device, or Parquet bytes. A later prepare of the same inputs reuses a completed directory when the identity matches. `--rebuild` is the explicit rebuild switch.

Historical EdNet IRT and AR-KT checkpoints were trained on `ednet_kt1_preproc_v1` split parquets, not on this corrected order. Those predictive numbers are not an oracle for the corrected dataset. See `docs/historical_psychometric_outputs.md`.

The question table's `bundle_id` is content identity. The interaction bundle is `(learner_id, solving_id)`.

### ASSISTments 2017

Obtain ASSISTments 2017 through the access procedure published by the ASSISTments data provider, and comply with the applicable terms of use. The provider page for the 2017 data-mining competition dataset is:

https://sites.google.com/view/assistmentsdatamining/dataset

That page states that access requires signup, that the dataset link is sent by email, and that use requires agreement to the provider's Terms of Use. The page text fetched for this release does not include a stable form URL, so this README does not invent one.

`load_assistments2017_main_v1` expects columns `student_id`, `problem_id`, `skill_id`, `correct`, `timestamp`, and `attempt_count`. There is no required path inside this repository. Blank skills are dropped and counted.

```bash
python -m kclearner.experiments.cli prepare --dataset assistments2017 --raw-file /path/to/primary.csv
```

This also requires `pyarrow`. The default output is `generated/assistments2017_main_v1/`. Rows are ordered by `student_id`, then `timestamp`, then source-file order. The execution bundle is `(student_id, timestamp)`.

On ASSISTments TEST, 949 item-OOV rows stay in the IRT and AR-KT sequence, use problem index 0, and are excluded from metrics. DKT and DKVMN drop those same rows from state and from metrics. The two rules are intentionally different.

## Installation

Python 3.10 or newer.

```bash
pip install -e .
```

Neural models:

```bash
pip install -e ".[neural]"
```

Parquet reads and writes import `pyarrow`. Core import and the synthetic example do not. There is no version pin:

```bash
pip install -e ".[parquet]"
```

The console script is `kclearner`. If that script is not on `PATH`, use `python -m kclearner.experiments.cli`.

Core import does not import torch or pyarrow. `prepare`, and any run that reads a prepared Parquet dataset, imports pyarrow. Neural model construction imports torch and pyKT only for `dkt_q`, `dkt_qc`, `dkvmn_q`, and `dkvmn_qc`.

One CPU smoke that completed on prepared EdNet data used Python 3.10.18, torch `2.13.0+cpu`, and pykt-toolkit `0.0.38`, with CUDA unavailable. That check covered all four neural models for one epoch, checkpoint write, and checkpoint restore. It does not promise the same bitwise result on every torch build or GPU. A machine that cannot load torch is an environment problem, not a change to the model mathematics.

## Requirements

| Piece | Declared requirement |
| --- | --- |
| Python | `>=3.10` |
| Core | `numpy>=1.23` |
| Neural extra | `torch>=2.0`, `pykt-toolkit==0.0.38` |
| Parquet extra | `pyarrow` (no minimum declared) |
| Build | setuptools `>=68`, wheel |

IRT and AR-KT use NumPy and do not import PyTorch. DKT and DKVMN import PyTorch and pyKT only when that model is constructed.

## Quick start

### Quick verification without third-party datasets

A fresh clone can check the package without EdNet or ASSISTments.

```bash
pip install -e .
python examples/toy_dataset.py
```

Expected first line:

```text
interactions=7 targets=7 ok=True
```

### Full dataset experiments

Corrected-EdNet CLI runs expect a locally built `generated/ednet_kt1_corrected_v1/` directory. ASSISTments runs expect a CSV obtained from the provider. Neither dataset is in this repository. See [Data availability and third-party datasets](#data-availability-and-third-party-datasets).

EdNet tag parsing, with no data files:

```python
from kclearner import parse_ednet_tags

parse_ednet_tags("5;2;182")  # (5, 2, 182)
parse_ednet_tags("-1")       # ()
```

`parse_ednet_tags("5;-1;2")` raises `EdNetTagParseError`.

## Configuration-driven experiments

The CLI loads `configs/ednet_corrected/protocol.json` unless `--protocol` points elsewhere. That file's `protocol_id` must be `ednet_corrected_publication_v1`. A run is the protocol plus `--model` and, for a neural model, `--seed`.

`--device` accepts `auto`, `cpu`, or `cuda`. The default in the protocol file is `auto`.

Dry-run checks the local dataset identity and constructs the model. It does not train.

```bash
python -m kclearner.experiments.cli run --model irt --device cpu --dry-run
```

The default dataset directory is `generated/ednet_kt1_corrected_v1`. That directory is produced locally and is not in this repository. Without it, the command stops. A neural dry-run also needs the neural extra and a seed:

```bash
python -m kclearner.experiments.cli run --model dkt_q --seed 42 --device cpu --dry-run
```

`--dry-run`, `--smoke`, and `--authorize-formal` are mutually exclusive. `--smoke` runs a few source-order bundles and writes a non-publication artifact. Omitting all three stops before training. `--authorize-formal` authorizes TRAIN then VALID only. It does not authorize TEST.

Dataset logical identity is fixed before the model, seed, optimizer, or device is chosen. Changing those run settings does not change the prepared dataset hash.

```bash
python -m kclearner.experiments.cli run --dataset assistments2017 --model irt --dry-run --dataset-dir /path/to/prepared
python -m kclearner.experiments.cli run --dataset ednet_kt1 --model dkt_q --seed 7 --smoke --device cpu --dataset-dir /path/to/prepared
```

Seed 7 is an exploratory smoke seed. It is outside the EdNet publication set, and the artifact records `publication_protocol: false`.

## Freeze and TEST

Formal TRAIN and VALID write `freeze_manifest.json`. TEST is a separate command. It requires the freeze, the TRAIN-only checkpoint, and `--enable-test-execution`. There is no `--phase test` shortcut.

```bash
python -m kclearner.experiments.cli evaluate --dataset assistments2017 --model irt --freeze /path/to/freeze_manifest.json --dataset-dir /path/to/prepared --checkpoint /path/to/state_train_only.npz --output-dir /path/to/test-output --enable-test-execution
```

Psychometric TEST replays VALID from the TRAIN-only state and stops before TEST if that replay misses the frozen VALID metrics. Neural TEST checks freeze identity and then continues the TRAIN then VALID state into TEST. It does not add a numerical VALID replay gate. An existing TEST result for the same freeze id is left in place.

## Sequential and bundle semantics

One row is one prediction target. Rows in a bundle are scored from the state that exists before that bundle, and the state updates after those predictions. For the corrected EdNet protocol the bundle is `(learner_id, solving_id)`.

IRT and AR-KT streaming walks physical row order and closes a contiguous run when `(student_idx, group_id)` changes. DKT and DKVMN take each learner's subsequence and cut bundles where `solving_id` changes. Those execution details are family-specific. The package does not collapse them into one shared state update.

`solving_id` and timestamp do not reorder the history. Inversions are reported. Duplicate responses are kept.

## Knowledge-component handling

`kc_ids` is a tuple. Length zero means the source recorded no knowledge component. Length greater than one stays on the same interaction. Repeated ids stay repeated.

EdNet `tags="-1"` and a blank tag cell become `()`. A `-1` token mixed with other tags is an error. ASSISTments blank skills are dropped rather than mapped to a placeholder id. Out-of-vocabulary behavior remains dataset- and model-specific.

## Outputs and experiment artifacts

A dry-run writes a directory under `runs/<run_id>/` (or `--runs-dir`):

- `config.json`: canonical run config
- `metadata.json`: version, dataset ids, hashes, model, seed, device
- `environment.json`: Python, NumPy, and, when installed, torch, pyKT, and CUDA
- `dry_run.json`: identity check and whether the model was constructed

These artifacts are marked `publication: false`. `runs/` is gitignored.

## Reproducibility and provenance

Preparation identity is a logical hash of protocol, cohort, split counts, vocabulary, and ordered interactions. It is not a hash of Parquet bytes, paths, or clocks. Two files can differ as bytes and still be the same dataset. Neural training is seed- and environment-dependent. This package does not claim bitwise identity across hardware or torch builds.

Each resolved run has a SHA-256 `config_hash` over the canonical config JSON. Metadata records the package version, dataset id, cohort hash, vocabulary hashes, seed role, requested device, and resolved device. IRT and AR-KT always execute on CPU. A CUDA request for those models is recorded and is not treated as a neural device. Neural `cuda` fails when CUDA is absent. Formal neural runs use a publication seed. Dry-run and smoke do not.

`project_c/` is not part of this repository. It is an external read-only regression reference when a local checkout exists. Integration tests that need it skip when those files are absent.

## Reviewer checklist

Without third-party data, after `pip install -e .`:

```bash
python -c "import kclearner, kclearner.data"
python examples/toy_dataset.py
python -m unittest discover -s tests -v
python -m kclearner.experiments.cli prepare --help
python -m kclearner.experiments.cli run --help
python -m kclearner.experiments.cli evaluate --help
```

Success indicators: the import prints nothing and returns, the toy example prints `interactions=7 targets=7 ok=True`, unit tests finish with failures=0, and each `--help` lists the subcommand. Neural tests skip when torch is not installed. That skip is not a core-install failure.

With EdNet, install the parquet extra, run the `prepare` command above, then:

```bash
python -m kclearner.experiments.cli run --dataset ednet_kt1 --model irt --dry-run --dataset-dir generated/ednet_kt1_corrected_v1
```

With ASSISTments, prepare from the provider CSV, then the same `run` form with `--dataset assistments2017`. Reviewers do not need full publication training or TEST to verify the package.

## Testing

From the repository root, after `pip install -e .`:

```bash
python -m unittest discover -s tests -v
python examples/toy_dataset.py
python -m kclearner.experiments.cli --help
```

`tests/` uses synthetic inputs. DKT and DKVMN tests skip when torch or pyKT cannot be imported. This command does not read EdNet, ASSISTments, or TEST partitions.

Optional local regression, only when the corresponding private artifacts exist:

```bash
python -m unittest discover -s integration -v
```

See `integration/FIXTURE_MANIFEST.md`. That command is not required to install or import the package.

## CPU and CUDA execution

`auto` uses CUDA when `torch.cuda.is_available()` is true, otherwise CPU. `cpu` forces CPU. `cuda` raises `DeviceError` when CUDA is missing; it does not fall back. IRT and AR-KT do not require a GPU. A CUDA run needs a CUDA build of PyTorch installed by the user. This repository does not pin a CUDA wheel.

## Repository structure

```text
src/kclearner/          package
  data/                 schema, validation, toy data, adapters
  models/               IRT, AR-KT, DKT, DKVMN
  experiments/          protocol, CLI, dry-run, smoke, external contracts
  sequence/             bundle pre-state contract
configs/                corrected-EdNet and ASSISTments protocols
docs/                   protocol notes, logical identity, external contracts
examples/               toy interactions and two architecture-only adapters
tests/                  synthetic unit and semantic tests
integration/            optional local regression; skips if artifacts are absent
```

`generated/` and `runs/` are local and ignored.

## Citation

If you use KC-Learner in research, please cite this release:

KC-Learner v0.2.0 — https://doi.org/10.5281/zenodo.23265408

Fu X, Chen Z, Lu W. KC-Learner. Version 0.2.0. Zenodo. https://doi.org/10.5281/zenodo.23265408

The previous release remains KC-Learner v0.1.0 — https://doi.org/10.5281/zenodo.23227279

Citation metadata are also available in [`CITATION.cff`](CITATION.cff).

## License

KC-Learner is released under the MIT License. The terms are in [`Licence.txt`](Licence.txt).

EdNet, ASSISTments, and any other third-party dataset are not covered by the KC-Learner MIT License. See [Data availability and third-party datasets](#data-availability-and-third-party-datasets).

## Support

Questions about KC-Learner: luwentao@sairi.com.cn

## Limitations / scope

- The Zenodo archive for this release is KC-Learner v0.2.0. The v0.1.0 archive remains the record of that earlier release.
- The public interface is the Python package, the examples, and the CLI above. External formal freeze and TEST are Python functions, not CLI commands. Protocol values live in `configs/*/protocol.json`.
- Extensions run under declared contracts. They do not add automatic raw-dataset interpretation or a claim that an external model generalizes. See [Extending KC-Learner](#extending-kc-learner).
- Default CLI invocation without `--dry-run` or `--smoke` does not run a full experiment; the formal entry point is blocked unless explicitly authorized.
- Corrected-EdNet dry-run and smoke require a locally built dataset directory that is not distributed here.
- Neural training depends on the optional PyTorch and pyKT extra.
- This release does not publish new corrected-EdNet predictive results.
- Knowledge-component fields are carried and used by the KC-aware variants. That is a software capability, not a claim that knowledge-component information has been shown to improve learner modeling.
