# KC-Learner

Reproducible framework for knowledge-component-aware sequential learner modeling across heterogeneous model representations.

The Python package name is `kclearner`. The version recorded in `pyproject.toml` and `kclearner.__version__` is `0.1.0`.

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

`Q` variants do not consume knowledge-component ids. `QC` variants do. IRT and AR-KT are deterministic; seed 42 is stored as provenance only. The four neural variants require a publication seed of 42, 43, or 44. Their LSTM and DKVMN memory modules come from `pykt-toolkit==0.0.38`.

## Supported datasets

- EdNet-KT1, through the corrected adapter `ednet_kt1_corrected_v1`
- ASSISTments 2017, through `load_assistments2017_main_v1`
- Synthetic interactions from `make_toy_interactions()`

## Data availability and third-party datasets

KC-Learner does not redistribute EdNet-KT1 or ASSISTments 2017. Raw archives, per-user interaction dumps, processed cohorts, and locally generated publication files are not part of this repository. `generated/` and `runs/` are gitignored.

These datasets are third-party resources. The KC-Learner software license does not cover them. Users obtain each dataset from its provider and comply with that provider's access rules and terms. Do not commit dataset files, access tokens, or provider credentials into this repository.

### EdNet-KT1

The provider repository is [riiid/ednet](https://github.com/riiid/ednet). Its README publishes the KT1 archive link `https://bit.ly/ednet_kt1` and the contents archive link `https://bit.ly/ednet-content`. That README states that the dataset is released under Creative Commons Attribution-NonCommercial 4.0 International for research purposes. KC-Learner does not restate that statement as its own license grant.

KT1 is a directory of per-user files named `{user_id}.csv`. The contents archive supplies the question table with columns `question_id`, `bundle_id`, `explanation_id`, `correct_answer`, `part`, `tags`, and `deployed_at`. Pass that questions CSV to the adapter. The fixture loader is:

```python
from kclearner import load_ednet_kt1_interactions

result = load_ednet_kt1_interactions("KT1.csv", "questions.csv")
```

Required interaction columns are `user_id`, `timestamp`, `solving_id`, `question_id`, and `user_answer`. The corrected cohort builder `kclearner.data.ednet_corrected.build_corrected_rows` reads `raw_root/{user_id}.csv` plus the questions CSV. `write_corrected_dataset` writes a local directory, by convention `generated/ednet_kt1_corrected_v1/`, containing `dev5000_users.txt`, `manifest.json`, `publication_vocab.json`, and `train.parquet`, `valid.parquet`, and `test.parquet`. That directory stays on the machine that built it.

The question table's `bundle_id` is content identity. The interaction bundle is `(learner_id, solving_id)`.

### ASSISTments 2017

Obtain ASSISTments 2017 through the access procedure published by the ASSISTments data provider, and comply with the applicable terms of use. The provider page for the 2017 data-mining competition dataset is:

https://sites.google.com/view/assistmentsdatamining/dataset

That page states that access requires signup, that the dataset link is sent by email, and that use requires agreement to the provider's Terms of Use. The page text fetched for this release does not include a stable form URL, so this README does not invent one.

`load_assistments2017_main_v1` takes a CSV path chosen by the caller. It expects columns `student_id`, `problem_id`, `skill_id`, `correct`, `timestamp`, and `attempt_count`. There is no required path inside this repository. Blank skills are dropped and counted. They are not replaced with an invented knowledge-component id. This loader does not score TEST.

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

`--smoke` runs a few source-order bundles and writes a non-publication artifact. Omitting both `--dry-run` and `--smoke` calls the formal path, which this CLI refuses unless `--authorize-formal` is passed. Formal execution is not the quick start, and it does not by itself authorize TEST scoring.

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

Each resolved run has a SHA-256 `config_hash` over the canonical config JSON. Metadata records the package version, dataset id, cohort hash, vocabulary hashes, seed role, and the resolved device. IRT and AR-KT record seed 42 as provenance only. Neural runs use the requested publication seed.

`project_c/` is not part of this repository. It is an external read-only regression reference when a local checkout exists. Integration tests that need it skip when those files are absent.

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
  experiments/          protocol, CLI, dry-run, smoke
  sequence/             bundle pre-state contract
configs/                corrected-EdNet protocol
docs/                   protocol notes and the legacy reference
examples/toy_dataset.py synthetic quick start
tests/                  synthetic unit and semantic tests
integration/            optional local regression; skips if artifacts are absent
```

`generated/` and `runs/` are local and ignored.

## Citation

If you use KC-Learner in research, please cite the archived software release:

Fu X, Chen Z, Lu W. KC-Learner. Version 0.1.0. Zenodo. https://doi.org/10.5281/zenodo.23227279

Citation metadata are also available in [`CITATION.cff`](CITATION.cff).

## License

KC-Learner is released under the MIT License. The terms are in [`Licence.txt`](Licence.txt).

EdNet, ASSISTments, and any other third-party dataset are not covered by the KC-Learner MIT License. See [Data availability and third-party datasets](#data-availability-and-third-party-datasets).

## Support

Questions about KC-Learner: luwentao@sairi.com.cn

## Limitations / scope

- The public interface is the Python package, the toy example, and the corrected-EdNet CLI above. There is no separate YAML experiment schema.
- Default CLI invocation without `--dry-run` or `--smoke` does not run a full experiment; the formal entry point is blocked unless explicitly authorized.
- Corrected-EdNet dry-run and smoke require a locally built dataset directory that is not distributed here.
- Neural training depends on the optional PyTorch and pyKT extra.
- This release does not publish new corrected-EdNet predictive results.
- Knowledge-component fields are carried and used by the KC-aware variants. That is a software capability, not a claim that knowledge-component information has been shown to improve learner modeling.
