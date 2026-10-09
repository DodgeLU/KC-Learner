# Changelog

## Unreleased

Notes collected before the v0.2.0 heading. They are not a separate release.

- Restored the data package that v0.1.0 had omitted from Git. Root `/data/` stays ignored; `src/kclearner/data/` is source.
- Public deterministic `prepare` for corrected EdNet and ASSISTments 2017, with dataset logical identity separate from Parquet bytes and from the experiment.
- EdNet publication protocol reconstruction and ASSISTments public TRAIN/VALID/evaluate path.
- Freeze manifest after formal TRAIN and VALID. TEST uses `evaluate` and `--enable-test-execution`. `--authorize-formal` does not authorize TEST.
- Formal neural seeds stay the frozen sets. Dry-run and smoke may record another seed with `publication_protocol` false.
- Requested and resolved devices are recorded separately. Psychometric runs stay on CPU.
- ASSISTments TEST rows with a missing execution group id take a `(learner, timestamp)` bundle id. Historical TEST scripts left `group_id` unset and therefore updated one row at a time. See `docs/historical_psychometric_outputs.md`.
- External canonical dataset contract. The user preprocesses raw data into that contract. KC-Learner does not interpret arbitrary raw datasets.
- External adapter contract for two families: psychometric streaming and neural bundle. An importable class is not enough for a formal experiment.
- External formal lifecycle: TRAIN-only checkpoint, VALID exact replay, external freeze, and explicit TEST authorization. See `docs/external_formal_workflow.md`.

## v0.2.0

### Added

- Canonical dataset contract for external datasets. Users preprocess raw data into that contract.
- External model adapter contracts for the psychometric streaming and neural bundle families.
- External formal experiment lifecycle with a TRAIN-only checkpoint, VALID verification, freeze, and explicit TEST authorization.

### Changed

- Documentation for the external extension workflow.
- Release documentation for reproducible external experiments under those declared contracts.

### Validated

- External formal workflow tests.
- Canonical dataset and adapter contract tests.

## 0.1.0

Archived public release, DOI `10.5281/zenodo.23227279`. The data package was missing from that Git tree. The notes below describe the software that the release intended to contain.

Software in this version:

- Canonical interaction record: one row is one prediction target, with `kc_ids` as an ordered tuple that may be empty or contain repeats.
- Dataset validation for identity, order, bundle, and time.
- Synthetic dataset `make_toy_interactions()`, used by `examples/toy_dataset.py`, with no EdNet or ASSISTments files.
- EdNet-KT1 adapter. A standalone `tags="-1"` cell becomes no knowledge component. Mixed `-1` tags raise. Source order and repeated tokens are kept.
- Corrected EdNet protocol `ednet_corrected_publication_v1` in `configs/ednet_corrected/protocol.json`. The cohort file list is local and is not packaged.
- ASSISTments 2017 adapter `load_assistments2017_main_v1`. Blank skills are dropped and counted rather than replaced with a placeholder id.
- Six model variants: IRT, AR-KT, DKT-Q, DKT-QC, DKVMN-Q, and DKVMN-QC. The three families keep their own execution paths.
- Bundle-aware sequential runs, including predict-before-update scoring.
- Configuration-driven CLI for the corrected-EdNet protocol: `python -m kclearner.experiments.cli run`, with `--dry-run` and `--smoke`.
- Run artifacts for config, metadata, and environment. Dry-run and smoke artifacts are marked non-publication.
- Device selection `auto`, `cpu`, or `cuda`. CPU is supported. CUDA is optional and is not substituted when it was requested and is absent.
- Provenance fields: package version, config hash, dataset id, cohort hash, vocabulary hashes, seed, and resolved device.
- Synthetic tests under `tests/`. Optional integration checks skip when local regression artifacts are absent.

This note does not claim that knowledge-component fields have been shown to improve learner modeling, and it does not report corrected-EdNet predictive numbers.

EdNet-KT1 and ASSISTments 2017 are not included. Users obtain them from their providers.
