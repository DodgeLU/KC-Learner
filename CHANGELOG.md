# Changelog

## 0.1.0

First public release candidate. No release date is recorded here; this version has not been published.

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
