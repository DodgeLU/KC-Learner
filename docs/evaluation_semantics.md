# Evaluation semantics

`evaluation_semantics_v1` is defined in
`src/kclearner/experiments/evaluation_semantics.py`. Freeze identity and
TEST evaluation both import that constant. It is a compatibility
boundary, not a package version and not a source-code hash.

Increment the version when a change alters metric definitions, metric
eligibility, TEST masking, OOV state behavior, sequential updates,
predict-before-update or bundle behavior, or checkpoint interpretation
that changes predictions.

Do not increment it for documentation, log formatting, timestamps, or
filesystem and path handling.
