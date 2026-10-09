"""Evaluation-semantics compatibility boundary.

``EVALUATION_SEMANTICS`` is the scientific contract shared by freeze
identity and TEST evaluation. Increment it when a change alters:

* metric definitions;
* metric eligibility;
* TEST masking;
* OOV state behavior;
* sequential update semantics;
* predict-before-update or bundle behavior;
* checkpoint interpretation that changes predictions.

Do not increment it for documentation, log formatting, timestamps, or
filesystem and path handling. Do not hash this source file into the
version. The version string itself is the compatibility gate.
"""

EVALUATION_SEMANTICS = "evaluation_semantics_v1"
