"""KC-Learner: KC-aware sequential learner modeling.

This package is the new software line. It does not import or modify the
legacy ``project_c`` research workspace.
"""

from kclearner.data.adapters.base import ensure_bundle_id
from kclearner.data.adapters.ednet_diagnostics import EdNetDiagnostics
from kclearner.data.adapters.ednet_kt1 import (
    DropReasonCount,
    EdNetAdapterError,
    EdNetFilterStats,
    EdNetKT1LoadResult,
    load_ednet_kt1_interactions,
)
from kclearner.data.adapters.ednet_metadata import (
    CORRECTED_PREPROCESSING_ID,
    EdNetQuestionMetadata,
    EdNetTagParseError,
    load_ednet_question_metadata,
    parse_ednet_tags,
)
from kclearner.data.schema import Interaction, prediction_target_count
from kclearner.data.splitting import assign_fractional_bundle_split
from kclearner.data.toy import make_toy_interactions
from kclearner.data.validation import (
    DatasetValidator,
    ValidationIssue,
    ValidationResult,
)
from kclearner.models import ARKTModel, IRTModel, StreamingRow
from kclearner.sequence.contract import BundleStepRecord, replay_bundle_pre_state

__version__ = "0.2.0"

__all__ = [
    "CORRECTED_PREPROCESSING_ID",
    "BundleStepRecord",
    "DatasetValidator",
    "DropReasonCount",
    "EdNetAdapterError",
    "EdNetDiagnostics",
    "EdNetFilterStats",
    "EdNetKT1LoadResult",
    "EdNetQuestionMetadata",
    "EdNetTagParseError",
    "ARKTModel",
    "IRTModel",
    "Interaction",
    "StreamingRow",
    "ValidationIssue",
    "ValidationResult",
    "assign_fractional_bundle_split",
    "ensure_bundle_id",
    "load_ednet_kt1_interactions",
    "load_ednet_question_metadata",
    "make_toy_interactions",
    "parse_ednet_tags",
    "prediction_target_count",
    "replay_bundle_pre_state",
]
