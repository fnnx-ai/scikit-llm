"""Default prompt templates per task."""

from skllm.prompts.classification import (
    FEW_SHOT_MULTI_LABEL,
    FEW_SHOT_SINGLE_LABEL,
    ZERO_SHOT_MULTI_LABEL,
    ZERO_SHOT_SINGLE_LABEL,
    render_examples,
    render_labels,
)

__all__ = [
    "FEW_SHOT_MULTI_LABEL",
    "FEW_SHOT_SINGLE_LABEL",
    "ZERO_SHOT_MULTI_LABEL",
    "ZERO_SHOT_SINGLE_LABEL",
    "render_examples",
    "render_labels",
]
