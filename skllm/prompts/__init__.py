"""Default prompt templates per task."""

from skllm.prompts.classification import (
    FEW_SHOT_MULTI_LABEL,
    FEW_SHOT_SINGLE_LABEL,
    ZERO_SHOT_MULTI_LABEL,
    ZERO_SHOT_SINGLE_LABEL,
    render_examples,
    render_labels,
)
from skllm.prompts.text2text import SUMMARY, TRANSLATION, render_focus

__all__ = [
    "FEW_SHOT_MULTI_LABEL",
    "FEW_SHOT_SINGLE_LABEL",
    "SUMMARY",
    "TRANSLATION",
    "ZERO_SHOT_MULTI_LABEL",
    "ZERO_SHOT_SINGLE_LABEL",
    "render_examples",
    "render_focus",
    "render_labels",
]
