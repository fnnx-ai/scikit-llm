"""Default prompt templates per task."""

from skllm.prompts.classification import (
    FEW_SHOT_MULTI_LABEL,
    FEW_SHOT_SINGLE_LABEL,
    ZERO_SHOT_MULTI_LABEL,
    ZERO_SHOT_SINGLE_LABEL,
    render_examples,
    render_labels,
)
from skllm.prompts.extraction import EXTRACTION
from skllm.prompts.tagging import NER_TAGGING, render_entities
from skllm.prompts.text2text import SUMMARY, TRANSLATION, render_focus

__all__ = [
    "EXTRACTION",
    "FEW_SHOT_MULTI_LABEL",
    "FEW_SHOT_SINGLE_LABEL",
    "NER_TAGGING",
    "SUMMARY",
    "TRANSLATION",
    "ZERO_SHOT_MULTI_LABEL",
    "ZERO_SHOT_SINGLE_LABEL",
    "render_entities",
    "render_examples",
    "render_focus",
    "render_labels",
]
