from skllm.llm.vertex.mixin import VertexClassifierMixin as _VertexClassifierMixin
from skllm.models._base.classifier import (
    BaseZeroShotClassifier as _BaseZeroShotClassifier,
    SingleLabelMixin as _SingleLabelMixin,
    MultiLabelMixin as _MultiLabelMixin,
)
from typing import Optional
from skllm.model_constants import TEXT_BISON_MODEL, VERTEX_GEMINI_MODEL


class ZeroShotVertexClassifier(
    _BaseZeroShotClassifier, _SingleLabelMixin, _VertexClassifierMixin
):
    def __init__(
        self,
        model: str = VERTEX_GEMINI_MODEL,
        default_label: str = "Random",
        prompt_template: Optional[str] = None,
        **kwargs,
    ):
        """
        Zero-shot text classifier using Vertex AI models.

        Parameters
        ----------
        model : str, optional
            model to use.
        default_label : str, optional
            default label for failed prediction; if "Random" -> selects randomly based on class frequencies, by default "Random"
        prompt_template : Optional[str], optional
            custom prompt template to use, by default None
        """
        super().__init__(
            model=model,
            default_label=default_label,
            prompt_template=prompt_template,
            **kwargs,
        )


class MultiLabelZeroShotVertexClassifier(
    _BaseZeroShotClassifier, _MultiLabelMixin, _VertexClassifierMixin
):
    def __init__(
        self,
        model: str = VERTEX_GEMINI_MODEL,
        default_label: str = "Random",
        prompt_template: Optional[str] = None,
        max_labels: Optional[int] = 5,
        **kwargs,
    ):
        """
        Multi-label zero-shot text classifier using Vertex AI models.

        Parameters
        ----------
        model : str, optional
            model to use.
        default_label : str, optional
            default label for failed prediction; if "Random" -> selects randomly based on class frequencies, by default "Random"
        prompt_template : Optional[str], optional
            custom prompt template to use, by default None
        max_labels : Optional[int], optional
            maximum labels per sample, by default 5
        """
        super().__init__(
            model=model,
            default_label=default_label,
            prompt_template=prompt_template,
            max_labels=max_labels,
            **kwargs,
        )
