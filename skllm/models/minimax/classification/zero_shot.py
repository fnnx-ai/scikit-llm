from skllm.models._base.classifier import (
    SingleLabelMixin as _SingleLabelMixin,
    MultiLabelMixin as _MultiLabelMixin,
    BaseZeroShotClassifier as _BaseZeroShotClassifier,
    BaseCoTClassifier as _BaseCoTClassifier,
)
from skllm.llm.minimax.mixin import MiniMaxClassifierMixin as _MiniMaxClassifierMixin
from typing import Optional
from skllm.model_constants import MINIMAX_MODEL


class ZeroShotMiniMaxClassifier(
    _BaseZeroShotClassifier, _MiniMaxClassifierMixin, _SingleLabelMixin
):
    """Zero-shot text classifier using MiniMax models for single-label classification."""

    def __init__(
        self,
        model: str = MINIMAX_MODEL,
        default_label: str = "Random",
        prompt_template: Optional[str] = None,
        key: Optional[str] = None,
        **kwargs,
    ):
        """
        Zero-shot text classifier using MiniMax models.

        Parameters
        ----------
        model : str, optional
            Model to use, by default "MiniMax-M2.7".
        default_label : str, optional
            Default label for failed predictions; if "Random", selects
            randomly based on class frequencies, by default "Random".
        prompt_template : Optional[str], optional
            Custom prompt template to use, by default None.
        key : Optional[str], optional
            Estimator-specific API key; if None, retrieved from the
            global config, by default None.
        """
        super().__init__(
            model=model,
            default_label=default_label,
            prompt_template=prompt_template,
            **kwargs,
        )
        self._set_keys(key)


class CoTMiniMaxClassifier(
    _BaseCoTClassifier, _MiniMaxClassifierMixin, _SingleLabelMixin
):
    """Chain-of-thought text classifier using MiniMax models for single-label classification."""

    def __init__(
        self,
        model: str = MINIMAX_MODEL,
        default_label: str = "Random",
        prompt_template: Optional[str] = None,
        key: Optional[str] = None,
        **kwargs,
    ):
        """
        Chain-of-thought text classifier using MiniMax models.

        Parameters
        ----------
        model : str, optional
            Model to use, by default "MiniMax-M2.7".
        default_label : str, optional
            Default label for failed predictions; if "Random", selects
            randomly based on class frequencies, by default "Random".
        prompt_template : Optional[str], optional
            Custom prompt template to use, by default None.
        key : Optional[str], optional
            Estimator-specific API key; if None, retrieved from the
            global config, by default None.
        """
        super().__init__(
            model=model,
            default_label=default_label,
            prompt_template=prompt_template,
            **kwargs,
        )
        self._set_keys(key)


class MultiLabelZeroShotMiniMaxClassifier(
    _BaseZeroShotClassifier, _MiniMaxClassifierMixin, _MultiLabelMixin
):
    """Zero-shot text classifier using MiniMax models for multi-label classification."""

    def __init__(
        self,
        model: str = MINIMAX_MODEL,
        default_label: str = "Random",
        max_labels: Optional[int] = 5,
        prompt_template: Optional[str] = None,
        key: Optional[str] = None,
        **kwargs,
    ):
        """
        Multi-label zero-shot text classifier using MiniMax models.

        Parameters
        ----------
        model : str, optional
            Model to use, by default "MiniMax-M2.7".
        default_label : str, optional
            Default label for failed predictions; if "Random", selects
            randomly based on class frequencies, by default "Random".
        max_labels : Optional[int], optional
            Maximum number of labels per sample, by default 5.
        prompt_template : Optional[str], optional
            Custom prompt template to use, by default None.
        key : Optional[str], optional
            Estimator-specific API key; if None, retrieved from the
            global config, by default None.
        """
        super().__init__(
            model=model,
            default_label=default_label,
            max_labels=max_labels,
            prompt_template=prompt_template,
            **kwargs,
        )
        self._set_keys(key)
