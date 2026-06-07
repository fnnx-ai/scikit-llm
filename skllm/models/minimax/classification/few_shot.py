from skllm.models._base.classifier import (
    BaseFewShotClassifier,
    BaseDynamicFewShotClassifier,
    SingleLabelMixin,
    MultiLabelMixin,
)
from skllm.llm.minimax.mixin import MiniMaxClassifierMixin
from skllm.models.gpt.vectorization import GPTVectorizer
from skllm.models._base.vectorizer import BaseVectorizer
from skllm.memory.base import IndexConstructor
from typing import Optional
from skllm.model_constants import MINIMAX_MODEL, OPENAI_EMBEDDING_MODEL


class FewShotMiniMaxClassifier(
    BaseFewShotClassifier, MiniMaxClassifierMixin, SingleLabelMixin
):
    """Few-shot text classifier using MiniMax API for single-label classification tasks."""

    def __init__(
        self,
        model: str = MINIMAX_MODEL,
        default_label: str = "Random",
        prompt_template: Optional[str] = None,
        key: Optional[str] = None,
        **kwargs,
    ):
        """
        Few-shot text classifier using MiniMax API.

        Parameters
        ----------
        model : str, optional
            Model to use, by default "MiniMax-M3".
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


class MultiLabelFewShotMiniMaxClassifier(
    BaseFewShotClassifier, MiniMaxClassifierMixin, MultiLabelMixin
):
    """Few-shot text classifier using MiniMax API for multi-label classification tasks."""

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
        Multi-label few-shot text classifier using MiniMax API.

        Parameters
        ----------
        model : str, optional
            Model to use, by default "MiniMax-M3".
        default_label : str, optional
            Default label for failed predictions; if "Random", selects
            randomly based on class frequencies, by default "Random".
        max_labels : Optional[int], optional
            Maximum labels per sample, by default 5.
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


class DynamicFewShotMiniMaxClassifier(
    BaseDynamicFewShotClassifier, MiniMaxClassifierMixin, SingleLabelMixin
):
    """Dynamic few-shot text classifier using MiniMax API with dynamic example selection."""

    def __init__(
        self,
        model: str = MINIMAX_MODEL,
        default_label: str = "Random",
        prompt_template: Optional[str] = None,
        key: Optional[str] = None,
        n_examples: int = 3,
        memory_index: Optional[IndexConstructor] = None,
        vectorizer: Optional[BaseVectorizer] = None,
        metric: Optional[str] = "euclidean",
        **kwargs,
    ):
        """
        Dynamic few-shot text classifier using MiniMax API.
        For each sample, N closest examples are retrieved from the memory.

        Parameters
        ----------
        model : str, optional
            Model to use, by default "MiniMax-M3".
        default_label : str, optional
            Default label for failed predictions; if "Random", selects
            randomly based on class frequencies, by default "Random".
        prompt_template : Optional[str], optional
            Custom prompt template to use, by default None.
        key : Optional[str], optional
            Estimator-specific API key; if None, retrieved from the
            global config, by default None.
        n_examples : int, optional
            Number of closest examples per class, by default 3.
        memory_index : Optional[IndexConstructor], optional
            Custom memory index, for details check `skllm.memory` submodule.
        vectorizer : Optional[BaseVectorizer], optional
            Scikit-llm vectorizer; if None, `GPTVectorizer` is used.
        metric : Optional[str], optional
            Metric used for similarity search, by default "euclidean".
        """
        if vectorizer is None:
            vectorizer = GPTVectorizer(model=OPENAI_EMBEDDING_MODEL, key=key)
        super().__init__(
            model=model,
            default_label=default_label,
            prompt_template=prompt_template,
            n_examples=n_examples,
            memory_index=memory_index,
            vectorizer=vectorizer,
            metric=metric,
        )
        self._set_keys(key)
