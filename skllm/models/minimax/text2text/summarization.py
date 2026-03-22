from skllm.models._base.text2text import BaseSummarizer as _BaseSummarizer
from skllm.llm.minimax.mixin import (
    MiniMaxTextCompletionMixin as _MiniMaxTextCompletionMixin,
)
from typing import Optional
from skllm.model_constants import MINIMAX_MODEL


class MiniMaxSummarizer(_BaseSummarizer, _MiniMaxTextCompletionMixin):
    """Text summarizer using MiniMax API."""

    def __init__(
        self,
        model: str = MINIMAX_MODEL,
        key: Optional[str] = None,
        max_words: int = 15,
        focus: Optional[str] = None,
    ) -> None:
        """
        Text summarizer using MiniMax API.

        Parameters
        ----------
        model : str, optional
            Model to use, by default "MiniMax-M2.7".
        key : Optional[str], optional
            Estimator-specific API key; if None, retrieved from
            the global config, by default None.
        max_words : int, optional
            Soft limit of the summary length, by default 15.
        focus : Optional[str], optional
            Concept in the text to focus on, by default None.
        """
        self._set_keys(key)
        self.model = model
        self.max_words = max_words
        self.focus = focus
