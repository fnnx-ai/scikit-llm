from skllm.models._base.text2text import BaseTranslator as _BaseTranslator
from skllm.llm.minimax.mixin import (
    MiniMaxTextCompletionMixin as _MiniMaxTextCompletionMixin,
)
from typing import Optional
from skllm.model_constants import MINIMAX_MODEL


class MiniMaxTranslator(_BaseTranslator, _MiniMaxTextCompletionMixin):
    """Text translator using MiniMax API."""

    default_output = "Translation is unavailable."

    def __init__(
        self,
        model: str = MINIMAX_MODEL,
        key: Optional[str] = None,
        output_language: str = "English",
    ) -> None:
        """
        Text translator using MiniMax API.

        Parameters
        ----------
        model : str, optional
            Model to use, by default "MiniMax-M3".
        key : Optional[str], optional
            Estimator-specific API key; if None, retrieved from
            the global config, by default None.
        output_language : str, optional
            Target language, by default "English".
        """
        self._set_keys(key)
        self.model = model
        self.output_language = output_language
