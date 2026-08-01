"""Summarization and translation transformers.

Both answer with free text rather than a constrained value, so there is nothing
to validate an answer against: a sample fails only when its request does, and
fills with the empty string.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
from sklearn.base import TransformerMixin

from skllm.backends import Message
from skllm.core import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_MAX_CONCURRENCY,
    BasePromptEstimator,
    StatelessMixin,
    as_texts,
    check_count,
)
from skllm.prompts import SUMMARY, TRANSLATION, render_focus

__all__ = ["Summarizer", "Translator"]

DEFAULT_MAX_WORDS = 15
DEFAULT_OUTPUT_LANGUAGE = "English"


class _BaseText2Text(TransformerMixin, StatelessMixin, BasePromptEstimator):
    """One chat answer per text, in input order, empty where none arrived."""

    def transform(self, X: Any) -> np.ndarray:
        """Answer every text, returning one string per sample."""
        self._check_ready()
        prompts = [self._prompt(**self._prompt_values(text)) for text in as_texts(X)]
        answers = self._run_batch(prompts, self._answer, fill=str)
        return np.asarray(answers, dtype=object)

    def _answer(self, prompt: str) -> str:
        # Free-text answers arrive padded with newlines often enough that the
        # padding, not the text, would decide equality for whatever reads them.
        return self._chat([Message("user", prompt)]).strip()

    def _prompt_values(self, text: str) -> dict[str, Any]:
        raise NotImplementedError


class Summarizer(_BaseText2Text):
    """Summarize each text in at most ``max_words`` words.

    A ``focus`` steers the summary towards one concept where the text covers it,
    and is ignored where it does not.
    """

    prompt_placeholders: ClassVar[frozenset[str]] = frozenset(
        {"text", "max_words", "focus"}
    )
    default_prompt_template: ClassVar[str] = SUMMARY

    def __init__(
        self,
        model: str = DEFAULT_CHAT_MODEL,
        *,
        max_words: int = DEFAULT_MAX_WORDS,
        focus: str | None = None,
        prompt_template: str | None = None,
        model_params: Mapping[str, Any] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        alias: str | None = None,
        on_error: str = "fill",
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        verbose: int = 1,
    ) -> None:
        super().__init__(
            model=model,
            prompt_template=prompt_template,
            model_params=model_params,
            api_key=api_key,
            base_url=base_url,
            alias=alias,
            on_error=on_error,
            max_concurrency=max_concurrency,
            verbose=verbose,
        )
        self.max_words = max_words
        self.focus = focus

    def _check_params(self) -> None:
        super()._check_params()
        check_count("max_words", self.max_words, required=True)
        if self.focus is not None and not isinstance(self.focus, str):
            raise TypeError(
                f"focus must be a string or None, got {type(self.focus).__name__}."
            )

    def _prompt_values(self, text: str) -> dict[str, Any]:
        return {
            "text": text,
            "max_words": self.max_words,
            "focus": render_focus(self.focus),
        }


class Translator(_BaseText2Text):
    """Translate each text into ``output_language``, leaving it be if already there."""

    prompt_placeholders: ClassVar[frozenset[str]] = frozenset(
        {"text", "output_language"}
    )
    default_prompt_template: ClassVar[str] = TRANSLATION

    def __init__(
        self,
        model: str = DEFAULT_CHAT_MODEL,
        *,
        output_language: str = DEFAULT_OUTPUT_LANGUAGE,
        prompt_template: str | None = None,
        model_params: Mapping[str, Any] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        alias: str | None = None,
        on_error: str = "fill",
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        verbose: int = 1,
    ) -> None:
        super().__init__(
            model=model,
            prompt_template=prompt_template,
            model_params=model_params,
            api_key=api_key,
            base_url=base_url,
            alias=alias,
            on_error=on_error,
            max_concurrency=max_concurrency,
            verbose=verbose,
        )
        self.output_language = output_language

    def _check_params(self) -> None:
        super()._check_params()
        if not isinstance(self.output_language, str):
            raise TypeError(
                "output_language must be a string, got "
                f"{type(self.output_language).__name__}."
            )
        if not self.output_language.strip():
            raise ValueError("output_language must name a language, e.g. 'German'.")

    def _prompt_values(self, text: str) -> dict[str, Any]:
        return {"text": text, "output_language": self.output_language}
