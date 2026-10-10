"""Rubric-based LLM-judge scoring.

The rubric is the whole task definition: the model reads it, reads the sample,
and answers with the one number the sample earns. What comes back is bounded on
both ends — a score outside the scale is clamped into it, since a judge saying
15 out of 10 still means the top of the range — while an answer that is no
number at all is a sample failure rather than a score nobody can distinguish
from a real one.

A System One model rates on discrete levels instead: every whole number of the
scale becomes one, and its probability-weighted level is the score.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from functools import partial
from typing import Any, ClassVar

import numpy as np
from sklearn.base import RegressorMixin

from skllm.backends import BackendError, Message, SystemOneBackend
from skllm.core import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_MAX_CONCURRENCY,
    BasePromptEstimator,
    StatelessMixin,
    as_texts,
)
from skllm.prompts import RUBRIC_QUESTION, RUBRIC_SCORING

__all__ = ["RubricScorer"]

DEFAULT_SCALE = (0, 10)

#: How much of an unusable answer an error message quotes.
_QUOTED_ANSWER_CHARS = 200

#: The id the scorer asks its one System One question under.
_SCORE_ID = "score"

#: The answer shape the judge is constrained to. The scale itself is not a schema
#: constraint — numeric bounds are not part of every provider's structured mode,
#: and clamping covers an out-of-range answer whether they are declared or not.
_SCORE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"score": {"type": "number"}},
    "required": ["score"],
    "additionalProperties": False,
}


class RubricScorer(RegressorMixin, StatelessMixin, BasePromptEstimator):
    """Score each text against a free-text rubric, on a numeric scale.

    ``rubric`` states what earns which score, in whatever detail the task needs;
    ``scale`` is the ``(minimum, maximum)`` pair the answers live between,
    ``(0, 10)`` by default. Scores come back as floats, with ``NaN`` marking the
    samples the model could not be made to answer for.

    A System One model needs a scale of whole numbers with no more steps than
    it has levels, e.g. ``(1, 10)``.
    """

    _system_one_capable: ClassVar[bool] = True

    prompt_placeholders: ClassVar[frozenset[str]] = frozenset(
        {"rubric", "text", "minimum", "maximum"}
    )
    default_prompt_template: ClassVar[str] = RUBRIC_SCORING

    def __init__(
        self,
        model: str = DEFAULT_CHAT_MODEL,
        *,
        rubric: str | None = None,
        scale: tuple[float, float] = DEFAULT_SCALE,
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
        self.rubric = rubric
        self.scale = scale

    def predict(self, X: Any) -> np.ndarray:
        """Score every text, ``NaN`` where the sample failed."""
        self._check_ready()
        texts = as_texts(X)
        system_one = self._system_one()
        if system_one is None:
            prompts = [self._prompt(**self._prompt_values(text)) for text in texts]
            scores = self._run_batch(
                prompts,
                partial(self._score, bounds=self._bounds()),
                fill=lambda: math.nan,
            )
        else:
            scores = self._run_batch(
                texts,
                partial(self._rate, levels=self._levels(system_one)),
                fill=lambda: math.nan,
            )
        return np.asarray(scores, dtype=float)

    def _check_params(self) -> None:
        super()._check_params()
        self._rubric()
        self._bounds()
        system_one = self._system_one()
        if system_one is not None:
            self._levels(system_one)

    def _prompt_values(self, text: str) -> dict[str, Any]:
        minimum, maximum = self._bounds()
        return {
            "rubric": self._rubric(),
            "text": text,
            "minimum": minimum,
            "maximum": maximum,
        }

    def _score(self, prompt: str, *, bounds: tuple[float, float]) -> float:
        answer = self._structured([Message("user", prompt)], _SCORE_SCHEMA)
        score = answer.get("score") if isinstance(answer, Mapping) else None
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            raise BackendError(
                f"The model answered without a numeric 'score': {_quote(answer)}."
            )
        minimum, maximum = bounds
        return min(max(float(score), minimum), maximum)

    def _rate(self, text: str, *, levels: list[float]) -> float:
        """Score a text on a System One model's discrete levels."""
        question = {
            "type": "score",
            "instructions": {"rubric": self._rubric(), "question": RUBRIC_QUESTION},
            "criteria": [f"{level:g}" for level in levels],
        }
        answer = self._evaluate(text, {_SCORE_ID: question}).get(_SCORE_ID)
        level = answer.get("score") if isinstance(answer, Mapping) else None
        if (
            isinstance(level, bool)
            or not isinstance(level, (int, float))
            or not math.isfinite(level)
        ):
            raise BackendError(
                f"The model answered without a numeric 'score': {_quote(answer)}."
            )
        # The answer is the expected level index; levels are one apart.
        return min(max(levels[0] + float(level), levels[0]), levels[-1])

    def _levels(self, backend: SystemOneBackend) -> list[float]:
        """Every whole number of the scale, one System One level each."""
        minimum, maximum = self._bounds()
        steps = maximum - minimum
        if (
            not minimum.is_integer()
            or not maximum.is_integer()
            or steps + 1 > backend.max_score_levels
        ):
            raise ValueError(
                f"{self.model!r} is a System One model, which rates on at most "
                f"{backend.max_score_levels} levels, so the scale must be a pair "
                f"of whole numbers at most {backend.max_score_levels - 1} apart, "
                f"e.g. (1, {backend.max_score_levels}); got {self.scale!r}."
            )
        return [minimum + step for step in range(int(steps) + 1)]

    def _rubric(self) -> str:
        """The validated criteria the judge scores against."""
        rubric = self.rubric
        if not isinstance(rubric, str) or not rubric.strip():
            raise ValueError(
                "rubric must state what the score measures, e.g. 'How clearly the "
                "review explains the problem it reports: 0 not at all, 10 "
                "exhaustively.'"
            )
        return rubric

    def _bounds(self) -> tuple[float, float]:
        """The validated ``(minimum, maximum)`` every score is clamped into."""
        scale = self.scale
        if not isinstance(scale, (tuple, list)) or len(scale) != 2:
            raise TypeError(
                "scale must be a (minimum, maximum) pair of numbers, e.g. (0, 10), "
                f"got {scale!r}."
            )
        minimum, maximum = (_as_bound(value) for value in scale)
        if minimum >= maximum:
            raise ValueError(
                f"The scale minimum {minimum:g} must lie below its maximum {maximum:g}."
            )
        return minimum, maximum


def _as_bound(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            "scale must be a (minimum, maximum) pair of numbers, e.g. (0, 10), but "
            f"one end is a {type(value).__name__}."
        )
    if not math.isfinite(value):
        raise ValueError(f"Every end of the scale must be finite, got {value!r}.")
    return float(value)


def _quote(answer: Any) -> str:
    return repr(str(answer)[:_QUOTED_ANSWER_CHARS])
