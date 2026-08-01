from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pytest
from sklearn.base import clone, is_regressor
from sklearn.pipeline import Pipeline
from sklearn.utils import get_tags

from skllm.core import SampleError, UsageRecord
from skllm.scoring import RubricScorer
from skllm.text2text import Summarizer
from tests.fakes import FakeBackend, FakeCall

FAKE_MODEL = "fake::m"

RUBRIC = "How clearly the review explains the problem it reports."

REVIEWS = [
    "The battery drains from full to empty in three hours of reading.",
    "bad",
]


def score_per_sample(scores: Mapping[str, Any]) -> Callable[[FakeCall], Any]:
    """Answer each sample by the text its prompt quotes, in whatever order they arrive."""

    def answer(call: FakeCall) -> Any:
        return next(value for text, value in scores.items() if text in call.prompt)

    return answer


@pytest.fixture
def scorer(fake_backend: FakeBackend) -> RubricScorer:
    return RubricScorer(model=FAKE_MODEL, rubric=RUBRIC, verbose=0)


# --- scoring -----------------------------------------------------------------


def test_every_text_is_scored_in_one_structured_request(
    fake_backend: FakeBackend, scorer: RubricScorer
) -> None:
    fake_backend.structured_script = score_per_sample(
        {REVIEWS[0]: {"score": 9}, REVIEWS[1]: {"score": 2}}
    )

    scores = scorer.predict(REVIEWS)

    assert [call.kind for call in fake_backend.calls] == ["structured", "structured"]
    assert scores.tolist() == [9.0, 2.0]
    assert scores.dtype == np.dtype(float)


def test_the_answer_is_constrained_to_a_number(
    fake_backend: FakeBackend, scorer: RubricScorer
) -> None:
    """The score is asked for through structured outputs, not parsed out of prose."""
    fake_backend.structured_script = {"score": 7}

    scorer.predict(REVIEWS[:1])

    schema = fake_backend.calls[0].schema
    assert schema is not None
    assert schema["properties"]["score"] == {"type": "number"}
    assert schema["required"] == ["score"]


@pytest.mark.parametrize(
    ("answered", "expected"),
    [
        (7, 7.0),  # inside the scale
        (15, 10.0),  # above it: clamped to the maximum
        (-3, 0.0),  # below it: clamped to the minimum
        (7.5, 7.5),  # fractional scores survive
        (0, 0.0),
        (10, 10.0),
    ],
)
def test_scores_are_clamped_into_the_scale(
    answered: float, expected: float, fake_backend: FakeBackend, scorer: RubricScorer
) -> None:
    fake_backend.structured_script = {"score": answered}

    scores = scorer.set_params(scale=(0, 10)).predict(REVIEWS[:1])

    assert scores.tolist() == [expected]


def test_a_custom_scale_bounds_the_answers(
    fake_backend: FakeBackend, scorer: RubricScorer
) -> None:
    fake_backend.structured_script = score_per_sample(
        {REVIEWS[0]: {"score": 9}, REVIEWS[1]: {"score": -2}}
    )

    scores = scorer.set_params(scale=(1, 5)).predict(REVIEWS)

    assert scores.tolist() == [5.0, 1.0]


def test_the_prompt_states_the_rubric_the_scale_and_the_sample(
    fake_backend: FakeBackend, scorer: RubricScorer
) -> None:
    fake_backend.structured_script = {"score": 7}

    scorer.set_params(scale=(1, 5)).predict(REVIEWS[:1])

    prompt = fake_backend.prompts[0]
    assert RUBRIC in prompt
    assert REVIEWS[0] in prompt
    assert "from 1 to 5" in prompt


# --- failures ----------------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        {"score": "seven"},  # a string, not a number
        {"score": None},
        {"score": True},  # a bool is not a score
        {"verdict": 7},  # the wrong field
        {},
        "seven out of ten",  # nothing salvage could turn into a score
    ],
)
def test_an_answer_that_is_no_number_is_a_sample_failure(
    answer: Any, fake_backend: FakeBackend, scorer: RubricScorer
) -> None:
    fake_backend.structured_script = lambda call: answer

    with pytest.warns(UserWarning, match="1 of 1 samples"):
        scores = scorer.predict(REVIEWS[:1])

    assert np.isnan(scores).tolist() == [True]
    assert [error.index for error in scorer.errors_] == [0]
    assert "numeric 'score'" in scorer.errors_[0].error


def test_a_failed_sample_is_filled_with_nan_and_leaves_the_others_alone(
    fake_backend: FakeBackend, scorer: RubricScorer
) -> None:
    fake_backend.structured_script = score_per_sample(
        {
            REVIEWS[0]: {"score": 9},
            REVIEWS[1]: RuntimeError("the provider rejected the request"),
        }
    )

    with pytest.warns(UserWarning, match="1 of 2 samples"):
        scores = scorer.predict(REVIEWS)

    assert scores[0] == 9.0
    assert np.isnan(scores[1])
    assert scorer.errors_ == (
        SampleError(index=1, error="RuntimeError: the provider rejected the request"),
    )
    assert scorer.usage_.last.failures == 1
    assert scorer.usage_.last.requests == 1


def test_the_raise_policy_aborts_the_batch(
    fake_backend: FakeBackend, scorer: RubricScorer
) -> None:
    fake_backend.structured_script = RuntimeError("the provider rejected the request")
    scorer.set_params(on_error="raise")

    with pytest.raises(RuntimeError, match="the provider rejected the request"):
        scorer.predict(REVIEWS[:1])


# --- parameters --------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "   "])
def test_a_rubric_is_required(value: Any, fake_backend: FakeBackend) -> None:
    with pytest.raises(ValueError, match="rubric"):
        RubricScorer(model=FAKE_MODEL, rubric=value).predict(REVIEWS[:1])


@pytest.mark.parametrize("value", [(0,), (0, 5, 10), 10, "0-10", None])
def test_a_scale_that_is_no_pair_is_rejected(
    value: Any, fake_backend: FakeBackend
) -> None:
    with pytest.raises(TypeError, match="scale"):
        RubricScorer(model=FAKE_MODEL, rubric=RUBRIC, scale=value).predict(REVIEWS[:1])


@pytest.mark.parametrize("value", [("0", "10"), (0, True), (None, 10)])
def test_scale_ends_that_are_no_numbers_are_rejected(
    value: Any, fake_backend: FakeBackend
) -> None:
    with pytest.raises(TypeError, match="scale"):
        RubricScorer(model=FAKE_MODEL, rubric=RUBRIC, scale=value).predict(REVIEWS[:1])


@pytest.mark.parametrize("value", [(10, 0), (5, 5), (0, float("nan"))])
def test_a_scale_that_bounds_nothing_is_rejected(
    value: Any, fake_backend: FakeBackend
) -> None:
    with pytest.raises(ValueError, match="scale"):
        RubricScorer(model=FAKE_MODEL, rubric=RUBRIC, scale=value).predict(REVIEWS[:1])


# --- prompts and sklearn compliance ------------------------------------------


def test_the_default_template_matches_its_documented_placeholders() -> None:
    """A default template is exactly what an override is held to."""
    RubricScorer._check_prompt_template(RubricScorer.default_prompt_template)


def test_an_invalid_prompt_template_names_the_placeholders(
    fake_backend: FakeBackend,
) -> None:
    with pytest.raises(ValueError) as excinfo:
        RubricScorer(
            model=FAKE_MODEL,
            rubric=RUBRIC,
            prompt_template="Score {text} against {rubric} up to {top}",
        ).predict(REVIEWS[:1])

    message = str(excinfo.value)
    assert "missing ['maximum', 'minimum']" in message
    assert "unknown ['top']" in message


def test_the_scorer_is_a_stateless_regressor(fake_backend: FakeBackend) -> None:
    fake_backend.structured_script = {"score": 7}
    scorer = RubricScorer(model=FAKE_MODEL, rubric=RUBRIC, verbose=0)

    assert is_regressor(scorer)
    assert not get_tags(scorer).requires_fit
    assert scorer.fit(REVIEWS) is scorer
    assert fake_backend.calls == []  # fit only validates params

    assert scorer.predict(REVIEWS[:1]).tolist() == [7.0]


def test_params_survive_a_clone() -> None:
    scorer = RubricScorer(
        model=FAKE_MODEL, rubric=RUBRIC, scale=(1, 5), api_key="sk-secret"
    )

    cloned = clone(scorer)

    assert cloned.get_params() == scorer.get_params()
    assert cloned.scale == (1, 5)
    assert cloned.api_key == "sk-secret"
    assert vars(scorer) == scorer.get_params()


def test_an_empty_batch_issues_no_requests(
    fake_backend: FakeBackend, scorer: RubricScorer
) -> None:
    fake_backend.structured_script = {"score": 7}
    scorer.predict(REVIEWS[:1])
    before = scorer.usage_.total
    fake_backend.calls.clear()

    scores = scorer.predict([])

    assert scores.shape == (0,)
    assert fake_backend.calls == []
    assert scorer.usage_.last == UsageRecord()
    assert scorer.usage_.total == before


def test_the_scorer_terminates_a_pipeline(fake_backend: FakeBackend) -> None:
    fake_backend.chat_script = "the battery dies within hours"
    fake_backend.structured_script = {"score": 4}
    pipeline = Pipeline(
        [
            ("summarize", Summarizer(model=FAKE_MODEL, verbose=0)),
            ("score", RubricScorer(model=FAKE_MODEL, rubric=RUBRIC, verbose=0)),
        ]
    )

    scores = pipeline.fit(REVIEWS).predict(REVIEWS)

    assert scores.tolist() == [4.0, 4.0]
    # The summary, not the review, is what was scored.
    assert "the battery dies within hours" in fake_backend.prompts[-1]
