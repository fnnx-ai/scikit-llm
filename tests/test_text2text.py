from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.utils import get_tags

from skllm.core import SampleError, UsageRecord
from skllm.text2text import Summarizer, Translator
from tests.fakes import FakeBackend, FakeCall

FAKE_MODEL = "fake::m"

REVIEWS = [
    "The battery lasts two days and the screen is bright.",
    "It broke on day one and support never answered.",
    "Setup took ten minutes, everything worked afterwards.",
]


def answer_per_sample(answers: Mapping[str, Any]) -> Callable[[FakeCall], Any]:
    """Answer each sample by the text its prompt quotes, whatever order they arrive in."""

    def answer(call: FakeCall) -> Any:
        return next(value for text, value in answers.items() if text in call.prompt)

    return answer


@pytest.fixture
def summarizer(fake_backend: FakeBackend) -> Summarizer:
    return Summarizer(model=FAKE_MODEL, verbose=0)


@pytest.fixture
def translator(fake_backend: FakeBackend) -> Translator:
    return Translator(model=FAKE_MODEL, output_language="German", verbose=0)


# --- summarization -----------------------------------------------------------


def test_every_text_is_summarized_in_one_request(
    fake_backend: FakeBackend, summarizer: Summarizer
) -> None:
    fake_backend.chat_script = answer_per_sample(
        {REVIEWS[0]: "long battery life", REVIEWS[1]: "broke immediately"}
    )

    summaries = summarizer.transform(REVIEWS[:2])

    assert [call.kind for call in fake_backend.calls] == ["chat", "chat"]
    assert list(summaries) == ["long battery life", "broke immediately"]


def test_the_prompt_states_the_word_budget_and_quotes_the_sample(
    fake_backend: FakeBackend, summarizer: Summarizer
) -> None:
    summarizer.set_params(max_words=7).transform(REVIEWS[:1])

    prompt = fake_backend.prompts[0]
    assert "at most 7 words" in prompt
    assert REVIEWS[0] in prompt


def test_a_focus_steers_the_prompt_only_when_one_is_given(
    fake_backend: FakeBackend, summarizer: Summarizer
) -> None:
    summarizer.transform(REVIEWS[:1])
    assert "Concentrate on" not in fake_backend.prompts[0]

    summarizer.set_params(focus="battery life").transform(REVIEWS[:1])
    assert "what the text says about battery life" in fake_backend.prompts[1]


def test_surrounding_whitespace_is_trimmed_from_the_answer(
    fake_backend: FakeBackend, summarizer: Summarizer
) -> None:
    fake_backend.chat_script = "\n  long battery life\n"

    assert list(summarizer.transform(REVIEWS[:1])) == ["long battery life"]


@pytest.mark.parametrize("value", [0, -1, 1.5, None])
def test_invalid_max_words_are_rejected(value: Any, fake_backend: FakeBackend) -> None:
    with pytest.raises(ValueError, match="max_words"):
        Summarizer(model=FAKE_MODEL, max_words=value).transform(REVIEWS[:1])


def test_a_focus_must_be_a_string(fake_backend: FakeBackend) -> None:
    with pytest.raises(TypeError, match="focus"):
        Summarizer(model=FAKE_MODEL, focus=["battery life"]).transform(REVIEWS[:1])  # type: ignore[arg-type]


# --- translation -------------------------------------------------------------


def test_one_request_carries_the_output_language_and_the_sample(
    fake_backend: FakeBackend, translator: Translator
) -> None:
    fake_backend.chat_script = "Guten Morgen"

    translations = translator.transform(["Good morning"])

    assert len(fake_backend.calls) == 1
    assert "German" in fake_backend.prompts[0]
    assert "Good morning" in fake_backend.prompts[0]
    assert translations.shape == (1,)
    assert list(translations) == ["Guten Morgen"]


@pytest.mark.parametrize("value", ["", "   "])
def test_an_output_language_naming_nothing_is_rejected(
    value: str, fake_backend: FakeBackend
) -> None:
    with pytest.raises(ValueError, match="output_language"):
        Translator(model=FAKE_MODEL, output_language=value).transform(["Good morning"])


def test_the_output_language_must_be_a_string(fake_backend: FakeBackend) -> None:
    with pytest.raises(TypeError, match="output_language"):
        Translator(model=FAKE_MODEL, output_language=None).transform(["Good morning"])  # type: ignore[arg-type]


# --- failures ----------------------------------------------------------------


def test_a_failed_sample_is_filled_with_an_empty_string(
    fake_backend: FakeBackend, summarizer: Summarizer
) -> None:
    fake_backend.chat_script = answer_per_sample(
        {
            REVIEWS[0]: "long battery life",
            REVIEWS[1]: RuntimeError("the provider rejected the request"),
            REVIEWS[2]: "quick setup",
        }
    )

    with pytest.warns(UserWarning, match="1 of 3 samples"):
        summaries = summarizer.transform(REVIEWS)

    assert list(summaries) == ["long battery life", "", "quick setup"]
    assert summarizer.errors_ == (
        SampleError(index=1, error="RuntimeError: the provider rejected the request"),
    )
    assert summarizer.usage_.last.failures == 1
    assert summarizer.usage_.last.requests == 2


def test_the_raise_policy_aborts_the_batch(
    fake_backend: FakeBackend, translator: Translator
) -> None:
    fake_backend.chat_script = RuntimeError("the provider rejected the request")
    translator.set_params(on_error="raise")

    with pytest.raises(RuntimeError, match="the provider rejected the request"):
        translator.transform(["Good morning"])


# --- prompts and sklearn compliance ------------------------------------------


@pytest.mark.parametrize("transformer", [Summarizer, Translator])
def test_default_templates_match_their_documented_placeholders(
    transformer: type[Any],
) -> None:
    """A default template is exactly what an override is held to."""
    transformer._check_prompt_template(transformer.default_prompt_template)


def test_an_invalid_prompt_template_names_the_placeholders(
    fake_backend: FakeBackend,
) -> None:
    with pytest.raises(ValueError) as excinfo:
        Summarizer(
            model=FAKE_MODEL, prompt_template="Summarize {text} in {words} words"
        ).transform(REVIEWS[:1])

    message = str(excinfo.value)
    assert "missing ['focus', 'max_words']" in message
    assert "unknown ['words']" in message


@pytest.mark.parametrize("transformer", [Summarizer, Translator])
def test_the_transformers_have_no_predict_alias(transformer: type[Any]) -> None:
    """v1 aliased predict to transform on transformers; v2 does not."""
    assert not hasattr(transformer, "predict")


@pytest.mark.parametrize("transformer", [Summarizer, Translator])
def test_the_transformers_are_stateless(
    transformer: type[Any], fake_backend: FakeBackend
) -> None:
    estimator = transformer(model=FAKE_MODEL, verbose=0)

    assert not get_tags(estimator).requires_fit
    assert estimator.fit(REVIEWS[:1]) is estimator
    assert fake_backend.calls == []  # fit only validates params

    assert list(estimator.transform(REVIEWS[:1])) == ["fake summary"]


@pytest.mark.parametrize("transformer", [Summarizer, Translator])
def test_params_survive_a_clone(transformer: type[Any]) -> None:
    estimator = transformer(model=FAKE_MODEL, api_key="sk-secret", alias="team_a")

    cloned = clone(estimator)

    assert cloned.get_params() == estimator.get_params()
    assert cloned.api_key == "sk-secret"
    assert vars(estimator) == estimator.get_params()


def test_an_empty_batch_issues_no_requests(
    fake_backend: FakeBackend, summarizer: Summarizer
) -> None:
    summarizer.transform(REVIEWS[:1])
    before = summarizer.usage_.total
    fake_backend.calls.clear()

    summaries = summarizer.transform([])

    assert summaries.shape == (0,)
    assert fake_backend.calls == []
    assert summarizer.usage_.last == UsageRecord()
    assert summarizer.usage_.total == before


def test_the_transformers_chain_in_a_pipeline(fake_backend: FakeBackend) -> None:
    fake_backend.chat_script = lambda call: (
        "Kurzfassung" if "Translate" in call.prompt else "short summary"
    )
    pipeline = Pipeline(
        [
            ("summarize", Summarizer(model=FAKE_MODEL, verbose=0)),
            ("translate", Translator(model=FAKE_MODEL, verbose=0)),
        ]
    )

    translations = pipeline.fit_transform(REVIEWS[:1])

    assert list(translations) == ["Kurzfassung"]
    assert "short summary" in fake_backend.prompts[1]  # the summary was translated
