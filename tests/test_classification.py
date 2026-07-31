from __future__ import annotations

import math
import pickle
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils import get_tags

from skllm.backends import parse_model
from skllm.classification import (
    DynamicFewShotClassifier,
    FewShotClassifier,
    MultiLabelFewShotClassifier,
    MultiLabelZeroShotClassifier,
    ZeroShotClassifier,
)
from skllm.config import config_context
from skllm.core import DEFAULT_CHAT_MODEL, Requirement, SampleError
from skllm.vectorization import Vectorizer
from tests.fakes import FakeBackend, FakeCall

FAKE_MODEL = "fake::m"

REVIEWS = ["I love it", "It broke on day one"]
SENTIMENTS = ["positive", "negative"]


def label(answer: str) -> dict[str, str]:
    return {"label": answer}


def labels(*answers: str) -> dict[str, list[str]]:
    return {"labels": list(answers)}


@pytest.fixture(autouse=True)
def use_the_fake_backend(fake_backend: FakeBackend) -> None:
    """Every classifier here answers through ``fake::m``, scripted per test."""


@pytest.fixture
def zero_shot(fake_backend: FakeBackend) -> ZeroShotClassifier:
    fake_backend.structured_script = label("positive")
    return ZeroShotClassifier(model=FAKE_MODEL, verbose=0).fit(None, SENTIMENTS)


@pytest.fixture
def multi_label(fake_backend: FakeBackend) -> MultiLabelZeroShotClassifier:
    return MultiLabelZeroShotClassifier(model=FAKE_MODEL, verbose=0, max_labels=3).fit(
        None, [["quality", "price"], ["delivery"]]
    )


# --- fitting -----------------------------------------------------------------


def test_zero_shot_derives_sorted_classes_and_predicts_a_label(
    fake_backend: FakeBackend,
) -> None:
    fake_backend.structured_script = label("positive")
    classifier = ZeroShotClassifier(model=FAKE_MODEL, verbose=0)

    classifier.fit(None, SENTIMENTS)
    predictions = classifier.predict(["I love it"])

    assert classifier.classes_ == ["negative", "positive"]
    assert len(fake_backend.calls) == 1
    assert fake_backend.calls[0].kind == "structured"
    assert list(predictions) == ["positive"]


def test_a_bare_model_string_routes_to_the_openai_backend(
    fake_provider: Callable[[str], FakeBackend],
) -> None:
    backend = fake_provider("openai")
    backend.structured_script = label("positive")

    ZeroShotClassifier(model="gpt-5", verbose=0).fit(None, SENTIMENTS).predict(
        ["I love it"]
    )

    assert backend.calls[-1].model == "gpt-5"


def test_a_classifier_built_without_a_model_asks_openai_for_the_default(
    fake_provider: Callable[[str], FakeBackend],
) -> None:
    backend = fake_provider("openai")
    backend.structured_script = label("positive")

    ZeroShotClassifier(verbose=0).fit(None, SENTIMENTS).predict(["I love it"])

    assert backend.calls[-1].model == parse_model(DEFAULT_CHAT_MODEL).model_id


def test_the_label_schema_constrains_the_answer_to_the_classes(
    zero_shot: ZeroShotClassifier, fake_backend: FakeBackend
) -> None:
    zero_shot.predict(["I love it"])

    schema = fake_backend.calls[0].schema or {}
    assert schema["properties"]["label"]["enum"] == ["negative", "positive"]
    assert schema["required"] == ["label"]
    assert schema["additionalProperties"] is False


def test_classes_come_from_y_not_from_the_training_texts(
    fake_backend: FakeBackend,
) -> None:
    classifier = FewShotClassifier(model=FAKE_MODEL, verbose=0).fit(
        REVIEWS, ["positive", "negative"]
    )

    assert classifier.classes_ == ["negative", "positive"]
    assert classifier.examples_ == (
        ("I love it", "positive"),
        ("It broke on day one", "negative"),
    )


def test_numpy_labels_become_plain_strings(fake_backend: FakeBackend) -> None:
    classifier = ZeroShotClassifier(model=FAKE_MODEL, verbose=0).fit(
        None, np.array(SENTIMENTS)
    )

    assert classifier.classes_ == ["negative", "positive"]
    assert all(type(label) is str for label in classifier.classes_)


@pytest.mark.parametrize(
    "classifier", [ZeroShotClassifier, MultiLabelZeroShotClassifier]
)
def test_non_string_labels_are_rejected(classifier: type[Any]) -> None:
    y = [0, 1] if classifier is ZeroShotClassifier else [[0], [1]]

    with pytest.raises(TypeError, match="labels must be strings"):
        classifier(model=FAKE_MODEL).fit(None, y)


def test_multi_label_targets_must_be_collections() -> None:
    with pytest.raises(TypeError, match="collections of labels"):
        MultiLabelZeroShotClassifier(model=FAKE_MODEL).fit(None, ["quality", "price"])


def test_fitting_without_labels_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ZeroShotClassifier(model=FAKE_MODEL).fit(None, [])


def test_few_shot_requires_as_many_texts_as_labels() -> None:
    with pytest.raises(ValueError, match="few-shot"):
        FewShotClassifier(model=FAKE_MODEL).fit(["only one text"], SENTIMENTS)


@pytest.mark.parametrize(
    ("X", "y"), [(["only one text"], SENTIMENTS), ([], [])], ids=["mismatch", "empty"]
)
def test_a_failed_fit_leaves_the_estimator_unfitted(X: Any, y: Any) -> None:
    classifier = FewShotClassifier(model=FAKE_MODEL)

    with pytest.raises(ValueError):
        classifier.fit(X, y)

    with pytest.raises(NotFittedError):
        classifier.predict(["I love it"])


# --- prompts -----------------------------------------------------------------


def test_the_prompt_lists_every_class_and_the_sample(
    zero_shot: ZeroShotClassifier, fake_backend: FakeBackend
) -> None:
    zero_shot.predict(["I love it"])

    prompt = fake_backend.prompts[0]
    assert "- negative\n- positive" in prompt
    assert "I love it" in prompt


def test_few_shot_examples_reach_the_prompt(fake_backend: FakeBackend) -> None:
    fake_backend.structured_script = label("positive")
    texts = ["I love it", "It broke on day one", "Fine, I guess"]
    classifier = FewShotClassifier(model=FAKE_MODEL, verbose=0).fit(
        texts, ["positive", "negative", "neutral"]
    )

    classifier.predict(["Works great"])

    prompt = fake_backend.prompts[0]
    for text, sentiment in zip(texts, ["positive", "negative", "neutral"], strict=True):
        assert text in prompt
        assert f"Categories: {sentiment}" in prompt
    assert "Works great" in prompt


def test_multi_label_prompts_state_the_cap(
    multi_label: Any, fake_backend: FakeBackend
) -> None:
    fake_backend.structured_script = labels("delivery", "price")

    multi_label.predict(["Fast delivery, high price"])
    assert "at most 3" in fake_backend.prompts[0]

    multi_label.set_params(max_labels=None).predict(["Fast delivery, high price"])
    assert "at most 3" in fake_backend.prompts[1]  # unset: the number of classes


@pytest.mark.parametrize(
    "classifier",
    [
        ZeroShotClassifier,
        FewShotClassifier,
        DynamicFewShotClassifier,
        MultiLabelZeroShotClassifier,
        MultiLabelFewShotClassifier,
    ],
)
def test_default_templates_match_their_documented_placeholders(
    classifier: type[Any],
) -> None:
    """A default template is exactly what an override is held to."""
    classifier._check_prompt_template(classifier.default_prompt_template)


def test_an_invalid_prompt_template_names_the_placeholders() -> None:
    with pytest.raises(ValueError) as excinfo:
        ZeroShotClassifier(
            model=FAKE_MODEL, prompt_template="Classify {text} as {tone}"
        ).fit(None, SENTIMENTS)

    message = str(excinfo.value)
    assert "missing ['labels']" in message
    assert "unknown ['tone']" in message
    assert "must use exactly ['labels', 'text']" in message


# --- failures ----------------------------------------------------------------


def test_a_label_outside_the_classes_is_a_sample_failure(
    fake_backend: FakeBackend,
) -> None:
    fake_backend.structured_script = [label("positive"), label("maybe")]
    classifier = ZeroShotClassifier(model=FAKE_MODEL, verbose=0, max_concurrency=1).fit(
        None, SENTIMENTS
    )

    with pytest.warns(UserWarning, match="1 of 2 samples"):
        predictions = classifier.predict(REVIEWS)

    assert list(predictions) == ["positive", None]
    assert classifier.errors_ == (
        SampleError(
            index=1,
            error="BackendError: The model answered 'maybe', which is not one of "
            "the fitted classes ['negative', 'positive'].",
        ),
    )
    assert classifier.usage_.last.failures == 1


def test_a_failed_sample_takes_the_default_label(fake_backend: FakeBackend) -> None:
    fake_backend.structured_script = RuntimeError("the provider rejected the request")
    classifier = ZeroShotClassifier(
        model=FAKE_MODEL, verbose=0, default_label="unknown"
    ).fit(None, SENTIMENTS)

    with pytest.warns(UserWarning):
        predictions = classifier.predict(["I love it"])

    assert list(predictions) == ["unknown"]


def test_a_malformed_answer_is_a_sample_failure(fake_backend: FakeBackend) -> None:
    fake_backend.structured_script = {"category": "positive"}
    classifier = ZeroShotClassifier(model=FAKE_MODEL, verbose=0).fit(None, SENTIMENTS)

    with pytest.warns(UserWarning):
        predictions = classifier.predict(["I love it"])

    assert list(predictions) == [None]
    assert "without a 'label' string" in classifier.errors_[0].error


def test_default_label_must_be_a_string() -> None:
    with pytest.raises(TypeError, match="default_label"):
        ZeroShotClassifier(model=FAKE_MODEL, default_label=1).fit(None, SENTIMENTS)  # type: ignore[arg-type]


def test_the_raise_policy_aborts_the_batch(fake_backend: FakeBackend) -> None:
    fake_backend.structured_script = RuntimeError("the provider rejected the request")
    classifier = ZeroShotClassifier(model=FAKE_MODEL, verbose=0, on_error="raise").fit(
        None, SENTIMENTS
    )

    with pytest.raises(RuntimeError, match="the provider rejected the request"):
        classifier.predict(REVIEWS)


# --- multi-label -------------------------------------------------------------


def test_multi_label_predictions_are_ragged_lists(
    multi_label: Any, fake_backend: FakeBackend
) -> None:
    fake_backend.structured_script = [labels("quality", "price"), labels("delivery")]

    predictions = multi_label.predict(["Great but pricey", "Arrived late"])

    assert predictions == [["quality", "price"], ["delivery"]]
    assert multi_label.classes_ == ["delivery", "price", "quality"]


def test_multi_label_truncates_to_max_labels(
    multi_label: Any, fake_backend: FakeBackend
) -> None:
    """Truncation is a backstop, not a failure: the first max_labels survive."""
    multi_label.set_params(max_labels=2)
    fake_backend.structured_script = labels("quality", "price", "delivery")

    predictions = multi_label.predict(["Great but pricey and late"])

    assert predictions == [["quality", "price"]]
    assert multi_label.errors_ == ()


def test_multi_label_answers_are_deduplicated(
    multi_label: Any, fake_backend: FakeBackend
) -> None:
    fake_backend.structured_script = labels("price", "price", "quality")

    assert multi_label.predict(["Pricey"]) == [["price", "quality"]]


def test_an_unknown_multi_label_answer_is_a_sample_failure(
    multi_label: Any, fake_backend: FakeBackend
) -> None:
    fake_backend.structured_script = labels("quality", "vibes")

    with pytest.warns(UserWarning, match="1 of 1 samples"):
        predictions = multi_label.predict(["Great but odd"])

    assert predictions == [[]]
    assert "vibes" in multi_label.errors_[0].error


def test_the_multi_label_schema_is_an_array_over_the_classes(
    multi_label: Any, fake_backend: FakeBackend
) -> None:
    fake_backend.structured_script = labels("quality")

    multi_label.predict(["Great"])

    schema = fake_backend.calls[0].schema or {}
    array = schema["properties"]["labels"]
    assert array["type"] == "array"
    assert array["items"]["enum"] == ["delivery", "price", "quality"]
    assert "maxItems" not in array  # the fake backend declares no schema constraints


def test_the_cap_becomes_a_schema_constraint_where_the_provider_takes_one(
    multi_label: Any, fake_backend: FakeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        type(fake_backend), "schema_constraints", frozenset({"maxItems"})
    )
    fake_backend.structured_script = labels("quality")

    multi_label.predict(["Great"])

    schema = fake_backend.calls[0].schema or {}
    assert schema["properties"]["labels"]["maxItems"] == 3


def test_multi_label_few_shot_examples_render_every_label(
    fake_backend: FakeBackend,
) -> None:
    fake_backend.structured_script = labels("quality")
    classifier = MultiLabelFewShotClassifier(model=FAKE_MODEL, verbose=0).fit(
        ["Great but pricey"], [["quality", "price"]]
    )

    classifier.predict(["Solid"])

    assert "Categories: quality, price" in fake_backend.prompts[0]


# --- sampled voting ----------------------------------------------------------


def test_votes_are_aggregated_by_majority(fake_backend: FakeBackend) -> None:
    fake_backend.structured_script = [label("A")] * 3 + [label("B")] * 2
    classifier = ZeroShotClassifier(
        model=FAKE_MODEL, verbose=0, n_predictors=5, max_concurrency=1
    ).fit(None, ["A", "B"])

    predictions = classifier.predict(["a text"])

    assert len(fake_backend.calls) == 5
    assert list(predictions) == ["A"]
    assert all("temperature" not in call.params for call in fake_backend.calls)


def test_predict_proba_returns_the_vote_fractions(fake_backend: FakeBackend) -> None:
    fake_backend.structured_script = [label("A")] * 3 + [label("B")] * 2
    classifier = ZeroShotClassifier(
        model=FAKE_MODEL, verbose=0, n_predictors=5, max_concurrency=1
    ).fit(None, ["A", "B"])

    probabilities = classifier.predict_proba(["a text"])

    assert probabilities.tolist() == [[0.6, 0.4]]
    assert len(fake_backend.calls) == 5


def test_predict_proba_is_one_hot_without_n_predictors(
    fake_backend: FakeBackend,
) -> None:
    fake_backend.structured_script = label("A")
    classifier = ZeroShotClassifier(model=FAKE_MODEL, verbose=0).fit(None, ["A", "B"])

    probabilities = classifier.predict_proba(["a text"])

    assert probabilities.tolist() == [[1.0, 0.0]]
    assert len(fake_backend.calls) == 1
    assert fake_backend.calls[0].params == {"temperature": 0}


def test_a_single_vote_keeps_the_determinism_default(
    fake_backend: FakeBackend,
) -> None:
    fake_backend.structured_script = label("A")
    classifier = ZeroShotClassifier(model=FAKE_MODEL, verbose=0, n_predictors=1).fit(
        None, ["A", "B"]
    )

    classifier.predict(["a text"])

    assert fake_backend.calls[0].params == {"temperature": 0}


def test_vote_ties_are_broken_by_class_order(fake_backend: FakeBackend) -> None:
    fake_backend.structured_script = [label("B"), label("A")]
    classifier = ZeroShotClassifier(
        model=FAKE_MODEL, verbose=0, n_predictors=2, max_concurrency=1
    ).fit(None, ["A", "B"])

    assert list(classifier.predict(["a text"])) == ["A"]


def test_failed_votes_are_tolerated(fake_backend: FakeBackend) -> None:
    fake_backend.structured_script = [RuntimeError("boom"), label("B"), label("B")]
    classifier = ZeroShotClassifier(
        model=FAKE_MODEL, verbose=0, n_predictors=3, max_concurrency=1
    ).fit(None, ["A", "B"])

    predictions = classifier.predict(["a text"])

    assert list(predictions) == ["B"]
    assert classifier.errors_ == ()
    assert classifier.usage_.last.failures == 0


def test_a_sample_fails_only_when_every_vote_fails(fake_backend: FakeBackend) -> None:
    fake_backend.structured_script = RuntimeError("boom")
    classifier = ZeroShotClassifier(
        model=FAKE_MODEL, verbose=0, n_predictors=3, default_label="unknown"
    ).fit(None, ["A", "B"])

    with pytest.warns(UserWarning, match="1 of 1 samples"):
        predictions = classifier.predict(["a text"])

    assert list(predictions) == ["unknown"]
    assert classifier.errors_[0].index == 0
    assert "all 3 attempts failed" in classifier.errors_[0].error


def test_the_raise_policy_tolerates_a_losing_vote(fake_backend: FakeBackend) -> None:
    """The raise policy aborts on failed samples, and a voted sample needs one vote."""
    fake_backend.structured_script = [RuntimeError("boom"), label("B"), label("B")]
    classifier = ZeroShotClassifier(
        model=FAKE_MODEL,
        verbose=0,
        n_predictors=3,
        max_concurrency=1,
        on_error="raise",
    ).fit(None, ["A", "B"])

    assert list(classifier.predict(["a text"])) == ["B"]


def test_the_raise_policy_aborts_when_every_vote_fails(
    fake_backend: FakeBackend,
) -> None:
    fake_backend.structured_script = RuntimeError("boom")
    classifier = ZeroShotClassifier(
        model=FAKE_MODEL, verbose=0, n_predictors=3, on_error="raise"
    ).fit(None, ["A", "B"])

    with pytest.raises(RuntimeError, match="boom"):
        classifier.predict(["a text"])


def test_multi_label_votes_need_a_strict_majority(fake_backend: FakeBackend) -> None:
    votes = [labels("price", "quality"), labels("price", "delivery"), labels("price")]
    fake_backend.structured_script = votes + votes
    classifier = MultiLabelZeroShotClassifier(
        model=FAKE_MODEL, verbose=0, n_predictors=3, max_concurrency=1
    ).fit(None, [["price", "quality"], ["delivery"]])

    predictions = classifier.predict(["a text"])
    probabilities = classifier.predict_proba(["a text"])

    assert predictions == [["price"]]  # only price is on more than half the votes
    assert probabilities[0].tolist() == pytest.approx([1 / 3, 1.0, 1 / 3])


def test_multi_label_probabilities_are_memberships_without_voting(
    multi_label: Any, fake_backend: FakeBackend
) -> None:
    fake_backend.structured_script = labels("quality", "delivery")

    probabilities = multi_label.predict_proba(["Great, arrived fast"])

    assert probabilities.tolist() == [[1.0, 0.0, 1.0]]  # delivery, price, quality
    assert len(fake_backend.calls) == 1


def test_multi_label_probabilities_are_per_class_frequencies(
    fake_backend: FakeBackend,
) -> None:
    fake_backend.structured_script = [labels("price", "quality"), labels("price")]
    classifier = MultiLabelZeroShotClassifier(
        model=FAKE_MODEL, verbose=0, n_predictors=2, max_concurrency=1
    ).fit(None, [["price", "quality"], ["delivery"]])

    probabilities = classifier.predict_proba(["a text"])

    assert probabilities.tolist() == [[0.0, 1.0, 0.5]]  # delivery, price, quality


def test_sampled_voting_rejects_a_pinned_temperature_of_zero() -> None:
    with pytest.raises(ValueError, match="n_predictors"):
        ZeroShotClassifier(
            model=FAKE_MODEL, n_predictors=5, model_params={"temperature": 0}
        ).fit(None, SENTIMENTS)


def test_an_override_cannot_pin_the_temperature_of_a_voting_classifier(
    fake_backend: FakeBackend,
) -> None:
    """The guard is on the effective temperature, so it must hold at call time."""
    fake_backend.structured_script = label("positive")
    classifier = ZeroShotClassifier(
        model=FAKE_MODEL, verbose=0, alias="careful", n_predictors=3
    ).fit(None, SENTIMENTS)

    with config_context(credentials={"careful": "sk-careful"}):
        classifier.predict(["I love it"])

        with (
            config_context(overrides={"careful": {"temperature": 0}}),
            pytest.raises(ValueError, match="n_predictors"),
        ):
            classifier.predict(["I love it"])


def test_a_pinned_temperature_of_zero_is_allowed_without_voting(
    fake_backend: FakeBackend,
) -> None:
    fake_backend.structured_script = label("positive")
    classifier = ZeroShotClassifier(
        model=FAKE_MODEL, verbose=0, model_params={"temperature": 0}
    ).fit(None, SENTIMENTS)

    assert list(classifier.predict(["I love it"])) == ["positive"]


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_invalid_n_predictors_are_rejected(value: Any) -> None:
    with pytest.raises(ValueError, match="n_predictors"):
        ZeroShotClassifier(model=FAKE_MODEL, n_predictors=value).fit(None, SENTIMENTS)


def test_invalid_max_labels_are_rejected() -> None:
    with pytest.raises(ValueError, match="max_labels"):
        MultiLabelZeroShotClassifier(model=FAKE_MODEL, max_labels=0).fit(
            None, [["a"], ["b"]]
        )


# --- dynamic few-shot --------------------------------------------------------

#: Where each text sits on the unit circle, so which examples are nearest is a
#: fact of the test rather than of the fake backend's own embedding.
ANGLES = {
    "cheap and cheerful": 0.0,
    "great value": 0.1,
    "worth every cent": 0.2,
    "a bargain": 0.3,
    "costs too much": 0.4,
    "arrived next day": 1.5,
    "how much does it cost?": 0.24,  # the query, nearest to 0.2 then 0.3 then 0.1
}
PRICE_TEXTS = [
    "cheap and cheerful",
    "great value",
    "worth every cent",
    "a bargain",
    "costs too much",
]
DELIVERY_TEXTS = ["arrived next day"]


def embed_by_angle(call: FakeCall) -> list[list[float]]:
    return [[math.cos(ANGLES[text]), math.sin(ANGLES[text])] for text in call.texts]


@pytest.fixture
def dynamic(fake_backend: FakeBackend) -> DynamicFewShotClassifier:
    fake_backend.structured_script = label("positive")
    return DynamicFewShotClassifier(
        model=FAKE_MODEL, verbose=0, vectorizer=Vectorizer(model=FAKE_MODEL, verbose=0)
    ).fit(REVIEWS, SENTIMENTS)


def test_each_class_contributes_its_own_nearest_examples(
    fake_backend: FakeBackend,
) -> None:
    """A class with fewer examples than n_examples contributes what it has."""
    fake_backend.embedding_script = embed_by_angle
    fake_backend.structured_script = label("price")
    classifier = DynamicFewShotClassifier(
        model=FAKE_MODEL,
        verbose=0,
        n_examples=3,
        vectorizer=Vectorizer(model=FAKE_MODEL, verbose=0),
    ).fit(PRICE_TEXTS + DELIVERY_TEXTS, ["price"] * 5 + ["delivery"])

    classifier.predict(["how much does it cost?"])

    prompt = fake_backend.prompts[-1]
    assert prompt.count("Categories: price") == 3
    assert prompt.count("Categories: delivery") == 1
    assert "worth every cent" in prompt  # the three nearest of the large class
    assert "a bargain" in prompt
    assert "great value" in prompt
    assert "cheap and cheerful" not in prompt
    assert "costs too much" not in prompt
    assert "arrived next day" in prompt  # the only example the small class has


def test_the_query_batch_is_embedded_once(
    dynamic: DynamicFewShotClassifier, fake_backend: FakeBackend
) -> None:
    fake_backend.calls.clear()

    dynamic.predict(["one", "two", "three"])

    kinds = [call.kind for call in fake_backend.calls]
    assert kinds.count("embed") == 1
    assert kinds.count("structured") == 3
    assert fake_backend.calls[0].texts == ("one", "two", "three")


def test_an_empty_batch_embeds_nothing(
    dynamic: DynamicFewShotClassifier, fake_backend: FakeBackend
) -> None:
    fake_backend.calls.clear()

    assert list(dynamic.predict([])) == []
    assert fake_backend.calls == []


def test_the_supplied_vectorizer_is_cloned_into_the_fitted_state(
    fake_backend: FakeBackend,
) -> None:
    vectorizer = Vectorizer(model=FAKE_MODEL, verbose=0)
    classifier = DynamicFewShotClassifier(
        model=FAKE_MODEL, verbose=0, vectorizer=vectorizer
    ).fit(REVIEWS, SENTIMENTS)

    assert classifier.vectorizer is vectorizer
    assert classifier.vectorizer_ is not vectorizer
    assert classifier.vectorizer_.get_params() == vectorizer.get_params()


@pytest.mark.parametrize(
    ("provider", "embedding_model"),
    [
        ("openai", "openai::text-embedding-3-small"),
        ("google", "google::gemini-embedding-001"),
    ],
)
def test_providers_with_a_default_embedding_model_need_no_vectorizer(
    provider: str,
    embedding_model: str,
    fake_provider: Callable[[str], FakeBackend],
) -> None:
    backend = fake_provider(provider)
    classifier = DynamicFewShotClassifier(model=f"{provider}::a-chat-model", verbose=0)

    classifier.fit(REVIEWS, SENTIMENTS)

    assert classifier.vectorizer_.model == embedding_model
    assert backend.calls[0].model == embedding_model.split("::")[1]
    # The fit-created vectorizer needs the same credential the classifier does.
    assert classifier.llm_requirements() == (Requirement(provider, None),)


def test_the_default_vectorizer_inherits_the_connection_config(
    fake_provider: Callable[[str], FakeBackend],
) -> None:
    backend = fake_provider("openai")

    DynamicFewShotClassifier(
        model="openai::a-chat-model",
        verbose=0,
        api_key="sk-param",
        base_url="http://proxy/v1",
    ).fit(REVIEWS, SENTIMENTS)

    assert backend.calls[0].api_key == "sk-param"
    assert backend.calls[0].base_url == "http://proxy/v1"


def test_the_inherited_connection_config_is_not_pickled(
    fake_provider: Callable[[str], FakeBackend],
) -> None:
    """The fit-created vectorizer holds the key in memory, never in the payload."""
    fake_provider("openai")
    classifier = DynamicFewShotClassifier(
        model="openai::a-chat-model", verbose=0, api_key="sk-secret"
    ).fit(REVIEWS, SENTIMENTS)

    assert classifier.vectorizer_.api_key == "sk-secret"
    payload = pickle.dumps(classifier)

    assert b"sk-secret" not in payload
    assert pickle.loads(payload).vectorizer_.api_key is None


def test_the_default_vectorizer_shares_the_alias(
    fake_provider: Callable[[str], FakeBackend],
) -> None:
    backend = fake_provider("openai")
    classifier = DynamicFewShotClassifier(
        model="openai::a-chat-model", verbose=0, alias="team_a"
    )

    with config_context(credentials={"team_a": "sk-a"}):
        classifier.fit(REVIEWS, SENTIMENTS)

    assert backend.calls[0].api_key == "sk-a"
    assert classifier.llm_requirements() == (Requirement("openai", "team_a"),)


def test_an_override_aimed_at_the_classifier_spares_the_embedding_call(
    fake_provider: Callable[[str], FakeBackend],
) -> None:
    """The shared alias routes the credential, not the generation params."""
    backend = fake_provider("openai")
    backend.structured_script = label("positive")
    classifier = DynamicFewShotClassifier(
        model="openai::a-chat-model", verbose=0, alias="careful"
    )

    with config_context(credentials={"careful": "sk-a"}):
        classifier.fit(REVIEWS, SENTIMENTS)
        with config_context(overrides={"careful": {"temperature": 0.9}}):
            classifier.predict(["I love it"])

    embedding, chat = backend.calls[-2], backend.calls[-1]
    assert (embedding.kind, chat.kind) == ("embed", "structured")
    assert "temperature" not in embedding.params
    assert chat.params["temperature"] == 0.9


def test_a_provider_without_a_default_embedding_model_demands_a_vectorizer() -> None:
    classifier = DynamicFewShotClassifier(model="anthropic::claude-sonnet-5")

    with pytest.raises(ValueError) as excinfo:
        classifier.fit(REVIEWS, SENTIMENTS)

    message = str(excinfo.value)
    assert "anthropic" in message
    assert "no embeddings" in message
    assert "vectorizer=Vectorizer(model=" in message


@pytest.mark.parametrize("value", [0, None, 1.5])
def test_invalid_n_examples_are_rejected(value: Any) -> None:
    with pytest.raises(ValueError, match="n_examples"):
        DynamicFewShotClassifier(model=FAKE_MODEL, n_examples=value).fit(
            REVIEWS, SENTIMENTS
        )


def test_a_foreign_vectorizer_is_rejected() -> None:
    with pytest.raises(TypeError, match="skllm Vectorizer"):
        DynamicFewShotClassifier(model=FAKE_MODEL, vectorizer=ZeroShotClassifier()).fit(
            REVIEWS, SENTIMENTS
        )


def test_dynamic_predictions_come_from_the_fitted_classes(
    dynamic: DynamicFewShotClassifier, fake_backend: FakeBackend
) -> None:
    assert dynamic.classes_ == ["negative", "positive"]
    assert list(dynamic.predict(["I love it"])) == ["positive"]


def test_the_retrieval_indexes_survive_a_pickle_round_trip(
    dynamic: DynamicFewShotClassifier, fake_backend: FakeBackend
) -> None:
    restored = pickle.loads(pickle.dumps(dynamic))

    assert list(dynamic.predict(["I love it"])) == ["positive"]
    assert list(restored.predict(["I love it"])) == ["positive"]

    prompts = [call.prompt for call in fake_backend.calls if call.kind == "structured"]
    assert prompts[-1] == prompts[-2]  # the same examples were retrieved
    assert "It broke on day one" in prompts[-1]


# --- sklearn compliance ------------------------------------------------------


@pytest.mark.parametrize(
    "classifier",
    [
        ZeroShotClassifier,
        FewShotClassifier,
        DynamicFewShotClassifier,
        MultiLabelZeroShotClassifier,
        MultiLabelFewShotClassifier,
    ],
)
def test_params_survive_a_clone(classifier: type[Any]) -> None:
    estimator = classifier(
        model=FAKE_MODEL, api_key="sk-secret", alias="team_a", n_predictors=3
    )

    cloned = clone(estimator)

    assert cloned.get_params() == estimator.get_params()
    assert cloned.api_key == "sk-secret"
    assert vars(estimator) == estimator.get_params()


@pytest.mark.parametrize(
    "classifier",
    [
        ZeroShotClassifier,
        FewShotClassifier,
        DynamicFewShotClassifier,
        MultiLabelZeroShotClassifier,
        MultiLabelFewShotClassifier,
    ],
)
def test_unfitted_classifiers_raise(classifier: type[Any]) -> None:
    assert get_tags(classifier()).requires_fit

    with pytest.raises(NotFittedError):
        classifier().predict(["I love it"])


def test_an_empty_batch_issues_no_requests(
    zero_shot: ZeroShotClassifier, fake_backend: FakeBackend
) -> None:
    assert list(zero_shot.predict([])) == []
    assert zero_shot.predict_proba([]).shape == (0, 2)
    assert fake_backend.calls == []


def test_pickling_keeps_the_fitted_state_and_drops_the_key(
    fake_backend: FakeBackend,
) -> None:
    fake_backend.structured_script = label("positive")
    classifier = FewShotClassifier(
        model=FAKE_MODEL, verbose=0, api_key="sk-secret"
    ).fit(REVIEWS, ["positive", "negative"])

    payload = pickle.dumps(classifier)
    restored = pickle.loads(payload)

    assert b"sk-secret" not in payload
    assert restored.api_key is None
    assert restored.classes_ == ["negative", "positive"]
    assert restored.examples_ == classifier.examples_

    with config_context(fake_api_key="sk-new"):
        assert list(restored.predict(["I love it"])) == ["positive"]
    assert fake_backend.calls[-1].api_key == "sk-new"


def test_predictions_keep_the_input_order(fake_backend: FakeBackend) -> None:
    def echo_the_sample(call: FakeCall) -> dict[str, str]:
        return label("positive" if "love" in call.prompt else "negative")

    fake_backend.structured_script = echo_the_sample
    classifier = ZeroShotClassifier(model=FAKE_MODEL, verbose=0).fit(None, SENTIMENTS)

    assert list(classifier.predict(REVIEWS)) == ["positive", "negative"]
