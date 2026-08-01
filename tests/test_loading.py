"""Loading a bundle back: what ``skllm.load`` hands over, and how it calls.

The facade is exercised the way a user meets it — save a fitted estimator, load
the bundle, call it — with the fake adapter standing in for the providers. What
is asserted throughout is equivalence: the loaded model answers what the live
one answered, in the same container, authenticating the same way.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, ClassVar, NamedTuple

import numpy as np
import pytest
from fnnx.handlers.local import LocalHandler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import skllm
from skllm import (
    NER,
    ClusterLabeler,
    ConsensusClassifier,
    DynamicFewShotClassifier,
    FewShotClassifier,
    MultiLabelFewShotClassifier,
    MultiLabelZeroShotClassifier,
    RubricScorer,
    StructuredExtractor,
    Summarizer,
    Translator,
    Vectorizer,
    ZeroShotClassifier,
)
from skllm.backends import register_backend, registry
from skllm.config import MissingCredentialError, config_context
from tests.fakes import FakeBackend, OpenAIFake

CHAT_MODEL = "openai::gpt-5"
EMBED_MODEL = "openai::text-embedding-3-small"

REVIEWS = ["I love it", "It broke on day one"]
SENTIMENTS = ["positive", "negative"]
ASPECTS = [["price", "quality"], ["delivery"]]
CLUSTERED = ["great", "awful", "fine"]

TEXT = "Tim Cook leads Apple."
TAGGED = "<PER>Tim Cook</PER> leads Apple."

LABEL = {"label": "positive"}
MOVIE_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "year": {"type": "integer"}},
    "required": ["name", "year"],
}


class Case(NamedTuple):
    """One estimator, the batch to call it with, and what its backend answers."""

    name: str
    build: Callable[[], Any]
    X: Any
    chat: Any = "a summary"
    structured: Any = LABEL


CASES = [
    Case(
        "zero_shot",
        lambda: ZeroShotClassifier(model=CHAT_MODEL, verbose=0).fit(None, SENTIMENTS),
        REVIEWS,
    ),
    Case(
        "few_shot",
        lambda: FewShotClassifier(model=CHAT_MODEL, verbose=0).fit(REVIEWS, SENTIMENTS),
        REVIEWS,
    ),
    Case(
        "dynamic_few_shot",
        lambda: DynamicFewShotClassifier(model=CHAT_MODEL, n_examples=1, verbose=0).fit(
            REVIEWS, SENTIMENTS
        ),
        REVIEWS,
    ),
    Case(
        "multi_label_zero_shot",
        lambda: MultiLabelZeroShotClassifier(model=CHAT_MODEL, verbose=0).fit(
            None, ASPECTS
        ),
        REVIEWS,
        structured={"labels": ["price"]},
    ),
    Case(
        "multi_label_few_shot",
        lambda: MultiLabelFewShotClassifier(model=CHAT_MODEL, verbose=0).fit(
            REVIEWS, ASPECTS
        ),
        REVIEWS,
        structured={"labels": ["price"]},
    ),
    Case(
        "consensus",
        lambda: ConsensusClassifier(
            [
                ZeroShotClassifier(model="openai::fast", verbose=0),
                ZeroShotClassifier(model="openai::careful", verbose=0),
            ],
            verbose=0,
        ).fit(REVIEWS, SENTIMENTS),
        REVIEWS,
    ),
    Case("summarizer", lambda: Summarizer(model=CHAT_MODEL, verbose=0), REVIEWS),
    Case(
        "translator",
        lambda: Translator(model=CHAT_MODEL, output_language="German", verbose=0),
        REVIEWS,
    ),
    Case(
        "ner",
        lambda: NER(model=CHAT_MODEL, entities={"PER": "a person"}, verbose=0),
        [TEXT],
        chat=TAGGED,
    ),
    Case(
        "extractor",
        lambda: StructuredExtractor(model=CHAT_MODEL, schema=MOVIE_SCHEMA, verbose=0),
        REVIEWS,
        structured={"name": "Dune", "year": 1965},
    ),
    Case(
        "scorer",
        lambda: RubricScorer(model=CHAT_MODEL, rubric="clarity", verbose=0),
        REVIEWS,
        structured={"score": 7},
    ),
    Case(
        "cluster_labeler",
        lambda: ClusterLabeler(model=CHAT_MODEL, verbose=0).fit(CLUSTERED, [0, 1, -1]),
        [0, 1, -1],
        chat="Praise",
    ),
    Case("vectorizer", lambda: Vectorizer(model=EMBED_MODEL, verbose=0), REVIEWS),
]


class RecordingHandler(LocalHandler):
    """The default handler, plus a record of what the facade forwarded to it.

    The forwarding is invisible in-process — a local bundle would resolve the
    same context itself — but it is the whole of what an isolated handler, whose
    process holds none of this scope, has to go on.
    """

    seen: ClassVar[list[dict[str, Any]]] = []

    def compute(self, inputs: dict, dynamic_attributes: dict) -> dict:
        RecordingHandler.seen.append(dynamic_attributes)
        answer: dict = super().compute(inputs, dynamic_attributes)
        return answer


@pytest.fixture
def openai(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeBackend]:
    """Answer ``openai::`` with the fake, with a key ambiently available."""
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    backend = OpenAIFake(structured=LABEL)
    register_backend("openai", backend)
    try:
        yield backend
    finally:
        registry._BACKENDS.pop("openai", None)


@pytest.fixture
def bundle(tmp_path: Any) -> Callable[..., str]:
    def save(estimator: Any, name: str = "model.fnnx") -> str:
        path = str(tmp_path / name)
        skllm.save(estimator, path)
        return path

    return save


@pytest.fixture
def recorded() -> Iterator[list[dict[str, Any]]]:
    RecordingHandler.seen.clear()
    yield RecordingHandler.seen
    RecordingHandler.seen.clear()


def fitted_classifier(**params: Any) -> ZeroShotClassifier:
    return ZeroShotClassifier(model=CHAT_MODEL, verbose=0, **params).fit(
        None, SENTIMENTS
    )


def assert_same_answer(loaded: Any, live: Any) -> None:
    """The loaded answer is the live one: same container, same values."""
    assert type(loaded) is type(live)
    if not isinstance(live, np.ndarray):
        assert loaded == live
        return
    assert (loaded.dtype, loaded.shape) == (live.dtype, live.shape)
    if live.dtype == object:
        assert loaded.tolist() == live.tolist()
    else:
        assert np.array_equal(loaded, live, equal_nan=True)


# --- round-trip equivalence ---------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_a_loaded_bundle_answers_what_the_estimator_answered(
    openai: FakeBackend, bundle: Callable[..., str], case: Case
) -> None:
    openai.chat_script, openai.structured_script = case.chat, case.structured
    estimator = case.build()

    model = skllm.load(bundle(estimator))
    answer = getattr(model, model.method)(case.X)

    assert_same_answer(answer, getattr(estimator, model.method)(case.X))


def test_a_pipeline_bundle_answers_what_the_pipeline_answered(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    """The skllm step embeds the new text, the foreign step classifies it.

    The step's key travels nowhere: stripping applies inside a foreign structure
    too, and the loaded bundle takes the calling process's key instead.
    """
    pipeline = Pipeline(
        [
            ("embed", Vectorizer(model=EMBED_MODEL, api_key="sk-secret", verbose=0)),
            ("classify", LogisticRegression()),
        ]
    ).fit(CLUSTERED, ["p", "n", "p"])
    path = bundle(pipeline)
    live = pipeline.predict(["some text"])

    model = skllm.load(path)
    with config_context(openai_api_key="sk-new"):
        answer = model.predict(["some text"])

    assert list(answer) == list(live)
    assert openai.calls[-1].api_key == "sk-new"
    assert b"sk-secret" not in Path(path).read_bytes()


def test_a_bundle_exposes_only_the_task_method_it_recorded(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    classifier = skllm.load(bundle(fitted_classifier()))
    summarizer = skllm.load(bundle(Summarizer(model=CHAT_MODEL, verbose=0), "s.fnnx"))

    assert (classifier.method, summarizer.method) == ("predict", "transform")
    assert not hasattr(classifier, "transform")
    assert not hasattr(summarizer, "predict")
    with pytest.raises(AttributeError, match="answers through 'transform'"):
        summarizer.predict(REVIEWS)


def test_a_batch_may_arrive_in_any_of_the_usual_containers(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    model = skllm.load(bundle(fitted_classifier()))

    assert list(model.predict(np.array(REVIEWS))) == ["positive"] * 2


def test_a_batch_that_is_not_text_is_refused_in_the_estimator_s_own_words(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    model = skllm.load(bundle(fitted_classifier()))

    with pytest.raises(TypeError, match="every sample must be a string"):
        model.predict([1, 2])


def test_an_empty_batch_answers_empty_without_a_request(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    vectorizer = Vectorizer(model=EMBED_MODEL, verbose=0)
    classifier = skllm.load(bundle(fitted_classifier()))
    embedder = skllm.load(bundle(vectorizer, "vectors.fnnx"))
    issued = len(openai.calls)

    assert list(classifier.predict([])) == []
    # Two-dimensional, as the live Vectorizer answers an empty batch.
    assert embedder.transform([]).shape == vectorizer.transform([]).shape == (0, 0)
    assert len(openai.calls) == issued


# --- credentials and overrides come from the calling process ------------------


def test_the_key_comes_from_the_calling_process(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    classifier = FewShotClassifier(
        model=CHAT_MODEL, api_key="sk-secret", verbose=0
    ).fit(REVIEWS, SENTIMENTS)
    model = skllm.load(bundle(classifier))

    with config_context(openai_api_key="sk-new"):
        model.predict(["I love it"])
    assert openai.calls[-1].api_key == "sk-new"

    model.predict(["I love it"])
    assert openai.calls[-1].api_key == "env-key"


def test_a_missing_key_is_reported_by_the_bundle_that_needs_it(
    openai: FakeBackend, bundle: Callable[..., str], monkeypatch: pytest.MonkeyPatch
) -> None:
    path = bundle(fitted_classifier())
    monkeypatch.delenv("OPENAI_API_KEY")

    model = skllm.load(path)

    with pytest.raises(MissingCredentialError, match=r"openai_api_key.*OPENAI_API_KEY"):
        model.predict(["I love it"])


def test_the_facade_forwards_what_the_bundle_declared_and_nothing_else(
    openai: FakeBackend, bundle: Callable[..., str], recorded: list[dict[str, Any]]
) -> None:
    """A scope may hold entries for estimators this bundle knows nothing about."""
    classifier = fitted_classifier(alias="team_a")
    model = skllm.load(bundle(classifier), handler=RecordingHandler)

    with config_context(
        credentials={"team_a": "sk-a", "team_b": "sk-b"},
        overrides={"team_a": {"temperature": 0.9}, "stranger": {"temperature": 0.1}},
    ):
        model.predict(["I love it"])

    assert recorded[-1] == {
        "credential_team_a": "sk-a",
        "overrides": {"temperature": 0.9},
    }
    assert openai.calls[-1].api_key == "sk-a"
    assert openai.calls[-1].params == {"temperature": 0.9}


def test_a_lone_estimator_is_tuned_by_its_own_alias_not_a_component_s(
    openai: FakeBackend, bundle: Callable[..., str], recorded: list[dict[str, Any]]
) -> None:
    """Its vectorizer is aliased too, but serving tunes only the estimator itself."""
    keys = {"chat": "sk-chat", "a_embed": "sk-embed"}

    with config_context(credentials=keys):
        classifier = DynamicFewShotClassifier(
            model=CHAT_MODEL,
            alias="chat",
            vectorizer=Vectorizer(model=EMBED_MODEL, alias="a_embed", verbose=0),
            n_examples=1,
            verbose=0,
        ).fit(REVIEWS, SENTIMENTS)
        model = skllm.load(bundle(classifier), handler=RecordingHandler)

        with config_context(
            overrides={"chat": {"temperature": 0.9}, "a_embed": {"temperature": 0.1}}
        ):
            model.predict(["I love it"])

    assert recorded[-1] == {
        "credential_chat": "sk-chat",
        "credential_a_embed": "sk-embed",
        "overrides": {"temperature": 0.9},
    }


def test_an_override_reaches_exactly_the_member_it_names(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    ensemble = ConsensusClassifier(
        [
            ZeroShotClassifier(model="openai::fast", alias="fast", verbose=0),
            ZeroShotClassifier(model="openai::careful", alias="careful", verbose=0),
        ],
        verbose=0,
    ).fit(REVIEWS, SENTIMENTS)
    model = skllm.load(bundle(ensemble))
    keys = {"fast": "sk-a", "careful": "sk-b"}

    with config_context(credentials=keys, overrides={"careful": {"temperature": 0.9}}):
        model.predict(["I love it"])

    assert {call.model: call.params for call in openai.calls[-2:]} == {
        "fast": {"temperature": 0},
        "careful": {"temperature": 0.9},
    }
    assert {call.model: call.api_key for call in openai.calls[-2:]} == {
        "fast": "sk-a",
        "careful": "sk-b",
    }
    with (
        config_context(credentials={"fast": "sk-a"}),
        pytest.raises(MissingCredentialError, match="careful"),
    ):
        model.predict(["I love it"])


def test_an_override_reaches_a_member_as_the_value_it_was_given(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    """A member's params are undeclared, so the bundle retypes none of them.

    ``user`` is a string OpenAI rejects as an integer: a scope that reads one
    way around the live ensemble must not read another way around its bundle.
    """
    ensemble = ConsensusClassifier(
        [ZeroShotClassifier(model="openai::fast", alias="fast", verbose=0)], verbose=0
    )
    with config_context(credentials={"fast": "sk-a"}):
        ensemble.fit(REVIEWS, SENTIMENTS)
    model = skllm.load(bundle(ensemble))

    with config_context(
        credentials={"fast": "sk-a"}, overrides={"fast": {"user": "42"}}
    ):
        ensemble.predict(["I love it"])
        live = openai.calls[-1].params
        model.predict(["I love it"])

    assert openai.calls[-1].params == live == {"temperature": 0, "user": "42"}


def test_a_base_url_reaches_the_provider(
    openai: FakeBackend, bundle: Callable[..., str], recorded: list[dict[str, Any]]
) -> None:
    model = skllm.load(bundle(fitted_classifier()), handler=RecordingHandler)

    with config_context(openai_base_url="http://localhost:8000/v1"):
        model.predict(["I love it"])

    assert recorded[-1]["openai_base_url"] == "http://localhost:8000/v1"
    assert openai.calls[-1].base_url == "http://localhost:8000/v1"


# --- back to a live object ----------------------------------------------------


def test_unwrap_restores_the_fitted_estimator_without_its_key(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    classifier = FewShotClassifier(
        model=CHAT_MODEL,
        api_key="sk-secret",
        model_params={"temperature": 0},
        verbose=0,
    ).fit(REVIEWS, SENTIMENTS)

    restored = skllm.load(bundle(classifier)).unwrap()

    assert type(restored) is FewShotClassifier
    assert restored.get_params() == {**classifier.get_params(), "api_key": None}
    assert list(restored.classes_) == list(classifier.classes_)
    assert restored.examples_ == classifier.examples_
    with config_context(openai_api_key="sk-new"):
        assert list(restored.predict(REVIEWS)) == list(classifier.predict(REVIEWS))


def test_unwrap_restores_a_whole_composition(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    pipeline = Pipeline(
        [
            ("embed", Vectorizer(model=EMBED_MODEL, verbose=0)),
            ("classify", LogisticRegression()),
        ]
    ).fit(CLUSTERED, ["p", "n", "p"])

    restored = skllm.load(bundle(pipeline)).unwrap()

    assert isinstance(restored, Pipeline)
    assert restored.named_steps["embed"].model == EMBED_MODEL
    assert list(restored.predict(["some text"])) == list(
        pipeline.predict(["some text"])
    )


def test_a_bundle_scikit_llm_did_not_write_is_refused(tmp_path: Any) -> None:
    (tmp_path / "variant_config.json").write_text(json.dumps({"extra_values": {}}))

    with pytest.raises(ValueError, match="not one scikit-llm wrote"):
        skllm.load(str(tmp_path))
