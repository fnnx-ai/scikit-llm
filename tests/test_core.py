from __future__ import annotations

import pickle
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytest
from sklearn.base import ClassifierMixin, TransformerMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils import get_tags
from sklearn.utils.validation import check_is_fitted

from skllm.backends import Message
from skllm.config import MissingCredentialError, config_context
from skllm.core import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    BaseLLMEstimator,
    BasePromptEstimator,
    Requirement,
    SampleError,
    StatelessMixin,
    UsageRecord,
    UsageReport,
    as_texts,
)
from tests.fakes import FakeBackend, FakeCall, FakeTransientError

FAKE_MODEL = "fake::m"


class ToyClassifier(ClassifierMixin, BasePromptEstimator):
    """A stateful, prompted estimator standing in for the task classifiers."""

    prompt_placeholders = frozenset({"labels", "text"})
    default_prompt_template = "Pick one of {labels} for: {text}"

    def __init__(
        self,
        model: str = FAKE_MODEL,
        *,
        default_label: str | None = None,
        prompt_template: str | None = None,
        model_params: Mapping[str, Any] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        alias: str | None = None,
        on_error: str = "fill",
        max_concurrency: int = 8,
        verbose: int = 0,
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
        self.default_label = default_label

    def fit(self, X: Any, y: Sequence[str]) -> ToyClassifier:
        self._check_params()
        self.classes_ = sorted(set(y))
        return self

    def predict(self, X: Any) -> np.ndarray:
        self._check_ready()
        labels = self._run_batch(
            as_texts(X), self._label, fill=lambda: self.default_label
        )
        return np.asarray(labels, dtype=object)

    def _label(self, text: str) -> str:
        prompt = self._prompt(labels=", ".join(self.classes_), text=text)
        return self._chat([Message("user", prompt)])


class ToyRewriter(TransformerMixin, StatelessMixin, BasePromptEstimator):
    """A stateless, prompted transformer standing in for the text2text tasks."""

    prompt_placeholders = frozenset({"text"})
    default_prompt_template = "Rewrite: {text}"

    def transform(self, X: Any) -> np.ndarray:
        self._check_ready()
        rewrites = self._run_batch(as_texts(X), self._rewrite, fill=str)
        return np.asarray(rewrites, dtype=object)

    def _rewrite(self, text: str) -> str:
        return self._chat([Message("user", self._prompt(text=text))])


class ToyVectorizer(TransformerMixin, StatelessMixin, BaseLLMEstimator):
    """A stateless embedding transformer, for the nesting and embedding paths."""

    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        *,
        model_params: Mapping[str, Any] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        alias: str | None = None,
        on_error: str = "fill",
        max_concurrency: int = 8,
        verbose: int = 0,
    ) -> None:
        super().__init__(
            model=model,
            model_params=model_params,
            api_key=api_key,
            base_url=base_url,
            alias=alias,
            on_error=on_error,
            max_concurrency=max_concurrency,
            verbose=verbose,
        )

    def transform(self, X: Any) -> np.ndarray:
        self._check_ready()
        batches = [[text] for text in as_texts(X)]
        vectors: list[list[list[float]]] = self._run_batch(
            batches, self._embed, fill=list
        )
        return np.asarray([vector for batch in vectors for vector in batch])


class ToyRetriever(BaseLLMEstimator):
    """Builds a component at fit, like the dynamic few-shot classifier does."""

    def __init__(
        self,
        model: str = FAKE_MODEL,
        *,
        vectorizer: ToyVectorizer | None = None,
        model_params: Mapping[str, Any] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        alias: str | None = None,
        on_error: str = "fill",
        max_concurrency: int = 8,
        verbose: int = 0,
    ) -> None:
        super().__init__(
            model=model,
            model_params=model_params,
            api_key=api_key,
            base_url=base_url,
            alias=alias,
            on_error=on_error,
            max_concurrency=max_concurrency,
            verbose=verbose,
        )
        self.vectorizer = vectorizer

    def fit(self, X: Any, y: Any = None) -> ToyRetriever:
        self._check_params()
        self.vectorizer_ = self.vectorizer or ToyVectorizer()
        return self


class ToyFitOnlyEstimator(ToyRetriever):
    """Calls its LLM only at fit, so it declares no serving-time credential."""

    def _own_requirements(self) -> tuple[Requirement, ...]:
        return ()


class ToyNamer(BaseLLMEstimator):
    """Names its inputs at fit, like the cluster labeler: a batch before any state."""

    def fit(self, X: Any, y: Any = None) -> ToyNamer:
        self._check_params()
        names = self._run_batch(as_texts(X), self._name, fill=str)
        if not all(names):
            raise RuntimeError("naming failed")
        self.names_ = names
        return self

    def _name(self, text: str) -> str:
        return self._chat([Message("user", text)])


@pytest.fixture
def classifier(fake_backend: FakeBackend) -> ToyClassifier:
    fake_backend.chat_script = "positive"
    return ToyClassifier().fit(None, ["negative", "positive"])


def answer_with_the_input(call: FakeCall) -> str:
    """Echo the sample back, which the toy prompt puts after the last colon."""
    return call.prompt.rsplit(": ", 1)[-1]


# --- sklearn compliance ------------------------------------------------------


def test_params_survive_clone_and_set_params() -> None:
    estimator = ToyClassifier(
        model="fake::other",
        api_key="sk-secret",
        base_url="http://proxy/v1",
        alias="team_a",
        model_params={"temperature": 0.4},
        default_label="unknown",
    )

    cloned = clone(estimator)

    assert cloned.get_params() == estimator.get_params()
    assert cloned.api_key == "sk-secret"
    assert cloned.base_url == "http://proxy/v1"
    assert cloned.set_params(alias="team_b").alias == "team_b"


def test_init_only_assigns_params() -> None:
    estimator = ToyClassifier(model="fake::x")

    assert vars(estimator) == estimator.get_params()


def test_estimators_declare_text_input() -> None:
    tags = get_tags(ToyClassifier())

    assert tags.input_tags.string
    assert not tags.input_tags.two_d_array
    assert tags.non_deterministic


def test_stateful_estimators_require_fit() -> None:
    assert get_tags(ToyClassifier()).requires_fit

    with pytest.raises(NotFittedError):
        ToyClassifier().predict(["anything"])


def test_bookkeeping_is_never_mistaken_for_fitted_state(
    fake_backend: FakeBackend,
) -> None:
    """A fit that batches before it stores state must not end up half-fitted."""
    fake_backend.chat_script = ""  # an unusable name, so fit aborts after the batch
    namer = ToyNamer(model=FAKE_MODEL)

    with pytest.raises(RuntimeError, match="naming failed"):
        namer.fit(["a cluster"])

    assert namer.usage_.total.requests == 1
    with pytest.raises(NotFittedError):
        check_is_fitted(namer)


def test_stateless_estimators_work_without_fit(fake_backend: FakeBackend) -> None:
    fake_backend.chat_script = "rewritten"
    rewriter = ToyRewriter(model=FAKE_MODEL)

    assert not get_tags(rewriter).requires_fit
    assert list(rewriter.transform(["a text"])) == ["rewritten"]


def test_stateless_fit_only_validates_params(fake_backend: FakeBackend) -> None:
    rewriter = ToyRewriter(model=FAKE_MODEL)

    assert rewriter.fit(["a text"]) is rewriter
    assert fake_backend.calls == []

    with pytest.raises(ValueError, match="cohere"):
        ToyRewriter(model="cohere::command-r").fit(["a text"])


def test_estimators_work_inside_a_pipeline(fake_backend: FakeBackend) -> None:
    fake_backend.chat_script = answer_with_the_input
    pipeline = Pipeline(
        [("rewrite", ToyRewriter(model=FAKE_MODEL)), ("classify", ToyClassifier())]
    )

    pipeline.fit(["good", "bad"], ["positive", "negative"])

    assert list(pipeline.predict(["good"])) == ["good"]
    assert pipeline.named_steps["classify"].classes_ == ["negative", "positive"]


# --- input handling ----------------------------------------------------------


def test_supported_text_containers() -> None:
    class Series:
        """A pandas Series is duck-typed: only tolist() and ndim are used."""

        ndim = 1

        def tolist(self) -> list[str]:
            return ["a", "b"]

    assert as_texts(["a", "b"]) == ["a", "b"]
    assert as_texts(("a", "b")) == ["a", "b"]
    assert as_texts(np.array(["a", "b"])) == ["a", "b"]
    assert as_texts(Series()) == ["a", "b"]
    assert as_texts([]) == []


@pytest.mark.parametrize(
    ("X", "message"),
    [
        (None, "sequence of strings"),
        ("a single text", "sequence of strings"),
        (np.array([["a"], ["b"]]), "one-dimensional"),
        ([1, 2], "must be a string"),
    ],
)
def test_unsupported_inputs_are_rejected(X: Any, message: str) -> None:
    with pytest.raises(TypeError, match=message):
        as_texts(X)


def test_dataframe_input_names_the_fix() -> None:
    class DataFrame:
        columns = ("text", "label")

        def __iter__(self) -> Any:
            return iter(self.columns)

    with pytest.raises(TypeError, match=r"df\['text'\]"):
        as_texts(DataFrame())


def test_empty_batch_issues_no_requests(classifier: ToyClassifier) -> None:
    classifier.predict(["one text"])
    before = classifier.usage_.total

    assert list(classifier.predict([])) == []
    assert classifier.usage_.last == UsageRecord()
    assert classifier.usage_.total == before
    assert classifier.errors_ == ()


# --- param validation --------------------------------------------------------


def test_unknown_provider_is_reported_at_fit() -> None:
    with pytest.raises(ValueError) as excinfo:
        ToyClassifier(model="cohere::command-r").fit(None, ["a", "b"])

    assert "cohere" in str(excinfo.value)
    assert "openai" in str(excinfo.value)


def test_invalid_alias_is_reported_at_fit() -> None:
    with pytest.raises(ValueError, match=r"\[a-z\]\[a-z0-9_\]\*"):
        ToyClassifier(alias="Team A").fit(None, ["a", "b"])


def test_invalid_alias_is_reported_on_a_stateless_first_use(
    fake_backend: FakeBackend,
) -> None:
    with pytest.raises(ValueError, match=r"\[a-z\]\[a-z0-9_\]\*"):
        ToyRewriter(model=FAKE_MODEL, alias="Team A").transform(["a text"])

    assert fake_backend.calls == []


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"on_error": "ignore"}, "on_error"),
        ({"max_concurrency": 0}, "max_concurrency"),
        ({"max_concurrency": 1.5}, "max_concurrency"),
    ],
)
def test_invalid_params_are_rejected(params: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ToyClassifier(**params).fit(None, ["a", "b"])


def test_non_mapping_model_params_are_rejected() -> None:
    with pytest.raises(TypeError, match="model_params"):
        ToyClassifier(model_params=[("temperature", 0)]).fit(None, ["a", "b"])  # type: ignore[arg-type]


def test_prompt_template_placeholders_are_validated(fake_backend: FakeBackend) -> None:
    with pytest.raises(ValueError) as excinfo:
        ToyClassifier(prompt_template="Classify {txt} as {labels} for {tone}").fit(
            None, ["a", "b"]
        )

    message = str(excinfo.value)
    assert "missing ['text']" in message
    assert "unknown ['tone', 'txt']" in message
    assert "['labels', 'text']" in message


def test_malformed_prompt_template_is_reported(fake_backend: FakeBackend) -> None:
    with pytest.raises(ValueError, match="prompt_template"):
        ToyClassifier(prompt_template="{labels} {text} }").fit(None, ["a", "b"])


@pytest.mark.parametrize(
    "template",
    [
        "Pick {labels} for {text}, answering with {}",
        "Pick {labels} for {text} as {0}",
        "Pick {labels} for {text:{width}}",
    ],
)
def test_unnamed_prompt_placeholders_are_rejected_at_fit(
    template: str, fake_backend: FakeBackend
) -> None:
    """Left to format(), they would fail once per sample as filled predictions."""
    with pytest.raises(ValueError, match="prompt_template"):
        ToyClassifier(prompt_template=template).fit(None, ["a", "b"])


def test_custom_prompt_template_reaches_the_provider(
    fake_backend: FakeBackend,
) -> None:
    fake_backend.chat_script = "positive"
    estimator = ToyClassifier(prompt_template="Labels: {labels}\nText: {text}").fit(
        None, ["negative", "positive"]
    )

    estimator.predict(["I love it"])

    assert fake_backend.prompts == ["Labels: negative, positive\nText: I love it"]


def test_params_are_validated_on_a_stateless_first_use(
    fake_backend: FakeBackend,
) -> None:
    with pytest.raises(ValueError, match="on_error"):
        ToyRewriter(model=FAKE_MODEL, on_error="ignore").transform(["a text"])


# --- the request path --------------------------------------------------------


def test_default_models_follow_the_spec() -> None:
    assert DEFAULT_CHAT_MODEL == "openai::gpt-5.6-sol"
    assert BaseLLMEstimator().model == DEFAULT_CHAT_MODEL
    assert ToyVectorizer().model == DEFAULT_EMBEDDING_MODEL


def test_model_params_and_credentials_reach_the_backend(
    fake_backend: FakeBackend,
) -> None:
    fake_backend.chat_script = "positive"
    estimator = ToyClassifier(
        api_key="sk-param", base_url="http://proxy/v1", model_params={"top_p": 0.3}
    ).fit(None, ["negative", "positive"])

    estimator.predict(["I love it"])

    call = fake_backend.calls[0]
    assert call.api_key == "sk-param"
    assert call.base_url == "http://proxy/v1"
    assert call.params == {"top_p": 0.3, "temperature": 0}


def test_credential_layers_resolve_through_the_estimator(
    fake_backend: FakeBackend,
) -> None:
    """The estimator param outranks the context, which outranks the environment."""
    fake_backend.chat_script = "positive"
    labels = ["negative", "positive"]

    with config_context(fake_api_key="ctx-key"):
        ToyClassifier(api_key="param-key").fit(None, labels).predict(["I love it"])
        ToyClassifier().fit(None, labels).predict(["I love it"])
    ToyClassifier().fit(None, labels).predict(["I love it"])

    assert [call.api_key for call in fake_backend.calls] == [
        "param-key",
        "ctx-key",
        "fake-key",  # the FAKE_API_KEY environment fallback
    ]


def test_an_aliased_estimator_never_falls_back_to_the_provider_key(
    fake_backend: FakeBackend,
) -> None:
    """An unresolved alias is a configuration error, not a filled sample failure."""
    estimator = ToyClassifier(alias="team_a").fit(None, ["negative", "positive"])

    with pytest.raises(MissingCredentialError, match="team_a"):
        estimator.predict(["I love it"])

    assert fake_backend.calls == []


def test_alias_overrides_beat_stored_model_params(fake_backend: FakeBackend) -> None:
    fake_backend.chat_script = "positive"
    estimator = ToyClassifier(alias="careful", model_params={"temperature": 0.2}).fit(
        None, ["negative", "positive"]
    )

    with config_context(
        credentials={"careful": "sk-careful"},
        overrides={"careful": {"temperature": 0.9}},
    ):
        estimator.predict(["I love it"])

    assert fake_backend.calls[0].params["temperature"] == 0.9
    assert fake_backend.calls[0].api_key == "sk-careful"


def test_embedding_requests_go_through_the_same_path(
    fake_backend: FakeBackend,
) -> None:
    vectors = ToyVectorizer(model=FAKE_MODEL).transform(["one", "two"])

    assert vectors.shape == (2, 16)
    assert [call.kind for call in fake_backend.calls] == ["embed", "embed"]
    assert fake_backend.calls[0].api_key == "fake-key"


# --- concurrency -------------------------------------------------------------


def test_results_keep_the_input_order(fake_backend: FakeBackend) -> None:
    def answer_slowest_first(call: Any) -> str:
        text = answer_with_the_input(call)
        time.sleep(0.02 * (4 - len(text)))
        return text

    fake_backend.chat_script = answer_slowest_first
    estimator = ToyClassifier().fit(None, ["a"])

    assert list(estimator.predict(["a", "bb", "ccc"])) == ["a", "bb", "ccc"]


def test_samples_are_answered_concurrently(fake_backend: FakeBackend) -> None:
    barrier = threading.Barrier(3, timeout=5)

    def meet(call: Any) -> str:
        barrier.wait()
        return "positive"

    fake_backend.chat_script = meet
    estimator = ToyClassifier(max_concurrency=3).fit(None, ["negative", "positive"])

    assert list(estimator.predict(["a", "b", "c"])) == ["positive"] * 3


def test_max_concurrency_bounds_the_batch(fake_backend: FakeBackend) -> None:
    """Three samples that must meet never can, because only two ever run."""
    barrier = threading.Barrier(3, timeout=0.2)

    def meet(call: Any) -> str:
        barrier.wait()
        return "positive"

    fake_backend.chat_script = meet
    estimator = ToyClassifier(max_concurrency=2).fit(None, ["negative", "positive"])

    with pytest.warns(UserWarning, match="3 of 3 samples"):
        predictions = estimator.predict(["a", "b", "c"])

    assert list(predictions) == [None, None, None]


def test_config_context_reaches_the_worker_threads(fake_backend: FakeBackend) -> None:
    fake_backend.chat_script = "positive"
    estimator = ToyClassifier(max_concurrency=4).fit(None, ["negative", "positive"])

    with config_context(fake_api_key="ctx-key"):
        estimator.predict(["a", "b", "c", "d"])

    assert {call.api_key for call in fake_backend.calls} == {"ctx-key"}


def test_threads_predicting_on_one_estimator_keep_their_own_credentials(
    fake_backend: FakeBackend,
) -> None:
    fake_backend.chat_script = "positive"
    estimator = ToyClassifier().fit(None, ["negative", "positive"])
    seen: dict[str, set[str | None]] = {}
    start = threading.Barrier(2, timeout=5)

    def predict_with(key: str) -> None:
        with config_context(fake_api_key=key):
            start.wait()
            estimator.predict([key])
        seen[key] = {
            call.api_key for call in fake_backend.calls if call.prompt.endswith(key)
        }

    threads = [
        threading.Thread(target=predict_with, args=(key,)) for key in ("k1", "k2")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert seen == {"k1": {"k1"}, "k2": {"k2"}}
    assert estimator.usage_.total.requests == 2


def test_progress_is_shown_only_when_asked(
    fake_backend: FakeBackend, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_backend.chat_script = "positive"
    texts = ["a", "b", "c"]

    ToyClassifier(verbose=0).fit(None, ["positive"]).predict(texts)
    assert capsys.readouterr().err == ""

    ToyClassifier(verbose=1).fit(None, ["positive"]).predict(texts)
    assert "ToyClassifier" in capsys.readouterr().err


# --- error policy ------------------------------------------------------------


def fail_one_sample(failing: str) -> Any:
    def answer(call: Any) -> str:
        text = answer_with_the_input(call)
        if text == failing:
            raise RuntimeError("the provider rejected the request")
        return text

    return answer


def test_fill_policy_reports_every_failed_sample(fake_backend: FakeBackend) -> None:
    fake_backend.chat_script = fail_one_sample("b")
    estimator = ToyClassifier(default_label="unknown").fit(None, ["a", "b", "c"])

    with pytest.warns(UserWarning, match="1 of 3 samples"):
        predictions = estimator.predict(["a", "b", "c"])

    assert list(predictions) == ["a", "unknown", "c"]
    assert estimator.errors_ == (
        SampleError(index=1, error="RuntimeError: the provider rejected the request"),
    )
    assert estimator.usage_.last.failures == 1


@pytest.mark.parametrize(
    "run",
    [
        lambda X: ToyClassifier().fit(None, ["a", "b"]).predict(X),
        lambda X: ToyRewriter(model=FAKE_MODEL, verbose=0).transform(X),
    ],
    ids=["predict", "transform"],
)
def test_the_fill_warning_points_at_the_calling_code(
    run: Any, fake_backend: FakeBackend
) -> None:
    """sklearn wraps transform, so a fixed stacklevel would name its shim instead."""
    fake_backend.chat_script = fail_one_sample("b")

    with pytest.warns(UserWarning) as caught:
        run(["a", "b"])

    assert caught[0].filename == __file__


def test_fill_values_are_not_shared_between_samples(fake_backend: FakeBackend) -> None:
    fake_backend.embedding_script = RuntimeError("the provider rejected the request")
    vectorizer = ToyVectorizer(model=FAKE_MODEL)

    with pytest.warns(UserWarning, match="2 of 2 samples"):
        vectorizer.transform(["one", "two"])

    assert [error.index for error in vectorizer.errors_] == [0, 1]


def test_raise_policy_aborts_after_the_retries_are_exhausted(
    fake_backend: FakeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    fake_backend.chat_script = FakeTransientError("500 from the provider")
    estimator = ToyClassifier(on_error="raise", max_concurrency=1).fit(None, ["a", "b"])

    with pytest.raises(FakeTransientError, match="500 from the provider"):
        estimator.predict(["a", "b", "c"])

    assert sleeps[:3] == [0.5, 1.0, 2.0]  # backed off before the batch gave up


def test_an_aborted_batch_reports_what_it_spent_and_drops_the_old_errors(
    fake_backend: FakeBackend,
) -> None:
    """An abort must not leave the previous operation's report standing."""
    fake_backend.chat_script = fail_one_sample("b")
    estimator = ToyClassifier(max_concurrency=1).fit(None, ["a", "b", "c"])

    with pytest.warns(UserWarning):
        estimator.predict(["b"])
    assert [error.index for error in estimator.errors_] == [0]

    estimator.set_params(on_error="raise")
    with pytest.raises(RuntimeError, match="the provider rejected the request"):
        estimator.predict(["a", "c", "b"])

    assert estimator.errors_ == ()
    assert estimator.usage_.last.requests == 2  # what the aborted batch paid for
    assert estimator.usage_.total.requests == 2


def test_reports_describe_the_most_recent_operation_only(
    fake_backend: FakeBackend,
) -> None:
    fake_backend.chat_script = fail_one_sample("b")
    estimator = ToyClassifier().fit(None, ["a", "b"])

    with pytest.warns(UserWarning):
        estimator.predict(["b"])
    estimator.predict(["a"])

    assert estimator.errors_ == ()
    assert estimator.usage_.last.failures == 0
    assert estimator.usage_.total.failures == 1


# --- usage and telemetry -----------------------------------------------------


def test_usage_is_reported_per_operation_and_cumulatively(
    fake_backend: FakeBackend,
) -> None:
    fake_backend.chat_script = "positive"
    estimator = ToyClassifier().fit(None, ["negative", "positive"])

    assert estimator.usage_ == UsageReport()

    estimator.predict(["a"] * 10)
    first = estimator.usage_.last
    estimator.predict(["a"] * 2)

    assert first.requests == 10
    assert first.output_tokens == 10  # the scripted answer is a single word
    assert first.input_tokens == 10 * len(fake_backend.prompts[0].split())
    assert first.cost is None
    assert estimator.usage_.last.requests == 2
    assert estimator.usage_.total.requests == 12
    assert estimator.usage_.total.input_tokens == first.input_tokens * 1.2


def test_usage_records_add_up_and_keep_costs_optional() -> None:
    priced = UsageRecord(requests=1, input_tokens=2, output_tokens=3, cost=0.5)
    unpriced = UsageRecord(requests=1, failures=1)

    assert priced + unpriced == UsageRecord(
        requests=2, input_tokens=2, output_tokens=3, failures=1, cost=0.5
    )
    assert (unpriced + unpriced).cost is None
    assert (priced + priced).cost == 1.0


def test_costs_are_estimated_from_the_configured_prices(
    fake_backend: FakeBackend,
) -> None:
    fake_backend.chat_script = "positive"
    estimator = ToyClassifier().fit(None, ["negative", "positive"])

    with config_context(
        pricing={FAKE_MODEL: {"input": 1_000_000, "output": 2_000_000}}
    ):
        estimator.predict(["a", "b"])

    usage = estimator.usage_.last
    assert usage.cost == pytest.approx(usage.input_tokens + 2 * usage.output_tokens)


def test_spans_are_exported_when_a_provider_is_configured(
    fake_backend: FakeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "_TRACER_PROVIDER", provider)

    fake_backend.chat_script = "positive"
    estimator = ToyClassifier().fit(None, ["negative", "positive"])
    with config_context(
        pricing={FAKE_MODEL: {"input": 1_000_000, "output": 2_000_000}}
    ):
        estimator.predict(["a"] * 10)

    spans = exporter.get_finished_spans()
    assert len(spans) == 10
    attributes: dict[str, Any] = dict(spans[0].attributes or {})
    assert spans[0].name == "chat m"
    assert attributes["gen_ai.operation.name"] == "chat"
    assert attributes["gen_ai.provider.name"] == "fake"
    assert attributes["gen_ai.request.model"] == "m"
    assert attributes["gen_ai.usage.input_tokens"] > 0
    assert attributes["gen_ai.usage.output_tokens"] == 1
    assert attributes["gen_ai.usage.cost"] > 0


def test_no_span_machinery_without_a_configured_provider(
    fake_backend: FakeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("opentelemetry")
    from opentelemetry import trace

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a tracer was requested without a configured provider")

    monkeypatch.setattr(trace, "get_tracer", fail)
    fake_backend.chat_script = "positive"

    predictions = ToyClassifier().fit(None, ["positive"]).predict(["a"])

    assert list(predictions) == ["positive"]


def test_predictions_work_without_opentelemetry_installed(
    fake_backend: FakeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "opentelemetry", None)
    fake_backend.chat_script = "positive"

    predictions = ToyClassifier().fit(None, ["positive"]).predict(["a"])

    assert list(predictions) == ["positive"]


# --- credentials in serialized forms -----------------------------------------


def test_pickling_strips_the_connection_config(fake_backend: FakeBackend) -> None:
    fake_backend.chat_script = "positive"
    estimator = ToyClassifier(
        api_key="sk-secret", base_url="http://proxy/v1", model_params={"top_p": 0.3}
    ).fit(None, ["negative", "positive"])

    payload = pickle.dumps(estimator)
    restored = pickle.loads(payload)

    assert b"sk-secret" not in payload
    assert restored.api_key is None
    assert restored.base_url is None
    assert restored.classes_ == ["negative", "positive"]
    assert restored.model_params == {"top_p": 0.3}

    with config_context(fake_api_key="sk-new"):
        restored.predict(["I love it"])
    assert fake_backend.calls[-1].api_key == "sk-new"


def test_a_stripped_estimator_needs_a_credential_again(
    fake_backend: FakeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing key is a configuration error, never N filled sample failures."""
    fake_backend.chat_script = "positive"
    estimator = ToyClassifier(api_key="sk-secret").fit(None, ["negative", "positive"])
    restored = pickle.loads(pickle.dumps(estimator))
    monkeypatch.delenv("FAKE_API_KEY")

    with pytest.raises(MissingCredentialError, match="FAKE_API_KEY"):
        restored.predict(["I love it"])

    assert restored.errors_ == ()


# --- credential discovery ----------------------------------------------------


def test_an_estimator_reports_its_own_provider_and_alias() -> None:
    assert ToyClassifier().llm_requirements() == (Requirement("fake", None),)
    assert ToyClassifier(alias="team_a").llm_requirements() == (
        Requirement("fake", "team_a"),
    )


def test_requirements_include_fit_created_components(
    fake_backend: FakeBackend,
) -> None:
    retriever = ToyRetriever().fit(["a text"])

    assert retriever.llm_requirements() == (
        Requirement("fake", None),
        Requirement("openai", None),
    )


def test_requirements_are_deduplicated(fake_backend: FakeBackend) -> None:
    retriever = ToyRetriever(vectorizer=ToyVectorizer(model=FAKE_MODEL)).fit(["a text"])

    assert retriever.llm_requirements() == (Requirement("fake", None),)


def test_requirements_reach_estimators_inside_containers(
    fake_backend: FakeBackend,
) -> None:
    ensemble = ToyRetriever()
    ensemble.estimators_ = [
        ToyClassifier(alias="fast"),
        ToyVectorizer(model="google::gemini-embedding-001"),
    ]

    assert ensemble.llm_requirements() == (
        Requirement("fake", None),
        Requirement("fake", "fast"),
        Requirement("google", None),
    )


def test_fit_only_estimators_declare_nothing_of_their_own() -> None:
    assert ToyFitOnlyEstimator().llm_requirements() == ()
