from __future__ import annotations

import pickle
import time
from typing import Any

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils import get_tags

from skllm.backends import BackendError
from skllm.core import DEFAULT_EMBEDDING_MODEL, UsageRecord
from skllm.vectorization import Vectorizer
from tests.fakes import EMBEDDING_DIM, FakeBackend, FakeCall

FAKE_MODEL = "fake::e"

REVIEWS = ["I love it", "It broke on day one"]


@pytest.fixture
def vectorizer(fake_backend: FakeBackend) -> Vectorizer:
    return Vectorizer(model=FAKE_MODEL, verbose=0)


def test_transform_returns_a_float_matrix_with_one_row_per_text(
    vectorizer: Vectorizer,
) -> None:
    vectors = vectorizer.transform(["one", "two", "three"])

    assert vectors.shape == (3, EMBEDDING_DIM)
    assert vectors.dtype == np.dtype(float)


def test_texts_are_embedded_in_batches(
    fake_backend: FakeBackend, vectorizer: Vectorizer
) -> None:
    vectorizer.set_params(batch_size=2, max_concurrency=1)

    vectorizer.transform(["a", "b", "c", "d", "e"])

    assert [call.texts for call in fake_backend.calls] == [
        ("a", "b"),
        ("c", "d"),
        ("e",),
    ]
    assert vectorizer.usage_.last.requests == 3


def test_a_batch_that_fits_in_one_request_makes_one(
    fake_backend: FakeBackend, vectorizer: Vectorizer
) -> None:
    vectorizer.transform(["a"] * 5)

    assert len(fake_backend.calls) == 1


def test_rows_keep_the_input_order_when_batches_finish_out_of_order(
    fake_backend: FakeBackend, vectorizer: Vectorizer
) -> None:
    def answer_shortest_last(call: FakeCall) -> list[list[float]]:
        time.sleep(0.02 * (5 - len(call.texts[0])))
        return [[float(len(text))] for text in call.texts]

    fake_backend.embedding_script = answer_shortest_last
    vectorizer.set_params(batch_size=1, max_concurrency=4)

    vectors = vectorizer.transform(["a", "bb", "ccc", "dddd"])

    assert vectors.tolist() == [[1.0], [2.0], [3.0], [4.0]]


def test_an_empty_batch_issues_no_requests(
    fake_backend: FakeBackend, vectorizer: Vectorizer
) -> None:
    vectorizer.transform(["a"])
    before = vectorizer.usage_.total
    fake_backend.calls.clear()

    vectors = vectorizer.transform([])

    assert vectors.shape == (0, 0)
    assert fake_backend.calls == []
    assert vectorizer.usage_.last == UsageRecord()
    assert vectorizer.usage_.total == before


# --- error behavior ----------------------------------------------------------


def test_a_failed_batch_raises_under_the_fill_policy(
    fake_backend: FakeBackend, vectorizer: Vectorizer
) -> None:
    """A zero-vector fill would poison every model downstream, silently."""
    fake_backend.embedding_script = RuntimeError("the provider rejected the request")

    assert vectorizer.on_error == "fill"
    with pytest.raises(RuntimeError, match="the provider rejected the request"):
        vectorizer.transform(["a"])

    assert vectorizer.errors_ == ()


def test_one_failed_batch_aborts_the_whole_transform(
    fake_backend: FakeBackend, vectorizer: Vectorizer
) -> None:
    def fail_the_third_text(call: FakeCall) -> list[list[float]]:
        if "c" in call.texts:
            raise RuntimeError("the provider rejected the request")
        return [[1.0] for _ in call.texts]

    fake_backend.embedding_script = fail_the_third_text
    vectorizer.set_params(batch_size=1, max_concurrency=1)

    with pytest.raises(RuntimeError, match="the provider rejected the request"):
        vectorizer.transform(["a", "b", "c"])

    assert vectorizer.usage_.last.requests == 2  # what the aborted batch paid for


def test_a_provider_without_embeddings_raises_at_first_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic")

    with pytest.raises(BackendError, match="does not provide embeddings"):
        Vectorizer(model="anthropic::claude-sonnet-5", verbose=0).transform(["a"])


@pytest.mark.parametrize("value", [0, -1, 1.5, None])
def test_invalid_batch_sizes_are_rejected(
    value: Any, fake_backend: FakeBackend
) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        Vectorizer(model=FAKE_MODEL, batch_size=value).transform(["a"])


# --- sklearn compliance ------------------------------------------------------


def test_the_default_model_is_the_openai_embedding_model() -> None:
    assert Vectorizer().model == DEFAULT_EMBEDDING_MODEL


def test_the_vectorizer_is_stateless(
    fake_backend: FakeBackend, vectorizer: Vectorizer
) -> None:
    assert not get_tags(vectorizer).requires_fit
    assert vectorizer.fit(["a text"]) is vectorizer
    assert fake_backend.calls == []  # fit only validates params

    assert vectorizer.transform(["a text"]).shape == (1, EMBEDDING_DIM)


def test_params_survive_a_clone() -> None:
    estimator = Vectorizer(model=FAKE_MODEL, api_key="sk-secret", batch_size=4)

    cloned = clone(estimator)

    assert cloned.get_params() == estimator.get_params()
    assert cloned.api_key == "sk-secret"
    assert vars(estimator) == estimator.get_params()


def test_pickling_drops_the_key(fake_backend: FakeBackend) -> None:
    payload = pickle.dumps(Vectorizer(model=FAKE_MODEL, api_key="sk-secret"))

    assert b"sk-secret" not in payload
    assert pickle.loads(payload).api_key is None


def test_the_vectorizer_feeds_a_classical_sklearn_estimator(
    fake_backend: FakeBackend,
) -> None:
    pipeline = Pipeline(
        [
            ("embed", Vectorizer(model=FAKE_MODEL, verbose=0)),
            ("classify", LogisticRegression()),
        ]
    )

    pipeline.fit(REVIEWS, ["positive", "negative"])

    assert list(pipeline.predict(["I love it"])) == ["positive"]
