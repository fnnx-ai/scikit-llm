from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx2
import pytest
from typesafe_sdk import (
    SystemOneResponse,
    TypeSafeAPIConnectionError,
    TypeSafeAuthenticationError,
    TypeSafeInternalServerError,
    TypeSafeRateLimitError,
)

from skllm.backends import (
    BackendError,
    Connection,
    Message,
    SystemOneBackend,
    Usage,
    get_backend,
)
from skllm.backends import typesafe as typesafe_backend
from skllm.backends.typesafe import DEFAULT_BASE_URL, TypeSafeBackend
from skllm.config import MissingCredentialError, config_context

STATE = "I was charged twice. Please help ASAP."
QUESTIONS: dict[str, dict[str, Any]] = {
    "tone": {
        "type": "choice",
        "instructions": "What is the tone?",
        "criteria": {"calm": None, "angry": None},
    },
    "urgent": {"type": "noul", "instructions": "Is this urgent?"},
}
ANSWERS: dict[str, dict[str, Any]] = {
    "tone": {
        "type": "choice",
        "choice": "angry",
        "confidence": 0.8,
        "probabilities": {"calm": 0.1, "angry": 0.9},
    },
    "urgent": {"type": "noul", "noul": 0.95},
}


def response(
    answers: dict[str, Any] = ANSWERS, input_tokens: int = 12, output_tokens: int = 3
) -> SystemOneResponse:
    # Validated from JSON, as the SDK does: score maps are keyed by strings there.
    return SystemOneResponse.model_validate_json(
        json.dumps(
            {
                "model": "jev-1.13.0",
                "answers": answers,
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            }
        )
    )


def api_error(
    kind: type[Any], status: int, headers: dict[str, str] | None = None
) -> Any:
    return kind(status, {"error": "failed"}, httpx2.Headers(headers or {}))


class StubClient:
    """Stands in for ``typesafe_sdk.TypeSafeClient``, recording clients and calls."""

    def __init__(self) -> None:
        self.clients: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self.answers: list[Any] = []
        self.fallback: Any = response()

    def __call__(self, **options: Any) -> SimpleNamespace:
        self.clients.append(options)
        return SimpleNamespace(system_one=self._respond)

    def _respond(self, state: Any, questions: Any, **options: Any) -> Any:
        self.requests.append({"state": state, "questions": questions, **options})
        answer = self.answers.pop(0) if self.answers else self.fallback
        if isinstance(answer, BaseException):
            raise answer
        return answer


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> StubClient:
    stub = StubClient()
    monkeypatch.setattr(typesafe_backend, "TypeSafeClient", stub)
    return stub


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr("time.sleep", recorded.append)
    return recorded


@pytest.fixture
def api_key(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-env")
    return "ts-env"


def test_the_registry_serves_a_system_one_adapter() -> None:
    backend = get_backend("typesafe")

    assert isinstance(backend, TypeSafeBackend)
    assert isinstance(backend, SystemOneBackend)


def test_the_state_and_questions_are_sent_as_given(
    stub: StubClient, api_key: str
) -> None:
    TypeSafeBackend().evaluate("jev-latest", STATE, QUESTIONS)

    assert stub.requests == [
        {
            "state": STATE,
            "questions": QUESTIONS,
            "model": "jev-latest",
            "extra_body": None,
        }
    ]


def test_answers_come_back_as_json_keyed_by_question(
    stub: StubClient, api_key: str
) -> None:
    result = TypeSafeBackend().evaluate("jev-latest", STATE, QUESTIONS)

    assert result.content == ANSWERS
    assert result.usage == Usage(requests=1, input_tokens=12, output_tokens=3)


def test_score_answers_keep_string_level_keys(stub: StubClient, api_key: str) -> None:
    score = {
        "type": "score",
        "score": 1.05,
        "confidence": 0.92,
        "legend": {"0": "Calm", "1": "Frustrated"},
        "probabilities": {"0": 0.0, "1": 1.0},
    }
    stub.fallback = response({"frustration": score})

    result = TypeSafeBackend().evaluate(
        "jev-latest",
        STATE,
        {"frustration": {"type": "score", "criteria": ["Calm", "Frustrated"]}},
    )

    assert result.content == {"frustration": score}


def test_a_question_left_unanswered_raises(stub: StubClient, api_key: str) -> None:
    """The SDK silently drops answer kinds it does not know."""
    stub.fallback = response({"tone": ANSWERS["tone"]})

    with pytest.raises(BackendError, match="urgent"):
        TypeSafeBackend().evaluate("jev-latest", STATE, QUESTIONS)


def test_model_params_ride_in_extra_body(stub: StubClient, api_key: str) -> None:
    TypeSafeBackend().evaluate("jev-latest", STATE, QUESTIONS, params={"beam_width": 4})

    assert stub.requests[0]["extra_body"] == {"beam_width": 4}


@pytest.mark.parametrize(
    "call",
    [
        lambda backend: backend.chat("jev-latest", [Message("user", "hi")]),
        lambda backend: backend.structured(
            "jev-latest", [Message("user", "hi")], {"type": "object"}
        ),
        lambda backend: backend.embed("jev-latest", ["hi"]),
    ],
    ids=["chat", "structured", "embed"],
)
def test_text_and_embedding_requests_are_refused(
    stub: StubClient, api_key: str, call: Any
) -> None:
    with pytest.raises(BackendError):
        call(TypeSafeBackend())

    assert stub.requests == []


def test_the_client_carries_the_resolved_connection(stub: StubClient) -> None:
    with config_context(timeout=7.0, client_options={"typesafe": {"headers": {}}}):
        TypeSafeBackend().evaluate(
            "jev-latest", STATE, QUESTIONS, connection=Connection(api_key="ts-own")
        )

    (options,) = stub.clients
    assert options["api_key"] == "ts-own"
    assert options["base_url"] == DEFAULT_BASE_URL
    assert options["timeout"] == 7.0
    assert options["retry"].max_retries == 0
    assert options["headers"] == {}


def test_a_configured_base_url_replaces_the_default(
    stub: StubClient, api_key: str
) -> None:
    with config_context(typesafe_base_url="https://gateway.example/typesafe"):
        TypeSafeBackend().evaluate("jev-latest", STATE, QUESTIONS)

    assert stub.clients[0]["base_url"] == "https://gateway.example/typesafe"


def test_clients_are_reused_per_connection(stub: StubClient, api_key: str) -> None:
    backend = TypeSafeBackend()
    backend.evaluate("jev-latest", STATE, QUESTIONS)
    backend.evaluate("jev-latest", STATE, QUESTIONS)

    assert len(stub.clients) == 1


def test_a_missing_key_is_reported_before_any_client_is_built(
    stub: StubClient,
) -> None:
    with pytest.raises(MissingCredentialError, match="TYPESAFE_API_KEY"):
        TypeSafeBackend().evaluate("jev-latest", STATE, QUESTIONS)

    assert stub.clients == []


def test_transient_failures_are_retried(
    stub: StubClient, api_key: str, sleeps: list[float]
) -> None:
    stub.answers = [
        api_error(TypeSafeRateLimitError, 429),
        api_error(TypeSafeInternalServerError, 529),
        TypeSafeAPIConnectionError("connection reset"),
    ]

    with config_context(max_retries=3):
        result = TypeSafeBackend().evaluate("jev-latest", STATE, QUESTIONS)

    assert result.content == ANSWERS
    assert len(stub.requests) == 4
    assert sleeps == [0.5, 1.0, 2.0]


@pytest.mark.parametrize(
    ("headers", "expected"),
    [({"retry-after-ms": "1500"}, 1.5), ({"retry-after": "3"}, 3.0)],
)
def test_retry_after_hints_are_honored(
    stub: StubClient,
    api_key: str,
    sleeps: list[float],
    headers: dict[str, str],
    expected: float,
) -> None:
    stub.answers = [api_error(TypeSafeRateLimitError, 429, headers)]

    TypeSafeBackend().evaluate("jev-latest", STATE, QUESTIONS)

    assert sleeps == [expected]


def test_authentication_failures_are_not_retried(
    stub: StubClient, api_key: str, sleeps: list[float]
) -> None:
    stub.answers = [api_error(TypeSafeAuthenticationError, 401)]

    with pytest.raises(TypeSafeAuthenticationError):
        TypeSafeBackend().evaluate("jev-latest", STATE, QUESTIONS)

    assert len(stub.requests) == 1
    assert sleeps == []
