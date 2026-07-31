from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx
import litellm
import pytest
from litellm import EmbeddingResponse, ModelResponse
from litellm.types.utils import Usage as LiteLLMUsage

from skllm.backends import BackendError, Connection, Message, Usage
from skllm.backends.litellm import LiteLLMBackend, _supports_response_schema
from skllm.config import config_context

MODEL = "vertex_ai/gemini-2.5-pro"
PROMPT = [Message("system", "classify"), Message("user", "I love it")]
LABEL_SCHEMA = {
    "type": "object",
    "properties": {"label": {"enum": ["positive", "negative"]}},
    "required": ["label"],
}


def completion(
    content: str | None = "positive",
    *,
    prompt_tokens: int = 11,
    completion_tokens: int = 7,
) -> ModelResponse:
    return ModelResponse(
        id="chatcmpl-1",
        model=MODEL,
        choices=[
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )


def embeddings(*vectors: list[float], prompt_tokens: int = 5) -> EmbeddingResponse:
    return EmbeddingResponse(
        model="text-embedding-004",
        data=[
            {"embedding": vector, "index": index, "object": "embedding"}
            for index, vector in reversed(list(enumerate(vectors)))
        ],
        usage=LiteLLMUsage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens),
    )


def api_error(
    kind: type[Any], status: int, *, headers: dict[str, str] | None = None
) -> Any:
    request = httpx.Request("POST", "https://litellm.example/chat")
    response = httpx.Response(status, request=request, headers=headers or {})
    return kind(
        message=f"Error code: {status}",
        model=MODEL,
        llm_provider="vertex_ai",
        response=response,
    )


class StubLiteLLM:
    """Records the calls litellm would have translated into provider requests."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.answers: list[Any] = []
        self.fallback: Any = completion()

    def __call__(self, **request: Any) -> Any:
        self.requests.append(request)
        answer = self.answers.pop(0) if self.answers else self.fallback
        if isinstance(answer, BaseException):
            raise answer
        return answer


@pytest.fixture(autouse=True)
def clear_capability_cache() -> Iterator[None]:
    _supports_response_schema.cache_clear()
    yield
    _supports_response_schema.cache_clear()


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> StubLiteLLM:
    stub = StubLiteLLM()
    monkeypatch.setattr(litellm, "completion", stub)
    monkeypatch.setattr(litellm, "embedding", stub)
    return stub


@pytest.fixture
def schema_support(monkeypatch: pytest.MonkeyPatch) -> Any:
    def set_support(supported: bool) -> None:
        monkeypatch.setattr(
            litellm, "supports_response_schema", lambda model, **_: supported
        )

    return set_support


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr("time.sleep", recorded.append)
    return recorded


def test_the_model_id_reaches_litellm_verbatim(stub: StubLiteLLM) -> None:
    response = LiteLLMBackend().chat(MODEL, PROMPT)

    request = stub.requests[0]
    assert request["model"] == MODEL
    assert request["messages"] == [
        {"role": "system", "content": "classify"},
        {"role": "user", "content": "I love it"},
    ]
    assert request["temperature"] == 0
    assert request["num_retries"] == 0  # scikit-llm owns the retry policy
    assert response.content == "positive"
    assert response.usage == Usage(requests=1, input_tokens=11, output_tokens=7)


def test_no_credential_is_required(stub: StubLiteLLM) -> None:
    """litellm reads each provider's own env vars, so skllm demands no key."""
    LiteLLMBackend().chat(MODEL, PROMPT)

    assert "api_key" not in stub.requests[0]


def test_supplied_connection_settings_are_forwarded(stub: StubLiteLLM) -> None:
    with config_context(litellm_api_key="sk-ctx", timeout=12.0):
        LiteLLMBackend().chat(
            MODEL, PROMPT, connection=Connection(base_url="http://proxy/v1")
        )

    request = stub.requests[0]
    assert request["api_key"] == "sk-ctx"
    assert request["api_base"] == "http://proxy/v1"
    assert request["timeout"] == 12.0


def test_model_params_may_shadow_a_connection_setting(stub: StubLiteLLM) -> None:
    """timeout is a legal litellm request arg, so an overlap must not be a TypeError."""
    with config_context(timeout=12.0):
        LiteLLMBackend().chat(MODEL, PROMPT, params={"timeout": 30})

    assert stub.requests[0]["timeout"] == 30


def test_structured_calls_use_native_schemas_where_litellm_reports_support(
    stub: StubLiteLLM, schema_support: Any
) -> None:
    schema_support(True)
    stub.fallback = completion('{"label": "positive"}')

    response = LiteLLMBackend().structured(MODEL, PROMPT, LABEL_SCHEMA)

    request = stub.requests[0]
    assert request["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": LABEL_SCHEMA, "strict": True},
    }
    assert request["messages"] == [
        {"role": "system", "content": "classify"},
        {"role": "user", "content": "I love it"},
    ]
    assert response.content == {"label": "positive"}


def test_structured_calls_fall_back_to_a_prompt_embedded_schema(
    stub: StubLiteLLM, schema_support: Any
) -> None:
    schema_support(False)
    stub.fallback = completion('Sure! {"label": "positive"}')

    response = LiteLLMBackend().structured(MODEL, PROMPT, LABEL_SCHEMA)

    # The schema folds into the last turn: an extra turn would put two user
    # messages in a row, which some of litellm's providers reject.
    sent = stub.requests[0]["messages"]
    assert "response_format" not in stub.requests[0]
    assert [message["role"] for message in sent] == ["system", "user"]
    assert sent[-1]["content"].startswith("I love it")
    assert json.dumps(LABEL_SCHEMA, indent=2) in sent[-1]["content"]
    assert response.content == {"label": "positive"}


def test_capability_lookups_happen_once_per_model(
    stub: StubLiteLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[str] = []

    def record(model: str, **_: Any) -> bool:
        asked.append(model)
        return True

    monkeypatch.setattr(litellm, "supports_response_schema", record)
    stub.fallback = completion('{"label": "positive"}')
    backend = LiteLLMBackend()

    backend.structured(MODEL, PROMPT, LABEL_SCHEMA)
    backend.structured(MODEL, PROMPT, LABEL_SCHEMA)
    backend.structured("openai/gpt-5", PROMPT, LABEL_SCHEMA)

    assert asked == [MODEL, "openai/gpt-5"]


def test_an_unknown_model_falls_back_instead_of_failing(
    stub: StubLiteLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unknown(model: str, **_: Any) -> bool:
        raise ValueError(f"litellm does not know {model}")

    monkeypatch.setattr(litellm, "supports_response_schema", unknown)
    stub.fallback = completion('{"label": "positive"}')

    assert LiteLLMBackend().structured(MODEL, PROMPT, LABEL_SCHEMA).content == {
        "label": "positive"
    }
    assert "response_format" not in stub.requests[0]


def test_user_response_format_is_never_overridden(
    stub: StubLiteLLM, schema_support: Any
) -> None:
    schema_support(True)
    stub.fallback = completion('{"label": "positive"}')

    LiteLLMBackend().structured(
        MODEL, PROMPT, LABEL_SCHEMA, params={"response_format": {"type": "json_object"}}
    )

    assert stub.requests[0]["response_format"] == {"type": "json_object"}


def test_embeddings_keep_the_input_order(stub: StubLiteLLM) -> None:
    stub.fallback = embeddings([1.0, 0.0], [0.0, 1.0])

    response = LiteLLMBackend().embed("text-embedding-004", ["one", "two"])

    assert stub.requests[0]["input"] == ["one", "two"]
    assert response.content == [[1.0, 0.0], [0.0, 1.0]]
    assert response.usage == Usage(requests=1, input_tokens=5)


def test_truncated_embedding_batches_are_rejected(stub: StubLiteLLM) -> None:
    stub.fallback = embeddings([1.0, 0.0])

    with pytest.raises(BackendError, match="1 embeddings for 2 inputs"):
        LiteLLMBackend().embed("text-embedding-004", ["one", "two"])


def test_unknown_user_params_reach_litellm_verbatim(stub: StubLiteLLM) -> None:
    LiteLLMBackend().chat(MODEL, PROMPT, params={"nonexistent_param": 1})

    assert stub.requests[0]["nonexistent_param"] == 1


def test_a_rejected_temperature_is_stripped_once_and_remembered(
    stub: StubLiteLLM,
) -> None:
    stub.answers = [
        litellm.UnsupportedParamsError(
            message=f"{MODEL} does not support parameters: ['temperature']",
            model=MODEL,
            llm_provider="vertex_ai",
        )
    ]
    backend = LiteLLMBackend()

    assert backend.chat(MODEL, PROMPT).content == "positive"
    assert backend.chat(MODEL, PROMPT).content == "positive"

    assert [("temperature" in request) for request in stub.requests] == [
        True,
        False,
        False,
    ]


def test_unparseable_model_output_can_never_retract_an_injected_default(
    stub: StubLiteLLM, schema_support: Any
) -> None:
    """A salvage failure quotes the model, whose prose may name a param by chance."""
    schema_support(True)
    stub.answers = [completion("Sorry, temperature data is not supported for Berlin.")]
    backend = LiteLLMBackend()

    with pytest.raises(BackendError, match="not valid JSON"):
        backend.structured(MODEL, PROMPT, LABEL_SCHEMA)
    backend.chat(MODEL, PROMPT)

    assert [("temperature" in request) for request in stub.requests] == [True, True]


def test_transient_failures_are_retried(stub: StubLiteLLM, sleeps: list[float]) -> None:
    stub.answers = [
        api_error(litellm.RateLimitError, 429),
        api_error(litellm.ServiceUnavailableError, 503),
        api_error(litellm.InternalServerError, 500),
    ]

    assert LiteLLMBackend().chat(MODEL, PROMPT).content == "positive"
    assert len(stub.requests) == 4
    assert sleeps == [0.5, 1.0, 2.0]


def test_retry_after_hints_are_honored(stub: StubLiteLLM, sleeps: list[float]) -> None:
    stub.answers = [
        api_error(litellm.RateLimitError, 429, headers={"retry-after": "3"})
    ]

    LiteLLMBackend().chat(MODEL, PROMPT)

    assert sleeps == [3.0]


def test_authentication_failures_are_not_retried(
    stub: StubLiteLLM, sleeps: list[float]
) -> None:
    stub.answers = [api_error(litellm.AuthenticationError, 401)]

    with pytest.raises(litellm.AuthenticationError):
        LiteLLMBackend().chat(MODEL, PROMPT)

    assert len(stub.requests) == 1
    assert sleeps == []


@pytest.mark.parametrize(
    ("answer", "match"),
    [(completion(None), "empty message"), (ModelResponse(choices=[]), "no choices")],
)
def test_unusable_responses_raise(
    stub: StubLiteLLM, answer: ModelResponse, match: str
) -> None:
    stub.fallback = answer

    with pytest.raises(BackendError, match=match):
        LiteLLMBackend().chat(MODEL, PROMPT)
