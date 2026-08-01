from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openai import (
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from openai.types import CompletionUsage, CreateEmbeddingResponse, Embedding
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.create_embedding_response import Usage as EmbeddingUsage

from skllm.backends import BackendError, Connection, Message, Usage
from skllm.backends import openai as openai_backend
from skllm.backends.openai import (
    KEYLESS_API_KEY,
    OLLAMA_BASE_URL,
    OllamaBackend,
    OpenAIBackend,
)
from skllm.config import MissingCredentialError, config_context

PROMPT = [Message("system", "classify"), Message("user", "I love it")]
LABEL_SCHEMA = {
    "type": "object",
    "properties": {"label": {"enum": ["positive", "negative"]}},
    "required": ["label"],
    "additionalProperties": False,
}


def completion(
    content: str | None = "positive",
    *,
    refusal: str | None = None,
    prompt_tokens: int = 11,
    completion_tokens: int = 7,
    choices: int = 1,
) -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-1",
        created=0,
        model="gpt-5",
        object="chat.completion",
        choices=[
            Choice(
                index=index,
                finish_reason="stop",
                message=ChatCompletionMessage(
                    role="assistant", content=content, refusal=refusal
                ),
            )
            for index in range(choices)
        ],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def embeddings(
    *vectors: list[float], prompt_tokens: int = 5
) -> CreateEmbeddingResponse:
    return CreateEmbeddingResponse(
        model="text-embedding-3-small",
        object="list",
        data=[
            Embedding(embedding=vector, index=index, object="embedding")
            for index, vector in reversed(list(enumerate(vectors)))
        ],
        usage=EmbeddingUsage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens),
    )


def api_error(
    kind: type[Any],
    status: int,
    *,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> Any:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status, request=request, headers=headers or {})
    # The SDK unwraps the body's "error" object and folds it into the message.
    message = f"Error code: {status}" + (f" - {{'error': {body}}}" if body else "")
    return kind(message, response=response, body=body)


def unsupported(param: str) -> Any:
    return api_error(
        BadRequestError,
        400,
        body={
            "message": f"Unsupported parameter: '{param}' is not supported with this model.",
            "type": "invalid_request_error",
            "param": param,
            "code": "unsupported_parameter",
        },
    )


class StubOpenAI:
    """Stands in for ``openai.OpenAI``, recording clients and requests."""

    def __init__(self) -> None:
        self.clients: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self.answers: list[Any] = []
        self.fallback: Any = completion()

    def __call__(self, **options: Any) -> SimpleNamespace:
        self.clients.append(options)
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=self._respond)),
            embeddings=SimpleNamespace(create=self._respond),
        )

    def _respond(self, **request: Any) -> Any:
        self.requests.append(request)
        answer = self.answers.pop(0) if self.answers else self.fallback
        if isinstance(answer, BaseException):
            raise answer
        return answer


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> StubOpenAI:
    stub = StubOpenAI()
    monkeypatch.setattr(openai_backend, "OpenAI", stub)
    return stub


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr("time.sleep", recorded.append)
    return recorded


@pytest.fixture
def api_key(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    return "sk-env"


def test_chat_sends_the_prompt_and_reports_usage(
    stub: StubOpenAI, api_key: str
) -> None:
    response = OpenAIBackend().chat("gpt-5.6-sol", PROMPT)

    request = stub.requests[0]
    assert request["model"] == "gpt-5.6-sol"
    assert request["messages"] == [
        {"role": "system", "content": "classify"},
        {"role": "user", "content": "I love it"},
    ]
    assert request["temperature"] == 0
    assert request["reasoning_effort"] == "none"
    assert response.content == "positive"
    assert response.usage == Usage(requests=1, input_tokens=11, output_tokens=7)


def test_user_params_replace_the_injected_defaults(
    stub: StubOpenAI, api_key: str
) -> None:
    OpenAIBackend().chat(
        "gpt-5",
        PROMPT,
        params={"reasoning_effort": "high", "max_completion_tokens": 64},
    )

    request = stub.requests[0]
    assert request["reasoning_effort"] == "high"
    assert request["max_completion_tokens"] == 64
    assert request["temperature"] == 0


def test_structured_calls_use_native_json_schema(
    stub: StubOpenAI, api_key: str
) -> None:
    stub.fallback = completion('```json\n{"label": "positive"}\n```')

    response = OpenAIBackend().structured("gpt-5", PROMPT, LABEL_SCHEMA)

    assert stub.requests[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": LABEL_SCHEMA, "strict": True},
    }
    assert response.content == {"label": "positive"}


def test_structured_calls_claim_strict_only_for_closed_schemas(
    stub: StubOpenAI, api_key: str
) -> None:
    """A user's extraction schema is rarely closed, and strict rejects the rest.

    Claiming it anyway would fail the request before the model saw the schema,
    so the schema still travels — just without the grammar constraint.
    """
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    stub.fallback = completion('{"name": "Ada Lovelace"}')

    response = OpenAIBackend().structured("gpt-5", PROMPT, schema)

    assert stub.requests[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": schema, "strict": False},
    }
    assert response.content == {"name": "Ada Lovelace"}


def test_embeddings_keep_the_input_order(stub: StubOpenAI, api_key: str) -> None:
    stub.fallback = embeddings([1.0, 0.0], [0.0, 1.0])

    response = OpenAIBackend().embed(
        "text-embedding-3-small", ["one", "two"], params={"dimensions": 2}
    )

    assert stub.requests[0]["input"] == ["one", "two"]
    assert stub.requests[0]["dimensions"] == 2
    assert "temperature" not in stub.requests[0]
    assert response.content == [[1.0, 0.0], [0.0, 1.0]]
    assert response.usage == Usage(requests=1, input_tokens=5, output_tokens=0)


def test_truncated_embedding_batches_are_rejected(
    stub: StubOpenAI, api_key: str
) -> None:
    stub.fallback = embeddings([1.0, 0.0])

    with pytest.raises(BackendError, match="1 embeddings for 2 inputs"):
        OpenAIBackend().embed("text-embedding-3-small", ["one", "two"])


def test_the_client_carries_the_resolved_connection(stub: StubOpenAI) -> None:
    with config_context(
        openai_api_key="sk-ctx",
        timeout=12.0,
        max_retries=5,
        client_options={"openai": {"organization": "org-1"}},
    ):
        OpenAIBackend().chat(
            "gpt-5", PROMPT, connection=Connection(base_url="http://x/v1")
        )

    assert stub.clients == [
        {
            "api_key": "sk-ctx",
            "base_url": "http://x/v1",
            "timeout": 12.0,
            "max_retries": 0,  # scikit-llm owns retries
            "organization": "org-1",
        }
    ]


def test_clients_are_reused_per_connection(stub: StubOpenAI, api_key: str) -> None:
    backend = OpenAIBackend()

    backend.chat("gpt-5", PROMPT)
    backend.chat("gpt-5", PROMPT)
    backend.chat("gpt-5", PROMPT, connection=Connection(api_key="sk-other"))

    assert [client["api_key"] for client in stub.clients] == ["sk-env", "sk-other"]


def test_base_url_layers_route_identically(stub: StubOpenAI, api_key: str) -> None:
    with config_context(openai_base_url="http://localhost:8000/v1"):
        OpenAIBackend().chat("my-model", PROMPT)

    assert stub.clients[0]["base_url"] == "http://localhost:8000/v1"
    assert stub.requests[0]["model"] == "my-model"


def test_a_missing_key_is_reported_before_any_client_is_built(stub: StubOpenAI) -> None:
    with pytest.raises(MissingCredentialError, match="OPENAI_API_KEY"):
        OpenAIBackend().chat("gpt-5", PROMPT)

    assert stub.clients == []


def test_env_fallback_can_be_disabled(stub: StubOpenAI, api_key: str) -> None:
    with config_context(allow_env_fallback=False):
        with pytest.raises(MissingCredentialError, match="fallback is disabled"):
            OpenAIBackend().chat("gpt-5", PROMPT)

        with config_context(openai_api_key="sk-ctx"):
            OpenAIBackend().chat("gpt-5", PROMPT)

    assert stub.clients[0]["api_key"] == "sk-ctx"


def test_ollama_runs_keyless_against_its_own_endpoint(stub: StubOpenAI) -> None:
    OllamaBackend().chat("llama3", PROMPT)

    assert stub.clients[0]["base_url"] == OLLAMA_BASE_URL
    assert stub.clients[0]["api_key"] == KEYLESS_API_KEY
    assert stub.requests[0]["temperature"] == 0
    assert "reasoning_effort" not in stub.requests[0]


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("http://ollama:11434", "http://ollama:11434/v1"),
        ("ollama:11434", "http://ollama:11434/v1"),
        ("http://ollama:11434/", "http://ollama:11434/v1"),
        ("http://ollama:11434/v1", "http://ollama:11434/v1"),
        # A port-less host means Ollama's own default port, not 80.
        ("ollama", "http://ollama:11434/v1"),
        ("127.0.0.1", "http://127.0.0.1:11434/v1"),
        ("http://ollama", "http://ollama:11434/v1"),
        # An https host is a fronted deployment; leave its implicit 443 alone.
        ("https://ollama.example.com", "https://ollama.example.com/v1"),
    ],
)
def test_ollama_host_becomes_an_openai_compatible_url(
    stub: StubOpenAI, monkeypatch: pytest.MonkeyPatch, host: str, expected: str
) -> None:
    monkeypatch.setenv("OLLAMA_HOST", host)

    OllamaBackend().chat("llama3", PROMPT)

    assert stub.clients[0]["base_url"] == expected


def test_a_rejected_temperature_is_stripped_once_and_remembered(
    stub: StubOpenAI, api_key: str
) -> None:
    stub.answers = [unsupported("temperature")]
    backend = OpenAIBackend()

    assert backend.chat("some-future-reasoning-model", PROMPT).content == "positive"
    assert backend.chat("some-future-reasoning-model", PROMPT).content == "positive"

    assert "temperature" in stub.requests[0]
    assert [("temperature" in request) for request in stub.requests] == [
        True,
        False,
        False,
    ]
    assert all(request["reasoning_effort"] == "none" for request in stub.requests)


def test_a_rejected_user_temperature_surfaces_the_provider_error(
    stub: StubOpenAI, api_key: str
) -> None:
    stub.answers = [unsupported("temperature")]

    with pytest.raises(BadRequestError):
        OpenAIBackend().chat(
            "some-future-reasoning-model", PROMPT, params={"temperature": 0}
        )

    assert len(stub.requests) == 1


def test_unsupported_params_are_recognized_without_a_param_field(
    stub: StubOpenAI, api_key: str
) -> None:
    stub.answers = [
        api_error(
            BadRequestError,
            400,
            body={"message": "reasoning_effort is not supported with this model."},
        )
    ]

    OpenAIBackend().chat("gpt-4o", PROMPT)

    assert "reasoning_effort" in stub.requests[0]
    assert "reasoning_effort" not in stub.requests[1]


def test_other_bad_requests_are_not_retried(stub: StubOpenAI, api_key: str) -> None:
    stub.answers = [api_error(BadRequestError, 400, body={"message": "context length"})]

    with pytest.raises(BadRequestError):
        OpenAIBackend().chat("gpt-5", PROMPT)

    assert len(stub.requests) == 1


def test_unknown_user_params_reach_the_provider_verbatim(
    stub: StubOpenAI, api_key: str
) -> None:
    stub.answers = [unsupported("nonexistent_param")]

    with pytest.raises(BadRequestError):
        OpenAIBackend().chat("gpt-5", PROMPT, params={"nonexistent_param": 1})

    assert stub.requests[0]["extra_body"] == {"nonexistent_param": 1}
    assert len(stub.requests) == 1


@pytest.mark.parametrize(
    ("params", "expected_body"),
    [
        ({"num_ctx": 8192}, {"num_ctx": 8192}),
        ({"nonexistent_param": 1}, {"nonexistent_param": 1}),
        ({"extra_body": {"a": 1}, "min_p": 0.1}, {"a": 1, "min_p": 0.1}),
    ],
)
def test_params_the_sdk_does_not_name_still_reach_the_wire(
    monkeypatch: pytest.MonkeyPatch,
    api_key: str,
    params: dict[str, Any],
    expected_body: dict[str, Any],
) -> None:
    """The SDK's create() has fixed kwargs, so unnamed params must ride along."""
    sent: dict[str, Any] = {}

    def handle(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return httpx.Response(200, json=completion().model_dump())

    client = OpenAI(
        api_key=api_key, http_client=httpx.Client(transport=httpx.MockTransport(handle))
    )
    monkeypatch.setattr(openai_backend, "OpenAI", lambda **options: client)

    assert OpenAIBackend().chat("llama3", PROMPT, params=params).content == "positive"

    for name, value in expected_body.items():
        assert sent[name] == value
    assert sent["temperature"] == 0


def test_seeded_reasoning_families_skip_the_wasted_round_trip(
    stub: StubOpenAI, api_key: str
) -> None:
    OpenAIBackend().chat("o3-mini", PROMPT)
    OpenAIBackend().chat("gpt-5.6-sol", PROMPT)

    assert "temperature" not in stub.requests[0]
    assert "reasoning_effort" not in stub.requests[0]
    assert stub.requests[1]["temperature"] == 0


def test_transient_failures_are_retried(
    stub: StubOpenAI, api_key: str, sleeps: list[float]
) -> None:
    stub.answers = [
        api_error(RateLimitError, 429),
        api_error(InternalServerError, 503),
        APITimeoutError(request=httpx.Request("POST", "https://api.openai.com")),
    ]

    assert OpenAIBackend().chat("gpt-5", PROMPT).content == "positive"
    assert len(stub.requests) == 4
    assert sleeps == [0.5, 1.0, 2.0]


def test_retry_after_hints_are_honored(
    stub: StubOpenAI, api_key: str, sleeps: list[float]
) -> None:
    stub.answers = [
        api_error(RateLimitError, 429, headers={"retry-after": "3"}),
        api_error(RateLimitError, 429, headers={"retry-after-ms": "1500"}),
        api_error(
            RateLimitError,
            429,
            headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"},
        ),
    ]

    OpenAIBackend().chat("gpt-5", PROMPT)

    assert sleeps == [3.0, 1.5, 2.0]


def test_authentication_failures_are_not_retried(
    stub: StubOpenAI, api_key: str, sleeps: list[float]
) -> None:
    stub.answers = [api_error(AuthenticationError, 401)]

    with pytest.raises(AuthenticationError):
        OpenAIBackend().chat("gpt-5", PROMPT)

    assert len(stub.requests) == 1
    assert sleeps == []


@pytest.mark.parametrize(
    ("response", "match"),
    [
        (completion(None, refusal="I cannot help with that"), "refused"),
        (completion(None), "empty message"),
        (completion(choices=0), "no choices"),
    ],
)
def test_unusable_responses_raise(
    stub: StubOpenAI, api_key: str, response: ChatCompletion, match: str
) -> None:
    stub.fallback = response

    with pytest.raises(BackendError, match=match):
        OpenAIBackend().chat("gpt-5", PROMPT)


def test_responses_without_usage_still_report_the_request(
    stub: StubOpenAI, api_key: str
) -> None:
    answer = completion()
    answer.usage = None
    stub.fallback = answer

    assert OpenAIBackend().chat("llama3", PROMPT).usage == Usage(requests=1)
