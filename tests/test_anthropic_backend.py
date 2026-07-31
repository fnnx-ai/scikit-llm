from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from anthropic import (
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    OverloadedError,
    RateLimitError,
)
from anthropic.types import Message as AnthropicMessage
from anthropic.types import TextBlock, ToolUseBlock
from anthropic.types import Usage as AnthropicUsage

from skllm.backends import BackendError, Connection, Message, Usage
from skllm.backends import anthropic as anthropic_backend
from skllm.backends.anthropic import (
    DEFAULT_MAX_TOKENS,
    RESPONSE_TOOL,
    AnthropicBackend,
)
from skllm.config import MissingCredentialError, config_context

PROMPT = [Message("system", "classify"), Message("user", "I love it")]
LABEL_SCHEMA = {
    "type": "object",
    "properties": {"label": {"enum": ["positive", "negative"]}},
    "required": ["label"],
}


def message(
    *blocks: Any, input_tokens: int = 11, output_tokens: int = 7
) -> AnthropicMessage:
    return AnthropicMessage(
        id="msg_1",
        model="claude-sonnet-5",
        role="assistant",
        type="message",
        stop_reason="end_turn",
        content=list(blocks) or [TextBlock(type="text", text="positive")],
        usage=AnthropicUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def tool_use(arguments: dict[str, Any]) -> ToolUseBlock:
    return ToolUseBlock(
        type="tool_use", id="toolu_1", name=RESPONSE_TOOL, input=arguments
    )


def api_error(
    kind: type[Any],
    status: int,
    *,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> Any:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status, request=request, headers=headers or {})
    return kind(f"Error code: {status}", response=response, body=body)


def unsupported(detail: str) -> Any:
    return api_error(
        BadRequestError,
        400,
        body={
            "type": "error",
            "error": {"type": "invalid_request_error", "message": detail},
        },
    )


class StubAnthropic:
    """Stands in for ``anthropic.Anthropic``, recording clients and requests."""

    def __init__(self) -> None:
        self.clients: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self.answers: list[Any] = []
        self.fallback: Any = message()

    def __call__(self, **options: Any) -> SimpleNamespace:
        self.clients.append(options)
        return SimpleNamespace(messages=SimpleNamespace(create=self._respond))

    def _respond(self, **request: Any) -> Any:
        self.requests.append(request)
        answer = self.answers.pop(0) if self.answers else self.fallback
        if isinstance(answer, BaseException):
            raise answer
        return answer


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> StubAnthropic:
    stub = StubAnthropic()
    monkeypatch.setattr(anthropic_backend, "Anthropic", stub)
    return stub


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr("time.sleep", recorded.append)
    return recorded


@pytest.fixture
def api_key(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    return "sk-ant-env"


def test_chat_lifts_the_system_prompt_out_of_the_turns(
    stub: StubAnthropic, api_key: str
) -> None:
    response = AnthropicBackend().chat("claude-sonnet-5", PROMPT)

    request = stub.requests[0]
    assert request["model"] == "claude-sonnet-5"
    assert request["system"] == "classify"
    assert request["messages"] == [{"role": "user", "content": "I love it"}]
    assert request["temperature"] == 0
    assert request["max_tokens"] == DEFAULT_MAX_TOKENS
    assert response.content == "positive"
    assert response.usage == Usage(requests=1, input_tokens=11, output_tokens=7)


def test_a_prompt_without_a_system_message_sends_no_system(
    stub: StubAnthropic, api_key: str
) -> None:
    AnthropicBackend().chat("claude-sonnet-5", [Message("user", "hi")])

    assert "system" not in stub.requests[0]


def test_user_params_replace_the_injected_defaults(
    stub: StubAnthropic, api_key: str
) -> None:
    AnthropicBackend().chat(
        "claude-sonnet-5", PROMPT, params={"max_tokens": 32, "temperature": 0.7}
    )

    assert stub.requests[0]["max_tokens"] == 32
    assert stub.requests[0]["temperature"] == 0.7


def test_structured_calls_force_the_response_tool(
    stub: StubAnthropic, api_key: str
) -> None:
    stub.fallback = message(tool_use({"label": "positive"}))

    response = AnthropicBackend().structured("claude-sonnet-5", PROMPT, LABEL_SCHEMA)

    request = stub.requests[0]
    assert request["tools"] == [
        {
            "name": RESPONSE_TOOL,
            "description": "Return the answer in the required structure.",
            "input_schema": LABEL_SCHEMA,
        }
    ]
    assert request["tool_choice"] == {"type": "tool", "name": RESPONSE_TOOL}
    assert response.content == {"label": "positive"}
    assert response.usage == Usage(requests=1, input_tokens=11, output_tokens=7)


def test_a_structured_answer_without_a_tool_call_raises(
    stub: StubAnthropic, api_key: str
) -> None:
    stub.fallback = message(TextBlock(type="text", text="I would say positive"))

    with pytest.raises(BackendError, match=RESPONSE_TOOL):
        AnthropicBackend().structured("claude-sonnet-5", PROMPT, LABEL_SCHEMA)


def test_text_blocks_are_joined_and_tool_blocks_ignored(
    stub: StubAnthropic, api_key: str
) -> None:
    stub.fallback = message(
        TextBlock(type="text", text="first"),
        tool_use({"label": "positive"}),
        TextBlock(type="text", text="second"),
    )

    assert AnthropicBackend().chat("claude-sonnet-5", PROMPT).content == "first\nsecond"


def test_an_answer_without_text_raises(stub: StubAnthropic, api_key: str) -> None:
    stub.fallback = message(tool_use({"label": "positive"}))

    with pytest.raises(BackendError, match="no text content"):
        AnthropicBackend().chat("claude-sonnet-5", PROMPT)


def test_embeddings_are_not_offered(api_key: str) -> None:
    with pytest.raises(BackendError, match="vectorizer"):
        AnthropicBackend().embed("claude-sonnet-5", ["one"])


def test_the_client_carries_the_resolved_connection(stub: StubAnthropic) -> None:
    with config_context(
        anthropic_api_key="sk-ant-ctx",
        timeout=12.0,
        client_options={"anthropic": {"default_headers": {"x-team": "a"}}},
    ):
        AnthropicBackend().chat(
            "claude-sonnet-5", PROMPT, connection=Connection(base_url="http://proxy/v1")
        )

    assert stub.clients == [
        {
            "api_key": "sk-ant-ctx",
            "base_url": "http://proxy/v1",
            "timeout": 12.0,
            "max_retries": 0,  # scikit-llm owns retries
            "default_headers": {"x-team": "a"},
        }
    ]


def test_clients_are_reused_per_connection(stub: StubAnthropic, api_key: str) -> None:
    backend = AnthropicBackend()

    backend.chat("claude-sonnet-5", PROMPT)
    backend.chat("claude-sonnet-5", PROMPT)
    backend.chat("claude-sonnet-5", PROMPT, connection=Connection(api_key="sk-other"))

    assert [client["api_key"] for client in stub.clients] == [api_key, "sk-other"]


def test_a_missing_key_is_reported_before_any_client_is_built(
    stub: StubAnthropic,
) -> None:
    with pytest.raises(MissingCredentialError, match="ANTHROPIC_API_KEY"):
        AnthropicBackend().chat("claude-sonnet-5", PROMPT)

    assert stub.clients == []


def test_params_the_sdk_does_not_name_ride_in_extra_body(
    stub: StubAnthropic, api_key: str
) -> None:
    AnthropicBackend().chat(
        "claude-sonnet-5", PROMPT, params={"top_k": 3, "nonexistent_param": 1}
    )

    assert stub.requests[0]["top_k"] == 3
    assert stub.requests[0]["extra_body"] == {"nonexistent_param": 1}


def test_a_rejected_temperature_is_stripped_once_and_remembered(
    stub: StubAnthropic, api_key: str
) -> None:
    stub.answers = [unsupported("temperature is not supported with this model")]
    backend = AnthropicBackend()

    assert backend.chat("claude-future", PROMPT).content == "positive"
    assert backend.chat("claude-future", PROMPT).content == "positive"

    assert [("temperature" in request) for request in stub.requests] == [
        True,
        False,
        False,
    ]


def test_thinking_suppresses_the_temperature_default_without_poisoning_the_model(
    stub: StubAnthropic, api_key: str
) -> None:
    """The API rejects temperature *with thinking* — a caller state, not a model one."""
    backend = AnthropicBackend()

    backend.chat("claude-opus-5", PROMPT, params={"thinking": {"type": "enabled"}})
    backend.chat("claude-opus-5", PROMPT, params={"thinking": {"type": "disabled"}})
    backend.chat("claude-opus-5", PROMPT)

    assert [("temperature" in request) for request in stub.requests] == [
        False,
        True,
        True,  # a later plain call keeps its determinism default
    ]


def test_only_the_provider_can_retract_an_injected_default(
    stub: StubAnthropic, api_key: str
) -> None:
    """A non-API error that happens to name a param must not disarm determinism."""
    stub.answers = [RuntimeError("temperature is not supported by this proxy")]
    backend = AnthropicBackend()

    with pytest.raises(RuntimeError):
        backend.chat("claude-sonnet-5", PROMPT)
    backend.chat("claude-sonnet-5", PROMPT)

    assert [("temperature" in request) for request in stub.requests] == [True, True]


def test_a_rejected_user_temperature_surfaces_the_provider_error(
    stub: StubAnthropic, api_key: str
) -> None:
    stub.answers = [unsupported("`temperature` may only be set to 1 ...")]

    with pytest.raises(BadRequestError):
        AnthropicBackend().chat("claude-opus-5", PROMPT, params={"temperature": 0})

    assert len(stub.requests) == 1


def test_other_bad_requests_are_not_retried(stub: StubAnthropic, api_key: str) -> None:
    stub.answers = [
        api_error(
            BadRequestError,
            400,
            body={"type": "error", "error": {"message": "prompt is too long"}},
        )
    ]

    with pytest.raises(BadRequestError):
        AnthropicBackend().chat("claude-sonnet-5", PROMPT)

    assert len(stub.requests) == 1


def test_transient_failures_are_retried(
    stub: StubAnthropic, api_key: str, sleeps: list[float]
) -> None:
    stub.answers = [
        api_error(RateLimitError, 429),
        api_error(OverloadedError, 529),
        api_error(InternalServerError, 503),
        APITimeoutError(request=httpx.Request("POST", "https://api.anthropic.com")),
    ]

    with config_context(max_retries=4):
        assert AnthropicBackend().chat("claude-sonnet-5", PROMPT).content == "positive"

    assert len(stub.requests) == 5
    assert sleeps == [0.5, 1.0, 2.0, 4.0]


def test_retry_after_hints_are_honored(
    stub: StubAnthropic, api_key: str, sleeps: list[float]
) -> None:
    stub.answers = [api_error(RateLimitError, 429, headers={"retry-after": "3"})]

    AnthropicBackend().chat("claude-sonnet-5", PROMPT)

    assert sleeps == [3.0]


def test_authentication_failures_are_not_retried(
    stub: StubAnthropic, api_key: str, sleeps: list[float]
) -> None:
    stub.answers = [api_error(AuthenticationError, 401)]

    with pytest.raises(AuthenticationError):
        AnthropicBackend().chat("claude-sonnet-5", PROMPT)

    assert len(stub.requests) == 1
    assert sleeps == []
