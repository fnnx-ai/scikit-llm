from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from google.genai import errors, types
from pydantic import ValidationError

from skllm.backends import BackendError, Connection, Message, Usage
from skllm.backends import google as google_backend
from skllm.backends.google import GoogleBackend
from skllm.config import MissingCredentialError, config_context

PROMPT = [Message("system", "classify"), Message("user", "I love it")]
LABEL_SCHEMA = {
    "type": "object",
    "properties": {"label": {"enum": ["positive", "negative"]}},
    "required": ["label"],
}


def generated(
    text: str | None = "positive",
    *,
    prompt_tokens: int = 9,
    output_tokens: int = 4,
    thoughts_tokens: int = 0,
    finish_reason: types.FinishReason = types.FinishReason.STOP,
) -> types.GenerateContentResponse:
    content = (
        types.Content(role="model", parts=[types.Part(text=text)])
        if text is not None
        else None
    )
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=content, finish_reason=finish_reason)],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt_tokens,
            candidates_token_count=output_tokens,
            thoughts_token_count=thoughts_tokens,
        ),
    )


def embedded(*vectors: list[float]) -> types.EmbedContentResponse:
    return types.EmbedContentResponse(
        embeddings=[types.ContentEmbedding(values=vector) for vector in vectors]
    )


def api_error(
    kind: type[errors.APIError], status: int, *, headers: dict[str, str] | None = None
) -> errors.APIError:
    request = httpx.Request("POST", "https://generativelanguage.googleapis.com")
    return kind(
        status,
        {"error": {"message": f"failure {status}"}},
        httpx.Response(status, request=request, headers=headers or {}),
    )


class StubClient:
    """Stands in for ``genai.Client``, recording clients and requests."""

    def __init__(self) -> None:
        self.clients: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self.answers: list[Any] = []
        self.fallback: Any = generated()

    def __call__(self, **options: Any) -> SimpleNamespace:
        self.clients.append(options)
        return SimpleNamespace(
            models=SimpleNamespace(
                generate_content=self._respond, embed_content=self._respond
            )
        )

    def _respond(self, **request: Any) -> Any:
        self.requests.append(request)
        answer = self.answers.pop(0) if self.answers else self.fallback
        if isinstance(answer, BaseException):
            raise answer
        return answer


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> StubClient:
    stub = StubClient()
    monkeypatch.setattr(google_backend, "Client", stub)
    return stub


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr("time.sleep", recorded.append)
    return recorded


@pytest.fixture
def api_key(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("GOOGLE_API_KEY", "goog-env")
    return "goog-env"


def test_chat_sends_the_system_instruction_and_reports_usage(
    stub: StubClient, api_key: str
) -> None:
    response = GoogleBackend().chat("gemini-2.5-flash", PROMPT)

    request = stub.requests[0]
    assert request["model"] == "gemini-2.5-flash"
    assert request["config"].system_instruction == "classify"
    assert request["config"].temperature == 0
    assert [
        (part.text, content.role)
        for content in request["contents"]
        for part in content.parts or []
    ] == [("I love it", "user")]
    assert response.content == "positive"
    assert response.usage == Usage(requests=1, input_tokens=9, output_tokens=4)


def test_thinking_tokens_count_as_output(stub: StubClient, api_key: str) -> None:
    """Gemini reports them separately but bills them as output."""
    stub.fallback = generated(thoughts_tokens=900)

    response = GoogleBackend().chat("gemini-2.5-pro", PROMPT)

    assert response.usage == Usage(requests=1, input_tokens=9, output_tokens=904)


def test_no_reasoning_default_is_injected(stub: StubClient, api_key: str) -> None:
    """Gemini's thinking budget is per-model, so skllm never guesses one."""
    GoogleBackend().chat("gemini-2.5-flash", PROMPT)

    assert stub.requests[0]["config"].thinking_config is None


def test_structured_calls_use_the_native_response_schema(
    stub: StubClient, api_key: str
) -> None:
    stub.fallback = generated('```json\n{"label": "positive"}\n```')

    response = GoogleBackend().structured("gemini-2.5-flash", PROMPT, LABEL_SCHEMA)

    config = stub.requests[0]["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == LABEL_SCHEMA
    assert response.content == {"label": "positive"}


def test_user_params_pass_through_and_win(stub: StubClient, api_key: str) -> None:
    GoogleBackend().chat(
        "gemini-2.5-flash",
        PROMPT,
        params={"temperature": 0.7, "max_output_tokens": 64, "top_k": 5},
    )

    config = stub.requests[0]["config"]
    assert (config.temperature, config.max_output_tokens, config.top_k) == (0.7, 64, 5)


def test_an_unknown_param_is_rejected_rather_than_silently_dropped(
    stub: StubClient, api_key: str
) -> None:
    """skllm allow-lists nothing; this SDK's own config model does the rejecting."""
    with pytest.raises(ValidationError, match="nonexistent_param"):
        GoogleBackend().chat(
            "gemini-2.5-flash", PROMPT, params={"nonexistent_param": 1}
        )


def test_embeddings_keep_the_input_order(stub: StubClient, api_key: str) -> None:
    stub.fallback = embedded([1.0, 0.0], [0.0, 1.0])

    response = GoogleBackend().embed(
        "gemini-embedding-001", ["one", "two"], params={"output_dimensionality": 2}
    )

    request = stub.requests[0]
    assert request["contents"] == ["one", "two"]
    assert request["config"].output_dimensionality == 2
    assert response.content == [[1.0, 0.0], [0.0, 1.0]]
    assert response.usage == Usage(requests=1)


def test_truncated_embedding_batches_are_rejected(
    stub: StubClient, api_key: str
) -> None:
    stub.fallback = embedded([1.0, 0.0])

    with pytest.raises(BackendError, match="1 embeddings for 2 inputs"):
        GoogleBackend().embed("gemini-embedding-001", ["one", "two"])


def test_the_client_carries_the_resolved_connection(stub: StubClient) -> None:
    with config_context(google_api_key="goog-ctx", timeout=12.0):
        GoogleBackend().chat(
            "gemini-2.5-flash", PROMPT, connection=Connection(base_url="http://proxy")
        )

    options = stub.clients[0]
    assert options["api_key"] == "goog-ctx"
    assert options["http_options"].base_url == "http://proxy"
    assert options["http_options"].timeout == 12_000  # the SDK counts milliseconds


def test_client_options_reach_the_sdk_transport_config(stub: StubClient) -> None:
    """Headers live only under http_options on this SDK, so they must merge in."""
    with config_context(
        google_api_key="goog-ctx",
        client_options={"google": {"http_options": {"headers": {"x-team": "a"}}}},
    ):
        GoogleBackend().chat("gemini-2.5-flash", PROMPT)

    http_options = stub.clients[0]["http_options"]
    assert http_options.headers == {"x-team": "a"}
    assert http_options.timeout == 60_000  # skllm's own settings survive the merge


def test_the_specific_gemini_key_variable_wins(
    stub: StubClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "goog-env")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-env")

    GoogleBackend().chat("gemini-2.5-flash", PROMPT)

    assert stub.clients[0]["api_key"] == "gemini-env"


def test_a_missing_key_is_reported_before_any_client_is_built(
    stub: StubClient,
) -> None:
    with pytest.raises(
        MissingCredentialError, match="GEMINI_API_KEY or GOOGLE_API_KEY"
    ):
        GoogleBackend().chat("gemini-2.5-flash", PROMPT)

    assert stub.clients == []


@pytest.mark.parametrize(
    ("text", "match"),
    [(None, "no text"), ("", "no text")],
)
def test_answers_without_text_raise(
    stub: StubClient, api_key: str, text: str | None, match: str
) -> None:
    stub.fallback = generated(text, finish_reason=types.FinishReason.MAX_TOKENS)

    with pytest.raises(BackendError, match=match):
        GoogleBackend().chat("gemini-2.5-flash", PROMPT)


def test_an_answer_without_candidates_raises(stub: StubClient, api_key: str) -> None:
    stub.fallback = types.GenerateContentResponse(candidates=[])

    with pytest.raises(BackendError, match="no candidates"):
        GoogleBackend().chat("gemini-2.5-flash", PROMPT)


def test_server_errors_and_rate_limits_are_retried(
    stub: StubClient, api_key: str, sleeps: list[float]
) -> None:
    stub.answers = [
        api_error(errors.ServerError, 503),
        api_error(errors.ClientError, 429),
        httpx.ConnectTimeout("timed out"),
    ]

    assert GoogleBackend().chat("gemini-2.5-flash", PROMPT).content == "positive"
    assert len(stub.requests) == 4
    assert sleeps == [0.5, 1.0, 2.0]


def test_client_errors_that_are_the_callers_fault_are_not_retried(
    stub: StubClient, api_key: str, sleeps: list[float]
) -> None:
    stub.answers = [api_error(errors.ClientError, 400)]

    with pytest.raises(errors.ClientError):
        GoogleBackend().chat("gemini-2.5-flash", PROMPT)

    assert len(stub.requests) == 1
    assert sleeps == []


def test_retry_after_hints_are_honored(
    stub: StubClient, api_key: str, sleeps: list[float]
) -> None:
    stub.answers = [api_error(errors.ClientError, 429, headers={"retry-after": "3"})]

    GoogleBackend().chat("gemini-2.5-flash", PROMPT)

    assert sleeps == [3.0]


def test_a_rejected_temperature_is_stripped_once_and_remembered(
    stub: StubClient, api_key: str
) -> None:
    stub.answers = [
        errors.ClientError(
            400,
            {"error": {"message": "temperature is not supported by this model"}},
            None,
        )
    ]
    backend = GoogleBackend()

    assert backend.chat("gemini-future", PROMPT).content == "positive"
    assert backend.chat("gemini-future", PROMPT).content == "positive"

    assert [
        (request["config"].temperature is not None) for request in stub.requests
    ] == [True, False, False]
