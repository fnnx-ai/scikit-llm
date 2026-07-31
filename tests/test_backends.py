from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, ClassVar

import pytest

from skllm.backends import (
    Backend,
    BackendError,
    BaseBackend,
    Connection,
    Message,
    ResolvedConnection,
    Response,
    Usage,
    available_providers,
    get_backend,
    parse_json,
    parse_model,
    register_backend,
    registry,
)
from skllm.backends.base import INITIAL_BACKOFF, MAX_BACKOFF
from skllm.backends.openai import OllamaBackend, OpenAIBackend
from skllm.config import MissingCredentialError, config_context
from tests.fakes import FakeBackend, FakeCall, FakeTransientError

PROMPT = [Message("system", "classify"), Message("user", "I love it")]


class UnsupportedParamError(RuntimeError):
    """Stands in for a provider rejecting one named request parameter."""

    def __init__(self, param: str) -> None:
        super().__init__(f"Unsupported parameter: {param!r}")
        self.param = param


class ReasoningFake(FakeBackend):
    injected_defaults: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {"temperature": 0, "reasoning_effort": "none"}
    )

    def _unsupported_param(self, exc: BaseException) -> str | None:
        param = getattr(exc, "param", None)
        return param if isinstance(param, str) else None


class SeededFake(ReasoningFake):
    def seed_unsupported(self, model: str) -> frozenset[str]:
        return (
            frozenset({"temperature"}) if model.startswith("thinker") else frozenset()
        )


class RetryAfterFake(FakeBackend):
    def _retry_after(self, exc: BaseException) -> float | None:
        return getattr(exc, "wait", None)


class MyCorpBackend(FakeBackend):
    provider: ClassVar[str] = "mycorp"
    credential_env_var: ClassVar[str | None] = "MYCORP_TOKEN"


class TextOnlyBackend(BaseBackend):
    """An adapter that offers chat but no embeddings, like Anthropic's."""

    provider: ClassVar[str] = "textonly"

    def _chat(
        self,
        model: str,
        messages: Sequence[Message],
        params: dict[str, Any],
        settings: ResolvedConnection,
    ) -> Response[str]:
        return Response(content="")

    def _structured(
        self,
        model: str,
        messages: Sequence[Message],
        schema: Mapping[str, Any],
        params: dict[str, Any],
        settings: ResolvedConnection,
    ) -> Response[Any]:
        return Response(content={})


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr("time.sleep", recorded.append)
    return recorded


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("openai::gpt-5", ("openai", "gpt-5")),
        ("gpt-5", ("openai", "gpt-5")),
        ("  gpt-5  ", ("openai", "gpt-5")),
        ("anthropic::claude-sonnet-5", ("anthropic", "claude-sonnet-5")),
        ("litellm::vertex_ai/gemini-2.5-pro", ("litellm", "vertex_ai/gemini-2.5-pro")),
        ("litellm::openai::gpt-5", ("litellm", "openai::gpt-5")),
        ("openai :: gpt-5", ("openai", "gpt-5")),
    ],
)
def test_model_strings_parse(model: str, expected: tuple[str, str]) -> None:
    assert parse_model(model) == expected


@pytest.mark.parametrize("model", ["", "   ", "::gpt-5", "openai::", "openai::   "])
def test_malformed_model_strings_are_rejected(model: str) -> None:
    with pytest.raises(ValueError, match="model"):
        parse_model(model)


def test_non_string_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="model"):
        parse_model(None)  # type: ignore[arg-type]


def test_builtin_backends_load_lazily_and_are_reused() -> None:
    backend = get_backend("openai")

    assert isinstance(backend, OpenAIBackend)
    assert isinstance(backend, Backend)
    assert get_backend("openai") is backend
    assert isinstance(get_backend("ollama"), OllamaBackend)


def test_unknown_prefix_lists_the_available_providers() -> None:
    with pytest.raises(ValueError) as excinfo:
        get_backend("cohere")

    message = str(excinfo.value)
    assert "cohere" in message
    for provider in ("openai", "anthropic", "google", "ollama", "litellm"):
        assert provider in message


def test_custom_backend_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MYCORP_TOKEN", "sk-corp")
    backend = MyCorpBackend(chat="internal answer")
    register_backend("mycorp", backend)

    try:
        provider, model_id = parse_model("mycorp::internal-1")
        assert get_backend(provider) is backend

        response = backend.chat(model_id, PROMPT)
        assert response.content == "internal answer"
        assert backend.calls[0].model == "internal-1"
        assert backend.calls[0].api_key == "sk-corp"
        assert "mycorp" in available_providers()

        with pytest.raises(ValueError, match="mycorp"):
            get_backend("cohere")
    finally:
        registry._BACKENDS.pop("mycorp", None)


@pytest.mark.parametrize("prefix", ["My Corp", "1corp", "", "corp::x", "corp\n"])
def test_invalid_provider_prefixes_are_rejected(prefix: str) -> None:
    with pytest.raises(ValueError, match="prefix"):
        register_backend(prefix, FakeBackend())


def test_registering_a_non_adapter_is_rejected() -> None:
    with pytest.raises(TypeError, match="backend adapter"):
        register_backend("broken", object())  # type: ignore[arg-type]


def test_injected_defaults_reach_the_provider(fake_backend: FakeBackend) -> None:
    fake_backend.chat("m", PROMPT)

    assert fake_backend.calls[0].params == {"temperature": 0}


def test_user_params_pass_through_and_win(fake_backend: FakeBackend) -> None:
    fake_backend.chat("m", PROMPT, params={"temperature": 0.7, "top_p": 0.1})

    assert fake_backend.calls[0].params == {"temperature": 0.7, "top_p": 0.1}


def test_sampled_calls_drop_only_the_determinism_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "fake-key")
    backend = ReasoningFake()

    backend.chat("m", PROMPT, deterministic=False)

    assert backend.calls[0].params == {"reasoning_effort": "none"}


def test_credentials_resolve_through_the_config_layers(
    fake_backend: FakeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_backend.chat("m", PROMPT, connection=Connection(api_key="param-key"))
    with config_context(fake_api_key="ctx-key"):
        fake_backend.chat("m", PROMPT)
    fake_backend.chat("m", PROMPT)

    assert [call.api_key for call in fake_backend.calls] == [
        "param-key",
        "ctx-key",
        "fake-key",
    ]


def test_alias_routes_the_credential(fake_backend: FakeBackend) -> None:
    with config_context(credentials={"team_a": "sk-a"}):
        fake_backend.chat("m", PROMPT, connection=Connection(alias="team_a"))

    assert fake_backend.calls[0].api_key == "sk-a"


def test_missing_key_raises_before_any_request(
    fake_backend: FakeBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FAKE_API_KEY")

    with pytest.raises(MissingCredentialError, match="FAKE_API_KEY"):
        fake_backend.chat("m", PROMPT)

    assert fake_backend.calls == []


def test_keyless_backends_resolve_without_a_key() -> None:
    assert OllamaBackend().resolve().api_key is None


def test_base_urls_and_transport_settings_resolve_per_call(
    fake_backend: FakeBackend,
) -> None:
    with config_context(fake_base_url="http://ctx/v1", timeout=5.0, max_retries=1):
        settings = fake_backend.resolve(Connection(base_url="http://param/v1"))
        assert settings.base_url == "http://param/v1"
        assert settings.timeout == 5.0
        assert settings.max_retries == 1

        fake_backend.chat("m", PROMPT)

    assert fake_backend.calls[0].base_url == "http://ctx/v1"


def test_rejected_injected_default_is_stripped_and_remembered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "fake-key")
    backend = ReasoningFake(chat=[UnsupportedParamError("temperature"), "a", "b", "c"])

    assert backend.chat("thinker-1", PROMPT).content == "a"
    assert backend.chat("thinker-1", PROMPT).content == "b"
    assert backend.chat("other-1", PROMPT).content == "c"

    sent = [call.params for call in backend.calls]
    assert sent[0] == {"temperature": 0, "reasoning_effort": "none"}
    assert sent[1] == {"reasoning_effort": "none"}
    assert sent[2] == {"reasoning_effort": "none"}
    assert sent[3] == {"temperature": 0, "reasoning_effort": "none"}


def test_an_unverified_strip_is_not_remembered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a stripped request that actually worked may drop the default for good."""
    monkeypatch.setenv("FAKE_API_KEY", "fake-key")
    backend = ReasoningFake(
        chat=[UnsupportedParamError("temperature"), ValueError("unrelated"), "ok"]
    )

    with pytest.raises(ValueError, match="unrelated"):
        backend.chat("thinker-1", PROMPT)

    assert backend.chat("thinker-1", PROMPT).content == "ok"
    assert [call.params for call in backend.calls] == [
        {"temperature": 0, "reasoning_effort": "none"},
        {"reasoning_effort": "none"},
        {"temperature": 0, "reasoning_effort": "none"},
    ]


def test_rejected_user_param_surfaces_the_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "fake-key")
    backend = ReasoningFake(chat=[UnsupportedParamError("temperature"), "unreachable"])

    with pytest.raises(UnsupportedParamError):
        backend.chat("thinker-1", PROMPT, params={"temperature": 0})

    assert len(backend.calls) == 1


def test_unknown_rejected_param_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "fake-key")
    backend = ReasoningFake(chat=[UnsupportedParamError("top_k"), "unreachable"])

    with pytest.raises(UnsupportedParamError):
        backend.chat("thinker-1", PROMPT)

    assert len(backend.calls) == 1


def test_seeded_families_skip_the_wasted_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "fake-key")
    backend = SeededFake()

    backend.chat("thinker-1", PROMPT)
    backend.chat("chatter-1", PROMPT)

    assert backend.calls[0].params == {"reasoning_effort": "none"}
    assert backend.calls[1].params == {"temperature": 0, "reasoning_effort": "none"}


def test_transient_failures_retry_with_exponential_backoff(
    fake_backend: FakeBackend, sleeps: list[float]
) -> None:
    fake_backend.chat_script = [
        FakeTransientError("429"),
        FakeTransientError("500"),
        "recovered",
    ]

    assert fake_backend.chat("m", PROMPT).content == "recovered"
    assert sleeps == [INITIAL_BACKOFF, INITIAL_BACKOFF * 2]
    assert len(fake_backend.calls) == 3


def test_retries_are_exhausted_and_the_error_propagates(
    fake_backend: FakeBackend, sleeps: list[float]
) -> None:
    fake_backend.chat_script = FakeTransientError("permanently down")

    with config_context(max_retries=2), pytest.raises(FakeTransientError):
        fake_backend.chat("m", PROMPT)

    assert len(fake_backend.calls) == 3
    assert len(sleeps) == 2


def test_retries_can_be_switched_off(
    fake_backend: FakeBackend, sleeps: list[float]
) -> None:
    fake_backend.chat_script = FakeTransientError("down")

    with config_context(max_retries=0), pytest.raises(FakeTransientError):
        fake_backend.chat("m", PROMPT)

    assert len(fake_backend.calls) == 1
    assert sleeps == []


def test_provider_retry_hints_are_honored(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "fake-key")
    rate_limited = FakeTransientError("slow down")
    rate_limited.wait = 4.0  # type: ignore[attr-defined]
    excessive = FakeTransientError("come back later")
    excessive.wait = 600.0  # type: ignore[attr-defined]
    backend = RetryAfterFake(chat=[rate_limited, excessive, "recovered"])

    assert backend.chat("m", PROMPT).content == "recovered"
    assert sleeps == [4.0, MAX_BACKOFF]


def test_non_transient_failures_are_not_retried(
    fake_backend: FakeBackend, sleeps: list[float]
) -> None:
    fake_backend.chat_script = ValueError("bad request")

    with pytest.raises(ValueError, match="bad request"):
        fake_backend.chat("m", PROMPT)

    assert len(fake_backend.calls) == 1
    assert sleeps == []


def test_usage_is_reported_per_call_and_sums(fake_backend: FakeBackend) -> None:
    first = fake_backend.chat("m", PROMPT)
    second = fake_backend.chat("m", PROMPT)

    assert first.usage == Usage(requests=1, input_tokens=4, output_tokens=2)
    assert first.usage + second.usage == Usage(
        requests=2, input_tokens=8, output_tokens=4
    )
    assert Usage() + Usage(requests=1) == Usage(requests=1)


def test_structured_and_embedding_calls_carry_their_payloads(
    fake_backend: FakeBackend,
) -> None:
    schema = {"type": "object", "properties": {"label": {"type": "string"}}}
    fake_backend.structured_script = {"label": "positive"}

    structured = fake_backend.structured("m", PROMPT, schema)
    embeddings = fake_backend.embed("e", ["one", "two"])

    assert structured.content == {"label": "positive"}
    assert fake_backend.calls[0].schema == schema
    assert len(embeddings.content) == 2
    assert embeddings.usage.requests == 1
    assert fake_backend.calls[1].texts == ("one", "two")


def test_backends_without_embeddings_say_so() -> None:
    with pytest.raises(BackendError, match="vectorizer"):
        TextOnlyBackend().embed("e", ["one"])


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"label": "positive"}', {"label": "positive"}),
        ('```json\n{"label": "positive"}\n```', {"label": "positive"}),
        (
            'Sure! Here it is: {"label": "positive"} — hope that helps',
            {"label": "positive"},
        ),
        ('[{"label": "positive"}]', [{"label": "positive"}]),
        ("[1, 2, 3]", [1, 2, 3]),
    ],
)
def test_json_salvage(text: str, expected: Any) -> None:
    assert parse_json(text) == expected


@pytest.mark.parametrize("text", ["", "no json here", "{unbalanced"])
def test_unsalvageable_json_raises(text: str) -> None:
    with pytest.raises(BackendError, match="not valid JSON"):
        parse_json(text)


def test_concurrent_calls_share_one_adapter(fake_backend: FakeBackend) -> None:
    fake_backend.chat_script = lambda call: call.messages[-1].content
    barrier = threading.Barrier(4, timeout=5)
    answers: dict[str, str] = {}

    def run(text: str) -> None:
        barrier.wait()
        answers[text] = fake_backend.chat(
            "m", [Message("user", text)], connection=Connection(api_key=f"sk-{text}")
        ).content

    threads = [threading.Thread(target=run, args=(str(index),)) for index in range(3)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5)

    assert answers == {"0": "0", "1": "1", "2": "2"}
    assert len(fake_backend.calls) == 3
    assert {call.api_key for call in fake_backend.calls} == {"sk-0", "sk-1", "sk-2"}


def test_recorded_calls_expose_the_prompt(fake_backend: FakeBackend) -> None:
    fake_backend.chat("m", PROMPT)

    assert fake_backend.prompts == ["classify\nI love it"]
    assert isinstance(fake_backend.calls[0], FakeCall)
