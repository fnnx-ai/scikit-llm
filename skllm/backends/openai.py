"""The OpenAI adapter, which also serves Ollama and OpenAI-compatible endpoints."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from inspect import signature
from operator import attrgetter
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, cast
from urllib.parse import urlsplit

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from openai.resources.chat.completions import Completions
from openai.resources.embeddings import Embeddings

from skllm.backends.base import (
    BackendError,
    BaseBackend,
    ClientCache,
    Connection,
    Message,
    ResolvedConnection,
    Response,
    Usage,
    named_unsupported_param,
    parse_json,
    request_args,
)

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

__all__ = ["OllamaBackend", "OpenAIBackend"]

# Ollama accepts any key; the SDK insists on having one.
KEYLESS_API_KEY = "skllm"
OLLAMA_PORT = 11434
OLLAMA_BASE_URL = f"http://localhost:{OLLAMA_PORT}/v1"

_CHAT_ARGS = frozenset(signature(Completions.create).parameters)
_EMBEDDING_ARGS = frozenset(signature(Embeddings.create).parameters)

# The o-series rejects `temperature` outright and only knows the graded
# `reasoning_effort` values, not "none". Seeding them saves the first
# round-trip; every other model is discovered from the provider's own error.
_SEEDED_REASONING_FAMILY = re.compile(r"^o[134](-|$)")

_CLIENTS: ClientCache[OpenAI] = ClientCache()


class OpenAIBackend(BaseBackend):
    provider: ClassVar[str] = "openai"
    credential_env_var: ClassVar[str | None] = "OPENAI_API_KEY"
    injected_defaults: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {"temperature": 0, "reasoning_effort": "none"}
    )
    transient_errors: ClassVar[tuple[type[BaseException], ...]] = (
        APITimeoutError,
        APIConnectionError,
        RateLimitError,
        InternalServerError,
    )

    def seed_unsupported(self, model: str) -> frozenset[str]:
        if _SEEDED_REASONING_FAMILY.match(model):
            return frozenset(self.injected_defaults)
        return frozenset()

    def _chat(
        self,
        model: str,
        messages: Sequence[Message],
        params: dict[str, Any],
        settings: ResolvedConnection,
    ) -> Response[str]:
        completion = self._client(settings).chat.completions.create(
            model=model,
            messages=_as_messages(messages),
            **request_args(params, _CHAT_ARGS),
        )
        return Response(content=_content(completion), usage=_usage(completion))

    def _structured(
        self,
        model: str,
        messages: Sequence[Message],
        schema: Mapping[str, Any],
        params: dict[str, Any],
        settings: ResolvedConnection,
    ) -> Response[Any]:
        completion = self._client(settings).chat.completions.create(
            model=model,
            messages=_as_messages(messages),
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": dict(schema),
                    "strict": True,
                },
            },
            **request_args(params, _CHAT_ARGS),
        )
        return Response(
            content=parse_json(_content(completion)), usage=_usage(completion)
        )

    def _embed(
        self,
        model: str,
        texts: Sequence[str],
        params: dict[str, Any],
        settings: ResolvedConnection,
    ) -> Response[list[list[float]]]:
        response = self._client(settings).embeddings.create(
            model=model, input=list(texts), **request_args(params, _EMBEDDING_ARGS)
        )
        vectors = [
            item.embedding for item in sorted(response.data, key=attrgetter("index"))
        ]
        if len(vectors) != len(texts):
            raise BackendError(
                f"The provider returned {len(vectors)} embeddings for {len(texts)} inputs."
            )
        return Response(content=vectors, usage=_usage(response))

    def _unsupported_param(self, exc: BaseException) -> str | None:
        # Only the provider may retract a default: skllm's own BackendError
        # quotes model output, which could otherwise name a param by accident.
        if not isinstance(exc, APIStatusError):
            return None

        param = getattr(exc, "param", None)
        if isinstance(param, str) and param in self.injected_defaults:
            return param

        body = exc.body
        message = str(body.get("message", "")) if isinstance(body, dict) else str(exc)
        return named_unsupported_param(message, self.injected_defaults)

    def _client(self, settings: ResolvedConnection) -> OpenAI:
        return _CLIENTS.get(
            self.provider,
            settings,
            lambda: OpenAI(
                api_key=settings.api_key or KEYLESS_API_KEY,
                base_url=settings.base_url,
                timeout=settings.timeout,
                max_retries=0,  # scikit-llm owns the retry policy
                **settings.client_options,
            ),
        )


class OllamaBackend(OpenAIBackend):
    """The OpenAI adapter pointed at Ollama's OpenAI-compatible endpoint."""

    provider: ClassVar[str] = "ollama"
    credential_env_var: ClassVar[str | None] = None
    default_base_url: ClassVar[str | None] = OLLAMA_BASE_URL
    # Ollama exposes no reasoning-control request param.
    injected_defaults: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {"temperature": 0}
    )

    def resolve(self, connection: Connection | None = None) -> ResolvedConnection:
        settings = super().resolve(connection)
        return replace(settings, base_url=_ollama_url(settings.base_url))


def _ollama_url(base_url: str | None) -> str:
    """Turn an ``OLLAMA_HOST``-style host into an OpenAI-compatible base URL.

    A bare host carries Ollama's own default port, as its clients assume: a plain
    ``OLLAMA_HOST=my-server`` means 11434, not 80. An explicit ``https`` host is
    left alone — that is a fronted deployment on 443.
    """
    if base_url is None:
        return OLLAMA_BASE_URL
    url = base_url if "://" in base_url else f"http://{base_url}"
    split = urlsplit(url)
    if split.port is None and split.scheme != "https":
        url = split._replace(netloc=f"{split.netloc}:{OLLAMA_PORT}").geturl()
    return url if url.rstrip("/").endswith("/v1") else f"{url.rstrip('/')}/v1"


def _as_messages(messages: Sequence[Message]) -> list[ChatCompletionMessageParam]:
    return cast(
        "list[ChatCompletionMessageParam]",
        [{"role": message.role, "content": message.content} for message in messages],
    )


def _content(completion: ChatCompletion) -> str:
    if not completion.choices:
        raise BackendError("The provider returned no choices.")

    message = completion.choices[0].message
    refusal = getattr(message, "refusal", None)
    if refusal:
        raise BackendError(f"The model refused to answer: {refusal}")
    if not message.content:
        raise BackendError("The provider returned an empty message.")
    return message.content


def _usage(response: Any) -> Usage:
    usage = getattr(response, "usage", None)
    return Usage(
        requests=1,
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
    )
