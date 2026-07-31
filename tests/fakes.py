"""The in-memory backend that stands in for a provider across the test suite.

It is a real adapter registered through the public registry, so tests exercise
the actual resolution, retry and default-injection paths — only the HTTP call is
replaced by a script.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from skllm.backends import BaseBackend, Message, ResolvedConnection, Response, Usage

EMBEDDING_DIM = 16

#: A scripted answer: a value, an exception to raise, a list consumed one call
#: at a time, or a callable computing one from the recorded call.
Script = Any


class FakeTransientError(RuntimeError):
    """A failure the fake backend retries, like a rate limit or a 5xx."""


@dataclass(frozen=True)
class FakeCall:
    kind: str
    model: str
    params: dict[str, Any] = field(default_factory=dict)
    messages: tuple[Message, ...] = ()
    schema: Mapping[str, Any] | None = None
    texts: tuple[str, ...] = ()
    api_key: str | None = None
    base_url: str | None = None

    @property
    def prompt(self) -> str:
        return "\n".join(message.content for message in self.messages)


class FakeBackend(BaseBackend):
    provider: ClassVar[str] = "fake"
    credential_env_var: ClassVar[str | None] = "FAKE_API_KEY"
    transient_errors: ClassVar[tuple[type[BaseException], ...]] = (FakeTransientError,)

    def __init__(
        self,
        chat: Script = "fake summary",
        structured: Script = None,
        embedding: Script = None,
    ) -> None:
        self.chat_script = chat
        self.structured_script = {} if structured is None else structured
        self.embedding_script = embedding
        self.calls: list[FakeCall] = []
        self._lock = threading.Lock()

    @property
    def prompts(self) -> list[str]:
        return [call.prompt for call in self.calls]

    def _chat(
        self,
        model: str,
        messages: Sequence[Message],
        params: dict[str, Any],
        settings: ResolvedConnection,
    ) -> Response[str]:
        call = self._record(
            FakeCall(
                kind="chat",
                model=model,
                params=params,
                messages=tuple(messages),
                api_key=settings.api_key,
                base_url=settings.base_url,
            )
        )
        content = str(_next(self.chat_script, call))
        return Response(content=content, usage=_usage(call.prompt, content))

    def _structured(
        self,
        model: str,
        messages: Sequence[Message],
        schema: Mapping[str, Any],
        params: dict[str, Any],
        settings: ResolvedConnection,
    ) -> Response[Any]:
        call = self._record(
            FakeCall(
                kind="structured",
                model=model,
                params=params,
                messages=tuple(messages),
                schema=schema,
                api_key=settings.api_key,
                base_url=settings.base_url,
            )
        )
        content = _next(self.structured_script, call)
        return Response(content=content, usage=_usage(call.prompt, str(content)))

    def _embed(
        self,
        model: str,
        texts: Sequence[str],
        params: dict[str, Any],
        settings: ResolvedConnection,
    ) -> Response[list[list[float]]]:
        call = self._record(
            FakeCall(
                kind="embed",
                model=model,
                params=params,
                texts=tuple(texts),
                api_key=settings.api_key,
                base_url=settings.base_url,
            )
        )
        if self.embedding_script is None:
            vectors = [_embedding(text) for text in texts]
        else:
            vectors = _next(self.embedding_script, call)
        return Response(content=vectors, usage=_usage(" ".join(texts), ""))

    def _record(self, call: FakeCall) -> FakeCall:
        with self._lock:
            self.calls.append(call)
        return call


def _next(script: Script, call: FakeCall) -> Any:
    """Resolve one scripted answer, raising it when it is an exception."""
    if callable(script):
        answer = script(call)
    elif isinstance(script, list):
        if not script:
            raise AssertionError(f"The fake backend ran out of answers for {call}.")
        answer = script.pop(0)
    else:
        answer = script

    if isinstance(answer, BaseException):
        raise answer
    return answer


def _embedding(text: str) -> list[float]:
    """A deterministic bag-of-characters vector: similar texts land close."""
    vector = [0.0] * EMBEDDING_DIM
    for character in text.lower():
        vector[ord(character) % EMBEDDING_DIM] += 1.0
    norm = sum(value**2 for value in vector) ** 0.5
    return [value / norm for value in vector] if norm else vector


def _usage(prompt: str, content: str) -> Usage:
    return Usage(
        requests=1,
        input_tokens=len(prompt.split()),
        output_tokens=len(content.split()),
    )
