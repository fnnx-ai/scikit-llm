"""Provider adapter protocol and the machinery every adapter shares.

Adapters are stateless singletons: everything that varies per call — the model
id, generation params and the caller's own connection settings — travels with
the call, so estimators never hold clients, keys or connections.
"""

from __future__ import annotations

import json
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from types import MappingProxyType
from typing import (
    Any,
    ClassVar,
    Generic,
    NamedTuple,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from skllm.config import (
    resolve_base_url,
    resolve_client_options,
    resolve_credential,
    resolve_max_retries,
    resolve_timeout,
)

__all__ = [
    "Backend",
    "BackendError",
    "BaseBackend",
    "Connection",
    "Message",
    "ResolvedConnection",
    "Response",
    "Usage",
    "parse_json",
]

INITIAL_BACKOFF = 0.5
MAX_BACKOFF = 30.0

_T = TypeVar("_T")


class BackendError(RuntimeError):
    """A provider replied, but with something scikit-llm cannot use."""


@dataclass(frozen=True)
class Usage:
    """What one or more requests cost, as far as the provider reports it."""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            requests=self.requests + other.requests,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class Message(NamedTuple):
    role: str
    content: str


@dataclass(frozen=True)
class Response(Generic[_T]):
    content: _T
    usage: Usage = field(default_factory=Usage)


@dataclass(frozen=True)
class Connection:
    """A caller's own connection settings — the most local resolution layer."""

    api_key: str | None = None
    base_url: str | None = None
    alias: str | None = None


@dataclass(frozen=True)
class ResolvedConnection:
    api_key: str | None
    base_url: str | None
    timeout: float
    max_retries: int
    client_options: Mapping[str, Any]


@runtime_checkable
class Backend(Protocol):
    """What an adapter must expose to be registered under a provider prefix.

    ``provider`` names the config namespace the adapter resolves connection
    settings from; ``credential_env_var`` is its canonical key variable, or
    ``None`` for keyless providers. Both are read-only, so adapters may declare
    them as class or instance attributes.
    """

    @property
    def provider(self) -> str: ...

    @property
    def credential_env_var(self) -> str | None: ...

    def chat(
        self,
        model: str,
        messages: Sequence[Message],
        *,
        params: Mapping[str, Any] | None = ...,
        connection: Connection | None = ...,
        deterministic: bool = ...,
    ) -> Response[str]: ...

    def structured(
        self,
        model: str,
        messages: Sequence[Message],
        schema: Mapping[str, Any],
        *,
        params: Mapping[str, Any] | None = ...,
        connection: Connection | None = ...,
        deterministic: bool = ...,
    ) -> Response[Any]: ...

    def embed(
        self,
        model: str,
        texts: Sequence[str],
        *,
        params: Mapping[str, Any] | None = ...,
        connection: Connection | None = ...,
    ) -> Response[list[list[float]]]: ...


class BaseBackend(ABC):
    """Shared adapter behavior: resolution, retries and advisory defaults.

    Subclasses implement the provider-specific ``_chat``/``_structured``/
    ``_embed`` hooks and receive already-resolved connection settings plus the
    effective generation params.
    """

    provider: ClassVar[str]
    credential_env_var: ClassVar[str | None] = None
    default_base_url: ClassVar[str | None] = None
    transient_errors: ClassVar[tuple[type[BaseException], ...]] = ()

    #: skllm-injected request defaults, applied only where the provider accepts
    #: them and only for params the caller did not set.
    injected_defaults: ClassVar[Mapping[str, Any]] = MappingProxyType(
        {"temperature": 0}
    )
    #: The injected defaults that exist purely for determinism; sampled calls
    #: (``deterministic=False``) drop them so votes can differ.
    determinism_defaults: ClassVar[frozenset[str]] = frozenset({"temperature"})

    @property
    def credential_env_vars(self) -> tuple[str, ...]:
        """Accepted key variables, most specific first; empty when keyless."""
        return () if self.credential_env_var is None else (self.credential_env_var,)

    def chat(
        self,
        model: str,
        messages: Sequence[Message],
        *,
        params: Mapping[str, Any] | None = None,
        connection: Connection | None = None,
        deterministic: bool = True,
    ) -> Response[str]:
        return self._invoke(
            model,
            params,
            connection,
            deterministic,
            partial(self._chat, model, messages),
        )

    def structured(
        self,
        model: str,
        messages: Sequence[Message],
        schema: Mapping[str, Any],
        *,
        params: Mapping[str, Any] | None = None,
        connection: Connection | None = None,
        deterministic: bool = True,
    ) -> Response[Any]:
        return self._invoke(
            model,
            params,
            connection,
            deterministic,
            partial(self._structured, model, messages, schema),
        )

    def embed(
        self,
        model: str,
        texts: Sequence[str],
        *,
        params: Mapping[str, Any] | None = None,
        connection: Connection | None = None,
    ) -> Response[list[list[float]]]:
        settings = self.resolve(connection)
        return self._with_retry(
            partial(self._embed, model, texts, dict(params or {}), settings), settings
        )

    def resolve(self, connection: Connection | None = None) -> ResolvedConnection:
        """Resolve this provider's connection settings for one call."""
        connection = connection or Connection()
        base_url = resolve_base_url(self.provider, explicit=connection.base_url)
        return ResolvedConnection(
            api_key=resolve_credential(
                self.provider,
                explicit=connection.api_key,
                alias=connection.alias,
                env_vars=self.credential_env_vars or None,
                required=self.credential_env_var is not None,
            ),
            base_url=base_url or self.default_base_url,
            timeout=resolve_timeout(),
            max_retries=resolve_max_retries(),
            client_options=resolve_client_options(self.provider),
        )

    @abstractmethod
    def _chat(
        self,
        model: str,
        messages: Sequence[Message],
        params: dict[str, Any],
        settings: ResolvedConnection,
    ) -> Response[str]: ...

    @abstractmethod
    def _structured(
        self,
        model: str,
        messages: Sequence[Message],
        schema: Mapping[str, Any],
        params: dict[str, Any],
        settings: ResolvedConnection,
    ) -> Response[Any]: ...

    def _embed(
        self,
        model: str,
        texts: Sequence[str],
        params: dict[str, Any],
        settings: ResolvedConnection,
    ) -> Response[list[list[float]]]:
        raise BackendError(
            f"The {self.provider} backend does not provide embeddings. Pass an "
            "explicit vectorizer backed by a provider that does."
        )

    def seed_unsupported(self, model: str) -> frozenset[str]:
        """Injected defaults a known model family rejects, to skip a round-trip.

        Only an optimization: the error-driven fallback in :meth:`_invoke` is the
        source of truth, so models unknown to the seed list still work.
        """
        return frozenset()

    def _unsupported_param(self, exc: BaseException) -> str | None:
        """The request param a provider error rejects, if it names one."""
        return None

    def _retry_after(self, exc: BaseException) -> float | None:
        """Seconds the provider asks the caller to wait, if it says so."""
        return None

    def _invoke(
        self,
        model: str,
        params: Mapping[str, Any] | None,
        connection: Connection | None,
        deterministic: bool,
        call: Callable[[dict[str, Any], ResolvedConnection], _T],
    ) -> _T:
        """Issue a request, adapting the defaults skllm injected if rejected.

        A provider that rejects one of *our* defaults costs one extra round-trip
        per model: the offending default is dropped here and remembered for the
        rest of the process. Caller-supplied params are never touched — an
        unsupported one surfaces the provider's own error.

        Only a stripped request that actually succeeded is remembered: caching an
        unverified guess would silently drop the default from every later call
        for that model, and a wasted round-trip is the cheaper mistake.
        """
        settings = self.resolve(connection)
        user_params = dict(params or {})
        stripped: set[str] = set()
        while True:
            injected = self._injected_params(
                model, user_params, deterministic, stripped
            )
            try:
                response = self._with_retry(
                    partial(call, {**injected, **user_params}, settings), settings
                )
            except Exception as exc:
                rejected = self._unsupported_param(exc)
                if rejected is None or rejected not in injected:
                    raise
                stripped.add(rejected)
            else:
                for name in stripped:
                    _remember_unsupported(self.provider, model, name)
                return response

    def _injected_params(
        self,
        model: str,
        user_params: Mapping[str, Any],
        deterministic: bool,
        stripped: set[str],
    ) -> dict[str, Any]:
        unsupported = (
            stripped
            | _unsupported_params(self.provider, model)
            | self.seed_unsupported(model)
        )
        return {
            name: value
            for name, value in self.injected_defaults.items()
            if name not in user_params
            and name not in unsupported
            and (deterministic or name not in self.determinism_defaults)
        }

    def _with_retry(self, call: Callable[[], _T], settings: ResolvedConnection) -> _T:
        transient = self.transient_errors
        attempt = 0
        while True:
            try:
                return call()
            except transient as exc:
                if attempt >= settings.max_retries:
                    raise
                hint = self._retry_after(exc)
                delay = INITIAL_BACKOFF * 2**attempt if hint is None else hint
                time.sleep(min(MAX_BACKOFF, max(0.0, delay)))
                attempt += 1


def parse_json(text: str) -> Any:
    """Parse a model's JSON answer, salvaging one embedded object if needed."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character in "{[":
            try:
                value, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            return value
    raise BackendError(f"The model response is not valid JSON: {text[:200]!r}")


_UNSUPPORTED: dict[tuple[str, str], frozenset[str]] = {}
_UNSUPPORTED_LOCK = threading.Lock()


def _unsupported_params(provider: str, model: str) -> frozenset[str]:
    with _UNSUPPORTED_LOCK:
        return _UNSUPPORTED.get((provider, model), frozenset())


def _remember_unsupported(provider: str, model: str, param: str) -> None:
    with _UNSUPPORTED_LOCK:
        key = (provider, model)
        _UNSUPPORTED[key] = _UNSUPPORTED.get(key, frozenset()) | {param}
