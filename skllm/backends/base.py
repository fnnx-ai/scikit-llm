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
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from operator import itemgetter
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
    "ClientCache",
    "Connection",
    "Message",
    "ResolvedConnection",
    "Response",
    "Usage",
    "missing_extra",
    "named_unsupported_param",
    "parse_json",
    "request_args",
    "strict_dialect",
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
    #: JSON Schema keywords beyond the common core that this provider's
    #: structured mode accepts. Opt-in per adapter: OpenAI's strict mode rejects
    #: an unsupported keyword outright rather than ignoring it, so a schema
    #: constraint that is merely advisory elsewhere would break requests here.
    schema_constraints: ClassVar[frozenset[str]] = frozenset()

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

    def _is_transient(self, exc: BaseException) -> bool:
        """Whether a failure is worth retrying — rate limits, timeouts, 5xx."""
        return isinstance(exc, self.transient_errors)

    def _retry_after(self, exc: BaseException) -> float | None:
        """Seconds the provider asks the caller to wait, per its Retry-After header."""
        headers = getattr(getattr(exc, "response", None), "headers", None)
        if not headers:
            return None
        for header, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
            raw = headers.get(header)
            if raw:
                try:
                    return float(raw) * scale
                except ValueError:  # the HTTP-date form; fall back to backoff
                    continue
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
        attempt = 0
        while True:
            try:
                return call()
            except Exception as exc:
                if attempt >= settings.max_retries or not self._is_transient(exc):
                    raise
                hint = self._retry_after(exc)
                delay = INITIAL_BACKOFF * 2**attempt if hint is None else hint
                time.sleep(min(MAX_BACKOFF, max(0.0, delay)))
                attempt += 1


class ClientCache(Generic[_T]):
    """One provider client per distinct connection, keeping HTTP pools warm.

    Adapters stay stateless — the cache is keyed by the resolved connection, so a
    call never sees a client built for someone else's credentials.
    """

    def __init__(self, max_size: int = 32) -> None:
        self._clients: dict[tuple[Any, ...], _T] = {}
        self._lock = threading.Lock()
        self._max_size = max_size
        _CLIENT_CACHES.append(self)

    def get(
        self, provider: str, settings: ResolvedConnection, build: Callable[[], _T]
    ) -> _T:
        key = (
            provider,
            settings.api_key,
            settings.base_url,
            settings.timeout,
            repr(sorted(settings.client_options.items(), key=itemgetter(0))),
        )
        with self._lock:
            client = self._clients.get(key)
            if client is None:
                if len(self._clients) >= self._max_size:
                    self._clients.clear()
                client = self._clients[key] = build()
            return client

    def clear(self) -> None:
        with self._lock:
            self._clients.clear()


#: Every cache ever built, so a process can drop all pooled connections at once.
_CLIENT_CACHES: list[ClientCache[Any]] = []


def missing_extra(extra: str, package: str) -> ImportError:
    return ImportError(
        f"The {extra} backend needs the {package} package, which is not installed. "
        f"Install it with: pip install scikit-llm[{extra}]"
    )


#: Words a provider uses when it blames one request parameter. The parameter name
#: alone is not enough — quota and validation errors mention it too.
_UNSUPPORTED_HINTS = (
    "support",
    "unrecognized",
    "unknown parameter",
    "only be set",
    "not allowed",
)


def named_unsupported_param(message: str, names: Iterable[str]) -> str | None:
    """The request param a provider's error message rejects, if it names one."""
    lowered = message.lower()
    if not any(hint in lowered for hint in _UNSUPPORTED_HINTS):
        return None
    return next((name for name in names if name in lowered), None)


def request_args(params: Mapping[str, Any], accepted: frozenset[str]) -> dict[str, Any]:
    """Route params an SDK does not name through its ``extra_body`` passthrough.

    OpenAI- and Anthropic-style SDKs take fixed keyword arguments, so a request
    param they do not name (Ollama's num_ctx, a proxy's own knobs, a typo) would
    die as a local TypeError. ``extra_body`` is merged into the request body
    verbatim, so every model_param reaches the provider and an unsupported one
    surfaces the provider's own error, never ours.
    """
    args = {name: value for name, value in params.items() if name in accepted}
    unnamed = {name: value for name, value in params.items() if name not in accepted}
    if unnamed:
        args["extra_body"] = {**args.get("extra_body", {}), **unnamed}
    return args


def strict_dialect(schema: Any) -> bool:
    """Whether every object in a schema is closed: no extra properties, none optional.

    Strict mode grammar-constrains the answer, but only accepts schemas in that
    closed subset — a request claiming strict for any other schema is rejected
    outright, before the model ever sees it. skllm builds its own task schemas
    that way; a user's extraction schema need not (a pydantic model compiles to
    neither marker), so the flag is asked for only where it holds. The same
    schema still travels with the request either way, and the estimator
    validates the answer against it regardless.
    """
    if isinstance(schema, (list, tuple)):
        return all(strict_dialect(item) for item in schema)
    if not isinstance(schema, Mapping):
        return True
    if "properties" in schema or schema.get("type") == "object":
        if schema.get("additionalProperties") is not False:
            return False
        if set(schema.get("required") or ()) != set(schema.get("properties") or ()):
            return False
    return all(strict_dialect(value) for value in schema.values())


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
