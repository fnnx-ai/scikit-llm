"""Layered configuration and credential resolution.

Connection settings (credentials, base URLs, transport) resolve through the same
layers, most local first: estimator parameter -> active :func:`config_context` ->
environment variable -> default. The environment layer is read-only: scikit-llm
never writes to ``os.environ``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, NamedTuple, TypeVar

__all__ = [
    "ALIAS_PATTERN",
    "MissingCredentialError",
    "Price",
    "config_context",
    "credential_env_vars",
    "resolve_allow_env_fallback",
    "resolve_base_url",
    "resolve_client_options",
    "resolve_credential",
    "resolve_max_retries",
    "resolve_model_params",
    "resolve_overrides",
    "resolve_price",
    "resolve_timeout",
    "validate_alias",
]

ALIAS_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 3

ALLOW_ENV_FALLBACK_VAR = "SKLLM_ALLOW_ENV_FALLBACK"
CREDENTIAL_VAR_PREFIX = "SKLLM_CREDENTIAL_"

# The more specific GEMINI_API_KEY wins over GOOGLE_API_KEY when both are set.
_KEY_ENV_VARS: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}
_BASE_URL_ENV_VARS: dict[str, tuple[str, ...]] = {
    "ollama": ("OLLAMA_HOST",),
}

_FALSY = frozenset({"0", "false", "no", "off"})

_SCALAR_SETTINGS = ("allow_env_fallback", "max_retries", "timeout")
_MAP_SETTINGS = ("client_options", "credentials", "overrides", "pricing")
_KEY_SUFFIX = "_api_key"
_BASE_URL_SUFFIX = "_base_url"


class MissingCredentialError(RuntimeError):
    """No credential could be resolved for a provider or an estimator alias."""


class Price(NamedTuple):
    """Price per million input/output tokens."""

    input: float
    output: float


@dataclass(frozen=True)
class _Scope:
    api_keys: dict[str, str]
    base_urls: dict[str, str]
    credentials: dict[str, str]
    overrides: dict[str, dict[str, Any]]
    client_options: dict[str, dict[str, Any]]
    pricing: dict[str, Price]
    allow_env_fallback: bool | None
    timeout: float | None
    max_retries: int | None


_CONFIG_STACK: ContextVar[tuple[_Scope, ...]] = ContextVar(
    "skllm_config_stack", default=()
)

_T = TypeVar("_T")


def validate_alias(alias: str) -> None:
    if not isinstance(alias, str) or not ALIAS_PATTERN.match(alias):
        raise ValueError(
            f"Invalid estimator alias {alias!r}: aliases must match "
            f"{ALIAS_PATTERN.pattern!r} (lowercase letters, digits and "
            "underscores, starting with a letter)."
        )


def credential_env_vars(provider: str) -> tuple[str, ...]:
    """Canonical credential environment variables, most specific first."""
    return _KEY_ENV_VARS.get(provider, (f"{provider.upper()}{_KEY_SUFFIX.upper()}",))


@contextmanager
def config_context(**settings: Any) -> Iterator[None]:
    """Scope connection settings to the current thread or async task.

    Accepts ``<provider>_api_key`` and ``<provider>_base_url`` for any provider,
    the ``credentials`` (alias -> key) and ``overrides`` (alias -> model params)
    maps, per-provider ``client_options``, a ``pricing`` map (model -> per-million
    input/output prices), ``timeout``, ``max_retries`` and ``allow_env_fallback``.
    Nested contexts override outer ones per setting; ``None`` values are ignored.
    """
    scope = _build_scope(settings)
    token = _CONFIG_STACK.set((*_CONFIG_STACK.get(), scope))
    try:
        yield
    finally:
        _CONFIG_STACK.reset(token)


def resolve_allow_env_fallback() -> bool:
    scoped = _innermost(lambda scope: scope.allow_env_fallback)
    if scoped is not None:
        return scoped
    raw = os.environ.get(ALLOW_ENV_FALLBACK_VAR, "").strip()
    return raw.lower() not in _FALSY if raw else True


def resolve_credential(
    provider: str,
    *,
    explicit: str | None = None,
    alias: str | None = None,
    env_vars: Sequence[str] | None = None,
    required: bool = True,
) -> str | None:
    """Resolve the API key for one request.

    An ``alias`` routes strictly by name — only ``explicit`` (the estimator's own
    ``api_key`` param) outranks it, and an unresolved alias never falls back to
    the provider-level key, which would silently merge separated credentials.
    """
    if explicit:
        return explicit

    allow_env = resolve_allow_env_fallback()
    if alias is not None:
        validate_alias(alias)
        env_var = f"{CREDENTIAL_VAR_PREFIX}{alias.upper()}"
        credential = _innermost(lambda scope: scope.credentials.get(alias))
        if credential is None and allow_env:
            credential = _from_env((env_var,))
        if credential is None and required:
            raise MissingCredentialError(
                f"No credential found for estimator alias {alias!r} (provider "
                f"{provider!r}). Supply it via config_context(credentials="
                f"{{{alias!r}: ...}}){_env_hint(allow_env, (env_var,))}. Aliased "
                "estimators never fall back to the provider-level key."
            )
        return credential

    env_names = tuple(env_vars) if env_vars else credential_env_vars(provider)
    key = _innermost(lambda scope: scope.api_keys.get(provider))
    if key is None and allow_env:
        key = _from_env(env_names)
    if key is None and required:
        raise MissingCredentialError(
            f"No API key found for provider {provider!r}. Supply it via the "
            f"estimator's api_key parameter, config_context({provider}"
            f"{_KEY_SUFFIX}=...){_env_hint(allow_env, env_names)}."
        )
    return key


def resolve_base_url(provider: str, *, explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    base_url = _innermost(lambda scope: scope.base_urls.get(provider))
    if base_url is not None:
        return base_url
    return _from_env(
        _BASE_URL_ENV_VARS.get(
            provider, (f"{provider.upper()}{_BASE_URL_SUFFIX.upper()}",)
        )
    )


def resolve_client_options(provider: str) -> dict[str, Any]:
    return _merged(lambda scope: scope.client_options, provider)


def resolve_overrides(alias: str | None) -> dict[str, Any]:
    return {} if alias is None else _merged(lambda scope: scope.overrides, alias)


def resolve_model_params(
    alias: str | None, stored: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Merge alias-targeted overrides over an estimator's stored ``model_params``.

    Precedence is inverted relative to credentials: an override sits *above* the
    stored value, because overrides exist to beat a stored param whereas
    credential layers exist to fill a missing one.
    """
    return {**(stored or {}), **resolve_overrides(alias)}


def resolve_timeout() -> float:
    timeout = _innermost(lambda scope: scope.timeout)
    return DEFAULT_TIMEOUT if timeout is None else timeout


def resolve_max_retries() -> int:
    max_retries = _innermost(lambda scope: scope.max_retries)
    return DEFAULT_MAX_RETRIES if max_retries is None else max_retries


def resolve_price(model: str) -> Price | None:
    """Look up a model's price, by full ``provider::model`` string or bare id."""
    bare_id = model.rpartition("::")[2]
    return _innermost(
        lambda scope: scope.pricing.get(model, scope.pricing.get(bare_id))
    )


def _innermost(select: Callable[[_Scope], _T | None]) -> _T | None:
    for scope in reversed(_CONFIG_STACK.get()):
        value = select(scope)
        if value is not None:
            return value
    return None


def _merged(
    select: Callable[[_Scope], Mapping[str, dict[str, Any]]], key: str
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for scope in _CONFIG_STACK.get():
        merged.update(select(scope).get(key, {}))
    return merged


def _from_env(names: Sequence[str]) -> str | None:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def _env_hint(allow_env: bool, env_vars: Sequence[str]) -> str:
    names = " or ".join(env_vars)
    if allow_env:
        return f", or the {names} environment variable"
    return (
        "; the environment fallback is disabled (allow_env_fallback=False), so "
        f"{names} is ignored"
    )


def _build_scope(settings: Mapping[str, Any]) -> _Scope:
    api_keys: dict[str, str] = {}
    base_urls: dict[str, str] = {}
    parsed: dict[str, Any] = {name: {} for name in _MAP_SETTINGS}
    parsed.update(dict.fromkeys(_SCALAR_SETTINGS))

    for name, value in settings.items():
        if value is None:
            continue
        if name in parsed:
            parsed[name] = _PARSERS[name](name, value)
        elif name.endswith(_KEY_SUFFIX) and len(name) > len(_KEY_SUFFIX):
            _store_connection_str(api_keys, name[: -len(_KEY_SUFFIX)], name, value)
        elif name.endswith(_BASE_URL_SUFFIX) and len(name) > len(_BASE_URL_SUFFIX):
            _store_connection_str(
                base_urls, name[: -len(_BASE_URL_SUFFIX)], name, value
            )
        else:
            raise ValueError(
                f"Unknown configuration setting {name!r}. Valid settings are "
                f"<provider>{_KEY_SUFFIX}, <provider>{_BASE_URL_SUFFIX}, "
                f"{', '.join(sorted(_MAP_SETTINGS + _SCALAR_SETTINGS))}."
            )

    return _Scope(api_keys=api_keys, base_urls=base_urls, **parsed)


def _as_str(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}.")
    return value


def _store_connection_str(
    target: dict[str, str], key: str, name: str, value: Any
) -> None:
    """Record a connection string, treating blanks as unset like the env layer.

    Otherwise a blank value (``config_context(openai_api_key=os.getenv(...))``
    with the variable absent) would mask the outer layers and reach the provider
    as an empty credential instead of raising :class:`MissingCredentialError`.
    """
    setting = _as_str(name, value).strip()
    if setting:
        target[key] = setting


def _as_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}.")
    for key in value:
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be strings, got {key!r}.")
    return value


def _as_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}.")
    return float(value)


def _parse_credentials(name: str, value: Any) -> dict[str, str]:
    credentials: dict[str, str] = {}
    for alias, key in _as_mapping(name, value).items():
        validate_alias(alias)
        _store_connection_str(credentials, alias, f"{name}[{alias!r}]", key)
    return credentials


def _parse_overrides(name: str, value: Any) -> dict[str, dict[str, Any]]:
    overrides = {}
    for alias, params in _as_mapping(name, value).items():
        validate_alias(alias)
        overrides[alias] = dict(_as_mapping(f"{name}[{alias!r}]", params))
    return overrides


def _parse_client_options(name: str, value: Any) -> dict[str, dict[str, Any]]:
    return {
        provider: dict(_as_mapping(f"{name}[{provider!r}]", options))
        for provider, options in _as_mapping(name, value).items()
    }


def _parse_pricing(name: str, value: Any) -> dict[str, Price]:
    pricing = {}
    for model, price in _as_mapping(name, value).items():
        if isinstance(price, Price):
            pricing[model] = price
            continue
        entry = _as_mapping(f"{name}[{model!r}]", price)
        missing = sorted({"input", "output"} - set(entry))
        if missing:
            raise ValueError(
                f"{name}[{model!r}] is missing {missing}; a price is a mapping "
                "with 'input' and 'output' prices per million tokens."
            )
        pricing[model] = Price(
            input=_as_number(f"{name}[{model!r}]['input']", entry["input"]),
            output=_as_number(f"{name}[{model!r}]['output']", entry["output"]),
        )
    return pricing


def _parse_allow_env_fallback(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}.")
    return value


def _parse_timeout(name: str, value: Any) -> float:
    timeout = _as_number(name, value)
    if timeout <= 0:
        raise ValueError(f"{name} must be positive, got {timeout}.")
    return timeout


def _parse_max_retries(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}.")
    if value < 0:
        raise ValueError(f"{name} must not be negative, got {value}.")
    return value


_PARSERS: dict[str, Any] = {
    "allow_env_fallback": _parse_allow_env_fallback,
    "client_options": _parse_client_options,
    "credentials": _parse_credentials,
    "max_retries": _parse_max_retries,
    "overrides": _parse_overrides,
    "pricing": _parse_pricing,
    "timeout": _parse_timeout,
}
