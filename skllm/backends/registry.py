"""The provider registry and ``provider::model_id`` dispatch."""

from __future__ import annotations

import re
import threading
from importlib import import_module
from typing import NamedTuple

from skllm.backends.base import Backend

__all__ = [
    "ModelRef",
    "available_providers",
    "get_backend",
    "parse_model",
    "register_backend",
]

PROVIDER_SEPARATOR = "::"
DEFAULT_PROVIDER = "openai"

# \Z, not $: a trailing newline would otherwise pass and register a prefix no
# model string can ever name.
_PROVIDER_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*\Z")

# Imported on first use so a missing provider extra only ever raises for the
# provider a model string actually asks for.
_BUILTIN_BACKENDS: dict[str, tuple[str, str]] = {
    "openai": ("skllm.backends.openai", "OpenAIBackend"),
    "ollama": ("skllm.backends.openai", "OllamaBackend"),
    "anthropic": ("skllm.backends.anthropic", "AnthropicBackend"),
    "google": ("skllm.backends.google", "GoogleBackend"),
    "litellm": ("skllm.backends.litellm", "LiteLLMBackend"),
}

_BACKENDS: dict[str, Backend] = {}
_LOCK = threading.Lock()


class ModelRef(NamedTuple):
    provider: str
    model_id: str


def parse_model(model: str) -> ModelRef:
    """Split a ``provider::model_id`` string; a bare id means OpenAI."""
    if not isinstance(model, str) or not model.strip():
        raise ValueError(
            f"Invalid model string {model!r}: expected 'provider{PROVIDER_SEPARATOR}"
            "model_id' (or a bare model id for OpenAI)."
        )

    provider, separator, model_id = model.partition(PROVIDER_SEPARATOR)
    if not separator:
        return ModelRef(DEFAULT_PROVIDER, provider.strip())

    provider, model_id = provider.strip(), model_id.strip()
    if not provider or not model_id:
        raise ValueError(
            f"Invalid model string {model!r}: expected 'provider{PROVIDER_SEPARATOR}"
            "model_id' with both parts non-empty."
        )
    return ModelRef(provider, model_id)


def register_backend(prefix: str, backend: Backend) -> None:
    """Make ``backend`` reachable as ``prefix::model_id``, replacing any existing one."""
    if not isinstance(prefix, str) or not _PROVIDER_PATTERN.match(prefix):
        raise ValueError(
            f"Invalid provider prefix {prefix!r}: prefixes must match "
            f"{_PROVIDER_PATTERN.pattern!r}."
        )
    if not isinstance(backend, Backend):
        raise TypeError(
            f"{type(backend).__name__} is not a backend adapter: it must expose "
            "'provider' and 'credential_env_var' attributes and chat(), "
            "structured() and embed() methods."
        )
    with _LOCK:
        _BACKENDS[prefix] = backend


def get_backend(provider: str) -> Backend:
    """Return the adapter registered for ``provider``, importing built-ins lazily."""
    with _LOCK:
        registered = _BACKENDS.get(provider)
    if registered is not None:
        return registered

    target = _BUILTIN_BACKENDS.get(provider)
    if target is None:
        raise ValueError(
            f"Unknown model provider {provider!r}. Available providers: "
            f"{', '.join(available_providers())}."
        )

    module_name, class_name = target
    backend: Backend = getattr(import_module(module_name), class_name)()
    with _LOCK:
        return _BACKENDS.setdefault(provider, backend)


def available_providers() -> tuple[str, ...]:
    with _LOCK:
        return tuple(sorted({*_BUILTIN_BACKENDS, *_BACKENDS}))
