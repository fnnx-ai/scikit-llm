"""The Google adapter: the Gemini API via ``google-genai`` (not Vertex)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from skllm.backends.base import (
    BackendError,
    BaseBackend,
    ClientCache,
    Message,
    ResolvedConnection,
    Response,
    Usage,
    missing_extra,
    named_unsupported_param,
    parse_json,
)

# Only a genuinely absent package becomes the install hint: a name that a newer
# SDK moved must surface as itself, not as "the extra is not installed". The
# submodule form is deliberate — `from google import genai` reports an absent
# google-genai as a plain ImportError, since `google` is a shared namespace
# package that other Google libraries also populate.
try:
    import httpx
    from google.genai import Client, errors, types
except ModuleNotFoundError as exc:  # pragma: no cover - install-profile dependent
    raise missing_extra("google", "google-genai") from exc

__all__ = ["GoogleBackend"]

#: Accepted key variables, most specific first. ``credential_env_var`` stays the
#: canonical name adapters declare (in bundle manifests, for instance).
KEY_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")

#: Status codes worth another attempt; every other 4xx is the caller's fault.
_TRANSIENT_STATUS = frozenset({408, 429})

_CLIENTS: ClientCache[Client] = ClientCache()


class GoogleBackend(BaseBackend):
    provider: ClassVar[str] = "google"
    credential_env_var: ClassVar[str | None] = "GOOGLE_API_KEY"
    # No reasoning default: Gemini's control is a per-model token budget whose
    # legal minimum differs per model, so a blanket "off" would be rejected more
    # often than honored. Callers set thinking_config through model_params.
    transient_errors: ClassVar[tuple[type[BaseException], ...]] = (
        errors.ServerError,
        httpx.TransportError,
    )

    @property
    def credential_env_vars(self) -> tuple[str, ...]:
        return KEY_ENV_VARS

    def _chat(
        self,
        model: str,
        messages: Sequence[Message],
        params: dict[str, Any],
        settings: ResolvedConnection,
    ) -> Response[str]:
        response = self._generate(model, messages, params, settings)
        return Response(content=_text(response), usage=_usage(response))

    def _structured(
        self,
        model: str,
        messages: Sequence[Message],
        schema: Mapping[str, Any],
        params: dict[str, Any],
        settings: ResolvedConnection,
    ) -> Response[Any]:
        params = {
            "response_mime_type": "application/json",
            "response_json_schema": dict(schema),
            **params,
        }
        response = self._generate(model, messages, params, settings)
        return Response(content=parse_json(_text(response)), usage=_usage(response))

    def _embed(
        self,
        model: str,
        texts: Sequence[str],
        params: dict[str, Any],
        settings: ResolvedConnection,
    ) -> Response[list[list[float]]]:
        response = self._client(settings).models.embed_content(
            model=model,
            contents=list(texts),
            config=types.EmbedContentConfig(**params),
        )
        vectors = [
            embedding.values
            for embedding in response.embeddings or []
            if embedding.values is not None
        ]
        if len(vectors) != len(texts):
            raise BackendError(
                f"The provider returned {len(vectors)} embeddings for {len(texts)} inputs."
            )
        # The embedding endpoint bills characters, not tokens, so there is no
        # token count to report.
        return Response(content=vectors, usage=Usage(requests=1))

    def _is_transient(self, exc: BaseException) -> bool:
        if isinstance(exc, errors.ClientError):
            return exc.code in _TRANSIENT_STATUS
        return super()._is_transient(exc)

    def _unsupported_param(self, exc: BaseException) -> str | None:
        # Only the provider may retract a default: skllm's own BackendError
        # quotes model output, which could otherwise name a param by accident.
        if not isinstance(exc, errors.APIError):
            return None
        return named_unsupported_param(str(exc.message), self.injected_defaults)

    def _generate(
        self,
        model: str,
        messages: Sequence[Message],
        params: dict[str, Any],
        settings: ResolvedConnection,
    ) -> types.GenerateContentResponse:
        system, turns = _split_system(messages)
        config: dict[str, Any] = dict(params)
        if system:
            config["system_instruction"] = system
        return self._client(settings).models.generate_content(
            model=model,
            contents=turns,
            config=types.GenerateContentConfig(**config),
        )

    def _client(self, settings: ResolvedConnection) -> Client:
        return _CLIENTS.get(self.provider, settings, lambda: _build_client(settings))


def _build_client(settings: ResolvedConnection) -> Client:
    """Build a client, folding client options into the SDK's transport config.

    ``http_options`` is this SDK's only home for custom headers and api_version,
    so a caller's own entries are merged over skllm's rather than colliding with
    them as a duplicate keyword argument.
    """
    options = dict(settings.client_options)
    http_options: dict[str, Any] = {
        "base_url": settings.base_url,
        "timeout": int(settings.timeout * 1000),  # this SDK counts in milliseconds
        **dict(options.pop("http_options", None) or {}),
    }
    return Client(
        api_key=settings.api_key,
        http_options=types.HttpOptions(**http_options),
        **options,
    )


def _split_system(messages: Sequence[Message]) -> tuple[str, list[types.Content]]:
    """Lift system messages into the system instruction, as this API expects."""
    system = "\n\n".join(
        message.content for message in messages if message.role == "system"
    )
    turns = [
        types.Content(
            role="model" if message.role == "assistant" else "user",
            parts=[types.Part(text=message.content)],
        )
        for message in messages
        if message.role != "system"
    ]
    return system, turns


def _text(response: types.GenerateContentResponse) -> str:
    if not response.candidates:
        raise BackendError("The provider returned no candidates.")

    text = response.text
    if not text:
        finish_reason = response.candidates[0].finish_reason
        raise BackendError(
            f"The provider returned no text (finish reason: {finish_reason})."
        )
    return text


def _usage(response: types.GenerateContentResponse) -> Usage:
    usage = response.usage_metadata
    if usage is None:
        return Usage(requests=1)
    return Usage(
        requests=1,
        input_tokens=usage.prompt_token_count or 0,
        # Thinking tokens are reported separately but billed as output, and
        # Gemini's current models think by default.
        output_tokens=(usage.candidates_token_count or 0)
        + (usage.thoughts_token_count or 0),
    )
