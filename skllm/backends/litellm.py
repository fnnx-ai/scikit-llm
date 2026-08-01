"""The LiteLLM catch-all adapter: the model id travels to litellm verbatim."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from functools import cache
from operator import itemgetter
from typing import Any, ClassVar

from openai import APIError

from skllm.backends.base import (
    BackendError,
    BaseBackend,
    Message,
    ResolvedConnection,
    Response,
    Usage,
    missing_extra,
    named_unsupported_param,
    parse_json,
    strict_dialect,
)

# Only a genuinely absent package becomes the install hint: a name that a newer
# SDK moved must surface as itself, not as "the extra is not installed".
try:
    import litellm
except ModuleNotFoundError as exc:  # pragma: no cover - install-profile dependent
    raise missing_extra("litellm", "litellm") from exc

__all__ = ["LiteLLMBackend"]

JSON_INSTRUCTION = (
    "Answer with a single JSON value that validates against this JSON Schema. "
    "Output only the JSON, with no prose and no code fences.\n\nJSON Schema:\n{schema}"
)


class LiteLLMBackend(BaseBackend):
    """Every provider litellm speaks, reached through litellm's own config.

    Credentials stay litellm's business: it reads each provider's native
    environment variables itself, so this adapter declares no canonical key and
    only forwards one when a caller supplies it explicitly.
    """

    provider: ClassVar[str] = "litellm"
    credential_env_var: ClassVar[str | None] = None
    transient_errors: ClassVar[tuple[type[BaseException], ...]] = (
        litellm.Timeout,
        litellm.APIConnectionError,
        litellm.RateLimitError,
        litellm.InternalServerError,
        litellm.ServiceUnavailableError,
    )

    def _chat(
        self,
        model: str,
        messages: Sequence[Message],
        params: dict[str, Any],
        settings: ResolvedConnection,
    ) -> Response[str]:
        response = self._complete(model, messages, params, settings)
        return Response(content=_content(response), usage=_usage(response))

    def _structured(
        self,
        model: str,
        messages: Sequence[Message],
        schema: Mapping[str, Any],
        params: dict[str, Any],
        settings: ResolvedConnection,
    ) -> Response[Any]:
        """Use native JSON-schema mode where litellm reports it, else prompt for it."""
        if _supports_response_schema(model):
            params = {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": dict(schema),
                        "strict": strict_dialect(schema),
                    },
                },
                **params,
            }
        else:
            messages = _with_schema_instruction(messages, schema)
        response = self._complete(model, messages, params, settings)
        return Response(content=parse_json(_content(response)), usage=_usage(response))

    def _embed(
        self,
        model: str,
        texts: Sequence[str],
        params: dict[str, Any],
        settings: ResolvedConnection,
    ) -> Response[list[list[float]]]:
        response = litellm.embedding(
            model=model, input=list(texts), **self._call_args(settings, params)
        )
        vectors = [
            item["embedding"] for item in sorted(response.data, key=itemgetter("index"))
        ]
        if len(vectors) != len(texts):
            raise BackendError(
                f"The provider returned {len(vectors)} embeddings for {len(texts)} inputs."
            )
        return Response(content=vectors, usage=_usage(response))

    def _unsupported_param(self, exc: BaseException) -> str | None:
        # Only the provider may retract a default: skllm's own BackendError
        # quotes model output, which could otherwise name a param by accident.
        # litellm builds its whole exception hierarchy on the OpenAI SDK's, so
        # openai.APIError — not litellm.APIError — is the root that covers it.
        if not isinstance(exc, APIError):
            return None
        return named_unsupported_param(str(exc), self.injected_defaults)

    def _complete(
        self,
        model: str,
        messages: Sequence[Message],
        params: dict[str, Any],
        settings: ResolvedConnection,
    ) -> Any:
        return litellm.completion(
            model=model,
            messages=[
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            **self._call_args(settings, params),
        )

    def _call_args(
        self, settings: ResolvedConnection, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Connection settings plus generation params, in one flat kwarg set.

        litellm takes both as keyword arguments, so they are merged rather than
        splatted separately: an overlap (``timeout`` is a legal model_param) would
        otherwise be a local TypeError. User params win, as everywhere.
        """
        args: dict[str, Any] = {
            "timeout": settings.timeout,
            "num_retries": 0,  # scikit-llm owns the retry policy
            **settings.client_options,
        }
        if settings.api_key:
            args["api_key"] = settings.api_key
        if settings.base_url:
            args["api_base"] = settings.base_url
        return {**args, **params}


def _with_schema_instruction(
    messages: Sequence[Message], schema: Mapping[str, Any]
) -> list[Message]:
    """Fold the schema into the last turn instead of appending one.

    litellm fans out to providers that reject consecutive same-role turns, and
    the instruction is just as binding at the end of the existing turn.
    """
    instruction = JSON_INSTRUCTION.format(schema=json.dumps(dict(schema), indent=2))
    if not messages:
        return [Message("user", instruction)]
    last = messages[-1]
    return [*messages[:-1], Message(last.role, f"{last.content}\n\n{instruction}")]


@cache
def _supports_response_schema(model: str) -> bool:
    """Ask litellm once per model; it warns loudly for ids it does not know."""
    try:
        return bool(litellm.supports_response_schema(model=model))
    except Exception:
        return False


def _content(response: Any) -> str:
    if not response.choices:
        raise BackendError("The provider returned no choices.")

    content = response.choices[0].message.content
    if not content:
        raise BackendError("The provider returned an empty message.")
    return str(content)


def _usage(response: Any) -> Usage:
    usage = getattr(response, "usage", None)
    return Usage(
        requests=1,
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
    )
