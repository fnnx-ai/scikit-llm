"""The Anthropic adapter: chat and tool-use structured outputs, no embeddings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from inspect import signature
from typing import TYPE_CHECKING, Any, ClassVar, cast

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
    request_args,
)

# Only a genuinely absent package becomes the install hint: a name that a newer
# SDK moved must surface as itself, not as "the extra is not installed".
try:
    from anthropic import (
        Anthropic,
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )
    from anthropic.resources.messages import Messages
except ModuleNotFoundError as exc:  # pragma: no cover - install-profile dependent
    raise missing_extra("anthropic", "anthropic") from exc

if TYPE_CHECKING:
    from anthropic.types import Message as AnthropicMessage
    from anthropic.types import MessageParam

__all__ = ["AnthropicBackend"]

DEFAULT_MAX_TOKENS = 4096

#: The name the forced tool is given; the model's arguments are the answer.
RESPONSE_TOOL = "respond"

#: "Overloaded" — a 5xx in spirit, but its own status code, and only some SDK
#: versions give it a dedicated exception class.
OVERLOADED_STATUS = 529

_MESSAGE_ARGS = frozenset(signature(Messages.create).parameters)

_CLIENTS: ClientCache[Anthropic] = ClientCache()


class AnthropicBackend(BaseBackend):
    provider: ClassVar[str] = "anthropic"
    credential_env_var: ClassVar[str | None] = "ANTHROPIC_API_KEY"
    # No reasoning default beyond the inherited temperature 0: extended thinking
    # is opt-in on this API, so there is nothing to switch off.
    # A tool's input_schema is passed to the model as written, so array bounds
    # reach it rather than being rejected.
    schema_constraints: ClassVar[frozenset[str]] = frozenset({"maxItems"})
    transient_errors: ClassVar[tuple[type[BaseException], ...]] = (
        APITimeoutError,
        APIConnectionError,
        RateLimitError,
        InternalServerError,
    )

    def _chat(
        self,
        model: str,
        messages: Sequence[Message],
        params: dict[str, Any],
        settings: ResolvedConnection,
    ) -> Response[str]:
        message = self._create(model, messages, params, settings)
        text = "\n".join(
            block.text for block in message.content if block.type == "text"
        )
        if not text:
            raise BackendError("The provider returned no text content.")
        return Response(content=text, usage=_usage(message))

    def _structured(
        self,
        model: str,
        messages: Sequence[Message],
        schema: Mapping[str, Any],
        params: dict[str, Any],
        settings: ResolvedConnection,
    ) -> Response[Any]:
        """Constrain the answer by forcing a tool call whose input is the schema."""
        params = {
            "tools": [
                {
                    "name": RESPONSE_TOOL,
                    "description": "Return the answer in the required structure.",
                    "input_schema": dict(schema),
                }
            ],
            "tool_choice": {"type": "tool", "name": RESPONSE_TOOL},
            **params,
        }
        message = self._create(model, messages, params, settings)
        arguments = next(
            (block.input for block in message.content if block.type == "tool_use"), None
        )
        if arguments is None:
            raise BackendError(
                f"The model answered without calling the {RESPONSE_TOOL!r} tool."
            )
        return Response(content=arguments, usage=_usage(message))

    def _is_transient(self, exc: BaseException) -> bool:
        return (
            isinstance(exc, APIStatusError) and exc.status_code == OVERLOADED_STATUS
        ) or super()._is_transient(exc)

    def _injected_params(
        self,
        model: str,
        user_params: Mapping[str, Any],
        deterministic: bool,
        stripped: set[str],
    ) -> dict[str, Any]:
        """Skip the temperature default when the caller turned thinking on.

        This API rejects a caller-set temperature together with extended
        thinking. Letting the error-driven fallback discover that would cache the
        rejection against the *model*, quietly disarming determinism for every
        later non-thinking call — the trigger is the caller's param, not the
        model.
        """
        injected = super()._injected_params(model, user_params, deterministic, stripped)
        if _thinking_enabled(user_params):
            injected.pop("temperature", None)
        return injected

    def _unsupported_param(self, exc: BaseException) -> str | None:
        # Only the provider may retract a default: skllm's own BackendError
        # quotes model output, which could otherwise name a param by accident.
        if not isinstance(exc, APIStatusError):
            return None
        error = exc.body.get("error") if isinstance(exc.body, dict) else None
        message = str(error.get("message", "")) if isinstance(error, dict) else str(exc)
        return named_unsupported_param(message, self.injected_defaults)

    def _create(
        self,
        model: str,
        messages: Sequence[Message],
        params: dict[str, Any],
        settings: ResolvedConnection,
    ) -> AnthropicMessage:
        system, turns = _split_system(messages)
        args = request_args(params, _MESSAGE_ARGS)
        if system:
            args["system"] = system
        return cast(
            "AnthropicMessage",  # never a Stream: skllm does not set stream=True
            self._client(settings).messages.create(
                model=model,
                messages=turns,
                # The API requires an output budget; skllm supplies a generous one
                # so estimators need not carry a token count of their own.
                max_tokens=args.pop("max_tokens", DEFAULT_MAX_TOKENS),
                **args,
            ),
        )

    def _client(self, settings: ResolvedConnection) -> Anthropic:
        return _CLIENTS.get(
            self.provider,
            settings,
            lambda: Anthropic(
                api_key=settings.api_key,
                base_url=settings.base_url,
                timeout=settings.timeout,
                max_retries=0,  # scikit-llm owns the retry policy
                **settings.client_options,
            ),
        )


def _thinking_enabled(params: Mapping[str, Any]) -> bool:
    thinking = params.get("thinking")
    return isinstance(thinking, Mapping) and thinking.get("type") != "disabled"


def _split_system(messages: Sequence[Message]) -> tuple[str, list[MessageParam]]:
    """Lift system messages out of the turn list, as this API expects."""
    system = "\n\n".join(
        message.content for message in messages if message.role == "system"
    )
    turns: list[MessageParam] = [
        {
            "role": "assistant" if message.role == "assistant" else "user",
            "content": message.content,
        }
        for message in messages
        if message.role != "system"
    ]
    return system, turns


def _usage(message: AnthropicMessage) -> Usage:
    return Usage(
        requests=1,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
    )
