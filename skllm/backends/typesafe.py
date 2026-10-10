"""The TypeSafe adapter: Jev and other System One models, typed decisions only.

A System One model never generates text — it answers typed questions about a
state with calibrated probabilities — so this adapter offers ``evaluate`` and
rejects chat, structured outputs and embeddings outright.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial
from types import MappingProxyType
from typing import Any, ClassVar, cast

from skllm.backends.base import (
    BackendError,
    BaseBackend,
    ClientCache,
    Connection,
    Message,
    ResolvedConnection,
    Response,
    Usage,
    missing_extra,
)

# Only a genuinely absent package becomes the install hint: a name that a newer
# SDK moved must surface as itself, not as "the extra is not installed".
try:
    from typesafe_sdk import (
        RetryPolicy,
        TypeSafeAPIConnectionError,
        TypeSafeAPIError,
        TypeSafeClient,
        TypeSafeInternalServerError,
        TypeSafeRateLimitError,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - install-profile dependent
    raise missing_extra("typesafe", "typesafe-sdk") from exc

__all__ = ["TypeSafeBackend"]

DEFAULT_BASE_URL = "https://api.typesafe.ai"

_NO_TEXT = (
    "{model!r} is a System One model: it answers typed questions with "
    "probabilities and never generates text, so it cannot serve {operation}. "
    "Use it with the classifiers or RubricScorer, or pick a text model."
)

_CLIENTS: ClientCache[TypeSafeClient] = ClientCache()


class TypeSafeBackend(BaseBackend):
    provider: ClassVar[str] = "typesafe"
    credential_env_var: ClassVar[str | None] = "TYPESAFE_API_KEY"
    default_base_url: ClassVar[str | None] = DEFAULT_BASE_URL
    # Answers are calibrated probabilities, not samples: there is no
    # temperature to pin, and nothing to strip for sampled calls either.
    injected_defaults: ClassVar[Mapping[str, Any]] = MappingProxyType({})
    determinism_defaults: ClassVar[frozenset[str]] = frozenset()
    # 529 "overloaded" is a 5xx, which the SDK maps to its server error.
    transient_errors: ClassVar[tuple[type[BaseException], ...]] = (
        TypeSafeAPIConnectionError,
        TypeSafeRateLimitError,
        TypeSafeInternalServerError,
    )
    max_choice_options: ClassVar[int] = 255
    max_score_levels: ClassVar[int] = 10

    def evaluate(
        self,
        model: str,
        state: Any,
        questions: Mapping[str, Mapping[str, Any]],
        *,
        params: Mapping[str, Any] | None = None,
        connection: Connection | None = None,
    ) -> Response[dict[str, dict[str, Any]]]:
        settings = self.resolve(connection)
        return self._with_retry(
            partial(
                self._evaluate, model, state, questions, dict(params or {}), settings
            ),
            settings,
        )

    def _evaluate(
        self,
        model: str,
        state: Any,
        questions: Mapping[str, Mapping[str, Any]],
        params: dict[str, Any],
        settings: ResolvedConnection,
    ) -> Response[dict[str, dict[str, Any]]]:
        # model_params ride in extra_body, so every one of them reaches the API
        # and an unsupported one surfaces the provider's own error.
        response = self._client(settings).system_one(
            state,
            cast("Any", questions),
            model=model,
            extra_body=params or None,
        )
        answers = {
            name: answer.model_dump(mode="json")
            for name, answer in response.answers.items()
        }
        # The SDK drops answer kinds newer than itself instead of failing.
        if missing := sorted(set(questions) - answers.keys()):
            raise BackendError(
                f"The provider returned no usable answer for {missing}; upgrading "
                "typesafe-sdk may add support for the answer type it sent."
            )
        usage = Usage(
            requests=1,
            input_tokens=response.usage.input_tokens or 0,
            output_tokens=response.usage.output_tokens or 0,
        )
        return Response(content=answers, usage=usage)

    def _chat(
        self,
        model: str,
        messages: Sequence[Message],
        params: dict[str, Any],
        settings: ResolvedConnection,
    ) -> Response[str]:
        raise BackendError(_NO_TEXT.format(model=model, operation="chat requests"))

    def _structured(
        self,
        model: str,
        messages: Sequence[Message],
        schema: Mapping[str, Any],
        params: dict[str, Any],
        settings: ResolvedConnection,
    ) -> Response[Any]:
        raise BackendError(
            _NO_TEXT.format(model=model, operation="free-form structured outputs")
        )

    def _retry_after(self, exc: BaseException) -> float | None:
        """The SDK keeps response headers on the error itself."""
        if not isinstance(exc, TypeSafeAPIError):
            return None
        for header, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
            raw = exc.headers.get(header)
            if raw:
                try:
                    return float(raw) * scale
                except ValueError:  # the HTTP-date form; fall back to backoff
                    continue
        return None

    def _client(self, settings: ResolvedConnection) -> TypeSafeClient:
        return _CLIENTS.get(
            self.provider,
            settings,
            lambda: TypeSafeClient(
                api_key=settings.api_key,
                base_url=settings.base_url,
                timeout=settings.timeout,
                retry=RetryPolicy(max_retries=0),  # scikit-llm owns the retry policy
                **settings.client_options,
            ),
        )
