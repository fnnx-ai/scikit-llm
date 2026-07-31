"""Best-effort OpenTelemetry ``gen_ai`` spans, one per LLM request.

Inert unless the application both installed ``opentelemetry-api`` and configured
a tracer provider: scikit-llm takes no opentelemetry dependency and never
installs or mutates a provider of its own.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from skllm.backends import Usage

__all__ = ["llm_span", "set_usage"]

TRACER_NAME = "scikit-llm"


@contextmanager
def llm_span(operation: str, provider: str, model: str) -> Iterator[Any | None]:
    """Wrap one provider request in a span, or in nothing at all."""
    tracer = _tracer()
    if tracer is None:
        yield None
        return

    with tracer.start_as_current_span(
        f"{operation} {model}",
        attributes={
            "gen_ai.operation.name": operation,
            "gen_ai.provider.name": provider,
            "gen_ai.request.model": model,
        },
    ) as span:
        yield span


def set_usage(span: Any | None, usage: Usage, cost: float | None) -> None:
    """Attach what the request cost; must run before the span is closed."""
    if span is None:
        return
    span.set_attribute("gen_ai.usage.input_tokens", usage.input_tokens)
    span.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)
    if cost is not None:
        span.set_attribute("gen_ai.usage.cost", cost)


def _tracer() -> Any | None:
    """The application's tracer, or ``None`` when nobody configured one.

    A proxy or no-op provider means the application never installed a real one,
    so no span is started at all rather than a non-recording one.
    """
    try:
        from opentelemetry import trace
    except ImportError:
        return None

    provider = trace.get_tracer_provider()
    if isinstance(provider, (trace.NoOpTracerProvider, trace.ProxyTracerProvider)):
        return None
    return trace.get_tracer(TRACER_NAME)
