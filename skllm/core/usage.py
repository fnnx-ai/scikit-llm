"""Per-operation and cumulative usage accounting, with optional cost estimates.

Requests are collected through a contextvar rather than passed around: the
request engine runs every worker in a copy of the caller's context, so the calls
of one operation land in that operation's collector no matter which thread — and
concurrent operations on a shared estimator never mix.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from skllm.backends import Usage
from skllm.config import Price, resolve_price

__all__ = [
    "UsageCollector",
    "UsageRecord",
    "UsageReport",
    "collect_usage",
    "estimate_cost",
    "record_request",
]

TOKENS_PER_PRICE_UNIT = 1_000_000


@dataclass(frozen=True)
class UsageRecord:
    """What a set of requests cost, as far as the provider reports it."""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    failures: int = 0
    cost: float | None = None

    def __add__(self, other: UsageRecord) -> UsageRecord:
        return UsageRecord(
            requests=self.requests + other.requests,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            failures=self.failures + other.failures,
            cost=_total_cost(self.cost, other.cost),
        )


@dataclass(frozen=True)
class UsageReport:
    """What the most recent operation cost, and everything since construction."""

    last: UsageRecord = UsageRecord()
    total: UsageRecord = UsageRecord()


class UsageCollector:
    """Thread-safe accumulation of the requests issued by one operation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._record = UsageRecord()

    @property
    def record(self) -> UsageRecord:
        with self._lock:
            return self._record

    def add(self, record: UsageRecord) -> None:
        with self._lock:
            self._record = self._record + record


_ACTIVE: ContextVar[UsageCollector | None] = ContextVar(
    "skllm_usage_collector", default=None
)


@contextmanager
def collect_usage() -> Iterator[UsageCollector]:
    """Collect every request issued in this scope, worker threads included."""
    collector = UsageCollector()
    token = _ACTIVE.set(collector)
    try:
        yield collector
    finally:
        _ACTIVE.reset(token)


def record_request(usage: Usage, model: str) -> float | None:
    """Add one provider response to the active collection, returning its cost."""
    cost = estimate_cost(usage, resolve_price(model))
    collector = _ACTIVE.get()
    if collector is not None:
        collector.add(
            UsageRecord(
                requests=usage.requests,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost=cost,
            )
        )
    return cost


def estimate_cost(usage: Usage, price: Price | None) -> float | None:
    """Price one response, or ``None`` when the model has no configured price."""
    if price is None:
        return None
    tokens = usage.input_tokens * price.input + usage.output_tokens * price.output
    return tokens / TOKENS_PER_PRICE_UNIT


def _total_cost(left: float | None, right: float | None) -> float | None:
    """Sum two costs, keeping ``None`` only when neither side was priced."""
    if left is None:
        return right
    if right is None:
        return left
    return left + right
