"""The bounded, order-preserving concurrency behind every batch of requests."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextvars import copy_context
from typing import Any, TypeVar

from tqdm.auto import tqdm

__all__ = ["map_concurrently"]

_T = TypeVar("_T")
_R = TypeVar("_R")


def map_concurrently(
    call: Callable[[_T], _R],
    items: Sequence[_T],
    *,
    max_concurrency: int,
    verbose: int = 0,
    description: str = "",
    fail_fast: bool = False,
) -> list[_R | Exception]:
    """Run ``call`` over ``items`` concurrently, in input order.

    A task that fails contributes its exception in place of a result, unless
    ``fail_fast`` is set, in which case the first failure aborts the batch and
    propagates. Every task runs in its own copy of the caller's contextvars
    context — they do not reach threads on their own, so an active
    ``config_context`` would otherwise be invisible to every worker.
    """
    if not items:
        return []

    outcomes: list[Any] = [None] * len(items)
    with ThreadPoolExecutor(
        max_workers=max(1, min(max_concurrency, len(items))),
        thread_name_prefix="skllm",
    ) as executor:
        futures = {
            executor.submit(copy_context().run, _guarded, call, item): index
            for index, item in enumerate(items)
        }
        try:
            for future in _tracked(
                as_completed(futures), len(items), verbose, description
            ):
                outcome = future.result()
                if fail_fast and isinstance(outcome, Exception):
                    raise outcome
                outcomes[futures[future]] = outcome
        except BaseException:
            executor.shutdown(wait=False, cancel_futures=True)
            raise
    return outcomes


def _guarded(call: Callable[[_T], _R], item: _T) -> _R | Exception:
    try:
        return call(item)
    except Exception as exc:
        return exc


def _tracked(
    futures: Iterable[Future[Any]], total: int, verbose: int, description: str
) -> Iterator[Future[Any]]:
    """Report completions on a progress bar, disabled for silent or tiny batches."""
    with tqdm(total=total, desc=description, disable=verbose < 1 or total < 2) as bar:
        for future in futures:
            yield future
            bar.update(1)
