"""The per-sample error policy shared by every estimator."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ON_ERROR_POLICIES", "SampleError"]

#: ``"fill"`` gives a failed sample the estimator's documented fill value and
#: reports it; ``"raise"`` aborts the batch with the underlying provider error.
ON_ERROR_POLICIES = ("fill", "raise")


@dataclass(frozen=True)
class SampleError:
    """Why one sample of a batch carries a fill value instead of an answer.

    The reason is kept as text rather than as the exception: reports travel with
    the estimator through pickling and bundling, and provider exceptions are not
    reliably picklable.
    """

    index: int
    error: str
