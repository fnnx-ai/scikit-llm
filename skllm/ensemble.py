"""Multi-model consensus ensembles.

Asking several models and taking the majority answer is a cheap way to trade
requests for reliability, and it is only worth the cost when the members
disagree independently — which is why the panel is free to mix providers,
models and prompts, and why every member keeps its own credentials and error
policy.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace
from math import isclose
from numbers import Real
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import numpy as np
from sklearn.base import ClassifierMixin, clone

from skllm.backends import BackendError
from skllm.classification import (
    _check_labels,
    _distinct,
    _SingleLabelClassifier,
)
from skllm.core import (
    DEFAULT_MAX_CONCURRENCY,
    BaseLLMEstimator,
    Requirement,
    SampleError,
    UsageRecord,
    as_texts,
)
from skllm.core.engine import map_concurrently

if TYPE_CHECKING:
    from typing import Self

__all__ = ["ConsensusClassifier"]

_NO_VOTE = "no member voted: every one of them filled this sample under its own policy"


class _Vote(NamedTuple):
    label: str
    weight: float


class ConsensusClassifier(ClassifierMixin, BaseLLMEstimator):
    """Decide each sample by majority vote across a panel of classifiers.

    The members may mix providers and models freely; the ensemble itself holds
    no model, so everything that describes one — the model string, its
    generation params, its credentials — is configured on the members. Like
    sklearn's ``VotingClassifier`` it fits clones, leaving the panel it was
    constructed with untouched.
    """

    def __init__(
        self,
        estimators: Sequence[_SingleLabelClassifier],
        *,
        weights: Sequence[float] | None = None,
        default_label: str | None = None,
        on_error: str = "fill",
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        verbose: int = 1,
    ) -> None:
        self.estimators = estimators
        self.weights = weights
        self.default_label = default_label
        self.on_error = on_error
        self.max_concurrency = max_concurrency
        self.verbose = verbose

    def fit(self, X: Any, y: Any) -> Self:
        """Fit a clone of every member on the same data, pre-fitted or not."""
        self._check_params()
        classes = _distinct(_check_labels(y))
        estimators = [clone(estimator) for estimator in self.estimators]
        for estimator in estimators:
            estimator.fit(X, y)
        self.estimators_ = estimators
        self.classes_ = classes
        return self

    def predict(self, X: Any) -> np.ndarray:
        """The label with the most votes, ``default_label`` where none was cast."""
        return np.asarray(
            [self._decide(votes) for votes in self._collect_votes(X)], dtype=object
        )

    def predict_proba(self, X: Any) -> np.ndarray:
        """Each class's share of the weight that voted, aligned with ``classes_``."""
        rows = [self._fractions(votes) for votes in self._collect_votes(X)]
        return np.asarray(rows, dtype=float).reshape(len(rows), len(self.classes_))

    def _check_params(self) -> None:
        self._check_batch_params()
        self._check_panel()
        self._check_weights()
        if self.default_label is not None and not isinstance(self.default_label, str):
            raise TypeError(
                "default_label must be a string or None, got "
                f"{type(self.default_label).__name__}."
            )

    def _check_panel(self) -> None:
        if isinstance(self.estimators, str) or not isinstance(
            self.estimators, Sequence
        ):
            raise TypeError(
                "estimators must be a list of skllm classifiers, got "
                f"{type(self.estimators).__name__}."
            )
        if not self.estimators:
            raise ValueError(
                "estimators is empty: a consensus needs at least one classifier."
            )
        for index, estimator in enumerate(self.estimators):
            if not isinstance(estimator, _SingleLabelClassifier):
                raise TypeError(
                    f"estimators[{index}] is a {type(estimator).__name__}, but a "
                    "consensus is voted by single-label skllm classifiers "
                    "(ZeroShotClassifier, FewShotClassifier, "
                    "DynamicFewShotClassifier); multi-label ensembles are out of "
                    "scope."
                )

    def _check_weights(self) -> None:
        if self.weights is None:
            return
        if len(self.weights) != len(self.estimators):
            raise ValueError(
                f"weights has {len(self.weights)} entries but there are "
                f"{len(self.estimators)} estimators: pass one weight per member, "
                "or None to weigh them equally."
            )
        for index, weight in enumerate(self.weights):
            if isinstance(weight, bool) or not isinstance(weight, Real) or weight <= 0:
                raise ValueError(
                    f"weights[{index}] is {weight!r}, but a member's weight must be "
                    "a positive number."
                )

    def _own_requirements(self) -> tuple[Requirement, ...]:
        """None of its own: the ensemble votes, its members call."""
        return ()

    def _collect_votes(self, X: Any) -> list[list[_Vote]]:
        """What each member answered per sample, minus the samples it filled.

        A member that fills ``None`` simply abstains; only a sample no member
        voted for is a failure of the ensemble's own, which its ``on_error``
        policy then decides.
        """
        self._check_ready()
        texts = as_texts(X)
        votes: list[list[_Vote]] = [[] for _ in texts]
        try:
            ballots = self._ballots(texts)
        except BaseException:
            # The panel still paid for the answers it did get before a member's
            # own policy aborted the batch, and a report left describing the
            # previous operation would be read as describing this one.
            self._record_usage(self._spent(), ())
            raise
        for ballot, weight in zip(ballots, self._member_weights(), strict=True):
            for index, label in enumerate(ballot):
                if label is not None:
                    votes[index].append(_Vote(label, weight))

        errors = tuple(
            SampleError(index, _NO_VOTE)
            for index, sample_votes in enumerate(votes)
            if not sample_votes
        )
        # The members counted the samples they themselves filled; these are the
        # ones the ensemble could not answer at all.
        self._record_usage(replace(self._spent(), failures=len(errors)), errors)
        if errors:
            if self.on_error == "raise":
                raise BackendError(
                    f"{len(errors)} of {len(votes)} samples got no vote: every "
                    "member filled them under its own error policy."
                )
            self._warn_filled(len(errors), len(votes))
        return votes

    def _ballots(self, texts: Sequence[str]) -> list[np.ndarray]:
        """Every member's predictions for the whole batch, asked concurrently."""
        outcomes = map_concurrently(
            lambda estimator: estimator.predict(texts),
            self.estimators_,
            max_concurrency=self.max_concurrency,
            verbose=self.verbose,
            description=type(self).__name__,
            # Nothing is filled at this level: a member that raises rather than
            # fills is one whose own policy says a failed sample aborts.
            fail_fast=True,
        )
        return cast("list[np.ndarray]", outcomes)

    def _member_weights(self) -> Sequence[float]:
        return [1.0] * len(self.estimators_) if self.weights is None else self.weights

    def _spent(self) -> UsageRecord:
        """What the members' last operation cost — the ensemble issues no request."""
        return sum(
            (estimator.usage_.last for estimator in self.estimators_), UsageRecord()
        )

    def _decide(self, votes: Sequence[_Vote]) -> str | None:
        if not votes:
            return self.default_label
        weights = _weight_per_label(votes)
        winning = max(weights.values())
        # A tie goes to the earliest-listed member that voted for one of the
        # tied labels, so the same panel decides the same way every run. Summed
        # float weights land a tie a rounding error apart (0.1 + 0.2 != 0.3),
        # which exact comparison would hand to whichever side rounded up.
        return next(
            vote.label for vote in votes if isclose(weights[vote.label], winning)
        )

    def _fractions(self, votes: Sequence[_Vote]) -> list[float]:
        weights = _weight_per_label(votes)
        total = sum(weights.values())
        return [
            weights.get(label, 0.0) / total if total else 0.0 for label in self.classes_
        ]


def _weight_per_label(votes: Iterable[_Vote]) -> dict[str, float]:
    tally: dict[str, float] = {}
    for vote in votes:
        tally[vote.label] = tally.get(vote.label, 0.0) + vote.weight
    return tally
