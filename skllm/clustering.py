"""Naming the clusters a clustering model produced.

Every name is decided once, at ``fit``: one call per cluster over a sample of
the texts it holds. ``transform`` is then a lookup from cluster id to name,
which is what makes the naming all-or-nothing — a cluster left unnamed would
answer every later ``transform`` with a hole no caller could tell from a name.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.utils import check_random_state

from skllm.backends import BackendError, Message
from skllm.core import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_MAX_CONCURRENCY,
    BasePromptEstimator,
    Requirement,
    UsageRecord,
    as_texts,
    check_count,
)
from skllm.prompts import CLUSTER_NAMING, render_exemplars

if TYPE_CHECKING:
    from typing import Self

__all__ = ["ClusterLabeler"]

DEFAULT_N_EXEMPLARS = 10

#: What density-based clusterers assign to the samples they grouped with
#: nothing. It stands for no cluster, so it is never named.
_NOISE_ID = -1


class ClusterLabeler(TransformerMixin, BasePromptEstimator):
    """Name every cluster from a sample of its texts, then map ids to names.

    ``fit(X, y)`` takes the texts and the cluster ids a clusterer assigned them;
    ``transform(y)`` takes ids alone, since it is the clustering that was named
    rather than the texts. The noise id ``-1`` maps to ``None`` without ever
    reaching a model, and an id that was not present at ``fit`` raises.
    """

    #: A cluster nobody could name aborts the fit, whatever ``on_error`` says:
    #: every later transform reads the mapping this leaves behind.
    _all_or_nothing: ClassVar[bool] = True

    prompt_placeholders: ClassVar[frozenset[str]] = frozenset({"texts"})
    default_prompt_template: ClassVar[str] = CLUSTER_NAMING

    def __init__(
        self,
        model: str = DEFAULT_CHAT_MODEL,
        *,
        n_exemplars: int = DEFAULT_N_EXEMPLARS,
        random_state: int | np.random.RandomState | None = None,
        prompt_template: str | None = None,
        model_params: Mapping[str, Any] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        alias: str | None = None,
        on_error: str = "fill",
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        verbose: int = 1,
    ) -> None:
        super().__init__(
            model=model,
            prompt_template=prompt_template,
            model_params=model_params,
            api_key=api_key,
            base_url=base_url,
            alias=alias,
            on_error=on_error,
            max_concurrency=max_concurrency,
            verbose=verbose,
        )
        self.n_exemplars = n_exemplars
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> Self:
        """Name every cluster in ``y`` from a sample of the texts it holds."""
        self._check_params()
        exemplars = self._exemplars(as_texts(X), _as_cluster_ids(y))
        names = self._run_batch(list(exemplars.values()), self._name, fill=str)
        self.names_ = dict(zip(exemplars, names, strict=True))
        return self

    def transform(self, y: Any) -> np.ndarray:
        """Map every cluster id to its name, ``None`` for the noise id.

        The names were decided at ``fit``, so this issues no request at all —
        but it is still an operation, and a report left describing the naming
        run would be read as describing this lookup. It reports nothing spent,
        leaving the cumulative total intact.
        """
        self._check_ready()
        names = [self._name_of(cluster) for cluster in _as_cluster_ids(y)]
        self._record_usage(UsageRecord(), ())
        return np.asarray(names, dtype=object)

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        """Name the clustering just fitted.

        The inherited contract would transform ``X``, but the texts are what
        this estimator names clusters *from*; the clustering in ``y`` is what it
        names.
        """
        return self.fit(X, y).transform(y)

    def _check_params(self) -> None:
        super()._check_params()
        check_count("n_exemplars", self.n_exemplars, required=True)

    def _own_requirements(self) -> tuple[Requirement, ...]:
        """None: the names are decided at fit, so serving one issues no request."""
        return ()

    def _exemplars(
        self, texts: Sequence[str], clusters: Sequence[int]
    ) -> dict[int, tuple[str, ...]]:
        """Up to ``n_exemplars`` texts per cluster, the noise id left out."""
        if len(texts) != len(clusters):
            raise ValueError(
                f"X has {len(texts)} samples but y has {len(clusters)} cluster ids: "
                "every text must carry the id of the cluster it was assigned to."
            )
        members: dict[int, list[str]] = {}
        for text, cluster in zip(texts, clusters, strict=True):
            if cluster != _NOISE_ID:
                members.setdefault(cluster, []).append(text)

        rng = check_random_state(self.random_state)
        # Clusters are drawn from in id order, so that one seed samples the same
        # exemplars however the clusters happened to be encountered.
        return {
            cluster: _sample(cluster_texts, self.n_exemplars, rng)
            for cluster, cluster_texts in sorted(members.items())
        }

    def _name(self, exemplars: Sequence[str]) -> str:
        prompt = self._prompt(texts=render_exemplars(exemplars))
        name = self._chat([Message("user", prompt)]).strip()
        if not name:
            raise BackendError(
                "The model answered with an empty name for a cluster of "
                f"{len(exemplars)} samples."
            )
        return name

    def _name_of(self, cluster: int) -> str | None:
        if cluster == _NOISE_ID:
            return None
        if cluster not in self.names_:
            raise ValueError(
                f"Cluster {cluster} was not seen at fit, so it has no name; the "
                f"clusters named at fit are {sorted(self.names_)}."
            )
        name: str = self.names_[cluster]
        return name


def _sample(
    texts: Sequence[str], count: int, rng: np.random.RandomState
) -> tuple[str, ...]:
    """Up to ``count`` of a cluster's texts, kept in the order they appeared."""
    positions = rng.permutation(len(texts))[:count]
    return tuple(texts[position] for position in sorted(positions))


def _as_cluster_ids(y: Any) -> list[int]:
    """Normalize cluster ids (list, numpy array or pandas Series) to a list."""
    if y is None or isinstance(y, (str, int, float)):
        raise TypeError(
            f"y must be a sequence of cluster ids, got {type(y).__name__}; pass the "
            "labels a clusterer assigned, e.g. KMeans().fit_predict(vectors)."
        )
    dimensions = getattr(y, "ndim", 1)
    if dimensions != 1:
        raise TypeError(
            f"y must be one-dimensional, got a {dimensions}-dimensional array."
        )

    values = y.tolist() if hasattr(y, "tolist") else list(y)
    clusters = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(
                f"y[{index}] is a {type(value).__name__}, but cluster ids are "
                "integers, as every sklearn clusterer assigns them."
            )
        clusters.append(int(value))
    return clusters
