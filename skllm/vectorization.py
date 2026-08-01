"""The embedding transformer, and which embedding model each provider defaults to."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
from sklearn.base import TransformerMixin

from skllm.core import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MAX_CONCURRENCY,
    BaseLLMEstimator,
    StatelessMixin,
    as_texts,
    check_count,
)

__all__ = ["DEFAULT_EMBEDDING_MODELS", "Vectorizer"]

DEFAULT_BATCH_SIZE = 32

#: The embedding model an estimator falls back to per chat provider. Only
#: providers whose embedding model is a fixed, hosted id belong here: anthropic
#: has no embeddings at all, and for ollama, litellm and custom backends the id
#: depends on the deployment, so guessing one would fail at the first request.
DEFAULT_EMBEDDING_MODELS: dict[str, str] = {
    "openai": DEFAULT_EMBEDDING_MODEL,
    "google": "google::gemini-embedding-001",
}


class Vectorizer(TransformerMixin, StatelessMixin, BaseLLMEstimator):
    """Embed texts through a provider's embedding model.

    Embeddings are all-or-nothing: a failed batch raises even under
    ``on_error="fill"``, because a zero-vector fill is indistinguishable from a
    real embedding to whatever consumes it.
    """

    _all_or_nothing: ClassVar[bool] = True

    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
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
            model_params=model_params,
            api_key=api_key,
            base_url=base_url,
            alias=alias,
            on_error=on_error,
            max_concurrency=max_concurrency,
            verbose=verbose,
        )
        self.batch_size = batch_size

    def transform(self, X: Any) -> np.ndarray:
        """Embed every text, returning one row per sample in input order."""
        self._check_ready()
        texts = as_texts(X)
        batches = [
            texts[start : start + self.batch_size]
            for start in range(0, len(texts), self.batch_size)
        ]
        embedded: list[list[list[float]]] = self._run_batch(
            batches, self._embed, fill=list
        )
        vectors = [vector for batch in embedded for vector in batch]
        # An empty batch goes through the request path too — issuing nothing, but
        # leaving a usage report that describes this operation rather than the
        # previous one. Only its width has to be spelled out: numpy cannot infer
        # one from no rows, and the result is documented as two-dimensional.
        if not vectors:
            return np.empty((0, 0), dtype=float)
        return np.asarray(vectors, dtype=float)

    def _request_params(self) -> dict[str, Any]:
        """Embedding requests take no alias-targeted override.

        Overrides carry generation params, and an alias is not always the
        vectorizer's alone: the one a ``DynamicFewShotClassifier`` builds shares
        the classifier's, so that both authenticate as one estimator. A
        temperature aimed at the classifier must not arrive here verbatim.
        """
        return dict(self.model_params or {})

    def _check_params(self) -> None:
        super()._check_params()
        check_count("batch_size", self.batch_size, required=True)
