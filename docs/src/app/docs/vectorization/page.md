---
title: Vectorization
---

`Vectorizer` turns texts into embedding vectors through a provider's embedding model. The result is an ordinary numeric feature matrix that any scikit-learn estimator can consume: a classifier, a clusterer, a nearest-neighbor index. [Dynamic few-shot classification](/docs/dynamic-few-shot-classification) and [Cluster labeling](/docs/cluster-labeling) build on it internally.

```python
from skllm import Vectorizer
from skllm.datasets import get_classification_dataset

X, y = get_classification_dataset()

vectors = Vectorizer().fit_transform(X)
vectors.shape   # (30, d) — one row per text, d fixed by the embedding model
```

The default model is `openai::text-embedding-3-small`, the one estimator whose default is an embedding model rather than the default chat model. Any provider that serves embeddings can be named instead, for example `Vectorizer(model="google::gemini-embedding-001")`. Anthropic serves none, and a `Vectorizer` pointed at it raises a `BackendError` saying so at the first request. See [Models and providers](/docs/models-and-providers).

`Vectorizer` is stateless: `fit` validates the constructor parameters, issues no request and returns `self`, so `transform` works on a freshly constructed instance and `fit_transform` is `fit` followed by `transform`.

## Requests and batching

`batch_size` (default `32`) is how many texts travel in one embedding request. A call over `n` texts becomes `ceil(n / batch_size)` requests, and `max_concurrency` bounds how many of those requests are in flight — batches, not individual texts. Rows come back in input order regardless of the order the batches complete in.

Options the embedding endpoint accepts are passed through `model_params` verbatim, such as `Vectorizer(model_params={"dimensions": 256})` for OpenAI's shortened vectors.

## Output

`transform` returns a `numpy.ndarray` of `dtype=float64` and shape `(n_samples, d)`, where `d` is whatever width the embedding model returns. An empty input returns an array of shape `(0, 0)` and issues no request, since a width cannot be inferred from zero rows.

## Failures are all or nothing

A failed embedding request aborts the whole `transform`, and it does so even under the default `on_error="fill"` policy.

```python
vectorizer = Vectorizer()   # on_error="fill"
vectorizer.transform(X)     # raises whatever the provider raised
vectorizer.errors_          # () — nothing was filled, so nothing was recorded
```

Every other estimator has a fill value that reads as "no answer here": an empty string, `None`, `NaN`. An embedding has no such value. The only candidate is a zero vector, and a zero vector is a well-formed row to whatever consumes the matrix — it would be trained on, clustered and compared like any other, contributing a point at the origin that no text belongs to. The failure would surface as a slightly worse downstream model rather than as an error, which is why the exception propagates instead. `usage_` still accounts for the requests that completed before the abort. See [Batch execution and error handling](/docs/batch-execution-and-error-handling).

## Embeddings in a pipeline

Because the output is a plain float matrix, the vectorizer slots in ahead of a classic scikit-learn estimator that never sees the text.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from skllm import Vectorizer
from skllm.datasets import get_classification_dataset

X, y = get_classification_dataset()

pipeline = Pipeline(
    [
        ("embed", Vectorizer(batch_size=32)),
        ("classify", LogisticRegression()),
    ]
)
pipeline.fit(X, y)
pipeline.predict(["The plot dragged, but the soundtrack was worth it."])
```

The language model supplies the representation and nothing else: the decision is `LogisticRegression`'s, trained on the labels, and it costs no requests to make.

Fitting embeds the training texts and every `predict` embeds its own input, so both issue requests. A grid search or a cross-validation loop over this pipeline re-embeds the same texts once per fit. Where that matters, embed once with `Vectorizer().transform(X)` and tune the downstream estimator on the resulting matrix.

## Retrieval for dynamic few-shot classification

`DynamicFewShotClassifier` selects its demonstrations by embedding similarity and takes a `Vectorizer` as its `vectorizer` parameter. Which providers supply a default one, and what an explicit one inherits, is covered in [Dynamic few-shot classification](/docs/dynamic-few-shot-classification).
