---
title: Dynamic few-shot classification
---

`DynamicFewShotClassifier` builds a different prompt for every sample. Instead of demonstrating the task with the whole training set, it retrieves the training examples nearest to the text being classified and shows only those. Every class contributes its own nearest examples, so the prompt stays the same size whether the training set holds thirty texts or thirty thousand.

```python
from skllm import DynamicFewShotClassifier
from skllm.datasets import get_classification_dataset

X, y = get_classification_dataset()

clf = DynamicFewShotClassifier(n_examples=2)
clf.fit(X, y)
labels = clf.predict(X)
```

`fit` pairs `X` with `y` position by position, so the two must have the same length, and labels must be strings. It embeds every training text and builds one neighbor index per class. `classes_` is the sorted set of labels seen. `predict` returns a `numpy` array of `dtype=object` holding one label per sample, and `predict_proba` returns a float array of shape `(n_samples, len(classes_))` whose columns follow `classes_`. A sample whose every attempt failed is filled with `default_label` (`None` unless you name one), gets an all-zero probability row, and is listed in `errors_`, as described in [Batch execution and error handling](/docs/batch-execution-and-error-handling).

Everything else matches [Few-shot classification](/docs/few-shot-classification): the same default template, the same `labels`, `text` and `examples` placeholders, and the same structured-output constraint that keeps every answer inside `classes_`. What differs is which examples fill the `examples` slot.

## Which examples are retrieved

`fit` groups the training pairs by label, embeds all of their texts in one pass, and fits a separate `sklearn.neighbors.NearestNeighbors` index over each class's vectors with cosine distance. `predict` embeds the incoming batch once, then queries every class index for that sample's `n_examples` nearest texts. The retrieved examples are assembled class by class in sorted label order, so one prompt carries at most `n_examples * len(classes_)` demonstrations, and a class with fewer training texts than `n_examples` contributes all it has. `n_examples` defaults to 3.

Splitting the index per class is what keeps the demonstrations balanced. A single index over the whole training set would return whichever examples sit nearest overall, which for an ambiguous sample can be several texts of one label and none of another. A label with no demonstration is one the model has seen only as a name in the list of categories. Per-class retrieval gives every label the same number of slots and lets the sample decide only which of that label's texts fill them.

Neither the metric nor the index type is configurable, and nearest in embedding space is not the same as most instructive. Retrieval finds texts that resemble the sample, which for a sample sitting between two labels can mean examples that are themselves hard to classify.

{% callout title="Why cosine" %}
Cosine distance ranks embeddings by direction. Direction is what similarity between two embeddings is normally taken to mean, while their magnitude varies with the length of the text.
{% /callout %}

## Choosing the embedding model

The `vectorizer` parameter takes a scikit-llm [`Vectorizer`](/docs/vectorization). Left as `None`, the classifier builds one at fit time from its own provider's default embedding model — `openai::text-embedding-3-small` for `openai`, `google::gemini-embedding-001` for `google`. That vectorizer inherits the classifier's `api_key`, `base_url`, `alias`, `max_concurrency` and `verbose`, so retrieval authenticates exactly as the chat calls do.

No other provider has a default. Anthropic serves no embedding model at all, and for `ollama`, `litellm` and custom backends the model id depends on the deployment. Fitting one of those without a `vectorizer` raises `ValueError`, so pass one explicitly. The embedding provider does not have to match the chat provider:

```python
from skllm import DynamicFewShotClassifier, Vectorizer

clf = DynamicFewShotClassifier(
    model="anthropic::claude-sonnet-5",
    vectorizer=Vectorizer(model="openai::text-embedding-3-small"),
)
clf.fit(X, y)
```

A vectorizer you pass is cloned at fit and kept on `vectorizer_`. It keeps its own connection settings instead of inheriting the classifier's, so the classifier above needs two credentials, one for Anthropic and one for OpenAI. [Credentials and configuration](/docs/credentials-and-configuration) covers how each of them resolves.

{% callout title="Overrides stop at the chat calls" %}
Embedding requests ignore alias-targeted `config_context(overrides=...)`. Overrides carry generation parameters, and the vectorizer a classifier builds for itself shares that classifier's alias, so a temperature meant for the chat calls must not arrive on an embedding request.
{% /callout %}

## What retrieval costs

`fit` issues one embedding pass over the training set, which is `ceil(len(X) / batch_size)` requests with the `Vectorizer` default `batch_size` of 32. Every `predict` or `predict_proba` call issues another pass over the texts it was handed, before any chat request goes out.

Nothing is cached between calls. Calling `predict` and then `predict_proba` on the same texts embeds them twice, and refitting embeds the training set again. `n_predictors` is the one repetition that does not multiply retrieval: it repeats the chat call, so the sampled votes for a sample all share a single prompt.

Embedding tokens normally cost far less than chat tokens, but the requests are real. They are issued by the vectorizer rather than by the classifier, so they are reported on `clf.vectorizer_.usage_` and not on `clf.usage_`.

Retrieval is all-or-nothing. An embedding request that still fails after its retries aborts the call it belongs to, even under the default `on_error="fill"`, because a zero-vector fill is indistinguishable from a real embedding. `on_error` governs the chat calls that follow it.

A fitted classifier also carries its training data: `examples_` holds every pair verbatim and `indexes_` holds their vectors. Both travel into anything saved from it, so a bundle grows with the training set. See [Saving and loading](/docs/saving-and-loading).
