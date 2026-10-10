---
title: Batch execution and error handling
---

A call to `predict` or `transform` becomes a batch of independent requests, usually one per sample. A few estimators issue requests from `fit` as well: `ClusterLabeler` names each cluster there, and `DynamicFewShotClassifier` embeds its training texts. Every estimator runs such a batch the same way, reports what it cost the same way, and decides what a failed sample becomes the same way.

## Concurrency

Requests within one batch are issued concurrently. `max_concurrency` caps how many are in flight at once and defaults to 8. The pool is created for the call and torn down when it returns, so nothing is shared between calls, and results always come back in input order regardless of which request finished first.

```python
from skllm import ZeroShotClassifier
from skllm.datasets import get_classification_dataset

X, y = get_classification_dataset()

clf = ZeroShotClassifier(max_concurrency=16)
clf.fit(None, y)
labels = clf.predict(X)
```

`verbose=1`, the default, shows a progress bar for batches of two or more units of work. `verbose=0` is silent. An empty batch issues no requests at all.

What counts as one unit of concurrency depends on the estimator. For most of them it is one sample. `Vectorizer` sends `batch_size` texts per embedding request, so the cap bounds requests rather than texts. Classifiers with `n_predictors` set ask each sample several times, and all of those attempts share the one pool, so the cap still holds. `ConsensusClassifier` runs each member's whole batch as a unit, and every member applies its own `max_concurrency` within that.

Worker threads run in a copy of the calling context, which is what lets an active `config_context` block reach every request the batch issues. See [Credentials and configuration](/docs/credentials-and-configuration).

## Timeouts and retries

Neither is an estimator parameter. Both are connection settings, scoped with `config_context`, because they describe the transport rather than the task.

```python
import skllm

with skllm.config_context(timeout=30.0, max_retries=5):
    labels = clf.predict(X)
```

`timeout` applies to a single request and defaults to 60 seconds. `max_retries` counts the retries after the first attempt and defaults to 3, so the default budget is four attempts. Only failures the provider's backend declares transient are retried — timeouts, connection errors, rate limits and server errors. Anything else propagates on the first attempt. Backoff starts at half a second and doubles up to a ceiling of 30 seconds, except where the provider sent a `Retry-After` header, which is honored instead. scikit-llm owns this policy and turns the provider SDK's own retries off, so `max_retries=0` means exactly one attempt per request.

Retries are invisible to the batch: a sample fails only once its attempts are exhausted.

## Error policy

`on_error` decides what a batch does with a sample it could not answer. Under the default `"fill"`, the sample takes its task's fill value, the batch finishes, and the failure is reported. Under `"raise"`, the first failed sample aborts the whole batch and the underlying exception propagates, so there is no partial result to inspect.

```python
clf = ZeroShotClassifier(on_error="raise")
```

A failure is not only a transport error. An answer scikit-llm cannot use is also a sample failure: a label outside `classes_`, a score that is not a finite number, an extracted record that does not validate against the schema, an NER rewrite that changed more than whitespace. These surface as `BackendError`.

Fill values are chosen per task so that a filled sample is distinguishable from a real answer wherever that is possible.

| Estimators | A failed sample becomes |
| --- | --- |
| `ZeroShotClassifier`, `FewShotClassifier`, `DynamicFewShotClassifier`, `ConsensusClassifier` | `default_label`, which is `None` unless you name a label |
| `MultiLabelZeroShotClassifier`, `MultiLabelFewShotClassifier` | `[]`, an empty label list |
| `Summarizer`, `Translator` | `""`, the empty string |
| `StructuredExtractor` | `None` in place of the record |
| `RubricScorer` | `NaN` |
| `NER` | that sample's own input text, untagged |
| `Vectorizer`, `ClusterLabeler` | nothing — they raise instead |

`predict_proba` gives a failed sample an all-zero row, which no successful prediction produces.

Two kinds of failure ignore `on_error` entirely. A `MissingCredentialError` aborts the batch immediately, because a configuration error hits every sample identically and filling it would report one missing key as a separate prediction failure per sample. Embeddings and cluster naming abort as well: a zero vector or an unnamed cluster is not something a downstream step could tell from a real answer, so `Vectorizer` and `ClusterLabeler` raise even under `on_error="fill"`.

## The two reports

Every operation replaces both reports, so `errors_` and `usage_.last` always describe the call that just ran.

```python
labels = clf.predict(X)
# UserWarning: ZeroShotClassifier: 1 of 30 samples failed and were filled;
# see the errors_ attribute for the reasons.

clf.errors_
# (SampleError(index=17, error="BackendError: The model answered 'mixed', which is
#  not one of the fitted classes ['negative', 'neutral', 'positive']."),)
```

One warning is emitted per batch, not per sample, and it is attributed to the line that called `predict` rather than to a file inside the library, so a warning filter keyed to your own module still matches it.

`errors_` is a tuple of `SampleError`, each holding the `index` of the sample in the batch and the reason as a string. The reason is text rather than the exception object because reports travel with the estimator through pickling and provider exceptions are not reliably picklable. An aborted batch leaves `errors_` empty: there is no per-sample report to give when the exception itself is the report.

`usage_` is a `UsageReport` with two `UsageRecord` fields. `last` covers the most recent operation, `total` accumulates over every operation since the estimator was constructed.

```python
clf.usage_.last
# UsageRecord(requests=29, input_tokens=8140, output_tokens=87, failures=1, cost=None)
clf.usage_.total
```

A record holds `requests`, `input_tokens`, `output_tokens`, `failures` and `cost`. Only responses that arrived are counted, so a batch of 30 with one failure reports 29 requests and one failure. Records are collected per operation through the same context mechanism the config layers use, which means two threads calling `predict` on one shared estimator do not contaminate each other's `last`.

A batch that aborts still records what it spent before aborting, then re-raises — the requests were paid for whether or not the batch finished. `ClusterLabeler.transform` is the opposite case: the names were decided at `fit`, so it issues no request and reports an empty record without touching `total`. `ConsensusClassifier` issues no request of its own either, and reports the sum of its members' last operations.

## Cost estimates

`cost` is `None` until prices are configured. Supply them per million tokens, keyed either by the full `provider::model` string or by the bare model id.

```python
with skllm.config_context(pricing={"gpt-5.6-sol": {"input": 1.25, "output": 10.0}}):
    clf.predict(X)

clf.usage_.last.cost
```

The estimate is exactly what the arithmetic suggests: reported input tokens times the input price plus reported output tokens times the output price, divided by a million. It knows nothing about cached-input discounts, batch pricing or free tiers. A provider that reports no token counts is priced as zero rather than left unpriced, so a cost of `0.0` under a configured price means the responses carried no usage numbers. `cost` stays `None` only where no price was configured for the model.

{% callout title="No built-in price table" %}
scikit-llm carries no prices of its own; published ones change, and a stale table would quietly produce wrong numbers. The `pricing` mapping is read from the innermost active `config_context`, so one block around a whole pipeline prices everything it runs.
{% /callout %}

## Tracing

Each provider request is wrapped in one OpenTelemetry span, from the tracer named `scikit-llm`. The span is named after the operation and the model, and carries the `gen_ai` semantic attributes: `gen_ai.operation.name` (`chat` for both plain and structured calls, `embeddings` for embedding calls), `gen_ai.provider.name` and `gen_ai.request.model` when it opens, then `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` and, where a price was configured, `gen_ai.usage.cost` before it closes.

This is inert by default. scikit-llm takes no opentelemetry dependency and never installs or mutates a tracer provider. If the package is not installed, or if the global provider is still the no-op or proxy one that means nobody configured tracing, no span is started at all. Spans appear once your application configures a real provider, and go wherever that provider sends them.
