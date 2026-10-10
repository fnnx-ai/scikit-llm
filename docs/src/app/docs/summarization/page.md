---
title: Summarization
---

`Summarizer` rewrites each input text as a short summary. It sends one chat request per text and returns the answers in input order.

```python
from skllm import Summarizer
from skllm.datasets import get_summarization_dataset

X = get_summarization_dataset()

summarizer = Summarizer(max_words=20)
summaries = summarizer.transform(X)
```

`Summarizer` is a transformer: it exposes `transform` and `fit_transform`, and has no `predict`. It is also stateless: `fit` validates the constructor parameters, issues no request and returns `self`. `transform` therefore works on a freshly constructed instance.

{% callout title="Changed in v2" %}
v1's `GPTSummarizer` aliased `predict` to `transform`. v2 has no such alias. See [Migrating from v1](/docs/migrating-from-v1).
{% /callout %}

## The word budget

`max_words` defaults to `15` and reaches the model as an instruction: *Summarize the text sample in at most `{max_words}` words.* Nothing truncates the answer afterwards, so this is a bound the model is asked to respect rather than one the estimator enforces. Summaries can come back longer, and a downstream step that depends on the limit has to check it. A hard cap on generated tokens, where the provider offers one, is a separate matter handled through [`model_params`](/docs/models-and-providers).

## Steering the summary

`focus` names the concept the summary should concentrate on.

```python
Summarizer(max_words=20, focus="pricing").transform(X)
```

The prompt then asks the model to concentrate on what the text says about that concept. Where the text says nothing about it, the prompt asks for a summary of the text in general, without noting the omission. Samples that do not cover the focus therefore come back as ordinary summaries, with nothing in the output marking them as such.

## Output and failed samples

`transform` returns a numpy array of `dtype=object` and shape `(n_samples,)`. Each entry is the model's answer with surrounding whitespace stripped. An empty input returns shape `(0,)` and issues no request.

Under the default `on_error="fill"` policy, a sample whose request failed fills with the empty string `""`, the batch warns once, and the reasons land in `errors_`. `on_error="raise"` aborts the batch with the provider's own error instead. An empty string is also what an empty answer produces, so `errors_` rather than the output is what distinguishes a filled sample from a summarized one.

```python
summaries = summarizer.transform(X)
summarizer.errors_   # (SampleError(index=3, error='...'),)
```

See [Batch execution and error handling](/docs/batch-execution-and-error-handling) for the policies, the warning and the usage counters.

## Summaries as a preprocessing step

Because the summarizer takes texts and returns texts, it chains with any other transformer over texts.

```python
from sklearn.pipeline import Pipeline

from skllm import Summarizer, Translator

pipeline = Pipeline(
    [
        ("summarize", Summarizer(max_words=20)),
        ("translate", Translator(output_language="German")),
    ]
)
german_summaries = pipeline.fit_transform(X)
```

Every step issues its own requests, so a two-step pipeline over `n` texts costs `2n` of them.

## Custom prompts

The default template is `skllm.prompts.SUMMARY`, and an override must use exactly the placeholders `{text}`, `{max_words}` and `{focus}`. `{focus}` receives a rendered instruction rather than the raw parameter: the full clause when a focus is set, and an empty string when it is not. See [Customizing prompts](/docs/customizing-prompts).
