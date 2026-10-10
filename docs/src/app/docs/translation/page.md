---
title: Translation
---

`Translator` translates each input text into one target language. It sends one chat request per text and returns the answers in input order.

```python
from skllm import Translator
from skllm.datasets import get_translation_dataset

X = get_translation_dataset()   # six sentences, six different source languages

translator = Translator(output_language="English")
translations = translator.transform(X)
```

`output_language` is the name of the target language as free text and defaults to `"English"`. It must be a non-blank string. Anything else is rejected before a request goes out. The source language is never declared, so one batch can mix languages — the model infers each sample's own. The prompt also asks for a sample already written in the target language to be repeated unchanged rather than paraphrased.

Like `Summarizer`, `Translator` is a transformer with no `predict`, and it is stateless: `fit` validates the constructor parameters, issues no request and returns `self`, so `transform` works on a freshly constructed instance. Because it takes texts and returns texts, it composes with any other transformer over texts, in either position. [Summarization](/docs/summarization) shows a summarize-then-translate pipeline.

## What the estimator is responsible for

The estimator formats the request and returns what comes back. A free-text answer has nothing to be validated against, so nothing checks that the answer is in the target language, that it carries the meaning of the source, or that anything was translated at all. Translation quality is the model's alone. The parameters here shape the request. The language pairs a model handles well, and how well it handles them, are properties of that model. Measuring them on your own data is the only way to know.

{% callout type="warning" title="Untrusted input" %}
Each sample is quoted inside the prompt and the answer is unconstrained free text, so a text that reads as instructions can influence what comes back. Treat a translation of untrusted input as model output, not as a faithful transformation of the source.
{% /callout %}

## Output and failed samples

`transform` returns a numpy array of `dtype=object` and shape `(n_samples,)`. Each entry is the model's answer with surrounding whitespace stripped. An empty input returns shape `(0,)` and issues no request.

Under the default `on_error="fill"` policy, a sample whose request failed fills with the empty string `""`, the batch warns once, and the reasons land in `errors_`. `on_error="raise"` aborts the batch with the provider's own error instead. Since an empty string is also what an empty answer produces, `errors_` rather than the output is what distinguishes a filled sample from a translated one. See [Batch execution and error handling](/docs/batch-execution-and-error-handling).

## Custom prompts

The default template is `skllm.prompts.TRANSLATION`, and an override must use exactly the placeholders `{text}` and `{output_language}`. See [Customizing prompts](/docs/customizing-prompts).
