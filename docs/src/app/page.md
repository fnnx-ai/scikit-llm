---
title: Quick start
---

Scikit-LLM exposes language models as scikit-learn estimators. Classification, extraction, scoring, tagging and embeddings all happen behind `fit`, `predict` and `transform`, against OpenAI, Anthropic, Google, Ollama and anything LiteLLM reaches.

## Installation

The core install covers OpenAI and every OpenAI-compatible endpoint, including a local Ollama server. Other providers ship as extras, so a missing package only affects the provider it belongs to.

```bash
pip install scikit-llm
```

| Command | Adds |
| --- | --- |
| `pip install scikit-llm[anthropic]` | Anthropic |
| `pip install scikit-llm[google]` | Google, through the Gemini API |
| `pip install scikit-llm[litellm]` | LiteLLM, which reaches Azure, Bedrock, Vertex and others |
| `pip install scikit-llm[all]` | all of the above |

Python 3.10 or newer is required.

## A first prediction

Every estimator is importable from the top-level `skllm` package. `skllm.datasets` carries a few small in-memory datasets for examples like this one.

```python
from skllm import ZeroShotClassifier
from skllm.datasets import get_classification_dataset

X, y = get_classification_dataset()  # 30 movie reviews; positive, negative, neutral

clf = ZeroShotClassifier()  # openai::gpt-5.6-sol, key read from OPENAI_API_KEY
clf.fit(None, y)
labels = clf.predict(X)
```

A zero-shot classifier never looks at the training texts, so `fit` ignores `X` and accepts `None` for it. All it takes from `y` is the label set: `clf.fit(None, ["positive", "negative", "neutral"])` produces the same estimator. The sorted unique labels end up in `clf.classes_`, and `predict` returns a numpy array holding one label per input text. Structured outputs constrain the model to those labels, so an answer outside the set becomes a reported failure rather than a silent prediction.

## Providing a credential

The example above reads the key from the environment.

```bash
export OPENAI_API_KEY="sk-..."
```

Connection settings resolve when a request is issued, not when the estimator is built. Nothing is cached on the estimator, and `api_key` and `base_url` are stripped from any pickle or saved bundle.

{% callout title="Two other ways to pass a key" %}
`ZeroShotClassifier(api_key="sk-...")` sets it per estimator, and `skllm.config_context(openai_api_key="sk-...")` scopes it to a block of code. The estimator parameter wins over the context, and the context wins over the environment. See [Credentials and configuration](/docs/credentials-and-configuration) for aliases, custom endpoints, timeouts and pricing.
{% /callout %}

## Choosing a model

Models are named `provider::model_id`. The prefix selects the provider, and the rest is that provider's own model id, passed through untouched. Built-in prefixes are `openai`, `ollama`, `anthropic`, `google` and `litellm`. A bare string means OpenAI, so `model="gpt-5"` and `model="openai::gpt-5"` are the same thing. Every estimator that issues chat requests takes this string and defaults to `openai::gpt-5.6-sol`. `Vectorizer` issues embedding requests instead, and defaults to `openai::text-embedding-3-small`.

```python
from skllm import ZeroShotClassifier

clf = ZeroShotClassifier(model="anthropic::claude-sonnet-5")
clf.fit(None, y)
clf.predict(X)
```

This one needs `pip install scikit-llm[anthropic]` and `ANTHROPIC_API_KEY`. [Models and providers](/docs/models-and-providers) covers what each provider supports, how to reach a self-hosted endpoint, and how to register a prefix of your own.

## Where to go next

Coming from v1, start at [Migrating from v1](/docs/migrating-from-v1). v2 is a rewrite with no compatibility layer: no v1 estimator survives under its old name, and `SKLLMConfig` is gone. That page maps every class, parameter and configuration call to its replacement, and says how to pin the old release if the move has to wait.

Classification continues with [few-shot](/docs/few-shot-classification) and [dynamic few-shot](/docs/dynamic-few-shot-classification) prompting, and with [consensus](/docs/consensus-classification) voting across several models. Beyond classification there are [structured extraction](/docs/structured-extraction), [named entity recognition](/docs/named-entity-recognition), [summarization](/docs/summarization), [translation](/docs/translation), [rubric scoring](/docs/rubric-scoring), [cluster labeling](/docs/cluster-labeling) and [vectorization](/docs/vectorization).

Machinery shared by all of them has its own pages. [Batch execution and error handling](/docs/batch-execution-and-error-handling) explains concurrency, what a failed sample becomes, and the `errors_` and `usage_` reports. [Customizing prompts](/docs/customizing-prompts) covers replacing a task's default template. When a fitted estimator has to leave the notebook, [Saving and loading](/docs/saving-and-loading) and [Serving bundles](/docs/serving-bundles) describe the `.fnnx` bundle format and how it runs in production.
