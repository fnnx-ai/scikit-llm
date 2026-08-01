<div align="center">
  <img alt="logo" src="https://gist.githubusercontent.com/OKUA1/55e2fb9dd55673ec05281e0247de6202/raw/41063fcd620d9091662fc6473f9331a7651b4465/scikit-llm.svg" height = "250">
</div>

# Scikit-LLM: Scikit-Learn Meets Large Language Models

Language models as scikit-learn estimators: classification, extraction, scoring,
tagging and embeddings behind `fit`/`predict`/`transform`, across OpenAI,
Anthropic, Google, Ollama and anything LiteLLM reaches.

> **v2 is a rewrite, and no v1 estimator survives it under its old name.** One
> estimator per task, the provider chosen by a model string, credentials that
> never touch a pickle, and `.fnnx` bundles as the deployment artifact. Coming
> from v1? Read **[MIGRATION.md](MIGRATION.md)**, or pin `scikit-llm<2`.

## Installation 💾

```bash
pip install scikit-llm                 # OpenAI, Ollama, any OpenAI-compatible endpoint
pip install scikit-llm[anthropic]      # + Anthropic
pip install scikit-llm[google]         # + Google (Gemini API)
pip install scikit-llm[litellm]        # + LiteLLM catch-all (Azure, Bedrock, Vertex, …)
pip install scikit-llm[all]            # everything
```

Python ≥ 3.10.

## Quick start 🚀

```python
from skllm import ZeroShotClassifier
from skllm.datasets import get_classification_dataset

X, y = get_classification_dataset()  # movie reviews; labels: positive, negative, neutral

clf = ZeroShotClassifier()  # openai::gpt-5.6-sol, key from OPENAI_API_KEY
clf.fit(None, y)            # zero-shot needs the label set, not the texts
labels = clf.predict(X)
```

Every estimator is importable from the top-level package, every one that talks to
a model takes a `provider::model_id` string, and each runs its batch
concurrently:

```python
from skllm import FewShotClassifier

clf = FewShotClassifier(model="anthropic::claude-sonnet-5", max_concurrency=16)
clf.fit(X, y).predict(["A film I will not be recommending to anyone."])
```

## Providers 🔌

| Prefix | Install | Chat | Structured outputs | Embeddings | Credentials |
| --- | --- | --- | --- | --- | --- |
| `openai::` | core | ✅ | native JSON Schema | ✅ | `OPENAI_API_KEY` |
| `ollama::` | core | ✅ | native (OpenAI-compatible) | ✅ | none — `OLLAMA_HOST`, default `http://localhost:11434/v1` |
| `anthropic::` | `[anthropic]` | ✅ | forced tool use | ❌ | `ANTHROPIC_API_KEY` |
| `google::` | `[google]` | ✅ | response schema | ✅ | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| `litellm::` | `[litellm]` | ✅ | native where LiteLLM reports support, prompt fallback otherwise | ✅ | LiteLLM's own per-provider variables |

A bare model string means OpenAI: `model="gpt-5"` is `model="openai::gpt-5"`. An
unknown prefix raises at first use and lists what is registered. Any
OpenAI-compatible endpoint is `openai::` plus a `base_url`; Azure, Bedrock and
Vertex go through `litellm::`.

```python
from skllm import Summarizer

Summarizer(model="ollama::llama3")                                  # local, keyless
Summarizer(model="openai::my-model", base_url="http://localhost:8000/v1")
Summarizer(model="litellm::azure/my-deployment")
```

## Credentials 🔑

Connection settings resolve at call time, in this order:

**estimator parameter → active `config_context` → environment variable → default**

Nothing is cached on the estimator, and nothing is serialized: `api_key` and
`base_url` are real sklearn params (so `clone` and grid search work), but
`__getstate__` strips them and `.fnnx` bundles never contain them.

```python
import skllm
from skllm import ZeroShotClassifier

# 1. environment — OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY / …
clf = ZeroShotClassifier()

# 2. scoped, thread-safe and async-safe (contextvars), nestable
with skllm.config_context(openai_api_key="sk-...", timeout=30.0, max_retries=5):
    clf.fit(None, y).predict(X)

# 3. per estimator
clf = ZeroShotClassifier(api_key="sk-...")
```

`config_context` also carries `client_options` (per-provider client arguments
such as an OpenAI organization), `pricing`, and `allow_env_fallback=False` —
which forbids ambient environment keys entirely, so a shared service can never
authenticate with a key it did not mean to use (`SKLLM_ALLOW_ENV_FALLBACK=0`
does the same for a whole process).

### Aliases

An `alias` gives an estimator a durable name — it is serialized, so it survives
saving. Two estimators on the same provider can then hold different keys, and a
single estimator inside a graph can be retuned from the outside:

```python
clf = ZeroShotClassifier(alias="team_a")

with skllm.config_context(
    credentials={"team_a": "sk-a"},                  # resolved strictly by name
    overrides={"team_a": {"temperature": 0.7}},      # beats the stored model_params
):
    clf.fit(None, y).predict(X)
```

An aliased estimator never falls back to the provider-level key: a missing alias
credential fails naming the alias rather than quietly merging keys you split on
purpose. The environment fallback is `SKLLM_CREDENTIAL_TEAM_A`.

## Tasks 🧰

### Classification

```python
from skllm import (
    DynamicFewShotClassifier,
    FewShotClassifier,
    MultiLabelZeroShotClassifier,
    Vectorizer,
    ZeroShotClassifier,
)

ZeroShotClassifier(default_label="neutral").fit(None, y).predict(X)
FewShotClassifier().fit(X, y).predict(X)          # training pairs go into the prompt

# retrieves the n nearest examples per class at predict time
DynamicFewShotClassifier(n_examples=3, vectorizer=Vectorizer()).fit(X, y).predict(X)
```

Labels are constrained by structured outputs, so a prediction is always a member
of `classes_` — anything else is a sample failure, not a silent label. Labels
must be strings, and `classes_` holds their sorted unique set.

Multi-label classifiers return ragged lists of labels, never padded rows.
`max_labels` becomes a schema constraint where the provider supports one, and
extra labels are truncated where it does not:

```python
from skllm.datasets import get_multilabel_classification_dataset

X, y = get_multilabel_classification_dataset()
MultiLabelZeroShotClassifier(max_labels=3).fit(None, y).predict(X)
# [['Quality', 'Price'], ['Delivery'], ...]
```

### Confidence through sampled voting

`n_predictors=k` issues k sampled calls per sample — skllm's temperature-0
default is suppressed so the votes can differ — then majority-votes them:

```python
clf = ZeroShotClassifier(n_predictors=5).fit(None, y)
clf.predict(X)        # the majority label
clf.predict_proba(X)  # vote fractions, columns aligned with clf.classes_
```

Left unset, `predict_proba` still works: it is one-hot on the single
deterministic prediction, at no extra cost.

### Consensus across providers

```python
from skllm import ConsensusClassifier, ZeroShotClassifier

ensemble = ConsensusClassifier(
    [
        ZeroShotClassifier(model="openai::gpt-5", alias="fast"),
        ZeroShotClassifier(model="anthropic::claude-sonnet-5", alias="careful"),
        ZeroShotClassifier(model="google::gemini-2.5-flash"),
    ],
    weights=[1.0, 2.0, 1.0],
)
ensemble.fit(X, y).predict(X)
```

Like sklearn's `VotingClassifier`, `fit` clones the members into `estimators_`
and leaves the ones you passed untouched. Members are queried concurrently.

### Summarization and translation

```python
from skllm import Summarizer, Translator

Summarizer(max_words=20, focus="pricing").transform(X)
Translator(output_language="German").transform(["Good morning"])
```

### Named entity recognition

`NER` returns the input text with entities wrapped in inline tags. Stripping the
tags reproduces the input exactly — an output that cannot be reconciled with the
source is a sample failure, filled with the untagged original.

```python
from skllm import NER
from skllm.tagging import render_tagged_html

ner = NER(entities={"PER": "a person", "ORG": "an organization"})
tagged = ner.transform(["Tim Cook leads Apple."])
# array(['<PER>Tim Cook</PER> leads <ORG>Apple</ORG>.'], dtype=object)

render_tagged_html(tagged, ner.entities)  # highlighted HTML, inline in notebooks
```

### Structured extraction

`schema` is a JSON Schema dict or a pydantic model class; each answer is
validated against it, and a sample that will not conform fills with `None`.

```python
from pydantic import BaseModel

from skllm import StructuredExtractor


class Movie(BaseModel):
    name: str
    year: int


StructuredExtractor(schema=Movie).transform(["Dune: Part Two came out in 2024."])
# [{'name': 'Dune: Part Two', 'year': 2024}]
```

### Rubric scoring

```python
from skllm import RubricScorer

scorer = RubricScorer(
    rubric="How clearly does the review justify its verdict?",
    scale=(0, 5),
)
scorer.predict(X)  # float array, clamped into the scale; NaN where a sample failed
```

### Cluster labeling

Names the clusters any clusterer produced, from a sample of each cluster's texts:

```python
from sklearn.cluster import KMeans

from skllm import ClusterLabeler, Vectorizer

embeddings = Vectorizer().fit_transform(X)
clusters = KMeans(n_clusters=3, random_state=0).fit_predict(embeddings)

labeler = ClusterLabeler(n_exemplars=8, random_state=0)
names = labeler.fit_transform(X, clusters)  # one naming call per cluster
labeler.names_                              # {0: '...', 1: '...', 2: '...'}
labeler.transform([0, 1, -1])               # noise id -1 maps to None
```

### Embeddings in a pipeline

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from skllm import Vectorizer

pipeline = Pipeline(
    [("embed", Vectorizer(batch_size=32)), ("clf", LogisticRegression())]
).fit(X, y)
```

## How predictions run ⚙️

**Concurrency and progress.** Every per-sample call in a batch runs concurrently,
bounded by `max_concurrency` (default 8). `verbose=1` shows a progress bar for
multi-sample batches; `verbose=0` is silent. An empty batch issues no requests.

**Error policy.** `on_error="fill"` (default) gives a failed sample its task's
documented fill value — `default_label` for single-label classifiers, `[]` for
multi-label, `""` for text, `None` for extraction, `NaN` for scores — warns once
per batch, and records what happened. `on_error="raise"` aborts the batch with
the provider's own error instead.

```python
clf.predict(X)     # 30 texts, one of which failed for good
clf.errors_        # (SampleError(index=3, error='...'),)
clf.usage_.last    # UsageRecord(requests=29, input_tokens=..., output_tokens=..., failures=1, cost=None)
clf.usage_.total   # the same, accumulated over every call on this estimator
```

Embeddings and cluster naming are all-or-nothing: they raise whatever the policy
says, because a zero-vector or a half-named mapping would poison everything
downstream.

**Cost.** Supply prices (per million tokens) and the usage report estimates cost:

```python
with skllm.config_context(pricing={"gpt-5.6-sol": {"input": 1.25, "output": 10.0}}):
    clf.predict(X)
clf.usage_.last.cost
```

**Telemetry.** Each request emits one OpenTelemetry span with `gen_ai.*`
attributes (operation, model, token counts, cost). It is inert unless the
application has configured a tracer provider — scikit-llm takes no
opentelemetry dependency and never installs a provider itself.

**Provider parameters.** `model_params` goes to the provider verbatim — it is the
home for temperature, token limits, reasoning effort, thinking budgets, whatever
your provider takes. scikit-llm validates nothing against allow-lists, so an
unsupported parameter surfaces the provider's own error.

```python
ZeroShotClassifier(model="openai::gpt-5", model_params={"reasoning_effort": "high"})
```

scikit-llm injects its own determinism defaults — temperature 0 everywhere, plus
`reasoning_effort: "none"` on OpenAI, which exposes a request param for it. They
are advisory: a model that rejects one has that default stripped and the request
retried without it, remembered per model for the rest of the process. Your
`model_params` are never rewritten or stripped that way.

**Prompts.** Pass `prompt_template` to replace a task's default. Placeholders are
validated at `fit` against the task's documented set (`{text}` and `{labels}` for
zero-shot classification, plus `{examples}` for few-shot, and so on) — a missing
or unknown placeholder raises then, not mid-batch.

## Serialization and serving 📦

`skllm.save` writes a self-describing **FNNX** bundle: the fitted graph, a
manifest with typed IO schemas, the credentials it needs declared as required
secrets, and pinned dependencies — but never a key.

```python
import skllm

clf.save("model.fnnx")                 # any skllm estimator
skllm.save(pipeline, "pipeline.fnnx")  # or any sklearn composition containing one
```

Loading gives an `InferenceModel` — the bundle answering through the same path a
deployment would use, with sklearn types on both sides:

```python
model = skllm.load("model.fnnx")

with skllm.config_context(openai_api_key="sk-..."):
    model.predict(["Absolutely worth the price."])

estimator = model.unwrap()  # the fitted graph itself, for inspection or refitting
```

The default handler runs the bundle in this process. `StdIOHandler` runs it in
its own environment, built from the bundle's pinned dependencies — exact
reproduction with no version matching against the host:

```python
from fnnx.handlers.stdio import StdIOHandler

model = skllm.load("model.fnnx", handler=StdIOHandler)
```

The same bundle runs under the FNNX runtime anywhere, taking credentials and
generation-parameter overrides as per-call dynamic attributes:

```python
from fnnx.runtime import Runtime

runtime = Runtime("model.fnnx")
runtime.compute(
    {"X": ["Absolutely worth the price."]},
    {"openai_api_key": "sk-...", "overrides": {"temperature": 0.2}},
)
```

`overrides` carries generation params straight for a bundle holding a single
estimator, and keyed by estimator alias for a composite one — a pipeline or a
`ConsensusClassifier` — so one member can be tuned without touching the others.

Structural identity — the model string, the prompt, the labels, the schema — is
fixed at save time and cannot be overridden per call. Credentials have no
default, by construction: they come from the dynamic attribute, or from the
serving process's `config_context`/environment, or the call fails saying which
attribute and which variable would have supplied them.

> Loading, unwrapping or running a bundle executes its packaged code and
> unpickles its state — load only bundles you trust.

## Custom backends 🧩

A provider prefix is one small adapter object, registered at runtime:

```python
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import skllm
from skllm.backends import BaseBackend, Message, ResolvedConnection, Response


class MyCorpBackend(BaseBackend):
    provider: ClassVar[str] = "mycorp"
    credential_env_var: ClassVar[str | None] = "MYCORP_API_KEY"
    # what the retry loop treats as worth another attempt; empty by default
    transient_errors: ClassVar[tuple[type[BaseException], ...]] = (TimeoutError,)

    def _chat(
        self,
        model: str,
        messages: Sequence[Message],
        params: dict[str, Any],
        settings: ResolvedConnection,
    ) -> Response[str]:
        ...

    def _structured(
        self,
        model: str,
        messages: Sequence[Message],
        schema: Mapping[str, Any],
        params: dict[str, Any],
        settings: ResolvedConnection,
    ) -> Response[Any]:
        ...


skllm.register_backend("mycorp", MyCorpBackend())
skllm.ZeroShotClassifier(model="mycorp::internal-1")
```

The base class supplies credential resolution, default injection and the retry
loop — which backs off over whatever the adapter declares transient, since only
the adapter knows which of its SDK's errors are worth a second attempt; the
adapter speaks to the provider and reports what it got back — content plus a
`Usage` record, which is what feeds `usage_` and the cost estimates.

## Support us 🤝

You can support the project in the following ways:

- ⭐ Star Scikit-LLM on GitHub (click the star button in the top right corner)
- 💡 Provide your feedback or propose ideas in the [issues](https://github.com/iryna-kondr/scikit-llm/issues) section or [Discord](https://discord.gg/YDAbwuWK7V)
- 📰 Post about Scikit-LLM on LinkedIn or other platforms

## Citation

You can cite Scikit-LLM using the following BibTeX:

```
@software{ScikitLLM,
  author = {Iryna Kondrashchenko and Oleh Kostromin},
  year = {2023},
  publisher = {beastbyte.ai},
  address = {Linz, Austria},
  title = {Scikit-LLM: Scikit-Learn Meets Large Language Models},
  url = {https://github.com/iryna-kondr/scikit-llm }
}
```
