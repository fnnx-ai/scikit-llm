---
title: Migrating from v1
---

v2 is a rewrite, and it ships no compatibility layer. Every estimator, the configuration object and the whole provider layer moved. Code that cannot move yet should pin the last v1 release.

```bash
pip install "scikit-llm<2"
```

Three changes account for most of a port.

**One estimator per task.** The provider is no longer part of the class name. It is a prefix on the model string, so `ZeroShotGPTClassifier`, `ZeroShotClaudeClassifier` and `ZeroShotVertexClassifier` are all `ZeroShotClassifier(model="<provider>::<model_id>")` now. Every estimator is importable straight from `skllm`.

**Credentials resolve at call time.** `SKLLMConfig`, which wrote to `os.environ`, is gone. A key comes from the estimator parameter, then an active `config_context`, then the provider's own standard environment variable. Keys are never serialized.

**Failures are visible.** An answer v1 could not turn into a label became an empty string, and then a random label indistinguishable from a real prediction. In v2 a failed sample takes the task's documented fill value, plus a warning and an entry in `errors_`. Under `on_error="raise"` it aborts the batch instead.

The tables below map the names most code touches. The repository's [MIGRATION.md](https://github.com/iryna-kondr/scikit-llm/blob/main/MIGRATION.md) lists every removed symbol.

{% callout title="What did not move" %}
`skllm.datasets` is untouched. `get_classification_dataset`, `get_multilabel_classification_dataset`, `get_summarization_dataset` and `get_translation_dataset` keep their names, their import path and their contents.
{% /callout %}

## Installation

| v1 | v2 | Notes |
| --- | --- | --- |
| `pip install scikit-llm` | `pip install scikit-llm` | OpenAI, and any OpenAI-compatible endpoint including Ollama, works out of the box |
| — | `pip install scikit-llm[anthropic]` | Anthropic backend |
| — | `pip install scikit-llm[google]` | Google Gemini API backend (`google-genai`) |
| — | `pip install scikit-llm[litellm]` | LiteLLM catch-all backend |
| — | `pip install scikit-llm[all]` | all of the above |
| `pip install scikit-llm[gguf]` | removed | run local models through `ollama::` instead |
| `pip install scikit-llm[annoy]` | removed | retrieval uses scikit-learn's `NearestNeighbors` |

Python 3.10 or newer is required, where v1 supported 3.9. `pandas` and `google-cloud-aiplatform` are no longer dependencies. pandas input is still accepted, it is just no longer installed for you.

## Classification

| v1 | v2 |
| --- | --- |
| `skllm.models.gpt.classification.zero_shot.ZeroShotGPTClassifier` | `skllm.ZeroShotClassifier` |
| `skllm.models.anthropic.classification.zero_shot.ZeroShotClaudeClassifier` | `skllm.ZeroShotClassifier` with `model="anthropic::…"` |
| `skllm.models.vertex.classification.zero_shot.ZeroShotVertexClassifier` | `skllm.ZeroShotClassifier` with `model="google::…"` (Gemini API) or `model="litellm::vertex_ai/…"` |
| `skllm.models.gpt.classification.few_shot.FewShotGPTClassifier` | `skllm.FewShotClassifier` |
| `skllm.models.anthropic.classification.few_shot.FewShotClaudeClassifier` | `skllm.FewShotClassifier` with `model="anthropic::…"` |
| `skllm.models.gpt.classification.few_shot.DynamicFewShotGPTClassifier` | `skllm.DynamicFewShotClassifier` |
| `skllm.models.anthropic.classification.few_shot.DynamicFewShotClaudeClassifier` | `skllm.DynamicFewShotClassifier` with an explicit `vectorizer` |
| `skllm.models.gpt.classification.zero_shot.MultiLabelZeroShotGPTClassifier` | `skllm.MultiLabelZeroShotClassifier` |
| `skllm.models.anthropic.classification.zero_shot.MultiLabelZeroShotClaudeClassifier` | `skllm.MultiLabelZeroShotClassifier` with `model="anthropic::…"` |
| `skllm.models.vertex.classification.zero_shot.MultiLabelZeroShotVertexClassifier` | `skllm.MultiLabelZeroShotClassifier` with `model="google::…"` |
| `skllm.models.gpt.classification.few_shot.MultiLabelFewShotGPTClassifier` | `skllm.MultiLabelFewShotClassifier` |
| `skllm.models.anthropic.classification.few_shot.MultiLabelFewShotClaudeClassifier` | `skllm.MultiLabelFewShotClassifier` with `model="anthropic::…"` |
| `skllm.classification.TunableGPTClassifier`, `skllm.models.gpt.classification.tunable.GPTClassifier`, `MultiLabelGPTClassifier` | removed, see [Fine-tuning](#fine-tuning) |
| `skllm.classification.TunableVertexClassifier`, `skllm.models.vertex.classification.tunable.VertexClassifier` | removed, see [Fine-tuning](#fine-tuning) |
| `CoTGPTClassifier`, `CoTClaudeClassifier` | removed, see the note below |
| — | `skllm.ConsensusClassifier`, new: a weighted vote across several classifiers, providers and models |

The task pages cover the v2 classifiers in full: [zero-shot](/docs/zero-shot-classification), [few-shot](/docs/few-shot-classification), [dynamic few-shot](/docs/dynamic-few-shot-classification) and [consensus](/docs/consensus-classification).

Constructor parameters moved along with the class names. Every v2 parameter except the leading `model` (`estimators` for `ConsensusClassifier`) is keyword-only.

| v1 | v2 |
| --- | --- |
| `key=`, `org=` | `api_key=`. The organization id is client-level connection config: `config_context(client_options={"openai": {"organization": …}})`, or the SDK's own `OPENAI_ORG_ID` |
| `default_label="Random"` | `default_label=None`. A failed sample is `None` unless you name a label, never a random one |
| `max_labels=5` | `max_labels=None`, meaning up to every fitted class. It becomes a schema constraint where the provider supports one, with truncation as the backstop |
| `n_examples`, `vectorizer` | same names on `DynamicFewShotClassifier`. `vectorizer` takes a scikit-llm `Vectorizer`, and is required for providers with no default embedding model |
| `memory_index=`, `metric=` | removed. Retrieval is scikit-learn `NearestNeighbors`, one index per class |
| `**kwargs` swallowed by the mixin stack | `model_params=`, a dict passed to the provider verbatim |
| `predict(X, num_workers=4)` | `predict(X)`. Concurrency is the `max_concurrency` constructor parameter |
| — | `n_predictors=`, new: k sampled votes per sample, aggregated by `predict` and turned into fractions by `predict_proba` |
| — | `predict_proba`, new, on every classifier |

Porting a zero-shot classifier therefore changes the import, the credential call and the concurrency argument, and names the provider on the model string:

```python
# v1
from skllm.config import SKLLMConfig
from skllm.models.gpt.classification.zero_shot import ZeroShotGPTClassifier

SKLLMConfig.set_openai_key("sk-...")

clf = ZeroShotGPTClassifier(model="gpt-4")
clf.fit(None, y)
labels = clf.predict(X, num_workers=4)
```

```python
import skllm
from skllm import ZeroShotClassifier

clf = ZeroShotClassifier(model="openai::gpt-4", max_concurrency=4)

with skllm.config_context(openai_api_key="sk-..."):
    labels = clf.fit(None, y).predict(X)
```

{% callout title="Chain-of-thought classifiers" %}
`CoTGPTClassifier` and `CoTClaudeClassifier` have no v2 counterpart. Reasoning models subsume the technique, and you can ask for it with `model_params={"reasoning_effort": "high"}` on a model that supports it. v2 classifiers return labels only, not the reasoning column v1 returned alongside them.
{% /callout %}

Three behaviors changed for classifiers.

**Labels must be strings.** `fit` raises `TypeError` on integer labels instead of letting them round-trip through a prompt. Cast them yourself, or keep a `LabelEncoder` outside the estimator.

**Multi-label output is ragged.** v1 padded every row to `max_labels` with `""`. v2 returns a list of label lists with no padding, and an empty list for a sample that failed. Wrap it where metrics need indicator columns:

```python
from sklearn.preprocessing import MultiLabelBinarizer
from skllm import MultiLabelZeroShotClassifier

clf = MultiLabelZeroShotClassifier().fit(None, y)
predictions = clf.predict(X)  # [["Quality", "Price"], ["Delivery"], []]
indicators = MultiLabelBinarizer(classes=clf.classes_).fit_transform(predictions)
```

**A label outside `classes_` is a failure, not a prediction.** v2 constrains the answer to the fitted classes through the provider's structured-output mode, so an unknown label is uncommon. When one arrives anyway, the sample fills with `default_label` and is reported in `errors_`, exactly as a transport failure would be. [Batch execution and error handling](/docs/batch-execution-and-error-handling) covers the policy.

## Text-to-text

| v1 | v2 |
| --- | --- |
| `skllm.models.gpt.text2text.summarization.GPTSummarizer` | `skllm.Summarizer` |
| `skllm.models.anthropic.text2text.summarization.ClaudeSummarizer` | `skllm.Summarizer` with `model="anthropic::…"` |
| `skllm.models.gpt.text2text.translation.GPTTranslator` | `skllm.Translator` |
| `skllm.models.anthropic.text2text.translation.ClaudeTranslator` | `skllm.Translator` with `model="anthropic::…"` |
| `skllm.text2text.TunableGPTText2Text`, `skllm.models.vertex.text2text.tunable.TunableVertexText2Text` | removed, see [Fine-tuning](#fine-tuning) |

`max_words`, `focus` and `output_language` keep their names and their defaults. Both estimators are transformers only: the v1 `predict` alias is gone, so call `transform`. Where v1 raised once its retries were exhausted, v2 fills the failed sample with `""` and reports it, unless you ask for the old behavior with `on_error="raise"`. See [Summarization](/docs/summarization) and [Translation](/docs/translation).

## Named entity recognition

| v1 | v2 |
| --- | --- |
| `skllm.models.gpt.tagging.ner.GPTExplainableNER` | `skllm.NER` |
| `skllm.models.anthropic.tagging.ner.AnthropicExplainableNER` | `skllm.NER` with `model="anthropic::…"` |
| `skllm.utils.rendering.render_ner`, `display_ner` | `skllm.tagging.render_tagged_html` |
| `skllm.utils.xml.filter_xml_tags`, `filter_unwanted_entities`, `replace_all_at_once`, `json_to_xml` | removed, internal to v1's JSON-to-XML machinery |

The `entities={label: description}` mapping survives by name, keyword-only like every other v2 parameter. Everything else about the task moved. `transform` returns the input text with entity mentions wrapped in inline `<LABEL>…</LABEL>` tags, so there is no `<entity><reasoning>…` envelope, no `sparse_output` switch and no `display_predictions` side effect. `num_workers` is now the `max_concurrency` constructor parameter. Stripping the tags off an output reproduces the input exactly, and an output that cannot be reconciled with the source is a sample failure that fills with the untagged original. See [Named entity recognition](/docs/named-entity-recognition).

## Vectorization and memory indexes

| v1 | v2 |
| --- | --- |
| `skllm.models.gpt.vectorization.GPTVectorizer` | `skllm.Vectorizer` |
| `skllm.models._base.vectorizer.BaseVectorizer` | removed. Subclassing is not the extension point, register a backend instead |
| `skllm.memory.SklearnMemoryIndex` | removed. `DynamicFewShotClassifier` builds its own per-class `NearestNeighbors` index |
| `skllm.memory._annoy.AnnoyMemoryIndex` | removed with the `annoy` extra |
| `skllm.memory.base.IndexConstructor` | removed with the `memory_index` parameter |

`batch_size` defaults to 32, where v1 defaulted to 1, and `key`/`org` become `api_key`. Embeddings stay all-or-nothing: a failed batch raises even under `on_error="fill"`, because a zero-vector fill would quietly poison everything downstream. See [Vectorization](/docs/vectorization).

Retrieval for dynamic few-shot is no longer pluggable. There is one `sklearn.neighbors.NearestNeighbors` index per class, built at `fit` from the embeddings the estimator's `vectorizer` produced, and cosine distance decides which examples a prompt demonstrates.

## Credentials and configuration

`skllm.config.SKLLMConfig` is gone in full. It mutated process-global environment variables, which is neither thread-safe nor safe to pickle. v2 reads the environment but never writes it, and everything programmatic flows through `skllm.config_context`. That scope is backed by contextvars, so it is per thread, per async task, and nestable.

```python
# v1
from skllm.config import SKLLMConfig

SKLLMConfig.set_openai_key("sk-...")
SKLLMConfig.set_openai_org("org-...")
clf.predict(X)
```

```python
import skllm

with skllm.config_context(
    openai_api_key="sk-...",
    client_options={"openai": {"organization": "org-..."}},
):
    clf.predict(X)
```

| v1 | v2 |
| --- | --- |
| `SKLLMConfig.set_openai_key(k)`, `set_gpt_key(k)` | `skllm.config_context(openai_api_key=k)`, `ZeroShotClassifier(api_key=k)`, or `OPENAI_API_KEY` |
| `SKLLMConfig.get_openai_key()` | — resolution is internal, and there is no getter that could leak a key |
| `SKLLMConfig.set_openai_org(org)`, `set_gpt_org(org)` | `skllm.config_context(client_options={"openai": {"organization": org}})`, or the SDK's own `OPENAI_ORG_ID` |
| `SKLLMConfig.set_anthropic_key(k)` | `skllm.config_context(anthropic_api_key=k)` or `ANTHROPIC_API_KEY` |
| `SKLLMConfig.set_gpt_url(url)`, `reset_gpt_url()` | `ZeroShotClassifier(base_url=…)` or `skllm.config_context(openai_base_url=…)`, scoped, so nothing needs resetting |
| `SKLLMConfig.set_azure_api_base(…)`, `set_azure_api_version(…)` | removed. Reach Azure through `model="litellm::azure/<deployment>"` and LiteLLM's own `AZURE_*` variables |
| `SKLLMConfig.set_google_project(p)`, `get_google_project()` | removed. The Google backend is the Gemini API, not Vertex |
| `SKLLMConfig.set_gguf_max_gpu_layers(…)`, `set_gguf_verbose(…)`, `get_gguf_download_path()` | removed with the GGUF backend |

The environment variables changed with them.

| v1 | v2 |
| --- | --- |
| `SKLLM_CONFIG_OPENAI_KEY` | `OPENAI_API_KEY` |
| `SKLLM_CONFIG_OPENAI_ORG` | `OPENAI_ORG_ID`, read by the OpenAI SDK itself |
| `SKLLM_CONFIG_ANTHROPIC_KEY` | `ANTHROPIC_API_KEY` |
| `SKLLM_CONFIG_GPT_URL` | `OPENAI_BASE_URL`, or the `base_url` parameter |
| `GOOGLE_CLOUD_PROJECT` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| `SKLLM_CONFIG_AZURE_API_BASE`, `SKLLM_CONFIG_AZURE_API_VERSION`, `SKLLM_CONFIG_GGUF_*` | removed |

`config_context` also carries what v1 had no home for: `credentials={alias: key}` for per-estimator key routing, `overrides={alias: {...}}` for targeted generation-parameter changes, `pricing=` for cost estimates, `timeout=`, `max_retries=`, and `allow_env_fallback=False` to forbid ambient keys entirely. [Credentials and configuration](/docs/credentials-and-configuration) covers the resolution order and aliases.

## Model strings and backends

v1 routed a handful of prefixes through `skllm.llm.gpt.utils.split_to_api_and_model`: `openai`, `azure`, `gguf` and `custom_url`. v2 keeps the `provider::model_id` shape, and a bare string still means OpenAI, but the prefix now selects a backend and the set is different.

| v1 model string | v2 |
| --- | --- |
| `"gpt-4"` (bare) | `"gpt-4"`, still OpenAI |
| `"azure::my-deployment"` | `"litellm::azure/my-deployment"` |
| `"custom_url::my-model"` | `"openai::my-model"` plus `base_url="http://…/v1"` |
| `"gguf::llama3-8b-q4"` | `"ollama::llama3"`. Local inference is served by Ollama, never in-process |
| `"gpt4all::…"` | removed in v1.4 already, use `"ollama::…"` |
| Vertex `text-bison@002` classes (PaLM 2, shut down) | `"google::gemini-2.5-flash"` (Gemini API) or `"litellm::vertex_ai/gemini-2.5-pro"` |
| — | `"anthropic::claude-sonnet-5"`, and any prefix you add with `skllm.register_backend` |

`skllm.llm` is deleted wholesale. The `GPTMixin`, `ClaudeMixin` and `VertexMixin` families, `get_chat_completion`, `get_embedding`, `set_credentials`, `LlamaHandler` and `SUPPORTED_APIS` have no successors under those names. `skllm.backends` replaces them with one backend object per provider behind a protocol of `chat`, `structured` and `embed`, discovered by prefix. Adding a provider is one backend plus `skllm.register_backend("mycorp", backend)`, with no mixin stack and no new estimator classes. [Models and providers](/docs/models-and-providers) covers the registry and what each provider supports.

## Fine-tuning

`GPTClassifier`, `MultiLabelGPTClassifier`, `VertexClassifier`, `TunableGPTText2Text`, `TunableVertexText2Text`, their base classes in `skllm.models._base`, and the `skllm.llm.gpt.clients.openai.tuning` and `skllm.llm.vertex.tuning` machinery behind them are removed with no replacement.

They targeted `gpt-3.5-turbo-0613` and `text-bison`, both retired, and their `fit` mutated constructor parameters, which is the scikit-learn contract violation v2 exists to end. Tune with the provider's own tooling and point a v2 estimator at the resulting model id, for example `ZeroShotClassifier(model="openai::ft:…")`.

## Base classes, prompts and utilities

| v1 | v2 |
| --- | --- |
| `skllm.models._base.classifier.BaseClassifier`, `BaseZeroShotClassifier`, `BaseCoTClassifier`, `BaseFewShotClassifier`, `BaseDynamicFewShotClassifier`, `BaseTunableClassifier`, `SingleLabelMixin`, `MultiLabelMixin` | removed. v2 composes providers instead of inheriting them, and the shared machinery in `skllm.core` is not a public extension point |
| `skllm.models._base.text2text.*`, `skllm.models._base.tagger.*`, `skllm.models._base.vectorizer.*` | removed, same reason |
| `skllm.prompts.builders.build_*_prompt*` | removed. Pass `prompt_template="…"` to any estimator instead |
| `skllm.prompts.templates.*` | `skllm.prompts`: `ZERO_SHOT_SINGLE_LABEL`, `FEW_SHOT_SINGLE_LABEL`, `ZERO_SHOT_MULTI_LABEL`, `FEW_SHOT_MULTI_LABEL`, `SUMMARY`, `TRANSLATION`, `NER_TAGGING`, `EXTRACTION`, `RUBRIC_SCORING`, `CLUSTER_NAMING`. Different placeholders, validated at `fit` |
| `skllm.model_constants.OPENAI_GPT_MODEL`, `OPENAI_EMBEDDING_MODEL`, `ANTHROPIC_CLAUDE_MODEL`, `OPENAI_GPT_TUNABLE_MODEL`, `VERTEX_DEFAULT_MODEL` | removed. The defaults are `skllm.core.DEFAULT_CHAT_MODEL` (`openai::gpt-5.6-sol`) and `skllm.core.DEFAULT_EMBEDDING_MODEL` (`openai::text-embedding-3-small`) |
| `skllm.utils.to_numpy`, `find_json_in_string`, `re_naive_json_extractor`, `extract_json_key`, `retry` | removed. Input coercion, JSON salvage and retry are internal to `skllm.core` and the backends |
| `skllm.utils.rendering.get_random_color`, `is_running_in_jupyter`, `color_palettes` | removed. `skllm.tagging.render_tagged_html` returns HTML that notebooks render on their own |

A custom prompt is a constructor parameter now rather than a builder function, and it is validated against the estimator's placeholders at `fit`. See [Customizing prompts](/docs/customizing-prompts).

## What has no v1 counterpart

[Structured extraction](/docs/structured-extraction) drives extraction from a JSON Schema or a pydantic model. [Rubric scoring](/docs/rubric-scoring) scores text against a written rubric, as a regressor. [Cluster labeling](/docs/cluster-labeling) names the clusters any scikit-learn clusterer produced. [Consensus classification](/docs/consensus-classification) votes across providers and models.

A fitted estimator can also be saved as an FNNX bundle and served without the training code, with credentials excluded by construction. v1 had no serialization story at all: pickling an estimator embedded your key and, for few-shot models, your training corpus. See [Saving and loading](/docs/saving-and-loading) and [Serving bundles](/docs/serving-bundles).

Every estimator reports `usage_` and `errors_` after a call, and emits OpenTelemetry `gen_ai` spans when the application has configured a tracer.
