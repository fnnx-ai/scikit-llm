# Migrating from scikit-llm v1 to v2

v2 is a rewrite, and it ships no compatibility layer: every estimator, the
configuration object and the whole provider layer moved, and only
`skllm.datasets` is untouched. Code that cannot move yet should pin the old
release.

```bash
pip install "scikit-llm<2"
```

## What changed, in three lines

1. **One estimator per task.** The provider is no longer part of the class name —
   it is a prefix on the model string. `ZeroShotGPTClassifier`,
   `ZeroShotClaudeClassifier` and `ZeroShotVertexClassifier` are all
   `ZeroShotClassifier(model="<provider>::<model_id>")` now, and every estimator
   is importable straight from `skllm`.
2. **Credentials resolve at call time.** `SKLLMConfig` — which wrote to
   `os.environ` — is gone. A key comes from the estimator param, then an active
   `config_context`, then the provider's own standard environment variable.
   Keys are never serialized.
3. **Failures are visible.** An answer v1 could not turn into a label became an
   empty string, and then a random label indistinguishable from a real
   prediction. In v2 any failed sample — unusable answer or exhausted request
   alike — takes the task's documented fill value, plus a warning and an entry
   in `errors_`, or aborts the batch under `on_error="raise"`.

## Installation

| v1 | v2 | Notes |
| --- | --- | --- |
| `pip install scikit-llm` | `pip install scikit-llm` | OpenAI (and any OpenAI-compatible endpoint, including Ollama) works out of the box |
| — | `pip install scikit-llm[anthropic]` | Anthropic backend |
| — | `pip install scikit-llm[google]` | Google Gemini API backend (`google-genai`) |
| — | `pip install scikit-llm[litellm]` | LiteLLM catch-all backend |
| — | `pip install scikit-llm[all]` | all of the above |
| `pip install scikit-llm[gguf]` | removed | run local models through `ollama::` instead |
| `pip install scikit-llm[annoy]` | removed | retrieval uses scikit-learn's `NearestNeighbors` |

Python ≥ 3.10 is required (v1 supported 3.9). `pandas` and
`google-cloud-aiplatform` are no longer dependencies; pandas input is still
accepted, it is simply not installed for you.

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
| `skllm.classification.TunableGPTClassifier`, `skllm.models.gpt.classification.tunable.GPTClassifier`, `MultiLabelGPTClassifier` | **removed** — see [Fine-tuning](#fine-tuning-removed) |
| `skllm.classification.TunableVertexClassifier`, `skllm.models.vertex.classification.tunable.VertexClassifier` | **removed** — see [Fine-tuning](#fine-tuning-removed) |
| `CoTGPTClassifier`, `CoTClaudeClassifier` | **removed** — reasoning models subsume chain-of-thought prompting; ask for it with `model_params={"reasoning_effort": "high"}` on a reasoning model |
| — | `skllm.ConsensusClassifier` (new): majority vote across several classifiers, providers and models |

Parameter changes on the classifiers — and note that every v2 parameter except
the leading `model` (`estimators` for `ConsensusClassifier`) is keyword-only:

| v1 | v2 |
| --- | --- |
| `key=`, `org=` | `api_key=`; the organization id is client-level connection config — `config_context(client_options={"openai": {"organization": …}})` or the SDK's own `OPENAI_ORG_ID` |
| `default_label="Random"` | `default_label=None`; a failed sample is `None` unless you name a label, never a random one |
| `max_labels=5` | `max_labels=None` (unbounded); a schema constraint where the provider supports one, with truncation as the backstop |
| `n_examples`, `vectorizer` | same names on `DynamicFewShotClassifier`; `vectorizer` takes a skllm `Vectorizer`, and is required for providers with no default embedding model |
| `memory_index=`, `metric=` | **removed** — retrieval is scikit-learn `NearestNeighbors`, one index per class |
| `**kwargs` swallowed by the mixin stack | `model_params=` — a dict passed to the provider verbatim |
| `predict(X, num_workers=4)` | `predict(X)`; concurrency is the `max_concurrency` constructor param |
| — | `n_predictors=` (new): k sampled votes per sample, aggregated by `predict` and turned into fractions by `predict_proba` |
| — | `predict_proba` (new, on every classifier) |

Behaviour worth re-reading before you port:

- **Labels must be strings.** `fit` raises on integer labels instead of letting
  them round-trip through a prompt. Cast them yourself, or keep a
  `LabelEncoder` outside the estimator.
- **Multi-label output is ragged.** v1 padded each row to `max_labels` with
  `""`; v2 returns a list of label lists with no padding. Wrap with
  `sklearn.preprocessing.MultiLabelBinarizer` where metrics need indicators.
- **A label outside `classes_` is a failure**, not a prediction — it fills and
  is reported in `errors_`.

## Text-to-text

| v1 | v2 |
| --- | --- |
| `skllm.models.gpt.text2text.summarization.GPTSummarizer` | `skllm.Summarizer` |
| `skllm.models.anthropic.text2text.summarization.ClaudeSummarizer` | `skllm.Summarizer` with `model="anthropic::…"` |
| `skllm.models.gpt.text2text.translation.GPTTranslator` | `skllm.Translator` |
| `skllm.models.anthropic.text2text.translation.ClaudeTranslator` | `skllm.Translator` with `model="anthropic::…"` |
| `skllm.text2text.TunableGPTText2Text`, `skllm.models.vertex.text2text.tunable.TunableVertexText2Text` | **removed** — see [Fine-tuning](#fine-tuning-removed) |

`max_words`, `focus` and `output_language` keep their names and defaults. Both
estimators are transformers only: the v1 `predict` alias is gone, so call
`transform`. Where v1 raised once its retries were exhausted, v2 fills the
failed sample with `""` and reports it — unless you ask for the old behaviour
with `on_error="raise"`.

## Named entity recognition

| v1 | v2 |
| --- | --- |
| `skllm.models.gpt.tagging.ner.GPTExplainableNER` | `skllm.NER` |
| `skllm.models.anthropic.tagging.ner.AnthropicExplainableNER` | `skllm.NER` with `model="anthropic::…"` |
| `skllm.utils.rendering.render_ner`, `display_ner` | `skllm.tagging.render_tagged_html` |
| `skllm.utils.xml.filter_xml_tags`, `filter_unwanted_entities`, `replace_all_at_once`, `json_to_xml` | **removed** — internal to v1's dense/sparse JSON→XML machinery |

The `entities={label: description}` mapping survives by name — keyword-only now,
like every v2 parameter. Everything else about the task moved: `transform`
returns the input text with entities wrapped in inline `<LABEL>…</LABEL>` tags —
no `<entity><reasoning>…` envelope, no `sparse_output` switch, no
`display_predictions` side effect, and `num_workers` is now `max_concurrency`.
Stripping the tags off an output reproduces the input exactly; an output that
cannot be reconciled with the source is a sample failure that fills with the
untagged original.

## Vectorization

| v1 | v2 |
| --- | --- |
| `skllm.models.gpt.vectorization.GPTVectorizer` | `skllm.Vectorizer` |
| `skllm.models._base.vectorizer.BaseVectorizer` | **removed** — subclassing is not the extension point; register a backend instead |

`batch_size` defaults to 32 (was 1) and `key`/`org` become `api_key`. Embeddings
stay all-or-nothing: a failed batch raises even under `on_error="fill"`, because
a zero-vector fill would quietly poison everything downstream.

## Memory / vector stores

| v1 | v2 |
| --- | --- |
| `skllm.memory.SklearnMemoryIndex` | **removed** — `DynamicFewShotClassifier` builds its own per-class `NearestNeighbors` index |
| `skllm.memory._annoy.AnnoyMemoryIndex` | **removed** with the `annoy` extra |
| `skllm.memory.base.IndexConstructor` | **removed** with the `memory_index` parameter |

## Configuration and credentials

`skllm.config.SKLLMConfig` is gone in full. It mutated process-global
environment variables, which is neither thread-safe nor safe to pickle; v2 reads
the environment but never writes it, and everything programmatic flows through
`skllm.config_context` (contextvars — per thread, per async task, nestable).

| v1 | v2 |
| --- | --- |
| `SKLLMConfig.set_openai_key(k)` / `set_gpt_key(k)` | `skllm.config_context(openai_api_key=k)`, `ZeroShotClassifier(api_key=k)`, or `OPENAI_API_KEY` |
| `SKLLMConfig.get_openai_key()` | — (resolution is internal; there is no getter that could leak a key) |
| `SKLLMConfig.set_openai_org(org)` / `set_gpt_org(org)` | `skllm.config_context(client_options={"openai": {"organization": org}})`, or the SDK's own `OPENAI_ORG_ID` |
| `SKLLMConfig.set_anthropic_key(k)` | `skllm.config_context(anthropic_api_key=k)` or `ANTHROPIC_API_KEY` |
| `SKLLMConfig.set_gpt_url(url)` / `reset_gpt_url()` | `ZeroShotClassifier(base_url=…)` or `skllm.config_context(openai_base_url=…)` — scoped, so nothing needs resetting |
| `SKLLMConfig.set_azure_api_base(…)`, `set_azure_api_version(…)` | **removed** — reach Azure through `model="litellm::azure/<deployment>"` and litellm's own `AZURE_*` variables |
| `SKLLMConfig.set_google_project(p)` / `get_google_project()` | **removed** — the Google backend is the Gemini API (`GOOGLE_API_KEY`/`GEMINI_API_KEY`), not Vertex |
| `SKLLMConfig.set_gguf_max_gpu_layers(…)`, `set_gguf_verbose(…)`, `get_gguf_download_path()` | **removed** with the GGUF backend |
| env `SKLLM_CONFIG_OPENAI_KEY` | env `OPENAI_API_KEY` |
| env `SKLLM_CONFIG_OPENAI_ORG` | env `OPENAI_ORG_ID` (read by the OpenAI SDK itself) |
| env `SKLLM_CONFIG_ANTHROPIC_KEY` | env `ANTHROPIC_API_KEY` |
| env `SKLLM_CONFIG_GPT_URL` | env `OPENAI_BASE_URL`, or the `base_url` param |
| env `GOOGLE_CLOUD_PROJECT` | env `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| env `SKLLM_CONFIG_AZURE_API_BASE`, `SKLLM_CONFIG_AZURE_API_VERSION`, `SKLLM_CONFIG_GGUF_*` | **removed** |

`config_context` also carries what v1 had no home for: `credentials={alias: key}`
for per-estimator key routing, `overrides={alias: {...}}` for targeted
generation-param changes, `pricing=` for cost estimates, `timeout=`,
`max_retries=`, and `allow_env_fallback=False` to forbid ambient keys entirely
(also settable as `SKLLM_ALLOW_ENV_FALLBACK=0`).

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

## Model strings and backends

v1 routed a handful of prefixes through `skllm.llm.gpt.utils.split_to_api_and_model`
(`openai`, `azure`, `gguf`, `custom_url`). v2 keeps the `provider::model_id`
shape — and a bare string still means OpenAI — but the prefix now selects a
backend adapter, and the set is different:

| v1 model string | v2 |
| --- | --- |
| `"gpt-4"` (bare) | `"gpt-4"` — still OpenAI |
| `"azure::my-deployment"` | `"litellm::azure/my-deployment"` |
| `"custom_url::my-model"` | `"openai::my-model"` plus `base_url="http://…/v1"` |
| `"gguf::llama3-8b-q4"` | `"ollama::llama3"` — local inference is served by Ollama, never in-process |
| `"gpt4all::…"` | removed in v1.4 already; use `"ollama::…"` |
| Vertex `text-bison@002` classes (PaLM 2, shut down) | `"google::gemini-2.5-flash"` (Gemini API) or `"litellm::vertex_ai/gemini-2.5-pro"` |
| — | `"anthropic::claude-sonnet-5"`, and any prefix you add with `skllm.register_backend` |

`skllm.llm.*` (the `GPTMixin` / `ClaudeMixin` / `VertexMixin` families,
`get_chat_completion`, `get_embedding`, `set_credentials`, `LlamaHandler`,
`SUPPORTED_APIS`, …) is deleted wholesale. Its replacement is
`skllm.backends`: one small adapter object per provider behind a protocol of
`chat` / `structured` / `embed`, discovered by prefix. Adding a provider is one
adapter plus `skllm.register_backend("mycorp", adapter)` — no mixin stack, and
no new estimator classes.

## Fine-tuning (removed)

`GPTClassifier`, `MultiLabelGPTClassifier`, `VertexClassifier`,
`TunableGPTText2Text`, `TunableVertexText2Text`, `skllm.models._base.classifier.BaseTunableClassifier`,
`skllm.models._base.text2text.BaseTunableText2TextModel` and the whole
`skllm.llm.gpt.clients.openai.tuning` / `skllm.llm.vertex.tuning` machinery are
removed with no replacement.

They targeted `gpt-3.5-turbo-0613` and `text-bison`, both retired, and their
`fit` mutated constructor parameters — the sklearn-contract violation v2 exists
to end. Tune with the provider's own tooling and point a v2 estimator at the
resulting model id: `ZeroShotClassifier(model="openai::ft:…")`.

## Base classes, prompts and utilities

| v1 | v2 |
| --- | --- |
| `skllm.models._base.classifier.BaseClassifier`, `BaseZeroShotClassifier`, `BaseCoTClassifier`, `BaseFewShotClassifier`, `BaseDynamicFewShotClassifier`, `BaseTunableClassifier`, `SingleLabelMixin`, `MultiLabelMixin` | **removed** — v2 composes providers instead of inheriting them; the shared machinery lives in `skllm.core` and is not a public extension point |
| `skllm.models._base.text2text.*`, `skllm.models._base.tagger.*`, `skllm.models._base.vectorizer.*` | **removed**, same reason |
| `skllm.prompts.builders.build_*_prompt*` | **removed** — pass `prompt_template="…"` to any estimator instead |
| `skllm.prompts.templates.*` (`ZERO_SHOT_CLF_PROMPT_TEMPLATE`, …) | `skllm.prompts` (`ZERO_SHOT_SINGLE_LABEL`, `FEW_SHOT_SINGLE_LABEL`, `ZERO_SHOT_MULTI_LABEL`, `FEW_SHOT_MULTI_LABEL`, `SUMMARY`, `TRANSLATION`, `NER_TAGGING`, `EXTRACTION`, `RUBRIC_SCORING`, `CLUSTER_NAMING`) — different placeholders, validated at `fit` |
| `skllm.model_constants.OPENAI_GPT_MODEL`, `OPENAI_EMBEDDING_MODEL`, `ANTHROPIC_CLAUDE_MODEL`, `OPENAI_GPT_TUNABLE_MODEL`, `VERTEX_DEFAULT_MODEL` | **removed** — the defaults are `skllm.core.DEFAULT_CHAT_MODEL` (`openai::gpt-5.6-sol`) and `skllm.core.DEFAULT_EMBEDDING_MODEL` (`openai::text-embedding-3-small`) |
| `skllm.utils.to_numpy`, `find_json_in_string`, `re_naive_json_extractor`, `extract_json_key`, `retry` | **removed** — input coercion, JSON salvage and retry are internal to `skllm.core` and the backends |
| `skllm.utils.rendering.get_random_color`, `is_running_in_jupyter`, `color_palettes` | **removed** — `skllm.tagging.render_tagged_html` returns HTML that notebooks render on their own |

## Datasets (unchanged)

`get_classification_dataset`, `get_multilabel_classification_dataset`,
`get_summarization_dataset` and `get_translation_dataset` keep their names,
their import path (`skllm.datasets`) and their contents.

## New in v2

Things with no v1 equivalent, covered in the [README](README.md):

- `skllm.StructuredExtractor` — schema-driven extraction (JSON Schema or a pydantic model).
- `skllm.RubricScorer` — rubric-based numeric scoring, as a regressor.
- `skllm.ClusterLabeler` — names the clusters any clusterer produced.
- `skllm.ConsensusClassifier` — vote across providers and models.
- `skllm.save` / `skllm.load` / `skllm.InferenceModel` — FNNX bundles that run
  under the FNNX runtime and reload into a live estimator, with credentials
  excluded by construction (v1 had no serialization story at all — pickling an
  estimator embedded your key and, for few-shot models, your training corpus).
- `usage_` / `errors_` on every estimator, plus OpenTelemetry `gen_ai` spans.
