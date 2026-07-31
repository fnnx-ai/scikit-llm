# Proposals

## Problem

scikit-llm v1.4.x is ~3 years old and no longer solves modern problems:

1. **Incorrect and inconsistent sklearn usage.** Estimators validate and raise in `__init__`, mutate constructor params during `fit` (breaking `get_params`/`clone`), never use `check_is_fitted`, have non-standard signatures (`predict(X, num_workers=1)`), and contain outright bugs (a walrus-operator precedence bug leaks `True` into predicted labels; `import skllm.classification` fails at HEAD with an `ImportError`; all three tunable estimators crash on construction).
2. **Provider-as-inheritance architecture.** Each provider is a mixin stack inherited into every estimator, producing a 3-providers × ~10-tasks class explosion that is only ~40% filled and impossible to keep consistent. Adding a provider means adding a dozen classes.
3. **Dead backends and dead features.** PaLM/text-bison is shut down, the Vertex integration is un-importable, gpt4all was already removed, the llama.cpp/GGUF integration is a heavy maintenance burden, and the fine-tuning API targets `gpt-3.5-turbo-0613` — nobody tunes cloud LLMs for small tasks anymore.
4. **No modern LLM capabilities.** No reasoning-model support (the temperature carve-out checks the nonexistent prefix `gpt-o`), no native structured outputs outside plain OpenAI (Azure/Anthropic emulate JSON mode via prompt suffixes), no async or concurrent execution, primitive retry, and silent failure: a failed API call becomes `""`, which is then replaced by a random label — indistinguishable from a real prediction.
5. **Credential handling is inconvenient and unsafe.** Keys live in process-global env vars set via `SKLLMConfig`, and per-estimator keys are stored as plain attributes — so pickled models embed API keys.
6. **No serialization story.** There is no save/load API at all; users pickle estimators, which embeds keys and, for few-shot models, entire training corpora, with no portable inference artifact.

## Proposed solution

Rebuild the library as **scikit-llm v2** (a major, largely rewritten release):

- **Unified, task-based estimators** (classifier, summarizer, translator, NER, vectorizer) that are strictly sklearn-compliant, with the provider selected by configuration rather than by class — collapsing the N×M class explosion into one class per task.
- **API-only provider backends**: OpenAI, Anthropic, Google (Gemini API), and Ollama as first-class backends, plus **LiteLLM as an optional catch-all backend**. No in-process local models — local inference is served by Ollama.
- **Modern LLM features as core**: reasoning-model awareness, native structured outputs on every backend that supports them, concurrent batched inference with robust retry, and an explicit per-sample error policy instead of silent empty strings.
- **Per-estimator credentials that are never serialized**: keys resolve at call time through a layered mechanism (explicit per-estimator key → scoped context → environment fallback) and are re-supplied through those same layers after loading, never written into artifacts.
- **FNNX (pyfunc variant) as the serialization format** — scikit-llm is part of the FNNX ecosystem. Saving produces a self-describing `.fnnx` bundle that both runs under the FNNX `Runtime` (portable inference) and reloads into a live sklearn estimator (round-trip), with credentials excluded by construction.
- **Remove the training/tuning API and all dead backends** (Vertex/PaLM, GGUF/llama.cpp) outright.
- **New capabilities**: multi-model consensus ensembles (MoE-style voting across providers/models); `predict_proba`/confidence via self-consistency sampling; a redesigned NER that outputs inline-tagged text, built on modern models' ability to reproduce text with inline annotations reliably (v1's dense/sparse JSON→XML machinery predates that ability); a schema-driven structured-extraction transformer; an LLM-judge scorer (rubric-based numeric scoring); a cluster/topic labeler that names clusters from exemplars; and per-run usage tracking on every estimator, exposed to observability backends via OpenTelemetry `gen_ai` spans — which luml ingests natively (luml is FNNX-native; its registry accepts `.fnnx` artifacts directly).

## Why this approach

- **Composition over inheritance** is the single change that fixes most structural problems at once: one estimator per task, providers become small interchangeable adapter objects, new providers cost one adapter instead of a class family, and sklearn compliance is implemented once in shared base classes.
- **API-only** reflects reality: Ollama won local inference, and every backend speaking an HTTP API means estimators hold only lightweight, serialization-friendly configuration — which is exactly what FNNX pyfunc packaging and the key-reinjection model need.
- **FNNX over bare pickle/joblib files**: the artifact is self-describing (manifest with IO signature, declared env vars and per-call attributes), environment-pinned (exact dependency versions travel with the model), deployable through the FNNX ecosystem (luml registry/satellites), and credentials are excluded by construction. Internally, fitted state still travels as a joblib payload governed by custom credential-stripping pickling semantics — FNNX's value is the contract around the state, not a new state format.
- **Dropping tuning outright** rather than porting it: the feature is dead in practice, its v1 implementation is already broken, and removing it eliminates the worst sklearn-contract violations (fit mutating constructor params).
- **Clean break over a compat layer**: v2 ships only the new API; v1 users pin `scikit-llm<2` and a migration guide maps old names to new ones. Half of the v1 surface is already broken or calls shut-down APIs, so a compat layer would preserve the appearance of stability without the substance — and it would drag the N×M class explosion into v2's public API.

# Design

## Architecture overview

Three layers, composed rather than inherited:

1. **Config layer** (`skllm/config.py`) — layered, contextvars-based resolution of credentials and settings. No estimator or backend reads `os.environ` directly; everything goes through this layer.
2. **Backend layer** (`skllm/backends/`) — one small adapter per provider implementing a shared protocol: chat completion, structured (JSON-schema) completion, and embeddings, each returning content plus usage (token/request counts); each adapter also declares its credential requirement (a canonical key env var, or none), consumed by config resolution and save-time discovery. A registry maps provider prefixes to adapters and is user-extensible (a public function registers custom adapters under a new prefix).
3. **Estimator layer** (task modules) — sklearn estimators that hold only declarative configuration (model string, prompt options, policy options) and fitted state. They obtain a backend from the registry at call time and never store client objects, keys, or connections as instance state.

Shared estimator infrastructure (`skllm/core/`) implements once: sklearn plumbing (tags, fitted checks, input validation), the concurrent request engine, the error policy, usage aggregation, and OpenTelemetry emission.

## Package layout

New tree replacing the v1 internals entirely (v1's `skllm/llm/`, `skllm/models/`, `skllm/memory/`, old `config.py`, `model_constants.py` are deleted):

- `skllm/__init__.py` — re-exports every public estimator plus `config_context`, `save`, `load`, `InferenceModel`, `register_backend`, `__version__` (single source of truth; `pyproject.toml` reads it dynamically).
- `skllm/config.py` — config context and resolution.
- `skllm/backends/` — protocol/base, registry, `openai.py` (also serves Ollama and any OpenAI-compatible endpoint), `anthropic.py`, `google.py`, `litellm.py`.
- `skllm/core/` — base estimator classes, request engine, errors, usage, telemetry.
- Task modules (canonical import paths): `skllm/classification.py`, `skllm/ensemble.py`, `skllm/text2text.py`, `skllm/tagging.py`, `skllm/extraction.py`, `skllm/scoring.py`, `skllm/clustering.py`, `skllm/vectorization.py`.
- `skllm/prompts/` — default prompt templates per task.
- `skllm/serialization/` — FNNX packaging: the static PyFunc wrapper module (self-contained, shipped as part of the package) plus save/load.
- `skllm/datasets/` — retained demo datasets (as in v1).

## Provider backends

**Model strings.** Every estimator takes a `model` parameter of the form `provider::model_id`. The default is `openai::gpt-5.6-sol` for chat estimators and `openai::text-embedding-3-small` for the `Vectorizer`, so every estimator constructs and runs with no arguments. A bare string without `::` defaults to the OpenAI provider (v1 convention). Recognized prefixes: `openai`, `anthropic`, `google`, `ollama`, `litellm`, plus any user-registered prefix. An unknown prefix raises immediately at first use with the list of available providers. The `custom_url::` and `azure::` prefixes are gone: OpenAI-compatible endpoints are reached via `openai::` plus a `base_url`; Azure and every other exotic provider go through `litellm::`.

**Adapters.**
- *OpenAI* (core dependency): chat, native structured outputs (JSON schema), embeddings. The Ollama backend is this adapter pointed at Ollama's OpenAI-compatible endpoint (default `http://localhost:11434/v1`, overridable via `base_url` or `OLLAMA_HOST`), with no API key required.
- *Anthropic* (extra `anthropic`): chat; structured outputs via forced tool-use; no embeddings (embedding-dependent estimators raise a clear error telling the user to supply a vectorizer from another provider).
- *Google* (extra `google`): the Gemini API via the `google-genai` SDK — not Vertex. Chat, native structured outputs (response schema), embeddings.
- *LiteLLM* (extra `litellm`): catch-all; the model id after `litellm::` is passed to litellm verbatim. Structured outputs where litellm reports support, else prompt-based JSON fallback. Importing the backend without the extra installed raises with the exact `pip install scikit-llm[litellm]` hint (same pattern for all extras).

**Behavior contracts common to all adapters:**
- Every call returns content plus a usage record (request count, input/output tokens where the provider reports them).
- Retries with exponential backoff on transient failures (rate limits, timeouts, 5xx), honoring provider retry-after hints where available; retry count and timeout are configurable through the config layer. After retries are exhausted the error propagates to the estimator's error policy.
- Reasoning-model awareness — skllm-injected defaults are advisory, user params are contractual. skllm's determinism defaults — temperature 0, plus reasoning disabled where the provider exposes a reasoning-control param (OpenAI `reasoning_effort: "none"`), keeping the default model cheap and fast for classification-shaped tasks — are applied best-effort: if the provider rejects a request with a parameter-level "unsupported parameter" error, the adapter strips the offending *skllm-injected* default and retries once, caching the outcome per (provider, model) for the process lifetime. A static seed list of known reasoning families skips the first wasted round-trip but is never the source of truth — the error-driven fallback is, so new models work without library updates (v1's static `gpt-o` prefix check matched nothing and silently broke). User-supplied `model_params` are never stripped or rewritten — an explicitly set unsupported param surfaces the provider's error. This one-shot adaptation is separate from the transient-failure retry budget.
- `model_params`: an optional dict parameter on every estimator, passed through to the provider API verbatim — the home for all provider-specific *request-level* arguments (temperature, max output tokens, OpenAI reasoning effort, Anthropic thinking budgets, Google top_k, Ollama num_ctx, …). scikit-llm does not validate these against per-provider allow-lists; an unsupported param surfaces the provider's own error. (The one cross-param check: an explicit temperature of 0 combined with `n_predictors > 1` is rejected at validation — see Sampled voting.) User-supplied values always win over skllm defaults. Client-level provider options do not belong here (they are connection config; see Credentials and configuration).
- Structured outputs are the default path for every task with a constrained output (classification, extraction, scoring): native JSON-schema mode where the provider supports it, otherwise a prompt-embedded schema with salvage parsing. Parsing failure after salvage counts as a sample failure (error policy), never as a silent empty result.

## Credentials and configuration

Estimator configuration falls into two classes with different routing:

- **Behavioral config** — model string, `model_params`, prompt template, task params. Defines *what the model is*. Plain sklearn constructor params: per-estimator by nature, serialized with the estimator, no routing mechanism needed.
- **Connection config** — credentials, base URLs, transport settings (timeout/retry) and client-level provider options. Defines *where and as whom the model runs* — deployment topology, not model identity. Stripped from every serialized form and resolved at call time through the layers below. This is the only class that ever needs a routing mechanism in bundles, which is precisely what the credential alias provides.

The membership test: *should this value survive save/load?* Temperature yes → behavioral. Host, key, org id no → connection.

Resolution order for any connection setting (per provider): **estimator parameter → active `config_context` → environment variable → default**.

Environment variables are strictly a read-only, outermost fallback for ambient provisioning (containers, CI): **scikit-llm never writes to `os.environ`** — this is an invariant. v1's thread-safety hazard was `SKLLMConfig` setters mutating process-global env at runtime; in v2 all programmatic and per-call configuration flows through contextvars, and the env layer is only ever read.

The env layer can be switched off for credentials: `allow_env_fallback` (default true) is a config-layer setting; when false, credential resolution (provider keys and alias credentials) skips the environment entirely — for shared or multi-tenant services where an ambient key must never mask a missing per-request one. Set per scope via `config_context(allow_env_fallback=False)` or ambiently for a whole process via `SKLLM_ALLOW_ENV_FALLBACK=0`. A credential miss under disabled fallback says so explicitly in the error, so a present-but-ignored env key is not mystifying. Non-secret settings (base URLs, timeouts) are unaffected.

- Every estimator accepts `api_key` and `base_url` (applied to its provider). These are real sklearn params (visible to `get_params`/`clone` so in-memory workflows like grid search keep working) but are excluded from every serialized form: FNNX bundles never contain them by construction, and `__getstate__` strips them so even a raw pickle is key-free.
- `skllm.config_context(...)` is a context manager over contextvars — safe per-thread and per-async-task, nestable (inner overrides outer). It accepts per-provider keys (`openai_api_key`, `anthropic_api_key`, `google_api_key`), base URLs, retry/timeout settings, and the pricing map (below).
- Environment fallback uses the providers' standard variable names (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`/`GEMINI_API_KEY` — the more specific `GEMINI_API_KEY` wins when both are set — `OLLAMA_HOST`) — not v1's `SKLLM_CONFIG_*` names. LiteLLM reads its own provider env vars natively.
- **Client-level provider options** (e.g. OpenAI organization/project, custom headers) are connection config: supplied via a per-provider options map in `config_context`, read by the adapter when issuing requests, never present on estimators or in bundles. Provider SDKs' native env conventions (`OPENAI_ORG_ID`, …) work out of the box, since the SDKs read them themselves — read-only env, consistent with the invariant. Request-level provider args, by contrast, belong in the estimator's `model_params` (behavioral, serialized, passed through verbatim).
- **Estimator aliases** give an estimator a durable name in the config namespace: an optional `alias` param — a string matching `[a-z][a-z0-9_]*` (validated at `fit`/first use, keeping the `SKLLM_CREDENTIAL_<ALIAS>` env mapping unambiguous), serialized normally, so it survives bundling (unlike stripped connection config). Aliases serve two purposes:
  - *Credential routing*, for same-provider estimators needing different keys (which per-provider resolution cannot express, and which per-estimator `api_key` params cannot survive serialization): an aliased estimator resolves its key strictly by name (only an explicit `api_key` constructor param, the most local layer, still outranks the alias) — `config_context(credentials={"<alias>": ...})` → env `SKLLM_CREDENTIAL_<ALIAS>` (uppercased); set-but-unresolved errors naming the alias, never silently falling back to the provider-level key (which would quietly merge deliberately separated credentials). Without an alias, the provider-level layers apply as above.
  - *Targeted behavioral overrides*: `config_context(overrides={"<alias>": {...model_params...}})` reaches into a graph — a live fitted pipeline or a served bundle alike — and adjusts one specific estimator's generation params for the duration of the scope. Precedence is deliberately inverted relative to credentials: an override sits *above* the estimator's stored `model_params` (override > stored param > skllm default), because overrides exist to beat a stored value, whereas credential layers exist to fill a missing one. Estimators without an alias cannot be individually targeted; there is no broadcast override (a blanket temperature would also reach embedding calls verbatim).
  In v2 the alias routes credentials and behavioral overrides only; per-alias base URLs are deferred (the provider-granular base-url attribute covers serving topology; `base_url_<alias>` remains reserved as a compatible extension).
- Resolution happens at call time on every request batch; nothing resolved is cached onto the estimator.

## Estimator contracts

**Common sklearn behavior (implemented once in `skllm/core`):**
- `__init__` only assigns params verbatim; validation happens at `fit`/first use. `get_params`/`set_params`/`clone` round-trip exactly. Fitted state lives only in trailing-underscore attributes. Stateful estimators (all classifiers, `ConsensusClassifier`, `ClusterLabeler`) raise sklearn's `NotFittedError` when used unfitted; stateless ones (`Summarizer`, `Translator`, `NER`, `StructuredExtractor`, `RubricScorer`, `Vectorizer`) declare the corresponding sklearn tag — their `fit` only validates params, and `transform`/`predict` work without a prior `fit` (params are validated at first use either way).
- Estimators declare sklearn 1.6+ tags (`__sklearn_tags__`) marking text (non-numeric) input; X may be a list of strings, numpy array of strings, or pandas Series (pandas optional, duck-typed).
- All per-sample LLM calls within `predict`/`transform` run concurrently through the shared request engine, bounded by a `max_concurrency` parameter on every estimator (conservative default), with tqdm progress controlled by a `verbose` param on every estimator (default `1`: progress bar for multi-sample batches; `0`: silent). An empty input returns an empty result without issuing requests. The engine captures the caller's contextvars context and runs every worker task inside a copy of it (contextvars do not propagate to threads on their own), so `config_context` credentials reach all concurrent requests.
- Estimators are thread-safe for concurrent prediction on a shared fitted instance: credentials and settings resolve into call-locals (never onto `self`), and the mutable bookkeeping (usage counters, per-call error reports) is updated under a lock.
- Every estimator maintains a usage report attribute updated after each fit/predict/transform: request and token counters plus failure counts, for the most recent operation and cumulatively. When a pricing map (model id → price per million input/output tokens) is provided via the config layer, the report includes cost estimates. No bundled price table — prices go stale; users supply them.
- Error policy: an `on_error` parameter with values `"fill"` (default) and `"raise"`. Under `"fill"`, a sample whose call ultimately fails receives that estimator's documented fill value, a warning summarizing the failure count is emitted, and a per-sample error report (index, error) is available as an attribute after the call. Under `"raise"`, the first exhausted failure aborts the batch with the underlying provider error.
- An optional `prompt_template` parameter overrides the default task template; placeholders are validated at fit time against the task's documented set.

**Classifiers** (`skllm/classification.py`): `ZeroShotClassifier`, `FewShotClassifier`, `DynamicFewShotClassifier`, and `MultiLabelZeroShotClassifier`, `MultiLabelFewShotClassifier`.
- `fit(X, y)`: zero-shot ignores X (may be None); few-shot stores X/y as demonstration examples; all derive `classes_` from y (sklearn convention: sorted unique labels). Multi-label y is an iterable of label collections. Labels must be strings — non-string labels raise at `fit` with a message to cast them (int labels don't survive prompt/JSON round-trips faithfully); `default_label` must be a string or `None`.
- Label output is obtained via structured outputs constrained to `classes_`. A response with a label outside `classes_` counts as a sample failure.
- Single-label `predict` returns an array of labels; fill value is the `default_label` parameter (default `None`, no v1 "Random" magic). Multi-label `predict` returns a list of label lists (no padding); fill value is the empty list. `max_labels` is enforced as a schema constraint where the provider supports it, with deterministic truncation as backstop (the first `max_labels` labels of the response — not a sample failure). List-of-lists output is a deliberate departure from sklearn's binary-indicator convention (ragged, JSON-friendly, no up-front label universe); use `MultiLabelBinarizer` when sklearn metrics need indicators.
- **Sampled voting**: every classifier takes `n_predictors` (int ≥ 1 or `None`, default `None`). When unset (or 1), `predict` issues one call per sample under skllm's determinism default, and `predict_proba` is degenerate by design — a one-hot row over `classes_` from that single call (multi-label: per-class 0/1 memberships), at no hidden extra cost. When `n_predictors = k > 1`, *both* `predict` and `predict_proba` issue k concurrently sampled calls per sample: skllm suppresses its temperature-0 default on these calls (provider-default sampling supplies vote diversity; other injected defaults such as disabled reasoning still apply, and user `model_params` still pass through verbatim), `predict` majority-votes (ties broken by `classes_` order), and `predict_proba` returns vote fractions. Multi-label with k votes: a label is predicted when it appears in a strict majority of successful votes, then capped by `max_labels` most-frequent-first; probabilities are independent per-class frequencies. An explicit user temperature of 0 combined with `n_predictors > 1` is contradictory (identical votes at k× cost) and raises at param validation; the same contradiction arriving through an alias-targeted override (in-memory scope or serving attribute) is rejected at call time — the guard applies to the effective explicit temperature, not only the stored param. Individual vote failures are tolerated — aggregation runs over the successful votes; the sample fails only when all k calls fail. `n_predictors` is behavioral config: serialized with the estimator, so a served bundle's `predict` aggregates identically; request cost scales by k.
- `DynamicFewShotClassifier` additionally takes `n_examples` and a `vectorizer` (a skllm Vectorizer instance). At fit it embeds training texts and builds one nearest-neighbor index per class (sklearn `NearestNeighbors` only — the annoy option is dropped); at predict it retrieves up to `n_examples` per class (classes with fewer samples contribute what they have — fixing v1's crash). If no vectorizer is given, a default is built only for providers with a designated default embedding model — `openai::text-embedding-3-small`, `google::gemini-embedding-001`; for every other provider (anthropic, ollama, litellm, custom) `fit` raises requiring an explicit `vectorizer` (no silent cross-provider default as in v1, no guessing which local or proxied embedding model exists).
- v1's CoT classifier is dropped — reasoning models subsume it.

**Consensus ensemble** (`skllm/ensemble.py`): `ConsensusClassifier`, sklearn-VotingClassifier-flavored: takes a list of single-label skllm classifiers (any mix of providers/models), optional per-estimator weights, and its own `default_label` (default `None`). Like `VotingClassifier`, `fit` clones every member and fits the clones on the same data into `estimators_` — the constructor list is never mutated (a pre-fitted member is refit as a clone) — and derives the ensemble's own `classes_` from y. `predict` queries all fitted members concurrently and majority-votes per sample (weighted if weights given); a tie among top-voted labels is broken in favor of the label voted by the earliest-listed member whose vote is among them; `predict_proba` returns vote fractions — weighted, normalized over the weight of members that actually cast a vote for that sample. Member failures follow the member's own error policy (a member with `on_error="raise"` aborts the batch); a member that fills with `None` contributes no vote. A sample for which no member casts a vote is an ensemble-level sample failure: filled with the ensemble's `default_label` and recorded in its error report. Multi-label ensembles are out of scope for v2.

**Text2text** (`skllm/text2text.py`): `Summarizer` (`max_words`, optional `focus`) and `Translator` (`output_language`). Stateless transformers (`fit` only validates params); `transform` returns an array of strings. Fill value is the empty string. The v1 `predict` alias on transformers is dropped.

**NER** (`skllm/tagging.py`): `NER` with an `entities` parameter (mapping label → description). `fit` only validates the entity schema (the estimator is stateless); `transform` returns the input text with entities wrapped in inline XML-style tags named after the labels (flat, non-overlapping spans). Invariant: stripping tags from an output must reproduce the input text exactly; the implementation validates (and minimally repairs, e.g. via whitespace-tolerant alignment) model output against the source, and an output that cannot be reconciled counts as a sample failure with the untagged original text as fill value. A small utility renders tagged text as highlighted HTML for notebooks (successor of v1's rendering util).

**Structured extraction** (`skllm/extraction.py`): `StructuredExtractor` with a `schema` parameter — a JSON Schema dict or a pydantic model class (converted to its schema). Stateless transformer; `transform` returns a list of dicts, each validated against the schema; validation failure after salvage is a sample failure with `None` as fill value. A pydantic class is compiled to its JSON Schema at the pickling boundary: the versioned state dict stores only the compiled dict, so pickles and bundles never carry the class and serving environments never import pydantic — after any pickle or save/load round-trip, `schema` is the dict form (a documented normalization; in-memory `get_params`/`clone` still return the class verbatim).

**Scorer** (`skllm/scoring.py`): `RubricScorer` — regressor-style. Parameters: `rubric` (free-text criteria) and `scale` (numeric min/max, default 0–10). `fit` only validates params; `predict` returns a float array; out-of-range model outputs are clamped; unparseable outputs are sample failures with NaN as fill value.

**Cluster labeler** (`skllm/clustering.py`): `ClusterLabeler` — names clusters produced by any clusterer. `fit(X, y)` takes texts and cluster ids, samples up to `n_exemplars` texts per cluster (`random_state`-controlled, sklearn semantics), makes one naming call per cluster, and stores the resulting id → name mapping as fitted state. `transform(y)` maps a vector of cluster ids to their names; the noise id `-1` maps to `None` without an LLM call; unseen ids raise. Naming calls run at `fit` and are all-or-nothing like the Vectorizer's embeddings: a naming failure raises regardless of `on_error` — a half-named mapping would silently poison every later `transform`. Because `transform` consumes cluster ids while `fit` consumes texts, the inherited `fit_transform` contract would be nonsense; `fit_transform(X, y)` is overridden to `fit(X, y).transform(y)` — naming the clustering just fitted.

**Vectorizer** (`skllm/vectorization.py`): `Vectorizer` — embedding transformer with `batch_size`; `transform` returns a 2-D float numpy array. Providers without embeddings raise at first use. Under `"fill"`, failed batches raise anyway (a zero-vector fill would silently poison downstream models — embeddings are all-or-nothing).

## Usage telemetry (luml integration)

Beyond the local usage report, each LLM request emits one OpenTelemetry span carrying `gen_ai.*` semantic-convention attributes (`gen_ai.operation.name`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`) plus cost when a pricing map is configured. Emission is best-effort and inert by default: it activates only when `opentelemetry-api` is importable and the application has configured a tracer provider; scikit-llm never installs or mutates a provider itself and takes no opentelemetry dependency (optional import). This makes usage land automatically in luml runs (whose tracker installs the provider and stores `gen_ai` attributes as typed trace columns) and in any other OTel backend. Deeper luml coupling (registry sync, experiment logging) is explicitly out of scope — a future `skllm-luml` bridge package.

## Serialization (FNNX pyfunc)

**Format.** `skllm.save(estimator, path)` (also exposed as a method on estimators) writes an `.fnnx` bundle via `fnnx.extras.builder.PyfuncBuilder`, modeled on FNNX's MLflow converter:
- One static, self-contained PyFunc wrapper module shipped inside scikit-llm (only top-level import: the FNNX PyFunc base; everything else imported in `warmup`), registered as the bundle's pyfunc class.
- **The saved object travels as a single joblib payload** in `extra_files` — the whole object graph, uniformly. skllm estimators define custom pickling semantics (`__getstate__`/`__setstate__`): credentials are stripped, and what is pickled is a stable, versioned state dict rather than raw `__dict__`, making pickle compatibility across skllm versions a deliberate, migratable contract.
- Because the envelope is the whole graph, **any sklearn composition serializes without decomposition**: `Pipeline`, `FeatureUnion`, `ColumnTransformer`, meta-estimators, or a bare skllm estimator — there is no extraction of skllm steps from foreign structures. `skllm.save` accepts any fitted sklearn estimator containing at least one skllm estimator; a purely classical sklearn object is out of scope (that is luml's own sklearn flavor).
- `extra_values` carries human-readable identity metadata — top-level class, skllm version, task/compute mapping, and the model strings/providers discovered inside — so bundles stay introspectable without unpickling.

**Discovery.** Determining which providers — and therefore which credentials — a bundle needs is a two-level, read-only process:
- A generic graph walk locates every skllm estimator inside arbitrary sklearn structures: starting from the saved object, it recurses through `BaseEstimator` params (`get_params`) and standard containers (lists/tuples/dicts, e.g. `Pipeline.steps`), with cycle protection.
- Each located skllm estimator reports its own credential requirements through a base-class introspection hook that recurses through skllm-native nesting — including components created during `fit` (e.g. the default vectorizer a `DynamicFewShotClassifier` builds when none was passed, which is fitted state and invisible to `get_params`) and `ConsensusClassifier` members. skllm classes know their own structure, so the foreign-graph walk never needs to.

The hook reports provider and `alias` assignments together, and reports serving-time requirements only: an estimator whose LLM calls happen exclusively at `fit` (`ClusterLabeler`) reports nothing, so its bundle declares no credentials — a key nothing can consume must not be declared. Their deduplicated union derives the manifest `env_vars`, the credential dynamic attributes (`<provider>_api_key` — with its canonical env var — only for key-requiring providers with at least one unaliased estimator: adapters declare their credential requirement, so openai/anthropic/google declare a key (google's canonical declared name is `GOOGLE_API_KEY`; resolution also accepts `GEMINI_API_KEY`), ollama declares none, litellm relies on its native per-provider env vars provisioned in the serving environment, and custom backends declare whatever they registered; aliased estimators resolve strictly by alias, and a key nothing can consume must not be declared; `<provider>_base_url` per discovered provider regardless, base URLs staying provider-granular; plus one required `credential_<alias>` attribute per discovered alias with its `SKLLM_CREDENTIAL_<ALIAS>` env fallback), and the provider extras pinned in `env.json`. Discovery runs once, at save time.
- Manifest: one JSON input channel (`X`) and one JSON output channel (`predictions`); `env_vars` declares the provider key variables the model needs; producer name `skllm.fnnx.ai`, and every producer tag starts with `skllm.fnnx.ai::` — including a task tag (e.g. `skllm.fnnx.ai::classification:v1`, a packaging-schema version independent of the library version) for luml deployment-compatibility matching and UI hooks. `predict_proba` is deliberately absent from the serving surface: a classifier bundle serves `predict` (vote-aggregated when `n_predictors` is set); probabilities remain an in-memory affordance, recoverable via `unwrap()`.

**IO typing.** All channels use FNNX's `JSON` content type (a pyfunc-variant feature): batch semantics live in the JSON array itself, so no tensor shapes are declared and consumers stay numpy-free — text batches are ragged and outputs nullable, which tensor channels cannot express. Bundles never default to permissive `{}` schemas: `skllm.save` generates real JSON Schemas into `dtypes.json`, specific to the saved model:
- Input `X`: array of strings for text estimators; array of integers for `ClusterLabeler` (cluster ids).
- Output `predictions`, per task and fitted state: single-label classifiers declare an enum over the actual `classes_` — plus the stored `default_label` when it is a string outside them — nullable for `None` fills; multi-label declares arrays of the `classes_` enum; text2text and NER declare arrays of strings; `RubricScorer` arrays of nullable numbers (NaN fills serialize as JSON `null`); `Vectorizer` arrays of number arrays; `ClusterLabeler` arrays of nullable strings (`-1` → `null`).
- `StructuredExtractor`: the user's extraction schema itself becomes the declared output dtype — a pydantic model is compiled via `model_json_schema()` at save time (pydantic is a save-time-only dependency, never needed in the serving environment); a raw JSON Schema dict embeds as-is; items are nullable for fills.
For composites, typing is **edge-based and uniform**: the entry step determines the input schema and the exit step the output schema, by the same per-estimator derivation — precise schemas for skllm estimators, convention-derived ones for foreign sklearn steps (a fitted classifier with `classes_` yields a nullable enum over them; a regressor yields nullable numbers; a transformer yields number arrays), and a permissive JSON schema only as the last-resort fallback for unrecognizable steps. A single-estimator bundle is the degenerate case of the same rule. The derived entry/exit kinds are recorded in `extra_values`, where they also drive the `InferenceModel` IO conversion (below) — one derivation feeds both the manifest and the loader, so they cannot disagree.

FNNX validates inputs against these schemas at compute time; the FNNX local handler does not validate outputs — output schemas are the published contract (and drive luml's generated OpenAPI page), while actual output validation continues inside the estimators per the error policy.
- Declared runtime dependencies (`env.json`): pinned `scikit-llm` plus the provider extras required by the discovered model strings (none for `openai::`/`ollama::`, whose SDK is a core dependency). This is mandatory, not optional: luml satellites build the serving conda environment solely from `env.json`, falling back to a bare `fnnx[core]` env (no scikit-llm, no provider SDK) when it is missing.

**Dynamic attributes.** FNNX's pyfunc variant passes `dynamic_attributes` through to the wrapper unvalidated and manifest `Var`s carry no default values (defaults exist only at the op level, which pyfunc bundles don't use), so the wrapper implements the attribute contract itself, mirroring FNNX's op-level semantics:
- *Overridable per call*: credentials (namespaced per provider, e.g. `openai_api_key`), base URLs, and generation params (`model_params` members: temperature, reasoning effort, max output tokens, …) through the `overrides` mechanism below — per-member targeting inside composites requires aliases.
- *Defaults*: the serialized estimator config in `extra_values` is the default set — a call with no dynamic attributes reproduces the estimator's saved behavior exactly.
- *Secrets are always required*: never serialized, hence no default. Resolution per call: dynamic attribute → the serving process's config layers (an active `config_context` if any, then env, honoring `allow_env_fallback`); if nothing provides a needed key, `compute` fails with an error naming the attribute and env var. Warmup never needs credentials (no clients are built at load).
- *Reinjection never mutates estimators at serving time*: the wrapper translates credential dynamic attributes into a `config_context` entered around the underlying `predict`/`transform` call. Because every skllm estimator resolves credentials through the layered config at call time, one per-provider context key blankets the entire object graph regardless of nesting depth — there is no per-estimator key plumbing in the wrapper.
- *Fixed*: structural identity — the model string, prompt template, labels/entities/schema, `n_predictors`, task mapping — is not overridable.
- *Generation-param overrides*: every bundle declares a single JSON `overrides` dynamic attribute. Bundle kind: a bundle is *single-estimator* when the saved top-level object is itself a skllm task estimator other than `ConsensusClassifier` (whose LLM calls happen in its members) — chat-less estimators like a bare `Vectorizer` included. Internal skllm components (a param-supplied or fit-created vectorizer) don't change the kind: they count for credential discovery (their `credential_<alias>` attributes work as everywhere), but serving-time generation-param overrides reach only the top-level estimator. Everything else is *composite*: pipelines, foreign wrappers, `ConsensusClassifier`. In a composite bundle `overrides` maps estimator alias → generation-param overrides, validated against the aliases discovered at save time (unknown aliases rejected with the available list); members without an alias cannot be individually tuned at serving time — alias them before saving if that matters. Composite overrides feed the alias-targeted `config_context` the wrapper enters around the call. In a single-estimator bundle `overrides` is the generation-param mapping itself: since the estimator may have no alias, the wrapper applies it internally as a call-scoped merge over the top-level estimator's stored `model_params` (override > stored > skllm default — same precedence as the alias path), never mutating persistent state and never touching internal components' calls; the public config layer stays alias-only (no broadcast form). The kind is recorded at save time, so the wrapper knows which shape to expect. Single-estimator bundles additionally declare one flat dynamic attribute per key of the stored `model_params` (default = the stored value, e.g. `temperature`), each defined as sugar over an `overrides` entry; params absent from the stored config remain reachable through `overrides`.
- The wrapper rejects unknown dynamic-attribute names (FNNX would silently ignore them, turning typos into no-ops).
- Numeric attributes arriving as strings are coerced to their declared type (luml's satellite documents every dynamic attribute as a string in its generated OpenAPI, so `"0.3"` for temperature is a normal payload).
- The wrapper's `compute_async` is genuinely non-blocking (provider I/O awaited or offloaded, never a sync alias): luml's serving worker prefers `compute_async` and runs a single-process event loop, so a blocking implementation would serialize all inference on a deployment.
- Every attribute is declared in the manifest as a `Var`, with variable-level tags following ecosystem conventions: credentials (provider key and `credential_<alias>` dynamic attributes alike, and their `env_vars`) are tagged `skllm.fnnx.ai::provider_api_key:v1` **plus** luml's `dataforce.studio::runtime_secret:v1` — the exact string luml's deployment UI matches to render an orbit-secret picker instead of a plaintext field, inject the bound secret per request, and scrub the value from the published OpenAPI schema and telemetry; generation params are tagged `skllm.fnnx.ai::generation_param:v1`. (Manifest-level `producer_tags` all start with `skllm.fnnx.ai::`; variable-level tags may additionally carry foreign-namespace markers like luml's secret tag.)

**Round-trip.** `skllm.load(path, handler=...)` returns an **`InferenceModel`** — a lightweight inference facade over `fnnx.Runtime`, not the raw estimator. It exposes the bundle's task methods (`predict` and/or `transform`, per the recorded task mapping) with standard sklearn/skllm types on both sides: inputs accept the usual list/ndarray/Series forms; outputs decode from the JSON wire format into what the live object would have returned (label arrays, lists of label lists, float arrays with JSON `null` restored to NaN, lists of dicts), driven by the recorded entry/exit kinds. Credentials keep identical semantics everywhere: on each call the facade resolves the host process's config layers — credentials (`config_context`, aliases, env, honoring `allow_env_fallback`) and alias-targeted `overrides` — and forwards the results as dynamic attributes, filtered to the attributes and aliases the bundle declares (a host scope may hold entries for estimators outside this bundle; the facade ignores those, exactly as the estimator-agnostic in-memory layer would) — `config_context` works the same around a live estimator and a loaded bundle. The handler is selectable: the default in-process `LocalHandler` requires a compatible scikit-llm in the host environment; `StdIOHandler` runs the bundle in its own pinned environment built from `env.json`, giving exact local reproduction with no host version matching. `InferenceModel.unwrap()` restores and returns the raw fitted graph for in-Python reuse (inspection, refitting, recomposition) — cross-version best-effort, with the versioned state dict as the compatibility surface. Running or unwrapping a bundle executes its packaged code and unpickles its state (in-process for `LocalHandler`/`unwrap`, inside the isolated environment for `StdIOHandler`) — load only bundles you trust.

**Runtime execution.** The same bundle runs under `fnnx.Runtime`: the wrapper's `warmup` reconstructs the estimator inside the serving environment via the same restore path, and `compute` maps the JSON input channel to `predict`/`transform` (per-task mapping recorded in `extra_values`), resolving credentials from `dynamic_attributes` first, then context/env of the serving process.

**Nesting** comes for free from the whole-graph envelope: `ConsensusClassifier` members, `DynamicFewShotClassifier`'s vectorizer, and arbitrary sklearn wrappers around skllm estimators all ride in the single joblib payload, with every discovered provider's key variables declared in the manifest.

## Dependencies and tooling

- Python ≥ 3.10 (matches FNNX). Core deps: `scikit-learn>=1.6`, `numpy`, `openai`, `fnnx[extras]`, `tqdm`. `pandas` and `google-cloud-aiplatform` are dropped from core.
- Extras: `anthropic`, `google` (google-genai), `litellm`, `all`. The `gguf` and `annoy` extras are removed.
- Version 2.0.0; `uv` for development; ruff + mypy must pass; pytest with no-network tests.
- CI: a GitHub Actions workflow running pytest, ruff, and mypy on pushes/PRs (v1 has no test CI at all).
- Migration: a `MIGRATION.md` mapping every v1 public name to its v2 replacement (or documenting removal: tunables, Vertex/PaLM, GGUF, CoT, `SKLLMConfig`), plus a README rewritten for the v2 API.

## Testing strategy

- A fake in-memory backend registered through the public backend registry (deterministic, scriptable responses, records received requests) is the primary test double — tests exercise real estimator/prompt/policy code paths without network or SDK mocks.
- Adapter tests mock at the provider-SDK boundary per backend.
- sklearn-compliance tests: `get_params`/`clone` round-trips, pipeline and grid-search usage with the fake backend, `NotFittedError` behavior, tag correctness.
- Serialization tests: save→load round-trips for every estimator and for a mixed sklearn pipeline, byte-level assertion that key material never appears in bundles, and end-to-end `fnnx.Runtime` execution of a saved bundle against the fake backend.

# Scenarios

## Scenario: Zero-shot classification happy path
**Given** a `ZeroShotClassifier(model="openai::gpt-5")` and an OpenAI key available via `OPENAI_API_KEY`
**When** `fit(None, ["positive", "negative"])` then `predict(["I love it"])` is called
**Then** `classes_` is `["negative", "positive"]` (sorted, per sklearn convention), one structured-output chat request is issued, and predict returns an array containing one of the two class labels.

## Scenario: Bare model string defaults to OpenAI
**Given** a `ZeroShotClassifier(model="gpt-5")`
**When** any LLM call is made
**Then** it is routed to the OpenAI backend exactly as `openai::gpt-5` would be.

## Scenario: Default model
**Given** a `ZeroShotClassifier()` constructed without a `model` argument
**When** `predict` runs
**Then** requests go to the OpenAI backend for `gpt-5.6-sol` carrying skllm's injected defaults (temperature 0, `reasoning_effort: "none"`); constructing with `model_params={"reasoning_effort": "high"}` instead replaces the injected value.

## Scenario: Unknown provider prefix
**Given** a `ZeroShotClassifier(model="cohere::command-r")`
**When** `fit`/`predict` first needs the backend
**Then** an error is raised naming `cohere` as unknown and listing the registered providers.

## Scenario: Multi-label prediction shape
**Given** a fitted `MultiLabelZeroShotClassifier(max_labels=3)` with classes from `[["quality", "price"], ["delivery"]]`
**When** `predict` runs on two texts
**Then** the result is a list of two lists, each containing 0–3 labels drawn only from `classes_`, with no padding entries.

## Scenario: Non-string labels rejected
**Given** a `ZeroShotClassifier` and integer labels `y=[0, 1]`
**When** `fit` is called
**Then** it raises with a message that labels must be strings, to be cast by the caller.

## Scenario: Out-of-set label counts as failure
**Given** a fitted single-label classifier whose backend returns the label `"maybe"` which is not in `classes_`
**When** `predict` runs with default `on_error="fill"` and `default_label=None`
**Then** the sample's prediction is `None`, a warning reports one failed sample, and the per-sample error report identifies the index and reason.

## Scenario: Error policy raise
**Given** a classifier constructed with `on_error="raise"` and a backend that keeps returning HTTP 500 for one sample
**When** `predict` runs
**Then** after retries are exhausted the batch aborts with the underlying provider error.

## Scenario: Fill policy on API failure
**Given** a `Summarizer` (default `on_error="fill"`) and a backend where exactly one of three samples permanently fails
**When** `transform` runs
**Then** the two healthy samples return summaries, the failed sample returns `""`, a warning is emitted, and usage counters record the failure.

## Scenario: Translation
**Given** a `Translator(output_language="German")`
**When** `transform(["Good morning"])` runs
**Then** a single request carrying the output language is issued and the result is a one-element array with the German translation.

## Scenario: Aggregated prediction with n_predictors
**Given** a fitted `ZeroShotClassifier(n_predictors=5)` over classes A/B whose backend returns A three times and B twice for a sample
**When** `predict` and `predict_proba` run on that sample
**Then** each call issues five sampled requests without skllm's temperature-0 default, `predict` returns A (the majority), and the `predict_proba` row is `[0.6, 0.4]` aligned with `classes_`.

## Scenario: Degenerate predict_proba without n_predictors
**Given** a fitted `ZeroShotClassifier` with `n_predictors` unset whose single deterministic call predicts A
**When** `predict_proba` runs on that sample
**Then** exactly one request is issued and the row is `[1.0, 0.0]` — one-hot on A.

## Scenario: n_predictors with a pinned temperature of zero
**Given** a `ZeroShotClassifier(n_predictors=5, model_params={"temperature": 0})`
**When** params are validated at `fit`/first use
**Then** it raises immediately: sampled voting cannot produce vote diversity at temperature 0.

## Scenario: Few-shot examples reach the prompt
**Given** a `FewShotClassifier` fitted on three labeled examples
**When** `predict` runs
**Then** the outgoing prompt contains all three training examples with their labels.

## Scenario: Dynamic few-shot retrieval with a small class
**Given** a `DynamicFewShotClassifier(n_examples=3)` fitted on a class with five examples and another class with one example
**When** `predict` runs on a new text
**Then** the prompt contains the 3 nearest examples of the large class and the 1 available example of the small class, without error.

## Scenario: Dynamic few-shot with a non-embedding provider
**Given** a `DynamicFewShotClassifier(model="anthropic::claude-sonnet-5")` and no explicit `vectorizer`
**When** `fit` is called
**Then** it raises with a message stating Anthropic offers no embeddings and an explicit `vectorizer` is required.

## Scenario: Invalid prompt template
**Given** a `ZeroShotClassifier` whose `prompt_template` omits a required placeholder and adds an unknown one
**When** `fit` is called
**Then** it raises naming the missing and unknown placeholders and listing the task's documented placeholder set.

## Scenario: Consensus voting and tie-break
**Given** a fitted `ConsensusClassifier` with three members where two predict "spam" and one predicts "ham" for sample 1, and for sample 2 member votes split 1/1 with one member's sample failing to `None`
**When** `predict` runs
**Then** sample 1 is "spam"; sample 2 is decided by the earliest-listed voting member; and `predict_proba` rows equal the vote fractions.

## Scenario: Consensus sample with no votes
**Given** a fitted `ConsensusClassifier(default_label=None)` and one sample for which every member's own policy fills `None`
**When** `predict` runs
**Then** that sample's prediction is `None` and it appears in the ensemble's error report; other samples are unaffected.

## Scenario: Consensus fit does not mutate members
**Given** a `ConsensusClassifier` constructed over two unfitted classifiers
**When** `fit` runs
**Then** fitted clones live in `estimators_`, the constructor-supplied members remain unfitted, and `get_params`/`clone` round-trip the ensemble.

## Scenario: NER inline tagging invariant
**Given** an `NER(entities={"PER": "person", "ORG": "organization"})` fitted, and input `"Tim Cook leads Apple."`
**When** `transform` runs and the model returns a correctly tagged rewrite
**Then** the output is `"<PER>Tim Cook</PER> leads <ORG>Apple</ORG>."` and stripping tags reproduces the input exactly.

## Scenario: NER output that corrupts the source text
**Given** the same NER estimator and a model response whose untagged text cannot be reconciled with the input
**When** `transform` runs under `on_error="fill"`
**Then** the sample returns the original untagged text and is recorded in the error report.

## Scenario: Structured extraction with schema validation
**Given** a `StructuredExtractor` with a JSON schema requiring `{name: string, year: integer}`
**When** `transform` runs on two texts, one yielding conforming JSON and one yielding non-conforming JSON even after salvage
**Then** the first result is the parsed dict, the second is `None` with the failure recorded.

## Scenario: Rubric scoring bounds
**Given** a `RubricScorer(scale=(0, 10))` whose backend returns 7 for one text and 15 for another
**When** `predict` runs
**Then** the results are 7.0 and 10.0 (clamped), as floats.

## Scenario: Cluster labeling
**Given** a `ClusterLabeler` fitted on texts with cluster ids `[0, 0, 1, -1]`
**When** `fit` completes and `transform([0, 1, -1])` is called
**Then** exactly two naming calls were made (none for `-1`), and the result is the two generated names followed by `None`.

## Scenario: Credential layering
**Given** `OPENAI_API_KEY=env_key` in the environment, an active `config_context(openai_api_key="ctx_key")`, and a classifier constructed with `api_key="param_key"`
**When** the classifier issues a request
**Then** it authenticates with `param_key`; without the constructor param it would use `ctx_key`; without both, `env_key`.

## Scenario: In-memory alias credential routing
**Given** a classifier with `alias="team_a"`, `OPENAI_API_KEY=env_key` set, and an active `config_context(credentials={"team_a": "sk-a"})`
**When** `predict` runs
**Then** requests authenticate with `sk-a`; with the alias set but no alias credential in context or `SKLLM_CREDENTIAL_TEAM_A`, the call fails naming `team_a` rather than falling back to `env_key`.

## Scenario: Invalid alias
**Given** a classifier constructed with `alias="Team A"`
**When** params are validated at `fit`/first use
**Then** it raises stating aliases must match `[a-z][a-z0-9_]*`.

## Scenario: Env fallback disabled
**Given** `OPENAI_API_KEY` present in the environment and an active `config_context(allow_env_fallback=False)` with no key in the context
**When** `predict` runs
**Then** the call fails with an error stating that the OpenAI credential is missing *and* that the environment fallback is disabled; the same call succeeds when the key is supplied via `config_context` or the estimator param.

## Scenario: Env fallback disabled ambiently
**Given** `SKLLM_ALLOW_ENV_FALLBACK=0` and `OPENAI_API_KEY` both set in the environment, with no other key source
**When** `predict` runs
**Then** the call fails stating the credential is missing and that the environment fallback is disabled.

## Scenario: config_context isolation across threads
**Given** two threads each entering `config_context` with different OpenAI keys
**When** both run `predict` concurrently on their own estimators
**Then** each thread's requests authenticate with its own key, with no cross-talk.

## Scenario: Reasoning model rejects skllm's temperature default
**Given** a `ZeroShotClassifier(model="openai::some-future-reasoning-model")` for a model unknown to the seed list, whose API rejects `temperature` as unsupported, and no user-supplied temperature
**When** `predict` runs over several samples
**Then** the first request is retried once without the skllm-injected temperature default and succeeds, subsequent requests omit it from the start (cached per model), and predictions are unaffected; the same model with an explicit `model_params={"temperature": 0}` instead surfaces the provider's error.

## Scenario: Provider-specific request params pass through
**Given** a `ZeroShotClassifier(model="openai::gpt-5", model_params={"reasoning_effort": "high"})` and another with `model_params={"nonexistent_param": 1}`
**When** each issues a request
**Then** the first request carries the reasoning effort verbatim, and the second surfaces the provider's own error for the unknown parameter (scikit-llm performs no allow-list validation of `model_params`).

## Scenario: Ollama without credentials
**Given** a `ZeroShotClassifier(model="ollama::llama3")` and no API keys anywhere
**When** `predict` runs against a local Ollama server
**Then** requests go to `http://localhost:11434/v1` (or `OLLAMA_HOST`) and no missing-key error is raised.

## Scenario: OpenAI-compatible endpoint via base_url
**Given** a `ZeroShotClassifier(model="openai::my-model", base_url="http://localhost:8000/v1")`
**When** `predict` runs
**Then** requests go to the given base URL with model id `my-model`; supplying the base URL via `config_context(openai_base_url=...)` instead of the constructor routes identically.

## Scenario: LiteLLM passthrough
**Given** a `ZeroShotClassifier(model="litellm::vertex_ai/gemini-2.5-pro")` with the litellm extra installed
**When** `predict` runs
**Then** the id `vertex_ai/gemini-2.5-pro` reaches litellm verbatim, using native structured outputs when litellm reports support for the model and the prompt-embedded schema fallback otherwise.

## Scenario: Missing optional backend dependency
**Given** the `anthropic` package is not installed
**When** a `ZeroShotClassifier(model="anthropic::claude-sonnet-5")` first needs its backend
**Then** an ImportError is raised naming `pip install scikit-llm[anthropic]`.

## Scenario: Custom backend registration
**Given** `register_backend("mycorp", adapter)` called with a conforming adapter
**When** a `ZeroShotClassifier(model="mycorp::internal-1")` predicts
**Then** requests route through the custom adapter, and `mycorp` appears in the providers listed by the unknown-prefix error.

## Scenario: Keys never serialized
**Given** a fitted classifier constructed with `api_key="sk-secret"`
**When** it is saved via `skllm.save` and separately pickled
**Then** the string `sk-secret` appears in neither the `.fnnx` bundle bytes nor the pickle bytes.

## Scenario: Pickle round-trip stays usable
**Given** a fitted `FewShotClassifier` constructed with `api_key="sk-secret"`, pickled and unpickled in a fresh process
**When** the restored estimator's params are inspected and `predict` runs inside `config_context(openai_api_key="sk-new")`
**Then** `api_key` is `None`, `classes_` and the stored demonstration examples are intact, and requests authenticate with `sk-new`.

## Scenario: FNNX round-trip
**Given** a fitted `FewShotClassifier` saved to `model.fnnx`
**When** `skllm.load("model.fnnx")` is called in a fresh process and `predict` runs inside `config_context(openai_api_key="sk-new")`
**Then** the result is an `InferenceModel` whose `predict` returns the same label array (same values and container type) the original estimator would return, authenticating with `sk-new`; and `unwrap()` yields the underlying estimator with identical params (except credentials) and identical fitted state.

## Scenario: Multi-provider composite bundle
**Given** a fitted `ConsensusClassifier` over an OpenAI-backed and an Anthropic-backed classifier, saved to a bundle
**When** the manifest is inspected and the bundle is run under `fnnx.Runtime`
**Then** the manifest declares both providers' env vars and both `openai_api_key` and `anthropic_api_key` dynamic attributes (secret-tagged), and `compute` with both key attributes reaches each provider with its own key.

## Scenario: Targeted generation overrides in a composite
**Given** a `ConsensusClassifier` whose members are aliased `fast` and `careful`, fitted in memory and also saved to a bundle
**When** in memory `predict` runs inside `config_context(overrides={"careful": {"temperature": 0.9}})`, and under `fnnx.Runtime` `compute` is called with `{"overrides": {"careful": {"temperature": 0.9}}}`
**Then** in both cases only the `careful` member's requests carry temperature 0.9 while the `fast` member keeps its stored `model_params`; under `fnnx.Runtime` an override naming an alias unknown to the bundle is rejected with the list of available aliases (the in-memory config layer is estimator-agnostic and performs no such check — a scope may hold overrides for estimators outside the current call).

## Scenario: Same provider, different keys via aliases
**Given** a `ConsensusClassifier` over two OpenAI-backed classifiers constructed with `alias="team_a"` and `alias="team_b"`, saved and served under `fnnx.Runtime`
**When** the manifest is inspected and `compute` is called with `{"credential_team_a": "sk-a", "credential_team_b": "sk-b"}`
**Then** the manifest declares `credential_team_a` and `credential_team_b` as required secret attributes (env fallbacks `SKLLM_CREDENTIAL_TEAM_A`/`SKLLM_CREDENTIAL_TEAM_B`) and no provider-level `openai_api_key` requirement (both members are aliased, so nothing could consume it), each member's requests authenticate with its own key, and omitting one attribute (with no env fallback) fails with an error naming that alias — never silently reusing the provider-level key.

## Scenario: Discovery includes fit-created components
**Given** a `DynamicFewShotClassifier(model="openai::gpt-5")` fitted without an explicit vectorizer (so a default OpenAI vectorizer was created during fit)
**When** it is saved
**Then** the manifest declares the OpenAI credentials exactly once — the fit-created vectorizer is discovered via the estimator's introspection hook and deduplicated against the classifier's own provider.

## Scenario: Typed IO schemas in the manifest
**Given** three saved bundles — a fitted `ZeroShotClassifier` with `classes_ = ["positive", "negative"]`, a `StructuredExtractor` built from a pydantic model, and a `Pipeline` of a skllm `Vectorizer` followed by a fitted `LogisticRegression`
**When** each bundle's `dtypes.json` and manifest are inspected
**Then** the classifier bundle declares its input as an array of strings and its output as an array over the enum `["positive", "negative"]` plus null; the extractor bundle's output dtype is the compiled JSON Schema of the pydantic model (nullable items), loadable in a serving environment without pydantic installed; and the pipeline bundle's output is a nullable enum over the `LogisticRegression`'s `classes_` — derived by the same edge-typing rule.

## Scenario: Pipeline round-trip
**Given** a fitted `Pipeline([("embed", Vectorizer(model="openai::text-embedding-3-small")), ("clf", LogisticRegression())])`
**When** it is saved with `skllm.save` and loaded in a fresh process with a key injected
**Then** the manifest declares `OPENAI_API_KEY` (secret-tagged) and `env.json` pins scikit-llm (no provider extra — openai is a core dependency), the loaded `InferenceModel.predict` returns the same predictions as the original pipeline (embedding calls for new texts, local classifier inference), and the bundle bytes contain no key material.

## Scenario: FNNX Runtime execution
**Given** the same saved bundle and provider credentials present in the serving environment's env vars
**When** `fnnx.Runtime(bundle).compute({"X": ["some text"]}, {})` is called
**Then** the declared `predictions` output channel contains the same prediction the live estimator would produce.

## Scenario: Dynamic attribute overrides at compute time
**Given** a bundle saved from an estimator with temperature 0 in its `model_params`
**When** `compute` is called once with no dynamic attributes and once with a temperature override and a per-call `openai_api_key`
**Then** the first call issues requests with temperature 0 and env-resolved credentials (saved behavior), and the second call's requests carry the overridden temperature and key — without mutating the bundle's defaults for subsequent calls.

## Scenario: Unknown dynamic attribute rejected
**Given** a loaded bundle under `fnnx.Runtime`
**When** `compute` is called with `{"temperatura": 0.5}`
**Then** the wrapper raises an error naming `temperatura` as undeclared, listing the declared attributes.

## Scenario: Missing required secret at compute time
**Given** a bundle whose model needs an OpenAI key, a serving process with no `OPENAI_API_KEY`, and no key dynamic attribute
**When** `compute` is called
**Then** warmup has already succeeded (no credentials needed at load), and `compute` fails with an error naming both the `openai_api_key` dynamic attribute and the `OPENAI_API_KEY` env var as the ways to provide it.

## Scenario: String-typed dynamic attributes from serving platforms
**Given** a bundle under `fnnx.Runtime` saved from an estimator whose stored `model_params` include `temperature` (so the flat attribute is declared)
**When** `compute` is called with `{"temperature": "0.3"}` (string, as luml's generated OpenAPI encodes attributes)
**Then** the underlying requests use temperature 0.3, identically to passing the float.

## Scenario: compute_async does not block the event loop
**Given** a loaded bundle whose backend simulates slow provider responses
**When** two `compute_async` calls are issued concurrently on one running event loop
**Then** their provider requests overlap in time (the second call does not wait for the first to finish).

## Scenario: Empty input batch
**Given** any fitted estimator
**When** `predict`/`transform` runs on an empty list
**Then** the result is empty, no requests are issued, cumulative usage is unchanged, and the last-operation report shows zero requests.

## Scenario: sklearn compliance
**Given** any v2 estimator instance
**When** `clone(estimator)` and `get_params`/`set_params` round-trips run, and it is placed in a `Pipeline` behind a fake backend
**Then** params (including `api_key`) survive clone verbatim, the pipeline fits and runs its terminal method (`predict` or `transform`), and calling `predict` on an unfitted classifier raises `NotFittedError`.

## Scenario: Usage report and OTel spans
**Given** a classifier run over 10 samples behind a fake backend reporting token counts, with an OTel tracer provider configured and a pricing map in `config_context`
**When** `predict` completes
**Then** the usage report shows 10 requests with summed input/output tokens and a cost estimate, and 10 spans with `gen_ai.request.model` and token attributes were exported; with no tracer provider configured, no span machinery is touched.

# Tasks

- [x] Scaffold v2 package layout and tooling
  - [x] Delete v1 internals: `skllm/llm/`, `skllm/models/`, `skllm/memory/`, `skllm/config.py`, `skllm/model_constants.py`, `skllm/classification.py`, `skllm/text2text.py`, `skllm/vectorization.py`, old `skllm/utils/` content, stale `build/` and `scikit_llm.egg-info/` (leave the untracked `tmp.*` scratch files to the user — one contains a credential to rotate)
  - [x] Rewrite `pyproject.toml`: version 2.0.0 (dynamic from `skllm/__init__.py`), `requires-python>=3.10`, core deps (`scikit-learn>=1.6`, `numpy`, `openai`, `fnnx[extras]`, `tqdm`), extras (`anthropic`, `google`, `litellm`, `all`), drop `pandas`/`google-cloud-aiplatform`/`gguf`/`annoy`, modernize ruff/mypy config, add pytest to dev deps
  - [x] Create the new module tree (`backends/`, `core/`, task modules, `prompts/`, `serialization/`) with minimal placeholders, the `skllm/__init__.py` re-export surface (grown as tasks land), and a passing empty test suite
  - [x] GitHub Actions workflow running pytest, ruff, and mypy on push/PR — live from the first commit
- [x] Add layered config and credential resolution
  - [x] Implement `skllm/config.py`: contextvars store, `config_context` manager, per-provider keys/base URLs, named credentials map for aliases (with `SKLLM_CREDENTIAL_<ALIAS>` env fallback), alias-targeted `overrides` map, `allow_env_fallback` switch (context option + `SKLLM_ALLOW_ENV_FALLBACK`), per-provider client-options map, retry/timeout settings, pricing map, env-var fallback with standard provider names
  - [x] Tests: layering order, alias strictness (set-but-unresolved errors), override precedence (beats stored `model_params`), disabled-env-fallback errors, nesting, thread and asyncio isolation
- [x] Add backend layer with model-string dispatch
  - [x] Backend protocol (chat / structured / embeddings, each returning content + usage) and registry with public `register_backend`; model-string parser with bare-string→openai default and unknown-prefix error
  - [x] OpenAI adapter (chat, JSON-schema structured outputs, embeddings) and the Ollama profile over it (default base URL, keyless); advisory-default injection (temperature 0, `reasoning_effort: "none"` where accepted) with unsupported-param strip-and-retry-once, per-model caching, seed list for known reasoning families, user params never touched; retry with backoff
  - [x] Fake in-memory test backend registered via the registry, used as the standard test double
  - [x] Tests: parser, registry, adapter behavior against a mocked OpenAI SDK, retry/backoff, usage capture
- [x] Add anthropic, google, and litellm backends
  - [x] Anthropic adapter (tool-use structured outputs, no embeddings), Google adapter via `google-genai` (chat, response-schema, embeddings), LiteLLM passthrough adapter
  - [x] Lazy imports with `pip install scikit-llm[<extra>]` error messages
  - [x] Tests per adapter against mocked SDKs, including missing-dependency errors
- [ ] Add core estimator infrastructure
  - [ ] Base estimator classes in `skllm/core/`: verbatim param storage, `__sklearn_tags__`, fitted-state conventions, stateful/stateless split (`NotFittedError` only for stateful estimators), default model strings (`openai::gpt-5.6-sol` chat, `openai::text-embedding-3-small` embeddings), alias charset validation, `prompt_template` placeholder validation, text input handling (list/ndarray/Series), empty-batch short-circuit
  - [ ] Concurrent request engine with `max_concurrency`-bounded parallelism and `verbose`-controlled progress; `on_error` fill/raise policy with warning + per-sample error report attribute
  - [ ] Usage aggregation (per-call and cumulative, optional cost via pricing map) and best-effort OTel `gen_ai` span emission behind an optional import
  - [ ] `alias` param and requirement-introspection hook on the base classes (self-reporting provider + alias pairs, recursing through skllm-native nesting incl. fit-created components); alias-targeted override resolution in the request path (override > stored `model_params` > defaults)
  - [ ] `__getstate__` credential stripping
  - [ ] Tests: sklearn compliance (clone/get_params/pipeline), error policies, stateless transform-without-fit, usage math, span emission on/off, key stripping
- [ ] Add zero-shot and few-shot classifiers
  - [ ] `ZeroShotClassifier`, `FewShotClassifier`, `MultiLabelZeroShotClassifier`, `MultiLabelFewShotClassifier` in `skllm/classification.py` with structured-output label constraints, string-label validation, `default_label`/`max_labels` (schema constraint plus truncation backstop), prompt templates in `skllm/prompts/`, `prompt_template` override validation
  - [ ] `n_predictors` sampled voting: majority-vote `predict`, vote-fraction `predict_proba`, one-hot proba when unset, determinism-default suppression for vote calls, temperature-0 conflict error, per-vote failure tolerance (single- and multi-label semantics)
  - [ ] Tests: happy paths, label-out-of-set failure, non-string y rejection, multi-label shapes, proba fractions and one-hot degeneracy, temperature-conflict error, prompt content
- [ ] Add vectorizer and dynamic few-shot classifier
  - [ ] `Vectorizer` in `skllm/vectorization.py` (batching, 2-D float output, all-or-nothing error behavior)
  - [ ] `DynamicFewShotClassifier` with per-class sklearn `NearestNeighbors` retrieval, small-class handling, default vectorizer models (`openai::text-embedding-3-small`, `google::gemini-embedding-001`), explicit-vectorizer requirement for all other providers
  - [ ] Tests: embedding batching, retrieval composition, default-vectorizer selection, anthropic-without-vectorizer error
- [ ] Add summarizer and translator
  - [ ] `Summarizer` (`max_words`, `focus`) and `Translator` (`output_language`) in `skllm/text2text.py` as stateless transformers without `predict` alias
  - [ ] Tests: outputs, fill-on-failure with empty string, param handling
- [ ] Add NER with inline-tagged output
  - [ ] `NER` in `skllm/tagging.py`: entity schema, tagged-rewrite prompting, strip-tags-equals-input validation with whitespace-tolerant repair, failure → untagged original
  - [ ] HTML rendering utility for tagged text
  - [ ] Tests: tagging invariant, corrupted-output handling, rendering
- [ ] Add structured extractor
  - [ ] `StructuredExtractor` in `skllm/extraction.py`: JSON Schema dict or pydantic class input, pydantic→dict schema compilation at the pickling boundary, native structured outputs with salvage fallback, per-sample schema validation
  - [ ] Tests: both schema input forms, conforming/non-conforming outputs, pickle round-trip schema normalization
- [ ] Add rubric scorer and cluster labeler
  - [ ] `RubricScorer` in `skllm/scoring.py` (scale clamping, NaN fill) and `ClusterLabeler` in `skllm/clustering.py` (`random_state`-controlled exemplar sampling, per-cluster naming calls, `-1`→None, unseen-id error, `fit_transform` override)
  - [ ] Tests: clamping, NaN policy, naming-call counts, transform mapping, `fit_transform` semantics
- [ ] Add consensus ensemble classifier
  - [ ] `ConsensusClassifier` in `skllm/ensemble.py`: member cloning into `estimators_` at fit, concurrent voting, weights, deterministic tie-break, weighted vote-fraction `predict_proba`, `default_label` fill for no-vote samples
  - [ ] Tests: voting, ties, weighted votes, member-failure behavior, all-members-fail fill, constructor members left unmutated
- [ ] Add FNNX bundle saving and runtime execution
  - [ ] Stable `__getstate__`/`__setstate__` contract on all estimators: credential stripping plus a versioned state dict (not raw `__dict__`)
  - [ ] Static self-contained PyFunc wrapper module in `skllm/serialization/`; `skllm.save` via `PyfuncBuilder`: whole-graph joblib payload, two-level provider discovery (generic sklearn graph walk + the base-class introspection hook), `extra_values` identity metadata, per-bundle JSON Schema generation for IO dtypes via edge-typing (texts/ids input; task- and fitted-state-specific outputs incl. `classes_` enums and compiled extraction schemas; convention-derived schemas for foreign exit steps with permissive fallback; entry/exit kinds recorded in `extra_values`), manifest env_vars + dynamic_attributes with the secret/generation-param tag conventions, producer name `skllm.fnnx.ai` with `skllm.fnnx.ai::<task>:v1` tags, `env.json` pinning scikit-llm and discovered provider extras; sklearn composite (Pipeline) support; estimator `.save(path)` method delegating to `skllm.save`
  - [ ] Wrapper compute contract: per-task compute mapping (`predict`/`transform`), defaults from `extra_values`, per-call credential and `<provider>_base_url` attributes, the `overrides` attribute (params object for single-estimator bundles, alias-keyed for composites, unknown aliases rejected) plus flat sugar attributes for stored `model_params` keys, applied via `config_context` around the call, unknown-attribute rejection, string-to-numeric coercion, required-secret errors naming both the attribute and the env var, genuinely non-blocking `compute_async`
  - [ ] Tests: no-key-bytes assertion, manifest/dtypes content per task, end-to-end `fnnx.Runtime` execution with the fake backend, dynamic-attribute override and rejection cases
- [ ] Add bundle loading and the InferenceModel facade
  - [ ] `InferenceModel` returned by `skllm.load`: task-method surface from the recorded mapping, IO encode/decode to sklearn/skllm types from the recorded entry/exit kinds, host-config→dynamic-attributes credential and override forwarding, handler selection (`LocalHandler` default, `StdIOHandler` isolated), `unwrap()`
  - [ ] Tests: facade round-trip equivalence for every estimator and a Vectorizer→LogisticRegression pipeline, credential forwarding through `config_context`, `unwrap()` state fidelity
- [ ] Add migration guide and v2 README
  - [ ] `MIGRATION.md` mapping all v1 public names to v2 equivalents or documenting removals (tunables, Vertex/PaLM, GGUF, CoT, `SKLLMConfig`)
  - [ ] Rewrite `README.md` for the v2 API: install + extras, per-task quickstarts, credentials/`config_context`, serialization and serving, provider matrix
