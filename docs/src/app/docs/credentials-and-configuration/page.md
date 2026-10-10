---
title: Credentials and configuration
---

Connection settings — credentials, endpoints, transport — are resolved once per request, never cached on the estimator. Each setting is looked up in four layers, most local first: the estimator parameter, then the active `config_context`, then the environment variable, then the built-in default.

The environment layer is read-only. scikit-llm never writes to `os.environ`, so nothing it resolves leaks into other libraries that read the same variables.

```python
import skllm
from skllm import ZeroShotClassifier
from skllm.datasets import get_classification_dataset

X, y = get_classification_dataset()

# the environment: OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, ...
clf = ZeroShotClassifier().fit(None, y)
clf.predict(X)

# a scope around the call
with skllm.config_context(openai_api_key="sk-...", timeout=30.0):
    clf.predict(X)

# the estimator itself, which outranks both
ZeroShotClassifier(api_key="sk-...").fit(None, y).predict(X)
```

The credential variable a provider is read from is its own: `OPENAI_API_KEY` for `openai::`, `ANTHROPIC_API_KEY` for `anthropic::`, and `GEMINI_API_KEY` or `GOOGLE_API_KEY` for `google::`, the more specific name winning when both are set. A provider without a declared variable, including any [custom backend](/docs/models-and-providers), falls back to `<PROVIDER>_API_KEY`.

## Scoping settings with config_context

`config_context` is a context manager backed by `contextvars`, so a scope belongs to the thread or async task that opened it. Two threads can hold different keys at the same time, and nothing needs resetting: leaving the `with` block restores what was there before.

```python
with skllm.config_context(openai_api_key="sk-team", timeout=30.0):
    with skllm.config_context(timeout=5.0):
        clf.predict(X)  # key sk-team, timeout 5.0
    clf.predict(X)      # key sk-team, timeout 30.0
```

Scopes nest per setting rather than per scope: an inner context overrides only the settings it names, and everything else keeps resolving through the outer ones. A setting passed as `None` is ignored entirely, and so is a blank string — `config_context(openai_api_key=os.getenv("KEY"))` with the variable absent does not mask the outer layers with an empty credential.

Batches run their requests on a thread pool, and each task runs in a copy of the calling context. A scope opened around `predict` therefore reaches every worker thread that call spawns.

## Recognized settings

| Setting | Value | Meaning |
| --- | --- | --- |
| `<provider>_api_key` | `str` | The key for one provider: `openai_api_key`, `anthropic_api_key`, `google_api_key`, or `<prefix>_api_key` for any registered backend. |
| `<provider>_base_url` | `str` | The endpoint for one provider, ahead of that provider's environment variable. |
| `credentials` | `dict[str, str]` | Keys routed by estimator alias. |
| `overrides` | `dict[str, dict]` | Generation parameters applied to an estimator by alias. |
| `client_options` | `dict[str, dict]` | Extra arguments for a provider's SDK client, keyed by provider. |
| `pricing` | `dict[str, dict]` | Prices per million tokens, keyed by model, as `{"input": ..., "output": ...}`. |
| `timeout` | `float` | Request timeout in seconds. Must be positive; defaults to `60.0`. |
| `max_retries` | `int` | Retries per request on transient failures. Must not be negative; defaults to `3`. |
| `allow_env_fallback` | `bool` | Whether ambient environment credentials may be used at all. |

Any other keyword raises `ValueError` listing the valid names, and a value of the wrong type raises `TypeError`, both when the context is entered rather than when a request is issued.

Scalars and connection strings resolve to the innermost scope that set them. Two of the maps behave differently: `client_options` and `overrides` are merged across every scope on the stack, key by key, with inner scopes winning, so an outer context can carry an organization id while an inner one adds a header. A price is taken from the innermost scope that prices that model, looked up first by the full `provider::model_id` string and then by the bare model id.

```python
with skllm.config_context(
    client_options={"openai": {"organization": "org-..."}},
    pricing={"gpt-5.6-sol": {"input": 1.25, "output": 10.0}},
):
    clf.predict(X)

clf.usage_.last.cost
```

Prices are what turn the token counts in `usage_` into a cost estimate. Without them the cost stays `None`. `timeout` and `max_retries` resolve through the same layers as every other setting here. [Batch execution and error handling](/docs/batch-execution-and-error-handling) covers the usage report itself, which failures are retried, and how the backoff between attempts behaves.

## Routing credentials by alias

An `alias` names an estimator instance. Unlike `api_key`, it is part of the estimator's persisted state, so it survives pickling and saving, and it can be addressed from outside the object — inside a pipeline, an ensemble, or a loaded bundle. Aliases must match `^[a-z][a-z0-9_]*$`.

```python
clf = ZeroShotClassifier(alias="team_a").fit(None, y)

with skllm.config_context(
    credentials={"team_a": "sk-a"},
    overrides={"team_a": {"temperature": 0.7}},
):
    clf.predict(X)
```

A credential given in `credentials` is routed strictly by name, and an aliased estimator never falls back to the provider-level key. A missing alias credential fails naming the alias instead of quietly reusing a key that was separated on purpose. Only the estimator's own `api_key` parameter outranks the alias. The environment fallback for an alias is `SKLLM_CREDENTIAL_` followed by the alias in upper case, so `team_a` reads `SKLLM_CREDENTIAL_TEAM_A`.

`overrides` layers the other way around from credentials: an override sits above the estimator's stored `model_params`, because overrides exist to beat a stored value, whereas credential layers exist to fill in a missing one. Parameters the override does not mention keep their stored values.

## Refusing ambient credentials

`allow_env_fallback=False` closes the environment layer for the scope. Only an explicit parameter, a `config_context` key or an alias credential can then authenticate a request, so a shared service cannot reach a provider with a key it did not mean to use.

```python
with skllm.config_context(allow_env_fallback=False, openai_api_key=key_for_this_request):
    clf.predict(X)
```

Setting `SKLLM_ALLOW_ENV_FALLBACK` to `0`, `false`, `no` or `off` does the same for a whole process. Any other value, or leaving it unset, keeps the fallback on.

## When a credential is missing

`skllm.MissingCredentialError`, a subclass of `RuntimeError`, is raised when every layer comes up empty for a provider whose backend declares a credential variable. Keyless providers — Ollama, LiteLLM, any custom backend with `credential_env_var = None` — never raise it. The message names what would have supplied the key: the estimator parameter, the `config_context` setting and the accepted environment variables, or, when the fallback is off, that those variables are being ignored.

It surfaces as a request is prepared, which means during `fit`, `predict` or `transform` rather than at construction. It is also the one failure that `on_error="fill"` does not absorb: a missing credential hits every sample of a batch identically, so it aborts the batch immediately instead of reporting a column of indistinguishable per-sample failures. [Batch execution and error handling](/docs/batch-execution-and-error-handling) covers the failures that are per-sample.

## Keys are never serialized

`api_key` and `base_url` are ordinary sklearn parameters, so `get_params`, `clone` and grid search treat them like any other. They are stripped on the way out: `__getstate__` writes them as `None`, so a pickled estimator and an `.fnnx` bundle both carry the configuration without the credential.

A bundle instead declares the credentials it needs as required attributes, which the serving process supplies through the same `config_context` and environment variables described here. [Saving and loading](/docs/saving-and-loading) covers what a bundle declares and how a deployment fills it in.
