---
title: Serving bundles
---

An `.fnnx` bundle runs under the FNNX runtime without any scikit-llm client code. The serving process needs scikit-llm installed, and the bundle pins the version and the provider extras it requires. The code that calls it talks to `fnnx.runtime.Runtime` and sees only JSON.

```python
from fnnx.runtime import Runtime

runtime = Runtime("model.fnnx")

runtime.compute(
    {"X": ["Absolutely worth the price."]},
    {"openai_api_key": "sk-..."},
)
# {'predictions': ['positive']}
```

`compute` takes the input channels as its first argument and the dynamic attributes as its second. The bundle declares one channel of each: `X` for the batch and `predictions` for the answer. Both carry plain JSON — a list of strings for a text estimator, a list of integers for a fitted `ClusterLabeler`, and on the way back a list of labels, label lists, numbers, vectors or records. A filled sample serializes as `null`, including the `NaN` a `RubricScorer` fills with. Nothing is converted to numpy. That conversion is what [`skllm.load`](/docs/saving-and-loading) adds on top of this same path.

## Dynamic attributes

Everything a call may vary arrives as a dynamic attribute. Which names a given bundle accepts is fixed at save time and readable from its manifest.

| Attribute | Carries |
| --- | --- |
| `<provider>_api_key` | the key for an unaliased provider in the graph, e.g. `openai_api_key` |
| `credential_<alias>` | the key for one aliased estimator, e.g. `credential_team_a` |
| `<provider>_base_url` | where that provider is reached |
| `overrides` | generation params for this call |

A bundle whose estimators are all aliased declares no provider-level key at all, because an aliased estimator resolves its credential strictly by name. A keyless provider such as Ollama declares only its base URL.

Undeclared names are rejected rather than ignored, which is what makes a typo visible instead of silently ineffective:

```python
runtime.compute({"X": ["..."]}, {"temperatura": "0.3"})
# ValueError: Unknown dynamic attribute(s) ['temperatura'].
# This bundle declares ['openai_api_key', 'openai_base_url', 'overrides'].
```

`overrides` is a JSON object of generation params to apply over the ones stored in the bundle. For a bundle holding a single estimator the object is flat. For a composite one — a pipeline or a `ConsensusClassifier` — it is keyed by estimator alias, so one member can be tuned without touching the others. An alias the bundle does not hold is rejected the same way an undeclared attribute is.

```python
runtime.compute({"X": ["..."]}, {"overrides": {"temperature": 0.2}})

runtime.compute(
    {"X": ["..."]},
    {"overrides": {"careful": {"temperature": 0.9}}},  # composite bundle
)
```

A single-estimator bundle additionally declares one flat attribute per generation param it already stores, so a deployment UI can expose `temperature` as its own field instead of asking for a nested JSON blob. Params the bundle stores nothing for have no flat name and go through `overrides`.

{% callout title="Strings that mean numbers" %}
Attributes usually reach a deployment through an OpenAPI schema that types every one of them as a string, so `"0.3"` is a normal way to send a temperature. A single-estimator bundle coerces such values: `"0.3"` becomes `0.3`, `"128"` becomes `128`. A param the bundle stores as a string is left alone, since its declared type is text — retyping OpenAI's `user` or Ollama's `keep_alive` would break it. The alias-keyed overrides of a composite bundle travel as the JSON they arrive as, because nothing records which of a member's params are strings by intent.
{% /callout %}

The `overrides` object may itself arrive as JSON text, for gateways that cannot send a nested object:

```python
runtime.compute({"X": ["..."]}, {"overrides": '{"temperature": 0.5}'})
```

## Where credentials come from

A dynamic attribute wins if one is present. Otherwise the bundle resolves the credential through the serving process's own [config layers](/docs/credentials-and-configuration), exactly as a live estimator would: an active `config_context`, then the environment variable the declaration names. A process that exports `OPENAI_API_KEY` serves the bundle above with an empty attribute dict.

When nothing resolves, the call fails naming the attribute and the variable that would have supplied it:

```text
MissingCredentialError: No credential for the 'openai_api_key' dynamic attribute
of this bundle. Pass it in the call's dynamic attributes, or provide it to the
serving process as OPENAI_API_KEY.
```

If the serving process set `allow_env_fallback=False`, the message says the environment is being ignored rather than pointing at a variable that cannot be read. Credentials never touch estimator state: the resolved values are applied as a `config_context` around the call and are gone when it returns.

## Concurrency

`compute` is blocking. `compute_async` runs the same work on the handler's worker pool, so a server can await several calls without blocking its event loop:

```python
answer = await runtime.compute_async({"X": ["..."]}, {})
```

The pool belongs to the handler, not to scikit-llm. The default `LocalHandler` is configured by `LocalHandlerConfig`, whose `n_workers` defaults to `1` — so with the default configuration, concurrent `compute_async` calls serialize. Raise it to overlap them:

```python
from fnnx.handlers.local import LocalHandlerConfig

runtime = Runtime("model.fnnx", handler_config=LocalHandlerConfig(n_workers=8))
```

This is a separate knob from the concurrency inside one call. How many requests a single batch has in flight is the estimator's [`max_concurrency`](/docs/batch-execution-and-error-handling), which is part of the saved state and cannot be changed per call. `n_workers` bounds how many batches the runtime works on at once, `max_concurrency` bounds the requests within each of them, so the two multiply.

## Serving over HTTP

Nothing about the runtime is web-framework specific: a bundle is loaded once at startup and called per request.

```python
from fastapi import FastAPI
from fnnx.handlers.local import LocalHandlerConfig
from fnnx.runtime import Runtime
from pydantic import BaseModel

runtime = Runtime("model.fnnx", handler_config=LocalHandlerConfig(n_workers=8))
app = FastAPI()


class Batch(BaseModel):
    texts: list[str]


@app.post("/predict")
async def predict(batch: Batch) -> dict:
    answer = await runtime.compute_async({"X": batch.texts}, {})
    return {"predictions": answer["predictions"]}
```

The empty attribute dict leaves every credential to the process environment, which is where a container usually keeps it. Passing them per request instead — from a header, or from a secret the platform binds — is a matter of building that second argument, and the names to build it from are the ones the bundle declares.

{% callout type="warning" title="Load only bundles you trust" %}
Running a bundle executes its packaged code and unpickles its state.
{% /callout %}
