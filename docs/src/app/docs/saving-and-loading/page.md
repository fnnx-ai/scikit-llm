---
title: Saving and loading
---

A fitted estimator is persisted as an FNNX bundle: one archive holding the fitted object graph, a manifest describing what the model takes and returns, the credentials it needs declared as required attributes, and the dependencies to rebuild the environment it runs in. The same file reloads into Python and runs under the FNNX runtime.

```python
import skllm
from skllm import ZeroShotClassifier
from skllm.datasets import get_classification_dataset

X, y = get_classification_dataset()
clf = ZeroShotClassifier().fit(None, y)

clf.save("model.fnnx")
```

`estimator.save(path)` and `skllm.save(estimator, path)` are the same call — the instance method delegates to the function.

## What can be saved

`skllm.save` accepts any fitted sklearn object holding at least one scikit-llm estimator. A bare estimator, a `Pipeline` mixing embeddings with a classical model, a [`ConsensusClassifier`](/docs/consensus-classification): the graph is packaged whole rather than taken apart.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from skllm import Vectorizer

pipeline = Pipeline(
    [
        ("embed", Vectorizer()),
        ("classify", LogisticRegression()),
    ]
).fit(X, y)

skllm.save(pipeline, "pipeline.fnnx")
```

Saving runs sklearn's `check_is_fitted` first, so an unfitted estimator raises `NotFittedError` instead of producing a bundle that cannot answer anything. Estimators with no fitted state — `Summarizer`, `Translator`, `NER`, `StructuredExtractor`, `RubricScorer` and `Vectorizer` — count as fitted and may be saved straight after construction. An object holding no scikit-llm estimator raises `ValueError`, pointing at sklearn-flavored packagers instead.

## What travels in the bundle

The fitted graph is written as a single `joblib` payload, so everything the estimators hold survives: a classifier's `classes_`, a few-shot classifier's demonstration pairs, a dynamic few-shot classifier's fitted vectorizer and per-class neighbor indexes, a `ClusterLabeler`'s cluster names. The estimator's own configuration travels with it — the model string, the prompt template, `model_params`, `alias`, `on_error`, `max_concurrency`.

Around that payload the bundle records its contract. The single input channel is named `X` and the single output channel `predictions`, each typed by a JSON Schema derived from the fitted state: a classifier publishes its labels as an enum, a `StructuredExtractor` publishes the extraction schema, a `Vectorizer` publishes an array of numbers. The bundle also declares the dynamic attributes a call may supply (credentials, base URLs and generation-parameter overrides), an environment variable per declared credential as a fallback, and pinned dependencies — `scikit-llm[<extras>]==<version>` with the extras every provider in the graph needs, plus the FNNX runtime.

{% callout title="Structural identity is fixed at save time" %}
The model string, the prompt, the labels and the schema are part of the payload, not part of the call. A served bundle takes credentials and generation params per call and nothing else. Changing what the model is, or how it is asked, means saving a new bundle.
{% /callout %}

A `StructuredExtractor` given a pydantic model is the one estimator whose state is rewritten on the way out: the class is compiled to its plain JSON Schema, so serving needs neither your class nor pydantic itself. In memory, `get_params()` and `clone()` still return the class.

## What never travels in it

Credentials are excluded by construction rather than filtered on the way out. `__getstate__` sets `api_key` and `base_url` to `None` before the object is handed to `joblib` or `pickle`, so no serialized form of a scikit-llm estimator holds a key:

```python
clf = ZeroShotClassifier(api_key="sk-secret").fit(None, y)
clf.save("model.fnnx")

restored = skllm.load("model.fnnx").unwrap()
restored.api_key  # None
```

This is the behavior v1 lacked. A pickled v1 estimator carried whatever key it had been configured with into the file, which made a saved model a secret to be handled as carefully as the key itself.

What the bundle records instead is which credentials it will need. Each unaliased provider in the graph contributes a `<provider>_api_key` attribute, each alias a `credential_<alias>` attribute, and each declaration names the environment variables a serving process may answer it from. Keyless providers such as Ollama contribute nothing, and an estimator that calls a provider only during `fit` — a fitted `ClusterLabeler`, whose `transform` is a lookup — declares no credential at all. [Serving bundles](/docs/serving-bundles) describes how those attributes are supplied.

{% callout type="warning" title="A bundle is still sensitive" %}
No key travels, but the training data does. A `FewShotClassifier` bundle contains every demonstration pair it was fitted on, and a `DynamicFewShotClassifier` bundle contains those texts and their embeddings.
{% /callout %}

## Loading a bundle

`skllm.load` returns an `InferenceModel` rather than the estimator. A bundle is a deployable artifact, and running it through the FNNX runtime is what a deployment does, so the object you get locally answers through the same path a served copy answers through.

```python
model = skllm.load("model.fnnx")

model.predict(["Absolutely worth the price."])
```

The facade exposes exactly one task method, the one the bundle recorded: `predict` for a classifier or scorer, `transform` for a transformer. Inputs and outputs are the sklearn types the live object used — a list, array or `Series` in, and the object array, list of label lists or list of records out. Asking for the other method says so rather than failing obscurely:

```python
model.transform(["Absolutely worth the price."])
# AttributeError: This bundle answers through 'predict', not 'transform'.
```

Nothing else from the estimator surface is on the facade. `predict_proba`, `errors_` and `usage_` belong to the live object, which `unwrap()` hands back:

```python
estimator = model.unwrap()

estimator.predict_proba(["Absolutely worth the price."])
estimator.usage_.total
```

`unwrap()` restores the fitted graph in the current process for inspection, refitting or recomposition — everything a bundle, which only answers batches, cannot do.

{% callout title="Versioned state" %}
The pickled estimator state carries a version number. State written by a newer scikit-llm than the one reading it raises `ValueError` asking for an upgrade, instead of being restored with the wrong meaning.
{% /callout %}

## Credentials at call time

A loaded bundle resolves connection settings the same way a live estimator does, through the [config layers](/docs/credentials-and-configuration) of the process that loaded it. On every call the facade asks those layers for each credential and base URL the bundle declared, and forwards what resolves as dynamic attributes:

```python
with skllm.config_context(openai_api_key="sk-..."):
    model.predict(["Absolutely worth the price."])
```

An ambient `OPENAI_API_KEY` works identically, as does `config_context(credentials={"team_a": "sk-..."})` for a bundle whose members are aliased. Only what the bundle declared is forwarded: a scope may carry credentials for estimators that have nothing to do with this bundle, and those are passed over.

Because resolution happens per call, the same loaded bundle answers under whichever scope the call is made in, and the key never lands on the estimator.

## Running the bundle in its own environment

The default handler runs the bundle in the current process, which needs a scikit-llm compatible with the one that wrote it. `StdIOHandler` runs it in an environment built from the bundle's pinned dependencies instead, so a bundle from a different scikit-llm version reproduces exactly with no version matching against the host:

```python
from fnnx.handlers.stdio import StdIOHandler

model = skllm.load("model.fnnx", handler=StdIOHandler)
```

Either way `config_context` reads the same around the loaded bundle, because the credentials travel as dynamic attributes rather than as ambient state the subprocess would have to share.

{% callout type="warning" title="Load only bundles you trust" %}
Loading, unwrapping or running a bundle executes its packaged code and unpickles its state.
{% /callout %}
