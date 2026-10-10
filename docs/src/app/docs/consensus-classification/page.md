---
title: Consensus classification
---

`ConsensusClassifier` decides each sample by majority vote across a panel of classifiers. It holds no model of its own: everything that describes one, from the model string to its credentials, is configured on the members. A panel is therefore free to mix providers, models and prompting strategies.

```python
from skllm import ConsensusClassifier, FewShotClassifier, ZeroShotClassifier
from skllm.datasets import get_classification_dataset

X, y = get_classification_dataset()

clf = ConsensusClassifier([
    ZeroShotClassifier(model="openai::gpt-5"),
    ZeroShotClassifier(model="anthropic::claude-sonnet-5"),
    FewShotClassifier(model="google::gemini-2.5-flash"),
])
clf.fit(X, y)
labels = clf.predict(X)
```

`estimators` is the only positional parameter and the only required one. Members must be single-label scikit-llm classifiers, meaning [`ZeroShotClassifier`](/docs/zero-shot-classification), [`FewShotClassifier`](/docs/few-shot-classification) or [`DynamicFewShotClassifier`](/docs/dynamic-few-shot-classification). A multi-label classifier or any other object raises `TypeError`, and an empty panel raises `ValueError`.

`fit` derives `classes_` from `y` as the sorted set of its labels, then fits a clone of every member on the same `X` and `y`, one after another. As in sklearn's `VotingClassifier`, the objects you passed stay untouched and the fitted clones live on `estimators_`. A member that was already fitted is refitted like the rest. Each one fits the way it normally would, so a zero-shot member ignores `X` (a panel of only zero-shot members can be fitted with `fit(None, y)`), a few-shot member keeps the pairs, and a dynamic member embeds them.

## How the vote is counted

`predict` asks every member for the whole batch. Members are queried concurrently, and the ensemble's `max_concurrency` bounds how many members are in flight rather than how many samples: within its own batch each member still applies its own `max_concurrency`.

Every member contributes one vote per sample, worth its weight, and the label with the most weight wins. `predict` returns a `numpy` array of `dtype=object` with one label per sample. When two labels tie, the winner is the one voted for by the earliest member in `estimators` order, so the same panel decides the same way on a rerun.

`predict_proba` returns a float array of shape `(n_samples, len(classes_))`, columns aligned with `classes_`, holding each class's share of the weight that was cast. A row where every member voted sums to one, and a row where nobody voted is all zeros. These are vote fractions rather than calibrated probabilities: three equally weighted members can only ever produce 0, one third, two thirds and 1.

A panel of k members costs k times the requests of a single classifier, and it only pays for that when its members fail in different ways. Repeatedly sampling one model is a different tool: `n_predictors` on a single classifier votes across k sampled answers from the same model without an ensemble around it.

## Weighting members

`weights` takes one positive number per member and defaults to equal weighting. Its length must match `estimators`.

```python
clf = ConsensusClassifier(
    [
        ZeroShotClassifier(model="openai::gpt-5"),
        ZeroShotClassifier(model="anthropic::claude-sonnet-5"),
        ZeroShotClassifier(model="google::gemini-2.5-flash"),
    ],
    weights=[1.0, 2.0, 1.0],
)
```

Weights are summed per label, so the second member above counts as two votes. Only their ratios matter, since `predict_proba` reports each label's share of the weight actually cast.

## Abstention and failure

A member that fails a sample under its own `on_error="fill"` policy fills it with its own `default_label`. When that fill is `None`, which is the default, the ensemble reads it as an abstention: the vote is not counted, and the remaining members decide the sample between them.

{% callout type="warning" title="Leave default_label unset on members" %}
A member whose `default_label` is a string does not abstain. The fill is an ordinary label, so that member votes for it on every sample it failed, and a run of failures can carry it to a majority. Set `default_label` on the ensemble instead, where it applies only when no member voted at all.
{% /callout %}

A sample no member voted for is the ensemble's own failure. Under the default `on_error="fill"` it takes the ensemble's `default_label`, emits a warning, and appears in `errors_` with the reason `no member voted: every one of them filled this sample under its own policy`. Under `on_error="raise"` it aborts the call with a `BackendError` instead. A member configured with `on_error="raise"` overrides all of this: its exception propagates out of the ensemble call, because its own policy says a failed sample is not something to work around. [Batch execution and error handling](/docs/batch-execution-and-error-handling) covers both policies.

The ensemble issues no request of its own, so `usage_.last` is the sum of what the members spent on their last operation and `usage_.total` accumulates that across calls. Each member also keeps its own report.

## Configuring members individually

The ensemble takes no `model`, `model_params`, `prompt_template`, `api_key`, `base_url` or `alias`. Giving a member an `alias` makes its credentials and generation parameters addressable by name from `config_context`:

```python
from skllm import ConsensusClassifier, ZeroShotClassifier, config_context

clf = ConsensusClassifier([
    ZeroShotClassifier(model="openai::gpt-5", alias="fast"),
    ZeroShotClassifier(model="anthropic::claude-sonnet-5", alias="careful"),
])

with config_context(
    credentials={"fast": "sk-...", "careful": "sk-ant-..."},
    overrides={"careful": {"temperature": 0.2}},
):
    clf.fit(X, y)
    predictions = clf.predict(X)
```

An aliased member is routed strictly by its alias, so every alias in the panel needs an entry in `credentials` or its own environment variable. Members left unaliased resolve their provider's key as usual. [Credentials and configuration](/docs/credentials-and-configuration) covers alias naming and resolution. Aliases also survive into a saved bundle, so a deployed panel can still be tuned member by member.
