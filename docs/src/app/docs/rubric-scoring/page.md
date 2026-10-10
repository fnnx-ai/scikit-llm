---
title: Rubric scoring
---

`RubricScorer` judges each text against a rubric written in plain language. The rubric is the whole task definition: the model reads it, reads one sample, and answers with the score that sample earns. The estimator is a scikit-learn regressor, so the judgment comes back from `predict` as a float array with one entry per input text.

```python
from skllm import RubricScorer

replies = [
    "Sorry about that. I refunded order 41205 in full; the money is back on your "
    "card within three to five business days.",
    "Please consult the FAQ.",
    "Thanks for flagging this. The parcel was sent to your old address, so I "
    "reshipped it this morning and waived the shipping fee.",
]

scorer = RubricScorer(
    rubric=(
        "How far the reply resolves the customer's problem.\n"
        "1: no answer, or the customer is sent elsewhere.\n"
        "3: the problem is acknowledged and a next step is named.\n"
        "5: the problem is settled, and what happens next is stated concretely."
    ),
    scale=(1, 5),
)

scores = scorer.predict(replies)  # array([5., 1., 5.])
```

The scorer keeps no fitted state. `fit` only validates parameters, `predict` works without one, and a scorer can be saved and served as it is (see [Saving and loading](/docs/saving-and-loading)). Which model answers, and where its credentials come from, is decided the same way as for every other estimator — see [Models and providers](/docs/models-and-providers) and [Credentials and configuration](/docs/credentials-and-configuration).

## The rubric

`rubric` is free text, and it is required: a missing or blank one raises `ValueError` before any request is issued. scikit-llm imposes no structure on it. What the prompt adds around it is the scale, plus the instruction to judge by the rubric alone and use the whole range. A rubric that says what each level requires therefore leaves the judge less room to invent a scale of its own.

A rubric that states one criterion produces a score that can be checked against a human rating of that same criterion. A rubric that bundles several produces a score whose disagreements cannot be attributed to any of them. Two criteria are two scorers and two columns.

The prompt is a template like every other task's, with the placeholders `{rubric}`, `{text}`, `{minimum}` and `{maximum}`. Replacing it is described in [Customizing prompts](/docs/customizing-prompts).

## The scale and clamping

`scale` is a `(minimum, maximum)` pair, `(0, 10)` by default. Both ends must be finite numbers and the minimum must lie below the maximum. Anything else raises `TypeError` or `ValueError` at the same point the rubric is checked. Scores are ordinary floats, so a judge answering `7.5` on a `(0, 10)` scale returns `7.5`.

Both bounds are stated in the prompt, and they also bound the result: a score outside them is clamped into the range rather than rejected, so an answer of 15 on a scale of 0 to 10 comes back as 10. Clamping is what enforces the scale, because a numeric bound is not part of every provider's structured-output mode and cannot be relied on as a schema constraint.

## Samples the judge could not score

The answer is requested through structured output, constrained to the shape `{"score": <number>}`. An answer holding anything else — a string, a null, a boolean, a different field — is a sample failure rather than a score. Under the default `on_error="fill"` that sample scores `NaN`, one warning is emitted per batch stating how many samples were filled, and each failure's index and reason land in `errors_`. Under `on_error="raise"` the first failure aborts the batch with the provider's own error. [Batch execution and error handling](/docs/batch-execution-and-error-handling) covers both policies, `errors_` and `usage_`.

`NaN` travels through numpy without complaint, which is what makes it a safe fill value, but scikit-learn's metrics reject it. Mask the failures out before comparing scores with anything:

```python
import numpy as np

scores = scorer.predict(replies)
usable = ~np.isnan(scores)
scores[usable].mean()
```

## Repeatability

A rubric score is a model's answer, not a measurement, and every scikit-llm estimator carries scikit-learn's `non_deterministic` tag for that reason. Scores are also comparable only within one configuration: rewording the rubric, moving the scale or switching models shifts the whole range, so scores collected under different settings should not be pooled or held against a threshold picked under earlier ones.

The classifiers turn several sampled answers into a distribution through `n_predictors`. `RubricScorer` has no such parameter, since there is no majority to take over a continuous answer. To find out how much a judge moves, score the same batch several times with a sampling temperature and aggregate the runs, at that multiple of the cost:

```python
import numpy as np

scorer.set_params(model_params={"temperature": 0.7})
runs = np.vstack([scorer.predict(replies) for _ in range(5)])

runs.mean(axis=0)  # the score to use
runs.std(axis=0)   # how much the judge disagreed with itself
```

The temperature matters here. Left alone, the providers are asked for temperature 0 wherever they accept it, so the repeats mostly reproduce each other and a spread of zero would describe the decoding rather than the rubric. A wide spread on the deliberately sampled runs points at a rubric the model reads as ambiguous, which is a problem to fix in the rubric text rather than to average away.

{% callout title="A caveat on the spread" %}
Even at temperature 0 a provider does not guarantee that the same request returns the same answer twice, so a narrow spread is evidence of agreement and not a promise of it. Neither reading says anything about whether the scores are correct — only about whether the judge is consistent with itself.
{% /callout %}

## Checking the judge against human scores

Being a regressor, `RubricScorer` inherits `score(X, y)` from scikit-learn's `RegressorMixin`, which returns the coefficient of determination of its scores against reference ones. On a sample somebody has already rated by hand, that is a way to find out whether a rubric means to the model what it means to its author before the scorer is turned loose on unrated text.

```python
human = [5.0, 1.0, 4.0]

scorer.score(replies, human)
```

`score` calls `predict` internally, so it issues requests and pays for them, and it raises if any sample failed and left a `NaN` behind.
