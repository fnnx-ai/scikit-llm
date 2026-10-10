---
title: Zero-shot classification
---

A zero-shot classifier assigns each text one of the labels it was fitted with, without seeing a single labeled example. The label set is the whole task description: the model reads the labels, the text, and nothing else. `ZeroShotClassifier` picks exactly one label per text; `MultiLabelZeroShotClassifier` picks every label that applies.

```python
from skllm import ZeroShotClassifier
from skllm.datasets import get_classification_dataset

X, y = get_classification_dataset()

clf = ZeroShotClassifier()
clf.fit(None, y)
labels = clf.predict(X)
```

The default model is `openai::gpt-5.6-sol` and its key is read from `OPENAI_API_KEY`. Other providers are covered in [Models and providers](/docs/models-and-providers), and the other ways to supply a key in [Credentials and configuration](/docs/credentials-and-configuration).

## Fitting on the label set

`fit` ignores `X`, which is why the example passes `None`: no training text reaches the prompt. Only `y` matters, and it is read as a flat sequence of labels whose sorted unique set becomes `classes_`. Handing over a full column of training labels works, and so does naming each label once:

```python
clf = ZeroShotClassifier().fit(None, ["positive", "negative", "neutral"])
clf.classes_
# ['negative', 'neutral', 'positive']
```

Labels must be strings. `fit` raises `TypeError` on anything else instead of letting an integer round-trip through a prompt and back, so cast them first or keep a `LabelEncoder` outside the estimator.

Answers are constrained to `classes_` by the provider's structured-output mode, so no answer outside the fitted set becomes a prediction. A label outside the set is reported as a sample failure rather than passed through. The only value `predict` can return that is not in `classes_` is a `default_label` you chose yourself.

## Label wording

Zero-shot accuracy depends on how the labels are written, because the label list carries the entire definition of the task. Labels have to read as natural language and be self-explanatory on their own. Internal codes and enum-shaped names carry little for the model to work with; the same categories spelled out carry considerably more.

```python
opaque = ["SEM_POS", "SEM_NEG", "SEM_NEU"]

descriptive = [
    "the semantics of the provided text is positive",
    "the semantics of the provided text is negative",
    "the semantics of the provided text is neutral",
]
```

Since `predict` returns the fitted labels verbatim, descriptive labels do not have to leak into the rest of a program. Map them back to your own codes with a dictionary after prediction.

## Failed samples

Under the default `on_error="fill"` policy, a sample whose request failed is filled with `default_label`, a `UserWarning` names how many samples were filled, and the reasons land in `errors_`. `default_label` is `None` unless you set it, so a failed prediction stays visibly distinct from a real one.

```python
clf = ZeroShotClassifier(default_label="neutral").fit(None, y)
labels = clf.predict(X)
clf.errors_
# (SampleError(index=3, error='BackendError: ...'),)
```

The same sample's row in `predict_proba` is all zeros. `on_error="raise"` aborts the batch instead of filling. [Batch execution and error handling](/docs/batch-execution-and-error-handling) covers both policies, the warning, and the `usage_` report.

{% callout title="Changed in v2" %}
v1 defaulted to `default_label="Random"`, which drew a label from the class frequencies and produced a filled sample indistinguishable from a prediction. v2 fills with `None` unless you name a label, and always records the failure.
{% /callout %}

## Vote fractions and n_predictors

`predict_proba` returns an array of shape `(n_samples, len(classes_))` whose columns are aligned with `classes_`. Each row holds the fraction of votes every class received. With the default of one call per sample there is exactly one vote, so a row is one-hot, and a fully failed sample is a row of zeros.

`n_predictors=k` answers each sample k times and aggregates the answers. The k calls are sampled rather than deterministic: scikit-llm drops its own temperature default so that the votes can differ. `predict` then returns the label with the most votes, and `predict_proba` reports the counts as fractions.

```python
clf = ZeroShotClassifier(n_predictors=5).fit(None, y)
labels = clf.predict(X)
proba = clf.predict_proba(X)
```

Ties go to the earliest class in `classes_`, which keeps a repeated run reproducible. A sample fails only when all k of its attempts fail; whatever votes arrived decide the rest. The cost is k times the requests, and `max_concurrency` still bounds how many of them are in flight at once.

Setting an explicit `temperature` of 0 in `model_params` alongside `n_predictors` greater than 1 raises `ValueError` at `fit`: every vote would be identical at k times the price.

{% callout title="predict and predict_proba each issue their own requests" %}
Neither method caches for the other. Calling both on the same batch runs the batch twice and pays for it twice.
{% /callout %}

## Multiple labels per text

`MultiLabelZeroShotClassifier` assigns every applicable label instead of exactly one. Its `y` is a sequence of label collections, and `classes_` is the sorted union across all rows:

```python
from skllm import MultiLabelZeroShotClassifier
from skllm.datasets import get_multilabel_classification_dataset

X, y = get_multilabel_classification_dataset()

clf = MultiLabelZeroShotClassifier(max_labels=3)
clf.fit(None, y)
labels = clf.predict(X)
# [['Quality', 'Packaging'], ['Delivery'], ...]
```

`X` is ignored here too. A single row holding the whole candidate set, `fit(None, [candidate_labels])`, fits the same classifier as the ten labeled rows above.

`predict` returns a plain Python list of lists, one list per text, ragged and never padded to a fixed width. A sample that failed is an empty list. There is no `default_label` on the multi-label classifiers, so an empty prediction and a failed one look alike in the output and differ only in `errors_`.

`max_labels` caps how many labels one text may receive. The cap is stated in the prompt, and it becomes a `maxItems` JSON-Schema constraint on providers whose structured mode accepts one. Where the provider does not, truncation on the way back is the backstop, so the cap holds either way. Left at `None`, a text may receive up to every fitted class.

With `n_predictors` set, a label survives only if more than half the votes included it, and the survivors are ordered by descending vote count before the cap applies. `predict_proba` reports each class's share of the votes independently, so a row does not need to sum to one.

## Metrics on ragged output

Most multi-label metrics want an indicator matrix rather than lists of labels. `sklearn.preprocessing.MultiLabelBinarizer` converts both sides, and passing `classes=clf.classes_` gives it the same column order `predict_proba` uses:

```python
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

binarizer = MultiLabelBinarizer(classes=clf.classes_)
y_true = binarizer.fit_transform(y)
y_pred = binarizer.transform(labels)

f1_score(y_true, y_pred, average="micro")
```

A failed sample's empty list becomes a row of zeros, which scores as a prediction that named nothing. Where that would distort a metric, drop the indices listed in `errors_` before scoring.

## Custom prompts

`prompt_template` replaces the default template. `ZeroShotClassifier` renders `{labels}` and `{text}`; `MultiLabelZeroShotClassifier` renders `{labels}`, `{text}` and `{max_labels}`. A template must use exactly the placeholders its class declares, and the check runs at `fit`. [Customizing prompts](/docs/customizing-prompts) has the details.
