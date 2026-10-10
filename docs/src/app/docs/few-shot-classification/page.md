---
title: Few-shot classification
---

A few-shot classifier is a zero-shot classifier that demonstrates the task before asking about it. `fit` stores the labeled texts it is given, and every one of those pairs is rendered into the prompt of every prediction. `FewShotClassifier` assigns one label per text; `MultiLabelFewShotClassifier` assigns every label that applies.

```python
from sklearn.model_selection import train_test_split
from skllm import FewShotClassifier
from skllm.datasets import get_classification_dataset

X, y = get_classification_dataset()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=9, stratify=y, random_state=0
)

clf = FewShotClassifier()
clf.fit(X_train, y_train)
labels = clf.predict(X_test)
```

Everything else works as it does without examples. Labels must be strings, answers are constrained to `classes_`, a failed sample fills with `default_label` (or with an empty list in the multi-label case), `predict_proba` returns vote fractions, and `n_predictors` turns one answer per sample into k sampled votes. [Zero-shot classification](/docs/zero-shot-classification) documents all of that.

## What fit keeps

`fit` pairs `X` and `y` element by element, so both must be the same length; a mismatch raises `ValueError` rather than truncating. The pairs are stored verbatim and in order as `examples_`:

```python
clf.examples_
# one (text, label) tuple per training sample, in fit order
```

`classes_` is derived from the labels present in `y`. Unlike the zero-shot case, a label cannot be declared without also supplying at least one example of it: a class that no training pair carries is not part of the schema and can never be predicted. Check `clf.classes_` after fitting on a subset, since a small or unstratified split can silently drop a class.

## Cost and context

All of `examples_` is rendered into the prompt of every sample. The input tokens of one prediction are the fixed template plus the entire demonstration set, so a pair added at `fit` is not paid for once. It is paid for on every text that is later classified. Nine examples in front of a thousand samples means nine thousand rendered examples.

Keep the demonstration set small. The v1 documentation recommended at most ten examples per class, and that remains a reasonable ceiling: beyond it prompts grow linearly while additional near-duplicate examples buy progressively less. Where a larger labeled set exists, hold most of it out and fit on a slice, or use [Dynamic few-shot classification](/docs/dynamic-few-shot-classification), which keeps the whole set and retrieves only the nearest examples per class for each sample.

`usage_.last.input_tokens` reports what a batch actually spent, which is the direct way to see the prompt growth. [Batch execution and error handling](/docs/batch-execution-and-error-handling) covers the report.

## Example order

Models do not weigh the examples in a prompt evenly. The ones nearest the end tend to influence the answer more than the ones at the start. `examples_` preserves the order it was fitted in and reuses that order for every prompt, so a training set sorted by label puts the same class last for every single sample. The demo dataset is sorted that way, and so is most data that came out of a database grouped by category. Shuffle before fitting, and balance the classes while you are at it. `train_test_split` shuffles by default, and `random_state` keeps both the split and the resulting order reproducible.

{% callout title="Where this comes from" %}
Recency effects in the example list were reported against v1 in [this issue](https://github.com/iryna-kondr/scikit-llm/issues/104). The behavior belongs to the models rather than to the library, so it applies to any prompt built from a fixed list of demonstrations.
{% /callout %}

## Choosing between static and dynamic few-shot

`FewShotClassifier` shows every sample the same evidence, so the demonstration half of the prompt is identical across a batch and one inspection describes them all. Its prompt size is fixed by the training set, so the training set has to be small.

`DynamicFewShotClassifier` embeds the training texts at `fit` and retrieves the `n_examples` nearest ones per class for each sample at predict time. Prompt size is then bounded by `n_examples * len(classes_)` however large the labeled set grows, and each text is demonstrated with examples that resemble it. The cost is an embedding model: the training texts are embedded at `fit`, every batch is embedded before it is classified, and not all providers offer embeddings. Prefer it when the labeled set outgrows a prompt, when there are many classes, or when the examples that matter differ from text to text.

## Multiple labels per text

`MultiLabelFewShotClassifier` takes label collections as `y` and demonstrates each training text together with its full label set:

```python
from skllm import MultiLabelFewShotClassifier
from skllm.datasets import get_multilabel_classification_dataset

X, y = get_multilabel_classification_dataset()

clf = MultiLabelFewShotClassifier(max_labels=2)
clf.fit(X, y)
labels = clf.predict(["Arrived two days late, but the packaging was flawless."])
```

Output is a ragged list of lists, `max_labels` is stated in the prompt and enforced as a schema constraint or by truncation depending on the provider, and a failed sample is an empty list. [Zero-shot classification](/docs/zero-shot-classification) describes those in full, including how to score ragged predictions with `MultiLabelBinarizer`.

## Saving a fitted classifier

`examples_` is part of the fitted state, so the training texts and their labels travel inside any bundle written by `save`. Credentials never do. Where the labeled data is confidential, that is the thing to check before shipping the file — see [Saving and loading](/docs/saving-and-loading).

## Custom prompts

`FewShotClassifier` renders `{labels}`, `{text}` and `{examples}`; `MultiLabelFewShotClassifier` renders those three plus `{max_labels}`. A `prompt_template` must use exactly the placeholders its class declares, checked at `fit`. The `{examples}` block is rendered from `examples_` by the library, so a custom template controls the framing around the demonstrations rather than their format. [Customizing prompts](/docs/customizing-prompts) has the details.
