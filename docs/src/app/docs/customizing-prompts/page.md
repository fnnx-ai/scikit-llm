---
title: Customizing prompts
---

Every estimator that talks to a chat model builds its request from a prompt template. Passing `prompt_template` replaces the one the task ships with. The two estimators without a template are `Vectorizer`, which calls an embedding model rather than a chat model, and `ConsensusClassifier`, which holds no model of its own and prompts nothing — its members carry their own templates.

A template is an ordinary `str.format` template. The rendered result is sent as a single user message; there is no separate system message, so everything the model is told lives in the template.

## The stock templates

The defaults are exported from `skllm.prompts`.

```python
from skllm.prompts import ZERO_SHOT_SINGLE_LABEL

print(ZERO_SHOT_SINGLE_LABEL)
```

Each estimator declares which template it uses and which placeholders that template requires.

| Estimator | Default template | Placeholders |
| --- | --- | --- |
| `ZeroShotClassifier` | `ZERO_SHOT_SINGLE_LABEL` | `{labels}`, `{text}` |
| `FewShotClassifier` | `FEW_SHOT_SINGLE_LABEL` | `{labels}`, `{text}`, `{examples}` |
| `DynamicFewShotClassifier` | `FEW_SHOT_SINGLE_LABEL` | `{labels}`, `{text}`, `{examples}` |
| `MultiLabelZeroShotClassifier` | `ZERO_SHOT_MULTI_LABEL` | `{labels}`, `{text}`, `{max_labels}` |
| `MultiLabelFewShotClassifier` | `FEW_SHOT_MULTI_LABEL` | `{labels}`, `{text}`, `{max_labels}`, `{examples}` |
| `StructuredExtractor` | `EXTRACTION` | `{text}` |
| `NER` | `NER_TAGGING` | `{entities}`, `{text}` |
| `RubricScorer` | `RUBRIC_SCORING` | `{rubric}`, `{text}`, `{minimum}`, `{maximum}` |
| `Summarizer` | `SUMMARY` | `{text}`, `{max_words}`, `{focus}` |
| `Translator` | `TRANSLATION` | `{text}`, `{output_language}` |
| `ClusterLabeler` | `CLUSTER_NAMING` | `{texts}` |

## What the placeholders hold

`{text}` is one sample, verbatim, so its value differs from sample to sample. Two other placeholders also vary within a batch: `{examples}` on `DynamicFewShotClassifier`, which retrieves its demonstrations per sample, and `{texts}` on `ClusterLabeler`, which names one cluster per request. The rest — `{labels}`, `{max_labels}`, `{entities}`, `{rubric}`, `{minimum}` and `{maximum}`, `{max_words}`, `{focus}`, `{output_language}`, and `{examples}` on `FewShotClassifier` and `MultiLabelFewShotClassifier` — hold the same value across the whole batch.

Several placeholders are filled by helper functions, which `skllm.prompts` also exports so you can see exactly what arrives. `{labels}` is `render_labels(classes_)`, one `- label` line per fitted class, which lets a label contain any punctuation without ambiguity. `{examples}` is `render_examples(...)`, a fenced block per demonstration followed by its categories — the whole training set for `FewShotClassifier`, the retrieved neighbors for `DynamicFewShotClassifier`. `{entities}` is `render_entities(entities)`, one line per entity showing the tag that marks it and what it covers. `{texts}` is `render_exemplars(...)`, the sampled cluster members, each fenced. `{focus}` is `render_focus(focus)`, a complete sentence when a focus is set and an empty string otherwise, so a custom summarizer template must place it where a sentence fits rather than mid-clause.

The rest are plain values. `{max_labels}` is the cap, falling back to the number of fitted classes when `max_labels` is `None`. `{max_words}` and `{output_language}` are the constructor arguments as given. `{rubric}` is the rubric string. `{minimum}` and `{maximum}` are floats, which is why the stock scoring template formats them as `{minimum:g}` — written plainly they render as `0.0` and `10.0`.

## Validation

A template must use exactly the estimator's placeholders: every one of them, and nothing else. The check runs inside `fit`, and again before every batch, so a mistake surfaces at once rather than in the middle of a paid run.

```python
ZeroShotClassifier(prompt_template="Classify {text}").fit(None, ["positive", "negative"])
# ValueError: Invalid prompt_template for ZeroShotClassifier: missing ['labels']
# placeholders. The template must use exactly ['labels', 'text'].
```

An unrecognized placeholder is rejected the same way, since nothing would supply a value for it. Positional fields — `{}` or `{0}` — are rejected too: they have no name to check, and would otherwise fail once per sample as a filled failure. A literal brace is written doubled, `{{` and `}}`, as in any format string. Format specs count as uses, so `{minimum:g}` satisfies the requirement for `{minimum}`.

## Replacing a template

The task stays the same; only the instructions around it change. Here the categories are support-ticket types and the model is told which part of the ticket to judge.

```python
from skllm import ZeroShotClassifier

tickets = [
    "Charged twice for the March invoice and the second one has not been refunded.",
    "The export button does nothing on Firefox. Works in Chrome.",
]

TICKET_PROMPT = """\
You are triaging tickets for a payments company. Assign the ticket to exactly one
of these categories:
{labels}

Judge what the customer needs, not how upset they sound. A ticket that reports
something working incorrectly is a bug report even when it asks for a refund.

Ticket:
---
{text}
---
"""

clf = ZeroShotClassifier(prompt_template=TICKET_PROMPT)
clf.fit(None, ["billing", "bug report", "feature request"])
labels = clf.predict(tickets)
```

`prompt_template` is a constructor parameter like any other, so `get_params`, `set_params` and `clone` carry it, and it is part of what a saved bundle stores. A served bundle answers with the template it was saved with; unlike credentials, it cannot be changed per call. See [Saving and loading](/docs/saving-and-loading).

{% callout type="warning" title="Keep the sample separated from the instructions" %}
Every stock template wraps `{text}` in a fence or delimiters, so that a sample which itself looks like instructions reads as data. A custom template that drops the separation invites the sample to be followed rather than classified. The example above uses `---` lines for the same reason.
{% /callout %}

## What a template does not control

The answer shape is not set by the prompt. Labels, scores and extracted records are constrained through the provider's structured-output mode, from a JSON schema scikit-llm builds — from `classes_` for a classifier, from `schema` for `StructuredExtractor`, and a fixed one-number object for `RubricScorer`. Telling the model in the template to answer differently does not change what is parsed; it only conflicts with the constraint. That is also why `EXTRACTION` never restates the extraction schema: the provider already puts it in front of the model, and a second copy would only cost tokens.

Some invariants are enforced after the answer arrives, and a template cannot relax them. A classifier rejects a label outside `classes_`, `RubricScorer` clamps a score into its scale, and `NER` realigns every rewrite onto the original text and rejects one that changed more than whitespace. A custom template that asks for something these checks disallow turns every sample into a reported failure — see [Batch execution and error handling](/docs/batch-execution-and-error-handling).

Generation parameters live elsewhere as well. Temperature, token limits and reasoning effort belong in `model_params`, described in [Models and providers](/docs/models-and-providers).
