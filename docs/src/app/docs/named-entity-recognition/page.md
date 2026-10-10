---
title: Named entity recognition
---

`NER` marks the entity mentions in a text with inline XML tags named after the entity label. It is a transformer: `transform` returns one tagged string per input text, in input order, as a numpy object array.

```python
from skllm import NER

ner = NER(
    entities={
        "PER": "a person",
        "ORG": "a company or institution",
        "LOC": "a city, region or country",
    }
)

tagged = ner.transform(["Tim Cook opened an Apple store in Berlin."])
tagged[0]
# '<PER>Tim Cook</PER> opened an <ORG>Apple</ORG> store in <LOC>Berlin</LOC>.'
```

The estimator keeps no fitted state. `fit` validates its parameters and nothing else, so `transform` can be called on its own, and `fit_transform` is there for use inside an sklearn pipeline.

{% callout title="Changed in v2" %}
`GPTExplainableNER` and `AnthropicExplainableNER` are both `skllm.NER`, with the provider named in the model string. The `<entity><reasoning>` envelope, `sparse_output` and `display_predictions` are gone. See [Migrating from v1](/docs/migrating-from-v1).
{% /callout %}

## Declaring the entities

`entities` is keyword-only and maps each label to a description of what it covers. The label names the tag, and the description is the entire definition of the category the model works from. A description that defines the category carries more than one that restates the label: `{"ORG": "a company, agency or institution, but not a product name"}` says more than `{"ORG": "organization"}`.

A label has to be able to name an XML tag, so it must match `[A-Za-z_][A-Za-z0-9_.-]*`. Every description must be non-blank, and two labels that differ only in case are rejected, because tags are matched case-insensitively on the way back. The mapping is validated on the first `fit` or `transform` rather than in the constructor, so a mapping that is empty or missing raises `ValueError` there too.

Both label and description reach the model through the prompt, which renders each category as the tag that marks it: `- <PER>...</PER>: a person`. A custom `prompt_template` uses the `{entities}` and `{text}` placeholders. See [Customizing prompts](/docs/customizing-prompts).

## The round trip

The model is asked to repeat the sample with tags added, and answers as free text rather than through a constrained schema. Tagging therefore asks nothing of a provider beyond a chat call, whichever one the [model string](/docs/models-and-providers) names. That answer is not returned as it arrived. It is realigned onto the original text: the tagged spans are located by their non-whitespace characters, and the output is rebuilt out of the source string itself. Stripping the tags off a result therefore reproduces the input exactly, character for character, including any spacing the model normalized along the way.

```python
import re

texts = ["Tim Cook opened an Apple store in Berlin."]
tagged = ner.transform(texts)

[re.sub(r"</?[A-Za-z_][A-Za-z0-9_.-]*>", "", text) for text in tagged] == texts
# True
```

Spans are flat: never nested, never overlapping. A single label's mentions can therefore be read back with one expression:

```python
re.findall(r"<ORG>(.*?)</ORG>", tagged[0])
# ['Apple']
```

Only tags naming a declared entity are read as tags, so a text carrying angle brackets of its own passes through untouched. A tag the model invented outside the entity mapping stays literal text, which surfaces as a failure to reconcile rather than as characters silently dropped from the sample.

A rewrite that changed anything but whitespace cannot be reconciled with its source, and that is a sample failure. The fill value is the sample's own untagged input text rather than a value shared by the batch, so the output stays one string per input whatever happened, and code that strips tags is unaffected by a failure. The cost of that choice is that a filled sample is indistinguishable from a text that genuinely mentions no entity: which samples failed, and why, is in `errors_`.

```python
ner.errors_
# (SampleError(index=1, error="BackendError: The model's answer does not reproduce
#  the text sample: tagging may only add tags, and every other character must
#  survive unchanged."),)
```

Under the default `on_error="fill"` the batch also warns once, naming how many samples were filled. Under `on_error="raise"` the first failure aborts the batch instead. See [Batch execution and error handling](/docs/batch-execution-and-error-handling).

## Displaying tagged text

`render_tagged_html` turns tagged texts into HTML that shows each span as a colored badge carrying its label, above a legend naming what every label covers. It takes the tagged texts and the same `entities` mapping, assigns colors in the mapping's declaration order, and escapes the text it highlights. It is imported from the module rather than the package root:

```python
from skllm.tagging import render_tagged_html

render_tagged_html(tagged, ner.entities)
```

The return value is a string. A notebook renders it inline, because the string carries a `_repr_html_`. Anywhere else it can be written to a file or embedded in a report like any other HTML fragment. Several texts in one call are rendered one after another, separated by a rule.

{% callout title="Pass a sequence, not a single string" %}
`render_tagged_html` normalizes its input the way an estimator normalizes `X`, so a bare string raises `TypeError`. To render one text, pass `tagged[:1]` rather than `tagged[0]`.
{% /callout %}
