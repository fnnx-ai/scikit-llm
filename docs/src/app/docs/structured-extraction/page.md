---
title: Structured extraction
---

`StructuredExtractor` pulls the fields a schema describes out of a text. Each input text produces one record, so a batch of product reviews comes back as a list of dictionaries in input order.

```python
from skllm import StructuredExtractor

schema = {
    "type": "object",
    "properties": {
        "product": {"type": "string"},
        "rating": {"type": "integer"},
        "sentiment": {"type": "string", "enum": ["positive", "mixed", "negative"]},
        "complaint": {"type": ["string", "null"]},
    },
    "required": ["product", "rating", "sentiment", "complaint"],
    "additionalProperties": False,
}

reviews = [
    "The XR-200 headphones arrived with a scratched earcup. Sound is fine, but "
    "for 199 dollars I expected better. Two stars.",
    "The Aurora desk lamp has been perfect since day one. Assembly took five "
    "minutes and the dimmer is exactly what I wanted. Five stars.",
]

extractor = StructuredExtractor(schema=schema)
extractor.transform(reviews)
# [{'product': 'XR-200 headphones', 'rating': 2, 'sentiment': 'negative',
#   'complaint': 'arrived with a scratched earcup'},
#  {'product': 'Aurora desk lamp', 'rating': 5, 'sentiment': 'positive',
#   'complaint': None}]
```

The extractor keeps no fitted state. `fit` validates its parameters and nothing else, so `transform` can be called on its own, and `fit_transform` is there for use inside an sklearn pipeline.

## Describing the schema

`schema` is keyword-only, like every constructor parameter except the leading `model`, and it takes either a JSON Schema object or a pydantic model class.

```python
from pydantic import BaseModel

from skllm import StructuredExtractor


class Review(BaseModel):
    product: str
    rating: int
    complaint: str | None = None


extractor = StructuredExtractor(schema=Review)
```

A class is recognized by its `model_json_schema` method rather than by an `isinstance` check, so pydantic is an accepted input form and not a dependency of scikit-llm. `get_params` and `clone` hand the class back as it was passed. It is compiled to its schema dict at the pickling boundary, so a saved estimator carries the dict alone and the environment that serves it needs neither the class nor pydantic. See [Saving and loading](/docs/saving-and-loading).

The schema must describe an object, since one text is extracted into one record. A schema whose top level is anything else raises `ValueError` on the first call. To pull several records out of a single text, nest them under a property:

```python
schema = {
    "type": "object",
    "properties": {
        "people": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string"},
                },
                "required": ["name", "role"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["people"],
    "additionalProperties": False,
}
```

## Provider support for structured output

The schema travels with the request through whatever constrained-decoding mode the provider offers. [Models and providers](/docs/models-and-providers) covers what that mode is for each of them.

The keywords those modes accept are subsets of JSON Schema, and the subsets differ. OpenAI's strict mode rejects a keyword it does not support outright rather than ignoring it, so a schema that runs against one provider is not guaranteed to run against another.

Strict mode is requested only for a schema in which every object is closed: `additionalProperties` is `false` and every property is listed in `required`. That is why the review schema above marks `complaint` as nullable instead of leaving it out of `required`. The object stays closed, and the model can still report that the text says nothing about that field. A pydantic model compiles to neither marker, so a model class is sent without the strict flag. The schema itself travels with the request either way.

{% callout title="The schema is not in the prompt" %}
Every provider puts the schema in front of the model through its own channel, so scikit-llm does not render it into the prompt as well — a second copy would only cost tokens. The prompt carries the text sample alone, and a custom `prompt_template` therefore uses the single `{text}` placeholder. See [Customizing prompts](/docs/customizing-prompts).
{% /callout %}

Whatever the provider did or did not enforce, every answer is validated against the schema locally before it is handed back, using the same validator FNNX applies to a bundle's JSON channels. A result is a record that fits the schema, or a reported failure, never a half-filled dictionary.

## Output and failed samples

`transform` returns a plain Python list of `dict[str, Any] | None`, one entry per input text, in input order. A sample that failed is `None`, the fill value for this task.

Three things count as a failure: a request that could not be completed, an answer that is not an object at all, and an object that does not fit the schema. The last covers a field of the wrong type, a missing required field, and an extra field where the schema is closed. Under the default `on_error="fill"` the batch warns once, fills each failed sample with `None` and records the reasons. Under `on_error="raise"` the first failure aborts the batch instead. See [Batch execution and error handling](/docs/batch-execution-and-error-handling).

```python
records = extractor.transform(reviews)

[index for index, record in enumerate(records) if record is None]
extractor.errors_
# (SampleError(index=1, error="BackendError: The model's answer does not fit the
#  extraction schema: Missing required property: sentiment in {'product':
#  'Aurora desk lamp', 'rating': 5, 'complaint': None}"),)
```
