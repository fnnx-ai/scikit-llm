"""Default prompt templates for the classification tasks.

Every placeholder a template uses is part of the task's documented set, which
``prompt_template`` overrides are validated against.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = [
    "FEW_SHOT_MULTI_LABEL",
    "FEW_SHOT_SINGLE_LABEL",
    "ZERO_SHOT_MULTI_LABEL",
    "ZERO_SHOT_SINGLE_LABEL",
    "render_examples",
    "render_labels",
]

ZERO_SHOT_SINGLE_LABEL = """\
Classify the text sample into exactly one of the following categories:
{labels}

Text sample:
```
{text}
```
"""

FEW_SHOT_SINGLE_LABEL = """\
Classify the text sample into exactly one of the following categories:
{labels}

Labeled examples:
{examples}

Text sample:
```
{text}
```
"""

ZERO_SHOT_MULTI_LABEL = """\
Assign the text sample to every applicable category, at most {max_labels} of:
{labels}

Text sample:
```
{text}
```
"""

FEW_SHOT_MULTI_LABEL = """\
Assign the text sample to every applicable category, at most {max_labels} of:
{labels}

Labeled examples:
{examples}

Text sample:
```
{text}
```
"""


def render_labels(labels: Sequence[str]) -> str:
    """List the categories one per line, so a label may contain any punctuation."""
    return "\n".join(f"- {label}" for label in labels)


def render_examples(examples: Sequence[tuple[str, str | list[str]]]) -> str:
    return "\n\n".join(
        "Text sample:\n```\n{}\n```\nCategories: {}".format(
            text, labels if isinstance(labels, str) else ", ".join(labels)
        )
        for text, labels in examples
    )
