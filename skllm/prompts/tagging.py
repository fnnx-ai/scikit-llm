"""Default prompt template for the named-entity tagging task.

Every placeholder a template uses is part of the task's documented set, which
``prompt_template`` overrides are validated against.
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = ["NER_TAGGING", "render_entities"]

NER_TAGGING = """\
Repeat the text sample, wrapping every entity mention in an XML tag named after
its category. The categories are:
{entities}

Rules:
- Reproduce the sample exactly: adding tags is the only change allowed, and
  every other character must stay as it is.
- Tag the shortest span that names the entity; never nest or overlap tags.
- Repeat the sample unchanged where it mentions none of these categories.
- Answer with the tagged sample alone, without any preamble or code fence.

Text sample:
```
{text}
```
"""


def render_entities(entities: Mapping[str, str]) -> str:
    """Show each category as the tag that marks it, then what it covers."""
    return "\n".join(
        f"- <{label}>...</{label}>: {description}"
        for label, description in entities.items()
    )
