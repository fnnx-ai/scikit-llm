"""Default prompt template for the structured-extraction task.

Every placeholder a template uses is part of the task's documented set, which
``prompt_template`` overrides are validated against.
"""

from __future__ import annotations

__all__ = ["EXTRACTION"]

# The schema itself is never rendered here: every backend puts it in front of the
# model already — as a native response schema, a tool signature, or the prompt
# suffix the litellm fallback appends — so a second copy would only cost tokens.
EXTRACTION = """\
Extract the information the response schema describes from the text sample.
Base every value on what the sample itself states rather than on outside
knowledge, and leave a value empty where the sample states nothing about it.

Text sample:
```
{text}
```
"""
