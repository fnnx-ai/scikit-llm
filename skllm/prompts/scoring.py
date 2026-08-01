"""Default prompt template for the rubric-scoring task.

Every placeholder a template uses is part of the task's documented set, which
``prompt_template`` overrides are validated against.
"""

from __future__ import annotations

__all__ = ["RUBRIC_SCORING"]

# The scale is stated here rather than as a schema constraint: no adapter's
# structured mode is guaranteed to accept numeric bounds, and clamping is the
# backstop for an answer that leaves the range anyway.
RUBRIC_SCORING = """\
Score the text sample against the rubric below, on a scale from {minimum:g} to \
{maximum:g}.
Judge it by the rubric alone, and use the whole scale: {minimum:g} for a sample
that meets none of the criteria, {maximum:g} for one that meets all of them.

Rubric:
{rubric}

Text sample:
```
{text}
```
"""
