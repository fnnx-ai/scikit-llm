"""Default prompt templates for the text-to-text tasks.

Every placeholder a template uses is part of the task's documented set, which
``prompt_template`` overrides are validated against.
"""

from __future__ import annotations

__all__ = ["SUMMARY", "TRANSLATION", "render_focus"]

SUMMARY = """\
Summarize the text sample in at most {max_words} words.{focus}
Answer with the summary alone, without any preamble.

Text sample:
```
{text}
```
"""

TRANSLATION = """\
Translate the text sample into {output_language}, preserving its meaning and tone.
If the sample is already written in {output_language}, repeat it unchanged.
Answer with the translation alone, without any preamble.

Text sample:
```
{text}
```
"""


def render_focus(focus: str | None) -> str:
    """The focus instruction, or nothing at all when the summary is a general one.

    A text that does not cover the focus is summarized generally and silently:
    an answer explaining the mismatch would end up in the summary column.
    """
    if not focus:
        return ""
    return (
        f" Concentrate on what the text says about {focus}; if it says nothing "
        "about that, summarize it in general without noting the omission."
    )
