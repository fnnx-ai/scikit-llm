"""Default prompt template for the cluster-naming task.

Every placeholder a template uses is part of the task's documented set, which
``prompt_template`` overrides are validated against.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["CLUSTER_NAMING", "render_exemplars"]

CLUSTER_NAMING = """\
The text samples below were grouped together by a clustering model. Name the
topic that holds them together, in a few words, specifically enough to tell this
group apart from other groups of the same corpus.
Answer with the name alone, without any preamble or quotation marks.

Text samples:
{texts}
"""


def render_exemplars(texts: Sequence[str]) -> str:
    """Fence each sample, so a sample of its own is never read as instructions."""
    return "\n\n".join(f"```\n{text}\n```" for text in texts)
