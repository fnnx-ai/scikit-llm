"""Named entity recognition with inline-tagged output.

An answer is the input text with entity mentions wrapped in XML tags, which
makes what counts as a good answer simple to state: stripping the tags must give
the input back. Every rewrite is realigned onto its source text so that holds by
construction, and one that changed more than whitespace is a sample failure.
"""

from __future__ import annotations

import html
import re
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, NamedTuple

import numpy as np
from sklearn.base import TransformerMixin

from skllm.backends import BackendError, Message
from skllm.core import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_MAX_CONCURRENCY,
    BasePromptEstimator,
    StatelessMixin,
    as_texts,
)
from skllm.prompts import NER_TAGGING, render_entities

__all__ = ["NER", "render_tagged_html"]

#: What an entity label must look like to name an XML tag.
_LABEL = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")

#: An opening or closing tag. Only names that are entity labels are read as
#: tags, so a text holding angle brackets of its own passes through untouched.
_TAG = re.compile(r"</?(?P<name>[A-Za-z_][A-Za-z0-9_.-]*)>")

#: Badge colors, handed to entities in declaration order. All light enough to
#: carry near-black text, so one palette works on any notebook theme.
_PALETTE = (
    "#a8d8ea",
    "#f9d5a7",
    "#c5e6a6",
    "#f5b7c1",
    "#d7c6ee",
    "#a9e5dd",
    "#ecd9a0",
    "#e3c2b3",
)


class _Piece(NamedTuple):
    """One flat run of a rewrite: its text, and the entity it is tagged as."""

    text: str
    label: str | None


class NER(TransformerMixin, StatelessMixin, BasePromptEstimator):
    """Mark the entity mentions in each text with inline XML tags.

    ``entities`` maps a label to a description of what it covers; the labels
    become the tag names, so ``{"PER": "a person"}`` turns "Tim Cook leads" into
    ``"<PER>Tim Cook</PER> leads"``. Spans are flat — never nested or
    overlapping — and stripping the tags off an output reproduces its input
    exactly. A rewrite that changed anything but whitespace cannot be reconciled
    with its source and is a sample failure, filled with the untagged original.
    """

    prompt_placeholders: ClassVar[frozenset[str]] = frozenset({"entities", "text"})
    default_prompt_template: ClassVar[str] = NER_TAGGING

    def __init__(
        self,
        model: str = DEFAULT_CHAT_MODEL,
        *,
        entities: Mapping[str, str] | None = None,
        prompt_template: str | None = None,
        model_params: Mapping[str, Any] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        alias: str | None = None,
        on_error: str = "fill",
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        verbose: int = 1,
    ) -> None:
        super().__init__(
            model=model,
            prompt_template=prompt_template,
            model_params=model_params,
            api_key=api_key,
            base_url=base_url,
            alias=alias,
            on_error=on_error,
            max_concurrency=max_concurrency,
            verbose=verbose,
        )
        self.entities = entities

    def transform(self, X: Any) -> np.ndarray:
        """Tag every text, returning one string per sample in input order."""
        self._check_ready()
        texts = as_texts(X)
        # Not _run_batch: a failed sample fills with its own input text, which
        # one fill value shared by the whole batch cannot express.
        answers = self._run_repeated(texts, self._tag, repeats=1)
        return np.asarray(
            [
                tagged[0] if tagged else text
                for tagged, text in zip(answers, texts, strict=True)
            ],
            dtype=object,
        )

    def _check_params(self) -> None:
        super()._check_params()
        self._entity_schema()

    def _tag(self, text: str) -> str:
        entities = self._entity_schema()
        prompt = self._prompt(entities=render_entities(entities), text=text)
        rewrite = self._chat([Message("user", prompt)])
        return _realign(text, _pieces(rewrite, _by_tag_name(entities)))

    def _entity_schema(self) -> Mapping[str, str]:
        """The validated ``label -> description`` schema the tags are named after."""
        entities = self.entities
        if not isinstance(entities, Mapping) or not entities:
            raise ValueError(
                "entities must be a non-empty mapping of entity label to what it "
                "covers, e.g. {'PER': 'a person', 'ORG': 'an organization'}."
            )
        seen: dict[str, str] = {}
        for label, description in entities.items():
            if not isinstance(label, str) or not _LABEL.fullmatch(label):
                raise ValueError(
                    f"Entity label {label!r} cannot name an XML tag: labels must "
                    "match [A-Za-z_][A-Za-z0-9_.-]*, e.g. 'PER' or 'ORG'."
                )
            if not isinstance(description, str) or not description.strip():
                raise ValueError(
                    f"Entity {label!r} needs a description of what to tag, e.g. "
                    f"{{{label!r}: 'a person'}}."
                )
            if (other := seen.setdefault(label.lower(), label)) != label:
                raise ValueError(
                    f"Entity labels {other!r} and {label!r} differ only in case, "
                    "but tags are matched case-insensitively."
                )
        return entities


def render_tagged_html(texts: Any, entities: Mapping[str, str]) -> str:
    """Highlight the entities in tagged texts, above a legend of what each means.

    The result is HTML that a notebook displays inline and every other caller
    can treat as the string it is.
    """
    colors = {
        label: _PALETTE[position % len(_PALETTE)]
        for position, label in enumerate(entities)
    }
    legend = " ".join(
        _badge(label, title=f"{label}: {description}", color=colors[label])
        for label, description in entities.items()
    )
    body = "<hr>".join(_highlight(text, entities, colors) for text in as_texts(texts))
    return _Html(
        '<div style="white-space:pre-wrap;line-height:2;">'
        f'<div style="margin-bottom:.5em;">{legend}</div>{body}</div>'
    )


class _Html(str):
    """HTML that a notebook renders instead of printing."""

    def _repr_html_(self) -> str:
        return str(self)


def _highlight(
    text: str, entities: Mapping[str, str], colors: Mapping[str, str]
) -> str:
    return "".join(
        html.escape(piece.text)
        if piece.label is None
        else _badge(
            piece.text,
            title=f"{piece.label}: {entities[piece.label]}",
            color=colors[piece.label],
            label=piece.label,
        )
        for piece in _pieces(text, _by_tag_name(entities))
    )


def _badge(text: str, *, title: str, color: str, label: str = "") -> str:
    tail = (
        f'<span style="font-size:.7em;font-weight:700;"> {html.escape(label)}</span>'
        if label
        else ""
    )
    return (
        f'<span title="{html.escape(title, quote=True)}" style="background:{color};'
        'color:#1a1a1a;border-radius:4px;padding:1px 5px;">'
        f"{html.escape(text)}{tail}</span>"
    )


def _by_tag_name(entities: Mapping[str, str]) -> dict[str, str]:
    """The labels keyed by the lowercased tag name a rewrite may spell them with."""
    return {label.lower(): label for label in entities}


def _pieces(tagged: str, canonical: Mapping[str, str]) -> list[_Piece]:
    """Split a tagged text into flat runs, tagged and untagged.

    Only tags naming a known entity are read as tags; anything else — an unknown
    name, a close without an open, a nested open — stays literal text, and so
    shows up as a failure to reconcile rather than as deleted characters.
    """
    pieces: list[_Piece] = []
    start = 0
    open_label: str | None = None
    for match in _TAG.finditer(tagged):
        label = canonical.get(match["name"].lower())
        closing = match[0].startswith("</")
        if label is None or (
            label != open_label if closing else open_label is not None
        ):
            continue
        pieces.append(_Piece(tagged[start : match.start()], open_label))
        start = match.end()
        open_label = None if closing else label
    # A tag left open runs to the end of the text rather than voiding its span.
    pieces.append(_Piece(tagged[start:], open_label))
    return pieces


def _realign(source: str, pieces: Sequence[_Piece]) -> str:
    """Re-emit ``source`` with the rewrite's spans tagged, whitespace and all.

    Spans are located by their non-whitespace characters, so a rewrite that only
    re-spaced the text still lands on the right words, and one that put the tags
    around the spacing still tags the words alone. Because the output is built
    out of ``source`` itself, stripping its tags reproduces the input exactly.
    """
    runs = [(piece.label, _dense(piece.text)) for piece in pieces]
    if "".join(text for _, text in runs) != _dense(source):
        raise BackendError(
            "The model's answer does not reproduce the text sample: tagging may "
            "only add tags, and every other character must survive unchanged."
        )

    offsets = [position for position, char in enumerate(source) if not char.isspace()]
    tagged: list[str] = []
    emitted = 0
    consumed = 0
    for label, text in runs:
        if label is not None and text:
            start = offsets[consumed]
            end = offsets[consumed + len(text) - 1] + 1
            tagged.append(source[emitted:start])
            tagged.append(f"<{label}>{source[start:end]}</{label}>")
            emitted = end
        consumed += len(text)
    tagged.append(source[emitted:])
    return "".join(tagged)


def _dense(text: str) -> str:
    """A text stripped of the whitespace a rewrite is allowed to disagree on."""
    return "".join(text.split())
