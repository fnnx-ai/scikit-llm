"""Zero-shot, few-shot and multi-label text classifiers.

Labels are obtained through structured outputs constrained to ``classes_``, so a
prediction is either one of the fitted labels or a reported sample failure —
never an unconstrained string. A System One model is asked typed questions
instead — a choice among the classes, or a yes/no per class — and its
probabilities back ``predict_proba`` directly.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.neighbors import NearestNeighbors

from skllm.backends import BackendError, Message, parse_model
from skllm.core import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MAX_CONCURRENCY,
    BasePromptEstimator,
    as_texts,
    check_count,
)
from skllm.prompts import (
    CATEGORY_QUESTION,
    FEW_SHOT_MULTI_LABEL,
    FEW_SHOT_SINGLE_LABEL,
    MEMBERSHIP_QUESTION,
    ZERO_SHOT_MULTI_LABEL,
    ZERO_SHOT_SINGLE_LABEL,
    render_examples,
    render_labels,
)
from skllm.vectorization import DEFAULT_EMBEDDING_MODELS, Vectorizer

if TYPE_CHECKING:
    from typing import Self

__all__ = [
    "DynamicFewShotClassifier",
    "FewShotClassifier",
    "MultiLabelFewShotClassifier",
    "MultiLabelZeroShotClassifier",
    "ZeroShotClassifier",
]

#: How much of an unusable answer an error message quotes.
_QUOTED_ANSWER_CHARS = 200

#: The id a single-label classifier asks its one System One question under.
_CATEGORY_ID = "category"


class _BaseClassifier(ClassifierMixin, BasePromptEstimator):
    """What every classifier shares: fitting labels, voting and aggregation.

    One call per sample answers a fitted classifier; ``n_predictors`` turns that
    into k sampled calls whose votes are aggregated, at k times the cost.
    """

    _system_one_capable: ClassVar[bool] = True

    def __init__(
        self,
        model: str = DEFAULT_CHAT_MODEL,
        *,
        n_predictors: int | None = None,
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
        self.n_predictors = n_predictors

    def fit(self, X: Any, y: Any) -> Self:
        """Derive ``classes_`` from ``y``; few-shot variants also keep examples."""
        self._check_params()
        rows = self._check_targets(y)
        # Everything that can reject the data runs before any state is assigned:
        # a fit that fails halfway would otherwise leave the estimator looking
        # fitted to check_is_fitted while missing the classes predict needs.
        classes = _distinct(rows)
        self._check_classes(classes)
        self._fit_examples(X, rows)
        self.classes_ = classes
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Per-class probabilities, aligned with ``classes_``.

        For a System One model these are the model's own calibrated
        probabilities. Otherwise they are vote fractions, degenerate without
        ``n_predictors``: the single deterministic answer is the only vote, so
        its row is one-hot (per-class 0/1 for multi-label). A failed sample's
        row is all zeros.
        """
        self._check_ready()
        if self._system_one() is None:
            rows = [self._frequencies(votes) for votes in self._collect_votes(X)]
        else:
            rows = [
                [0.0] * len(self.classes_) if row is None else row
                for row in self._collect_probabilities(X)
            ]
        return np.asarray(rows, dtype=float).reshape(len(rows), len(self.classes_))

    def _check_params(self) -> None:
        super()._check_params()
        check_count("n_predictors", self.n_predictors)
        if self.n_predictors is not None and self._system_one() is not None:
            raise ValueError(
                f"n_predictors samples votes to estimate probabilities, but "
                f"{self.model!r} is a System One model, which returns calibrated "
                "probabilities from a single call. Leave n_predictors unset."
            )
        if self._repeats() > 1 and self._request_params().get("temperature") == 0:
            raise ValueError(
                f"n_predictors={self.n_predictors} samples that many answers per "
                "text, but an explicit temperature of 0 makes every vote "
                "identical at that many times the cost. Drop the temperature from "
                "model_params (skllm suppresses its own temperature default for "
                "sampled voting) or leave n_predictors unset."
            )

    def _repeats(self) -> int:
        return 1 if self.n_predictors is None else self.n_predictors

    def _collect_votes(self, X: Any) -> list[list[tuple[str, ...]]]:
        """The successful votes per sample, empty where every call failed."""
        repeats = self._repeats()
        return self._run_repeated(
            self._prompts(as_texts(X)),
            partial(self._vote, deterministic=repeats == 1),
            repeats=repeats,
        )

    def _prompts(self, texts: Sequence[str]) -> list[str]:
        """Render one prompt per sample, before the batch goes out.

        Rendering up front rather than inside each worker lets a task gather
        what its prompts need — the dynamic classifier's retrieval — for the
        whole batch at once, and renders it once per sample rather than once
        per vote.
        """
        return [self._prompt(**self._prompt_values(text)) for text in texts]

    def _collect_probabilities(self, X: Any) -> list[list[float] | None]:
        """A System One model's per-class probabilities, ``None`` where it failed."""
        texts = as_texts(X)
        requests = [
            (text, self._questions(examples))
            for text, examples in zip(texts, self._demonstrations(texts), strict=True)
        ]
        return self._run_batch(
            requests, lambda request: self._probabilities(*request), fill=lambda: None
        )

    def _probabilities(
        self, text: str, questions: Mapping[str, Mapping[str, Any]]
    ) -> list[float]:
        return self._read_probabilities(self._evaluate(text, questions))

    def _vote(self, prompt: str, *, deterministic: bool) -> tuple[str, ...]:
        answer = self._structured(
            [Message("user", prompt)], self._schema(), deterministic=deterministic
        )
        return self._parse(answer)

    def _frequencies(self, votes: Sequence[tuple[str, ...]]) -> list[float]:
        counts = Counter(label for vote in votes for label in vote)
        return [counts[label] / len(votes) if votes else 0.0 for label in self.classes_]

    def _prompt_values(self, text: str) -> dict[str, Any]:
        return {"labels": render_labels(self.classes_), "text": text}

    def _fit_examples(self, X: Any, rows: Sequence[Any]) -> None:
        """Zero-shot classifiers ignore X, which may be ``None``."""

    def _demonstrations(self, texts: Sequence[str]) -> list[Sequence[tuple[str, Any]]]:
        """Which examples to demonstrate the task with, per sample."""
        return [()] * len(texts)

    def _check_classes(self, classes: Sequence[str]) -> None:
        """Reject a label set the model cannot be asked about, before fitting."""

    def _check_targets(self, y: Any) -> list[Any]:
        """Validate y and return its labels row by row."""
        raise NotImplementedError

    def _schema(self) -> Mapping[str, Any]:
        raise NotImplementedError

    def _parse(self, answer: Any) -> tuple[str, ...]:
        raise NotImplementedError

    def _questions(
        self, examples: Sequence[tuple[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """The System One questions whose answers decide one sample."""
        raise NotImplementedError

    def _read_probabilities(self, answers: Mapping[str, Any]) -> list[float]:
        """Per-class probabilities, aligned with ``classes_``, from the answers."""
        raise NotImplementedError


class _SingleLabelClassifier(_BaseClassifier):
    """One label per sample, filled with ``default_label`` when a sample fails."""

    def __init__(
        self,
        model: str = DEFAULT_CHAT_MODEL,
        *,
        default_label: str | None = None,
        n_predictors: int | None = None,
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
            n_predictors=n_predictors,
            prompt_template=prompt_template,
            model_params=model_params,
            api_key=api_key,
            base_url=base_url,
            alias=alias,
            on_error=on_error,
            max_concurrency=max_concurrency,
            verbose=verbose,
        )
        self.default_label = default_label

    def predict(self, X: Any) -> np.ndarray:
        """The winning label per sample, or ``default_label`` where none arrived.

        The majority vote, or a System One model's most probable class.
        """
        self._check_ready()
        if self._system_one() is None:
            labels = [self._decide(votes) for votes in self._collect_votes(X)]
        else:
            labels = [
                self._most_probable(row) for row in self._collect_probabilities(X)
            ]
        return np.asarray(labels, dtype=object)

    def _check_params(self) -> None:
        super()._check_params()
        if self.default_label is not None and not isinstance(self.default_label, str):
            raise TypeError(
                "default_label must be a string or None, got "
                f"{type(self.default_label).__name__}."
            )

    def _decide(self, votes: Sequence[tuple[str, ...]]) -> str | None:
        if not votes:
            return self.default_label
        counts = Counter(label for (label,) in votes)
        winning = max(counts.values())
        # Ties go to the earliest class, which keeps a repeated run reproducible.
        return next(label for label in self.classes_ if counts[label] == winning)

    def _most_probable(self, row: Sequence[float] | None) -> str | None:
        if row is None:
            return self.default_label
        # argmax breaks ties toward the earliest class, as the vote count does.
        return str(self.classes_[int(np.argmax(row))])

    def _check_classes(self, classes: Sequence[str]) -> None:
        backend = self._system_one()
        if backend is not None and len(classes) > backend.max_choice_options:
            raise ValueError(
                f"y holds {len(classes)} classes, but {self.model!r} chooses among "
                f"at most {backend.max_choice_options} options per question."
            )

    def _check_targets(self, y: Any) -> list[str]:
        return _check_labels(y)

    def _schema(self) -> Mapping[str, Any]:
        return _object_schema("label", {"type": "string", "enum": list(self.classes_)})

    def _parse(self, answer: Any) -> tuple[str, ...]:
        label = answer.get("label") if isinstance(answer, Mapping) else None
        if not isinstance(label, str):
            raise BackendError(
                f"The model answered without a 'label' string: {_quote(answer)}."
            )
        if label not in self.classes_:
            raise BackendError(
                f"The model answered {label!r}, which is not one of the fitted "
                f"classes {list(self.classes_)}."
            )
        return (label,)

    def _questions(
        self, examples: Sequence[tuple[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        texts = _texts_per_label(examples)
        return {
            _CATEGORY_ID: {
                "type": "choice",
                "instructions": CATEGORY_QUESTION,
                "criteria": {
                    label: _demonstrated(texts.get(label)) for label in self.classes_
                },
            }
        }

    def _read_probabilities(self, answers: Mapping[str, Any]) -> list[float]:
        answer = answers.get(_CATEGORY_ID)
        probabilities = (
            answer.get("probabilities") if isinstance(answer, Mapping) else None
        )
        if not isinstance(probabilities, Mapping) or set(probabilities) != set(
            self.classes_
        ):
            raise BackendError(
                "The model answered without a probability per fitted class: "
                f"{_quote(answer)}."
            )
        return [float(probabilities[label]) for label in self.classes_]


class _MultiLabelClassifier(_BaseClassifier):
    """Zero or more labels per sample, filled with an empty list on failure."""

    def __init__(
        self,
        model: str = DEFAULT_CHAT_MODEL,
        *,
        max_labels: int | None = None,
        n_predictors: int | None = None,
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
            n_predictors=n_predictors,
            prompt_template=prompt_template,
            model_params=model_params,
            api_key=api_key,
            base_url=base_url,
            alias=alias,
            on_error=on_error,
            max_concurrency=max_concurrency,
            verbose=verbose,
        )
        self.max_labels = max_labels

    def predict(self, X: Any) -> list[list[str]]:
        """One label list per sample — ragged, never padded.

        The labels a majority voted for, or those a System One model deems more
        likely than not; either way the most supported first.
        """
        self._check_ready()
        if self._system_one() is None:
            return [self._decide(votes) for votes in self._collect_votes(X)]
        return [self._applicable(row) for row in self._collect_probabilities(X)]

    def _check_params(self) -> None:
        super()._check_params()
        check_count("max_labels", self.max_labels)

    def _decide(self, votes: Sequence[tuple[str, ...]]) -> list[str]:
        if len(votes) == 1:
            chosen = list(votes[0])
        else:
            counts = Counter(label for vote in votes for label in vote)
            chosen = sorted(
                (label for label in self.classes_ if counts[label] * 2 > len(votes)),
                key=lambda label: -counts[label],
            )
        return chosen[: self._label_cap()]

    def _applicable(self, row: Sequence[float] | None) -> list[str]:
        if row is None:
            return []
        # The same bar as the vote: strictly more likely than not.
        chosen = sorted(
            (
                (probability, label)
                for label, probability in zip(self.classes_, row, strict=True)
                if probability > 0.5
            ),
            key=lambda pair: -pair[0],
        )
        return [label for _, label in chosen[: self._label_cap()]]

    def _label_cap(self) -> int:
        """The cap the prompt states and the truncation backstop enforces."""
        return self.max_labels or len(self.classes_)

    def _prompt_values(self, text: str) -> dict[str, Any]:
        return {**super()._prompt_values(text), "max_labels": self._label_cap()}

    def _check_targets(self, y: Any) -> list[list[str]]:
        rows = []
        for index, row in enumerate(y):
            if isinstance(row, str) or not isinstance(row, Iterable):
                raise TypeError(
                    f"y[{index}] is a {type(row).__name__}, but multi-label targets "
                    "are collections of labels, e.g. [['quality', 'price'], ['delivery']]."
                )
            rows.append(_check_labels(row, at=f"y[{index}]"))
        return rows

    def _schema(self) -> Mapping[str, Any]:
        labels: dict[str, Any] = {
            "type": "array",
            "items": {"type": "string", "enum": list(self.classes_)},
        }
        backend, _ = self._backend()
        if "maxItems" in getattr(backend, "schema_constraints", frozenset()):
            labels["maxItems"] = self._label_cap()
        return _object_schema("labels", labels)

    def _parse(self, answer: Any) -> tuple[str, ...]:
        labels = answer.get("labels") if isinstance(answer, Mapping) else None
        if not isinstance(labels, list) or not all(
            isinstance(label, str) for label in labels
        ):
            raise BackendError(
                f"The model answered without a 'labels' array of strings: "
                f"{_quote(answer)}."
            )
        unknown = [label for label in labels if label not in self.classes_]
        if unknown:
            raise BackendError(
                f"The model answered {unknown}, which is not among the fitted "
                f"classes {list(self.classes_)}."
            )
        return tuple(dict.fromkeys(labels))

    def _questions(
        self, examples: Sequence[tuple[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """One yes/no question per class, all answered in the same request."""
        texts = _texts_per_label(examples)
        questions: dict[str, dict[str, Any]] = {}
        for label in self.classes_:
            question: dict[str, Any] = {
                "type": "noul",
                "instructions": {"category": label, "question": MEMBERSHIP_QUESTION},
            }
            if label in texts:
                question["criteria"] = {"true": _demonstrated(texts[label])}
            questions[label] = question
        return questions

    def _read_probabilities(self, answers: Mapping[str, Any]) -> list[float]:
        row = []
        for label in self.classes_:
            answer = answers.get(label)
            probability = answer.get("noul") if isinstance(answer, Mapping) else None
            if isinstance(probability, bool) or not isinstance(
                probability, (int, float)
            ):
                raise BackendError(
                    f"The model answered without a yes/no probability for {label!r}: "
                    f"{_quote(answer)}."
                )
            row.append(float(probability))
        return row


class _FewShotMixin(_BaseClassifier):
    """Carries fitted examples into every prompt as demonstrations."""

    def _fit_examples(self, X: Any, rows: Sequence[Any]) -> None:
        self.examples_ = _paired(X, rows)

    def _prompts(self, texts: Sequence[str]) -> list[str]:
        return [
            self._prompt(
                **self._prompt_values(text), examples=render_examples(examples)
            )
            for text, examples in zip(texts, self._demonstrations(texts), strict=True)
        ]

    def _demonstrations(self, texts: Sequence[str]) -> list[Sequence[tuple[str, Any]]]:
        return [self.examples_] * len(texts)


class ZeroShotClassifier(_SingleLabelClassifier):
    """Assign one of the labels seen at ``fit`` to each text, without examples."""

    prompt_placeholders: ClassVar[frozenset[str]] = frozenset({"labels", "text"})
    default_prompt_template: ClassVar[str] = ZERO_SHOT_SINGLE_LABEL


class FewShotClassifier(_FewShotMixin, _SingleLabelClassifier):
    """Assign one label per text, demonstrating the task with the training set."""

    prompt_placeholders: ClassVar[frozenset[str]] = frozenset(
        {"labels", "text", "examples"}
    )
    default_prompt_template: ClassVar[str] = FEW_SHOT_SINGLE_LABEL


class DynamicFewShotClassifier(_FewShotMixin, _SingleLabelClassifier):
    """Assign one label per text, demonstrating with the examples nearest to it.

    Every class contributes up to ``n_examples`` of its own training texts —
    those closest to the sample in embedding space — so the prompt stays small
    and balanced however large the training set is.
    """

    prompt_placeholders: ClassVar[frozenset[str]] = frozenset(
        {"labels", "text", "examples"}
    )
    default_prompt_template: ClassVar[str] = FEW_SHOT_SINGLE_LABEL

    def __init__(
        self,
        model: str = DEFAULT_CHAT_MODEL,
        *,
        n_examples: int = 3,
        vectorizer: Vectorizer | None = None,
        default_label: str | None = None,
        n_predictors: int | None = None,
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
            default_label=default_label,
            n_predictors=n_predictors,
            prompt_template=prompt_template,
            model_params=model_params,
            api_key=api_key,
            base_url=base_url,
            alias=alias,
            on_error=on_error,
            max_concurrency=max_concurrency,
            verbose=verbose,
        )
        self.n_examples = n_examples
        self.vectorizer = vectorizer

    def _check_params(self) -> None:
        super()._check_params()
        check_count("n_examples", self.n_examples, required=True)
        if self.vectorizer is not None and not isinstance(self.vectorizer, Vectorizer):
            raise TypeError(
                "vectorizer must be a skllm Vectorizer or None, got "
                f"{type(self.vectorizer).__name__}."
            )

    def _fit_examples(self, X: Any, rows: Sequence[Any]) -> None:
        examples = _paired(X, rows)
        vectorizer = (
            self._default_vectorizer()
            if self.vectorizer is None
            else clone(self.vectorizer)
        )
        vectors = vectorizer.transform([text for text, _ in examples])
        self.vectorizer_ = vectorizer
        self.indexes_ = _class_indexes(examples, vectors)

    def _demonstrations(self, texts: Sequence[str]) -> list[Sequence[tuple[str, Any]]]:
        """Embed the whole batch once, then retrieve per class for each sample."""
        return [self._nearest(query) for query in self.vectorizer_.transform(texts)]

    def _nearest(self, query: np.ndarray) -> list[tuple[str, str]]:
        examples: list[tuple[str, str]] = []
        for label, index in self.indexes_.items():
            count = min(self.n_examples, len(index.texts))
            positions = index.neighbors.kneighbors(
                [query], n_neighbors=count, return_distance=False
            )
            examples.extend((index.texts[position], label) for position in positions[0])
        return examples

    def _default_vectorizer(self) -> Vectorizer:
        """The chat provider's own embedding model, where it has a default one.

        Connection config carries over: the fit-created vectorizer talks to the
        same provider as the classifier, so it must authenticate as it does —
        which also keeps a bundle's declared credentials down to one per key.
        """
        provider = parse_model(self.model).provider
        model = DEFAULT_EMBEDDING_MODELS.get(provider)
        if model is None:
            raise ValueError(
                f"{type(self).__name__} retrieves its examples by embedding them, "
                f"and there is no default embedding model for the {provider!r} "
                "provider (anthropic and typesafe offer no embeddings; for ollama, "
                "litellm and custom backends the model id depends on the "
                "deployment). Pass one explicitly, e.g. vectorizer=Vectorizer(model="
                f"{DEFAULT_EMBEDDING_MODEL!r})."
            )
        return Vectorizer(
            model=model,
            api_key=self.api_key,
            base_url=self.base_url,
            alias=self.alias,
            max_concurrency=self.max_concurrency,
            verbose=self.verbose,
        )


class MultiLabelZeroShotClassifier(_MultiLabelClassifier):
    """Assign every applicable label to each text, without examples."""

    prompt_placeholders: ClassVar[frozenset[str]] = frozenset(
        {"labels", "text", "max_labels"}
    )
    default_prompt_template: ClassVar[str] = ZERO_SHOT_MULTI_LABEL


class MultiLabelFewShotClassifier(_FewShotMixin, _MultiLabelClassifier):
    """Assign every applicable label per text, demonstrated by the training set."""

    prompt_placeholders: ClassVar[frozenset[str]] = frozenset(
        {"labels", "text", "max_labels", "examples"}
    )
    default_prompt_template: ClassVar[str] = FEW_SHOT_MULTI_LABEL


class _ClassIndex(NamedTuple):
    """One class's demonstration texts and a neighbor index over their vectors."""

    texts: tuple[str, ...]
    neighbors: NearestNeighbors


def _class_indexes(
    examples: Sequence[tuple[str, Any]], vectors: np.ndarray
) -> dict[str, _ClassIndex]:
    """Index each class separately, so every class contributes its own examples."""
    positions: dict[str, list[int]] = {}
    for position, (_, label) in enumerate(examples):
        positions.setdefault(label, []).append(position)
    return {
        label: _ClassIndex(
            texts=tuple(examples[position][0] for position in rows),
            # Cosine ranks embeddings by direction, which is what their
            # similarity means; magnitude varies with text length.
            neighbors=NearestNeighbors(metric="cosine").fit(vectors[rows]),
        )
        for label, rows in sorted(positions.items())
    }


def _paired(X: Any, rows: Sequence[Any]) -> tuple[tuple[str, Any], ...]:
    texts = as_texts(X)
    if len(texts) != len(rows):
        raise ValueError(
            f"X has {len(texts)} samples but y has {len(rows)}: few-shot "
            "classifiers learn from labeled examples, so both must match."
        )
    return tuple(zip(texts, rows, strict=True))


def _texts_per_label(examples: Sequence[tuple[str, Any]]) -> dict[str, list[str]]:
    """Group demonstration texts under every label they carry."""
    texts: dict[str, list[str]] = {}
    for text, labels in examples:
        for label in (labels,) if isinstance(labels, str) else labels:
            texts.setdefault(label, []).append(text)
    return texts


def _demonstrated(texts: Sequence[str] | None) -> dict[str, list[str]] | None:
    """A System One option description: its examples, or none at all."""
    return {"examples": list(texts)} if texts else None


def _object_schema(name: str, value: Mapping[str, Any]) -> dict[str, Any]:
    """A single-property JSON Schema object, in the strictest common dialect."""
    return {
        "type": "object",
        "properties": {name: dict(value)},
        "required": [name],
        "additionalProperties": False,
    }


def _check_labels(y: Any, at: str = "y") -> list[str]:
    labels = []
    for index, label in enumerate(y):
        if not isinstance(label, str):
            raise TypeError(
                f"{at}[{index}] is a {type(label).__name__}, but labels must be "
                "strings: cast them first, e.g. [str(label) for label in y]. "
                "Non-string labels do not survive the prompt and JSON round-trip "
                "faithfully."
            )
        labels.append(str(label))  # a numpy str_ is a str, but not a plain one
    return labels


def _distinct(rows: Sequence[Any]) -> list[str]:
    labels: set[str] = set()
    for row in rows:
        labels.update((row,) if isinstance(row, str) else row)
    if not labels:
        raise ValueError("y holds no labels: a classifier needs at least one to fit.")
    return sorted(labels)


def _quote(answer: Any) -> str:
    return repr(str(answer)[:_QUOTED_ANSWER_CHARS])
