"""What a bundle declares it takes in and hands back.

Typing is edge-based and uniform: the step that sees the input decides the input
schema, the step that produces the answer decides the output schema, and a bare
estimator is the degenerate case where both are the same object. Each edge is
typed from what it actually is — a skllm estimator by task and fitted state, a
foreign sklearn step by convention, a permissive array only where neither says
anything.

The derived *kinds* travel in the bundle next to the schemas, so the loader
turning JSON answers back into sklearn types reads the same derivation the
manifest published — which is also why the conversions live here, beside the
schemas they have to agree with.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, is_classifier, is_regressor

from skllm.classification import _MultiLabelClassifier
from skllm.clustering import ClusterLabeler, as_cluster_ids
from skllm.core import BaseLLMEstimator, as_texts
from skllm.extraction import StructuredExtractor
from skllm.tagging import NER
from skllm.vectorization import Vectorizer

__all__ = [
    "decode_output",
    "edges",
    "encode_input",
    "entry_kind",
    "exit_kind",
    "input_schema",
    "output_schema",
    "task_tag",
]

_TASK_BY_EXIT_KIND = {
    "labels": "classification",
    "label_lists": "classification",
    "scores": "scoring",
    "vectors": "vectorization",
    "records": "extraction",
    "names": "clustering",
    "texts": "text2text",
    "unknown": "unknown",
}

_ANY_ARRAY: dict[str, Any] = {"type": "array"}


def edges(obj: Any) -> tuple[Any, Any]:
    """The steps that see a model's input and produce its answer."""
    return _entry(obj), _exit(obj)


def entry_kind(step: Any) -> str:
    """What the entry step consumes: ``texts``, ``cluster_ids`` or ``unknown``."""
    if isinstance(step, ClusterLabeler):
        return "cluster_ids"
    return "texts" if isinstance(step, BaseLLMEstimator) else "unknown"


def exit_kind(step: Any) -> str:
    """The shape of the answer the exit step produces."""
    if isinstance(step, _MultiLabelClassifier):
        return "label_lists"
    # Guarded: these read sklearn tags, which a pipeline's bare "passthrough"
    # step — a string, not an estimator — does not have.
    if isinstance(step, BaseEstimator):
        if is_classifier(step):
            return "labels"
        if is_regressor(step):
            return "scores"
    if isinstance(step, StructuredExtractor):
        return "records"
    if isinstance(step, ClusterLabeler):
        return "names"
    if isinstance(step, BaseLLMEstimator):
        return "vectors" if isinstance(step, Vectorizer) else "texts"
    # A foreign step gets the convention that fits it: a transformer answers
    # with vectors, but one that predicts something neither classified nor
    # scored — a clusterer — has no shape to declare beyond the permissive one.
    if hasattr(step, "predict"):
        return "unknown"
    return "vectors" if hasattr(step, "transform") else "unknown"


def task_tag(step: Any) -> str:
    """The task a bundle is published under, from what its exit step does."""
    kind = exit_kind(step)
    if kind == "texts":
        return "tagging" if isinstance(step, NER) else "text2text"
    return _TASK_BY_EXIT_KIND[kind]


def input_schema(kind: str) -> dict[str, Any]:
    if kind == "texts":
        return {"type": "array", "items": {"type": "string"}}
    if kind == "cluster_ids":
        return {"type": "array", "items": {"type": "integer"}}
    return dict(_ANY_ARRAY)


def output_schema(kind: str, step: Any) -> dict[str, Any]:
    """The answer schema for one exit step, as precise as the step allows."""
    if kind == "labels":
        return _array_of(_nullable(_enum(_labels(step))))
    if kind == "label_lists":
        return _array_of({"type": "array", "items": _enum(_labels(step))})
    if kind == "texts":
        return _array_of({"type": "string"})
    if kind == "scores":
        return _array_of(_nullable({"type": "number"}))
    if kind == "vectors":
        return _array_of({"type": "array", "items": {"type": "number"}})
    if kind == "names":
        return _array_of(_nullable({"type": "string"}))
    if kind == "records":
        return _array_of(_nullable(step._json_schema()))
    return dict(_ANY_ARRAY)


def encode_input(kind: str, X: Any) -> list[Any]:
    """A batch as the JSON array the bundle's input channel declares.

    Validation is the estimators' own, so a batch rejected at the bundle's edge
    is rejected in the same words the live object would have used.
    """
    if kind == "texts":
        return as_texts(X)
    if kind == "cluster_ids":
        return as_cluster_ids(X)
    return X.tolist() if hasattr(X, "tolist") else list(X)


def decode_output(kind: str, values: Sequence[Any]) -> Any:
    """A JSON answer as the container the live object would have returned."""
    return _DECODERS.get(kind, list)(values)


def _object_array(values: Sequence[Any]) -> np.ndarray:
    """Labels, texts and names alike: object dtype is what carries a null fill."""
    return np.asarray(values, dtype=object)


def _label_lists(values: Sequence[Any]) -> list[list[str]]:
    return [list(labels) for labels in values]


def _scores(values: Sequence[Any]) -> np.ndarray:
    """JSON has no NaN, so a filled sample arrives as null."""
    return np.asarray(
        [np.nan if value is None else value for value in values], dtype=float
    )


def _vectors(values: Sequence[Any]) -> np.ndarray:
    # An empty batch of embeddings is two-dimensional — the shape the Vectorizer
    # itself returns — where an empty JSON array would decode to a flat one.
    if not values:
        return np.empty((0, 0), dtype=float)
    return np.asarray(values, dtype=float)


_DECODERS: dict[str, Callable[[Sequence[Any]], Any]] = {
    "labels": _object_array,
    "label_lists": _label_lists,
    "texts": _object_array,
    "names": _object_array,
    "scores": _scores,
    "vectors": _vectors,
    "records": list,
}


def _entry(obj: Any) -> Any:
    steps = _steps(obj)
    return _entry(steps[0]) if steps else obj


def _exit(obj: Any) -> Any:
    steps = _steps(obj)
    return _exit(steps[-1]) if steps else obj


def _steps(obj: Any) -> list[Any]:
    """The ordered inner estimators of a composite; empty for anything else."""
    steps = getattr(obj, "steps", None)
    return [step for _, step in steps] if steps else []


def _labels(step: Any) -> list[Any]:
    """The values a fitted classifier can answer with, as JSON scalars.

    Empty where they are not scalars at all: a multi-output classifier answers
    with a vector per sample and its ``classes_`` is one set per output, which
    no enum over labels describes.
    """
    classes = getattr(step, "classes_", None)
    values = [] if classes is None else [_scalar(value) for value in classes]
    if any(_json_type(value) is None for value in values):
        return []
    fill = getattr(step, "default_label", None)
    if isinstance(fill, str) and fill not in values:
        values.append(fill)
    return values


def _scalar(value: Any) -> Any:
    """A numpy label as the plain Python value a JSON schema can carry."""
    return value.item() if getattr(value, "ndim", None) == 0 else value


def _enum(values: list[Any]) -> dict[str, Any]:
    """An enum over known values, staying permissive when there are none to name."""
    if not values:
        return {}
    types = {_json_type(value) for value in values}
    declared = types.pop() if len(types) == 1 else None
    return {"enum": values} if declared is None else {"type": declared, "enum": values}


def _json_type(value: Any) -> str | None:
    if isinstance(value, str):
        return "string"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    return "number" if isinstance(value, float) else None


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    """Allow the null a filled sample serializes to."""
    return {"anyOf": [schema, {"type": "null"}]}


def _array_of(items: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": items}
