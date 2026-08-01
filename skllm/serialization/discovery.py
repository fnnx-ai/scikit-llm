"""Finding the skllm estimators inside an arbitrary sklearn object graph.

Discovery is two-level and read-only. A generic walk locates the skllm
estimators held anywhere in a foreign structure — a pipeline, a meta-estimator,
a plain container — and each of them then reports its own credential
requirements, recursing through skllm-native nesting on its own. Only skllm
classes know about skllm nesting, so the foreign walk never has to.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, NamedTuple, TypeVar

from sklearn.base import BaseEstimator

from skllm.backends import parse_model
from skllm.core import BaseLLMEstimator, Requirement

__all__ = ["ModelUse", "components", "find_estimators", "model_uses", "requirements"]

_T = TypeVar("_T")


class ModelUse(NamedTuple):
    """One model a bundle talks to, for the bundle's identity metadata."""

    model: str
    provider: str
    alias: str | None


def find_estimators(obj: Any) -> list[BaseLLMEstimator]:
    """Every skllm estimator reachable from ``obj`` through foreign structures.

    Recursion stops at a skllm estimator: what it holds is its own business, and
    it reports that through :meth:`BaseLLMEstimator.llm_components`.
    """
    found: list[BaseLLMEstimator] = []
    _walk(obj, found, set())
    return found


def components(obj: Any) -> tuple[BaseLLMEstimator, ...]:
    """Every skllm estimator in the graph, skllm-native nesting included."""
    return _unique(
        component
        for estimator in find_estimators(obj)
        for component in estimator.llm_components()
    )


def requirements(obj: Any) -> tuple[Requirement, ...]:
    """The provider/alias pairs the graph needs credentials for, deduplicated."""
    return _unique(
        requirement
        for estimator in find_estimators(obj)
        for requirement in estimator.llm_requirements()
    )


def model_uses(obj: Any) -> tuple[ModelUse, ...]:
    """The distinct models the graph talks to, in discovery order.

    Estimators that name no model of their own — an ensemble votes, its members
    call — contribute nothing but are walked through all the same.
    """
    return _unique(
        ModelUse(
            component.model, parse_model(component.model).provider, component.alias
        )
        for component in components(obj)
        if getattr(component, "model", None)
    )


def _walk(value: Any, found: list[BaseLLMEstimator], seen: set[int]) -> None:
    if id(value) in seen:
        return
    seen.add(id(value))

    if isinstance(value, BaseLLMEstimator):
        found.append(value)
    elif isinstance(value, BaseEstimator):
        for param in value.get_params(deep=False).values():
            _walk(param, found, seen)
    elif isinstance(value, Mapping):
        for item in value.values():
            _walk(item, found, seen)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            _walk(item, found, seen)


def _unique(items: Iterable[_T]) -> tuple[_T, ...]:
    return tuple(dict.fromkeys(items))
