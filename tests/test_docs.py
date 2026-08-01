"""Keep README.md and MIGRATION.md honest about the code they document.

Documentation rots silently: a renamed parameter or a dropped export leaves the
examples looking plausible and failing only for a reader. So every name and
keyword the docs show is resolved against the live package here — with the one
exception marked in the source, the v1 snippets in the migration guide, which
document code that is deliberately gone.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
MIGRATION = ROOT / "MIGRATION.md"

#: The distributions whose names the docs are checked against: scikit-llm's own
#: and its serialization format's. Illustrative sklearn/pydantic imports are the
#: reader's problem, not ours.
CHECKED_ROOTS = ("skllm", "fnnx")

#: The first line marking a code block as v1 archaeology, exempt from checking.
V1_MARKER = "# v1"

_PYTHON_BLOCK = re.compile(r"^```python\n(.*?)^```", re.DOTALL | re.MULTILINE)
_BACKTICKED = re.compile(r"`([^`]+)`")
_DOTTED_PATH = re.compile(r"skllm\.[A-Za-z_][A-Za-z0-9_.]*")


def python_blocks(document: Path) -> list[str]:
    return [
        block
        for block in _PYTHON_BLOCK.findall(document.read_text())
        if not block.startswith(V1_MARKER)
    ]


def documented_paths(text: str) -> list[str]:
    """The dotted skllm names a passage claims exist, from its inline code."""
    return [
        path
        for span in _BACKTICKED.findall(text)
        for path in _DOTTED_PATH.findall(span)
    ]


def resolve(dotted: str) -> Any:
    """The object a dotted path names, importing the longest module prefix."""
    parts = dotted.split(".")
    for depth in range(len(parts), 0, -1):
        try:
            resolved: Any = importlib.import_module(".".join(parts[:depth]))
        except ImportError:
            continue
        for name in parts[depth:]:
            resolved = getattr(resolved, name)
        return resolved
    raise ModuleNotFoundError(dotted)


def bound_names(tree: ast.Module) -> dict[str, Any]:
    """What the block's imports of the checked packages bind, resolved."""
    bound: dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if not node.module or not node.module.startswith(CHECKED_ROOTS):
                continue
            for alias in node.names:
                bound[alias.asname or alias.name] = resolve(
                    f"{node.module}.{alias.name}"
                )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(CHECKED_ROOTS):
                    name = alias.asname or alias.name.split(".")[0]
                    bound[name] = importlib.import_module(
                        alias.name if alias.asname else name
                    )
    return bound


def called_object(func: ast.expr, bound: dict[str, Any]) -> Any:
    """The documented object a call targets, or None when it is someone else's."""
    if isinstance(func, ast.Name):
        return bound.get(func.id)
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        module = bound.get(func.value.id)
        return getattr(module, func.attr, None) if inspect.ismodule(module) else None
    return None


def unknown_keywords(target: Any, call: ast.Call) -> list[str]:
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):  # not introspectable: nothing to check against
        return []
    parameters = signature.parameters
    if any(
        parameter.kind is parameter.VAR_KEYWORD for parameter in parameters.values()
    ):
        return []
    return [
        keyword.arg
        for keyword in call.keywords
        if keyword.arg is not None and keyword.arg not in parameters
    ]


def documents() -> list[tuple[str, str]]:
    return [
        (document.name, block)
        for document in (README, MIGRATION)
        for block in python_blocks(document)
    ]


@pytest.mark.parametrize(("document", "block"), documents())
def test_documented_examples_are_valid_python(document: str, block: str) -> None:
    ast.parse(block)


@pytest.mark.parametrize(("document", "block"), documents())
def test_documented_examples_import_names_that_exist(document: str, block: str) -> None:
    bound_names(ast.parse(block))


@pytest.mark.parametrize(("document", "block"), documents())
def test_documented_examples_pass_parameters_that_exist(
    document: str, block: str
) -> None:
    tree = ast.parse(block)
    bound = bound_names(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = called_object(node.func, bound)
        if target is None:
            continue
        assert not unknown_keywords(target, node), (
            f"{document} calls {getattr(target, '__name__', target)} with parameters "
            f"it does not take: {unknown_keywords(target, node)}"
        )


def test_readme_mentions_only_names_that_exist() -> None:
    for path in documented_paths(README.read_text()):
        resolve(path)


def test_migration_targets_exist() -> None:
    """Every v2 replacement the guide points at resolves."""
    for row in _mapping_rows():
        for path in documented_paths(row[1]):
            resolve(path)


def test_migration_does_not_claim_a_living_name_is_gone() -> None:
    """Nothing in the v1 column survived — otherwise the guide is misleading."""
    for row in _mapping_rows():
        for path in documented_paths(row[0]):
            with pytest.raises((ImportError, AttributeError)):
                resolve(path)


def _mapping_rows() -> list[list[str]]:
    """The v1 → v2 rows of the migration tables, cells left as written."""
    rows = []
    for line in MIGRATION.read_text().splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        separator = all(set(cell) <= set("-: ") for cell in cells)
        if separator or cells[:2] == ["v1", "v2"] or len(cells) < 2:
            continue
        rows.append(cells)
    return rows
