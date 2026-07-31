import importlib
from pathlib import Path

import pytest

import skllm
from skllm.datasets import (
    get_classification_dataset,
    get_multilabel_classification_dataset,
    get_summarization_dataset,
    get_translation_dataset,
)

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

V2_MODULES = [
    "skllm.backends",
    "skllm.classification",
    "skllm.clustering",
    "skllm.config",
    "skllm.core",
    "skllm.datasets",
    "skllm.ensemble",
    "skllm.extraction",
    "skllm.prompts",
    "skllm.scoring",
    "skllm.serialization",
    "skllm.tagging",
    "skllm.text2text",
    "skllm.vectorization",
]

REMOVED_V1_MODULES = [
    "skllm.llm",
    "skllm.memory",
    "skllm.model_constants",
    "skllm.models",
]


@pytest.mark.parametrize("module", V2_MODULES)
def test_v2_module_is_importable(module: str) -> None:
    importlib.import_module(module)


@pytest.mark.parametrize("module", REMOVED_V1_MODULES)
def test_v1_internals_are_gone(module: str) -> None:
    with pytest.raises(ImportError):
        importlib.import_module(module)


def test_version_is_v2() -> None:
    assert skllm.__version__.split(".")[0] == "2"


def test_pyproject_takes_version_from_the_package() -> None:
    tomllib = pytest.importorskip("tomllib")
    pyproject = tomllib.loads(PYPROJECT.read_text())

    assert "version" not in pyproject["project"]
    assert "version" in pyproject["project"]["dynamic"]
    assert (
        pyproject["tool"]["setuptools"]["dynamic"]["version"]["attr"]
        == "skllm.__version__"
    )


def test_demo_datasets_are_retained() -> None:
    for get_labeled_dataset in (
        get_classification_dataset,
        get_multilabel_classification_dataset,
    ):
        X, y = get_labeled_dataset()
        assert len(X) == len(y) > 0

    for get_unlabeled_dataset in (get_summarization_dataset, get_translation_dataset):
        assert len(get_unlabeled_dataset()) > 0
