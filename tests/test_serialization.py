"""Packaging a fitted graph as an FNNX bundle, and running that bundle.

The bundles here are executed through ``fnnx.Runtime`` exactly as a deployment
would, with the fake adapter standing in for the providers — so the manifest,
the wrapper's attribute contract and the estimators all meet on the real path.
"""

from __future__ import annotations

import ast
import asyncio
import json
import pickle
import sys
import tarfile
import time
from collections.abc import Callable, Iterator
from typing import Any

import joblib
import numpy as np
import pytest
from fnnx.runtime import Runtime
from pydantic import BaseModel
from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

import skllm
from skllm import (
    NER,
    ClusterLabeler,
    ConsensusClassifier,
    DynamicFewShotClassifier,
    FewShotClassifier,
    MultiLabelZeroShotClassifier,
    RubricScorer,
    StructuredExtractor,
    Summarizer,
    Vectorizer,
    ZeroShotClassifier,
)
from skllm.backends import register_backend, registry
from skllm.config import MissingCredentialError, config_context
from skllm.core.base import STATE_VERSION, STATE_VERSION_KEY
from skllm.serialization.save import SECRET_TAGS
from tests.fakes import AnthropicFake, FakeBackend, FakeCall, OllamaFake, OpenAIFake

REVIEWS = ["I love it", "It broke on day one"]
SENTIMENTS = ["negative", "positive"]


class Movie(BaseModel):
    name: str
    year: int


@pytest.fixture
def openai(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeBackend]:
    """Answer ``openai::`` with the fake, with a key ambiently available."""
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    yield from _registered("openai", OpenAIFake(structured={"label": "positive"}))


@pytest.fixture
def anthropic() -> Iterator[FakeBackend]:
    yield from _registered("anthropic", AnthropicFake(structured={"label": "positive"}))


@pytest.fixture
def ollama() -> Iterator[FakeBackend]:
    yield from _registered("ollama", OllamaFake(structured={"label": "positive"}))


@pytest.fixture
def bundle(tmp_path: Any) -> Callable[..., str]:
    def save(estimator: Any, name: str = "model.fnnx") -> str:
        path = str(tmp_path / name)
        skllm.save(estimator, path)
        return path

    return save


def _registered(prefix: str, backend: FakeBackend) -> Iterator[FakeBackend]:
    register_backend(prefix, backend)
    try:
        yield backend
    finally:
        registry._BACKENDS.pop(prefix, None)


def read(path: str, member: str) -> Any:
    with tarfile.open(path) as tar:
        payload = tar.extractfile(member)
        assert payload is not None
        return json.loads(payload.read())


def manifest(path: str) -> Any:
    return read(path, "manifest.json")


def attributes(path: str) -> dict[str, Any]:
    return {var["name"]: var for var in manifest(path)["dynamic_attributes"]}


def extra_values(path: str) -> Any:
    return read(path, "variant_config.json")["extra_values"]["skllm"]


def output_dtype(path: str) -> Any:
    return read(path, "dtypes.json")["ext::skllm::predictions"]


def input_dtype(path: str) -> Any:
    return read(path, "dtypes.json")["ext::skllm::X"]


def payload(path: str) -> Any:
    """The estimator graph the bundle carries, restored straight from the tar."""
    with tarfile.open(path) as tar:
        state = tar.extractfile("variant_artifacts/extra_files/estimator.joblib")
        assert state is not None
        return joblib.load(state)


def dependencies(path: str) -> list[str]:
    env = read(path, "env.json")["python3::conda_pip"]
    return [dependency["package"] for dependency in env["dependencies"]]


def fitted_classifier(**params: Any) -> ZeroShotClassifier:
    return ZeroShotClassifier(model="openai::gpt-5", verbose=0, **params).fit(
        None, SENTIMENTS
    )


# --- credentials are never written -------------------------------------------


def test_neither_a_bundle_nor_a_pickle_carries_the_key(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    classifier = fitted_classifier(api_key="sk-secret")

    with open(bundle(classifier), "rb") as saved:
        assert b"sk-secret" not in saved.read()
    assert b"sk-secret" not in pickle.dumps(classifier)


def test_a_restored_estimator_keeps_its_state_and_takes_a_new_key(
    openai: FakeBackend,
) -> None:
    classifier = FewShotClassifier(
        model="openai::gpt-5", api_key="sk-secret", verbose=0
    ).fit(REVIEWS, SENTIMENTS)

    restored = pickle.loads(pickle.dumps(classifier))

    assert restored.api_key is None
    assert list(restored.classes_) == SENTIMENTS
    assert restored.examples_ == classifier.examples_
    with config_context(openai_api_key="sk-new"):
        restored.predict(["I love it"])
    assert openai.calls[-1].api_key == "sk-new"


def test_state_is_versioned_and_a_newer_layout_is_refused(openai: FakeBackend) -> None:
    state = fitted_classifier().__getstate__()
    assert state[STATE_VERSION_KEY] == STATE_VERSION
    assert state["attributes"]["classes_"] is not None

    with pytest.raises(ValueError, match="version"):
        ZeroShotClassifier().__setstate__(
            {**state, STATE_VERSION_KEY: STATE_VERSION + 1}
        )
    with pytest.raises(ValueError, match="Unrecognized"):
        ZeroShotClassifier().__setstate__({"classes_": ["a"]})


# --- what a bundle declares ---------------------------------------------------


def test_a_classifier_bundle_declares_its_labels(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    path = bundle(fitted_classifier())

    assert input_dtype(path) == {"type": "array", "items": {"type": "string"}}
    assert output_dtype(path) == {
        "type": "array",
        "items": {"anyOf": [{"type": "string", "enum": SENTIMENTS}, {"type": "null"}]},
    }
    assert manifest(path)["producer_name"] == "skllm.fnnx.ai"
    assert manifest(path)["producer_tags"] == ["skllm.fnnx.ai::classification:v1"]


def test_a_string_default_label_joins_the_declared_labels(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    path = bundle(fitted_classifier(default_label="unknown"))

    assert output_dtype(path)["items"]["anyOf"][0]["enum"] == [*SENTIMENTS, "unknown"]


def test_an_extractor_bundle_publishes_the_extraction_schema(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    path = bundle(StructuredExtractor(model="openai::gpt-5", schema=Movie, verbose=0))

    assert output_dtype(path) == {
        "type": "array",
        "items": {"anyOf": [Movie.model_json_schema(), {"type": "null"}]},
    }
    assert extra_values(path)["exit_kind"] == "records"
    # The payload carries the compiled schema, so serving needs neither the
    # user's model class nor pydantic itself.
    assert payload(path).schema == Movie.model_json_schema()


def test_an_extractor_bundle_serves_where_pydantic_is_not_installed(
    openai: FakeBackend, bundle: Callable[..., str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """pydantic is a save-time dependency; the schema travels already compiled."""
    record = {"name": "Dune: Part Two", "year": 2024}
    openai.structured_script = record
    path = bundle(StructuredExtractor(model="openai::gpt-5", schema=Movie, verbose=0))

    # None in sys.modules fails an `import pydantic` the way an absent
    # distribution would — including for modules that already imported it.
    for name in [n for n in sys.modules if n.startswith("pydantic")]:
        monkeypatch.setitem(sys.modules, name, None)
    answer = Runtime(path).compute({"X": ["Dune: Part Two came out in 2024."]}, {})

    assert answer["predictions"] == [record]


def test_a_pipeline_types_its_edges_from_the_steps_that_see_them(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    pipeline = Pipeline(
        [
            ("embed", Vectorizer(model="openai::text-embedding-3-small", verbose=0)),
            ("classify", LogisticRegression()),
        ]
    ).fit(["good", "bad", "fine"], ["p", "n", "p"])

    path = bundle(pipeline)

    assert input_dtype(path) == {"type": "array", "items": {"type": "string"}}
    assert output_dtype(path) == {
        "type": "array",
        "items": {"anyOf": [{"type": "string", "enum": ["n", "p"]}, {"type": "null"}]},
    }
    assert [var["name"] for var in manifest(path)["env_vars"]] == ["OPENAI_API_KEY"]
    assert manifest(path)["env_vars"][0]["tags"] == SECRET_TAGS
    assert dependencies(path)[0] == f"scikit-llm=={skllm.__version__}"


def test_a_foreign_entry_step_falls_back_to_a_permissive_input(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    pipeline = Pipeline(
        [
            ("passthrough", FunctionTransformer()),
            ("classify", ZeroShotClassifier(model="openai::gpt-5", verbose=0)),
        ]
    ).fit(REVIEWS, SENTIMENTS)

    path = bundle(pipeline)

    assert input_dtype(path) == {"type": "array"}
    assert output_dtype(path)["items"]["anyOf"][0]["enum"] == SENTIMENTS
    assert Runtime(path).compute({"X": ["I love it"]}, {}) == {
        "predictions": ["positive"]
    }


def test_edge_typing_reaches_through_nested_composites(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    pipeline = Pipeline(
        [
            (
                "prepare",
                Pipeline(
                    [
                        (
                            "embed",
                            Vectorizer(
                                model="openai::text-embedding-3-small", verbose=0
                            ),
                        )
                    ]
                ),
            ),
            ("classify", LogisticRegression()),
        ]
    ).fit(["good", "bad", "fine"], ["p", "n", "p"])

    path = bundle(pipeline)

    assert input_dtype(path) == {"type": "array", "items": {"type": "string"}}
    assert output_dtype(path)["items"]["anyOf"][0]["enum"] == ["n", "p"]
    assert [var["name"] for var in manifest(path)["env_vars"]] == ["OPENAI_API_KEY"]


@pytest.mark.parametrize(
    ("estimator", "entry_kind", "exit_kind", "items"),
    [
        (
            MultiLabelZeroShotClassifier(model="openai::m", verbose=0).fit(
                None, [["price"], ["delivery"]]
            ),
            "texts",
            "label_lists",
            {
                "type": "array",
                "items": {"type": "string", "enum": ["delivery", "price"]},
            },
        ),
        (
            Summarizer(model="openai::m", verbose=0),
            "texts",
            "texts",
            {"type": "string"},
        ),
        (
            NER(model="openai::m", entities={"PER": "a person"}, verbose=0),
            "texts",
            "texts",
            {"type": "string"},
        ),
        (
            RubricScorer(model="openai::m", rubric="clarity", verbose=0),
            "texts",
            "scores",
            {"anyOf": [{"type": "number"}, {"type": "null"}]},
        ),
        (
            Vectorizer(model="openai::text-embedding-3-small", verbose=0),
            "texts",
            "vectors",
            {"type": "array", "items": {"type": "number"}},
        ),
    ],
)
def test_every_task_declares_its_own_answer_shape(
    openai: FakeBackend,
    bundle: Callable[..., str],
    estimator: Any,
    entry_kind: str,
    exit_kind: str,
    items: dict[str, Any],
) -> None:
    path = bundle(estimator)

    assert extra_values(path)["entry_kind"] == entry_kind
    assert extra_values(path)["exit_kind"] == exit_kind
    assert output_dtype(path)["items"] == items


@pytest.mark.parametrize(
    "tail",
    ["passthrough", KMeans(n_clusters=2, n_init=1, random_state=0)],
    ids=["passthrough", "clusterer"],
)
def test_an_exit_step_with_no_convention_declares_a_permissive_output(
    openai: FakeBackend, bundle: Callable[..., str], tail: Any
) -> None:
    """Neither a disabled step nor a clusterer has an answer shape to publish.

    The clusterer is the case that matters: the pipeline answers through
    ``predict``, so typing it by the transformer convention would declare
    embeddings where cluster ids come out.
    """
    pipeline = Pipeline(
        [
            ("embed", Vectorizer(model="openai::text-embedding-3-small", verbose=0)),
            ("tail", tail),
        ]
    ).fit(["good", "bad", "fine"])

    path = bundle(pipeline)

    assert output_dtype(path) == {"type": "array"}
    assert extra_values(path)["exit_kind"] == "unknown"


def test_a_multi_output_classifier_declares_no_label_enum(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    """Its ``classes_`` is one label set per output, which no enum describes."""
    pipeline = Pipeline(
        [
            ("embed", Vectorizer(model="openai::text-embedding-3-small", verbose=0)),
            ("classify", MultiOutputClassifier(LogisticRegression())),
        ]
    ).fit(["good", "bad", "fine"], np.array([[0, 1], [1, 0], [0, 1]]))

    path = bundle(pipeline)

    assert output_dtype(path)["items"] == {"anyOf": [{}, {"type": "null"}]}


def test_a_bundle_records_what_it_is_made_of(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    values = extra_values(bundle(fitted_classifier(model_params={"temperature": 0})))

    assert values["estimator"] == "skllm.classification.ZeroShotClassifier"
    assert values["version"] == skllm.__version__
    assert values["kind"] == "single"
    assert values["method"] == "predict"
    assert values["task"] == "classification"
    assert values["models"] == [
        {"model": "openai::gpt-5", "provider": "openai", "alias": None}
    ]


def test_a_transformer_bundle_computes_through_transform(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    assert extra_values(bundle(Summarizer(verbose=0)))["method"] == "transform"


# --- which credentials a bundle asks for --------------------------------------


def test_every_provider_in_the_graph_declares_its_own_key(
    openai: FakeBackend, anthropic: FakeBackend, bundle: Callable[..., str]
) -> None:
    ensemble = ConsensusClassifier(
        [
            ZeroShotClassifier(model="openai::gpt-5", verbose=0),
            ZeroShotClassifier(model="anthropic::claude-sonnet-5", verbose=0),
        ],
        verbose=0,
    ).fit(REVIEWS, SENTIMENTS)

    path = bundle(ensemble)
    declared = attributes(path)

    assert [var["name"] for var in manifest(path)["env_vars"]] == [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ]
    assert declared["openai_api_key"]["tags"] == SECRET_TAGS
    assert declared["anthropic_api_key"]["tags"] == SECRET_TAGS
    assert extra_values(path)["kind"] == "composite"

    Runtime(path).compute(
        {"X": ["I love it"]},
        {"openai_api_key": "sk-openai", "anthropic_api_key": "sk-anthropic"},
    )
    assert openai.calls[-1].api_key == "sk-openai"
    assert anthropic.calls[-1].api_key == "sk-anthropic"


def test_aliased_members_are_credentialed_by_name_only(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    ensemble = ConsensusClassifier(
        [
            ZeroShotClassifier(model="openai::gpt-5", alias="team_a", verbose=0),
            ZeroShotClassifier(model="openai::gpt-5-mini", alias="team_b", verbose=0),
        ],
        verbose=0,
    ).fit(REVIEWS, SENTIMENTS)

    path = bundle(ensemble)
    declared = attributes(path)

    assert "openai_api_key" not in declared
    assert declared["credential_team_a"]["tags"] == SECRET_TAGS
    assert [var["name"] for var in manifest(path)["env_vars"]] == [
        "SKLLM_CREDENTIAL_TEAM_A",
        "SKLLM_CREDENTIAL_TEAM_B",
    ]

    runtime = Runtime(path)
    runtime.compute(
        {"X": ["I love it"]},
        {"credential_team_a": "sk-a", "credential_team_b": "sk-b"},
    )
    assert {call.model: call.api_key for call in openai.calls[-2:]} == {
        "gpt-5": "sk-a",
        "gpt-5-mini": "sk-b",
    }

    with pytest.raises(MissingCredentialError, match="team_b"):
        runtime.compute({"X": ["I love it"]}, {"credential_team_a": "sk-a"})


def test_a_fit_created_component_is_discovered_and_deduplicated(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    classifier = DynamicFewShotClassifier(
        model="openai::gpt-5", n_examples=1, verbose=0
    ).fit(REVIEWS, SENTIMENTS)

    path = bundle(classifier)

    assert classifier.vectorizer_ is not None
    assert [var["name"] for var in manifest(path)["env_vars"]] == ["OPENAI_API_KEY"]
    assert list(attributes(path)) == [
        "openai_api_key",
        "openai_base_url",
        "overrides",
    ]


def test_an_estimator_that_only_calls_at_fit_declares_no_credentials(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    openai.chat_script = "Praise"
    labeler = ClusterLabeler(model="openai::gpt-5", verbose=0).fit(
        ["great", "awful", "fine"], [0, 1, -1]
    )

    path = bundle(labeler)

    assert manifest(path)["env_vars"] == []
    assert list(attributes(path)) == ["overrides"]
    assert input_dtype(path) == {"type": "array", "items": {"type": "integer"}}
    assert Runtime(path).compute({"X": [0, 1, -1]}, {}) == {
        "predictions": ["Praise", "Praise", None]
    }


def test_a_keyless_provider_declares_only_its_base_url(
    ollama: FakeBackend, bundle: Callable[..., str]
) -> None:
    classifier = ZeroShotClassifier(model="ollama::llama3", verbose=0).fit(
        None, SENTIMENTS
    )

    path = bundle(classifier)

    assert manifest(path)["env_vars"] == []
    assert list(attributes(path)) == ["ollama_base_url", "overrides"]
    assert Runtime(path).compute({"X": ["I love it"]}, {}) == {
        "predictions": ["positive"]
    }


def test_a_provider_extra_travels_with_the_bundle(
    anthropic: FakeBackend, bundle: Callable[..., str]
) -> None:
    classifier = ZeroShotClassifier(model="anthropic::claude-sonnet-5", verbose=0).fit(
        None, SENTIMENTS
    )

    assert dependencies(bundle(classifier))[0] == (
        f"scikit-llm[anthropic]=={skllm.__version__}"
    )


def test_a_provider_called_only_at_fit_still_pins_its_extra(
    anthropic: FakeBackend, bundle: Callable[..., str]
) -> None:
    """It needs no credential to serve, but it does need its adapter to import.

    Every ``transform`` re-validates the model string, which resolves the
    backend — so an environment built from ``env.json`` without the extra dies
    on a bundle that declares no credentials at all.
    """
    with config_context(anthropic_api_key="sk-fit"):
        labeler = ClusterLabeler(model="anthropic::claude-sonnet-5", verbose=0).fit(
            ["great", "awful"], [0, 1]
        )

    path = bundle(labeler)

    assert manifest(path)["env_vars"] == []
    assert dependencies(path)[0] == f"scikit-llm[anthropic]=={skllm.__version__}"


def test_an_aliased_estimator_on_a_keyless_provider_declares_no_secret(
    ollama: FakeBackend, bundle: Callable[..., str]
) -> None:
    """Routing a credential by name does not invent a credential to route."""
    classifier = ZeroShotClassifier(
        model="ollama::llama3", alias="local", verbose=0
    ).fit(None, SENTIMENTS)

    path = bundle(classifier)

    assert "credential_local" not in attributes(path)
    assert manifest(path)["env_vars"] == []
    assert Runtime(path).compute({"X": ["I love it"]}, {}) == {
        "predictions": ["positive"]
    }


def test_an_alias_is_declared_once_however_many_providers_share_it(
    openai: FakeBackend, anthropic: FakeBackend, bundle: Callable[..., str]
) -> None:
    ensemble = ConsensusClassifier(
        [
            ZeroShotClassifier(model="openai::gpt-5", alias="shared", verbose=0),
            ZeroShotClassifier(
                model="anthropic::claude-sonnet-5", alias="shared", verbose=0
            ),
        ],
        verbose=0,
    ).fit(REVIEWS, SENTIMENTS)

    path = bundle(ensemble)

    assert [var["name"] for var in manifest(path)["env_vars"]] == [
        "SKLLM_CREDENTIAL_SHARED"
    ]
    Runtime(path).compute({"X": ["I love it"]}, {"credential_shared": "sk-shared"})
    assert {call.api_key for call in openai.calls[-1:] + anthropic.calls[-1:]} == {
        "sk-shared"
    }


# --- running the bundle -------------------------------------------------------


def test_the_shipped_wrapper_needs_nothing_but_fnnx_to_import(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    """The bundle's entry point is loaded before its environment is proven."""
    with tarfile.open(bundle(fitted_classifier())) as tar:
        source = tar.extractfile("variant_artifacts/__pyfunc__.py")
        assert source is not None
        module = ast.parse(source.read().decode())

    imported = {
        (alias.name if isinstance(node, ast.Import) else node.module or "").split(".")[
            0
        ]
        for node in module.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert imported - set(sys.stdlib_module_names) == {"fnnx"}


def test_a_bundle_answers_what_the_live_estimator_would(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    classifier = fitted_classifier()
    live = classifier.predict(["some text"])

    answer = Runtime(bundle(classifier)).compute({"X": ["some text"]}, {})

    assert answer["predictions"] == list(live)
    assert openai.calls[-1].api_key == "env-key"


def test_a_pipeline_answers_under_the_runtime_what_it_answers_live(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    """The skllm step embeds the new text, the foreign step classifies it."""
    pipeline = Pipeline(
        [
            ("embed", Vectorizer(model="openai::text-embedding-3-small", verbose=0)),
            ("classify", LogisticRegression()),
        ]
    ).fit(["good", "bad", "fine"], ["p", "n", "p"])
    live = pipeline.predict(["some text"])

    answer = Runtime(bundle(pipeline)).compute({"X": ["some text"]}, {})

    assert answer["predictions"] == list(live)
    assert openai.calls[-1].api_key == "env-key"


def test_ragged_and_numeric_answers_survive_the_json_channel(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    openai.structured_script = {"labels": ["price"]}
    multi_label = MultiLabelZeroShotClassifier(model="openai::gpt-5", verbose=0).fit(
        None, [["price", "quality"], ["delivery"]]
    )
    vectorizer = Vectorizer(model="openai::text-embedding-3-small", verbose=0)

    labels = Runtime(bundle(multi_label, "labels.fnnx")).compute({"X": ["a", "b"]}, {})
    vectors = Runtime(bundle(vectorizer, "vectors.fnnx")).compute({"X": ["a"]}, {})

    assert labels["predictions"] == [["price"], ["price"]]
    assert np.allclose(vectors["predictions"], vectorizer.transform(["a"]))


def test_a_failed_sample_serializes_as_null(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    openai.structured_script = lambda call: (
        RuntimeError("nope") if "bad" in call.prompt else {"score": 7}
    )
    scorer = RubricScorer(model="openai::gpt-5", rubric="clarity", verbose=0)

    with pytest.warns(UserWarning, match="1 of 2"):
        answer = Runtime(bundle(scorer)).compute({"X": ["good", "bad"]}, {})

    assert answer["predictions"] == [7.0, None]


def test_defaults_hold_until_a_call_overrides_them(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    runtime = Runtime(bundle(fitted_classifier(model_params={"temperature": 0})))

    runtime.compute({"X": ["some text"]}, {})
    assert openai.calls[-1].params == {"temperature": 0}
    assert openai.calls[-1].api_key == "env-key"

    runtime.compute(
        {"X": ["some text"]}, {"temperature": 0.7, "openai_api_key": "sk-1"}
    )
    assert openai.calls[-1].params == {"temperature": 0.7}
    assert openai.calls[-1].api_key == "sk-1"

    runtime.compute({"X": ["some text"]}, {})
    assert openai.calls[-1].params == {"temperature": 0}
    assert openai.calls[-1].api_key == "env-key"


def test_a_numeric_attribute_may_arrive_as_a_string(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    runtime = Runtime(bundle(fitted_classifier(model_params={"temperature": 0})))

    runtime.compute({"X": ["some text"]}, {"temperature": "0.3"})

    assert openai.calls[-1].params == {"temperature": 0.3}


def test_a_param_stored_as_text_is_not_read_back_as_a_number(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    """Its declared type is text; a platform round-trips it as the string it is.

    ``user`` is an OpenAI request param the API rejects as an integer, so
    re-sending the documented default must not silently retype it.
    """
    runtime = Runtime(bundle(fitted_classifier(model_params={"user": "12345"})))

    runtime.compute({"X": ["some text"]}, {"user": "12345"})

    assert openai.calls[-1].params["user"] == "12345"


def test_params_absent_from_the_stored_config_go_through_overrides(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    runtime = Runtime(bundle(fitted_classifier()))

    runtime.compute({"X": ["some text"]}, {"overrides": {"max_tokens": "128"}})

    assert openai.calls[-1].params["max_tokens"] == 128
    with pytest.raises(ValueError, match="max_tokens"):
        runtime.compute({"X": ["some text"]}, {"max_tokens": 128})


def test_an_overrides_object_may_arrive_as_json_text(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    runtime = Runtime(bundle(fitted_classifier(model_params={"temperature": 0})))

    runtime.compute({"X": ["some text"]}, {"overrides": '{"temperature": 0.5}'})

    assert openai.calls[-1].params == {"temperature": 0.5}


def test_an_override_reaches_exactly_the_member_it_names(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    ensemble = ConsensusClassifier(
        [
            ZeroShotClassifier(model="openai::fast", alias="fast", verbose=0),
            ZeroShotClassifier(model="openai::careful", alias="careful", verbose=0),
        ],
        verbose=0,
    ).fit(REVIEWS, SENTIMENTS)
    runtime = Runtime(bundle(ensemble))
    keys = {"credential_fast": "sk-a", "credential_careful": "sk-b"}

    runtime.compute(
        {"X": ["I love it"]}, {**keys, "overrides": {"careful": {"temperature": 0.9}}}
    )

    assert {call.model: call.params for call in openai.calls[-2:]} == {
        "fast": {"temperature": 0},
        "careful": {"temperature": 0.9},
    }
    with pytest.raises(ValueError, match=r"careless.*careful"):
        runtime.compute({"X": ["I love it"]}, {**keys, "overrides": {"careless": {}}})


def test_an_undeclared_attribute_is_rejected(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    runtime = Runtime(bundle(fitted_classifier(model_params={"temperature": 0})))

    with pytest.raises(ValueError, match=r"temperatura.*temperature"):
        runtime.compute({"X": ["some text"]}, {"temperatura": 0.5})


def test_a_missing_secret_names_both_ways_to_supply_it(
    openai: FakeBackend, monkeypatch: pytest.MonkeyPatch, bundle: Callable[..., str]
) -> None:
    path = bundle(fitted_classifier())
    monkeypatch.delenv("OPENAI_API_KEY")

    runtime = Runtime(path)  # warmup needs no credentials

    with pytest.raises(MissingCredentialError, match=r"openai_api_key.*OPENAI_API_KEY"):
        runtime.compute({"X": ["some text"]}, {})


def test_a_missing_secret_says_when_the_environment_is_being_ignored(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    """A key that is present but deliberately unread must not be mystifying."""
    runtime = Runtime(bundle(fitted_classifier()))

    with (
        config_context(allow_env_fallback=False),
        pytest.raises(MissingCredentialError, match="fallback is disabled"),
    ):
        runtime.compute({"X": ["some text"]}, {})


def test_a_base_url_reaches_the_provider(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    runtime = Runtime(bundle(fitted_classifier()))

    runtime.compute(
        {"X": ["some text"]}, {"openai_base_url": "http://localhost:8000/v1"}
    )

    assert openai.calls[-1].base_url == "http://localhost:8000/v1"


def test_the_serving_process_config_answers_when_a_call_does_not(
    openai: FakeBackend, monkeypatch: pytest.MonkeyPatch, bundle: Callable[..., str]
) -> None:
    path = bundle(fitted_classifier())
    monkeypatch.delenv("OPENAI_API_KEY")
    runtime = Runtime(path)

    with config_context(openai_api_key="sk-context"):
        runtime.compute({"X": ["some text"]}, {})

    assert openai.calls[-1].api_key == "sk-context"


def test_concurrent_computes_overlap_on_one_event_loop(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    windows: list[tuple[float, float]] = []

    def slow(call: FakeCall) -> dict[str, str]:
        started = time.perf_counter()
        time.sleep(0.05)
        windows.append((started, time.perf_counter()))
        return {"label": "positive"}

    openai.structured_script = slow
    runtime = Runtime(bundle(fitted_classifier()))

    async def both() -> Any:
        return await asyncio.gather(
            runtime.compute_async({"X": ["a"]}, {}),
            runtime.compute_async({"X": ["b"]}, {}),
        )

    answers = asyncio.run(both())

    assert answers == [{"predictions": ["positive"]}] * 2
    assert max(start for start, _ in windows) < min(end for _, end in windows)


# --- what may be saved --------------------------------------------------------


def test_an_estimator_saves_itself(openai: FakeBackend, tmp_path: Any) -> None:
    path = str(tmp_path / "self.fnnx")

    fitted_classifier().save(path)

    assert manifest(path)["producer_name"] == "skllm.fnnx.ai"


def test_an_unfitted_estimator_cannot_be_saved(
    openai: FakeBackend, bundle: Callable[..., str]
) -> None:
    with pytest.raises(NotFittedError):
        bundle(ZeroShotClassifier(model="openai::gpt-5"))


def test_a_classical_model_is_out_of_scope(bundle: Callable[..., str]) -> None:
    classical = LogisticRegression().fit(np.array([[0.0], [1.0]]), ["n", "p"])

    with pytest.raises(ValueError, match="no scikit-llm estimator"):
        bundle(classical)
