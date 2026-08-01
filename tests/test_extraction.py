from __future__ import annotations

import pickle
from typing import Any

import pytest
from pydantic import BaseModel
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.utils import get_tags

from skllm.core import UsageRecord
from skllm.extraction import StructuredExtractor
from tests.fakes import FakeBackend

FAKE_MODEL = "fake::m"

SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "year": {"type": "integer"}},
    "required": ["name", "year"],
    "additionalProperties": False,
}

TEXT = "Ada Lovelace was born in 1815."
OTHER_TEXT = "Alan Turing was born in 1912."
RECORD = {"name": "Ada Lovelace", "year": 1815}
OTHER_RECORD = {"name": "Alan Turing", "year": 1912}


class Address(BaseModel):
    city: str


class Person(BaseModel):
    name: str
    year: int
    address: Address | None = None


@pytest.fixture
def extractor(fake_backend: FakeBackend) -> StructuredExtractor:
    return StructuredExtractor(model=FAKE_MODEL, schema=SCHEMA, verbose=0)


# --- extraction --------------------------------------------------------------


def test_each_text_is_extracted_into_a_record(
    fake_backend: FakeBackend, extractor: StructuredExtractor
) -> None:
    fake_backend.structured_script = lambda call: (
        RECORD if TEXT in call.prompt else OTHER_RECORD
    )

    records = extractor.transform([TEXT, OTHER_TEXT])

    assert records == [RECORD, OTHER_RECORD]
    assert len(fake_backend.calls) == 2


def test_the_schema_constrains_the_request(
    fake_backend: FakeBackend, extractor: StructuredExtractor
) -> None:
    """The answer is asked for through structured outputs, not parsed out of prose."""
    fake_backend.structured_script = RECORD

    extractor.transform([TEXT])

    assert fake_backend.calls[0].kind == "structured"
    assert fake_backend.calls[0].schema == SCHEMA


def test_the_prompt_carries_the_text_sample(
    fake_backend: FakeBackend, extractor: StructuredExtractor
) -> None:
    fake_backend.structured_script = RECORD

    extractor.transform([TEXT])

    assert TEXT in fake_backend.prompts[0]


# --- schema validation -------------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        {"name": "Ada Lovelace", "year": "eighteen fifteen"},  # wrong field type
        {"name": "Ada Lovelace"},  # missing a required field
        {"name": "Ada Lovelace", "year": 1815, "born": "London"},  # extra field
        {"name": "Ada Lovelace", "year": True},  # a bool is not an integer
    ],
)
def test_a_record_that_does_not_fit_the_schema_is_a_sample_failure(
    answer: Any, fake_backend: FakeBackend, extractor: StructuredExtractor
) -> None:
    fake_backend.structured_script = answer

    with pytest.warns(UserWarning, match="1 of 1 samples"):
        records = extractor.transform([TEXT])

    assert records == [None]
    assert [error.index for error in extractor.errors_] == [0]
    assert "does not fit the extraction schema" in extractor.errors_[0].error


@pytest.mark.parametrize(
    "answer",
    [
        ["Ada Lovelace", 1815],  # an array, not the object the schema describes
        "Ada Lovelace, 1815",  # nothing salvage could turn into a record
    ],
)
def test_an_answer_that_is_not_an_object_is_a_sample_failure(
    answer: Any, fake_backend: FakeBackend, extractor: StructuredExtractor
) -> None:
    # A callable script rather than the value: the fake reads a bare list as a
    # queue of answers, which would hide the array case behind its first item.
    fake_backend.structured_script = lambda call: answer

    with pytest.warns(UserWarning, match="1 of 1 samples"):
        records = extractor.transform([TEXT])

    assert records == [None]
    assert "describes an object" in extractor.errors_[0].error


def test_only_the_failing_sample_is_filled(
    fake_backend: FakeBackend, extractor: StructuredExtractor
) -> None:
    fake_backend.structured_script = lambda call: (
        RECORD if TEXT in call.prompt else {"name": "Alan Turing"}
    )

    with pytest.warns(UserWarning, match="1 of 2 samples"):
        records = extractor.transform([TEXT, OTHER_TEXT])

    assert records == [RECORD, None]
    assert [error.index for error in extractor.errors_] == [1]
    assert "does not fit the extraction schema" in extractor.errors_[0].error


def test_a_request_failure_fills_with_none(
    fake_backend: FakeBackend, extractor: StructuredExtractor
) -> None:
    fake_backend.structured_script = RuntimeError("the provider rejected the request")

    with pytest.warns(UserWarning, match="1 of 1 samples"):
        records = extractor.transform([TEXT])

    assert records == [None]
    assert extractor.usage_.last.failures == 1


def test_the_raise_policy_aborts_the_batch(
    fake_backend: FakeBackend, extractor: StructuredExtractor
) -> None:
    fake_backend.structured_script = {"name": "Ada Lovelace"}
    extractor.set_params(on_error="raise")

    with pytest.raises(Exception, match="does not fit the extraction schema"):
        extractor.transform([TEXT])


# --- pydantic model classes --------------------------------------------------


def test_a_model_class_is_compiled_into_the_request(
    fake_backend: FakeBackend,
) -> None:
    fake_backend.structured_script = RECORD
    extractor = StructuredExtractor(model=FAKE_MODEL, schema=Person, verbose=0)

    records = extractor.transform([TEXT])

    assert records == [RECORD]
    assert fake_backend.calls[0].schema == Person.model_json_schema()


def test_a_nested_model_is_validated_through_its_references(
    fake_backend: FakeBackend,
) -> None:
    """A pydantic schema refers to nested models by $ref, which validation follows."""
    fake_backend.structured_script = lambda call: (
        {**RECORD, "address": {"city": "London"}}
        if TEXT in call.prompt
        else {**OTHER_RECORD, "address": {"city": 51}}
    )
    extractor = StructuredExtractor(model=FAKE_MODEL, schema=Person, verbose=0)

    with pytest.warns(UserWarning, match="1 of 2 samples"):
        records = extractor.transform([TEXT, OTHER_TEXT])

    assert records == [{**RECORD, "address": {"city": "London"}}, None]


def test_a_model_class_survives_get_params_and_clone() -> None:
    """The normalization is a serialization boundary, not an in-memory one."""
    extractor = StructuredExtractor(model=FAKE_MODEL, schema=Person)

    cloned = clone(extractor)

    assert cloned.get_params()["schema"] is Person
    assert cloned.schema is Person


def test_pickling_normalizes_a_model_class_to_its_schema(
    fake_backend: FakeBackend,
) -> None:
    """A restored estimator needs neither the model class nor pydantic."""
    fake_backend.structured_script = RECORD
    extractor = StructuredExtractor(
        model=FAKE_MODEL, schema=Person, api_key="sk-secret", verbose=0
    )

    restored = pickle.loads(pickle.dumps(extractor))

    assert restored.schema == Person.model_json_schema()
    assert isinstance(restored.schema, dict)
    assert restored.api_key is None
    assert restored.transform([TEXT]) == [RECORD]
    assert fake_backend.calls[0].schema == Person.model_json_schema()


def test_pickling_leaves_a_schema_dict_alone(fake_backend: FakeBackend) -> None:
    restored = pickle.loads(
        pickle.dumps(StructuredExtractor(model=FAKE_MODEL, schema=SCHEMA))
    )

    assert restored.schema == SCHEMA


# --- param validation --------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        None,
        "name: string",
        [("name", "string")],
        Person(name="Ada Lovelace", year=1815),  # an instance, not the class
    ],
)
def test_a_schema_that_is_neither_an_object_nor_a_model_class_is_rejected(
    value: Any, fake_backend: FakeBackend
) -> None:
    with pytest.raises(TypeError, match="schema must be a JSON Schema object"):
        StructuredExtractor(model=FAKE_MODEL, schema=value).transform([TEXT])


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"type": "array", "items": {"type": "string"}},
        {"properties": {"name": {"type": "string"}}},
    ],
)
def test_a_schema_that_does_not_describe_an_object_is_rejected(
    value: Any, fake_backend: FakeBackend
) -> None:
    with pytest.raises(ValueError, match="must describe an object"):
        StructuredExtractor(model=FAKE_MODEL, schema=value).transform([TEXT])


def test_fit_validates_the_schema(fake_backend: FakeBackend) -> None:
    with pytest.raises(TypeError, match="schema must be a JSON Schema object"):
        StructuredExtractor(model=FAKE_MODEL).fit([TEXT])


def test_the_default_template_matches_its_documented_placeholders() -> None:
    StructuredExtractor._check_prompt_template(
        StructuredExtractor.default_prompt_template
    )


def test_an_invalid_prompt_template_names_the_placeholders(
    fake_backend: FakeBackend,
) -> None:
    with pytest.raises(ValueError) as excinfo:
        StructuredExtractor(
            model=FAKE_MODEL,
            schema=SCHEMA,
            prompt_template="Extract {fields} from the sample",
        ).transform([TEXT])

    message = str(excinfo.value)
    assert "missing ['text']" in message
    assert "unknown ['fields']" in message


# --- sklearn compliance ------------------------------------------------------


def test_the_extractor_is_stateless(
    fake_backend: FakeBackend, extractor: StructuredExtractor
) -> None:
    assert not get_tags(extractor).requires_fit
    assert extractor.fit([TEXT]) is extractor
    assert fake_backend.calls == []  # fit only validates params

    fake_backend.structured_script = RECORD
    assert StructuredExtractor(model=FAKE_MODEL, schema=SCHEMA).transform([TEXT]) == [
        RECORD
    ]


def test_params_survive_a_clone() -> None:
    extractor = StructuredExtractor(
        model=FAKE_MODEL, schema=SCHEMA, api_key="sk-secret", alias="team_a"
    )

    cloned = clone(extractor)

    assert cloned.get_params() == extractor.get_params()
    assert vars(extractor) == extractor.get_params()


def test_the_extractor_runs_as_a_pipeline_step(fake_backend: FakeBackend) -> None:
    fake_backend.structured_script = RECORD
    pipeline = Pipeline(
        [("extract", StructuredExtractor(model=FAKE_MODEL, schema=SCHEMA, verbose=0))]
    )

    assert pipeline.fit_transform([TEXT]) == [RECORD]
    assert len(fake_backend.calls) == 1  # fitting the step issued none


def test_an_empty_batch_issues_no_requests(
    fake_backend: FakeBackend, extractor: StructuredExtractor
) -> None:
    fake_backend.structured_script = RECORD
    extractor.transform([TEXT])
    before = extractor.usage_.total
    fake_backend.calls.clear()

    assert extractor.transform([]) == []
    assert fake_backend.calls == []
    assert extractor.usage_.last == UsageRecord()
    assert extractor.usage_.total == before
