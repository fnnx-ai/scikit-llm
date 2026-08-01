"""Schema-driven extraction of one structured record per text.

The schema is the contract on both sides: it constrains the answer through the
provider's structured-output mode, and every answer is validated against it
before it is handed back. A result is therefore a record that fits the schema or
a reported sample failure — never a half-filled dict.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from typing import Any, ClassVar

# The validator FNNX itself checks a bundle's JSON channels with, so an answer
# this estimator accepts is one its served bundle accepts too.
from fnnx.validators.jsonschema import validate_jsonschema
from sklearn.base import TransformerMixin

from skllm.backends import BackendError, Message
from skllm.core import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_MAX_CONCURRENCY,
    BasePromptEstimator,
    StatelessMixin,
    as_texts,
)
from skllm.prompts import EXTRACTION

__all__ = ["StructuredExtractor"]

#: How much of an unusable answer an error message quotes.
_QUOTED_ANSWER_CHARS = 200


class StructuredExtractor(TransformerMixin, StatelessMixin, BasePromptEstimator):
    """Extract the fields a JSON Schema describes out of each text.

    ``schema`` is either a JSON Schema object or a pydantic model class. A class
    is compiled to its schema at the pickling boundary, so a restored estimator
    carries the dict form and a serving environment needs neither the class nor
    pydantic; in memory ``get_params`` and ``clone`` still hand back the class
    itself.
    """

    prompt_placeholders: ClassVar[frozenset[str]] = frozenset({"text"})
    default_prompt_template: ClassVar[str] = EXTRACTION

    def __init__(
        self,
        model: str = DEFAULT_CHAT_MODEL,
        *,
        schema: Mapping[str, Any] | type[Any] | None = None,
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
        self.schema = schema

    def transform(self, X: Any) -> list[dict[str, Any] | None]:
        """Extract one record per text, ``None`` where the sample failed."""
        self._check_ready()
        prompts = [self._prompt(text=text) for text in as_texts(X)]
        records: list[dict[str, Any] | None] = self._run_batch(
            prompts,
            partial(self._extract, schema=self._json_schema()),
            fill=lambda: None,
        )
        return records

    def __getstate__(self) -> dict[str, Any]:
        """Compile a pydantic model class to its schema on the way out.

        What a pickle — and what a bundle — carries is the dict form alone, so
        restoring never imports the user's model class or pydantic itself.
        """
        state = super().__getstate__()
        return {**state, "schema": _as_json_schema(state["schema"])}

    def _check_params(self) -> None:
        super()._check_params()
        self._json_schema()

    def _extract(self, prompt: str, *, schema: dict[str, Any]) -> dict[str, Any]:
        answer = self._structured([Message("user", prompt)], schema)
        if not isinstance(answer, dict):
            raise BackendError(
                f"The model answered with a {type(answer).__name__} where the "
                "extraction schema describes an object: "
                f"{str(answer)[:_QUOTED_ANSWER_CHARS]!r}."
            )
        try:
            validate_jsonschema(answer, schema)
        except ValueError as exc:
            raise BackendError(
                f"The model's answer does not fit the extraction schema: {exc}"
            ) from exc
        return answer

    def _json_schema(self) -> dict[str, Any]:
        """The extraction schema as a JSON Schema object, model class compiled."""
        schema = _as_json_schema(self.schema)
        if not isinstance(schema, Mapping):
            raise TypeError(
                "schema must be a JSON Schema object describing the fields to "
                f"extract, or a pydantic model class, got {type(schema).__name__}."
            )
        if schema.get("type") != "object":
            raise ValueError(
                'The extraction schema must describe an object ({"type": '
                '"object"}), since every sample is extracted into one record. '
                "A schema that pulls several records out of one text nests them "
                "under a property of an object schema."
            )
        return dict(schema)


def _as_json_schema(schema: Any) -> Any:
    """A pydantic model class compiled to its schema; anything else unchanged.

    Duck-typed rather than checked against ``pydantic.BaseModel``: the class is
    the only thing that needs pydantic, and importing it to recognize one would
    make a hard dependency out of an input form.
    """
    compile_schema = getattr(schema, "model_json_schema", None)
    if isinstance(schema, type) and callable(compile_schema):
        return compile_schema()
    return schema
