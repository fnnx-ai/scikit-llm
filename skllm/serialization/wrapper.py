"""Static runtime wrapper for the FNNX bundles ``skllm.save`` produces.

Every bundle uses this exact class as its pyfunc entry point; what varies per
model is driven by ``extra_values["skllm"]`` written at save time. The module is
copied verbatim into the bundle, so it imports only the FNNX PyFunc base at the
top level — scikit-llm itself, joblib and numpy are reached inside the methods,
which run in the serving environment.

The wrapper owns the dynamic-attribute contract, because the pyfunc variant
passes attributes through unvalidated and manifest ``Var``s carry no defaults:
it rejects undeclared names, coerces numeric strings, resolves credentials
through the same layers a live estimator uses, and translates per-call
overrides into a ``config_context`` around the call — never into estimator
state, which stays exactly as it was saved.
"""

import copy
import json
import math
import sys
from typing import Any

from fnnx.variants.pyfunc import PyFunc

#: Where the save side records everything this wrapper needs to know.
CONFIG_KEY = "skllm"
OVERRIDES_ATTRIBUTE = "overrides"


class SkllmModel(PyFunc):
    """One static pyfunc wrapper for any saved scikit-llm object graph."""

    def warmup(self) -> None:
        import joblib

        config = self.fnnx_context.get_value(CONFIG_KEY) or {}
        state_path = self.fnnx_context.get_filepath(config.get("state_file", ""))
        if state_path is None:
            raise FileNotFoundError(
                f"The bundle does not contain its estimator payload "
                f"{config.get('state_file')!r}."
            )
        self._config = config
        self._model = joblib.load(state_path)
        self._declared = frozenset(
            {
                OVERRIDES_ATTRIBUTE,
                *(entry["attribute"] for entry in config.get("credentials", [])),
                *(entry["attribute"] for entry in config.get("base_urls", [])),
                *(config.get("model_params") or {}),
            }
        )

    def compute(self, inputs: dict, dynamic_attributes: dict) -> dict:
        from skllm.config import config_context

        attributes = self._checked(dynamic_attributes)
        model, settings = self._call_scope(attributes)
        with config_context(**settings):
            answer = getattr(model, self._config["method"])(inputs["X"])
        return {"predictions": _jsonable(answer)}

    async def compute_async(self, inputs: dict, dynamic_attributes: dict) -> dict:
        """Run the batch off the event loop, so concurrent calls overlap."""
        import asyncio

        return await asyncio.to_thread(self.compute, inputs, dynamic_attributes)

    def _checked(self, dynamic_attributes: dict | None) -> dict:
        """Reject what the bundle never declared, which FNNX would silently ignore."""
        attributes = dict(dynamic_attributes or {})
        unknown = sorted(set(attributes) - self._declared)
        if unknown:
            raise ValueError(
                f"Unknown dynamic attribute(s) {unknown}. This bundle declares "
                f"{sorted(self._declared)}."
            )
        return attributes

    def _call_scope(self, attributes: dict) -> tuple[Any, dict]:
        """The estimator to call and the config scope to call it in."""
        settings = self._credentials(attributes)
        settings.update(self._base_urls(attributes))

        if self._config["kind"] == "composite":
            overrides = self._alias_overrides(attributes)
            if overrides:
                settings["overrides"] = overrides
            return self._model, settings
        return self._tuned(self._param_overrides(attributes)), settings

    def _credentials(self, attributes: dict) -> dict:
        """Resolve every credential the bundle declared, per call.

        A dynamic attribute wins; otherwise the serving process's own config
        layers answer, exactly as they would for a live estimator.
        """
        from skllm.config import (
            MissingCredentialError,
            resolve_allow_env_fallback,
            resolve_credential,
        )

        settings: dict[str, Any] = {}
        aliased: dict[str, str] = {}
        for entry in self._config.get("credentials", []):
            key = attributes.get(entry["attribute"]) or resolve_credential(
                entry["provider"],
                alias=entry.get("alias"),
                env_vars=entry.get("env_vars"),
                required=False,
            )
            if not key:
                names = " or ".join(entry["env_vars"])
                # Naming an env var the serving process is configured to ignore
                # would send the caller to set a key that cannot be read.
                fallback = (
                    f", or provide it to the serving process as {names}"
                    if resolve_allow_env_fallback()
                    else f"; the environment fallback is disabled, so {names} is "
                    "ignored"
                )
                raise MissingCredentialError(
                    f"No credential for the {entry['attribute']!r} dynamic attribute "
                    f"of this bundle. Pass it in the call's dynamic attributes"
                    f"{fallback}."
                )
            if entry.get("alias"):
                aliased[entry["alias"]] = key
            else:
                settings[f"{entry['provider']}_api_key"] = key

        if aliased:
            settings["credentials"] = aliased
        return settings

    def _base_urls(self, attributes: dict) -> dict:
        return {
            f"{entry['provider']}_base_url": attributes[entry["attribute"]]
            for entry in self._config.get("base_urls", [])
            if attributes.get(entry["attribute"])
        }

    def _alias_overrides(self, attributes: dict) -> dict:
        """Per-alias generation params for a composite bundle."""
        overrides = _as_mapping(
            attributes.get(OVERRIDES_ATTRIBUTE), OVERRIDES_ATTRIBUTE
        )
        available = self._config.get("aliases", [])
        unknown = sorted(set(overrides) - set(available))
        if unknown:
            raise ValueError(
                f"Cannot override {unknown}: this bundle has no estimator with that "
                f"alias. Available aliases: {sorted(available)}."
            )
        return {
            alias: _numeric(_as_mapping(params, f"{OVERRIDES_ATTRIBUTE}[{alias!r}]"))
            for alias, params in overrides.items()
        }

    def _param_overrides(self, attributes: dict) -> dict:
        """Generation params for a single-estimator bundle, flat sugar included."""
        stored = self._config.get("model_params") or {}
        flat = {name: attributes[name] for name in stored if name in attributes}
        overrides = _as_mapping(
            attributes.get(OVERRIDES_ATTRIBUTE), OVERRIDES_ATTRIBUTE
        )
        return _numeric({**flat, **overrides}, stored)

    def _tuned(self, params: dict) -> Any:
        """The estimator with per-call params merged over its stored ones.

        A shallow copy shares the fitted state and leaves the saved defaults
        untouched, so an override never outlives the call that asked for it.
        """
        if not params:
            return self._model
        tuned = copy.copy(self._model)
        tuned.model_params = {**(self._model.model_params or {}), **params}
        return tuned


def _as_mapping(value: Any, name: str) -> dict:
    """A JSON object attribute, accepting the string form a gateway may send."""
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError as exc:
            raise ValueError(f"{name} is not valid JSON: {value!r}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object, got {type(value).__name__}.")
    return value


def _numeric(params: dict, stored: dict | None = None) -> dict:
    """Turn the strings a serving platform sends back into the numbers they denote.

    Attributes reach a deployment through an OpenAPI schema that types every one
    of them as a string, so ``"0.3"`` is a normal way to send a temperature. A
    param the bundle stores as a string is left alone: its declared type is text,
    and coercing it would rewrite values that only work as text (OpenAI's
    ``user``, Ollama's ``keep_alive``). A param the bundle stores nothing for is
    coerced best-effort, a number being the likelier intent.
    """
    stored = stored or {}
    return {
        name: value if isinstance(stored.get(name), str) else _number(value)
        for name, value in params.items()
    }


def _number(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return value


def _jsonable(value: Any) -> Any:
    """An estimator's answer as JSON-safe data; NaN fills become null."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None

    numpy = sys.modules.get("numpy")
    if numpy is not None:
        if isinstance(value, numpy.ndarray):
            return _jsonable(value.tolist())
        if isinstance(value, numpy.generic):
            return _jsonable(value.item())

    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)
