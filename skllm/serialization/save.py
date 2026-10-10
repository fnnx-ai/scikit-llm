"""Packaging a fitted object graph as a self-describing FNNX bundle.

The whole graph travels as one joblib payload, so any sklearn composition — a
pipeline, a meta-estimator, a bare estimator — is packaged without being taken
apart. What the bundle adds around that payload is the contract: the IO schemas
it accepts and publishes, the credentials it needs and how to supply them, the
generation params a call may override, and the environment it runs in.

Credentials are excluded by construction: the payload is written through the
estimators' own credential-stripping state, and keys are re-supplied at serving
time through dynamic attributes or the serving process's config layers.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Sequence
from typing import Any

import joblib
from fnnx.extras.builder import PyfuncBuilder
from fnnx.extras.pydantic_models.manifest import JSON, Var
from sklearn.utils.validation import check_is_fitted

from skllm.backends import get_backend
from skllm.core import BaseLLMEstimator, Requirement
from skllm.ensemble import ConsensusClassifier
from skllm.serialization import discovery, schemas
from skllm.serialization.wrapper import (
    CONFIG_KEY,
    INPUT_CHANNEL,
    OUTPUT_CHANNEL,
    OVERRIDES_ATTRIBUTE,
    SkllmModel,
)

__all__ = ["save"]

PRODUCER_NAME = "skllm.fnnx.ai"
TAG_PREFIX = f"{PRODUCER_NAME}::"

#: Marks a variable as a credential. The second tag is luml's: its deployment UI
#: matches that exact string to render a secret picker instead of a plaintext
#: field, inject the bound secret per request and scrub it from the published
#: OpenAPI schema and telemetry.
SECRET_TAGS = [
    f"{TAG_PREFIX}provider_api_key:v1",
    "dataforce.studio::runtime_secret:v1",
]
GENERATION_TAGS = [f"{TAG_PREFIX}generation_param:v1"]

STATE_FILE = "estimator.joblib"
INPUT_DTYPE = "ext::skllm::X"
OUTPUT_DTYPE = "ext::skllm::predictions"
CREDENTIAL_ATTRIBUTE_PREFIX = "credential_"
CREDENTIAL_ENV_PREFIX = "SKLLM_CREDENTIAL_"

#: Providers whose SDK is not a core dependency, and the extra that installs it.
PROVIDER_EXTRAS = {
    "anthropic": "anthropic",
    "google": "google",
    "litellm": "litellm",
    "typesafe": "typesafe",
}


def save(estimator: Any, path: str) -> None:
    """Write a fitted sklearn object containing skllm estimators to ``path``.

    The object may be a skllm estimator or any sklearn composition holding at
    least one — the graph is packaged whole. Credentials are never written.
    """
    from skllm import __version__

    check_is_fitted(estimator)
    if not discovery.find_estimators(estimator):
        raise ValueError(
            f"{type(estimator).__name__} contains no scikit-llm estimator, so there "
            "is nothing for a scikit-llm bundle to serve. Save purely classical "
            "sklearn models with an sklearn-flavored packager instead."
        )

    entry, exit_ = schemas.edges(estimator)
    entry_kind, exit_kind = schemas.entry_kind(entry), schemas.exit_kind(exit_)
    single = isinstance(estimator, BaseLLMEstimator) and not isinstance(
        estimator, ConsensusClassifier
    )
    requirements = discovery.requirements(estimator)
    credentials = _credential_attributes(requirements)
    uses = discovery.model_uses(estimator)

    config = {
        "version": __version__,
        "state_file": STATE_FILE,
        "estimator": f"{type(estimator).__module__}.{type(estimator).__qualname__}",
        "kind": "single" if single else "composite",
        "method": "predict" if hasattr(estimator, "predict") else "transform",
        "task": schemas.task_tag(exit_),
        "entry_kind": entry_kind,
        "exit_kind": exit_kind,
        "credentials": credentials,
        "base_urls": _base_url_attributes(requirements),
        "aliases": sorted({alias for _, alias in requirements if alias}),
        # The top-level estimator's own alias, which the discovered set cannot
        # single out (an internal component may be aliased too): a
        # single-estimator bundle takes generation params unkeyed, so a loader
        # forwarding an alias-targeted override needs to know whose it is.
        "alias": estimator.alias if single else None,
        "model_params": dict(estimator.model_params or {}) if single else {},
        "models": [use._asdict() for use in uses],
    }

    builder = PyfuncBuilder(pyfunc=SkllmModel)
    builder.set_producer_info(
        name=PRODUCER_NAME,
        version=__version__,
        tags=[f"{TAG_PREFIX}{config['task']}:v1"],
    )
    builder.define_dtype(INPUT_DTYPE, schemas.input_schema(entry_kind))
    builder.define_dtype(OUTPUT_DTYPE, schemas.output_schema(exit_kind, exit_))
    builder.add_input(JSON(name=INPUT_CHANNEL, content_type="JSON", dtype=INPUT_DTYPE))
    builder.add_output(
        JSON(name=OUTPUT_CHANNEL, content_type="JSON", dtype=OUTPUT_DTYPE)
    )
    for variable in _dynamic_attributes(config):
        builder.add_dynamic_attribute(variable)
    for variable in _env_vars(credentials):
        builder.add_env_var(variable)
    builder.add_runtime_dependency(_dependency(uses, __version__))
    builder.add_fnnx_runtime_dependency()
    builder.set_extra_values({CONFIG_KEY: config})

    with tempfile.TemporaryDirectory(prefix="skllm_save_") as workspace:
        payload = os.path.join(workspace, STATE_FILE)
        joblib.dump(estimator, payload)
        builder.add_file(payload, STATE_FILE)
        builder.save(path)


def _credential_attributes(requirements: Sequence[Requirement]) -> list[dict[str, Any]]:
    """The keys a served bundle needs, per provider and per alias.

    Only a key something could actually consume is declared. A provider-level
    key needs an unaliased estimator, since an aliased one resolves strictly by
    name; and a keyless provider needs nothing at all, aliased or not — routing
    a credential by name does not invent a credential to route.
    """
    entries = []
    for provider in _unique(provider for provider, alias in requirements if not alias):
        backend = get_backend(provider)
        canonical = backend.credential_env_var
        if canonical is None:  # a keyless provider: Ollama, or litellm's own vars
            continue
        accepted = getattr(backend, "credential_env_vars", None) or (canonical,)
        entries.append(
            {
                "attribute": f"{provider}_api_key",
                "provider": provider,
                "env_var": canonical,
                "env_vars": list(accepted),
            }
        )
    declared: set[str] = set()
    for provider, alias in requirements:
        if not alias or alias in declared:
            continue
        if get_backend(provider).credential_env_var is None:
            continue
        declared.add(alias)
        env_var = f"{CREDENTIAL_ENV_PREFIX}{alias.upper()}"
        entries.append(
            {
                "attribute": f"{CREDENTIAL_ATTRIBUTE_PREFIX}{alias}",
                "provider": provider,
                "alias": alias,
                "env_var": env_var,
                "env_vars": [env_var],
            }
        )
    return entries


def _base_url_attributes(requirements: Sequence[Requirement]) -> list[dict[str, str]]:
    """Where each discovered provider is reached — serving topology, not identity."""
    return [
        {"attribute": f"{provider}_base_url", "provider": provider}
        for provider in _unique(provider for provider, _ in requirements)
    ]


def _dynamic_attributes(config: dict[str, Any]) -> list[Var]:
    variables = [
        Var(
            name=entry["attribute"],
            description=(
                f"API key for the estimator aliased {entry['alias']!r} "
                f"(provider {entry['provider']}). Required."
                if entry.get("alias")
                else f"API key for the {entry['provider']} provider. Required."
            ),
            tags=SECRET_TAGS,
        )
        for entry in config["credentials"]
    ]
    variables += [
        Var(
            name=entry["attribute"],
            description=f"Base URL for {entry['provider']} requests.",
        )
        for entry in config["base_urls"]
    ]
    variables.append(
        Var(
            name=OVERRIDES_ATTRIBUTE,
            description=(
                "JSON object of generation params to apply over the stored ones."
                if config["kind"] == "single"
                else "JSON object mapping estimator alias to the generation params "
                f"to apply for this call. Available aliases: {config['aliases']}."
            ),
            tags=GENERATION_TAGS,
        )
    )

    declared = {variable.name for variable in variables}
    variables += [
        Var(
            name=name,
            description=f"Generation param {name!r} (default: {value!r}).",
            tags=GENERATION_TAGS,
        )
        for name, value in config["model_params"].items()
        if name not in declared
    ]
    return variables


def _env_vars(credentials: Sequence[dict[str, Any]]) -> list[Var]:
    """The environment the serving process may answer the credential calls from."""
    return [
        Var(
            name=entry["env_var"],
            description=f"Fallback for the {entry['attribute']!r} dynamic attribute.",
            tags=SECRET_TAGS,
        )
        for entry in credentials
    ]


def _dependency(uses: Sequence[discovery.ModelUse], version: str) -> str:
    """This library pinned, with the extras the discovered models need.

    Mandatory rather than advisory: a satellite builds the serving environment
    from this alone, so an unlisted provider extra means an import error at
    warmup rather than a missing convenience. Derived from every model string in
    the graph, not from the credentials it needs: resolving a backend imports its
    SDK even where serving issues no request through it (a ``ClusterLabeler``
    validates its model on every ``transform``).
    """
    extras = sorted(
        {
            PROVIDER_EXTRAS[use.provider]
            for use in uses
            if use.provider in PROVIDER_EXTRAS
        }
    )
    package = f"scikit-llm[{','.join(extras)}]" if extras else "scikit-llm"
    return f"{package}=={version}"


def _unique(items: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))
