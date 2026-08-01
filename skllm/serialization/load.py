"""Loading a bundle back into something callable.

``load`` hands back an :class:`InferenceModel` rather than the estimator: a
bundle is a deployable artifact, and running it through ``fnnx.Runtime`` is what
a deployment does — so the object you get locally answers through exactly the
path a served copy answers through. The estimator itself is one
:meth:`~InferenceModel.unwrap` away for the in-Python work bundles are not for.

The facade owns the two edges the Runtime leaves open: it converts batches and
answers between sklearn types and the bundle's JSON channels, and it resolves
this process's connection config on every call, forwarding it as dynamic
attributes — so ``config_context`` reads the same around a loaded bundle as
around a live estimator, whichever handler runs it.
"""

from __future__ import annotations

import json
import os
import tarfile
from collections.abc import Iterator
from contextlib import contextmanager
from typing import IO, Any

import joblib
from fnnx.handlers._base import BaseHandler
from fnnx.runtime import Runtime

from skllm.config import resolve_base_url, resolve_credential, resolve_overrides
from skllm.serialization.schemas import decode_output, encode_input
from skllm.serialization.wrapper import (
    CONFIG_KEY,
    INPUT_CHANNEL,
    OUTPUT_CHANNEL,
    OVERRIDES_ATTRIBUTE,
)

__all__ = ["InferenceModel", "load"]

#: The methods a bundle may answer through, one of which it declares as its own.
TASK_METHODS = ("predict", "transform")

_VARIANT_CONFIG = "variant_config.json"
_EXTRA_FILES = "variant_artifacts/extra_files"


def load(
    path: str,
    *,
    handler: type[BaseHandler] | None = None,
    handler_config: Any = None,
) -> InferenceModel:
    """Load an ``.fnnx`` bundle written by :func:`skllm.save`.

    The default handler runs the bundle in this process, which needs a
    scikit-llm compatible with the one that wrote it; ``fnnx``'s
    ``StdIOHandler`` runs it in its own environment built from the bundle's
    pinned dependencies instead.

    Loading executes the bundle's packaged code and unpickles its state — load
    only bundles you trust.
    """
    return InferenceModel(path, handler=handler, handler_config=handler_config)


class InferenceModel:
    """A saved model, callable the way the estimator inside it was.

    It exposes the one task method the bundle recorded — ``predict`` for a
    classifier or scorer, ``transform`` for a transformer — taking the usual
    list, array or Series and answering with the array, list of lists or list of
    records the live object would have produced.
    """

    def __init__(
        self,
        path: str,
        *,
        handler: type[BaseHandler] | None = None,
        handler_config: Any = None,
    ) -> None:
        self.path = path
        self._config = _bundle_config(path)
        self._runtime = Runtime(path, handler=handler, handler_config=handler_config)
        self.method: str = self._config["method"]
        setattr(self, self.method, self._infer)

    def unwrap(self) -> Any:
        """The fitted object graph the bundle carries, restored in this process.

        For inspection, refitting or recomposition — everything a bundle, which
        only answers batches, cannot do. Unpickles the payload, so unwrap only
        bundles you trust.
        """
        return _read_payload(self.path, self._config["state_file"])

    def _infer(self, X: Any) -> Any:
        answer = self._runtime.compute(
            {INPUT_CHANNEL: encode_input(self._config["entry_kind"], X)},
            self._attributes(),
        )
        return decode_output(self._config["exit_kind"], answer[OUTPUT_CHANNEL])

    def _attributes(self) -> dict[str, Any]:
        """What this process's config layers answer for this call.

        Only what the bundle declared is forwarded: a scope may carry
        credentials and overrides for estimators outside this bundle, which the
        estimator-agnostic in-memory layer would likewise pass over — and which
        the bundle would reject as undeclared.
        """
        attributes: dict[str, Any] = {}
        for entry in self._config["credentials"]:
            key = resolve_credential(
                entry["provider"],
                alias=entry.get("alias"),
                env_vars=entry.get("env_vars"),
                required=False,
            )
            if key:
                attributes[entry["attribute"]] = key
        for entry in self._config["base_urls"]:
            base_url = resolve_base_url(entry["provider"])
            if base_url:
                attributes[entry["attribute"]] = base_url
        if overrides := self._overrides():
            attributes[OVERRIDES_ATTRIBUTE] = overrides
        return attributes

    def _overrides(self) -> dict[str, Any]:
        """Alias-targeted generation params, in the shape this bundle takes them.

        A composite is tuned per alias; a single estimator takes the params
        unkeyed, so only its own alias — if it has one — can name it.
        """
        if self._config["kind"] == "single":
            return resolve_overrides(self._config.get("alias"))
        return {
            alias: params
            for alias in self._config["aliases"]
            if (params := resolve_overrides(alias))
        }

    def __getattr__(self, name: str) -> Any:
        """Say which task method a bundle answers through, when asked the other."""
        if name in TASK_METHODS:
            raise AttributeError(
                f"This bundle answers through {self._config['method']!r}, not {name!r}."
            )
        raise AttributeError(name)


def _bundle_config(path: str) -> dict[str, Any]:
    """What ``skllm.save`` recorded about the model inside the bundle."""
    with _bundle_file(path, _VARIANT_CONFIG) as handle:
        variant_config = json.load(handle)
    config = (variant_config.get("extra_values") or {}).get(CONFIG_KEY)
    if not isinstance(config, dict):
        raise ValueError(
            f"{path} is an FNNX bundle, but not one scikit-llm wrote: it carries no "
            f"{CONFIG_KEY!r} configuration. Only bundles saved with skllm.save can be "
            "loaded here."
        )
    return config


def _read_payload(path: str, state_file: str) -> Any:
    with _bundle_file(path, f"{_EXTRA_FILES}/{state_file}") as handle:
        return joblib.load(handle)


@contextmanager
def _bundle_file(path: str, member: str) -> Iterator[IO[bytes]]:
    """One member of a bundle, archived or already unpacked."""
    if os.path.isdir(path):
        with open(os.path.join(path, *member.split("/")), "rb") as handle:
            yield handle
        return

    with tarfile.open(path) as archive:
        try:
            extracted = archive.extractfile(member)
        except KeyError:
            extracted = None
        if extracted is None:
            raise FileNotFoundError(f"The bundle {path} holds no {member!r}.")
        with extracted:
            yield extracted
