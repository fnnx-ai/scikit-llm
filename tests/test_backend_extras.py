"""What a user sees when a provider's optional dependency is not installed."""

from __future__ import annotations

import importlib
import re
import sys

import pytest

from skllm.backends import get_backend, registry

# (provider prefix, the SDK module the adapter imports, the scikit-llm extra)
OPTIONAL_BACKENDS = [
    ("anthropic", "anthropic", "anthropic"),
    ("google", "google.genai", "google"),
    ("litellm", "litellm", "litellm"),
]


@pytest.mark.parametrize(("provider", "package", "extra"), OPTIONAL_BACKENDS)
def test_a_missing_extra_names_the_install_command(
    monkeypatch: pytest.MonkeyPatch, provider: str, package: str, extra: str
) -> None:
    # A None entry makes the import machinery treat the package as unimportable.
    monkeypatch.setitem(sys.modules, package, None)
    monkeypatch.delitem(sys.modules, f"skllm.backends.{provider}", raising=False)
    monkeypatch.setattr(registry, "_BACKENDS", {})

    with pytest.raises(
        ImportError, match=re.escape(f"pip install scikit-llm[{extra}]")
    ):
        get_backend(provider)


@pytest.mark.parametrize(("provider", "package", "extra"), OPTIONAL_BACKENDS)
def test_an_installed_extra_yields_a_usable_adapter(
    provider: str, package: str, extra: str
) -> None:
    backend = get_backend(provider)

    assert backend.provider == provider
    assert get_backend(provider) is backend


# The backends that import names (not just modules) out of their SDK, and one
# such name each — an SDK version that dropped it must not read as "not installed".
@pytest.mark.parametrize(
    ("provider", "package", "dropped"),
    [("anthropic", "anthropic", "Anthropic"), ("google", "google.genai", "Client")],
)
def test_an_sdk_that_dropped_a_name_is_not_reported_as_a_missing_extra(
    monkeypatch: pytest.MonkeyPatch, provider: str, package: str, dropped: str
) -> None:
    """An installed-but-incompatible SDK must say so, not send the user to pip."""
    module_name = f"skllm.backends.{provider}"
    monkeypatch.delattr(importlib.import_module(package), dropped)
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setattr(registry, "_BACKENDS", {})

    with pytest.raises(ImportError) as excinfo:
        get_backend(provider)

    assert dropped in str(excinfo.value)
    assert "pip install" not in str(excinfo.value)
