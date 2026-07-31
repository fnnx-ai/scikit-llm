from __future__ import annotations

import os
from collections.abc import Callable, Iterator

import pytest

from skllm.backends import base, register_backend, registry
from tests.fakes import FakeBackend

FAKE_PROVIDER = "fake"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests hermetic against whatever the developer's shell exports."""
    for name in list(os.environ):
        if (
            name.startswith("SKLLM_")
            or name.endswith(("_API_KEY", "_BASE_URL"))
            or name == "OLLAMA_HOST"
        ):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def clean_process_caches() -> Iterator[None]:
    """Reset the caches adapters keep for the process lifetime."""
    yield
    base._UNSUPPORTED.clear()
    for cache in base._CLIENT_CACHES:
        cache.clear()


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeBackend]:
    backend = FakeBackend()
    monkeypatch.setenv("FAKE_API_KEY", "fake-key")
    register_backend(FAKE_PROVIDER, backend)
    try:
        yield backend
    finally:
        registry._BACKENDS.pop(FAKE_PROVIDER, None)


@pytest.fixture
def fake_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Callable[[str], FakeBackend]]:
    """Answer a real provider prefix with the fake adapter, for the tests that
    exercise what scikit-llm derives from the provider name itself."""
    monkeypatch.setenv("FAKE_API_KEY", "fake-key")
    registered: list[str] = []

    def register(prefix: str) -> FakeBackend:
        backend = FakeBackend()
        register_backend(prefix, backend)
        registered.append(prefix)
        return backend

    try:
        yield register
    finally:
        for prefix in registered:
            registry._BACKENDS.pop(prefix, None)
