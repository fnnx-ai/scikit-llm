import asyncio
import contextvars
import os
import threading

import pytest

from skllm.config import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    MissingCredentialError,
    Price,
    config_context,
    credential_env_vars,
    resolve_allow_env_fallback,
    resolve_base_url,
    resolve_client_options,
    resolve_credential,
    resolve_max_retries,
    resolve_model_params,
    resolve_overrides,
    resolve_price,
    resolve_timeout,
    validate_alias,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if (
            name.startswith("SKLLM_")
            or name.endswith(("_API_KEY", "_BASE_URL"))
            or name == "OLLAMA_HOST"
        ):
            monkeypatch.delenv(name, raising=False)


def test_credential_layering(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env_key")

    with config_context(openai_api_key="ctx_key"):
        assert resolve_credential("openai", explicit="param_key") == "param_key"
        assert resolve_credential("openai") == "ctx_key"

    assert resolve_credential("openai") == "env_key"


def test_context_is_popped_on_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env_key")

    with config_context(openai_api_key="ctx_key"):
        pass

    assert resolve_credential("openai") == "env_key"


def test_scope_is_popped_when_the_body_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env_key")

    with pytest.raises(RuntimeError), config_context(openai_api_key="ctx_key"):
        raise RuntimeError("boom")

    assert resolve_credential("openai") == "env_key"


def test_nested_contexts_override_per_setting() -> None:
    with config_context(openai_api_key="outer", anthropic_api_key="anthropic_outer"):
        with config_context(openai_api_key="inner"):
            assert resolve_credential("openai") == "inner"
            assert resolve_credential("anthropic") == "anthropic_outer"
        assert resolve_credential("openai") == "outer"


def test_inner_scope_wins_for_every_setting() -> None:
    outer = {
        "openai_base_url": "http://outer/v1",
        "credentials": {"team_a": "sk-outer"},
        "pricing": {"gpt-5": {"input": 1.0, "output": 2.0}},
        "timeout": 1.0,
        "max_retries": 1,
    }
    inner = {
        "openai_base_url": "http://inner/v1",
        "credentials": {"team_a": "sk-inner"},
        "pricing": {"gpt-5": {"input": 3.0, "output": 4.0}},
        "timeout": 2.0,
        "max_retries": 2,
    }

    with config_context(**outer):
        with config_context(**inner):
            assert resolve_base_url("openai") == "http://inner/v1"
            assert resolve_credential("openai", alias="team_a") == "sk-inner"
            assert resolve_price("gpt-5") == Price(input=3.0, output=4.0)
            assert resolve_timeout() == 2.0
            assert resolve_max_retries() == 2

        assert resolve_base_url("openai") == "http://outer/v1"
        assert resolve_credential("openai", alias="team_a") == "sk-outer"
        assert resolve_price("gpt-5") == Price(input=1.0, output=2.0)
        assert resolve_timeout() == 1.0
        assert resolve_max_retries() == 1


def test_none_values_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env_key")

    with config_context(openai_api_key=None, timeout=None):
        assert resolve_credential("openai") == "env_key"
        assert resolve_timeout() == DEFAULT_TIMEOUT


def test_missing_provider_key_names_the_ways_to_supply_it() -> None:
    with pytest.raises(MissingCredentialError) as excinfo:
        resolve_credential("openai")

    message = str(excinfo.value)
    assert "openai" in message
    assert "api_key" in message
    assert "OPENAI_API_KEY" in message


def test_optional_credential_is_not_required(monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_credential("ollama", required=False) is None

    monkeypatch.setenv("OLLAMA_API_KEY", "ollama_key")
    assert resolve_credential("ollama", required=False) == "ollama_key"


def test_gemini_key_wins_over_google_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "google_key")
    assert resolve_credential("google") == "google_key"

    monkeypatch.setenv("GEMINI_API_KEY", "gemini_key")
    assert resolve_credential("google") == "gemini_key"


def test_adapter_declared_env_vars_replace_the_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MYCORP_TOKEN", "custom_key")

    assert resolve_credential("mycorp", env_vars=("MYCORP_TOKEN",)) == "custom_key"
    assert credential_env_vars("mycorp") == ("MYCORP_API_KEY",)


def test_empty_env_value_is_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "  ")

    with pytest.raises(MissingCredentialError):
        resolve_credential("openai")


def test_blank_context_values_are_treated_as_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env_key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://env/v1")
    monkeypatch.setenv("SKLLM_CREDENTIAL_TEAM_A", "sk-a")

    with config_context(
        openai_api_key="", openai_base_url="  ", credentials={"team_a": ""}
    ):
        assert resolve_credential("openai") == "env_key"
        assert resolve_base_url("openai") == "http://env/v1"
        assert resolve_credential("openai", alias="team_a") == "sk-a"


def test_blank_context_key_does_not_mask_a_missing_credential() -> None:
    with config_context(openai_api_key=""), pytest.raises(MissingCredentialError):
        resolve_credential("openai")


def test_alias_credential_beats_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env_key")

    with config_context(credentials={"team_a": "sk-a"}):
        assert resolve_credential("openai", alias="team_a") == "sk-a"


def test_alias_falls_back_to_its_own_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKLLM_CREDENTIAL_TEAM_A", "sk-a")

    assert resolve_credential("openai", alias="team_a") == "sk-a"


def test_unresolved_alias_never_falls_back_to_the_provider_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env_key")

    with pytest.raises(MissingCredentialError) as excinfo:
        resolve_credential("openai", alias="team_a")

    message = str(excinfo.value)
    assert "team_a" in message
    assert "SKLLM_CREDENTIAL_TEAM_A" in message
    assert "env_key" not in message


def test_estimator_param_outranks_the_alias() -> None:
    with config_context(credentials={"team_a": "sk-a"}):
        assert (
            resolve_credential("openai", alias="team_a", explicit="param_key")
            == "param_key"
        )


def test_aliases_route_independently() -> None:
    with (
        config_context(credentials={"team_a": "sk-a"}),
        config_context(credentials={"team_b": "sk-b"}),
    ):
        assert resolve_credential("openai", alias="team_a") == "sk-a"
        assert resolve_credential("openai", alias="team_b") == "sk-b"


@pytest.mark.parametrize("alias", ["Team A", "team-a", "1team", "", "_team"])
def test_invalid_alias_is_rejected(alias: str) -> None:
    with pytest.raises(ValueError, match=r"\[a-z\]\[a-z0-9_\]\*"):
        validate_alias(alias)

    with pytest.raises(ValueError, match=r"\[a-z\]\[a-z0-9_\]\*"):
        config_context(credentials={alias: "sk-a"}).__enter__()

    with pytest.raises(ValueError, match=r"\[a-z\]\[a-z0-9_\]\*"):
        resolve_credential("openai", alias=alias)


def test_non_string_alias_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"\[a-z\]\[a-z0-9_\]\*"):
        validate_alias(None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="must be strings"):
        config_context(credentials={None: "sk-a"}).__enter__()


def test_env_fallback_can_be_disabled_in_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env_key")

    with config_context(allow_env_fallback=False):
        with pytest.raises(MissingCredentialError) as excinfo:
            resolve_credential("openai")

        message = str(excinfo.value)
        assert "No API key found for provider 'openai'" in message
        assert "environment fallback is disabled" in message
        assert "OPENAI_API_KEY" in message

        assert resolve_credential("openai", explicit="param_key") == "param_key"
        with config_context(openai_api_key="ctx_key"):
            assert resolve_credential("openai") == "ctx_key"


def test_env_fallback_disabled_also_blocks_alias_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKLLM_CREDENTIAL_TEAM_A", "sk-a")

    with config_context(allow_env_fallback=False):
        with pytest.raises(MissingCredentialError) as excinfo:
            resolve_credential("openai", alias="team_a")

        assert "environment fallback is disabled" in str(excinfo.value)


def test_env_fallback_can_be_disabled_ambiently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKLLM_ALLOW_ENV_FALLBACK", "0")
    monkeypatch.setenv("OPENAI_API_KEY", "env_key")

    assert resolve_allow_env_fallback() is False
    with pytest.raises(MissingCredentialError) as excinfo:
        resolve_credential("openai")

    message = str(excinfo.value)
    assert "environment fallback is disabled" in message

    with config_context(allow_env_fallback=True):
        assert resolve_credential("openai") == "env_key"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0", False),
        ("false", False),
        ("No", False),
        ("off", False),
        ("", True),
        ("1", True),
        ("true", True),
    ],
)
def test_ambient_env_fallback_switch_parsing(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool
) -> None:
    monkeypatch.setenv("SKLLM_ALLOW_ENV_FALLBACK", raw)

    assert resolve_allow_env_fallback() is expected


def test_disabled_env_fallback_leaves_non_secret_settings_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama:11434")

    with config_context(allow_env_fallback=False):
        assert resolve_base_url("ollama") == "http://ollama:11434"


def test_base_url_layering(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "http://env/v1")

    with config_context(openai_base_url="http://ctx/v1"):
        assert (
            resolve_base_url("openai", explicit="http://param/v1") == "http://param/v1"
        )
        assert resolve_base_url("openai") == "http://ctx/v1"

    assert resolve_base_url("openai") == "http://env/v1"
    assert resolve_base_url("anthropic") is None


def test_overrides_beat_stored_model_params() -> None:
    stored = {"temperature": 0.0, "max_tokens": 100}

    with config_context(overrides={"careful": {"temperature": 0.9}}):
        assert resolve_overrides("careful") == {"temperature": 0.9}
        assert resolve_model_params("careful", stored) == {
            "temperature": 0.9,
            "max_tokens": 100,
        }
        assert resolve_model_params("fast", stored) == stored
        assert resolve_model_params(None, stored) == stored

    assert resolve_model_params("careful", stored) == stored
    assert stored == {"temperature": 0.0, "max_tokens": 100}


def test_nested_overrides_merge_per_param() -> None:
    with (
        config_context(overrides={"careful": {"temperature": 0.9, "top_p": 0.5}}),
        config_context(overrides={"careful": {"temperature": 0.1}}),
    ):
        assert resolve_overrides("careful") == {"temperature": 0.1, "top_p": 0.5}


def test_client_options_merge_per_provider() -> None:
    with config_context(client_options={"openai": {"organization": "org-1"}}):
        with config_context(client_options={"openai": {"project": "proj-1"}}):
            assert resolve_client_options("openai") == {
                "organization": "org-1",
                "project": "proj-1",
            }
        assert resolve_client_options("anthropic") == {}

    assert resolve_client_options("openai") == {}


def test_transport_settings() -> None:
    assert resolve_timeout() == DEFAULT_TIMEOUT
    assert resolve_max_retries() == DEFAULT_MAX_RETRIES

    with config_context(timeout=5.0, max_retries=0):
        assert resolve_timeout() == 5.0
        assert resolve_max_retries() == 0


def test_pricing_lookup_by_full_model_string_or_bare_id() -> None:
    with config_context(pricing={"gpt-5": {"input": 1.25, "output": 10.0}}):
        assert resolve_price("openai::gpt-5") == Price(input=1.25, output=10.0)
        assert resolve_price("gpt-5") == Price(input=1.25, output=10.0)
        assert resolve_price("openai::gpt-4o") is None

    with config_context(pricing={"openai::gpt-5": Price(2.0, 8.0)}):
        assert resolve_price("openai::gpt-5") == Price(input=2.0, output=8.0)

    assert resolve_price("gpt-5") is None


@pytest.mark.parametrize(
    ("settings", "error"),
    [
        ({"openai_apikey": "sk-x"}, ValueError),
        ({"_api_key": "sk-x"}, ValueError),
        ({"openai_api_key": 1}, TypeError),
        ({"credentials": {"team_a": 1}}, TypeError),
        ({"credentials": ["team_a"]}, TypeError),
        ({"overrides": {"careful": 0.9}}, TypeError),
        ({"pricing": {"gpt-5": {"input": 1.0}}}, ValueError),
        ({"pricing": {"gpt-5": 1.0}}, TypeError),
        ({"pricing": {"gpt-5": {"input": "1.0", "output": 2.0}}}, TypeError),
        ({"allow_env_fallback": "no"}, TypeError),
        ({"timeout": 0}, ValueError),
        ({"max_retries": -1}, ValueError),
        ({"max_retries": 1.5}, TypeError),
    ],
)
def test_invalid_settings_are_rejected(
    settings: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error), config_context(**settings):
        pass


def test_unknown_setting_error_lists_the_valid_forms() -> None:
    with pytest.raises(ValueError) as excinfo, config_context(openai_apikey="sk-x"):
        pass

    message = str(excinfo.value)
    assert "openai_apikey" in message
    assert "<provider>_api_key" in message
    assert "credentials" in message


def test_config_context_never_writes_to_the_environment() -> None:
    before = dict(os.environ)

    with config_context(openai_api_key="sk-a", credentials={"team_a": "sk-b"}):
        assert dict(os.environ) == before

    assert dict(os.environ) == before


def test_config_context_is_isolated_across_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env_key")
    barrier = threading.Barrier(3, timeout=5)
    resolved: dict[str, str | None] = {}

    def run(name: str, key: str) -> None:
        with config_context(openai_api_key=key):
            barrier.wait()
            resolved[name] = resolve_credential("openai")

    threads = [
        threading.Thread(target=run, args=("a", "sk-a")),
        threading.Thread(target=run, args=("b", "sk-b")),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5)

    assert resolved == {"a": "sk-a", "b": "sk-b"}
    assert resolve_credential("openai") == "env_key"


def test_context_reaches_threads_only_through_a_copied_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env_key")
    resolved: dict[str, str | None] = {}

    def record(name: str) -> None:
        resolved[name] = resolve_credential("openai")

    with config_context(openai_api_key="ctx_key"):
        plain = threading.Thread(target=record, args=("plain",))
        context = contextvars.copy_context()
        copied = threading.Thread(target=context.run, args=(record, "copied"))
        for thread in (plain, copied):
            thread.start()
            thread.join(timeout=5)

    assert resolved == {"plain": "env_key", "copied": "ctx_key"}


def test_config_context_is_isolated_across_asyncio_tasks() -> None:
    async def worker(key: str) -> str | None:
        with config_context(openai_api_key=key):
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            return resolve_credential("openai")

    async def main() -> list[str | None]:
        return list(await asyncio.gather(worker("sk-a"), worker("sk-b")))

    assert asyncio.run(main()) == ["sk-a", "sk-b"]


def test_config_context_reaches_child_tasks_and_worker_threads() -> None:
    async def resolve() -> str | None:
        return resolve_credential("openai")

    async def main() -> list[str | None]:
        with config_context(openai_api_key="ctx_key"):
            in_tasks = await asyncio.gather(
                *(asyncio.create_task(resolve()) for _ in range(2))
            )
            in_thread = await asyncio.to_thread(resolve_credential, "openai")
        return [*in_tasks, in_thread]

    assert asyncio.run(main()) == ["ctx_key", "ctx_key", "ctx_key"]
