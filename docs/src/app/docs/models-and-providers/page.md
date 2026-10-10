---
title: Models and providers
---

Every estimator that talks to a language model takes a `model` string of the form `provider::model_id`. The prefix selects a provider, and the part after the separator travels to that provider verbatim.

A provider is the service that answers the request. A backend is the adapter object registered under its prefix, which is what `skllm.register_backend` adds and replaces.

```python
from skllm import ZeroShotClassifier

ZeroShotClassifier(model="openai::gpt-5.6-sol")
ZeroShotClassifier(model="anthropic::claude-sonnet-5")
ZeroShotClassifier(model="google::gemini-2.5-flash")
```

A string without a separator is read as OpenAI, so `model="gpt-5"` and `model="openai::gpt-5"` name the same model. Every estimator that issues chat requests defaults to `openai::gpt-5.6-sol`. `Vectorizer` defaults to `openai::text-embedding-3-small`, since it issues embedding requests instead.

The string is parsed when the estimator resolves its backend, which happens at `fit` and again before each `predict` or `transform`. A malformed string with an empty provider or an empty model id raises `ValueError` there. An unregistered prefix does the same, with a message listing the prefixes that are registered.

## Built-in providers

Five providers ship with the library. Their backends are imported on first use, so a missing extra only ever raises for the provider a model string actually asks for.

| Prefix | Install | Chat | Structured outputs | Embeddings | Credentials |
| --- | --- | --- | --- | --- | --- |
| `openai::` | `scikit-llm` | yes | native JSON Schema | yes | `OPENAI_API_KEY` |
| `ollama::` | `scikit-llm` | yes | native, OpenAI-compatible | yes | none required |
| `anthropic::` | `scikit-llm[anthropic]` | yes | forced tool use | no | `ANTHROPIC_API_KEY` |
| `google::` | `scikit-llm[google]` | yes | response schema | yes | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| `litellm::` | `scikit-llm[litellm]` | yes | native where LiteLLM reports support | yes | each provider's own variables |

Structured output is how classification, extraction, tagging and scoring constrain an answer, and each provider exposes it differently. OpenAI and Ollama take a JSON Schema as the response format. Anthropic gets the schema as the input schema of a tool it is forced to call, and the tool arguments are the answer. Google receives it as a response schema alongside a JSON mime type. LiteLLM asks its own catalog whether the target model supports response schemas, and where it does not, the schema is folded into the last message as an instruction to answer with matching JSON. The estimator validates the answer either way, so a provider that ignores the constraint produces a sample failure rather than an unusable prediction.

Anthropic offers no embedding models, and its backend has no embeddings path, so a `Vectorizer` pointed at it fails with a `BackendError` saying so. The same applies indirectly to [Dynamic few-shot classification](/docs/dynamic-few-shot-classification), which retrieves its examples by embedding them. On a provider with no default embedding model, pass a `vectorizer` explicitly. That means `anthropic`, but also `ollama`, `litellm` and custom backends, where the model id depends on the deployment.

```python
from skllm import DynamicFewShotClassifier, Vectorizer

DynamicFewShotClassifier(
    model="anthropic::claude-sonnet-5",
    vectorizer=Vectorizer(model="openai::text-embedding-3-small"),
)
```

{% callout title="Google means the Gemini API" %}
The `google::` prefix is the Gemini API through the `google-genai` package, authenticated with an API key. Vertex AI is a different service with a different authentication model. Reach it through `litellm::vertex_ai/...`.
{% /callout %}

## Custom endpoints

Any OpenAI-compatible endpoint is `openai::` plus a `base_url`. The model id is whatever that endpoint serves.

```python
from skllm import Summarizer

Summarizer(model="openai::my-model", base_url="http://localhost:8000/v1")
```

`base_url` resolves through the same layers as every other connection setting: the estimator parameter, then an active `config_context(<provider>_base_url=...)`, then the environment variable — `OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL` and so on, or `OLLAMA_HOST` for Ollama. See [Credentials and configuration](/docs/credentials-and-configuration) for the full order.

Ollama is the OpenAI backend pointed at Ollama's own OpenAI-compatible endpoint, defaulting to `http://localhost:11434/v1`.

```python
Summarizer(model="ollama::llama3")
```

{% callout title="How OLLAMA_HOST is normalized" %}
A host without a port gets Ollama's default port 11434, as its own clients assume. An `https` scheme is read as a fronted deployment on 443. A missing `/v1` suffix is appended, so `OLLAMA_HOST=my-server` becomes `http://my-server:11434/v1`.
{% /callout %}

Ollama requires no credential. An endpoint that does authenticate, such as a proxy in front of a shared server, still accepts one through `api_key=`, `config_context(ollama_api_key=...)` or `OLLAMA_API_KEY`.

Azure OpenAI, Bedrock and Vertex AI go through `litellm::`, where the model id is LiteLLM's own and the credentials are the ones LiteLLM reads for that provider.

```python
Summarizer(model="litellm::azure/my-deployment")
Summarizer(model="litellm::vertex_ai/gemini-2.5-pro")
```

{% callout title="LiteLLM keeps its own credentials" %}
The `litellm::` backend declares no canonical credential variable, so no key is required of it. One supplied through `api_key=`, `config_context(litellm_api_key=...)` or `LITELLM_API_KEY` is forwarded to LiteLLM. Everything else — `AZURE_API_KEY`, `AWS_*`, application default credentials — is read by LiteLLM itself, from the environment, exactly as its own documentation describes.
{% /callout %}

## Provider request parameters

`model_params` is a mapping passed to the provider verbatim: temperature, token limits, reasoning effort, thinking budgets, whatever that provider takes. Nothing is validated against an allow-list, so an unsupported parameter surfaces the provider's own error rather than a scikit-llm one.

```python
ZeroShotClassifier(
    model="openai::gpt-5",
    model_params={"reasoning_effort": "high"},
)
```

Where the SDK takes a fixed set of keyword arguments, as OpenAI's and Anthropic's do, a parameter it does not name is routed through the request body instead. A deployment-specific knob such as Ollama's `num_ctx` therefore reaches the provider rather than failing locally as a `TypeError`.

scikit-llm injects two defaults of its own, and only for parameters the caller did not set: `temperature: 0` on every provider, plus `reasoning_effort: "none"` on OpenAI, which exposes a request parameter for it. They are advisory. A model that rejects one has that default stripped, the request retried without it, and the rejection remembered for that model for the rest of the process. Values in `model_params` are never rewritten or stripped that way.

Sampled voting drops the temperature default so the votes can differ. See [Zero-shot classification](/docs/zero-shot-classification) for `n_predictors`. Generation parameters can also be changed from the outside, without touching the estimator, through `config_context(overrides={...})` — see [Credentials and configuration](/docs/credentials-and-configuration).

## Registering a custom backend

A provider prefix is one backend object, registered at runtime with `skllm.register_backend`. The prefix must match `^[a-z][a-z0-9_-]*\Z`, which is lowercase letters, digits, underscores and hyphens, starting with a letter. Registering a prefix that already exists replaces it, built-ins included.

```python
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import skllm
from skllm.backends import BaseBackend, Message, ResolvedConnection, Response


class MyCorpBackend(BaseBackend):
    provider: ClassVar[str] = "mycorp"
    credential_env_var: ClassVar[str | None] = "MYCORP_API_KEY"
    transient_errors: ClassVar[tuple[type[BaseException], ...]] = (TimeoutError,)

    def _chat(
        self,
        model: str,
        messages: Sequence[Message],
        params: dict[str, Any],
        settings: ResolvedConnection,
    ) -> Response[str]:
        ...

    def _structured(
        self,
        model: str,
        messages: Sequence[Message],
        schema: Mapping[str, Any],
        params: dict[str, Any],
        settings: ResolvedConnection,
    ) -> Response[Any]:
        ...


skllm.register_backend("mycorp", MyCorpBackend())
skllm.ZeroShotClassifier(model="mycorp::internal-1")
```

`register_backend` accepts anything satisfying the `Backend` protocol: a `provider` attribute, a `credential_env_var` attribute, and `chat()`, `structured()` and `embed()` methods. An object without that surface is rejected with a `TypeError` naming what a backend has to expose.

Subclassing `BaseBackend` is the shorter route to that surface. It resolves credentials, base URL, timeout and retry budget for each call, injects the determinism defaults and runs the retry loop, then hands the subclass an already-resolved `ResolvedConnection` and the effective request parameters. The subclass implements `_chat` and `_structured`, and `_embed` if the provider embeds. One that does not implement `_embed` raises `BackendError` when something asks it to, which is how the Anthropic backend behaves.

Turns arrive as `Message(role, content)` pairs, and each hook returns a `Response`: the content, plus a `Usage` record of requests and token counts. That record is what feeds `usage_` and the cost estimates described in [Batch execution and error handling](/docs/batch-execution-and-error-handling).

`transient_errors` lists the exception types that earn another attempt. Only the backend knows which of its SDK's errors are transient, so the base class retries exactly what the backend declares, bounded by the `max_retries` setting. `credential_env_var` names the canonical key variable and is what a saved bundle declares as a required secret. Set it to `None` for a keyless provider, and the credential becomes optional rather than required.

{% callout title="Inspecting the registry" %}
`available_providers()` from `skllm.backends` returns the sorted prefixes currently registered, built-ins and runtime registrations together.
{% /callout %}
