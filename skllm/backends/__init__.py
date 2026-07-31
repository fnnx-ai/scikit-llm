"""Provider backend adapters and the model-string registry."""

from skllm.backends.base import (
    Backend,
    BackendError,
    BaseBackend,
    Connection,
    Message,
    ResolvedConnection,
    Response,
    Usage,
    parse_json,
)
from skllm.backends.registry import (
    ModelRef,
    available_providers,
    get_backend,
    parse_model,
    register_backend,
)

__all__ = [
    "Backend",
    "BackendError",
    "BaseBackend",
    "Connection",
    "Message",
    "ModelRef",
    "ResolvedConnection",
    "Response",
    "Usage",
    "available_providers",
    "get_backend",
    "parse_json",
    "parse_model",
    "register_backend",
]
