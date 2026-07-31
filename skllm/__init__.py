from skllm.backends import register_backend
from skllm.classification import (
    FewShotClassifier,
    MultiLabelFewShotClassifier,
    MultiLabelZeroShotClassifier,
    ZeroShotClassifier,
)
from skllm.config import MissingCredentialError, config_context

__version__ = "2.0.0"
__author__ = "Iryna Kondrashchenko, Oleh Kostromin"

__all__ = [
    "FewShotClassifier",
    "MissingCredentialError",
    "MultiLabelFewShotClassifier",
    "MultiLabelZeroShotClassifier",
    "ZeroShotClassifier",
    "__version__",
    "config_context",
    "register_backend",
]
