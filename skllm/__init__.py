from skllm.backends import register_backend
from skllm.classification import (
    DynamicFewShotClassifier,
    FewShotClassifier,
    MultiLabelFewShotClassifier,
    MultiLabelZeroShotClassifier,
    ZeroShotClassifier,
)
from skllm.clustering import ClusterLabeler
from skllm.config import MissingCredentialError, config_context
from skllm.extraction import StructuredExtractor
from skllm.scoring import RubricScorer
from skllm.tagging import NER
from skllm.text2text import Summarizer, Translator
from skllm.vectorization import Vectorizer

__version__ = "2.0.0"
__author__ = "Iryna Kondrashchenko, Oleh Kostromin"

__all__ = [
    "NER",
    "ClusterLabeler",
    "DynamicFewShotClassifier",
    "FewShotClassifier",
    "MissingCredentialError",
    "MultiLabelFewShotClassifier",
    "MultiLabelZeroShotClassifier",
    "RubricScorer",
    "StructuredExtractor",
    "Summarizer",
    "Translator",
    "Vectorizer",
    "ZeroShotClassifier",
    "__version__",
    "config_context",
    "register_backend",
]
