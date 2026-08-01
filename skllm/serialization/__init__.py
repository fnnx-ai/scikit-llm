"""FNNX bundle packaging: the PyFunc wrapper plus save/load."""

from skllm.serialization.load import InferenceModel, load
from skllm.serialization.save import save

__all__ = ["InferenceModel", "load", "save"]
