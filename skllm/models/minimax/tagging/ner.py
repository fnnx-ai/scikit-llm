from skllm.models._base.tagger import ExplainableNER as _ExplainableNER
from skllm.llm.minimax.mixin import (
    MiniMaxTextCompletionMixin as _MiniMaxTextCompletionMixin,
)
from typing import Optional, Dict
from skllm.model_constants import MINIMAX_MODEL


class MiniMaxExplainableNER(_ExplainableNER, _MiniMaxTextCompletionMixin):
    """Named Entity Recognition model using MiniMax API for explainable entity extraction."""

    def __init__(
        self,
        entities: Dict[str, str],
        display_predictions: bool = False,
        sparse_output: bool = True,
        model: str = MINIMAX_MODEL,
        key: Optional[str] = None,
        num_workers: int = 1,
    ) -> None:
        """
        Named entity recognition using MiniMax API.

        Parameters
        ----------
        entities : dict
            Dictionary of entities to recognize, with keys as entity
            names and values as descriptions.
        display_predictions : bool, optional
            Whether to display predictions, by default False.
        sparse_output : bool, optional
            Whether to generate a sparse representation, by default True.
        model : str, optional
            Model to use, by default "MiniMax-M2.7".
        key : Optional[str], optional
            Estimator-specific API key; if None, retrieved from the
            global config, by default None.
        num_workers : int, optional
            Number of workers (threads) to use, by default 1.
        """
        self._set_keys(key)
        self.model = model
        self.entities = entities
        self.display_predictions = display_predictions
        self.sparse_output = sparse_output
        self.num_workers = num_workers
