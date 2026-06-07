from typing import Optional, Union, Any, List, Dict, Mapping
from skllm.config import SKLLMConfig as _Config
from skllm.llm.minimax.completion import get_chat_completion
from skllm.utils import extract_json_key
from skllm.llm.base import BaseTextCompletionMixin, BaseClassifierMixin
import json


class MiniMaxMixin:
    """A mixin class that provides MiniMax API key to other classes."""

    _prefer_json_output = False

    def _set_keys(self, key: Optional[str] = None) -> None:
        """Set the MiniMax API key."""
        self.key = key

    def _get_minimax_key(self) -> str:
        """Get the MiniMax key from the class or config."""
        key = self.key
        if key is None:
            key = _Config.get_minimax_key()
        if key is None:
            raise RuntimeError("MiniMax API key was not found")
        return key


class MiniMaxTextCompletionMixin(MiniMaxMixin, BaseTextCompletionMixin):
    """A mixin class that provides text completion capabilities using the MiniMax API."""

    def _get_chat_completion(
        self,
        model: str,
        messages: Union[str, List[Dict[str, str]]],
        system_message: Optional[str] = None,
        **kwargs: Any,
    ):
        """Gets a chat completion from the MiniMax API.

        Parameters
        ----------
        model : str
            The model to use.
        messages : Union[str, List[Dict[str, str]]]
            Input messages to use.
        system_message : Optional[str]
            A system message to use.

        Returns
        -------
        completion : dict
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, list):
            messages = [
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                for msg in messages
            ]

        completion = get_chat_completion(
            messages=messages,
            key=self._get_minimax_key(),
            model=model,
            system=system_message,
            json_response=self._prefer_json_output,
            **kwargs,
        )
        return completion

    def _convert_completion_to_str(self, completion: Mapping[str, Any]):
        """Converts MiniMax API completion to string."""
        try:
            if hasattr(completion, "choices"):
                return str(completion.choices[0].message.content)
            return str(completion["choices"][0]["message"]["content"])
        except Exception as e:
            print(f"Error converting completion to string: {str(e)}")
            return ""


class MiniMaxClassifierMixin(MiniMaxTextCompletionMixin, BaseClassifierMixin):
    """A mixin class that provides classification capabilities using MiniMax API."""

    _prefer_json_output = True

    def _extract_out_label(self, completion: Mapping[str, Any], **kwargs) -> str:
        """Extracts the label from a MiniMax API completion."""
        try:
            content = self._convert_completion_to_str(completion)
            if not self._prefer_json_output:
                return content.strip()
            try:
                label = extract_json_key(content, "label")
                if label is not None:
                    return label
            except Exception:
                pass
            return ""
        except Exception as e:
            print(f"Error extracting label: {str(e)}")
            return ""
