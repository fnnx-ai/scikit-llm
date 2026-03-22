import re
from typing import Dict, List, Optional
from skllm.llm.minimax.credentials import set_credentials
from skllm.utils import retry
from skllm.model_constants import MINIMAX_MODEL


def _strip_think_tags(text: str) -> str:
    """Strip <think>...</think> tags from MiniMax model responses.

    Also handles unclosed <think> tags (e.g. when the model runs out of tokens
    while still in the thinking phase).
    """
    # First strip properly closed think tags
    text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text)
    # Then strip unclosed think tags (content truncated during thinking)
    text = re.sub(r"<think>[\s\S]*$", "", text)
    return text.strip()


@retry(max_retries=3)
def get_chat_completion(
    messages: List[Dict],
    key: str,
    model: str = MINIMAX_MODEL,
    max_tokens: int = 1000,
    temperature: float = 0.0,
    system: Optional[str] = None,
    json_response: bool = False,
) -> dict:
    """Gets a chat completion from the MiniMax API via OpenAI-compatible endpoint.

    Parameters
    ----------
    messages : list
        Input messages to use.
    key : str
        The MiniMax API key to use.
    model : str, optional
        The MiniMax model to use.
    max_tokens : int, optional
        Maximum tokens to generate.
    temperature : float, optional
        Sampling temperature (0.0 to 1.0).
    system : str, optional
        System message to set the assistant's behavior.
    json_response : bool, optional
        Whether to request a JSON-formatted response.

    Returns
    -------
    response : dict
        The completion response from the API.
    """
    if not messages:
        raise ValueError("Messages list cannot be empty")
    if not isinstance(messages, list):
        raise TypeError("Messages must be a list")

    # Clamp temperature to MiniMax's supported range [0.0, 1.0]
    temperature = max(0.0, min(1.0, temperature))

    client = set_credentials(key)

    formatted_messages = []
    if system:
        if json_response:
            system = f"{system.rstrip('.')}. Respond in JSON format."
        formatted_messages.append({"role": "system", "content": system})
    elif json_response:
        formatted_messages.append(
            {"role": "system", "content": "Respond in JSON format."}
        )

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        formatted_messages.append({"role": role, "content": content})

    model_dict = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": formatted_messages,
    }

    if json_response:
        model_dict["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**model_dict)

    # Strip <think>...</think> tags from the response content
    if (
        response.choices
        and response.choices[0].message.content
        and isinstance(response.choices[0].message.content, str)
    ):
        response.choices[0].message.content = _strip_think_tags(
            response.choices[0].message.content
        )

    return response
