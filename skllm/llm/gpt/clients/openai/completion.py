from typing import TypeVar, Type
from pydantic import BaseModel
from openai import OpenAI
from skllm.llm.gpt.clients.openai.credentials import (
    set_azure_credentials,
    set_credentials,
)
from skllm.utils import retry
from skllm.model_constants import OPENAI_GPT_MODEL

T = TypeVar('T', bound=BaseModel)


@retry(max_retries=3)
def get_chat_completion(
    messages: dict,
    key: str,
    org: str,
    model: str = OPENAI_GPT_MODEL,
    api="openai",
    json_response=False,
):
    """Gets a chat completion from the OpenAI API.

    Parameters
    ----------
    messages : dict
        input messages to use.
    key : str
        The OPEN AI key to use.
    org : str
        The OPEN AI organization ID to use.
    model : str, optional
        The OPEN AI model to use.
    max_retries : int, optional
        The maximum number of retries to use. Defaults to 3.
    api : str
        The API to use. Must be one of "openai" or "azure". Defaults to "openai".

    Returns
    -------
    completion : dict
    """
    if api in ("openai", "custom_url"):
        client = set_credentials(key, org)
    elif api == "azure":
        client = set_azure_credentials(key, org)
    else:
        raise ValueError("Invalid API")
    model_dict: dict = {"model": model}
    if json_response and api == "openai":
        model_dict["response_format"] = {"type": "json_object"}
    if not model.startswith(("gpt-o", "gpt-5")):
        model_dict["temperature"] = 0.0
    completion = client.chat.completions.create(messages=messages, **model_dict)  # type: ignore
    return completion


@retry(max_retries=3)
def get_parsed_completion(
    messages: dict,
    output_model: Type[T],
    key: str,
    org: str,
    model: str = "gpt-3.5-turbo",
    api="openai",
) -> T:
    """Gets a chat completion parsed into the specified Pydantic model.

    Parameters
    ----------
    messages : dict
        input messages to use.
    output_model : Type[T]
        Pydantic model class to parse the response into.
    key : str
        The OPEN AI key to use.
    org : str
        The OPEN AI organization ID to use.
    model : str, optional
        The OPEN AI model to use. Defaults to "gpt-3.5-turbo".
    api : str
        The API to use. Must be one of "openai" or "azure". Defaults to "openai".

    Returns
    -------
    parsed_model : T
        Instance of the specified Pydantic model
    """
    if api in ("openai", "custom_url"):
        client = set_credentials(key, org)
    elif api == "azure":
        client = set_azure_credentials(key, org)
    else:
        raise ValueError("Invalid API")
    
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=output_model,
        temperature=0.0
    )
    
    return completion.choices[0].message.parsed
