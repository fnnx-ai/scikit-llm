from openai import OpenAI

MINIMAX_BASE_URL = "https://api.minimax.io/v1"


def set_credentials(key: str) -> OpenAI:
    """Set MiniMax credentials and return an OpenAI-compatible client.

    Parameters
    ----------
    key : str
        The MiniMax API key to use.

    Returns
    -------
    client : OpenAI
        An OpenAI client configured for MiniMax.
    """
    client = OpenAI(api_key=key, base_url=MINIMAX_BASE_URL)
    return client
