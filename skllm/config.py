import os
from typing import Optional

_OPENAI_KEY_VAR = "SKLLM_CONFIG_OPENAI_KEY"
_OPENAI_ORG_VAR = "SKLLM_CONFIG_OPENAI_ORG"
_AZURE_API_BASE_VAR = "SKLLM_CONFIG_AZURE_API_BASE"
_AZURE_API_VERSION_VAR = "SKLLM_CONFIG_AZURE_API_VERSION"
_GOOGLE_PROJECT = "GOOGLE_CLOUD_PROJECT"
_GPT_URL_VAR = "SKLLM_CONFIG_GPT_URL"
_ANTHROPIC_KEY_VAR = "SKLLM_CONFIG_ANTHROPIC_KEY"
_GGUF_DOWNLOAD_PATH = "SKLLM_CONFIG_GGUF_DOWNLOAD_PATH"
_GGUF_MAX_GPU_LAYERS = "SKLLM_CONFIG_GGUF_MAX_GPU_LAYERS"
_GGUF_VERBOSE = "SKLLM_CONFIG_GGUF_VERBOSE"
_MINIMAX_KEY_VAR = "SKLLM_CONFIG_MINIMAX_KEY"


class SKLLMConfig:
    @staticmethod
    def set_gpt_key(key: str) -> None:
        """Sets the GPT key.

        Parameters
        ----------
        key : str
            GPT key.
        """
        os.environ[_OPENAI_KEY_VAR] = key

    def set_gpt_org(key: str) -> None:
        """Sets the GPT organization ID.

        Parameters
        ----------
        key : str
            GPT organization ID.
        """
        os.environ[_OPENAI_ORG_VAR] = key

    @staticmethod
    def set_openai_key(key: str) -> None:
        """Sets the OpenAI key.

        Parameters
        ----------
        key : str
            OpenAI key.
        """
        os.environ[_OPENAI_KEY_VAR] = key

    @staticmethod
    def get_openai_key() -> Optional[str]:
        """Gets the OpenAI key.

        Returns
        -------
        Optional[str]
            OpenAI key.
        """
        return os.environ.get(_OPENAI_KEY_VAR, None)

    @staticmethod
    def set_openai_org(key: str) -> None:
        """Sets OpenAI organization ID.

        Parameters
        ----------
        key : str
            OpenAI organization ID.
        """
        os.environ[_OPENAI_ORG_VAR] = key

    @staticmethod
    def get_openai_org() -> str:
        """Gets the OpenAI organization ID.

        Returns
        -------
        str
            OpenAI organization ID.
        """
        return os.environ.get(_OPENAI_ORG_VAR, "")

    @staticmethod
    def get_azure_api_base() -> str:
        """Gets the API base for Azure.

        Returns
        -------
        str
            URL to be used as the base for the Azure API.
        """
        base = os.environ.get(_AZURE_API_BASE_VAR, None)
        if base is None:
            raise RuntimeError("Azure API base is not set")
        return base

    @staticmethod
    def set_azure_api_base(base: str) -> None:
        """Set the API base for Azure.

        Parameters
        ----------
        base : str
            URL to be used as the base for the Azure API.
        """
        os.environ[_AZURE_API_BASE_VAR] = base

    @staticmethod
    def set_azure_api_version(ver: str) -> None:
        """Set the API version for Azure.

        Parameters
        ----------
        ver : str
            Azure API version.
        """
        os.environ[_AZURE_API_VERSION_VAR] = ver

    @staticmethod
    def get_azure_api_version() -> str:
        """Gets the API version for Azure.

        Returns
        -------
        str
            Azure API version.
        """
        return os.environ.get(_AZURE_API_VERSION_VAR, "2023-05-15")

    @staticmethod
    def get_google_project() -> Optional[str]:
        """Gets the Google Cloud project ID.

        Returns
        -------
        Optional[str]
            Google Cloud project ID.
        """
        return os.environ.get(_GOOGLE_PROJECT, None)

    @staticmethod
    def set_google_project(project: str) -> None:
        """Sets the Google Cloud project ID.

        Parameters
        ----------
        project : str
            Google Cloud project ID.
        """
        os.environ[_GOOGLE_PROJECT] = project

    @staticmethod
    def set_gpt_url(url: str):
        """Sets the GPT URL.

        Parameters
        ----------
        url : str
            GPT URL.
        """
        os.environ[_GPT_URL_VAR] = url

    @staticmethod
    def get_gpt_url() -> Optional[str]:
        """Gets the GPT URL.

        Returns
        -------
        Optional[str]
            GPT URL.
        """
        return os.environ.get(_GPT_URL_VAR, None)
    
    @staticmethod
    def set_anthropic_key(key: str) -> None:
        """Sets the Anthropic key.

        Parameters
        ----------
        key : str
            Anthropic key.
        """
        os.environ[_ANTHROPIC_KEY_VAR] = key
    
    @staticmethod
    def get_anthropic_key() -> Optional[str]:
        """Gets the Anthropic key.

        Returns
        -------
        Optional[str]
            Anthropic key.
        """
        return os.environ.get(_ANTHROPIC_KEY_VAR, None)

    @staticmethod
    def set_minimax_key(key: str) -> None:
        """Sets the MiniMax API key.

        Parameters
        ----------
        key : str
            MiniMax API key.
        """
        os.environ[_MINIMAX_KEY_VAR] = key

    @staticmethod
    def get_minimax_key() -> Optional[str]:
        """Gets the MiniMax API key.

        Returns
        -------
        Optional[str]
            MiniMax API key.
        """
        return os.environ.get(_MINIMAX_KEY_VAR, None)

    @staticmethod
    def reset_gpt_url():
        """Resets the GPT URL."""
        os.environ.pop(_GPT_URL_VAR, None)

    @staticmethod
    def get_gguf_download_path() -> str:
        """Gets the path to store the downloaded GGUF files."""
        default_path = os.path.join(os.path.expanduser("~"), ".skllm", "gguf")
        return os.environ.get(_GGUF_DOWNLOAD_PATH, default_path)

    @staticmethod
    def get_gguf_max_gpu_layers() -> int:
        """Gets the maximum number of layers to use for the GGUF model."""
        return int(os.environ.get(_GGUF_MAX_GPU_LAYERS, 0))

    @staticmethod
    def set_gguf_max_gpu_layers(n_layers: int):
        """Sets the maximum number of layers to use for the GGUF model."""
        if not isinstance(n_layers, int):
            raise ValueError("n_layers must be an integer")
        if n_layers < -1:
            n_layers = -1
        os.environ[_GGUF_MAX_GPU_LAYERS] = str(n_layers)

    @staticmethod
    def set_gguf_verbose(verbose: bool):
        """Sets the verbosity of the GGUF model."""
        if not isinstance(verbose, bool):
            raise ValueError("verbose must be a boolean")
        os.environ[_GGUF_VERBOSE] = str(verbose)

    @staticmethod
    def get_gguf_verbose() -> bool:
        """Gets the verbosity of the GGUF model."""
        return os.environ.get(_GGUF_VERBOSE, "False").lower() == "true"
