import logging
import os

from src.providers.llm.azure_openai import (
    AzureOpenAILLMProvider,
)
from src.providers.llm.base import (
    LLMProvider,
)


logger = logging.getLogger(__name__)


def get_llm_provider() -> LLMProvider:
    """
    Create the configured LLM provider.

    Why this factory exists:
        Model selection is centralized so the RAG pipeline remains independent
        from provider-specific construction logic.

    Current provider:
        azure_openai

    Future implementations could include:
        - OpenAI
        - Anthropic
        - local models
        - other enterprise model gateways

    Returns:
        Configured LLMProvider implementation.

    Raises:
        ValueError:
            If LLM_PROVIDER references an unsupported implementation.
    """

    provider_name = os.getenv(
        "LLM_PROVIDER",
        "azure_openai",
    ).lower()

    logger.info(
        "Creating LLM provider | provider=%s",
        provider_name,
    )

    if provider_name == "azure_openai":
        return AzureOpenAILLMProvider()

    raise ValueError(
        f"Unsupported LLM provider: {provider_name}"
    )