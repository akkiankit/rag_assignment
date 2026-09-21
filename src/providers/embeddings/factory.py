import logging
import os

from src.providers.embeddings.azure_openai import (
    AzureOpenAIEmbeddingProvider,
)
from src.providers.embeddings.base import (
    EmbeddingProvider,
)


logger = logging.getLogger(__name__)


def get_embedding_provider() -> EmbeddingProvider:
    """
    Create the configured embedding provider.

    Why this factory exists:
        Provider selection is centralized so application code does not need
        conditional logic for individual vendors.

    Current implementation:
        azure_openai

    Future implementations could include:
        - Hugging Face
        - Cohere
        - local embedding models

    Returns:
        Configured EmbeddingProvider implementation.

    Raises:
        ValueError:
            If EMBEDDING_PROVIDER references an unsupported provider.
    """

    provider_name = os.getenv("EMBEDDING_PROVIDER", "azure_openai",).lower()

    logger.info("Creating embedding provider | provider=%s", provider_name,)

    if provider_name == "azure_openai":
        return AzureOpenAIEmbeddingProvider()

    raise ValueError(
        f"Unsupported embedding provider: "
        f"{provider_name}"
    )