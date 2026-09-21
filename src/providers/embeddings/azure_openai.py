import logging
import os

from dotenv import load_dotenv
from openai import AzureOpenAI

from src.observability.tracing import (
    trace_step,
)
from src.providers.embeddings.base import (
    EmbeddingProvider,
)


load_dotenv()

logger = logging.getLogger(__name__)


class AzureOpenAIEmbeddingProvider(EmbeddingProvider):
    """
    Azure OpenAI implementation of the embedding provider contract.

    The provider encapsulates Azure-specific SDK calls so the rest of the
    application interacts only with the generic EmbeddingProvider interface.

    Environment variables:
        AZURE_OPENAI_ENDPOINT
        AZURE_OPENAI_API_KEY
        AZURE_OPENAI_API_VERSION
        AZURE_OPENAI_EMBEDDING_DEPLOYMENT

    Current model characteristics:
        The configured deployment produces 3072-dimensional embeddings.

    Security:
        Credentials are loaded from environment variables and are never
        written to logs.
    """

    EMBEDDING_DIMENSION = 3072

    def __init__(self) -> None:
        """
        Initialize the Azure OpenAI client and validate configuration.

        Raises:
            ValueError:
                If a required environment variable is missing.
        """

        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

        missing = [
            name
            for name, value in {
                "AZURE_OPENAI_ENDPOINT": endpoint,
                "AZURE_OPENAI_API_KEY": api_key,
                "AZURE_OPENAI_API_VERSION": api_version,
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": deployment,
            }.items()
            if not value
        ]

        if missing:
            raise ValueError(
                "Missing Azure OpenAI configuration: "
                + ", ".join(missing)
            )

        self.deployment = deployment

        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
        )

        logger.info(
            "Azure embedding provider initialized | dimension=%s",
            self.dimension,
        )

    @property
    def dimension(self) -> int:
        """
        Return the embedding dimension required by the vector index.
        """

        return self.EMBEDDING_DIMENSION

    def embed_texts(self, texts: list[str],) -> list[list[float]]:
        """
        Generate embeddings for a batch of document texts.

        Args:
            texts:
                Non-empty list of document text values.

        Returns:
            Embedding vectors in the same order as the input texts.

        Raises:
            ValueError:
                If an empty input collection is supplied.

            RuntimeError:
                If the Azure OpenAI request fails.
        """

        if not texts:
            raise ValueError(
                "texts must contain at least one item"
            )

        logger.debug(
            "Generating document embeddings | count=%s",
            len(texts),
        )

        try:
            with trace_step(
                "azure_embedding_batch"
            ):
                response = (
                    self.client.embeddings.create(
                        model=self.deployment,
                        input=texts,
                    )
                )

        except Exception as exc:
            logger.exception(
                "Azure embedding request failed | count=%s",
                len(texts),
            )

            raise RuntimeError(
                "Failed to generate document embeddings"
            ) from exc

        vectors = [
            item.embedding
            for item in response.data
        ]

        if len(vectors) != len(texts):
            raise RuntimeError(
                "Embedding response count does not match input count"
            )

        return vectors

    def embed_query(self, query: str,) -> list[float]:
        """
        Generate an embedding for a single user query.

        Args:
            query:
                Natural-language retrieval query.

        Returns:
            Dense vector representation of the query.

        Raises:
            ValueError:
                If the query is empty.

            RuntimeError:
                If Azure OpenAI fails to generate an embedding.
        """

        if not query or not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        try:
            with trace_step("azure_query_embedding"):
                response = (
                    self.client.embeddings.create(
                        model=self.deployment,
                        input=query,
                    )
                )

        except Exception as exc:
            logger.exception(
                "Azure query embedding failed"
            )

            raise RuntimeError(
                "Failed to generate query embedding"
            ) from exc

        vector = response.data[0].embedding

        if len(vector) != self.dimension:
            logger.warning(
                "Unexpected embedding dimension | expected=%s | actual=%s",
                self.dimension,
                len(vector),
            )

        return vector