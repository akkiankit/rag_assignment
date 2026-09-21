from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """
    Contract implemented by all embedding providers.

    Why this abstraction exists:
        The RAG pipeline should depend on embedding capabilities rather than
        directly depending on a specific vendor SDK.

        The current implementation uses Azure OpenAI, but another provider can
        later be introduced without changing retrieval or orchestration logic.

    Important consideration:
        Changing embedding models generally requires regenerating document
        embeddings and rebuilding the vector index because query and document
        vectors must exist in the same embedding space.
    """

    @abstractmethod
    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Generate dense embeddings for multiple document texts.

        Args:
            texts:
                Text values to embed.

        Returns:
            One vector for each input text.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        """
        Generate an embedding for a retrieval query.

        Args:
            query:
                Natural-language query.

        Returns:
            Dense embedding vector.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(
        self,
    ) -> int:
        """
        Return the number of dimensions produced by the model.

        The vector database requires this value when creating a vector field.
        """
        raise NotImplementedError