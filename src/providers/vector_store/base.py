from abc import ABC, abstractmethod


class VectorStore(ABC):
    """
    Contract for vector-storage and retrieval implementations.

    Why this abstraction exists:
        Retrieval logic should not depend directly on Azure AI Search or any
        other specific vector database.

        The current assignment uses Azure AI Search, while alternative
        implementations such as Qdrant could later implement the same
        interface.

    This is a lightweight application of the adapter/ports pattern commonly
    used to reduce infrastructure coupling in larger systems.
    """

    @abstractmethod
    def create_index(
        self,
    ) -> None:
        """
        Create or validate the underlying search index.
        """
        raise NotImplementedError

    @abstractmethod
    def add_documents(
        self,
        documents: list[dict],
    ) -> None:
        """
        Persist document chunks and their embedding vectors.
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Retrieve the most similar chunks for a query vector.
        """
        raise NotImplementedError

    @abstractmethod
    def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Perform hybrid keyword + vector retrieval.

        Hybrid retrieval combines:

            - lexical/BM25 text matching
            - dense vector similarity

        The concrete backend is responsible for combining or fusing the
        independent ranking signals.

        Args:
            query_text:
                Original user query for lexical retrieval.

            query_vector:
                Embedded representation of the same user query.

            top_k:
                Number of final fused results returned to the caller.

        Returns:
            Ranked retrieved chunks.
        """
        raise NotImplementedError