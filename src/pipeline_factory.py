import logging

from src.providers.embeddings.factory import (
    get_embedding_provider,
)
from src.providers.llm.factory import (
    get_llm_provider,
)
from src.providers.vector_store.factory import (
    get_vector_store,
)
from src.rag_pipeline import RAGPipeline


logger = logging.getLogger(__name__)


def create_rag_pipeline(top_k: int = 5, retrieval_mode: Literal["dense", "hybrid",] = "dense",) -> RAGPipeline:
    """
    Construct the complete RAG pipeline using configured providers.

    Application entry points and notebooks should not need to know how
    individual infrastructure components are instantiated.

    Dependency construction is centralized here so provider changes can
    be made without modifying calling code.

    Args:
        top_k:
            Number of document chunks retrieved for each question.

    Returns:
        Fully initialized RAGPipeline.
    """

    logger.info(
        "Creating RAG pipeline"
    )

    embedding_provider = (
        get_embedding_provider()
    )

    vector_store = get_vector_store(
        embedding_dimension=(
            embedding_provider.dimension
        )
    )

    llm_provider = (
        get_llm_provider()
    )

    pipeline = RAGPipeline(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        llm_provider=llm_provider,
        top_k=top_k,
        retrieval_mode=(
            retrieval_mode
        ),
    )

    logger.info(
        "RAG pipeline created successfully"
    )

    return pipeline