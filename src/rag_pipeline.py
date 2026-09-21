import logging
from typing import Any

from src.observability.tracing import trace_step
from src.prompt_template.prompts import SYSTEM_PROMPT, build_user_prompt
from src.providers.embeddings.base import EmbeddingProvider
from src.providers.llm.base import LLMProvider
from src.providers.vector_store.base import VectorStore
from src.schemas import RAGResponse
from typing import Literal

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Orchestrate retrieval and grounded response generation.

    The individual provider classes have intentionally narrow
    responsibilities:

        EmbeddingProvider
            converts text into vectors

        VectorStore
            stores vectors and retrieves relevant chunks

        LLMProvider
            generates validated structured responses

    RAGPipeline coordinates those components into the online
    question-answering workflow.

    Request flow:
        user question
            -> query embedding
            -> vector retrieval
            -> prompt construction
            -> structured LLM generation
            -> validated RAGResponse

    The pipeline depends only on provider interfaces rather than
    Azure-specific implementations. This keeps orchestration logic
    independent from the underlying infrastructure.

    Args:
        embedding_provider:
            Provider used to embed user queries.

        vector_store:
            Search backend used to retrieve relevant document chunks.

        llm_provider:
            Language model provider used for structured generation.

        top_k:
            Number of chunks retrieved for each question.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        llm_provider: LLMProvider,
        top_k: int = 5,
        retrieval_mode: Literal[
            "dense",
            "hybrid",
        ] = "dense",
    ) -> None:

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.top_k = top_k
        self.retrieval_mode = (retrieval_mode)

        logger.info(
            "RAG pipeline initialized | top_k=%s",
            self.top_k,
        )

    def ask(self, query: str,) -> RAGResponse:
        """
        Answer a user question using retrieval-augmented generation.

        Args:
            query:
                Natural-language question about the indexed document corpus.

        Returns:
            Validated RAGResponse containing:
                - answer
                - evidence
                - sources
                - confidence
                - insufficient_context

        Raises:
            ValueError:
                If the query is empty.

            RuntimeError:
                If retrieval, prompt construction, or generation fails.
        """

        if not query or not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        clean_query = query.strip()

        logger.info(
            "Processing RAG query"
        )

        try:
            with trace_step(
                "rag_query_embedding"
            ):
                query_vector = (
                    self.embedding_provider
                    .embed_query(
                        clean_query
                    )
                )

            with trace_step("rag_retrieval"):
                # retrieved_chunks = (
                #     self.vector_store.search(
                #         query_vector=query_vector,
                #         top_k=self.top_k,
                #     )
                # )
                if self.retrieval_mode == "dense":
                    retrieved_chunks = (
                        self.vector_store.search(
                            query_vector=query_vector,
                            top_k=self.top_k,
                        )
                    )

                else:

                    retrieved_chunks = (
                        self.vector_store.hybrid_search(
                            query_text=query,
                            query_vector=query_vector,
                            top_k=self.top_k,
                        )
                    )

            if not retrieved_chunks:
                logger.warning(
                    "No chunks retrieved for query"
                )

                return RAGResponse(
                    answer=(
                        "The indexed documents do not contain "
                        "sufficient information to answer this question."
                    ),
                    evidence=[],
                    sources=[],
                    confidence="low",
                    insufficient_context=True,
                )

            logger.info(
                "Chunks retrieved | count=%s",
                len(retrieved_chunks),
            )

            with trace_step("rag_prompt_construction"):
                user_prompt = build_user_prompt(
                    query=clean_query,
                    retrieved_chunks=retrieved_chunks,
                )

            with trace_step("rag_generation"):
                response = (
                    self.llm_provider
                    .generate_structured_response(
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=user_prompt,
                    )
                )

        except Exception as exc:
            logger.exception(
                "RAG pipeline failed"
            )

            raise RuntimeError(
                "Failed to process RAG query"
            ) from exc

        logger.info(
            "RAG query completed | "
            "confidence=%s | insufficient_context=%s",
            response.confidence,
            response.insufficient_context,
        )

        return response