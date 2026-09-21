import logging
import os

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
    HnswParameters, 
)

from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv

from src.observability.tracing import trace_step
from src.providers.vector_store.base import VectorStore


load_dotenv()

logger = logging.getLogger(__name__)


class AzureAISearchVectorStore(VectorStore):
    """
    Azure AI Search implementation of the VectorStore contract.

    Why this class exists:
        The core RAG pipeline should depend on a generic vector-store
        abstraction rather than directly depending on Azure AI Search.

        This class contains all Azure-specific index creation, document
        upload, and vector-search logic.

    Current responsibilities:
        - create the Azure AI Search index
        - store document chunks and embedding vectors
        - perform dense vector similarity search

    Future responsibilities:
        Hybrid search and metadata filtering can be added here without
        changing the core RAG orchestration layer.

    Security:
        Azure AI Search credentials are loaded from environment variables.
        API keys are never written to application logs.

    Required environment variables:
        AZURE_SEARCH_ENDPOINT
        AZURE_SEARCH_API_KEY
        AZURE_SEARCH_INDEX_NAME
    """

    VECTOR_PROFILE_NAME = "vector-profile"
    HNSW_CONFIG_NAME = "hnsw-config"
    VECTOR_FIELD_NAME = "embedding"

    def __init__(self, embedding_dimension: int,) -> None:
        """
        Initialize Azure AI Search clients.

        Args:
            embedding_dimension:
                Number of dimensions produced by the active embedding model.
                The search index vector field must use exactly the same
                dimension.

        Raises:
            ValueError:
                If required Azure AI Search configuration is missing or
                embedding_dimension is invalid.
        """

        if embedding_dimension <= 0:
            raise ValueError(
                "embedding_dimension must be greater than zero"
            )

        self.endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        self.api_key = os.getenv("AZURE_SEARCH_API_KEY")
        self.index_name = os.getenv("AZURE_SEARCH_INDEX_NAME","rag-index",)
        self.embedding_dimension = (embedding_dimension)

        missing = [
            name
            for name, value in {
                "AZURE_SEARCH_ENDPOINT": self.endpoint,
                "AZURE_SEARCH_API_KEY": self.api_key,
            }.items()
            if not value
        ]

        if missing:
            raise ValueError(
                "Missing Azure AI Search configuration: "
                + ", ".join(missing)
            )

        credential = AzureKeyCredential(self.api_key)

        self.index_client = SearchIndexClient(endpoint=self.endpoint, credential=credential,)

        self.search_client = SearchClient( endpoint=self.endpoint, index_name=self.index_name, credential=credential,)

        logger.info(
            "Azure AI Search vector store initialized | "
            "index=%s | dimension=%s",
            self.index_name,
            self.embedding_dimension,
        )

    def create_index(self,) -> None:
        """
        Create or update the Azure AI Search index.

        The vector index schema is defined in code rather than manually
        in the Azure portal. This makes the project reproducible and
        allows another developer to recreate the environment from the
        repository.

        Index fields:
            id:
                Unique chunk identifier and Azure Search document key.

            text:
                Original chunk text. Marked searchable so keyword/hybrid
                search can be added later.

            source:
                Original PDF filename. Filterable for future company/document
                filtering.

            page:
                PDF page number.

            chunk_index:
                Chunk sequence within the page.

            token_count:
                Number of tokens contained in the chunk.

            embedding:
                Dense embedding vector used for semantic retrieval.

        Vector search:
            HNSW is configured as the approximate nearest-neighbor algorithm.

        Raises:
            RuntimeError:
                If Azure AI Search cannot create or update the index.
        """

        logger.info(
            "Creating Azure AI Search index | index=%s",
            self.index_name,
        )

        fields = [
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
            ),

            SearchableField(
                name="text",
                type=SearchFieldDataType.String,
            ),

            SimpleField(
                name="source",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
            ),

            SimpleField(
                name="page",
                type=SearchFieldDataType.Int32,
                filterable=True,
                sortable=True,
            ),

            SimpleField(
                name="chunk_index",
                type=SearchFieldDataType.Int32,
                filterable=True,
            ),

            SimpleField(
                name="token_count",
                type=SearchFieldDataType.Int32,
                filterable=True,
            ),

            SearchField(
                name=self.VECTOR_FIELD_NAME,
                type=SearchFieldDataType.Collection(
                    SearchFieldDataType.Single
                ),
                searchable=True,
                vector_search_dimensions=(
                    self.embedding_dimension
                ),
                vector_search_profile_name=(
                    self.VECTOR_PROFILE_NAME
                ),
            ),
        ]

        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name=self.HNSW_CONFIG_NAME,
                    # parameters=HnswParameters(metric="cosine",), - cosine similarity
                ),
            ],
            profiles=[
                VectorSearchProfile(
                    name=self.VECTOR_PROFILE_NAME,
                    algorithm_configuration_name=(
                        self.HNSW_CONFIG_NAME
                    ),
                ),
            ],
        )

        index = SearchIndex(name=self.index_name,  fields=fields, vector_search=vector_search,)

        try:
            with trace_step(
                "azure_search_create_index"
            ):
                self.index_client.create_or_update_index(
                    index
                )

        except Exception as exc:
            logger.exception(
                "Azure AI Search index creation failed | "
                "index=%s",
                self.index_name,
            )

            raise RuntimeError(
                f"Failed to create Azure AI Search index "
                f"'{self.index_name}'"
            ) from exc

        logger.info(
            "Azure AI Search index ready | index=%s",
            self.index_name,
        )

    def add_documents(self, documents: list[dict],) -> None:
        """
        Upload embedded document chunks to Azure AI Search.

        Args:
            documents:
                List of embedded chunk dictionaries.

                Each document must contain:
                    - chunk_id
                    - text
                    - source
                    - page
                    - chunk_index
                    - token_count
                    - embedding

        Design choice:
            The local file path is intentionally not stored in the search
            index because machine-specific file-system paths are not useful
            retrieval metadata.

        Raises:
            ValueError:
                If no documents are supplied or required fields are missing.

            RuntimeError:
                If Azure AI Search rejects one or more documents.
        """

        if not documents:
            raise ValueError(
                "documents must contain at least one item"
            )

        required_fields = {
            "chunk_id",
            "text",
            "source",
            "page",
            "chunk_index",
            "token_count",
            "embedding",
        }

        azure_documents = []

        for document in documents:

            missing_fields = (
                required_fields
                - document.keys()
            )

            if missing_fields:
                raise ValueError(
                    "Document is missing required fields: "
                    + ", ".join(
                        sorted(missing_fields)
                    )
                )

            embedding = document["embedding"]

            if len(embedding) != (
                self.embedding_dimension
            ):
                raise ValueError(
                    "Embedding dimension mismatch for "
                    f"chunk '{document['chunk_id']}': "
                    f"expected "
                    f"{self.embedding_dimension}, "
                    f"received {len(embedding)}"
                )

            azure_documents.append(
                {
                    "id": document["chunk_id"],
                    "text": document["text"],
                    "source": document["source"],
                    "page": document["page"],
                    "chunk_index": document["chunk_index"],
                    "token_count": document["token_count"],
                    "embedding": embedding,
                }
            )

        logger.info(
            "Uploading documents to Azure AI Search | "
            "count=%s | index=%s",
            len(azure_documents),
            self.index_name,
        )

        try:
            with trace_step(
                "azure_search_document_upload"
            ):
                results = (
                    self.search_client
                    .upload_documents(
                        documents=azure_documents
                    )
                )

        except Exception as exc:
            logger.exception(
                "Azure AI Search upload request failed | "
                "count=%s",
                len(azure_documents),
            )

            raise RuntimeError(
                "Failed to upload documents to "
                "Azure AI Search"
            ) from exc

        failures = [
            result
            for result in results
            if not result.succeeded
        ]

        if failures:

            failed_keys = [
                result.key
                for result in failures
            ]

            logger.error(
                "Some Azure AI Search documents failed | "
                "failed_count=%s | keys=%s",
                len(failures),
                failed_keys,
            )

            raise RuntimeError(
                f"{len(failures)} document(s) failed "
                "during Azure AI Search indexing"
            )

        logger.info(
            "Azure AI Search upload completed | "
            "uploaded=%s",
            len(results),
        )

    def search(self, query_vector: list[float], top_k: int = 5,) -> list[dict]:
        """
        Retrieve document chunks using dense vector similarity.

        Args:
            query_vector:
                Embedding representation of the user's query.

            top_k:
                Maximum number of chunks to retrieve.

        Returns:
            Retrieved chunks ordered by Azure AI Search vector similarity.

            Each result contains:
                - chunk_id
                - text
                - source
                - page
                - chunk_index
                - token_count
                - score

        Raises:
            ValueError:
                If the query vector dimension is incorrect or top_k is
                invalid.

            RuntimeError:
                If the Azure AI Search query fails.
        """

        if not query_vector:
            raise ValueError(
                "query_vector must not be empty"
            )

        if len(query_vector) != (
            self.embedding_dimension
        ):
            raise ValueError(
                "Query embedding dimension mismatch: "
                f"expected "
                f"{self.embedding_dimension}, "
                f"received {len(query_vector)}"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields=self.VECTOR_FIELD_NAME,
        )

        logger.info(
            "Executing vector search | index=%s | top_k=%s",
            self.index_name,
            top_k,
        )

        try:
            with trace_step(
                "azure_search_vector_query"
            ):
                results = (
                    self.search_client.search(
                        search_text=None,
                        vector_queries=[
                            vector_query
                        ],
                        select=[
                            "id",
                            "text",
                            "source",
                            "page",
                            "chunk_index",
                            "token_count",
                        ],
                        top=top_k,
                    )
                )

                retrieved_documents = [
                    {
                        "chunk_id": result["id"],
                        "text": result["text"],
                        "source": result["source"],
                        "page": result["page"],
                        "chunk_index": result["chunk_index"],
                        "token_count": result["token_count"],
                        "score": result.get("@search.score"),
                    }
                    for result in results
                ]

        except Exception as exc:
            logger.exception(
                "Azure AI Search vector query failed"
            )

            raise RuntimeError(
                "Failed to execute vector search"
            ) from exc

        logger.info(
            "Vector search completed | results=%s",
            len(retrieved_documents),
        )

        return retrieved_documents


    def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Perform Azure AI Search hybrid retrieval.

        Hybrid retrieval executes two retrieval strategies in parallel:

            1. BM25 keyword search over the searchable text field.
            2. Dense vector similarity search over the embedding field.

        Azure AI Search then combines the independent ranked result sets using
        Reciprocal Rank Fusion (RRF).

        The ``@search.score`` returned by hybrid retrieval is an RRF score.
        It should NOT be compared numerically with the similarity score from
        dense-only vector retrieval.

        Compare ranking metrics such as Hit@K, MRR, Recall@K, and Precision@K
        instead.

        Args:
            query_text:
                Original user query used by Azure's full-text search.

            query_vector:
                Embedding of the user query.

            top_k:
                Number of fused results returned to the RAG pipeline.

        Returns:
            Ranked list of retrieved chunks.

        Raises:
            ValueError:
                If the query, vector, or top_k value is invalid.

            RuntimeError:
                If Azure AI Search fails.
        """

        if not query_text or not query_text.strip():
            raise ValueError(
                "query_text must not be empty"
            )

        if not query_vector:
            raise ValueError(
                "query_vector must not be empty"
            )

        if len(query_vector) != self.embedding_dimension:
            raise ValueError(
                "Query vector dimension does not match the "
                f"configured index dimension. "
                f"Expected={self.embedding_dimension}, "
                f"received={len(query_vector)}"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        # Give the vector side a broader candidate set before RRF fusion.
        candidate_k = max(
            top_k * 5,
            20,
        )

        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=candidate_k,
            fields=self.VECTOR_FIELD_NAME,
        )

        try:

            with trace_step(
                "azure_search_hybrid_query"
            ):

                results = self.search_client.search(
                    search_text=query_text,
                    vector_queries=[
                        vector_query
                    ],
                    select=[
                        "id",
                        "text",
                        "source",
                        "page",
                        "chunk_index",
                        "token_count",
                    ],
                    top=top_k,
                )

                retrieved = []

                for result in results:

                    retrieved.append(
                        {
                            "chunk_id": result[
                                "id"
                            ],
                            "text": result[
                                "text"
                            ],
                            "source": result[
                                "source"
                            ],
                            "page": result[
                                "page"
                            ],
                            "chunk_index": result[
                                "chunk_index"
                            ],
                            "token_count": result[
                                "token_count"
                            ],
                            "score": result.get(
                                "@search.score"
                            ),
                        }
                    )

                logger.info(
                    "Hybrid retrieval completed | "
                    "query=%r | top_k=%s | "
                    "vector_candidates=%s | returned=%s",
                    query_text[:100],
                    top_k,
                    candidate_k,
                    len(retrieved),
                )

                return retrieved

        except Exception as exc:

            logger.exception(
                "Azure AI Search hybrid query failed"
            )

            raise RuntimeError(
                "Hybrid retrieval failed"
            ) from exc