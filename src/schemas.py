from typing import Literal

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    """
    One piece of retrieved evidence supporting the generated answer.

    Fields:
        statement:
            Short factual statement supported by the retrieved context.

        chunk_id:
            Identifier of the retrieved chunk containing the evidence.
    """

    statement: str = Field(
        ...,
        description=(
            "A factual statement supported by the retrieved context."
        ),
    )

    chunk_id: str = Field(
        ...,
        description=(
            "Chunk identifier containing the supporting evidence."
        ),
    )


class SourceItem(BaseModel):
    """
    Source metadata returned with the generated answer.

    Financial-document answers should be traceable back to the original
    PDF and page rather than relying only on opaque model-generated text.
    """

    document: str = Field(
        ...,
        description="Original source PDF filename.",
    )

    page: int = Field(
        ...,
        description="PDF page number containing the evidence.",
    )

    chunk_id: str = Field(
        ...,
        description="Retrieved chunk identifier.",
    )


class RAGResponse(BaseModel):
    """
    Structured response returned by the RAG pipeline.
    """

    answer: str = Field(
        ...,
        description=(
            "Answer grounded only in the supplied retrieved context."
        ),
    )

    evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description=(
            "Supporting factual statements mapped to retrieved chunks."
        ),
    )

    sources: list[SourceItem] = Field(
        default_factory=list,
        description=(
            "Source documents and pages used to support the answer."
        ),
    )

    confidence: Literal[
        "high",
        "medium",
        "low",
    ] = Field(
        ...,
        description=(
            "Evidence-strength category derived from the supplied context."
        ),
    )

    insufficient_context: bool = Field(
        ...,
        description=(
            "True when the retrieved context does not contain enough "
            "information to answer reliably."
        ),
    )