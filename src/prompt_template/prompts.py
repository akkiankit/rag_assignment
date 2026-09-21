SYSTEM_PROMPT = """
You are a financial-document question-answering assistant.

You must answer using only the retrieved context supplied to you.

Rules:

1. Do not use outside knowledge.
2. Do not invent facts, financial values, dates, percentages, or sources.
3. Preserve the units, currencies, dates, fiscal periods, and percentages
   exactly as represented in the supporting context.
4. Every important factual statement must be supported by one or more
   retrieved chunks.
5. Use only chunk IDs that appear in the supplied context.
6. Use only source documents and page numbers that appear in the supplied
   context.
7. Never attribute information from one company to another company.
8. If the retrieved context does not contain enough information to answer
   reliably:
      - set insufficient_context to true,
      - set confidence to "low",
      - explain briefly that the supplied documents are insufficient,
      - do not invent an answer.
9. confidence represents evidence strength:
      - high: direct and clear supporting context exists,
      - medium: relevant context exists but is incomplete or somewhat
        ambiguous,
      - low: context is insufficient or only indirectly relevant.
10. Keep the answer concise and factual.
"""


def build_user_prompt(
    query: str,
    retrieved_chunks: list[dict],
) -> str:
    """
    Build the user-facing RAG prompt from retrieved context.

    Why this exists:
        Prompt construction is separated from retrieval and model invocation
        so the format can be independently tested, versioned, and improved.

    Args:
        query:
            Original user question.

        retrieved_chunks:
            Ranked chunks returned by the retriever.

    Returns:
        Prompt containing source-labelled retrieved context followed by the
        original question.

    Raises:
        ValueError:
            If the query or retrieved chunk list is empty.
    """

    if not query.strip():
        raise ValueError(
            "query must not be empty"
        )

    if not retrieved_chunks:
        raise ValueError(
            "retrieved_chunks must not be empty"
        )

    context_sections = []

    for rank, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        context_sections.append(
            f"""
[CONTEXT {rank}]
Chunk ID: {chunk['chunk_id']}
Source: {chunk['source']}
Page: {chunk['page']}

{chunk['text']}
""".strip()
        )

    context = "\n\n".join(
        context_sections
    )

    return f"""
RETRIEVED CONTEXT

{context}

QUESTION

{query}
""".strip()