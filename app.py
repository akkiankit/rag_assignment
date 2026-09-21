"""
Streamlit demo application for the financial RAG system.

Purpose
-------
This module provides a lightweight presentation layer for demonstrating
the RAG pipeline during interviews or stakeholder reviews.

The application intentionally contains very little business logic.
All retrieval, prompting, grounding, and generation behavior remains
inside the reusable RAG pipeline under src/.

This separation keeps:
    - UI concerns in app.py
    - retrieval/generation logic in src/
    - evaluation logic in evaluation/

That makes the application easier to maintain, test, and evolve.
"""

import logging

import streamlit as st

from src.observability.logging_config import configure_logging
from src.pipeline_factory import create_rag_pipeline


configure_logging()

logger = logging.getLogger(__name__)


@st.cache_resource
def get_pipeline():
    """
    Create and cache the RAG pipeline.

    Streamlit reruns the script whenever the user interacts with the UI.
    Without caching, the application would recreate Azure/OpenAI/Search
    clients repeatedly.

    Returns:
        Configured RAG pipeline instance.
    """

    logger.info(
        "Initializing RAG pipeline for Streamlit application"
    )

    return create_rag_pipeline(
        top_k=5
    )


def render_sidebar() -> None:
    """
    Render current RAG configuration.

    The sidebar makes the architecture and benchmark configuration visible
    during a demo so the audience knows exactly which system version is
    being tested.
    """

    st.sidebar.header(
        "RAG Configuration"
    )

    st.sidebar.write(
        "**Version:** baseline_v1"
    )

    st.sidebar.write(
        "**Parser:** PyMuPDF"
    )

    st.sidebar.write(
        "**Chunk Size:** 1000 tokens"
    )

    st.sidebar.write(
        "**Chunk Overlap:** 150 tokens"
    )

    st.sidebar.write(
        "**Retrieval:** Dense Vector Search"
    )

    st.sidebar.write(
        "**Top-K:** 5"
    )

    st.sidebar.write(
        "**Embedding:** Azure OpenAI"
    )

    st.sidebar.write(
        "**Vector Store:** Azure AI Search"
    )

    st.sidebar.write(
        "**Conversation Memory:** Disabled"
    )


def render_response(
    response,
) -> None:
    """
    Render structured RAG response.

    Displays:
        - final answer
        - confidence
        - insufficient-context warning
        - evidence statements
        - source document / page / chunk traceability
    """

    st.subheader(
        "Answer"
    )

    st.write(
        response.answer
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Confidence",
            response.confidence,
        )

    with col2:
        st.metric(
            "Insufficient Context",
            "Yes"
            if response.insufficient_context
            else "No",
        )

    if response.insufficient_context:
        st.warning(
            "The retrieved context was insufficient to support "
            "a confident answer."
        )

    if response.evidence:

        st.subheader(
            "Evidence"
        )

        for index, evidence in enumerate(
            response.evidence,
            start=1,
        ):
            st.write(
                f"{index}. {evidence.statement}"
            )

            st.caption(
                f"Chunk: {evidence.chunk_id}"
            )

    if response.sources:

        st.subheader(
            "Sources"
        )

        for index, source in enumerate(
            response.sources,
            start=1,
        ):

            st.write(
                f"{index}. **{source.document}** "
                f"— Page {source.page}"
            )

            st.caption(
                f"Chunk: {source.chunk_id}"
            )


def main() -> None:
    """
    Application entry point.

    The UI delegates all RAG functionality to the reusable pipeline.
    """

    logger.info(
        "RAG demo application started"
    )

    st.set_page_config(
        page_title="Financial RAG Demo",
        page_icon="📊",
        layout="wide",
    )

    st.title(
        "Financial Document RAG"
    )

    st.caption(
        "Grounded question answering over quarterly financial filings "
        "using Azure OpenAI and Azure AI Search."
    )

    render_sidebar()

    pipeline = get_pipeline()

    st.subheader(
        "Ask a Question"
    )

    question = st.text_area(
        "Question",
        placeholder=(
            "Example: What were Apple's total net sales "
            "for the quarter ended July 1, 2023?"
        ),
        height=100,
    )

    if st.button(
        "Ask",
        type="primary",
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

            return

        logger.info(
            "Processing user query"
        )

        try:

            with st.spinner(
                "Retrieving evidence and generating answer..."
            ):

                response = pipeline.ask(
                    question
                )

            render_response(
                response
            )

        except Exception:

            logger.exception(
                "RAG query failed"
            )

            st.error(
                "The request could not be completed. "
                "Please check the application logs."
            )


if __name__ == "__main__":

    try:
        main()

    except Exception:

        logger.exception(
            "Application terminated unexpectedly"
        )

        raise