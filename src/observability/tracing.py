import logging
import time
from contextlib import contextmanager
from typing import Iterator


logger = logging.getLogger(__name__)


@contextmanager
def trace_step(
    step_name: str,
) -> Iterator[None]:
    """
    Measure and log the execution time of a pipeline step.
    A RAG request passes through several stages such as query embedding,
    retrieval, prompt construction, and LLM generation. When latency or
    failures occur, it is useful to know which stage was responsible.

    This project uses lightweight timing-based tracing for the assignment
    instead of introducing a full distributed tracing platform at the
    beginning.

    The abstraction is intentionally small so it can later be replaced
    with OpenTelemetry spans or Azure Application Insights without
    changing the RAG pipeline significantly.

    Args:
        step_name:
            Human-readable name for the operation being measured.

    Raises:
        Exception:
            Any exception raised inside the traced block is logged and then
            re-raised. The tracing layer does not suppress application errors.

    Example:
        with trace_step("query_embedding"):
            query_vector = embedding_provider.embed_query(query)
    """

    start_time = time.perf_counter()

    logger.debug(
        "Trace started | step=%s",
        step_name,
    )

    try:
        yield

    except Exception:
        elapsed = time.perf_counter() - start_time

        logger.exception(
            "Trace failed | step=%s | duration_seconds=%.3f",
            step_name,
            elapsed,
        )

        raise

    else:
        elapsed = time.perf_counter() - start_time

        logger.info(
            "Trace completed | step=%s | duration_seconds=%.3f",
            step_name,
            elapsed,
        )