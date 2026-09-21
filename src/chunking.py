from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
from dotenv import load_dotenv
import os
import logging
import re
load_dotenv() # Loads variables from .env
logger = logging.getLogger(__name__)

encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(encoding.encode(text))


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Create a token-aware recursive text splitter.

    We prefer paragraph and newline boundaries before falling back
    to smaller separators.
    """

    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 150)),
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

def make_safe_chunk_id(
    source: str,
    page: int,
    chunk_index: int,
) -> str:
    """
    Create a deterministic document key that is safe for vector stores
    such as Azure AI Search.

    Why this exists:
        Azure AI Search document keys only allow letters, digits,
        underscore, dash, or equals sign. File names frequently contain
        spaces, periods, parentheses, and other characters that cannot be
        used directly as index keys.

    Design choice:
        The ID remains human-readable and deterministic so the same source
        page/chunk generates the same key across repeated ingestion runs.

    Args:
        source:
            Original PDF filename.

        page:
            Source PDF page number.

        chunk_index:
            Chunk sequence within the page.

    Returns:
        Vector-store-safe chunk identifier.
    """

    source_stem = source.rsplit(".", 1)[0]

    safe_source = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        source_stem,
    )

    safe_source = safe_source.strip("_")

    return (
        f"{safe_source}"
        f"_p{page}"
        f"_c{chunk_index}"
    )


def chunk_pages(
    pages: list[dict],
) -> list[dict]:
    """
    Split page-level documents into retrieval chunks.

    Why this exists:
        Embedding an entire financial report or even very large pages would
        reduce retrieval precision. Chunking creates smaller semantic units
        that can be independently embedded and retrieved.

    Important design decision:
        Each PDF page is chunked independently. Text from separate pages is
        never merged before splitting. This preserves reliable page-level
        source attribution.

    Noise filtering:
        Chunks below MIN_CHUNK_TOKENS are removed after manual inspection
        showed that extremely short chunks were primarily trademark notices,
        isolated footer text, and appendix labels.

    Args:
        pages:
            Page dictionaries returned by the ingestion pipeline.

    Returns:
        A list of chunk dictionaries containing:
            - chunk_id
            - text
            - source
            - page
            - file_path
            - chunk_index
            - token_count
    """

    if not pages:
        logger.warning(
            "Chunking requested with empty page list"
        )
        return []

    splitter = create_text_splitter()

    all_chunks = []
    filtered_count = 0

    logger.info(
        "Starting chunking | page_count=%s",
        len(pages),
    )

    for page in pages:

        try:
            page_chunks = splitter.split_text(
                page["text"]
            )

            for chunk_index, chunk_text in enumerate(
                page_chunks,
                start=1,
            ):
                token_count = count_tokens(
                    chunk_text
                )

                if token_count < int(os.getenv("MIN_CHUNK_TOKENS", 25)):
                    filtered_count += 1

                    logger.debug(
                        "Filtered tiny chunk | source=%s | page=%s | tokens=%s",
                        page.get("source"),
                        page.get("page"),
                        token_count,
                    )

                    continue

                chunk_id = make_safe_chunk_id(
                                source=page["source"],
                                page=page["page"],
                                chunk_index=chunk_index,
                            )

                all_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                        "source": page["source"],
                        "page": page["page"],
                        "file_path": page["file_path"],
                        "chunk_index": chunk_index,
                        "token_count": token_count,
                    }
                )

        except KeyError as exc:
            logger.exception(
                "Page metadata missing required field | page=%s",
                page,
            )

            raise ValueError(
                "Page document is missing required metadata"
            ) from exc

    logger.info(
        "Chunking completed | chunks=%s | filtered=%s",
        len(all_chunks),
        filtered_count,
    )

    return all_chunks