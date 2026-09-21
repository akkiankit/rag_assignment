import logging
import re
from pathlib import Path

import pymupdf


logger = logging.getLogger(__name__)


def clean_text(
    text: str,
) -> str:
    """
    Apply conservative cleanup to PDF-extracted text.

    Args:
        text:
            Raw text extracted from a PDF page.

    Returns:
        Cleaned text with basic whitespace normalization applied.
    """

    if not text:
        return ""

    text = text.replace("\x00", "")

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def extract_pdf_pages(
    pdf_path: Path,
) -> list[dict]:
    """
    Extract text from a PDF one page at a time.

    Why page-level extraction is used:
        Preserving page boundaries allows retrieved chunks to be traced back
        to the exact source document and page number. This is important for
        auditability and source attribution in financial-document RAG.

    Args:
        pdf_path:
            Path to the PDF file.

    Returns:
        A list of dictionaries. Each dictionary contains:
            - text
            - page
            - source
            - file_path

    Raises:
        FileNotFoundError:
            If the supplied PDF path does not exist.

        RuntimeError:
            If PyMuPDF fails to open or process the document.
    """

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file does not exist: {pdf_path}"
        )

    logger.info(
        "Extracting PDF | source=%s",
        pdf_path.name,
    )

    pages = []

    try:
        with pymupdf.open(pdf_path) as document:

            for page_number, page in enumerate(
                document,
                start=1,
            ):
                raw_text = page.get_text("text")

                cleaned_text = clean_text(
                    raw_text
                )

                if not cleaned_text:
                    logger.debug(
                        "Skipping empty page | source=%s | page=%s",
                        pdf_path.name,
                        page_number,
                    )
                    continue

                pages.append(
                    {
                        "text": cleaned_text,
                        "page": page_number,
                        "source": pdf_path.name,
                        "file_path": str(pdf_path),
                    }
                )

    except Exception as exc:
        logger.exception(
            "PDF extraction failed | source=%s",
            pdf_path.name,
        )

        raise RuntimeError(
            f"Failed to extract PDF: {pdf_path.name}"
        ) from exc

    logger.info(
        "PDF extraction completed | source=%s | pages=%s",
        pdf_path.name,
        len(pages),
    )

    return pages


def load_pdf_directory(
    pdf_directory: str,
) -> list[dict]:
    """
    Load and extract all PDF files from a directory.

    Why this exists:
        This function acts as the corpus-level ingestion entry point. It
        discovers PDF files and processes each document independently so one
        malformed file does not prevent the remaining corpus from loading.

    Args:
        pdf_directory:
            Directory containing input PDF files.

    Returns:
        A combined list of page-level document dictionaries.

    Raises:
        FileNotFoundError:
            If the input directory does not exist.
    """

    directory = Path(pdf_directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"PDF directory does not exist: {directory}"
        )

    pdf_files = sorted(
        directory.glob("*.pdf")
    )

    logger.info(
        "Starting corpus ingestion | directory=%s | pdf_count=%s",
        directory,
        len(pdf_files),
    )

    all_pages = []

    for pdf_path in pdf_files:

        try:
            pages = extract_pdf_pages(
                pdf_path
            )

            all_pages.extend(
                pages
            )

        except Exception:
            logger.exception(
                "Skipping failed PDF | source=%s",
                pdf_path.name,
            )

    logger.info(
        "Corpus ingestion completed | pdf_count=%s | extracted_pages=%s",
        len(pdf_files),
        len(all_pages),
    )

    return all_pages