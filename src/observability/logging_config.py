import logging
import os
from typing import Optional


DEFAULT_LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)s | "
    "%(name)s | "
    "%(message)s"
)


def configure_logging(
    level: Optional[str] = None,
) -> None:
    """
    Configure application-wide logging.

    Args:
        level:
            Optional logging level such as "DEBUG", "INFO", "WARNING",
            or "ERROR". If omitted, LOG_LEVEL is read from the environment.
            Defaults to "INFO".

    Example:
        configure_logging()

        logger = logging.getLogger(__name__)
        logger.info("Application started")
    """

    configured_level = (
        level
        or os.getenv("LOG_LEVEL", "INFO")
    ).upper()

    numeric_level = getattr(
        logging,
        configured_level,
        logging.INFO,
    )

    logging.basicConfig(
        level=numeric_level,
        format=DEFAULT_LOG_FORMAT,
        force=True,
    )

    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging configured | level=%s",
        configured_level,
    )