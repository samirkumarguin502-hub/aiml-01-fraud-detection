"""Structured logging that excludes transaction data and sensitive values.

Only whitelisted scalar fields are allowed.
String values are sanitized to prevent log injection.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any


SAFE_FIELDS = {
    "request_id",
    "method",
    "route",
    "status",
    "latency_ms",
    "rows",
    "flagged",
    "model_state",
    "error_code",
    "exc_type",
    "batch_id",
    "count",
}

MAX_EVENT_LENGTH = 64
MAX_VALUE_LENGTH = 256

SAFE_LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

SAFE_EVENT_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]+$")
SAFE_STRING_PATTERN = re.compile(r"[\r\n\t\x00-\x1f\x7f]")


def _clean_string(value: str, max_length: int) -> str:
    """Sanitize a string before adding it to a log record."""

    value = SAFE_STRING_PATTERN.sub("_", value)
    value = value[:max_length]

    return value


def _safe_value(value: Any) -> str | int | float | bool | None:
    """Return a safe scalar value or None."""

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value

    if isinstance(value, str):
        return _clean_string(value, MAX_VALUE_LENGTH)

    return None


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure the application logger."""

    logger = logging.getLogger("fraudapp")

    normalized_level = (
        level.upper()
        if isinstance(level, str)
        else "INFO"
    )

    numeric_level = SAFE_LOG_LEVELS.get(
        normalized_level,
        logging.INFO,
    )

    if not logger.handlers:
        handler = logging.StreamHandler()

        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(message)s"
            )
        )

        logger.addHandler(handler)

    logger.setLevel(numeric_level)

    # Prevent duplicate records through the root logger.
    logger.propagate = False

    return logger


def log_event(
    logger: logging.Logger,
    event: str,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Write an event using only approved scalar fields."""

    if not isinstance(event, str):
        event = "unknown_event"

    event = _clean_string(event, MAX_EVENT_LENGTH)

    if not SAFE_EVENT_PATTERN.fullmatch(event):
        event = "invalid_event"

    safe_fields = {}

    for key, value in fields.items():
        if key not in SAFE_FIELDS:
            continue

        safe_value = _safe_value(value)

        if safe_value is None:
            continue

        safe_fields[key] = safe_value

    details = " ".join(
        f"{key}={value}"
        for key, value in sorted(safe_fields.items())
    )

    if not isinstance(level, int) or isinstance(level, bool):
        level = logging.INFO

    logger.log(
        level,
        "%s %s",
        event,
        details,
    )