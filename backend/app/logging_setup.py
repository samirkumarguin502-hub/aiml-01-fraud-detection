"""Structured logging that cannot carry transaction data: only whitelisted scalar fields."""
from __future__ import annotations

import logging

SAFE_FIELDS = {"request_id", "method", "route", "status", "latency_ms", "rows", "flagged",
               "model_state", "error_code", "exc_type", "batch_id", "count"}


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("fraudapp")
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(h)
    logger.setLevel(getattr(logging, level, logging.INFO))
    logger.propagate = True
    return logger


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields) -> None:
    safe = {k: v for k, v in fields.items() if k in SAFE_FIELDS and isinstance(v, (str, int, float, bool))}
    logger.log(level, "%s %s", event, " ".join(f"{k}={v}" for k, v in sorted(safe.items())))
