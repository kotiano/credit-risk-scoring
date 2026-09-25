"""Structured logging configuration (key=value records, one line per event)."""

from __future__ import annotations

import logging
import sys
from typing import Any


class KeyValueFormatter(logging.Formatter):
    """Render records as ``ts level logger msg key=value ...`` for grep-able logs."""

    def format(self, record: logging.LogRecord) -> str:
        base = (
            f"{self.formatTime(record, '%Y-%m-%dT%H:%M:%S')} {record.levelname:<7} "
            f"{record.name} | {record.getMessage()}"
        )
        context: dict[str, Any] = getattr(record, "context", {})
        if context:
            base += " | " + " ".join(f"{k}={v}" for k, v in context.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(KeyValueFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, message: str, **context: Any) -> None:
    """Log an INFO event with structured context fields."""
    logger.info(message, extra={"context": context})
