from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from .events import sanitize_for_log


LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR"})


def _root_logger() -> logging.Logger:
    return logging.getLogger()


class SafeDiagnosticFormatter(logging.Formatter):
    """Render bounded JSON diagnostics without serializing arbitrary objects."""

    def format(self, record: logging.LogRecord) -> str:
        try:
            rendered = record.getMessage()
        except Exception:
            rendered = "diagnostic_format_error"
        message = sanitize_for_log(rendered)
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname if record.levelname in LOG_LEVELS else "WARNING",
            "logger": sanitize_for_log(record.name),
            "message": message,
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_logging(level: str = "WARNING") -> None:
    """Configure the process diagnostic stream with a safe bounded formatter."""

    normalized = str(level or "WARNING").upper()
    if normalized not in LOG_LEVELS:
        raise ValueError(f"unsupported log level: {normalized}")
    root = _root_logger()
    root.setLevel(getattr(logging, normalized))
    handler = logging.StreamHandler(sys.stderr)
    setattr(handler, "_deep_agent_safe", True)
    handler.setLevel(getattr(logging, normalized))
    handler.setFormatter(SafeDiagnosticFormatter())
    for existing in tuple(root.handlers):
        root.removeHandler(existing)
        existing.close()
    root.addHandler(handler)


__all__ = ["LOG_LEVELS", "SafeDiagnosticFormatter", "configure_logging"]
