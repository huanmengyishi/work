"""Structured domain failures used at architectural recovery boundaries."""

from __future__ import annotations


class DeepAgentError(Exception):
    """Base class for locally classified Deep Agent failures."""


class ToolExecutionError(DeepAgentError, RuntimeError):
    """A managed tool failed after permission evaluation."""


class ContextCompactionError(DeepAgentError, RuntimeError):
    """A context reduction stage could not produce a protocol-safe result."""


class SessionInconsistencyError(DeepAgentError, ValueError):
    """Persisted Session data violates the supported bounded schema."""


class ArtifactLifecycleError(DeepAgentError, ValueError):
    """An artifact lifecycle transition or receipt is invalid."""


class IntentJournalError(DeepAgentError, RuntimeError):
    """A durable intent-journal operation failed safely."""


class ModelCompatibilityError(DeepAgentError, RuntimeError):
    """A model response violates the supported DeepSeek response shape."""

    def __init__(self, message: str, *, http_attempt_count: int = 0) -> None:
        super().__init__(message)
        self.http_attempt_count = max(0, int(http_attempt_count))


__all__ = [
    "ArtifactLifecycleError",
    "ContextCompactionError",
    "DeepAgentError",
    "IntentJournalError",
    "ModelCompatibilityError",
    "SessionInconsistencyError",
    "ToolExecutionError",
]
