from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .config import AppConfig


class ErrorCategory(StrEnum):
    INTERRUPTED = "interrupted"
    PROTOCOL = "protocol"
    CONTEXT_LIMIT = "context_limit"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    TRANSIENT_NETWORK = "transient_network"
    TIMEOUT = "timeout"
    PERMISSION = "permission"
    RESOURCE_LIMIT = "resource_limit"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ResiliencePolicy:
    """Runtime recovery policy that deliberately does not duplicate HTTP retry.

    DeepSeekClient owns network retry, key rotation, and backoff.  Runtime uses
    this policy only for protocol correction and abnormal finish responses,
    both of which are safe because no unaccepted tool request is executed.
    """

    max_corrective_rounds: int = 2
    max_abnormal_finish_recoveries: int = 1

    @classmethod
    def from_config(cls, config: AppConfig) -> "ResiliencePolicy":
        return cls(
            max_corrective_rounds=_bounded_int(
                config.get("runtime.resilience.max_corrective_rounds", 2),
                default=2,
                minimum=0,
                maximum=8,
            ),
            max_abnormal_finish_recoveries=_bounded_int(
                config.get("runtime.resilience.max_abnormal_finish_recoveries", 1),
                default=1,
                minimum=0,
                maximum=4,
            ),
        )

    @staticmethod
    def classify(error: BaseException) -> ErrorCategory:
        name = type(error).__name__.casefold()
        message = str(error).casefold()
        combined = f"{name} {message}"
        if "interrupt" in combined or "cancel" in combined or "keyboard" in combined:
            return ErrorCategory.INTERRUPTED
        if "context" in combined and any(item in combined for item in ("overflow", "length", "limit")):
            return ErrorCategory.CONTEXT_LIMIT
        if any(item in combined for item in ("401", "403", "authentication", "api key", "unauthorized")):
            return ErrorCategory.AUTHENTICATION
        if "429" in combined or "rate limit" in combined:
            return ErrorCategory.RATE_LIMIT
        if "timeout" in combined or "timed out" in combined:
            return ErrorCategory.TIMEOUT
        if any(item in combined for item in ("connection", "network", "dns", "temporarily unavailable")):
            return ErrorCategory.TRANSIENT_NETWORK
        if any(item in combined for item in ("permission", "denied", "not allowed")):
            return ErrorCategory.PERMISSION
        if any(item in combined for item in ("budget", "resource", "memoryerror", "disk full", "too large")):
            return ErrorCategory.RESOURCE_LIMIT
        if any(item in combined for item in ("protocol", "finish_reason", "tool-call")):
            return ErrorCategory.PROTOCOL
        return ErrorCategory.UNKNOWN


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


__all__ = ["ErrorCategory", "ResiliencePolicy"]
