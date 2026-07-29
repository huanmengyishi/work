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
    capability_backoff_enabled: bool = True
    max_capability_backoff_rounds: int = 8
    circuit_recovery_rounds: int = 4

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
            capability_backoff_enabled=bool(config.get("runtime.resilience.capability_backoff_enabled", True)),
            max_capability_backoff_rounds=_bounded_int(
                config.get("runtime.resilience.max_capability_backoff_rounds", 8),
                default=8,
                minimum=1,
                maximum=32,
            ),
            circuit_recovery_rounds=_bounded_int(
                config.get("runtime.resilience.circuit_recovery_rounds", 4),
                default=4,
                minimum=1,
                maximum=64,
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


@dataclass(frozen=True)
class CapabilityRecoveryDecision:
    allowed: bool
    action: str
    blocked_through_round: int
    reason: str


class CapabilityRecoveryController:
    """Connect persisted capability health to deterministic Runtime recovery."""

    MAX_RECORDS = 64

    def __init__(self, policy: ResiliencePolicy) -> None:
        self.policy = policy

    def before_call(
        self, convergence: dict[str, Any], capability: str, *, current_round: int
    ) -> CapabilityRecoveryDecision:
        records = self._records(convergence)
        record = records.get(capability)
        if not self.policy.capability_backoff_enabled or not isinstance(record, dict):
            return CapabilityRecoveryDecision(True, "execute", 0, "ready")
        blocked = _bounded_int(record.get("blocked_through_round"), default=0, minimum=0, maximum=10_000)
        if current_round <= blocked:
            action = "skip_broken" if bool(record.get("circuit_open")) else "backoff"
            return CapabilityRecoveryDecision(
                False,
                action,
                blocked,
                f"{capability} recovery {action} remains active through tool round {blocked}",
            )
        return CapabilityRecoveryDecision(True, "execute", blocked, "cooldown complete")

    def observe(
        self,
        convergence: dict[str, Any],
        capability: str,
        *,
        current_round: int,
        success: bool,
        health_failure: bool,
        health_status: str,
    ) -> CapabilityRecoveryDecision:
        records = self._records(convergence)
        if success:
            records.pop(capability, None)
            return CapabilityRecoveryDecision(True, "reset", 0, "successful execution reset recovery state")
        if not self.policy.capability_backoff_enabled or not health_failure:
            return CapabilityRecoveryDecision(True, "diagnose", 0, "failure is not a capability-health failure")
        previous = records.get(capability) if isinstance(records.get(capability), dict) else {}
        failures = _bounded_int(previous.get("failures"), default=0, minimum=0, maximum=1_000) + 1
        circuit_open = str(health_status) == "Broken"
        delay = (
            self.policy.circuit_recovery_rounds
            if circuit_open
            else min(self.policy.max_capability_backoff_rounds, 2 ** min(10, failures - 1))
        )
        blocked = min(10_000, current_round + delay)
        action = "skip_broken" if circuit_open else "backoff"
        if capability not in records and len(records) >= self.MAX_RECORDS:
            oldest = next(iter(records), None)
            if oldest is not None:
                records.pop(oldest, None)
        records[capability] = {
            "failures": failures,
            "circuit_open": circuit_open,
            "blocked_through_round": blocked,
            "action": action,
        }
        return CapabilityRecoveryDecision(
            False,
            action,
            blocked,
            f"{capability} entered {action} through tool round {blocked}; choose another healthy capability",
        )

    @staticmethod
    def _records(convergence: dict[str, Any]) -> dict[str, dict[str, Any]]:
        value = convergence.get("capability_recovery")
        if not isinstance(value, dict):
            value = {}
            convergence["capability_recovery"] = value
        return value


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


__all__ = [
    "CapabilityRecoveryController",
    "CapabilityRecoveryDecision",
    "ErrorCategory",
    "ResiliencePolicy",
]
