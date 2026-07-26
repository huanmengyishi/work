from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from .config import AppConfig
from .state import AgentState


class ExecutionBudgetExceeded(RuntimeError):
    """Raised at a protocol boundary before more external work is started."""

    def __init__(self, reason: str) -> None:
        self.reason = str(reason).strip() or "execution budget exhausted"
        super().__init__(self.reason)


@dataclass(frozen=True)
class BudgetRemaining:
    model_requests: int
    tokens: int
    seconds: float


class ExecutionBudgetController:
    """Bound one explicit Session turn without adding a second retry layer.

    Provider/network retry policy remains owned by ``DeepSeekClient``.  This
    controller only stops new model requests or tool batches at Runtime
    protocol boundaries.  Its bounded scalar snapshot is kept in AgentState's
    existing ``convergence`` mapping so checkpoints and Resume remain safe.
    """

    _MAX_REQUEST_LIMIT = 10_000
    _MAX_TOKEN_LIMIT = 100_000_000
    _MAX_SECONDS_LIMIT = 7 * 24 * 60 * 60

    def __init__(self, config: AppConfig, *, clock: Any = time.monotonic) -> None:
        self.enabled = bool(config.get("runtime.budget.enabled", True))
        self.max_model_requests = self._bounded_int(
            config.get("runtime.budget.max_model_requests_per_turn", 64),
            default=64,
            minimum=1,
            maximum=self._MAX_REQUEST_LIMIT,
        )
        self.max_total_tokens = self._bounded_int(
            config.get("runtime.budget.max_total_tokens_per_turn", 1_000_000),
            default=1_000_000,
            minimum=1_024,
            maximum=self._MAX_TOKEN_LIMIT,
        )
        self.max_elapsed_seconds = float(
            self._bounded_int(
                config.get("runtime.budget.max_elapsed_seconds_per_turn", 3_600),
                default=3_600,
                minimum=1,
                maximum=self._MAX_SECONDS_LIMIT,
            )
        )
        self._clock = clock
        self._turn = 0
        self._started: float | None = None

    def bind(self, state: AgentState) -> None:
        self._turn = state.turn
        self._started = float(self._clock())
        self._sync(state, stop_reason="", reserved_tokens=0)

    def before_model_request(
        self,
        state: AgentState,
        *,
        phase: str,
        estimated_input_tokens: int,
        requested_output_tokens: int,
    ) -> BudgetRemaining:
        self._ensure_bound(state)
        remaining = self.remaining(state)
        if not self.enabled:
            self._sync(state, stop_reason="", reserved_tokens=0)
            return remaining
        if remaining.seconds <= 0:
            self._stop(state, "elapsed-time limit reached")
        if remaining.model_requests <= 0:
            self._stop(state, "model-request limit reached")
        reservation = max(0, int(estimated_input_tokens)) + max(0, int(requested_output_tokens))
        if reservation > remaining.tokens:
            self._stop(
                state,
                f"token limit would be exceeded (remaining={remaining.tokens}, requested={reservation}, phase={phase})",
            )
        self._sync(state, stop_reason="", reserved_tokens=reservation, phase=phase)
        return remaining

    def before_tool_batch(self, state: AgentState) -> BudgetRemaining:
        self._ensure_bound(state)
        remaining = self.remaining(state)
        if self.enabled and remaining.seconds <= 0:
            self._stop(state, "elapsed-time limit reached before tool execution")
        self._sync(state, stop_reason="", reserved_tokens=0, phase="tool_batch")
        return remaining

    def remaining(self, state: AgentState) -> BudgetRemaining:
        self._ensure_bound(state)
        requests_used = max(0, int(state.model_request_count))
        tokens_used = self._tokens_used(state)
        elapsed = max(0.0, float(self._clock()) - self._started)
        return BudgetRemaining(
            model_requests=max(0, self.max_model_requests - requests_used),
            tokens=max(0, self.max_total_tokens - tokens_used),
            seconds=max(0.0, self.max_elapsed_seconds - elapsed),
        )

    def snapshot(self, state: AgentState) -> dict[str, Any]:
        self._ensure_bound(state)
        self._sync(state)
        value = state.convergence.get("execution_budget")
        return dict(value) if isinstance(value, dict) else {}

    def _ensure_bound(self, state: AgentState) -> None:
        if self._turn != state.turn or self._started is None:
            self.bind(state)

    def _stop(self, state: AgentState, reason: str) -> None:
        self._sync(state, stop_reason=reason, reserved_tokens=0)
        raise ExecutionBudgetExceeded(reason)

    def _sync(
        self,
        state: AgentState,
        *,
        stop_reason: str | None = None,
        reserved_tokens: int | None = None,
        phase: str | None = None,
    ) -> None:
        existing = state.convergence.get("execution_budget")
        previous = dict(existing) if isinstance(existing, dict) else {}
        elapsed = max(0.0, float(self._clock()) - self._started) if self._started is not None else 0.0
        remaining = BudgetRemaining(
            model_requests=max(0, self.max_model_requests - max(0, int(state.model_request_count))),
            tokens=max(0, self.max_total_tokens - self._tokens_used(state)),
            seconds=max(0.0, self.max_elapsed_seconds - elapsed),
        )
        previous.update(
            {
                "turn": state.turn,
                "enabled": self.enabled,
                "limits": {
                    "model_requests": self.max_model_requests,
                    "tokens": self.max_total_tokens,
                    "seconds": int(self.max_elapsed_seconds),
                },
                "used": {
                    "model_requests": max(0, int(state.model_request_count)),
                    "tokens": self._tokens_used(state),
                    "elapsed_seconds": round(elapsed, 3),
                },
                "remaining": {
                    **asdict(remaining),
                    "seconds": round(remaining.seconds, 3),
                },
            }
        )
        if stop_reason is not None:
            previous["stop_reason"] = str(stop_reason)[:500]
        if reserved_tokens is not None:
            previous["reserved_tokens"] = max(0, int(reserved_tokens))
        if phase is not None:
            previous["phase"] = str(phase)[:64]
        state.convergence["execution_budget"] = previous
        state.touch()

    @staticmethod
    def _tokens_used(state: AgentState) -> int:
        value = (state.model_metrics or {}).get("total_tokens", 0)
        return max(0, int(value)) if isinstance(value, int) and not isinstance(value, bool) else 0

    @staticmethod
    def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
        if isinstance(value, bool):
            return default
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return max(minimum, min(maximum, parsed))


__all__ = ["BudgetRemaining", "ExecutionBudgetController", "ExecutionBudgetExceeded"]
