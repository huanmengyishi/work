from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .state import AgentState


@dataclass(frozen=True)
class ProgressSnapshot:
    """Conservative task progress derived only from persisted state.

    ``percent`` is in the inclusive ``0..100`` range.  An active task never
    reports 100 merely because every plan step is marked satisfied; final
    completion remains owned by ``AgentState.status``.
    """

    percent: float
    completed_steps: int
    skipped_steps: int
    total_steps: int
    current_step: str | None
    model_requests_used: int
    tokens_used: int
    model_requests_remaining: int | None = None
    tokens_remaining: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProgressTracker:
    """Build read-only progress snapshots without inventing partial work."""

    @staticmethod
    def snapshot(
        state: AgentState,
        *,
        model_request_budget: int | None = None,
        token_budget: int | None = None,
    ) -> ProgressSnapshot:
        return derive_progress(
            state,
            model_request_budget=model_request_budget,
            token_budget=token_budget,
        )


def derive_progress(
    state: AgentState,
    *,
    model_request_budget: int | None = None,
    token_budget: int | None = None,
) -> ProgressSnapshot:
    """Derive trustworthy progress and optional remaining budgets.

    Only completed steps and skips accepted by ``AgentState`` contribute to
    the percentage.  In-progress work receives no guessed fractional credit.
    """

    if not isinstance(state, AgentState):
        raise TypeError("progress requires an AgentState")
    model_request_budget = _optional_budget(model_request_budget, "model_request_budget")
    token_budget = _optional_budget(token_budget, "token_budget")

    total_weight = 0.0
    completed_weight = 0.0
    completed_steps = 0
    skipped_steps = 0
    for step in state.plan:
        weight = _step_weight(step)
        total_weight += weight
        if not state.plan_step_satisfied(step):
            continue
        completed_weight += weight
        if step.status == "completed":
            completed_steps += 1
        elif step.status == "skipped":
            skipped_steps += 1

    if state.status == "completed":
        percent = 100.0
    elif not state.plan:
        percent = 0.0
    else:
        raw_percent = 100.0 * completed_weight / total_weight if total_weight else 0.0
        percent = min(99.0, round(raw_percent, 1))

    metrics = state.model_metrics if isinstance(state.model_metrics, dict) else {}
    prompt_tokens = _non_negative_int(metrics.get("prompt_tokens"))
    completion_tokens = _non_negative_int(metrics.get("completion_tokens"))
    reported_total = _non_negative_int(metrics.get("total_tokens"))
    tokens_used = max(reported_total, prompt_tokens + completion_tokens)
    model_requests_used = _non_negative_int(state.model_request_count)

    return ProgressSnapshot(
        percent=percent,
        completed_steps=completed_steps,
        skipped_steps=skipped_steps,
        total_steps=len(state.plan),
        current_step=_current_step(state),
        model_requests_used=model_requests_used,
        tokens_used=tokens_used,
        model_requests_remaining=_remaining(model_request_budget, model_requests_used),
        tokens_remaining=_remaining(token_budget, tokens_used),
    )


def _current_step(state: AgentState) -> str | None:
    if state.status == "completed":
        return None
    by_id = {step.id: step for step in state.plan}
    selected = by_id.get(state.current_step or "")
    if selected is not None and not state.plan_step_satisfied(selected):
        return selected.id
    active = next(
        (step for step in state.plan if step.status == "in_progress" and not state.plan_step_satisfied(step)),
        None,
    )
    if active is not None:
        return active.id
    satisfied = {step.id for step in state.plan if state.plan_step_satisfied(step)}
    ready = next(
        (
            step
            for step in state.plan
            if step.status == "pending" and all(dependency in satisfied for dependency in step.dependencies)
        ),
        None,
    )
    return ready.id if ready is not None else None


def _step_weight(step: Any) -> float:
    value = getattr(step, "progress_weight", 1.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 1.0
    converted = float(value)
    return converted if math.isfinite(converted) and 0.0 < converted <= 10_000.0 else 1.0


def _optional_budget(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer or None")
    return value


def _non_negative_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _remaining(limit: int | None, used: int) -> int | None:
    return None if limit is None else max(0, limit - used)
