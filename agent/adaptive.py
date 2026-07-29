from __future__ import annotations

import math
from dataclasses import dataclass

from .config import AppConfig
from .model_router import ModelRoute
from .optimizer import PerformanceHistory
from .task_router import TaskRoute


@dataclass(frozen=True)
class PerformanceProfile:
    task_type: str
    samples: int
    exploration_samples: int
    success_rate: float
    p95_exploration_rounds: int


@dataclass(frozen=True)
class StrategyAdjustment:
    enabled: bool
    applied: bool
    samples: int
    reason: str
    exploration_round_limit: int
    model_tier: str


class StrategyAdjuster:
    """Read bounded scalar history and recommend conservative local policy."""

    def __init__(self, config: AppConfig, history: PerformanceHistory, *, project_id: str) -> None:
        self.history = history
        self.project_id = project_id
        self.strategy_adjustment_enabled = bool(config.get("optimizer.strategy_adjustment_enabled", False))
        self.adaptive_convergence_enabled = bool(config.get("optimizer.adaptive_convergence_enabled", False))
        self.enabled = self.strategy_adjustment_enabled or self.adaptive_convergence_enabled
        self.min_samples = _bounded_int(config.get("optimizer.min_samples", 8), 8, 3, 100)
        self.history_limit = _bounded_int(config.get("optimizer.history_limit", 100), 100, 3, 200)
        self.failure_upgrade_threshold = _bounded_float(
            config.get("optimizer.failure_upgrade_threshold", 0.4),
            0.4,
            0.05,
            0.9,
        )
        self.success_downgrade_threshold = _bounded_float(
            config.get("optimizer.success_downgrade_threshold", 0.9),
            0.9,
            0.5,
            1.0,
        )

    def profile(self, task_type: str) -> PerformanceProfile:
        records = [
            item
            for item in self.history.recent(limit=self.history_limit)
            if item.project_id == self.project_id and item.task_type == task_type
        ]
        success = sum(item.outcome == "completed" for item in records)
        exploration = sorted(
            item.exploration_rounds
            for item in records
            if _has_observed_exploration_rounds(item.schema_version, item.exploration_rounds)
        )
        index = max(0, math.ceil(0.95 * len(exploration)) - 1) if exploration else 0
        p95 = exploration[index] if exploration else 0
        return PerformanceProfile(
            task_type=task_type,
            samples=len(records),
            exploration_samples=len(exploration),
            success_rate=(success / len(records)) if records else 0.0,
            p95_exploration_rounds=p95,
        )

    def adjust(
        self,
        task: TaskRoute,
        model: ModelRoute,
        *,
        configured_exploration_limit: int,
    ) -> StrategyAdjustment:
        profile = self.profile(task.task_type)
        exploration_limit = configured_exploration_limit
        tier = model.tier
        reasons: list[str] = []
        adaptive_ready = self.adaptive_convergence_enabled and profile.exploration_samples >= self.min_samples
        strategy_ready = self.strategy_adjustment_enabled and profile.samples >= self.min_samples
        effective_samples = profile.samples if self.strategy_adjustment_enabled else profile.exploration_samples
        if not self.enabled or not (adaptive_ready or strategy_ready):
            return StrategyAdjustment(
                self.enabled,
                False,
                effective_samples,
                "disabled" if not self.enabled else "insufficient_samples",
                exploration_limit,
                tier,
            )
        if adaptive_ready:
            exploration_limit = max(2, min(32, profile.p95_exploration_rounds))
            reasons.append("p95_exploration")
        if strategy_ready:
            failure_rate = 1.0 - profile.success_rate
            if tier == "standard" and failure_rate >= self.failure_upgrade_threshold:
                tier = "deep"
                reasons.append("low_recent_success")
            elif (
                tier == "deep"
                and profile.success_rate >= self.success_downgrade_threshold
                and task.risk != "high"
                and task.mode != "deep"
                and task.task_type not in {"architecture", "refactor"}
            ):
                tier = "standard"
                reasons.append("stable_recent_success")
        return StrategyAdjustment(
            True,
            bool(reasons),
            effective_samples,
            ",".join(reasons) or "no_change",
            exploration_limit,
            tier,
        )


def _bounded_int(value: object, default: int, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _has_observed_exploration_rounds(schema_version: object, value: object) -> bool:
    """Reject legacy migration defaults while retaining genuine zero-round observations."""

    return (
        isinstance(schema_version, int)
        and not isinstance(schema_version, bool)
        and schema_version >= 3
        and isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    )


def _bounded_float(value: object, default: float, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        parsed = default
    if not math.isfinite(parsed):
        parsed = default
    return max(minimum, min(maximum, parsed))


__all__ = ["PerformanceProfile", "StrategyAdjuster", "StrategyAdjustment"]
