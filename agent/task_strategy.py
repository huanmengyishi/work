from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

from .config import AppConfig
from .model_router import ModelRoute, ModelRouter
from .task_plan import TaskPlanFactory
from .task_router import TASK_MODES, TaskRoute, TaskRouter


__all__ = ["TASK_MODES", "TaskStrategy", "TaskStrategySelector"]


@dataclass(frozen=True)
class TaskStrategy:
    """Deprecated v0.8 DTO kept for saved-state and third-party compatibility."""

    mode: str
    score: int
    reasons: tuple[str, ...]
    thinking_enabled: bool
    reasoning_effort: str | None
    max_tool_rounds: int
    require_plan: bool
    chunked_context: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "score": self.score,
            "reasons": list(self.reasons),
            "thinking_enabled": self.thinking_enabled,
            "reasoning_effort": self.reasoning_effort,
            "max_tool_rounds": self.max_tool_rounds,
            "require_plan": self.require_plan,
            "chunked_context": self.chunked_context,
        }


class TaskStrategySelector:
    """Deprecated compatibility facade; TaskRouter is the only classifier.

    New code must compose ``TaskRouter``, ``ModelRouter``, and
    ``TaskPlanFactory`` directly.  This class contains no scoring or
    classification rules and will be removed after the v1 compatibility
    window.
    """

    def __init__(self, config: AppConfig) -> None:
        warnings.warn(
            "TaskStrategySelector is deprecated; use TaskRouter, ModelRouter, and TaskPlanFactory directly",
            DeprecationWarning,
            stacklevel=2,
        )
        self.config = config
        self.task_router = TaskRouter(config)
        self.model_router = ModelRouter(config)
        self.plan_factory = TaskPlanFactory()

    def select(
        self,
        prompt: str,
        *,
        source_file_count: int = 0,
        file_count: int = 0,
        explicit_mode: str | None = None,
        failure_count: int = 0,
    ) -> TaskStrategy:
        task = self.task_router.route(
            prompt,
            source_file_count=source_file_count,
            file_count=file_count,
            explicit_mode=explicit_mode,
            failure_count=failure_count,
        )
        model = self.model_router.route(task)
        return TaskStrategy(
            mode=task.mode,
            score=task.score,
            reasons=task.reasons,
            thinking_enabled=model.thinking_enabled,
            reasoning_effort=model.reasoning_effort,
            max_tool_rounds=task.max_tool_rounds,
            require_plan=task.require_plan,
            chunked_context=task.chunked_context,
        )

    def route(
        self,
        prompt: str,
        *,
        source_file_count: int = 0,
        file_count: int = 0,
        explicit_mode: str | None = None,
        failure_count: int = 0,
    ) -> tuple[TaskRoute, ModelRoute]:
        """Return the separate v0.9 routes while the old `select` API remains stable."""

        task = self.task_router.route(
            prompt,
            source_file_count=source_file_count,
            file_count=file_count,
            explicit_mode=explicit_mode,
            failure_count=failure_count,
        )
        return task, self.model_router.route(task)

    def initial_plan(self, prompt: str, strategy: TaskStrategy | TaskRoute) -> list[dict[str, Any]]:
        if isinstance(strategy, TaskRoute):
            route = strategy
        else:
            warnings.warn(
                "TaskStrategy planning compatibility re-routes the prompt; pass TaskRoute to avoid duplicate work",
                DeprecationWarning,
                stacklevel=2,
            )
            route = self.task_router.route(prompt, explicit_mode=strategy.mode)
        return self.plan_factory.build(route)

    @staticmethod
    def _thinking_enabled(value: Any) -> bool:
        """Retained for callers that used the v0.8 helper directly."""

        if isinstance(value, dict):
            return str(value.get("type") or "").lower() != "disabled"
        if isinstance(value, bool):
            return value
        return str(value or "").lower() in {"enabled", "true", "on", "1"}
