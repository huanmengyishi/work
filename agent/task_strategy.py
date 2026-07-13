from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .config import AppConfig
from .model_router import ModelRoute, ModelRouter
from .task_router import TASK_MODES, TaskRoute, TaskRouter


__all__ = ["TASK_MODES", "TaskStrategy", "TaskStrategySelector"]


@dataclass(frozen=True)
class TaskStrategy:
    """Legacy combined policy kept for runtime and third-party compatibility."""

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
    """Compatibility facade over the v0.9 TaskRouter and ModelRouter."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.task_router = TaskRouter(config)
        self.model_router = ModelRouter(config)

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
        if not strategy.require_plan:
            return []
        change_task = bool(
            re.search(
                r"\b(fix|implement|change|edit|debug|build|write|refactor|migrate)\b|"
                r"修复|实现|修改|编辑|调试|构建|编写|重构|迁移",
                prompt,
                re.IGNORECASE,
            )
        )
        middle_title = "Implement bounded changes" if change_task else "Synthesize the inspected evidence"
        middle_done = (
            "Requested changes are applied through the managed file workflow."
            if change_task
            else "Findings are reconciled across all inspected chunks without unsupported claims."
        )
        return [
            {
                "id": "scope",
                "title": "Map the request, constraints, and relevant project areas",
                "status": "in_progress",
                "max_retries": 1,
                "completion_criteria": "Scope, constraints, and bounded inspection targets are explicit.",
            },
            {
                "id": "inspect-chunks",
                "title": "Inspect relevant text or code in bounded chunks",
                "dependencies": ["scope"],
                "max_retries": 2,
                "allow_parallel": strategy.mode == "deep",
                "completion_criteria": "Each relevant chunk has evidence and unresolved questions recorded.",
            },
            {
                "id": "implement" if change_task else "synthesize",
                "title": middle_title,
                "dependencies": ["inspect-chunks"],
                "max_retries": 2,
                "completion_criteria": middle_done,
            },
            {
                "id": "verify",
                "title": "Verify the result and reconcile it with the original request",
                "dependencies": ["implement" if change_task else "synthesize"],
                "max_retries": 1,
                "completion_criteria": "Checks pass and the final answer states evidence, limits, and remaining risk.",
            },
        ]

    @staticmethod
    def _thinking_enabled(value: Any) -> bool:
        """Retained for callers that used the v0.8 helper directly."""

        if isinstance(value, dict):
            return str(value.get("type") or "").lower() != "disabled"
        if isinstance(value, bool):
            return value
        return str(value or "").lower() in {"enabled", "true", "on", "1"}
