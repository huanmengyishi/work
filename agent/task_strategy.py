from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .config import AppConfig


TASK_MODES = {"simple", "standard", "large", "deep"}


@dataclass(frozen=True)
class TaskStrategy:
    """Bounded execution policy selected before the first model request."""

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
    """Classify requests locally so easy work stays fast and hard work is staged."""

    _DEEP_MARKERS = re.compile(
        r"\b(architecture|migration|refactor|security|audit|root cause|end[- ]to[- ]end|"
        r"comprehensive|all issues|deep analysis|large[- ]scale)\b|"
        r"架构|迁移|重构|安全|审计|根因|端到端|全面|所有问题|深度|大规模",
        re.IGNORECASE,
    )
    _LARGE_MARKERS = re.compile(
        r"\b(repository|codebase|workspace|all files|entire project|long document|dataset)\b|"
        r"仓库|代码库|工作区|所有文件|整个项目|长文档|数据集|批量",
        re.IGNORECASE,
    )
    _ACTION_MARKERS = re.compile(
        r"\b(fix|implement|change|edit|test|debug|review|analy[sz]e|inspect|build|write)\b|"
        r"修复|实现|修改|编辑|测试|调试|审查|分析|检查|构建|编写",
        re.IGNORECASE,
    )
    _SIMPLE_MARKERS = re.compile(
        r"^(what is|who is|when is|where is|define|translate)\b|"
        r"^(什么是|谁是|何时是|哪里是|定义|翻译)",
        re.IGNORECASE,
    )

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def select(
        self,
        prompt: str,
        *,
        source_file_count: int = 0,
        file_count: int = 0,
        explicit_mode: str | None = None,
    ) -> TaskStrategy:
        configured_mode = str(explicit_mode or self.config.get("runtime.task_mode", "auto")).lower()
        if configured_mode != "auto" and configured_mode not in TASK_MODES:
            configured_mode = "auto"

        text = prompt.strip()
        score = 0
        reasons: list[str] = []
        if len(text) >= 2_000:
            score += 2
            reasons.append("long-request")
        elif len(text) >= 600:
            score += 1
            reasons.append("detailed-request")
        if self._ACTION_MARKERS.search(text):
            score += 1
            reasons.append("project-action")
        if self._LARGE_MARKERS.search(text):
            score += 2
            reasons.append("large-scope")
        if self._DEEP_MARKERS.search(text):
            score += 3
            reasons.append("deep-reasoning")
        if source_file_count >= int(self.config.get("runtime.large_project_source_files", 500)):
            score += 2
            reasons.append("large-codebase")
        elif file_count >= int(self.config.get("runtime.large_project_files", 2_000)):
            score += 1
            reasons.append("many-files")

        if configured_mode in TASK_MODES:
            mode = configured_mode
            reasons.append("configured-mode")
        elif score >= 5:
            mode = "deep"
        elif score >= 3:
            mode = "large"
        elif score >= 1:
            mode = "standard"
        elif self._SIMPLE_MARKERS.search(text):
            mode = "simple"
        else:
            mode = "standard"

        defaults = {
            "simple": (False, None, 4, False, False),
            "standard": (True, "high", 8, False, False),
            "large": (True, "high", 16, True, True),
            "deep": (True, "max", 24, True, True),
        }
        thinking, effort, rounds, require_plan, chunked = defaults[mode]
        configured_rounds = max(1, int(self.config.get("runtime.max_tool_rounds", 8)))
        if mode == "simple":
            rounds = min(rounds, configured_rounds)
        elif mode == "standard":
            rounds = configured_rounds
        else:
            rounds = max(rounds, configured_rounds)
        if not bool(self.config.get("runtime.adaptive_thinking", True)):
            configured_thinking = self.config.get("model.thinking")
            thinking = self._thinking_enabled(configured_thinking)
            effort = str(self.config.get("model.reasoning_effort") or "") or None
        rounds = min(rounds, int(self.config.get("runtime.max_tool_rounds_hard_limit", 32)))
        return TaskStrategy(
            mode=mode,
            score=score,
            reasons=tuple(dict.fromkeys(reasons)) or ("default",),
            thinking_enabled=thinking,
            reasoning_effort=effort,
            max_tool_rounds=max(1, rounds),
            require_plan=require_plan,
            chunked_context=chunked,
        )

    def initial_plan(self, prompt: str, strategy: TaskStrategy) -> list[dict[str, Any]]:
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
        if isinstance(value, dict):
            return str(value.get("type") or "").lower() != "disabled"
        if isinstance(value, bool):
            return value
        return str(value or "").lower() in {"enabled", "true", "on", "1"}
