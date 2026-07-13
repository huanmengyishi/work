from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

from .config import AppConfig


TASK_MODES = {"simple", "standard", "large", "deep"}
TASK_TYPES = {
    "question",
    "code_explanation",
    "bug_fix",
    "feature_development",
    "review",
    "architecture",
    "refactor",
}
TASK_SCALES = {"small", "medium", "large"}
TASK_RISKS = {"low", "medium", "high"}
TASK_MODE_RANK = {"simple": 0, "standard": 1, "large": 2, "deep": 3}
TASK_RISK_RANK = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True)
class TaskRoute:
    """A deterministic, serializable classification and execution policy."""

    task_type: str
    scale: str
    risk: str
    mode: str
    score: int
    reasons: tuple[str, ...]
    max_tool_rounds: int
    require_plan: bool
    chunked_context: bool
    failure_count: int = 0
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_type": self.task_type,
            "scale": self.scale,
            "risk": self.risk,
            "mode": self.mode,
            "score": self.score,
            "reasons": list(self.reasons),
            "max_tool_rounds": self.max_tool_rounds,
            "require_plan": self.require_plan,
            "chunked_context": self.chunked_context,
            "failure_count": self.failure_count,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TaskRoute":
        """Load v0.9 routes and safely promote a legacy task_strategy mapping."""

        mode = _enum_value(value.get("mode"), TASK_MODES, "standard")
        default_scale = "large" if mode in {"large", "deep"} else "small" if mode == "simple" else "medium"
        default_rounds = {"simple": 4, "standard": 8, "large": 16, "deep": 24}[mode]
        return cls(
            task_type=_enum_value(value.get("task_type"), TASK_TYPES, "question"),
            scale=_enum_value(value.get("scale"), TASK_SCALES, default_scale),
            risk=_enum_value(value.get("risk"), TASK_RISKS, "low"),
            mode=mode,
            score=_bounded_int(value.get("score"), default=0, minimum=0, maximum=10_000),
            reasons=_reasons_from_value(value.get("reasons")),
            max_tool_rounds=_bounded_int(
                value.get("max_tool_rounds"),
                default=default_rounds,
                minimum=1,
                maximum=10_000,
            ),
            require_plan=bool(value.get("require_plan", mode in {"large", "deep"})),
            chunked_context=bool(value.get("chunked_context", mode in {"large", "deep"})),
            failure_count=_bounded_int(value.get("failure_count"), default=0, minimum=0, maximum=10_000),
            schema_version=_bounded_int(value.get("schema_version"), default=1, minimum=1, maximum=10_000),
        )


class TaskRouter:
    """Classify a request locally without consuming a model request."""

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
        r"\b(fix|implement|change|edit|test|debug|review|audit|analy[sz]e|inspect|build|write|"
        r"refactor|migrate)\b|修复|实现|修改|编辑|测试|调试|审查|审计|分析|检查|构建|编写|重构|迁移",
        re.IGNORECASE,
    )
    _MUTATION_MARKERS = re.compile(
        r"\b(fix|implement|change|edit|debug|build|write|refactor|migrat(?:e|ion)|delete|remove|deploy)\b|"
        r"修复|实现|修改|编辑|调试|构建|编写|重构|迁移|删除|移除|部署",
        re.IGNORECASE,
    )
    _SIMPLE_MARKERS = re.compile(
        r"^(what is|who is|when is|where is|define|translate)\b|"
        r"^(什么是|谁是|何时是|哪里是|定义|翻译)",
        re.IGNORECASE,
    )
    _EXPLANATION_MARKERS = re.compile(
        r"\b(explain|describe|summarize|read this|how does)\b|解释|说明|概述|总结|怎么看|如何工作",
        re.IGNORECASE,
    )
    _ARCHITECTURE_MARKERS = re.compile(r"\b(?:architecture|system design)\b|架构|系统设计", re.IGNORECASE)
    _REFACTOR_MARKERS = re.compile(r"\b(?:refactor|migration)\b|重构|迁移", re.IGNORECASE)
    _BUG_MARKERS = re.compile(
        r"\b(bug|fix|debug|error|failure|root cause)\b|缺陷|修复|调试|错误|失败|根因",
        re.IGNORECASE,
    )
    _FEATURE_MARKERS = re.compile(
        r"\b(implement|add|feature|build|create)\b|实现|新增|添加|功能|构建|创建",
        re.IGNORECASE,
    )
    _REVIEW_MARKERS = re.compile(
        r"\b(review|audit|inspect|analy[sz]e)\b|审查|审计|检查|分析",
        re.IGNORECASE,
    )
    _HIGH_RISK_MARKERS = re.compile(
        r"\b(security|credential|secret|token|password|permission|authorization|authentication|"
        r"production|deploy|payment|database schema|destructive|delete|wipe|migration)\b|"
        r"安全|凭据|密钥|令牌|密码|权限|授权|认证|生产|部署|支付|数据库架构|破坏性|删除|清空|迁移",
        re.IGNORECASE,
    )

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def route(
        self,
        prompt: str,
        *,
        source_file_count: int = 0,
        file_count: int = 0,
        explicit_mode: str | None = None,
        failure_count: int = 0,
    ) -> TaskRoute:
        text = str(prompt or "").strip()
        failures = _bounded_int(failure_count, default=0, minimum=0, maximum=10_000)
        source_files = _bounded_int(source_file_count, default=0, minimum=0, maximum=10_000_000)
        files = _bounded_int(file_count, default=0, minimum=0, maximum=10_000_000)
        action = bool(self._ACTION_MARKERS.search(text))
        mutation = bool(self._MUTATION_MARKERS.search(text))
        large_signal = bool(self._LARGE_MARKERS.search(text))
        deep_signal = bool(self._DEEP_MARKERS.search(text))

        score = 0
        reasons: list[str] = []
        if len(text) >= 2_000:
            score += 2
            reasons.append("long-request")
        elif len(text) >= 600:
            score += 1
            reasons.append("detailed-request")
        if action:
            score += 1
            reasons.append("project-action")
        if mutation:
            reasons.append("mutation-request")
        if large_signal:
            score += 2
            reasons.append("large-scope")
        if deep_signal:
            # A short factual question about a deep topic is not itself a deep
            # engineering task. Mutating and repository-wide requests are.
            score += 3 if action or mutation or large_signal else 1
            reasons.append("deep-reasoning" if action or mutation or large_signal else "deep-topic")

        large_source_threshold = _bounded_int(
            self.config.get("runtime.large_project_source_files", 500),
            default=500,
            minimum=1,
            maximum=10_000_000,
        )
        large_file_threshold = _bounded_int(
            self.config.get("runtime.large_project_files", 2_000),
            default=2_000,
            minimum=1,
            maximum=10_000_000,
        )
        if source_files >= large_source_threshold:
            score += 2
            reasons.append("large-codebase")
        elif files >= large_file_threshold:
            score += 1
            reasons.append("many-files")

        task_type = self._task_type(text)
        scale = self._scale(
            text,
            action=action or mutation,
            large_signal=large_signal,
            source_files=source_files,
            files=files,
            source_threshold=large_source_threshold,
            file_threshold=large_file_threshold,
        )
        risk = self._risk(text, task_type=task_type, mutation=mutation, action=action)
        if risk == "high":
            score += 2
            reasons.append("high-risk")
        if failures >= 2:
            score += 2
            reasons.append("repeated-failure")
        elif failures == 1:
            score += 1
            reasons.append("prior-failure")

        configured_mode = str(explicit_mode or self.config.get("runtime.task_mode", "auto")).strip().lower()
        if configured_mode not in TASK_MODES | {"auto"}:
            configured_mode = "auto"
            reasons.append("invalid-mode-auto")
        if configured_mode in TASK_MODES:
            mode = configured_mode
            reasons.append("configured-mode")
        elif score >= 5 or (risk == "high" and mutation):
            mode = "deep"
        elif scale == "large" or score >= 3:
            mode = "large"
        elif score >= 1:
            mode = "standard"
        elif self._SIMPLE_MARKERS.search(text) or task_type == "code_explanation":
            mode = "simple"
        else:
            mode = "standard"

        require_plan = mode in {"large", "deep"}
        chunked_context = mode in {"large", "deep"}
        return TaskRoute(
            task_type=task_type,
            scale=scale,
            risk=risk,
            mode=mode,
            score=score,
            reasons=tuple(dict.fromkeys(reasons)) or ("default",),
            max_tool_rounds=self._round_limit(mode),
            require_plan=require_plan,
            chunked_context=chunked_context,
            failure_count=failures,
        )

    def select(self, prompt: str, **kwargs: Any) -> TaskRoute:
        """Compatibility spelling for callers moving from TaskStrategySelector."""

        return self.route(prompt, **kwargs)

    def _round_limit(self, mode: str) -> int:
        defaults = {"simple": 4, "standard": 8, "large": 16, "deep": 24}
        configured = _bounded_int(
            self.config.get("runtime.max_tool_rounds", 8),
            default=8,
            minimum=1,
            maximum=10_000,
        )
        if mode == "simple":
            rounds = min(defaults[mode], configured)
        elif mode == "standard":
            rounds = configured
        else:
            rounds = max(defaults[mode], configured)
        hard_limit = _bounded_int(
            self.config.get("runtime.max_tool_rounds_hard_limit", 32),
            default=32,
            minimum=1,
            maximum=10_000,
        )
        return max(1, min(rounds, hard_limit))

    @classmethod
    def _task_type(cls, text: str) -> str:
        if cls._ARCHITECTURE_MARKERS.search(text):
            return "architecture"
        if cls._REFACTOR_MARKERS.search(text):
            return "refactor"
        if cls._BUG_MARKERS.search(text):
            return "bug_fix"
        if cls._FEATURE_MARKERS.search(text):
            return "feature_development"
        if cls._REVIEW_MARKERS.search(text):
            return "review"
        if cls._EXPLANATION_MARKERS.search(text):
            return "code_explanation"
        return "question"

    @classmethod
    def _scale(
        cls,
        text: str,
        *,
        action: bool,
        large_signal: bool,
        source_files: int,
        files: int,
        source_threshold: int,
        file_threshold: int,
    ) -> str:
        if len(text) >= 2_000 or large_signal or source_files >= source_threshold or files >= file_threshold:
            return "large"
        if len(text) >= 600 or action:
            return "medium"
        return "small"

    @classmethod
    def _risk(cls, text: str, *, task_type: str, mutation: bool, action: bool) -> str:
        if cls._HIGH_RISK_MARKERS.search(text) and (mutation or action):
            return "high"
        if mutation or task_type in {"bug_fix", "feature_development", "architecture", "refactor"}:
            return "medium"
        return "low"


def more_capable_task_route(previous: TaskRoute, selected: TaskRoute) -> TaskRoute:
    """Upgrade a resume route without letting a generic continuation replace it."""

    previous_mode = TASK_MODE_RANK.get(previous.mode, 1)
    selected_mode = TASK_MODE_RANK.get(selected.mode, 1)
    if selected_mode > previous_mode:
        return selected
    if selected_mode == previous_mode and TASK_RISK_RANK.get(selected.risk, 0) > TASK_RISK_RANK.get(previous.risk, 0):
        return selected
    if selected_mode == previous_mode and selected.task_type != "question" and selected.task_type != previous.task_type:
        return selected
    if selected_mode == previous_mode and selected.score > previous.score and selected.task_type != "question":
        return selected
    if selected.failure_count > previous.failure_count:
        return replace(
            previous,
            score=max(previous.score, selected.score),
            reasons=tuple(dict.fromkeys((*previous.reasons, *selected.reasons))),
            failure_count=selected.failure_count,
        )
    return previous


# A more explicit alias for new runtime code.
monotonic_task_route = more_capable_task_route


def _enum_value(value: Any, allowed: set[str], default: str) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in allowed else default


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _reasons_from_value(value: Any) -> tuple[str, ...]:
    items = value if isinstance(value, (list, tuple)) else []
    return tuple(str(item) for item in items if str(item).strip()) or ("restored",)
