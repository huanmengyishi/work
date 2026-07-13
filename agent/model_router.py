from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import AppConfig
from .task_router import TaskRoute


MODEL_TIERS = {"fast", "standard", "deep"}
MODEL_TIER_RANK = {"fast": 0, "standard": 1, "deep": 2}


@dataclass(frozen=True)
class ModelRoute:
    """One DeepSeek-only model decision persisted with the Agent state."""

    provider: str
    tier: str
    model: str
    thinking_enabled: bool
    reasoning_effort: str | None
    max_tokens: int
    reasons: tuple[str, ...]
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider": self.provider,
            "tier": self.tier,
            "model": self.model,
            "thinking_enabled": self.thinking_enabled,
            "reasoning_effort": self.reasoning_effort,
            "max_tokens": self.max_tokens,
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelRoute":
        provider = str(value.get("provider") or "deepseek").strip().lower()
        if provider != "deepseek":
            raise ValueError("Deep Agent supports only the DeepSeek model provider")
        tier = str(value.get("tier") or "standard").strip().lower()
        if tier not in MODEL_TIERS:
            tier = "standard"
        model = str(value.get("model") or "").strip()
        if not model:
            raise ValueError("saved model route does not contain a model name")
        effort = value.get("reasoning_effort")
        return cls(
            provider="deepseek",
            tier=tier,
            model=model,
            thinking_enabled=bool(value.get("thinking_enabled", False)),
            reasoning_effort=str(effort) if effort else None,
            max_tokens=_positive_int(value.get("max_tokens"), default=4096),
            reasons=_reasons_from_value(value.get("reasons")),
            schema_version=_positive_int(value.get("schema_version"), default=1),
        )


class ModelRouter:
    """Select a capability tier locally while keeping DeepSeek as the sole provider."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        provider = str(config.get("model.provider", "deepseek")).strip().lower()
        if provider != "deepseek":
            raise ValueError("Deep Agent supports only the DeepSeek model provider")

    def route(self, task: TaskRoute, *, explicit_tier: str | None = None) -> ModelRoute:
        configured_tier = (
            str(explicit_tier if explicit_tier is not None else self.config.get("model.routing.tier", "auto"))
            .strip()
            .lower()
        )
        if configured_tier not in MODEL_TIERS | {"auto"}:
            raise ValueError("model.routing.tier must be auto, fast, standard, or deep")

        reasons: list[str] = []
        if configured_tier in MODEL_TIERS:
            tier = configured_tier
            reasons.append("configured-tier")
        else:
            tier = self._automatic_tier(task)
            reasons.append(f"task-{task.mode}")
            if task.risk == "high":
                reasons.append("high-risk")
            if task.failure_count >= 2:
                reasons.append("repeated-failure")

        routing_enabled = bool(self.config.get("model.routing.enabled", True))
        model, used_override = self._model_for_tier(tier, routing_enabled=routing_enabled)
        if not routing_enabled:
            reasons.append("routing-disabled")
        reasons.append("tier-model" if used_override else "base-model-fallback")
        thinking_enabled, reasoning_effort = self._thinking_for_tier(tier)
        return ModelRoute(
            provider="deepseek",
            tier=tier,
            model=model,
            thinking_enabled=thinking_enabled,
            reasoning_effort=reasoning_effort,
            max_tokens=_positive_int(self.config.get("model.max_tokens", 4096), default=4096),
            reasons=tuple(dict.fromkeys(reasons)),
        )

    def select(self, task: TaskRoute, *, explicit_tier: str | None = None) -> ModelRoute:
        """Compatibility spelling for router callers."""

        return self.route(task, explicit_tier=explicit_tier)

    @staticmethod
    def _automatic_tier(task: TaskRoute) -> str:
        if "configured-mode" in task.reasons:
            return {"simple": "fast", "standard": "standard", "large": "standard", "deep": "deep"}.get(
                task.mode, "standard"
            )
        if (
            task.mode == "deep"
            or task.risk == "high"
            or task.failure_count >= 2
            or task.task_type in {"architecture", "refactor"}
        ):
            return "deep"
        if task.mode == "simple" and task.risk == "low":
            return "fast"
        return "standard"

    def _model_for_tier(self, tier: str, *, routing_enabled: bool) -> tuple[str, bool]:
        # `*_model` is the canonical v0.9 shape. The nested form is accepted so
        # hand-written configs can group model names without requiring migration.
        if routing_enabled:
            candidate = self.config.get(f"model.routing.{tier}_model")
            if not str(candidate or "").strip():
                candidate = self.config.get(f"model.routing.models.{tier}")
            model = str(candidate or "").strip()
            if model:
                return model, True
        base_model = str(self.config.get("model.model", "deepseek-v4-pro") or "").strip()
        return base_model or "deepseek-v4-pro", False

    def _thinking_for_tier(self, tier: str) -> tuple[bool, str | None]:
        if bool(self.config.get("runtime.adaptive_thinking", True)):
            return {
                "fast": (False, None),
                "standard": (True, "high"),
                "deep": (True, "max"),
            }[tier]
        configured = self.config.get("model.thinking")
        effort = str(self.config.get("model.reasoning_effort") or "").strip() or None
        return _thinking_enabled(configured), effort


def more_capable_model_route(previous: ModelRoute, selected: ModelRoute) -> ModelRoute:
    """Upgrade a resumed Session, but retain its exact model on equal/lower tiers."""

    if previous.provider != "deepseek" or selected.provider != "deepseek":
        raise ValueError("Deep Agent supports only the DeepSeek model provider")
    if MODEL_TIER_RANK.get(selected.tier, 1) > MODEL_TIER_RANK.get(previous.tier, 1):
        return selected
    return previous


monotonic_model_route = more_capable_model_route


def _thinking_enabled(value: Any) -> bool:
    if isinstance(value, dict):
        return str(value.get("type") or "").strip().lower() == "enabled"
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"enabled", "true", "on", "1"}


def _positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        parsed = default
    return max(1, min(parsed, 10_000_000))


def _reasons_from_value(value: Any) -> tuple[str, ...]:
    items = value if isinstance(value, (list, tuple)) else []
    return tuple(str(item) for item in items if str(item).strip()) or ("restored",)
