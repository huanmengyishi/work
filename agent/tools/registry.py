from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable
from uuid import uuid4

from ..config import AppConfig
from .base import ToolRequest, ToolResult


ToolHandler = Callable[..., ToolResult]


@dataclass(frozen=True)
class ToolCapability:
    tool: str
    action: str
    model_name: str
    description: str
    properties: dict[str, Any]
    required: tuple[str, ...] = ()
    permissions: tuple[str, ...] = ("read",)
    timeout_seconds: int = 120
    supports_stream: bool = False
    enabled: bool = True
    input_formats: tuple[str, ...] = ()
    output_formats: tuple[str, ...] = ()
    available: bool = True
    unavailable_reason: str = ""
    requires_confirmation: bool = False

    @property
    def name(self) -> str:
        return f"{self.tool}.{self.action}"

    @property
    def active(self) -> bool:
        return self.enabled and self.available

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.model_name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.properties,
                    "required": list(self.required),
                    "additionalProperties": False,
                },
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_name": self.model_name,
            "permissions": list(self.permissions),
            "timeout_seconds": self.timeout_seconds,
            "supports_stream": self.supports_stream,
            "enabled": self.enabled,
            "available": self.available,
            "active": self.active,
            "unavailable_reason": self.unavailable_reason,
            "requires_confirmation": self.requires_confirmation,
            "input": self.properties,
            "input_formats": list(self.input_formats),
            "output_formats": list(self.output_formats),
        }


class ToolCapabilityRegistry:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._capabilities: dict[str, ToolCapability] = {}
        self._model_names: dict[str, str] = {}
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, capability: ToolCapability, handler: ToolHandler) -> None:
        configured = self._configured(capability)
        self._capabilities[configured.name] = configured
        self._model_names[configured.model_name] = configured.name
        self._model_names[configured.name] = configured.name
        self._handlers[configured.name] = handler

    def resolve(self, name: str) -> tuple[ToolCapability | None, ToolHandler | None]:
        canonical = self._model_names.get(name, name)
        return self._capabilities.get(canonical), self._handlers.get(canonical)

    def capabilities(self, *, enabled_only: bool = False) -> list[ToolCapability]:
        values = list(self._capabilities.values())
        if enabled_only:
            values = [item for item in values if item.active]
        return values

    def schemas(self) -> list[dict[str, Any]]:
        return [item.schema() for item in self.capabilities(enabled_only=True)]

    def request(self, name: str, args: dict[str, Any], *, request_id: str | None = None) -> ToolRequest:
        capability, _ = self.resolve(name)
        if capability:
            return ToolRequest(
                tool=capability.tool,
                action=capability.action,
                args=args,
                request_id=request_id or str(uuid4()),
                model_name=name,
            )
        if "." in name:
            tool, action = name.split(".", 1)
        elif "_" in name:
            tool, action = name.split("_", 1)
        else:
            tool, action = name or "unknown", "unknown"
        return ToolRequest(
            tool=tool,
            action=action,
            args=args,
            request_id=request_id or str(uuid4()),
            model_name=name,
        )

    def _configured(self, capability: ToolCapability) -> ToolCapability:
        base_enabled = bool(self.config.get(f"tools.{capability.tool}.enabled", True))
        legacy_enabled = bool(self.config.get(f"tools.allow_{capability.tool}", True))
        override = self.config.get(f"tools.capabilities.{capability.name}", {})
        if not isinstance(override, dict):
            override = {}
        permissions = override.get("permissions", capability.permissions)
        if not isinstance(permissions, (list, tuple)):
            permissions = capability.permissions
        if capability.tool == "http":
            legacy_enabled = True
            override = {key: value for key, value in override.items() if key != "enabled"}
        return replace(
            capability,
            enabled=base_enabled and legacy_enabled and bool(override.get("enabled", capability.enabled)),
            timeout_seconds=int(override.get("timeout_seconds", capability.timeout_seconds)),
            supports_stream=bool(override.get("supports_stream", capability.supports_stream)),
            permissions=tuple(str(item) for item in permissions),
            input_formats=self._string_tuple(override.get("input", capability.input_formats)),
            output_formats=self._string_tuple(override.get("output", capability.output_formats)),
            requires_confirmation=bool(override.get("requires_confirmation", capability.requires_confirmation)),
        )

    @staticmethod
    def _string_tuple(value: Any) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            return ()
        return tuple(str(item) for item in value)
