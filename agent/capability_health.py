from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from .config import AppConfig
from .timeutil import utc_now_iso
from .tools.registry import ToolCapability


HEALTH_STATES = {"Available", "Unavailable", "Need Config", "Disabled", "Broken"}


@dataclass(frozen=True)
class CapabilityHealth:
    name: str
    status: str
    reason: str
    consecutive_failures: int = 0
    last_failure: str = ""
    checked_at: str = ""


class CapabilityHealthManager:
    def __init__(self, config: AppConfig, project_id: str) -> None:
        self.config = config
        self.project_id = project_id
        self.path = config.data_dir / "capability-health" / f"{project_id}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.failure_threshold = max(1, int(config.get("runtime.capability_failure_threshold", 3)))
        self.records = self._read()

    def evaluate(self, capability: ToolCapability) -> CapabilityHealth:
        stored = self.records.get(capability.name, {})
        failures = max(0, int(stored.get("consecutive_failures") or 0))
        last_failure = str(stored.get("last_failure") or "")
        if not capability.enabled:
            status, reason = "Disabled", "disabled by configuration"
        elif not capability.available:
            status, reason = "Unavailable", capability.unavailable_reason or "dependency unavailable"
        elif capability.name == "http.request" and not self.config.get("tools.http.allowed_domains", []):
            status, reason = "Need Config", "configure tools.http.allowed_domains"
        elif capability.name.startswith("mcp.") and not bool(self.config.get("mcp.enabled", False)):
            status, reason = "Need Config", "enable and configure MCP"
        elif failures >= self.failure_threshold:
            status, reason = "Broken", last_failure or f"failed {failures} consecutive times"
        else:
            status, reason = "Available", "ready"
        return CapabilityHealth(
            capability.name,
            status,
            reason,
            failures,
            last_failure,
            str(stored.get("checked_at") or utc_now_iso()),
        )

    def record(self, capability_name: str, *, success: bool, error: str = "") -> None:
        current = self.records.get(capability_name, {})
        failures = 0 if success else max(0, int(current.get("consecutive_failures") or 0)) + 1
        self.records[capability_name] = {
            "consecutive_failures": failures,
            "last_failure": "" if success else error[:1000],
            "checked_at": utc_now_iso(),
        }
        self._write()

    def reset(self, capability_name: str | None = None) -> None:
        if capability_name:
            self.records.pop(capability_name, None)
        else:
            self.records.clear()
        self._write()

    def report(self, capabilities: list[ToolCapability]) -> list[CapabilityHealth]:
        return [self.evaluate(capability) for capability in capabilities]

    def _read(self) -> dict[str, dict[str, Any]]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        records = value.get("records") if isinstance(value, dict) else None
        return records if isinstance(records, dict) else {}

    def _write(self) -> None:
        payload = {
            "schema_version": 1,
            "project_id": self.project_id,
            "updated_at": utc_now_iso(),
            "records": self.records,
        }
        temp = self.path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(self.path)


def health_to_dict(item: CapabilityHealth) -> dict[str, Any]:
    return asdict(item)
