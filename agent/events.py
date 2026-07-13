from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from . import paths
from .timeutil import utc_now_iso


@dataclass(frozen=True)
class Event:
    name: str
    payload: dict[str, Any]
    project_id: str | None = None
    session_id: str | None = None
    timestamp: str = field(default_factory=utc_now_iso)
    id: str = field(default_factory=lambda: str(uuid4()))


EventHandler = Callable[[Event], None]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[EventHandler]] = {}
        self.last_errors: list[str] = []

    def subscribe(self, event_name: str, handler: EventHandler) -> None:
        handlers = self._subscribers.setdefault(event_name, [])
        if handler not in handlers:
            handlers.append(handler)

    def publish(
        self,
        event_name: str,
        payload: dict[str, Any] | None = None,
        *,
        project_id: str | None = None,
        session_id: str | None = None,
    ) -> Event:
        event = Event(
            name=event_name,
            payload=payload or {},
            project_id=project_id,
            session_id=session_id,
        )
        self.last_errors = []
        for handler in [*self._subscribers.get(event_name, []), *self._subscribers.get("*", [])]:
            try:
                handler(event)
            except Exception as exc:
                self.last_errors.append(f"{event_name}: {handler!r}: {exc}")
        return event


class JsonlEventLogger:
    def __init__(self, log_dir: Path | None = None) -> None:
        self.log_dir = log_dir or paths.logs_dir()
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, event: Event) -> None:
        day = event.timestamp[:10]
        path = self.log_dir / f"events-{day}.jsonl"
        record = {
            "id": event.id,
            "name": event.name,
            "timestamp": event.timestamp,
            "project_id": event.project_id,
            "session_id": event.session_id,
            "payload": sanitize_for_log(event.payload),
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def sanitize_for_log(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return "[depth-limited]"
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(secret in lowered for secret in ("api_key", "authorization", "token", "messages")):
                result[str(key)] = "[redacted]"
            else:
                result[str(key)] = sanitize_for_log(item, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [sanitize_for_log(item, depth=depth + 1) for item in value[:50]]
    if isinstance(value, str):
        return value if len(value) <= 2000 else value[:2000] + "...[truncated]"
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return str(value)
