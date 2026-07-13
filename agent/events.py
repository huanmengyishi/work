from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from . import paths
from .contracts import EVENT_SCHEMA_VERSION, EVENT_SERIALIZED_FIELDS
from .timeutil import utc_now_iso


@dataclass(frozen=True)
class Event:
    """Immutable, versioned notification exchanged by runtime components.

    The bus deliberately stays synchronous and process-local in v0.9.1;
    persistence, replay, and delivery guarantees remain outside this minimal
    contract. Publishers should pass a fresh payload and handlers should not
    mutate it while another subscriber may still inspect it.
    """

    name: str
    payload: dict[str, Any]
    project_id: str | None = None
    session_id: str | None = None
    run_id: str | None = None
    timestamp: str = field(default_factory=utc_now_iso)
    id: str = field(default_factory=lambda: str(uuid4()))
    schema_version: int = EVENT_SCHEMA_VERSION

    SERIALIZED_FIELDS = EVENT_SERIALIZED_FIELDS

    def __post_init__(self) -> None:
        if not isinstance(self.payload, dict):
            raise TypeError("event payload must be a dictionary")
        object.__setattr__(self, "payload", dict(self.payload))
        if not isinstance(self.schema_version, int) or isinstance(self.schema_version, bool):
            raise TypeError("event schema_version must be an integer")
        if self.schema_version < 1:
            raise ValueError("event schema_version must be positive")
        for field_name in ("id", "name", "timestamp"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"event {field_name} must be a non-empty string")
        for field_name in ("project_id", "session_id", "run_id"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"event {field_name} must be null or a non-empty string")
        _validate_timestamp(self.timestamp)

    @property
    def effective_run_id(self) -> str | None:
        """Return explicit correlation metadata, then the legacy payload value."""

        payload_run_id = self.payload.get("run_id")
        if self.run_id:
            return self.run_id
        if isinstance(payload_run_id, str) and payload_run_id.strip():
            return payload_run_id
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "name": self.name,
            "timestamp": self.timestamp,
            "project_id": self.project_id,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Event":
        if not isinstance(value, dict):
            raise TypeError("event record must be a dictionary")
        payload = value.get("payload")
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise TypeError("event payload must be a dictionary")
        return cls(
            schema_version=_positive_int(value.get("schema_version"), default=EVENT_SCHEMA_VERSION),
            id=str(value.get("id") or uuid4()),
            name=str(value.get("name") or ""),
            timestamp=str(value.get("timestamp") or utc_now_iso()),
            project_id=_optional_identifier(value.get("project_id")),
            session_id=_optional_identifier(value.get("session_id")),
            run_id=_optional_identifier(value.get("run_id")),
            payload=dict(payload),
        )


EventHandler = Callable[[Event], None]


class EventBus:
    """Minimal synchronous publish/subscribe boundary.

    A handler failure is isolated in ``last_errors`` and never prevents later
    subscribers from receiving the event.  This is not yet a durable queue.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[EventHandler]] = {}
        self.last_errors: list[str] = []

    def subscribe(self, event_name: str, handler: EventHandler) -> Callable[[], None]:
        event_name = _event_name(event_name)
        if not callable(handler):
            raise TypeError("event handler must be callable")
        handlers = self._subscribers.setdefault(event_name, [])
        if handler not in handlers:
            handlers.append(handler)
        return lambda: self.unsubscribe(event_name, handler)

    def unsubscribe(self, event_name: str, handler: EventHandler) -> bool:
        event_name = _event_name(event_name)
        handlers = self._subscribers.get(event_name)
        if not handlers or handler not in handlers:
            return False
        handlers.remove(handler)
        if not handlers:
            self._subscribers.pop(event_name, None)
        return True

    def publish(
        self,
        event_name: str | Event,
        payload: dict[str, Any] | None = None,
        *,
        project_id: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> Event:
        if isinstance(event_name, Event):
            if payload is not None or project_id is not None or session_id is not None or run_id is not None:
                raise ValueError("an Event cannot be combined with publish metadata")
            event = event_name
        else:
            if payload is not None and not isinstance(payload, dict):
                raise TypeError("event payload must be a dictionary")
            event = Event(
                name=_event_name(event_name),
                payload=dict(payload) if payload is not None else {},
                project_id=_optional_identifier(project_id),
                session_id=_optional_identifier(session_id),
                run_id=_optional_identifier(run_id),
            )
        handlers = [*self._subscribers.get(event.name, []), *self._subscribers.get("*", [])]
        publish_errors: list[str] = []
        for handler in handlers:
            try:
                handler(event)
            except Exception as exc:
                publish_errors.append(f"{event.name}: {handler!r}: {exc}")
        self.last_errors = publish_errors
        return event


class JsonlEventLogger:
    def __init__(self, log_dir: Path | None = None) -> None:
        self.log_dir = log_dir or paths.logs_dir()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.log_dir.chmod(0o700)
        except OSError:
            pass

    def __call__(self, event: Event) -> None:
        day = event.timestamp[:10]
        path = self.log_dir / f"events-{day}.jsonl"
        record = event.to_dict()
        record["payload"] = sanitize_for_log(event.payload)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass


def sanitize_for_log(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return "[depth-limited]"
    if isinstance(value, dict) or hasattr(value, "items"):
        result: dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if any(
                secret in lowered
                for secret in (
                    "api_key",
                    "apikey",
                    "authorization",
                    "cookie",
                    "password",
                    "secret",
                    "token",
                    "messages",
                )
            ):
                result[str(key)] = "[redacted]"
            else:
                result[str(key)] = sanitize_for_log(item, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [sanitize_for_log(item, depth=depth + 1) for item in value[:50]]
    if isinstance(value, str):
        redacted = _redact_secret_patterns(value)
        return redacted if len(redacted) <= 2000 else redacted[:2000] + "...[truncated]"
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return _redact_secret_patterns(str(value))


_SECRET_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"(?i)\b(DEEPSEEK_API_KEY|API_KEY|ACCESS_TOKEN|REFRESH_TOKEN)\s*[=:]\s*[^\s,;]+"),
)


def _redact_secret_patterns(value: str) -> str:
    redacted = value
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[redacted]", redacted)
    return redacted


def _event_name(value: Any) -> str:
    name = str(value or "").strip()
    if not name:
        raise ValueError("event name must be a non-empty string")
    return name


def _optional_identifier(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("event identifiers must be null or non-empty strings")
    return normalized


def _positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        parsed = default
    if parsed < 1:
        raise ValueError("event schema_version must be positive")
    return parsed


def _validate_timestamp(value: str) -> None:
    try:
        datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("event timestamp must be an ISO-8601 timestamp") from exc
