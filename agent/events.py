from __future__ import annotations

import json
import os
import re
import stat
import threading
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

    The bus deliberately stays synchronous and process-local in v0.10.0;
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


@dataclass(frozen=True)
class EventDelivery:
    """One subscriber delivery result without retaining private payload data."""

    event_name: str
    subscriber_name: str
    required: bool
    success: bool
    error: str = ""


@dataclass(frozen=True)
class EventDispatch:
    """Synchronous publication outcome used by required side-effect events."""

    event: Event
    deliveries: tuple[EventDelivery, ...]

    @property
    def handler_count(self) -> int:
        return len(self.deliveries)

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(item.error for item in self.deliveries if not item.success)

    @property
    def required_errors(self) -> tuple[str, ...]:
        return tuple(item.error for item in self.deliveries if item.required and not item.success)

    def subscriber_succeeded(self, subscriber_name: str) -> bool:
        return any(item.subscriber_name == subscriber_name and item.success for item in self.deliveries)


class EventDispatchError(RuntimeError):
    """Raised when a required event has no subscriber or one required handler fails."""

    def __init__(
        self,
        event_name: str,
        errors: tuple[str, ...],
        *,
        dispatch: EventDispatch | None = None,
    ) -> None:
        self.event_name = event_name
        self.errors = errors
        self.dispatch = dispatch
        super().__init__(f"required event dispatch failed for {event_name}: {'; '.join(errors)}")

    def subscriber_succeeded(self, subscriber_name: str) -> bool:
        return bool(self.dispatch and self.dispatch.subscriber_succeeded(subscriber_name))


@dataclass(frozen=True)
class _Subscription:
    handler: EventHandler
    required: bool
    name: str


class EventBus:
    """Minimal synchronous publish/subscribe boundary.

    A handler failure is isolated in ``last_errors`` and never prevents later
    subscribers from receiving the event.  This is not yet a durable queue.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[_Subscription]] = {}
        self.last_errors: list[str] = []
        self.last_dispatch: EventDispatch | None = None
        self._delivery_depth = 0
        # Read-only tools may execute concurrently, but their Event side
        # effects update shared Health, Audit, and Metrics stores.  Serialize
        # delivery while keeping the lock reentrant for nested publication.
        self._delivery_lock = threading.RLock()

    def subscribe(
        self,
        event_name: str,
        handler: EventHandler,
        *,
        required: bool = False,
        name: str | None = None,
    ) -> Callable[[], None]:
        event_name = _event_name(event_name)
        if not callable(handler):
            raise TypeError("event handler must be callable")
        subscription = _Subscription(
            handler=handler,
            required=bool(required),
            name=str(name or _handler_name(handler))[:200],
        )
        with self._delivery_lock:
            handlers = self._subscribers.setdefault(event_name, [])
            conflicting_name = next(
                (item for item in handlers if item.name == subscription.name and item.handler != handler),
                None,
            )
            if conflicting_name is not None:
                raise ValueError(f"event subscriber name is already registered: {event_name}: {subscription.name}")
            if not any(item.handler == handler for item in handlers):
                handlers.append(subscription)
        return lambda: self.unsubscribe(event_name, handler)

    def unsubscribe(self, event_name: str, handler: EventHandler) -> bool:
        event_name = _event_name(event_name)
        with self._delivery_lock:
            handlers = self._subscribers.get(event_name)
            if not handlers:
                return False
            match = next((item for item in handlers if item.handler == handler), None)
            if match is None:
                return False
            handlers.remove(match)
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
        event = self._event(
            event_name,
            payload,
            project_id=project_id,
            session_id=session_id,
            run_id=run_id,
        )
        return self._deliver(event).event

    def dispatch_required(
        self,
        event_name: str | Event,
        payload: dict[str, Any] | None = None,
        *,
        project_id: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> EventDispatch:
        """Publish a critical side-effect event and fail closed on required errors."""

        event = self._event(
            event_name,
            payload,
            project_id=project_id,
            session_id=session_id,
            run_id=run_id,
        )
        dispatch = self._deliver(event)
        if self.required_subscriber_count(event.name) == 0:
            raise EventDispatchError(event.name, ("no required subscribers registered",), dispatch=dispatch)
        if dispatch.required_errors:
            raise EventDispatchError(event.name, dispatch.required_errors, dispatch=dispatch)
        return dispatch

    def subscriber_count(self, event_name: str) -> int:
        """Return exact subscribers; wildcard audit handlers are not owners."""

        with self._delivery_lock:
            return len(self._subscribers.get(_event_name(event_name), ()))

    def required_subscriber_count(self, event_name: str) -> int:
        """Return exact required subscribers; observers cannot own persistence."""

        with self._delivery_lock:
            return sum(1 for item in self._subscribers.get(_event_name(event_name), ()) if item.required)

    @staticmethod
    def _event(
        event_name: str | Event,
        payload: dict[str, Any] | None,
        *,
        project_id: str | None,
        session_id: str | None,
        run_id: str | None,
    ) -> Event:
        if isinstance(event_name, Event):
            if payload is not None or project_id is not None or session_id is not None or run_id is not None:
                raise ValueError("an Event cannot be combined with publish metadata")
            return event_name
        if payload is not None and not isinstance(payload, dict):
            raise TypeError("event payload must be a dictionary")
        return Event(
            name=_event_name(event_name),
            payload=dict(payload) if payload is not None else {},
            project_id=_optional_identifier(project_id),
            session_id=_optional_identifier(session_id),
            run_id=_optional_identifier(run_id),
        )

    def _deliver(self, event: Event) -> EventDispatch:
        with self._delivery_lock:
            outermost = self._delivery_depth == 0
            self._delivery_depth += 1
            subscriptions = [*self._subscribers.get(event.name, ()), *self._subscribers.get("*", ())]
            deliveries: list[EventDelivery] = []
            try:
                for subscription in subscriptions:
                    try:
                        subscription.handler(event)
                        deliveries.append(EventDelivery(event.name, subscription.name, subscription.required, True))
                    except Exception as exc:
                        message = f"{event.name}: {subscription.name}: {exc}"
                        deliveries.append(
                            EventDelivery(
                                event.name,
                                subscription.name,
                                subscription.required,
                                False,
                                message[:2000],
                            )
                        )
            finally:
                self._delivery_depth -= 1
            dispatch = EventDispatch(event, tuple(deliveries))
            if outermost:
                self.last_dispatch = dispatch
                self.last_errors = list(dispatch.errors)
            return dispatch


class JsonlEventLogger:
    def __init__(self, log_dir: Path | None = None) -> None:
        self.log_dir = log_dir or paths.logs_dir()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.log_dir.is_symlink():
            raise OSError("audit log directory must not be a symbolic link")
        try:
            self.log_dir.chmod(0o700)
        except OSError:
            pass

    def __call__(self, event: Event) -> None:
        if self.log_dir.is_symlink():
            raise OSError("audit log directory must not be a symbolic link")
        path = self.log_dir / f"events-{_audit_day(event.timestamp)}.jsonl"
        try:
            existing = path.lstat()
        except FileNotFoundError:
            existing = None
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise OSError("audit log path must be a regular file")
        record = audit_event_projection(event)
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass


class AuditEventSubscriber(JsonlEventLogger):
    """Named Event Bus subscriber boundary for metadata-only JSONL audit."""


_AUDIT_SENSITIVE_KEYS = frozenset(
    {
        "agentstate",
        "accesstoken",
        "apikey",
        "args",
        "arguments",
        "authorization",
        "body",
        "content",
        "cookie",
        "cookies",
        "credential",
        "final",
        "finalanswer",
        "headers",
        "message",
        "messages",
        "password",
        "privatekey",
        "prompt",
        "reasoning",
        "reasoningcontent",
        "requestbody",
        "refreshtoken",
        "responsebody",
        "secret",
        "state",
        "stderr",
        "stdin",
        "stdout",
        "token",
        "toolargs",
        "toolarguments",
        "toolbody",
        "toolcontent",
        "tooloutput",
        "toolresult",
        "userrequest",
    }
)
_AUDIT_ALLOWED_FIELDS: dict[str, frozenset[str]] = {
    "task.started": frozenset(),
    "task.finished": frozenset(),
    "task.failed": frozenset(),
    "model.requested": frozenset({"round", "message_count", "model_tier", "model"}),
    "model.responded": frozenset({"round", "tool_call_count"}),
    "tool.started": frozenset({"request"}),
    "tool.finished": frozenset({"request", "result"}),
    "tool.denied": frozenset({"request", "result"}),
    "memory.summary.persisted": frozenset({"memory_id"}),
    "memory.experience.persisted": frozenset({"memory_id", "kind"}),
    "memory.reflection.persisted": frozenset({"memory_id"}),
    "memory.usage.recorded": frozenset({"memory_ids"}),
    "memory.usage.persisted": frozenset({"memory_ids"}),
    "session.checkpoint.requested": frozenset(),
    "session.finalize.requested": frozenset(),
    "ui.progress.updated": frozenset(),
}
_AUDIT_REQUEST_FIELDS = frozenset({"tool", "action", "capability", "request_id", "argument_count"})
_AUDIT_RESULT_FIELDS = frozenset({"success", "duration_ms", "request_id", "data_field_count", "health_failure"})
_AUDIT_MAX_KEYS = 32
_AUDIT_MAX_ITEMS = 50
_AUDIT_MAX_STRING_CHARS = 512


def audit_event_projection(event: Event) -> dict[str, Any]:
    """Return a bounded audit record that never serializes event payload objects.

    Audit is deliberately an allow-list projection rather than a general event
    serializer.  Internal persistence events may contain a live ``AgentState``
    and full message history for their required owner; those values must never
    reach JSONL even through an object's ``__str__`` implementation.
    """

    return {
        "schema_version": _bounded_audit_integer(event.schema_version),
        "id": _bounded_text(event.id, maximum=200),
        "name": _bounded_text(event.name, maximum=200),
        "timestamp": _bounded_text(event.timestamp, maximum=80),
        "project_id": _optional_audit_identifier(event.project_id),
        "session_id": _optional_audit_identifier(event.session_id),
        "run_id": _optional_audit_identifier(event.run_id),
        "payload": _audit_payload(event.name, event.payload),
    }


def sanitize_for_log(value: Any, *, depth: int = 0) -> Any:
    """Sanitize standalone diagnostic values without stringifying objects.

    ``JsonlEventLogger`` uses the stricter :func:`audit_event_projection`; this
    helper remains for bounded, explicit diagnostic values and compatibility.
    """

    if depth > 4:
        return "[depth-limited]"
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in list(value.items())[:_AUDIT_MAX_KEYS]:
            normalized = _normalize_audit_key(key)
            safe_key = _bounded_text(key, maximum=80)
            if _is_sensitive_audit_key(normalized):
                result[safe_key] = "[redacted]"
            else:
                result[safe_key] = sanitize_for_log(item, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [sanitize_for_log(item, depth=depth + 1) for item in value[:_AUDIT_MAX_ITEMS]]
    if isinstance(value, str):
        redacted = _redact_secret_patterns(value)
        return _bounded_text(redacted, maximum=_AUDIT_MAX_STRING_CHARS)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return "[unsupported]"


def _audit_payload(event_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    allowed = _AUDIT_ALLOWED_FIELDS.get(event_name, frozenset())
    projected: dict[str, Any] = {}
    for key in allowed:
        if key not in payload:
            continue
        value = payload[key]
        if key == "request" and isinstance(value, dict):
            projected[key] = _audit_named_fields(value, _AUDIT_REQUEST_FIELDS)
        elif key == "result" and isinstance(value, dict):
            projected[key] = _audit_named_fields(value, _AUDIT_RESULT_FIELDS)
        elif key in {"round", "message_count", "tool_call_count", "memory_id"}:
            projected[key] = _bounded_audit_integer(value)
        elif key == "memory_ids" and isinstance(value, list):
            projected[key] = [
                parsed for item in value[:_AUDIT_MAX_ITEMS] if (parsed := _bounded_audit_integer(item)) is not None
            ]
        elif key == "kind":
            projected[key] = _bounded_text(value, maximum=80)
        elif key in {"model", "model_tier"}:
            projected[key] = _bounded_text(value, maximum=120)
    return projected


def _audit_named_fields(value: dict[str, Any], allowed: frozenset[str]) -> dict[str, Any]:
    projected: dict[str, Any] = {}
    normalized_allowed = {_normalize_audit_key(item): item for item in allowed}
    for key, item in list(value.items())[:_AUDIT_MAX_KEYS]:
        normalized = _normalize_audit_key(key)
        canonical = normalized_allowed.get(normalized)
        if canonical is None:
            continue
        if canonical in {"success", "health_failure"}:
            projected[canonical] = item if isinstance(item, bool) else False
        elif canonical == "duration_ms":
            projected[canonical] = _bounded_audit_integer(item)
        elif canonical in {"argument_count", "data_field_count"}:
            projected[canonical] = _bounded_audit_integer(item)
        else:
            projected[canonical] = _bounded_text(item, maximum=160)
    return projected


def _normalize_audit_key(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[^a-z0-9]", "", value.casefold())[:100]


def _is_sensitive_audit_key(normalized: str) -> bool:
    if not normalized:
        return True
    return any(marker in normalized for marker in _AUDIT_SENSITIVE_KEYS)


def _bounded_audit_integer(value: Any) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    return max(0, min(value, 2**63 - 1))


def _bounded_text(value: Any, *, maximum: int) -> str:
    if not isinstance(value, str):
        return ""
    redacted = _redact_secret_patterns(value)
    if len(redacted) <= maximum:
        return redacted
    return redacted[:maximum] + "...[truncated]"


def _optional_audit_identifier(value: str | None) -> str | None:
    if value is None:
        return None
    return _bounded_text(value, maximum=240)


def _audit_day(timestamp: str) -> str:
    day = timestamp[:10]
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        raise ValueError("audit event timestamp has an invalid date prefix")
    return day


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


def _handler_name(handler: EventHandler) -> str:
    module = str(getattr(handler, "__module__", "") or "")
    qualified = str(getattr(handler, "__qualname__", "") or "")
    if qualified:
        return f"{module}.{qualified}".strip(".")
    owner = getattr(handler, "__self__", None)
    method = str(getattr(handler, "__name__", "") or "")
    if owner is not None and method:
        return f"{owner.__class__.__module__}.{owner.__class__.__qualname__}.{method}"
    return handler.__class__.__qualname__


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
