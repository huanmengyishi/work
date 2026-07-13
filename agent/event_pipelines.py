from __future__ import annotations

import json
import os
import stat
from collections import Counter
from pathlib import Path
from typing import Any
from uuid import uuid4

from .capability_health import CapabilityHealthManager
from .config import AppConfig
from .events import AuditEventSubscriber, Event, EventBus
from .memory import MemoryStore
from .memory_pipeline import MemoryPipeline
from .paths import storage_key
from .project import Project
from .session import SessionManager
from .state import AgentState
from .timeutil import utc_now_iso


SESSION_CHECKPOINT_REQUESTED = "session.checkpoint.requested"
SESSION_FINALIZE_REQUESTED = "session.finalize.requested"
MEMORY_USAGE_RECORDED = "memory.usage.recorded"
PROGRESS_UPDATED = "ui.progress.updated"


class SessionEventPipeline:
    """Own Session writes requested through the in-process Event Bus."""

    def __init__(self, sessions: SessionManager, events: EventBus) -> None:
        self.sessions = sessions
        self.events = events
        events.subscribe(
            SESSION_CHECKPOINT_REQUESTED,
            self.checkpoint,
            required=True,
            name="session.checkpoint-writer",
        )
        events.subscribe(
            SESSION_FINALIZE_REQUESTED,
            self.finalize,
            required=True,
            name="session.finalize-writer",
        )

    def checkpoint(self, event: Event) -> None:
        state, messages = _state_and_messages(event)
        self.sessions.checkpoint(state, messages)

    def finalize(self, event: Event) -> None:
        state, messages = _state_and_messages(event)
        self.sessions.finalize(state, messages)


class MemoryUsageEventPipeline:
    """Record only Memory IDs that entered a ContextPackage."""

    def __init__(self, memory: MemoryStore, events: EventBus) -> None:
        self.memory = memory
        events.subscribe(
            MEMORY_USAGE_RECORDED,
            self.handle,
            required=True,
            name="memory.usage-writer",
        )

    def handle(self, event: Event) -> None:
        raw_ids = event.payload.get("memory_ids", [])
        if not isinstance(raw_ids, list):
            raise TypeError("memory usage event requires a memory_ids list")
        memory_ids = tuple(dict.fromkeys(int(item) for item in raw_ids))
        if not memory_ids or len(memory_ids) > 1000 or any(item <= 0 for item in memory_ids):
            raise ValueError("memory usage event requires 1 to 1000 positive IDs")
        usage_id = str(event.payload.get("usage_id") or "").strip()
        run_id = str(event.effective_run_id or "").strip()
        if not usage_id or not run_id:
            raise ValueError("memory usage event requires usage_id and run_id")
        self.memory.record_usage_once(
            usage_id,
            memory_ids,
            run_id=run_id,
            project_id=event.project_id,
        )


class CapabilityHealthEventPipeline:
    """Project tool outcomes into the existing persisted health store."""

    def __init__(self, health: CapabilityHealthManager, events: EventBus) -> None:
        self.health = health
        events.subscribe("tool.finished", self.handle, name="capability-health")

    def handle(self, event: Event) -> None:
        request = event.payload.get("request")
        result = event.payload.get("result")
        if not isinstance(request, dict) or not isinstance(result, dict):
            return
        capability = str(request.get("capability") or "").strip()
        if not capability:
            return
        success = bool(result.get("success"))
        health_failure = bool(result.get("health_failure", False))
        if success:
            self.health.record(capability, success=True)
        elif health_failure:
            self.health.record(capability, success=False, error=str(result.get("error") or ""))


class ProgressEventPipeline:
    """Forward ephemeral progress events without making UI availability critical."""

    def __init__(self, handler: Any, events: EventBus) -> None:
        if not callable(handler):
            raise TypeError("progress handler must be callable")
        self.handler = handler
        events.subscribe(PROGRESS_UPDATED, self.handle, name="ui.progress")

    def handle(self, event: Event) -> None:
        value = event.payload.get("value")
        if not isinstance(value, dict):
            return
        self.handler(dict(value))


class EventMetricsCollector:
    """Persist bounded counters only; never Prompt, reasoning, or tool output."""

    SCHEMA_VERSION = 1
    MAX_COUNTER = 1_000_000_000_000
    MAX_TOOL_DURATION_MS = 10 * 365 * 24 * 60 * 60 * 1000
    MAX_EVENT_DURATION_MS = 7 * 24 * 60 * 60 * 1000
    MAX_FILE_BYTES = 64 * 1024

    ALLOWED_EVENTS = frozenset(
        {
            "task.started",
            "task.finished",
            "task.failed",
            "model.requested",
            "model.responded",
            "tool.started",
            "tool.finished",
            "tool.denied",
        }
    )

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.parent.is_symlink():
            raise OSError("metrics directory must not be a symbolic link")
        try:
            self.path.parent.chmod(0o700)
        except OSError:
            pass
        self.counts: Counter[str] = Counter()
        self.total_tool_duration_ms = 0
        self.failed_tools = 0
        self._read()

    def __call__(self, event: Event) -> None:
        if event.name not in self.ALLOWED_EVENTS:
            return
        self.counts[event.name] = _saturating_add(
            self.counts[event.name],
            1,
            maximum=self.MAX_COUNTER,
        )
        result = event.payload.get("result")
        if event.name in {"tool.finished", "tool.denied"} and isinstance(result, dict):
            duration_ms = _bounded_non_negative_int(
                result.get("duration_ms"),
                maximum=self.MAX_EVENT_DURATION_MS,
            )
            self.total_tool_duration_ms = _saturating_add(
                self.total_tool_duration_ms,
                duration_ms,
                maximum=self.MAX_TOOL_DURATION_MS,
            )
            success = result.get("success")
            if success is False:
                self.failed_tools = _saturating_add(
                    self.failed_tools,
                    1,
                    maximum=self.MAX_COUNTER,
                )
        self._write()

    def _read(self) -> None:
        try:
            existing = self.path.lstat()
            if not stat.S_ISREG(existing.st_mode):
                return
            if existing.st_size > self.MAX_FILE_BYTES:
                return
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return
        if not isinstance(value, dict):
            return
        counts = value.get("counts")
        if isinstance(counts, dict):
            for key, item in list(counts.items())[: len(self.ALLOWED_EVENTS)]:
                if isinstance(key, str) and key in self.ALLOWED_EVENTS:
                    self.counts[key] = _bounded_non_negative_int(
                        item,
                        maximum=self.MAX_COUNTER,
                    )
        self.total_tool_duration_ms = _bounded_non_negative_int(
            value.get("total_tool_duration_ms"),
            maximum=self.MAX_TOOL_DURATION_MS,
        )
        self.failed_tools = _bounded_non_negative_int(
            value.get("failed_tools"),
            maximum=self.MAX_COUNTER,
        )

    def _write(self) -> None:
        if self.path.parent.is_symlink():
            raise OSError("metrics directory must not be a symbolic link")
        try:
            existing = self.path.lstat()
        except FileNotFoundError:
            existing = None
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise OSError("metrics path must be a regular file")
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "updated_at": utc_now_iso(),
            "counts": dict(sorted(self.counts.items())),
            "total_tool_duration_ms": self.total_tool_duration_ms,
            "failed_tools": self.failed_tools,
        }
        temp = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        try:
            descriptor = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            temp.replace(self.path)
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass
        finally:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass


class RuntimeEventPipelines:
    """Register all automatic Runtime side-effect subscribers in one place."""

    def __init__(
        self,
        *,
        config: AppConfig,
        project: Project,
        sessions: SessionManager,
        memory: MemoryStore,
        health: CapabilityHealthManager,
        events: EventBus,
        progress_handler: Any = None,
    ) -> None:
        self.session = SessionEventPipeline(sessions, events)
        self.memory_usage = MemoryUsageEventPipeline(memory, events)
        self.capability_health = CapabilityHealthEventPipeline(health, events)
        self.progress = ProgressEventPipeline(progress_handler, events) if callable(progress_handler) else None
        self.memory = MemoryPipeline(config=config, project=project, memory=memory, events=events)
        self.audit = None
        if bool(config.get("events.jsonl_log", True)):
            self.audit = AuditEventSubscriber(config.data_dir / "logs")
            events.subscribe("*", self.audit, name="audit.jsonl")
        self.metrics = None
        if bool(config.get("events.metrics_enabled", True)):
            self.metrics = EventMetricsCollector(config.data_dir / "metrics" / f"{storage_key(project.id)}.json")
            events.subscribe("*", self.metrics, name="metrics.project")


def _state_and_messages(event: Event) -> tuple[AgentState, list[dict[str, Any]]]:
    state = event.payload.get("state")
    messages = event.payload.get("messages")
    if not isinstance(state, AgentState):
        raise TypeError(f"{event.name} requires an AgentState")
    if not isinstance(messages, list) or not all(isinstance(item, dict) for item in messages):
        raise TypeError(f"{event.name} requires a message list")
    if event.session_id and event.session_id != state.session_id:
        raise ValueError(f"{event.name} session identity does not match AgentState")
    if event.run_id and event.run_id != state.run_id:
        raise ValueError(f"{event.name} run identity does not match AgentState")
    return state, messages


def _bounded_non_negative_int(value: Any, *, maximum: int) -> int:
    """Accept JSON integers only and clamp persisted metrics to a fixed bound."""

    if not isinstance(value, int) or isinstance(value, bool):
        return 0
    return min(maximum, max(0, value))


def _saturating_add(current: Any, increment: Any, *, maximum: int) -> int:
    left = _bounded_non_negative_int(current, maximum=maximum)
    right = _bounded_non_negative_int(increment, maximum=maximum)
    return min(maximum, left + right)
