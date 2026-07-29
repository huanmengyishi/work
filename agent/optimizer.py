from __future__ import annotations

import math
import os
import re
import sqlite3
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .events import Event, EventBus


_TASK_TYPES = frozenset(
    {
        "question",
        "code_explanation",
        "document_workflow",
        "bug_fix",
        "feature_development",
        "review",
        "architecture",
        "refactor",
    }
)
_TASK_MODES = frozenset({"simple", "standard", "large", "deep"})
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@+-]*\Z")


@dataclass(frozen=True)
class TaskPerformance:
    """A terminal-task projection containing bounded metadata scalars only."""

    run_id: str
    project_id: str
    task_type: str
    task_mode: str
    outcome: str
    model_requests_total: int
    model_requests_main_loop: int
    model_requests_context_compaction: int
    model_requests_final_synthesis: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tool_calls: int
    tool_failures: int
    plan_steps_total: int
    plan_steps_completed: int
    elapsed_seconds: float
    recorded_at: str
    exploration_rounds: int = 0
    model_requests_memory_refinement: int = 0
    schema_version: int = 3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TaskPerformanceAnalyzer:
    """Project an allowlisted terminal Event without retaining task content."""

    MAX_COUNT = 1_000_000
    MAX_COLLECTION_ITEMS = 10_000
    MAX_TOKENS = 1_000_000_000
    MAX_ELAPSED_SECONDS = 7 * 24 * 60 * 60

    def analyze(self, event: Event) -> TaskPerformance | None:
        if not isinstance(event, Event) or event.name not in {"task.finished", "task.failed"}:
            return None
        state = event.payload.get("state")
        if not isinstance(state, Mapping):
            return None

        task_route = _mapping(state.get("task_route"))
        task_strategy = _mapping(state.get("task_strategy"))
        model_metrics = _mapping(state.get("model_metrics"))
        model_requests = {
            "main_loop": _bounded_int(
                state.get("main_loop_model_request_count"),
                maximum=self.MAX_COUNT,
            ),
            "context_compaction": _bounded_int(
                state.get("context_compaction_model_request_count"),
                maximum=self.MAX_COUNT,
            ),
            "final_synthesis": _bounded_int(
                state.get("final_synthesis_model_request_count"),
                maximum=self.MAX_COUNT,
            ),
            "memory_refinement": _bounded_int(
                state.get("memory_refinement_model_request_count"),
                maximum=self.MAX_COUNT,
            ),
        }
        phase_request_total = min(self.MAX_COUNT, sum(model_requests.values()))
        request_total = _bounded_int(state.get("model_request_count"), maximum=self.MAX_COUNT)
        if request_total == 0 and phase_request_total:
            request_total = phase_request_total

        tool_calls = state.get("tool_calls")
        tool_records = tool_calls[: self.MAX_COLLECTION_ITEMS] if isinstance(tool_calls, list) else []
        hot_tool_records = [
            item for item in tool_records if isinstance(item, Mapping) and item.get("type") != "pruned_history"
        ]
        marker_count = sum(isinstance(item, Mapping) and item.get("type") == "pruned_history" for item in tool_records)
        tool_history_summary = _mapping(state.get("tool_history_summary"))
        pruned_tool_calls = _bounded_int(tool_history_summary.get("count"), maximum=self.MAX_COUNT)
        tool_failures = sum(
            1
            for item in hot_tool_records
            if isinstance(item.get("result"), Mapping) and item["result"].get("success") is False
        )
        tool_failures += _bounded_int(tool_history_summary.get("failure_count"), maximum=self.MAX_COUNT)

        plan = state.get("plan")
        plan_records = plan[: self.MAX_COLLECTION_ITEMS] if isinstance(plan, list) else []
        plan_completed = sum(
            1 for item in plan_records if isinstance(item, Mapping) and item.get("status") == "completed"
        )

        execution_budget = _mapping(_mapping(state.get("convergence")).get("execution_budget"))
        convergence = _mapping(state.get("convergence"))
        budget_used = _mapping(execution_budget.get("used"))
        elapsed_seconds = _bounded_float(
            budget_used.get("elapsed_seconds"),
            maximum=float(self.MAX_ELAPSED_SECONDS),
        )

        project = _mapping(state.get("project"))
        task_type = task_route.get("task_type")
        task_mode = task_route.get("mode") or task_strategy.get("mode")
        return TaskPerformance(
            run_id=_identifier(event.effective_run_id or state.get("run_id"), fallback="unknown-run"),
            project_id=_identifier(event.project_id or project.get("id"), fallback="unknown-project"),
            task_type=task_type if isinstance(task_type, str) and task_type in _TASK_TYPES else "unknown",
            task_mode=task_mode if isinstance(task_mode, str) and task_mode in _TASK_MODES else "unknown",
            outcome="completed" if event.name == "task.finished" else "failed",
            model_requests_total=request_total,
            model_requests_main_loop=model_requests["main_loop"],
            model_requests_context_compaction=model_requests["context_compaction"],
            model_requests_final_synthesis=model_requests["final_synthesis"],
            model_requests_memory_refinement=model_requests["memory_refinement"],
            prompt_tokens=_bounded_int(model_metrics.get("prompt_tokens"), maximum=self.MAX_TOKENS),
            completion_tokens=_bounded_int(model_metrics.get("completion_tokens"), maximum=self.MAX_TOKENS),
            total_tokens=_bounded_int(model_metrics.get("total_tokens"), maximum=self.MAX_TOKENS),
            tool_calls=min(self.MAX_COUNT, len(tool_records) - marker_count + pruned_tool_calls),
            tool_failures=min(self.MAX_COUNT, tool_failures),
            plan_steps_total=len(plan_records),
            plan_steps_completed=min(self.MAX_COUNT, plan_completed),
            elapsed_seconds=elapsed_seconds,
            recorded_at=str(event.timestamp)[:64],
            exploration_rounds=_bounded_int(
                convergence.get("exploration_rounds_observed"),
                maximum=self.MAX_COUNT,
            ),
        )


class PerformanceHistory:
    """Bounded, idempotent SQLite history for safe TaskPerformance records.

    Storage is deliberately best effort: a read-only filesystem, a malformed
    database, or a lock timeout returns ``False``/an empty result and never
    changes task execution.
    """

    SCHEMA_VERSION = 3
    DEFAULT_MAX_RECORDS = 200
    MAX_RECORDS = 10_000
    MAX_DB_BYTES = 16 * 1024 * 1024
    _COLUMNS = (
        "run_id",
        "project_id",
        "task_type",
        "task_mode",
        "outcome",
        "model_requests_total",
        "model_requests_main_loop",
        "model_requests_context_compaction",
        "model_requests_final_synthesis",
        "model_requests_memory_refinement",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "tool_calls",
        "tool_failures",
        "plan_steps_total",
        "plan_steps_completed",
        "elapsed_seconds",
        "recorded_at",
        "exploration_rounds",
        "schema_version",
    )

    def __init__(self, path: Path, *, max_records: int = DEFAULT_MAX_RECORDS) -> None:
        self.path = Path(path)
        self.max_records = _bounded_int(max_records, maximum=self.MAX_RECORDS) or self.DEFAULT_MAX_RECORDS

    def record(self, performance: TaskPerformance) -> bool:
        if not isinstance(performance, TaskPerformance):
            return False
        try:
            self._prepare_parent()
            with self._connect() as connection:
                self._ensure_schema(connection)
                values = performance.to_dict()
                placeholders = ",".join("?" for _ in self._COLUMNS)
                connection.execute(
                    f"insert or ignore into task_performance ({','.join(self._COLUMNS)}) values ({placeholders})",
                    tuple(values[column] for column in self._COLUMNS),
                )
                connection.execute(
                    """
                    delete from task_performance
                    where sequence not in (
                        select sequence from task_performance order by sequence desc limit ?
                    )
                    """,
                    (self.max_records,),
                )
            self._secure_file()
            return True
        except (OSError, sqlite3.Error, TypeError, ValueError):
            return False

    def recent(self, *, limit: int = 20) -> list[TaskPerformance]:
        bounded_limit = min(self.max_records, _bounded_int(limit, maximum=self.MAX_RECORDS) or 1)
        if not self._readable_path():
            return []
        try:
            with self._connect() as connection:
                self._ensure_schema(connection)
                rows = connection.execute(
                    f"select {','.join(self._COLUMNS)} from task_performance order by sequence desc limit ?",
                    (bounded_limit,),
                ).fetchall()
        except (OSError, sqlite3.Error, TypeError, ValueError):
            return []
        return [TaskPerformance(**dict(row)) for row in rows]

    def count(self) -> int:
        if not self._readable_path():
            return 0
        try:
            with self._connect() as connection:
                self._ensure_schema(connection)
                row = connection.execute("select count(*) from task_performance").fetchone()
        except (OSError, sqlite3.Error, TypeError, ValueError):
            return 0
        return _bounded_int(row[0] if row else 0, maximum=self.max_records)

    def _prepare_parent(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.parent.is_symlink():
            raise OSError("performance directory must not be a symbolic link")
        try:
            self.path.parent.chmod(0o700)
        except OSError:
            pass
        if self.path.exists() or self.path.is_symlink():
            metadata = self.path.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                raise OSError("performance path must be a regular file")
            if metadata.st_size > self.MAX_DB_BYTES:
                raise OSError("performance database exceeds its size limit")

    def _readable_path(self) -> bool:
        try:
            if self.path.parent.is_symlink():
                return False
            metadata = self.path.lstat()
        except OSError:
            return False
        return stat.S_ISREG(metadata.st_mode) and metadata.st_size <= self.MAX_DB_BYTES

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=1.0)
        connection.row_factory = sqlite3.Row
        connection.execute("pragma busy_timeout = 1000")
        connection.execute("pragma journal_mode = delete")
        return connection

    @staticmethod
    def _ensure_schema(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            create table if not exists task_performance (
                sequence integer primary key,
                run_id text not null unique,
                project_id text not null,
                task_type text not null,
                task_mode text not null,
                outcome text not null check(outcome in ('completed', 'failed')),
                model_requests_total integer not null,
                model_requests_main_loop integer not null,
                model_requests_context_compaction integer not null,
                model_requests_final_synthesis integer not null,
                model_requests_memory_refinement integer not null default 0,
                prompt_tokens integer not null,
                completion_tokens integer not null,
                total_tokens integer not null,
                tool_calls integer not null,
                tool_failures integer not null,
                plan_steps_total integer not null,
                plan_steps_completed integer not null,
                elapsed_seconds real not null,
                recorded_at text not null,
                exploration_rounds integer not null default 0,
                schema_version integer not null
            )
            """
        )
        columns = {str(row[1]) for row in connection.execute("pragma table_info(task_performance)").fetchall()}
        if "model_requests_memory_refinement" not in columns:
            connection.execute(
                "alter table task_performance add column model_requests_memory_refinement integer not null default 0"
            )
        if "exploration_rounds" not in columns:
            connection.execute("alter table task_performance add column exploration_rounds integer not null default 0")
        connection.execute("create index if not exists idx_task_performance_recent on task_performance(sequence desc)")

    def _secure_file(self) -> None:
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass


class PerformanceAnalysisPipeline:
    """Observe terminal events only; never adjust Runtime strategy or config."""

    def __init__(
        self,
        history: PerformanceHistory,
        events: EventBus,
        *,
        analyzer: TaskPerformanceAnalyzer | None = None,
    ) -> None:
        self.history = history
        self.analyzer = analyzer or TaskPerformanceAnalyzer()
        events.subscribe("task.finished", self.handle, name="performance.history")
        events.subscribe("task.failed", self.handle, name="performance.history")

    def handle(self, event: Event) -> None:
        try:
            performance = self.analyzer.analyze(event)
            if performance is not None:
                self.history.record(performance)
        except Exception:
            # Performance observation is explicitly best effort and must never
            # turn a completed/failed task into a different Runtime outcome.
            return


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _bounded_int(value: Any, *, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        return 0
    return min(maximum, max(0, value))


def _bounded_float(value: Any, *, maximum: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return 0.0
    parsed = float(value)
    if not math.isfinite(parsed):
        return 0.0
    return min(maximum, max(0.0, parsed))


def _identifier(value: Any, *, fallback: str) -> str:
    if not isinstance(value, str):
        return fallback
    candidate = value.strip()
    if not candidate or len(candidate) > 128 or _SAFE_IDENTIFIER.fullmatch(candidate) is None:
        return fallback
    return candidate


__all__ = [
    "PerformanceAnalysisPipeline",
    "PerformanceHistory",
    "TaskPerformance",
    "TaskPerformanceAnalyzer",
]
