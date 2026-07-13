from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable
from uuid import uuid4

from .project import Project
from .timeutil import utc_now_iso


@dataclass
class QueueTask:
    id: str
    prompt: str
    status: str = "pending"
    session_id: str | None = None
    result: str = ""
    error: str = ""
    updated_at: str = field(default_factory=utc_now_iso)


@dataclass
class QueueRecord:
    id: str
    project_id: str
    status: str
    tasks: list[QueueTask]
    created_at: str
    updated_at: str
    schema_version: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


QueueRunner = Callable[[str], tuple[str, str | None, str]]


class TaskQueueManager:
    def __init__(self, project: Project) -> None:
        self.project = project
        self.queue_dir = project.agent_dir / "queues"
        self.queue_dir.mkdir(parents=True, exist_ok=True)

    def create(self, prompts: list[str]) -> QueueRecord:
        values = [prompt.strip() for prompt in prompts if prompt.strip()]
        if not values:
            raise ValueError("queue requires at least one non-empty task")
        stamp = utc_now_iso().replace("+00:00", "Z").replace("-", "").replace(":", "")
        queue_id = f"{stamp}-{uuid4().hex[:8]}"
        now = utc_now_iso()
        record = QueueRecord(
            id=queue_id,
            project_id=self.project.id,
            status="pending",
            tasks=[QueueTask(id=f"task-{index + 1}", prompt=prompt) for index, prompt in enumerate(values)],
            created_at=now,
            updated_at=now,
        )
        self.save(record)
        return record

    def run(self, record: QueueRecord, runner: QueueRunner, *, stop_on_failure: bool = True) -> QueueRecord:
        if record.project_id != self.project.id:
            raise ValueError("queue belongs to a different project")
        record.status = "running"
        self.save(record)
        for task in record.tasks:
            if task.status == "completed":
                continue
            if task.status in {"running", "paused"}:
                task.status = "pending"
            task.status = "running"
            task.error = ""
            task.updated_at = utc_now_iso()
            self.save(record)
            try:
                result, session_id, session_status = runner(task.prompt)
                task.result = result
                task.session_id = session_id
                task.status = "completed" if session_status == "completed" else "failed"
                if task.status == "failed":
                    task.error = f"session ended with status {session_status}"
            except KeyboardInterrupt:
                task.status = "paused"
                task.error = "interrupted by user"
                task.updated_at = utc_now_iso()
                record.status = "paused"
                self.save(record)
                raise
            except Exception as exc:
                task.status = "failed"
                task.error = str(exc)
            task.updated_at = utc_now_iso()
            self.save(record)
            if task.status == "failed" and stop_on_failure:
                record.status = "paused"
                self.save(record)
                return record
        record.status = "completed" if all(task.status == "completed" for task in record.tasks) else "paused"
        self.save(record)
        return record

    def save(self, record: QueueRecord) -> Path:
        record.updated_at = utc_now_iso()
        path = self._path(record.id)
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(record.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(path)
        return path

    def load(self, queue_id: str | None = None) -> QueueRecord:
        if queue_id is None:
            records = self.list(limit=1)
            if not records:
                raise FileNotFoundError("no saved queue is available")
            return records[0]
        matches = sorted(self.queue_dir.glob(f"{queue_id}*.json"))
        if len(matches) != 1:
            if not matches:
                raise FileNotFoundError(f"queue not found: {queue_id}")
            raise ValueError(f"queue prefix is ambiguous: {queue_id}")
        return self._read(matches[0])

    def list(self, limit: int = 20) -> list[QueueRecord]:
        records: list[QueueRecord] = []
        for path in sorted(self.queue_dir.glob("*.json"), reverse=True):
            try:
                records.append(self._read(path))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            if len(records) >= max(1, limit):
                break
        return records

    def _read(self, path: Path) -> QueueRecord:
        value = json.loads(path.read_text(encoding="utf-8"))
        tasks = [
            QueueTask(
                id=str(item.get("id") or f"task-{index + 1}"),
                prompt=str(item.get("prompt") or ""),
                status=str(item.get("status") or "pending"),
                session_id=str(item["session_id"]) if item.get("session_id") else None,
                result=str(item.get("result") or ""),
                error=str(item.get("error") or ""),
                updated_at=str(item.get("updated_at") or utc_now_iso()),
            )
            for index, item in enumerate(value.get("tasks", []))
            if isinstance(item, dict)
        ]
        return QueueRecord(
            id=str(value["id"]),
            project_id=str(value["project_id"]),
            status=str(value.get("status") or "pending"),
            tasks=tasks,
            created_at=str(value.get("created_at") or ""),
            updated_at=str(value.get("updated_at") or ""),
            schema_version=int(value.get("schema_version") or 1),
        )

    def _path(self, queue_id: str) -> Path:
        if not queue_id or any(
            ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in queue_id
        ):
            raise ValueError("invalid queue id")
        return self.queue_dir / f"{queue_id}.json"
