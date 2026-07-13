from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any

from .config import AppConfig
from .context import ContextBuilder
from .memory import MemoryStore
from .project import Project
from .task_queue import TaskQueueManager
from .timeutil import utc_now_iso


@dataclass(frozen=True)
class DaemonStatus:
    running: bool
    pid: int | None
    project_id: str
    project_root: str
    state: dict[str, Any]


class ProjectDaemon:
    def __init__(self, config: AppConfig, project: Project, memory: MemoryStore) -> None:
        self.config = config
        self.project = project
        self.memory = memory
        self.base_dir = config.data_dir / "daemon" / project.id
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.base_dir / "daemon.lock"
        self.pid_path = self.base_dir / "daemon.pid"
        self.stop_path = self.base_dir / "stop.requested"
        self.state_path = self.base_dir / "state.json"
        self.log_path = self.base_dir / "daemon.log"
        self._stop = Event()

    def start(self) -> int:
        status = self.status()
        if status.running and status.pid:
            return status.pid
        self.stop_path.unlink(missing_ok=True)
        command = [
            sys.executable,
            "-m",
            "agent",
            "daemon",
            "run",
            "--project",
            str(self.project.root),
        ]
        with self.log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=self.project.root,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = self.status()
            if status.running and status.pid:
                return status.pid
            if process.poll() is not None:
                raise RuntimeError(f"daemon exited during startup; inspect {self.log_path}")
            time.sleep(0.1)
        process.terminate()
        raise RuntimeError(f"daemon did not become ready; inspect {self.log_path}")

    def run(self, *, once: bool = False) -> int:
        lock_handle = self.lock_path.open("a+")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_handle.close()
            raise RuntimeError("daemon is already running for this project") from None

        self.stop_path.unlink(missing_ok=True)
        self._install_signal_handlers()
        self._write_pid(os.getpid())
        interval = max(2, min(int(self.config.get("daemon.poll_interval_seconds", 10)), 3600))
        maintenance_interval = max(
            60,
            min(int(self.config.get("daemon.memory_maintenance_seconds", 3600)), 7 * 24 * 3600),
        )
        last_fingerprint = ""
        last_maintenance = 0.0
        try:
            while not self._should_stop():
                started = time.monotonic()
                context = ContextBuilder(self.config).build(self.project)
                fingerprint = str(context.index.get("fingerprint") or "")
                context_changed = fingerprint != last_fingerprint
                last_fingerprint = fingerprint
                maintenance: dict[str, Any] | None = None
                now = time.monotonic()
                if now - last_maintenance >= maintenance_interval or last_maintenance == 0:
                    maintenance = self.memory.maintain(project_id=self.project.id, apply=True)
                    last_maintenance = now
                queued = self._run_one_pending_queue() if bool(self.config.get("daemon.queue_enabled", False)) else None
                self._write_state(
                    {
                        "status": "running",
                        "pid": os.getpid(),
                        "project_id": self.project.id,
                        "project_root": str(self.project.root),
                        "last_poll_at": utc_now_iso(),
                        "context_changed": context_changed,
                        "context_fingerprint": fingerprint,
                        "memory_maintenance": maintenance,
                        "queue_run": queued,
                        "poll_duration_ms": round((time.monotonic() - started) * 1000),
                    }
                )
                if once:
                    break
                self._stop.wait(interval)
        finally:
            self._write_state(
                {
                    **self._read_state(),
                    "status": "stopped",
                    "pid": None,
                    "stopped_at": utc_now_iso(),
                }
            )
            self.pid_path.unlink(missing_ok=True)
            self.stop_path.unlink(missing_ok=True)
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()
        return 0

    def stop(self, *, timeout: float = 10.0) -> bool:
        status = self.status()
        if not status.running or not status.pid:
            self._remove_stale_pid()
            return False
        self.stop_path.write_text(utc_now_iso() + "\n", encoding="utf-8")
        try:
            os.kill(status.pid, signal.SIGTERM)
        except ProcessLookupError:
            self._remove_stale_pid()
            return False
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self._pid_alive(status.pid):
                self._remove_stale_pid()
                return True
            time.sleep(0.1)
        return False

    def status(self) -> DaemonStatus:
        pid = self._read_pid()
        running = bool(pid and self._is_our_process(pid))
        if pid and not running:
            self._remove_stale_pid()
            pid = None
        state = self._read_state()
        state_project = str(state.get("project_id") or self.project.id)
        state_root = str(state.get("project_root") or self.project.root)
        return DaemonStatus(running, pid, state_project, state_root, state)

    def _run_one_pending_queue(self) -> dict[str, Any] | None:
        pending = next(
            (record for record in TaskQueueManager(self.project).list(limit=100) if record.status == "pending"), None
        )
        if pending is None:
            return None
        command = [
            sys.executable,
            "-m",
            "agent",
            "--auto-approve",
            "queue",
            "resume",
            "--id",
            pending.id,
        ]
        completed = subprocess.run(
            command,
            cwd=self.project.root,
            stdin=subprocess.DEVNULL,
            text=True,
            capture_output=True,
            timeout=max(60, int(self.config.get("tools.shell.timeout_seconds", 120)) * max(1, len(pending.tasks))),
            check=False,
        )
        with self.log_path.open("a", encoding="utf-8") as log:
            if completed.stdout:
                log.write(completed.stdout)
            if completed.stderr:
                log.write(completed.stderr)
        return {"id": pending.id, "returncode": completed.returncode}

    def _install_signal_handlers(self) -> None:
        def request_stop(_signum, _frame) -> None:
            self._stop.set()

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)

    def _should_stop(self) -> bool:
        return self._stop.is_set() or self.stop_path.exists()

    def _write_pid(self, pid: int) -> None:
        temp = self.pid_path.with_suffix(".tmp")
        temp.write_text(f"{pid}\n", encoding="ascii")
        temp.replace(self.pid_path)

    def _read_pid(self) -> int | None:
        try:
            return int(self.pid_path.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            return None

    def _remove_stale_pid(self) -> None:
        self.pid_path.unlink(missing_ok=True)

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            return False
        return True

    def _is_our_process(self, pid: int) -> bool:
        if not self._pid_alive(pid):
            return False
        try:
            command = (
                (Path("/proc") / str(pid) / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode("utf-8", errors="replace")
            )
        except OSError:
            return False
        return "agent daemon run" in command and str(self.project.root) in command

    def _read_state(self) -> dict[str, Any]:
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _write_state(self, value: dict[str, Any]) -> None:
        temp = self.state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(self.state_path)
