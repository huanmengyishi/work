from __future__ import annotations

import fcntl
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from .project import Project
from .state import AgentState
from .timeutil import utc_now_iso


SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass
class SessionRecord:
    state: AgentState
    messages: list[dict[str, Any]]


@dataclass(frozen=True)
class SessionInfo:
    session_id: str
    status: str
    turn: int
    user_request: str
    updated_at: str
    path: Path


class SessionLease:
    def __init__(self, handle) -> None:
        self.handle = handle

    def __enter__(self) -> SessionLease:
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        return False


class SessionManager:
    MAX_SESSION_FILE_BYTES = 64 * 1024 * 1024
    MAX_MESSAGES = 20_000

    def __init__(self, project: Project) -> None:
        self.project = project
        self.session_dir = project.agent_dir / "sessions"
        self.session_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def new_session_id() -> str:
        stamp = utc_now_iso().replace("+00:00", "Z").replace("-", "").replace(":", "")
        return f"{stamp}-{uuid4().hex[:8]}"

    def checkpoint(self, state: AgentState, messages: list[dict[str, Any]]) -> Path:
        if len(messages) > self.MAX_MESSAGES or not all(isinstance(item, dict) for item in messages):
            raise ValueError(f"session messages exceed the {self.MAX_MESSAGES} item limit or contain invalid records")
        state.touch()
        path = self._json_path(state.session_id)
        payload = {
            "schema_version": 1,
            "state": state.to_dict(),
            "messages": messages,
        }
        content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        if len(content.encode("utf-8")) > self.MAX_SESSION_FILE_BYTES:
            raise ValueError(f"session checkpoint exceeds the {self.MAX_SESSION_FILE_BYTES} byte limit")
        self._atomic_write(path, content)
        return path

    def finalize(self, state: AgentState, messages: list[dict[str, Any]]) -> tuple[Path, Path]:
        json_path = self.checkpoint(state, messages)
        markdown_path = self._markdown_path(state.session_id)
        plan_lines = [f"- [{self._mark(step.status)}] {step.id}: {step.title} ({step.status})" for step in state.plan]
        tool_lines = []
        for item in state.tool_calls:
            request = item.get("request") or {}
            result = item.get("result") or {}
            tool_lines.append(
                f"- tool turn {item.get('round', '-')}: {request.get('tool', '?')}.{request.get('action', '?')} "
                f"success={result.get('success', False)} duration_ms={result.get('duration_ms', 0)}"
            )
        content = "\n".join(
            [
                f"# Session {state.session_id}",
                "",
                f"- Status: `{state.status}`",
                f"- Turn: `{state.turn}`",
                f"- Updated: `{state.updated_at}`",
                f"- Project: `{state.project.get('name', '')}`",
                "",
                "## User Request",
                "",
                "### Original Objective",
                "",
                state.objective.strip(),
                "",
                "### Current Turn Request",
                "",
                state.user_request.strip(),
                "",
                "## Plan",
                "",
                *(plan_lines or ["No explicit plan was recorded."]),
                "",
                "## Tool Calls",
                "",
                *(tool_lines or ["No tool calls were recorded."]),
                "",
                "## Final Answer",
                "",
                state.final_answer.strip(),
                "",
                "## Error",
                "",
                state.error.strip() or "None.",
                "",
            ]
        )
        self._atomic_write(markdown_path, content)
        return json_path, markdown_path

    def load(self, session_id: str | None = None) -> SessionRecord:
        resolved = self.resolve_session_id(session_id)
        path = self._json_path(resolved)
        payload = self._read_payload(path)
        state_data = payload.get("state")
        messages = payload.get("messages")
        if (
            not isinstance(state_data, dict)
            or not isinstance(messages, list)
            or len(messages) > self.MAX_MESSAGES
            or not all(isinstance(item, dict) for item in messages)
        ):
            raise ValueError(f"invalid session file: {path}")
        return SessionRecord(state=AgentState.from_dict(state_data), messages=messages)

    def acquire(self, session_id: str) -> SessionLease:
        """Exclusively lease one Session turn so concurrent resumes cannot replay it."""
        path = self.session_dir / f".{session_id}.lock"
        handle = path.open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            raise RuntimeError(f"session is already being resumed: {session_id}") from None
        return SessionLease(handle)

    def list_sessions(self, limit: int = 20) -> list[SessionInfo]:
        items: list[SessionInfo] = []
        paths = sorted(self.session_dir.glob("*.json"), key=lambda item: item.lstat().st_mtime_ns, reverse=True)
        for path in paths:
            try:
                payload = self._read_payload(path)
                state = payload.get("state") or {}
                items.append(
                    SessionInfo(
                        session_id=str(state.get("session_id") or path.stem),
                        status=str(state.get("status") or "unknown"),
                        turn=int(state.get("turn") or 1),
                        user_request=str(state.get("objective") or state.get("user_request") or ""),
                        updated_at=str(state.get("updated_at") or ""),
                        path=path,
                    )
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
            if len(items) >= limit:
                break
        return items

    def _read_payload(self, path: Path) -> dict[str, Any]:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"session path is not a regular file: {path}")
        if metadata.st_size > self.MAX_SESSION_FILE_BYTES:
            raise ValueError(f"session file exceeds the {self.MAX_SESSION_FILE_BYTES} byte limit: {path}")
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"invalid session file: {path}")
        return payload

    def resolve_session_id(self, session_id: str | None) -> str:
        if not session_id:
            sessions = self.list_sessions(limit=1)
            if not sessions:
                raise FileNotFoundError("no saved session is available")
            return sessions[0].session_id
        if not SESSION_ID_RE.fullmatch(session_id):
            raise ValueError("session id contains unsupported characters")
        exact = self._json_path(session_id)
        if exact.exists():
            return session_id
        matches = sorted(self.session_dir.glob(f"{session_id}*.json"))
        if len(matches) == 1:
            return matches[0].stem
        if not matches:
            raise FileNotFoundError(f"session not found: {session_id}")
        raise ValueError(f"session prefix is ambiguous: {session_id}")

    def _json_path(self, session_id: str) -> Path:
        if not SESSION_ID_RE.fullmatch(session_id):
            raise ValueError("invalid session id")
        return self.session_dir / f"{session_id}.json"

    def _markdown_path(self, session_id: str) -> Path:
        return self.session_dir / f"{session_id}.md"

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temp.write_text(content, encoding="utf-8")
        temp.replace(path)

    @staticmethod
    def _mark(status: str) -> str:
        return "x" if status in {"completed", "skipped"} else " "
