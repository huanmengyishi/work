from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from .artifact import (
    MANAGED_DOCUMENT_ARTIFACT_ID,
    MAX_ARTIFACT_BYTES_HARD_LIMIT,
    ArtifactSpec,
    ArtifactVerifier,
)
from .file_lock import FileLockUnavailable, lock_exclusive, unlock
from .progress import ProgressTracker
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
        unlock(self.handle)
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
        progress = ProgressTracker.snapshot(state)
        progress_lines = [
            f"- Plan progress: `{progress.percent:.1f}%`",
            f"- Satisfied steps: `{progress.completed_steps + progress.skipped_steps}/{progress.total_steps}`",
            f"- Model requests: `{progress.model_requests_used}`",
            f"- Reported tokens: `{progress.tokens_used}`",
        ]
        deliverable_lines = self._deliverable_lines(state)
        verification_lines = self._verification_lines(state)
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
                "## Progress",
                "",
                *progress_lines,
                "",
                "## Deliverables",
                "",
                *(deliverable_lines or ["No explicit deliverable was requested."]),
                "",
                "## Verification",
                "",
                *(verification_lines or ["No managed verification attempt was recorded."]),
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
            lock_exclusive(handle, nonblocking=True)
        except FileLockUnavailable:
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

    @staticmethod
    def _deliverable_lines(state: AgentState) -> list[str]:
        hints = list((state.task_route or {}).get("artifact_hints") or [])
        for step in state.plan:
            hints.extend(step.artifact_ids)
        hints = list(dict.fromkeys(str(item) for item in hints if str(item).strip()))[:64]
        active_applies = SessionManager._active_file_applies(state)
        directory_paths: list[str] = []
        parsed: list[dict[str, Any]] = []
        for index, item in enumerate(state.tool_calls):
            request = item.get("request") if isinstance(item.get("request"), dict) else {}
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            data = result.get("data") if isinstance(result.get("data"), dict) else {}
            if not bool(result.get("success")) or bool(data.get("not_executed")):
                continue
            capability = (str(request.get("tool") or ""), str(request.get("action") or ""))
            args = request.get("args") if isinstance(request.get("args"), dict) else {}
            if capability == ("template", "make_dir"):
                directory_paths.append(str(data.get("path") or args.get("path") or ""))
            elif capability == ("document", "parse"):
                parsed.append(
                    {
                        "index": index,
                        "path": str(args.get("path") or data.get("path") or ""),
                        "result": result,
                    }
                )
        lines: list[str] = []
        for hint in hints:
            matching_applies = [
                item for item in active_applies if SessionManager._hint_matches(hint, str(item["path"]))
            ]
            generated = bool(matching_applies) or any(
                SessionManager._hint_matches(hint, path) for path in directory_paths
            )
            verified = any(
                SessionManager._parse_verifies_apply(state, applied, parse_record)
                for applied in matching_applies
                for parse_record in parsed
            )
            status = "verified" if verified else "generated" if generated else "missing"
            lines.append(f"- `{hint}`: `{status}` (derived from managed tool receipts)")
        return lines

    @staticmethod
    def _active_file_applies(state: AgentState) -> list[dict[str, Any]]:
        successful: list[dict[str, Any]] = []
        for item in state.tool_calls:
            if not isinstance(item, dict):
                continue
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            data = result.get("data") if isinstance(result.get("data"), dict) else {}
            if bool(result.get("success")) and not bool(data.get("not_executed")):
                successful.append(item)
        undone = {
            str(((item.get("result") or {}).get("data") or {}).get("snapshot_id") or "")
            for item in successful
            if (
                str((item.get("request") or {}).get("tool") or ""),
                str((item.get("request") or {}).get("action") or ""),
            )
            == ("file", "undo")
        }
        undone.discard("")
        latest_by_path: dict[str, dict[str, Any]] = {}
        for index, item in enumerate(state.tool_calls):
            if item not in successful:
                continue
            request = item.get("request") if isinstance(item.get("request"), dict) else {}
            if (str(request.get("tool") or ""), str(request.get("action") or "")) != ("file", "apply"):
                continue
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            data = result.get("data") if isinstance(result.get("data"), dict) else {}
            snapshot_id = str(data.get("snapshot_id") or "")
            if snapshot_id and snapshot_id in undone:
                continue
            path = str(data.get("path") or "").strip().replace("\\", "/")
            if not path:
                continue
            latest_by_path[path] = {
                "index": index,
                "path": path,
                "after_exists": data.get("after_exists") if isinstance(data.get("after_exists"), bool) else None,
            }
        return [item for item in latest_by_path.values() if item["after_exists"] is True]

    @staticmethod
    def _parse_verifies_apply(
        state: AgentState,
        applied: dict[str, Any],
        parsed: dict[str, Any],
    ) -> bool:
        if int(parsed.get("index") or 0) <= int(applied.get("index") or 0):
            return False
        path = str(applied.get("path") or "")
        if not SessionManager._same_recorded_path(state, path, str(parsed.get("path") or "")):
            return False
        if not path.lower().endswith(".docx"):
            return True
        try:
            spec = ArtifactSpec(
                MANAGED_DOCUMENT_ARTIFACT_ID,
                path,
                format="docx",
                max_bytes=MAX_ARTIFACT_BYTES_HARD_LIMIT,
            )
        except ValueError:
            return False
        result = parsed.get("result") if isinstance(parsed.get("result"), dict) else {}
        return ArtifactVerifier.verify_receipt(spec, result).passed

    @staticmethod
    def _verification_lines(state: AgentState) -> list[str]:
        lines: list[str] = []
        for item in state.tool_calls:
            request = item.get("request") if isinstance(item.get("request"), dict) else {}
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            data = result.get("data") if isinstance(result.get("data"), dict) else {}
            capability = f"{request.get('tool', '')}.{request.get('action', '')}"
            if capability not in {"template.run_tests", "lsp.diagnostics", "document.parse"}:
                continue
            if bool(data.get("runtime_denied")) or bool(data.get("not_executed")):
                continue
            passed = bool(result.get("success"))
            args = request.get("args") if isinstance(request.get("args"), dict) else {}
            path = str(args.get("path") or data.get("path") or "")
            if capability == "document.parse" and path.lower().endswith(".docx") and passed:
                try:
                    relative = SessionManager._relative_recorded_path(state, path)
                    spec = ArtifactSpec(
                        MANAGED_DOCUMENT_ARTIFACT_ID,
                        relative,
                        format="docx",
                        max_bytes=MAX_ARTIFACT_BYTES_HARD_LIMIT,
                    )
                    passed = ArtifactVerifier.verify_receipt(spec, result).passed
                except ValueError:
                    passed = False
            outcome = "passed" if passed else "failed"
            lines.append(f"- `{capability}`: `{outcome}`")
        return lines[-50:]

    @staticmethod
    def _relative_recorded_path(state: AgentState, value: str) -> str:
        raw = str(value).strip().replace("\\", "/")
        if not raw:
            raise ValueError("recorded path is empty")
        path = Path(raw)
        root = Path(str((state.project or {}).get("root") or state.working_directory)).resolve(strict=False)
        resolved = path.resolve(strict=False) if path.is_absolute() else (root / path).resolve(strict=False)
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError("recorded path is outside the project") from exc

    @staticmethod
    def _same_recorded_path(state: AgentState, left: str, right: str) -> bool:
        try:
            return SessionManager._relative_recorded_path(state, left) == SessionManager._relative_recorded_path(
                state, right
            )
        except ValueError:
            return False

    @staticmethod
    def _hint_matches(hint: str, path: str) -> bool:
        normalized_hint = str(hint).strip().replace("\\", "/").rstrip("/")
        normalized_path = str(path).strip().replace("\\", "/").rstrip("/")
        if not normalized_hint or not normalized_path:
            return False
        basename = normalized_path.rsplit("/", maxsplit=1)[-1]
        return (
            basename.lower().endswith(normalized_hint.lower())
            if normalized_hint.startswith(".")
            else (basename == normalized_hint)
        )
