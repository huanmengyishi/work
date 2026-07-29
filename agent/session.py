from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO
from uuid import uuid4

from .artifact import (
    MANAGED_DOCUMENT_ARTIFACT_ID,
    MAX_ARTIFACT_BYTES_HARD_LIMIT,
    ArtifactSpec,
    ArtifactVerifier,
)
from .constants import RESUME_WINDOW_ROUNDS
from .exceptions import SessionInconsistencyError
from .file_lock import FileLockUnavailable, lock_exclusive, unlock
from .intent_journal import IntentJournal
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
    MAX_RESUME_PREFIX_MESSAGES = 32
    MAX_RESUME_ROUND_MESSAGES = 256
    MAX_COLD_MESSAGE_GENERATIONS = 4
    MAX_COLD_MESSAGE_BYTES = 128 * 1024 * 1024
    PAYLOAD_SCHEMA_VERSION = 2
    RESUME_PROJECTION_PREFIX = "[Deep Agent resume window: "

    def __init__(self, project: Project) -> None:
        self.project = project
        self.session_dir = project.agent_dir / "sessions"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.intent_journal = IntentJournal(project.agent_dir / "intent-journal")

    @staticmethod
    def new_session_id() -> str:
        stamp = utc_now_iso().replace("+00:00", "Z").replace("-", "").replace(":", "")
        return f"{stamp}-{uuid4().hex[:8]}"

    def checkpoint(self, state: AgentState, messages: list[dict[str, Any]]) -> Path:
        if len(messages) > self.MAX_MESSAGES or not all(isinstance(item, dict) for item in messages):
            raise ValueError(f"session messages exceed the {self.MAX_MESSAGES} item limit or contain invalid records")
        state.touch()
        state.intent_journal_head = self.intent_journal.append(state, event="checkpoint")
        path = self._json_path(state.session_id)
        previous_generations: list[dict[str, Any]] = []
        transient_paths: set[Path] = set()
        committed = False
        try:
            if path.exists() or path.is_symlink():
                previous_generations, transient_paths = self._existing_message_generations(path)
            generation = uuid4().hex
            message_name = f"{state.session_id}.{generation}.messages.jsonl"
            message_path = self.session_dir / message_name
            message_content = self._encode_messages(messages)
            message_metadata = self._message_metadata(
                message_name=message_name,
                message_content=message_content,
                message_count=len(messages),
                turn=state.turn,
                complete=not self._contains_resume_projection(messages),
                recorded_at=state.updated_at,
            )
            cold_messages = self._retain_cold_generations(previous_generations)
            payload = {
                "schema_version": self.PAYLOAD_SCHEMA_VERSION,
                "state": state.to_dict(),
                "messages": message_metadata,
                "cold_messages": cold_messages,
            }
            content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
            if len(content.encode("utf-8")) + len(message_content) > self.MAX_SESSION_FILE_BYTES:
                raise ValueError(f"session checkpoint exceeds the {self.MAX_SESSION_FILE_BYTES} byte limit")
            self._atomic_write_bytes(message_path, message_content)
            transient_paths.add(message_path)
            self._atomic_write(path, content)
            committed = True
        finally:
            if not committed:
                self._remove_message_paths(transient_paths)
        retained_paths = {self._message_store_path(path, item)[0] for item in cold_messages}
        obsolete_paths = {
            self._message_store_path(path, item)[0]
            for item in previous_generations
            if self._message_store_path(path, item)[0] not in retained_paths
        }
        self._remove_message_paths(obsolete_paths)
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
        state_data = self._state_data(payload, path)
        messages = self._load_messages(payload, path)
        return SessionRecord(state=AgentState.from_dict(state_data), messages=messages)

    def load_for_resume(
        self,
        session_id: str | None = None,
        *,
        max_rounds: int = RESUME_WINDOW_ROUNDS,
    ) -> SessionRecord:
        """Load only the hot model-conversation window needed by Resume.

        The complete checkpoint remains on disk.  Older message bodies are not
        retained in the returned object; one metadata-only marker records the
        projection without copying Prompt, tool output, or credentials.
        """

        resolved = self.resolve_session_id(session_id)
        path = self._json_path(resolved)
        payload = self._read_payload(path)
        state_data = self._state_data(payload, path)
        bounded_rounds = max(1, min(int(max_rounds), 128))
        schema_version = payload.get("schema_version", 1)
        message_store = payload.get("messages")
        if schema_version == self.PAYLOAD_SCHEMA_VERSION and isinstance(message_store, dict):
            self._validate_active_and_cold_paths(payload, path, message_store)
            projected = self._stream_resume_messages(path, message_store, max_rounds=bounded_rounds)
        elif schema_version == 1 and isinstance(message_store, list):
            self._validate_legacy_messages(message_store, path)
            projected = self._project_resume_messages(message_store, max_rounds=bounded_rounds)
        else:
            raise SessionInconsistencyError(f"invalid session file: {path}")
        return SessionRecord(state=AgentState.from_dict(state_data), messages=projected)

    @staticmethod
    def _project_resume_messages(
        messages: list[dict[str, Any]],
        *,
        max_rounds: int,
    ) -> list[dict[str, Any]]:
        assistant_indices = [index for index, item in enumerate(messages) if item.get("role") == "assistant"]
        if len(assistant_indices) <= max_rounds:
            return [dict(item) for item in messages]
        cutoff = assistant_indices[-max_rounds]
        prefix_indices: list[int] = []
        first_system = next(
            (index for index, item in enumerate(messages[:cutoff]) if item.get("role") == "system"), None
        )
        first_user = next((index for index, item in enumerate(messages[:cutoff]) if item.get("role") == "user"), None)
        for index in (first_system, first_user):
            if index is not None and index not in prefix_indices:
                prefix_indices.append(index)
        projected = [dict(messages[index]) for index in sorted(prefix_indices)]
        removed_count = cutoff - len(prefix_indices)
        projected.append(
            {
                "role": "system",
                "content": (
                    SessionManager.RESUME_PROJECTION_PREFIX
                    + f"{removed_count} older message(s) / {len(assistant_indices) - max_rounds} model round(s) "
                    "remain in the durable Session checkpoint and were omitted from hot context.]"
                ),
            }
        )
        projected.extend(dict(item) for item in messages[cutoff:])
        return projected

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
            raise SessionInconsistencyError(f"invalid session file: {path}")
        return payload

    def _state_data(self, payload: dict[str, Any], path: Path) -> dict[str, Any]:
        state_data = payload.get("state")
        if not isinstance(state_data, dict):
            raise SessionInconsistencyError(f"invalid session file: {path}")
        return state_data

    def _load_messages(self, payload: dict[str, Any], path: Path) -> list[dict[str, Any]]:
        schema_version = payload.get("schema_version", 1)
        message_store = payload.get("messages")
        if schema_version == 1 and isinstance(message_store, list):
            self._validate_legacy_messages(message_store, path)
            return message_store
        if schema_version != self.PAYLOAD_SCHEMA_VERSION or not isinstance(message_store, dict):
            raise SessionInconsistencyError(f"invalid session file: {path}")
        self._validate_active_and_cold_paths(payload, path, message_store)
        return self._read_message_store(path, message_store)

    def load_cold_messages(self, session_id: str | None = None) -> list[list[dict[str, Any]]]:
        """Read hash-verified cold message generations, oldest first."""

        resolved = self.resolve_session_id(session_id)
        path = self._json_path(resolved)
        payload = self._read_payload(path)
        if payload.get("schema_version", 1) == 1:
            return []
        if payload.get("schema_version") != self.PAYLOAD_SCHEMA_VERSION:
            raise SessionInconsistencyError(f"invalid session file: {path}")
        return [self._read_message_store(path, item) for item in self._cold_generation_metadata(payload, path)]

    def _validate_active_and_cold_paths(
        self,
        payload: dict[str, Any],
        session_path: Path,
        active: dict[str, Any],
    ) -> None:
        active_path, _ = self._message_store_path(session_path, active)
        cold_paths = {
            self._message_store_path(session_path, item)[0]
            for item in self._cold_generation_metadata(payload, session_path)
        }
        if active_path in cold_paths:
            raise SessionInconsistencyError(
                f"active Session messages are duplicated in cold generations: {session_path}"
            )

    def _read_message_store(
        self,
        path: Path,
        message_store: dict[str, Any],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        with self._open_message_store(path, message_store) as handle:
            digest = hashlib.sha256()
            for raw_line in handle:
                digest.update(raw_line)
                item = self._decode_message_line(raw_line, path)
                messages.append(item)
                if len(messages) > self.MAX_MESSAGES:
                    raise SessionInconsistencyError(f"invalid session file: {path}")
        self._verify_message_store(path, message_store, len(messages), digest.hexdigest())
        return messages

    def _stream_resume_messages(
        self,
        session_path: Path,
        message_store: dict[str, Any],
        *,
        max_rounds: int,
    ) -> list[dict[str, Any]]:
        prefix: list[dict[str, Any]] = []
        first_system: dict[str, Any] | None = None
        first_user: dict[str, Any] | None = None
        rounds: deque[list[dict[str, Any]]] = deque(maxlen=max_rounds)
        current_round: list[dict[str, Any]] | None = None
        assistant_count = 0
        message_count = 0
        digest = hashlib.sha256()
        with self._open_message_store(session_path, message_store) as handle:
            for raw_line in handle:
                digest.update(raw_line)
                item = self._decode_message_line(raw_line, session_path)
                message_count += 1
                if message_count > self.MAX_MESSAGES:
                    raise SessionInconsistencyError(f"invalid session file: {session_path}")
                role = item.get("role")
                if role == "assistant":
                    assistant_count += 1
                    current_round = [item]
                    rounds.append(current_round)
                    continue
                if current_round is not None:
                    if len(current_round) >= self.MAX_RESUME_ROUND_MESSAGES:
                        raise SessionInconsistencyError(
                            f"session model round exceeds {self.MAX_RESUME_ROUND_MESSAGES} messages: {session_path}"
                        )
                    current_round.append(item)
                    continue
                if len(prefix) < self.MAX_RESUME_PREFIX_MESSAGES:
                    prefix.append(item)
                if first_system is None and role == "system":
                    first_system = item
                if first_user is None and role == "user":
                    first_user = item
        self._verify_message_store(session_path, message_store, message_count, digest.hexdigest())

        retained_rounds = [dict(item) for round_items in rounds for item in round_items]
        if assistant_count <= max_rounds and message_count <= len(prefix) + len(retained_rounds):
            return [dict(item) for item in prefix] + retained_rounds
        preserved_prefix: list[dict[str, Any]] = []
        for item in (first_system, first_user):
            if item is not None and item not in preserved_prefix:
                preserved_prefix.append(dict(item))
        removed_rounds = max(0, assistant_count - max_rounds)
        removed_messages = max(0, message_count - len(preserved_prefix) - len(retained_rounds))
        return [
            *preserved_prefix,
            {
                "role": "system",
                "content": (
                    self.RESUME_PROJECTION_PREFIX
                    + f"{removed_messages} older message(s) / {removed_rounds} model round(s) "
                    "remain in the durable Session message store and were omitted from hot context.]"
                ),
            },
            *retained_rounds,
        ]

    def _open_message_store(self, session_path: Path, metadata: dict[str, Any]) -> BinaryIO:
        message_path, expected_bytes = self._message_store_path(session_path, metadata)
        try:
            file_metadata = message_path.lstat()
        except OSError as exc:
            raise SessionInconsistencyError(f"session message path is unavailable: {message_path}") from exc
        if not stat.S_ISREG(file_metadata.st_mode):
            raise SessionInconsistencyError(f"session message path is not a regular file: {message_path}")
        if file_metadata.st_size != expected_bytes:
            raise SessionInconsistencyError(f"session message byte count does not match its manifest: {message_path}")
        if session_path.lstat().st_size + file_metadata.st_size > self.MAX_SESSION_FILE_BYTES:
            raise ValueError(f"session file exceeds the {self.MAX_SESSION_FILE_BYTES} byte limit: {session_path}")
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(message_path, flags)
        except OSError as exc:
            raise SessionInconsistencyError(f"session message path cannot be opened safely: {message_path}") from exc
        return os.fdopen(descriptor, "rb")

    def _message_store_path(self, session_path: Path, metadata: dict[str, Any]) -> tuple[Path, int]:
        if metadata.get("format") != "jsonl":
            raise SessionInconsistencyError(f"invalid session message format: {session_path}")
        name = metadata.get("path")
        count = metadata.get("count")
        byte_count = metadata.get("bytes")
        digest = metadata.get("sha256")
        turn = metadata.get("turn", 1)
        complete = metadata.get("complete", True)
        recorded_at = metadata.get("recorded_at", "legacy")
        pattern = re.compile(rf"^{re.escape(session_path.stem)}\.[0-9a-f]{{32}}\.messages\.jsonl$")
        if (
            not set(metadata) <= {"format", "path", "count", "bytes", "sha256", "turn", "complete", "recorded_at"}
            or not isinstance(name, str)
            or not pattern.fullmatch(name)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or not 0 <= count <= self.MAX_MESSAGES
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or not 0 <= byte_count <= self.MAX_SESSION_FILE_BYTES
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or isinstance(turn, bool)
            or not isinstance(turn, int)
            or turn < 1
            or not isinstance(complete, bool)
            or not isinstance(recorded_at, str)
            or not recorded_at
            or len(recorded_at.encode("utf-8")) > 128
        ):
            raise SessionInconsistencyError(f"invalid session message manifest: {session_path}")
        return self.session_dir / name, byte_count

    @staticmethod
    def _decode_message_line(raw_line: bytes, session_path: Path) -> dict[str, Any]:
        try:
            item = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SessionInconsistencyError(f"invalid session message record: {session_path}") from exc
        if not isinstance(item, dict):
            raise SessionInconsistencyError(f"invalid session message record: {session_path}")
        return item

    def _verify_message_store(
        self,
        session_path: Path,
        metadata: dict[str, Any],
        actual_count: int,
        actual_sha256: str,
    ) -> None:
        if actual_count != metadata.get("count") or actual_sha256 != metadata.get("sha256"):
            raise SessionInconsistencyError(f"session message store does not match its manifest: {session_path}")

    def _validate_legacy_messages(self, messages: list[Any], path: Path) -> None:
        if len(messages) > self.MAX_MESSAGES or not all(isinstance(item, dict) for item in messages):
            raise SessionInconsistencyError(f"invalid session file: {path}")

    def _existing_message_generations(
        self,
        session_path: Path,
    ) -> tuple[list[dict[str, Any]], set[Path]]:
        payload = self._read_payload(session_path)
        state_data = self._state_data(payload, session_path)
        existing_state = AgentState.from_dict(state_data)
        if existing_state.session_id != session_path.stem:
            raise SessionInconsistencyError(f"session identity does not match its path: {session_path}")
        schema_version = payload.get("schema_version", 1)
        message_store = payload.get("messages")
        if schema_version == 1 and isinstance(message_store, list):
            self._validate_legacy_messages(message_store, session_path)
            content = self._encode_messages(message_store)
            generation = uuid4().hex
            name = f"{session_path.stem}.{generation}.messages.jsonl"
            path = self.session_dir / name
            metadata = self._message_metadata(
                message_name=name,
                message_content=content,
                message_count=len(message_store),
                turn=existing_state.turn,
                complete=not self._contains_resume_projection(message_store),
                recorded_at=existing_state.updated_at,
            )
            self._atomic_write_bytes(path, content)
            return [metadata], {path}
        if schema_version != self.PAYLOAD_SCHEMA_VERSION or not isinstance(message_store, dict):
            raise SessionInconsistencyError(f"invalid session file: {session_path}")
        cold = self._cold_generation_metadata(payload, session_path)
        active = self._normalize_generation_metadata(message_store, existing_state)
        self._assert_message_store_on_disk(session_path, active)
        paths = [self._message_store_path(session_path, item)[0] for item in [*cold, active]]
        if len(paths) != len(set(paths)):
            raise SessionInconsistencyError(f"session message generations contain duplicate paths: {session_path}")
        return [*cold, active], set()

    def _cold_generation_metadata(
        self,
        payload: dict[str, Any],
        session_path: Path,
    ) -> list[dict[str, Any]]:
        raw = payload.get("cold_messages", [])
        if not isinstance(raw, list) or len(raw) > self.MAX_COLD_MESSAGE_GENERATIONS:
            raise SessionInconsistencyError(f"invalid cold session generations: {session_path}")
        generations: list[dict[str, Any]] = []
        total_bytes = 0
        for item in raw:
            if not isinstance(item, dict):
                raise SessionInconsistencyError(f"invalid cold session generation: {session_path}")
            normalized = dict(item)
            _, byte_count = self._message_store_path(session_path, normalized)
            self._assert_message_store_on_disk(session_path, normalized)
            total_bytes += byte_count
            generations.append(normalized)
        if total_bytes > self.MAX_COLD_MESSAGE_BYTES:
            raise SessionInconsistencyError(f"cold session generations exceed the byte limit: {session_path}")
        paths = [self._message_store_path(session_path, item)[0] for item in generations]
        if len(paths) != len(set(paths)):
            raise SessionInconsistencyError(f"cold session generations contain duplicate paths: {session_path}")
        return generations

    @staticmethod
    def _normalize_generation_metadata(
        metadata: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        normalized = dict(metadata)
        normalized.setdefault("turn", state.turn)
        normalized.setdefault("complete", True)
        normalized.setdefault("recorded_at", state.updated_at)
        return normalized

    def _retain_cold_generations(self, generations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not generations:
            return []
        newest_complete = next((item for item in reversed(generations) if item.get("complete") is True), None)
        mandatory = [generations[-1]]
        if newest_complete is not None and newest_complete is not generations[-1]:
            mandatory.append(newest_complete)
        selected_paths: set[str] = set()
        selected: list[dict[str, Any]] = []
        total_bytes = 0

        def retain(item: dict[str, Any], *, required: bool = False) -> None:
            nonlocal total_bytes
            name = str(item.get("path") or "")
            byte_count = int(item.get("bytes") or 0)
            if name in selected_paths:
                return
            if (
                len(selected) >= self.MAX_COLD_MESSAGE_GENERATIONS
                or total_bytes + byte_count > self.MAX_COLD_MESSAGE_BYTES
            ):
                if required:
                    raise SessionInconsistencyError("required cold Session generations exceed the bounded quota")
                return
            selected_paths.add(name)
            selected.append(item)
            total_bytes += byte_count

        for item in mandatory:
            retain(item, required=True)
        for item in reversed(generations):
            retain(item)
        order = {str(item.get("path") or ""): index for index, item in enumerate(generations)}
        return sorted(selected, key=lambda item: order[str(item.get("path") or "")])

    def _assert_message_store_on_disk(self, session_path: Path, metadata: dict[str, Any]) -> None:
        message_path, expected_bytes = self._message_store_path(session_path, metadata)
        try:
            file_metadata = message_path.lstat()
        except OSError as exc:
            raise SessionInconsistencyError(f"session message path is unavailable: {message_path}") from exc
        if not stat.S_ISREG(file_metadata.st_mode) or file_metadata.st_size != expected_bytes:
            raise SessionInconsistencyError(f"session message path does not match its manifest: {message_path}")

    @staticmethod
    def _message_metadata(
        *,
        message_name: str,
        message_content: bytes,
        message_count: int,
        turn: int,
        complete: bool,
        recorded_at: str,
    ) -> dict[str, Any]:
        return {
            "format": "jsonl",
            "path": message_name,
            "count": message_count,
            "bytes": len(message_content),
            "sha256": hashlib.sha256(message_content).hexdigest(),
            "turn": max(1, int(turn)),
            "complete": bool(complete),
            "recorded_at": str(recorded_at)[:128] or "unknown",
        }

    @staticmethod
    def _encode_messages(messages: list[dict[str, Any]]) -> bytes:
        try:
            return b"".join(
                (json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
                for item in messages
            )
        except (TypeError, ValueError, RecursionError, UnicodeError) as exc:
            raise ValueError("session messages cannot be serialized as JSONL") from exc

    @classmethod
    def _contains_resume_projection(cls, messages: list[dict[str, Any]]) -> bool:
        return any(
            item.get("role") == "system"
            and isinstance(item.get("content"), str)
            and item["content"].startswith(cls.RESUME_PROJECTION_PREFIX)
            for item in messages
        )

    @staticmethod
    def _remove_message_paths(paths: set[Path]) -> None:
        for path in paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

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
        try:
            temp.write_text(content, encoding="utf-8")
            temp.replace(path)
        finally:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _atomic_write_bytes(path: Path, content: bytes) -> None:
        temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temp.write_bytes(content)
            temp.replace(path)
        finally:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass

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
