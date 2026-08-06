from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from .constants import MAX_ARTIFACT_REGISTRY_ITEMS, MAX_INTENT_JOURNAL_ENTRIES
from .exceptions import IntentJournalError
from .file_lock import lock_exclusive, unlock
from .memory_refinement import redact_sensitive_text
from .timeutil import utc_now_iso

if TYPE_CHECKING:
    from .state import AgentState


_HASH_RE = re.compile(r"[0-9a-f]{64}")


class IntentJournal:
    """Append-only, hash-chained recovery metadata for long-running Sessions.

    The active file is bounded.  Once it reaches its record or byte limit, a
    compact projection replaces the validated prefix and commits to its final
    hash.  The next checkpoint then links to that projection, so checkpointing
    continues without discarding the integrity anchor for the older chain.
    """

    SCHEMA_VERSION = 1
    MAX_FILE_BYTES = 8 * 1024 * 1024
    MAX_LINE_BYTES = 32 * 1024
    PLAN_PROJECTION_BYTES = 6 * 1024
    IMPACT_PROJECTION_BYTES = 16 * 1024
    OBJECTIVE_BYTES = 2 * 1024

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if not stat.S_ISDIR(self.directory.lstat().st_mode):
            raise IntentJournalError("intent journal directory must not be a symbolic link")
        try:
            self.directory.chmod(0o700)
        except OSError:
            pass

    def append(self, state: AgentState, *, event: str = "checkpoint") -> dict[str, Any]:
        path = self._path(state.session_id)
        lock_handle = self._open_lock(state.session_id)
        try:
            lock_exclusive(lock_handle)
            entries = self._read_entries(path) if path.exists() or path.is_symlink() else []
            previous = entries[-1] if entries else {}
            sequence = int(previous.get("sequence") or 0) + 1
            entry = self._checkpoint_entry(state, event=event, sequence=sequence, previous=previous)
            encoded = self._render_entry(entry)
            current_size = path.lstat().st_size if path.exists() else 0
            if len(entries) >= MAX_INTENT_JOURNAL_ENTRIES or current_size + len(encoded) > self.MAX_FILE_BYTES:
                if not previous:
                    raise IntentJournalError("intent journal cannot project an empty chain")
                projection = self._chain_projection(previous, compacted_entry_count=len(entries))
                projection_bytes = self._render_entry(projection)
                entry["previous_hash"] = projection["entry_hash"]
                encoded = self._render_entry(entry)
                self._replace_bytes(path, projection_bytes + encoded)
            else:
                self._append_bytes(path, encoded)
            return self._head(path, entry)
        finally:
            unlock(lock_handle)
            lock_handle.close()

    def read(self, session_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        path = self._path(session_id)
        if not path.exists() and not path.is_symlink():
            return []
        bounded = max(1, min(int(limit), MAX_INTENT_JOURNAL_ENTRIES))
        return self._read_entries(path)[-bounded:]

    def _read_entries(self, path: Path) -> list[dict[str, Any]]:
        self._validate_path(path)
        expected_session_id = path.name.removesuffix(".intent.jsonl")
        lines = self._read_bytes(path).splitlines()
        if len(lines) > MAX_INTENT_JOURNAL_ENTRIES:
            raise IntentJournalError("intent journal exceeds its entry-count limit")

        entries: list[dict[str, Any]] = []
        previous_hash = ""
        previous_sequence = 0
        for index, line in enumerate(lines):
            if len(line) > self.MAX_LINE_BYTES:
                raise IntentJournalError("intent journal entry exceeds its bounded line size")
            try:
                value = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise IntentJournalError("intent journal contains invalid JSON") from exc
            if not isinstance(value, dict):
                raise IntentJournalError("intent journal entry must be an object")

            recorded_hash = value.get("entry_hash")
            canonical_value = dict(value)
            canonical_value.pop("entry_hash", None)
            calculated_hash = hashlib.sha256(self._canonical(canonical_value)).hexdigest()
            sequence = value.get("sequence")
            is_projection = value.get("event") == "projection"
            valid_root = (
                index == 0
                and not is_projection
                and isinstance(sequence, int)
                and not isinstance(sequence, bool)
                and sequence == 1
                and value.get("previous_hash") == ""
            )
            valid_projection_root = index == 0 and is_projection and self._valid_chain_projection(value)
            valid_link = (
                index > 0
                and not is_projection
                and isinstance(sequence, int)
                and not isinstance(sequence, bool)
                and sequence == previous_sequence + 1
                and value.get("previous_hash") == previous_hash
            )
            if (
                value.get("schema_version") != self.SCHEMA_VERSION
                or value.get("session_id") != expected_session_id
                or not (valid_root or valid_projection_root or valid_link)
                or not isinstance(recorded_hash, str)
                or not _HASH_RE.fullmatch(recorded_hash)
                or recorded_hash != calculated_hash
            ):
                raise IntentJournalError("intent journal hash chain is invalid")
            previous_hash = recorded_hash
            previous_sequence = int(sequence)
            entries.append(value)
        return entries

    def _checkpoint_entry(
        self,
        state: AgentState,
        *,
        event: str,
        sequence: int,
        previous: dict[str, Any],
    ) -> dict[str, Any]:
        plans = [
            {
                "id": self._bounded_text(redact_sensitive_text(step.id, maximum=80), 160),
                "status": self._bounded_text(step.status, 32),
                "step_type": self._bounded_text(step.step_type, 32),
            }
            for step in state.plan[:128]
        ]
        projected_plan, plan_projection = self._project_records(plans, budget=self.PLAN_PROJECTION_BYTES)
        impacts = self._external_impacts(state)
        projected_impacts, impact_projection = self._project_records(
            impacts,
            budget=self.IMPACT_PROJECTION_BYTES,
        )
        event_name = self._bounded_text(str(event).strip(), 64) or "checkpoint"
        if event_name == "projection":
            event_name = "checkpoint"
        return {
            "schema_version": self.SCHEMA_VERSION,
            "sequence": sequence,
            "event": event_name,
            "session_id": state.session_id,
            "run_id": self._bounded_text(state.run_id, 300),
            "turn": max(1, int(state.turn)),
            "round": max(0, int(state.round)),
            "status": self._bounded_text(state.status, 32),
            "objective": self._bounded_text(
                redact_sensitive_text(state.objective, maximum=2_000),
                self.OBJECTIVE_BYTES,
            ),
            "plan": projected_plan,
            "plan_projection": plan_projection,
            "current_step": self._bounded_text(
                redact_sensitive_text(str(state.current_step or ""), maximum=80),
                160,
            ),
            "execution_decision": self._execution_decision(state),
            "error": self._bounded_text(
                redact_sensitive_text(str(state.error or ""), maximum=1_000),
                1_024,
            ),
            "recent_error": self._bounded_text(
                redact_sensitive_text(
                    str(state.execution_context.recent_error if state.execution_context is not None else ""),
                    maximum=1_000,
                ),
                1_024,
            ),
            "external_impacts": projected_impacts,
            "external_impacts_projection": impact_projection,
            "resume_checkpoint": self._safe_checkpoint(state.resume_checkpoint),
            "previous_hash": str(previous.get("entry_hash") or ""),
            "recorded_at": utc_now_iso(),
        }

    def _chain_projection(self, previous: dict[str, Any], *, compacted_entry_count: int) -> dict[str, Any]:
        previous_hash = str(previous.get("entry_hash") or "")
        sequence = int(previous.get("sequence") or 0)
        if sequence < 1 or not _HASH_RE.fullmatch(previous_hash):
            raise IntentJournalError("intent journal cannot project an invalid chain head")
        return {
            "schema_version": self.SCHEMA_VERSION,
            "sequence": sequence,
            "event": "projection",
            "session_id": str(previous.get("session_id") or ""),
            "run_id": self._bounded_text(str(previous.get("run_id") or ""), 300),
            "turn": max(1, int(previous.get("turn") or 1)),
            "round": max(0, int(previous.get("round") or 0)),
            "status": self._bounded_text(str(previous.get("status") or ""), 32),
            "current_step": self._bounded_text(str(previous.get("current_step") or ""), 160),
            "previous_hash": previous_hash,
            "projection": {
                "compacted_entry_count": max(sequence, int(compacted_entry_count)),
                "compacted_through_sequence": sequence,
                "prior_head_hash": previous_hash,
                "prior_recorded_at": self._bounded_text(str(previous.get("recorded_at") or ""), 64),
            },
            "recorded_at": utc_now_iso(),
        }

    @staticmethod
    def _valid_chain_projection(value: dict[str, Any]) -> bool:
        sequence = value.get("sequence")
        projection = value.get("projection")
        previous_hash = value.get("previous_hash")
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 1
            or not isinstance(projection, dict)
            or not isinstance(previous_hash, str)
            or not _HASH_RE.fullmatch(previous_hash)
        ):
            return False
        compacted_count = projection.get("compacted_entry_count")
        return (
            isinstance(compacted_count, int)
            and not isinstance(compacted_count, bool)
            and compacted_count >= 1
            and projection.get("compacted_through_sequence") == sequence
            and projection.get("prior_head_hash") == previous_hash
        )

    def _render_entry(self, entry: dict[str, Any]) -> bytes:
        """Hash and serialize one line, trimming only projected detail if needed."""

        entry.pop("entry_hash", None)
        while True:
            canonical = self._canonical(entry)
            entry_hash = hashlib.sha256(canonical).hexdigest()
            rendered_entry = {**entry, "entry_hash": entry_hash}
            encoded = self._canonical(rendered_entry) + b"\n"
            if len(encoded) <= self.MAX_LINE_BYTES:
                entry["entry_hash"] = entry_hash
                return encoded
            impacts = entry.get("external_impacts")
            plan = entry.get("plan")
            if isinstance(impacts, list) and impacts:
                impacts.pop()
                self._update_projection_count(entry.get("external_impacts_projection"), len(impacts))
                continue
            if isinstance(plan, list) and plan:
                plan.pop()
                self._update_projection_count(entry.get("plan_projection"), len(plan))
                continue
            objective = str(entry.get("objective") or "")
            if objective:
                entry["objective"] = self._bounded_text(objective, max(0, len(objective.encode("utf-8")) // 2))
                continue
            raise IntentJournalError("intent journal entry exceeds its bounded line size")

    @staticmethod
    def _update_projection_count(value: object, included: int) -> None:
        if not isinstance(value, dict):
            return
        total = max(0, int(value.get("total") or 0))
        value["included"] = included
        value["omitted"] = max(0, total - included)

    @classmethod
    def _project_records(
        cls, values: list[dict[str, Any]], *, budget: int
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        digest = hashlib.sha256(cls._canonical(values)).hexdigest()
        projected: list[dict[str, Any]] = []
        for value in values:
            candidate = [*projected, value]
            if len(cls._canonical(candidate)) > budget:
                break
            projected.append(value)
        return projected, {
            "total": len(values),
            "included": len(projected),
            "omitted": len(values) - len(projected),
            "sha256": digest,
        }

    @staticmethod
    def _canonical(value: object) -> bytes:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @staticmethod
    def _bounded_text(value: object, maximum_bytes: int) -> str:
        text = str(value)
        encoded = text.encode("utf-8")
        if len(encoded) <= maximum_bytes:
            return text
        if maximum_bytes <= 0:
            return ""
        suffix = "...[truncated]"
        suffix_bytes = suffix.encode("utf-8")
        if maximum_bytes <= len(suffix_bytes):
            return encoded[:maximum_bytes].decode("utf-8", errors="ignore")
        prefix = encoded[: maximum_bytes - len(suffix_bytes)].decode("utf-8", errors="ignore")
        return prefix + suffix

    def _open_lock(self, session_id: str):
        lock_path = self.directory / f".{session_id}.intent.lock"
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(lock_path, flags, 0o600)
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                os.close(descriptor)
                raise IntentJournalError("intent journal lock must be a regular file")
            return os.fdopen(descriptor, "a+", encoding="utf-8")
        except OSError as exc:
            raise IntentJournalError(f"could not open intent journal lock: {exc}") from exc

    def _append_bytes(self, path: Path, content: bytes) -> None:
        if path.exists() or path.is_symlink():
            self._validate_path(path)
            if path.stat().st_size + len(content) > self.MAX_FILE_BYTES:
                raise IntentJournalError("intent journal exceeds its file-size limit")
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags, 0o600)
            with os.fdopen(descriptor, "ab", closefd=True) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(path, 0o600)
        except OSError as exc:
            raise IntentJournalError(f"could not append intent journal: {exc}") from exc

    def _replace_bytes(self, path: Path, content: bytes) -> None:
        if len(content) > self.MAX_FILE_BYTES:
            raise IntentJournalError("intent journal projection exceeds its file-size limit")
        if path.exists() or path.is_symlink():
            self._validate_path(path)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        descriptor: int | None = None
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(temporary, flags, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = None
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            if path.exists() or path.is_symlink():
                self._validate_path(path)
            os.replace(temporary, path)
            os.chmod(path, 0o600)
            try:
                directory_descriptor = os.open(self.directory, os.O_RDONLY)
            except OSError:
                directory_descriptor = None
            if directory_descriptor is not None:
                try:
                    os.fsync(directory_descriptor)
                finally:
                    os.close(directory_descriptor)
        except IntentJournalError:
            raise
        except OSError as exc:
            raise IntentJournalError(f"could not replace intent journal projection: {exc}") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def _validate_path(self, path: Path) -> None:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise IntentJournalError(f"could not inspect intent journal: {exc}") from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise IntentJournalError("intent journal path must be a regular file")
        if metadata.st_size > self.MAX_FILE_BYTES:
            raise IntentJournalError("intent journal exceeds its file-size limit")

    def _read_bytes(self, path: Path) -> bytes:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
            with os.fdopen(descriptor, "rb", closefd=True) as handle:
                metadata = os.fstat(handle.fileno())
                if not stat.S_ISREG(metadata.st_mode):
                    raise IntentJournalError("intent journal path must be a regular file")
                if metadata.st_size > self.MAX_FILE_BYTES:
                    raise IntentJournalError("intent journal exceeds its file-size limit")
                content = handle.read(self.MAX_FILE_BYTES + 1)
        except IntentJournalError:
            raise
        except OSError as exc:
            raise IntentJournalError(f"could not read intent journal: {exc}") from exc
        if len(content) > self.MAX_FILE_BYTES:
            raise IntentJournalError("intent journal exceeds its file-size limit")
        return content

    def _path(self, session_id: str) -> Path:
        if (
            not session_id
            or len(session_id) > 200
            or any(
                char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in session_id
            )
        ):
            raise IntentJournalError("invalid intent journal session id")
        return self.directory / f"{session_id}.intent.jsonl"

    def _head(self, path: Path, entry: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "path": f"intent-journal/{path.name}",
            "sequence": max(0, int(entry.get("sequence") or 0)),
            "entry_hash": str(entry.get("entry_hash") or "")[:64],
            "recorded_at": str(entry.get("recorded_at") or "")[:64],
        }

    @classmethod
    def _safe_checkpoint(cls, value: dict[str, Any]) -> dict[str, Any]:
        numeric = {
            "schema_version",
            "turn",
            "round",
            "tool_call_count",
            "pruned_tool_call_count",
            "model_request_count",
            "message_count",
        }
        result: dict[str, Any] = {}
        for key in numeric:
            item = value.get(key)
            if isinstance(item, int) and not isinstance(item, bool):
                result[key] = max(0, min(item, 10_000_000))
        for key, maximum in (("phase", 64), ("recorded_at", 64)):
            item = value.get(key)
            if isinstance(item, str):
                result[key] = cls._bounded_text(item, maximum)
        return result

    @classmethod
    def _external_impacts(cls, state: AgentState) -> list[dict[str, Any]]:
        artifacts = (state.artifact_registry or {}).get("artifacts")
        if not isinstance(artifacts, dict):
            return []
        impacts: list[dict[str, Any]] = []
        for path, value in list(artifacts.items())[:MAX_ARTIFACT_REGISTRY_ITEMS]:
            if not isinstance(value, dict):
                continue
            artifact_state = str(value.get("state") or "")
            workflow_id = str(value.get("workflow_id") or "")
            is_document_workflow_progress = (
                artifact_state in {"planned", "in_progress"} and 1 <= len(workflow_id) <= 200
            )
            if artifact_state not in {"generated", "verified"} and not is_document_workflow_progress:
                continue
            all_step_ids = [
                cls._bounded_text(redact_sensitive_text(str(item), maximum=80), 160)
                for item in list(value.get("step_ids") or [])[:32]
            ]
            all_parent_artifacts = [
                cls._bounded_text(redact_sensitive_text(str(item), maximum=500), 512)
                for item in list(value.get("parent_artifacts") or [])[:32]
            ]
            step_ids, step_projection = cls._project_values(all_step_ids, budget=1_024)
            parent_artifacts, parent_projection = cls._project_values(all_parent_artifacts, budget=2_048)
            impacts.append(
                {
                    "path": cls._bounded_text(redact_sensitive_text(str(path), maximum=500), 512),
                    "kind": cls._bounded_text(str(value.get("kind") or "file"), 32),
                    "state": cls._bounded_text(artifact_state or "generated", 32),
                    "workflow_id": cls._bounded_text(workflow_id, 200),
                    "workflow_status": cls._bounded_text(str(value.get("workflow_status") or ""), 64),
                    "workflow_event": cls._bounded_text(str(value.get("workflow_event") or ""), 64),
                    "completed_chapters": cls._bounded_non_negative_int(value.get("completed_chapters")),
                    "total_chapters": cls._bounded_non_negative_int(value.get("total_chapters")),
                    "chapter_id": cls._bounded_text(str(value.get("chapter_id") or ""), 64),
                    "chapter_sha256": cls._bounded_text(str(value.get("chapter_sha256") or ""), 64),
                    "preview_id": cls._bounded_text(str(value.get("workflow_render_preview_id") or ""), 200),
                    "workflow_finalized": value.get("workflow_finalized") is True,
                    "snapshot_id": cls._bounded_text(
                        redact_sensitive_text(str(value.get("snapshot_id") or ""), maximum=200),
                        256,
                    ),
                    "step_ids": step_ids,
                    "step_ids_projection": step_projection,
                    "parent_artifacts": parent_artifacts,
                    "parent_artifacts_projection": parent_projection,
                }
            )
        return impacts

    @staticmethod
    def _bounded_non_negative_int(value: object) -> int:
        return max(0, min(value, 1_000_000)) if isinstance(value, int) and not isinstance(value, bool) else 0

    @classmethod
    def _execution_decision(cls, state: AgentState) -> dict[str, str]:
        route = state.task_route if isinstance(state.task_route, dict) else {}
        strategy = state.task_strategy if isinstance(state.task_strategy, dict) else {}
        values = {
            "mode": strategy.get("mode"),
            "task_type": route.get("task_type"),
            "scale": route.get("scale"),
            "risk": route.get("risk"),
        }
        return {
            key: cls._bounded_text(redact_sensitive_text(str(value), maximum=80), 160)
            for key, value in values.items()
            if value is not None and str(value).strip()
        }

    @classmethod
    def _project_values(cls, values: list[str], *, budget: int) -> tuple[list[str], dict[str, Any]]:
        digest = hashlib.sha256(cls._canonical(values)).hexdigest()
        projected: list[str] = []
        for value in values:
            candidate = [*projected, value]
            if len(cls._canonical(candidate)) > budget:
                break
            projected.append(value)
        return projected, {
            "total": len(values),
            "included": len(projected),
            "omitted": len(values) - len(projected),
            "sha256": digest,
        }


__all__ = ["IntentJournal"]
