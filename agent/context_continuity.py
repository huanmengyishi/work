from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Sequence
from uuid import uuid4

from .memory_refinement import redact_sensitive_text

if TYPE_CHECKING:
    from .state import AgentState


PROJECT_CONTEXT_MARKER = "[Deep Agent restored project context]"
PLAN_CONTEXT_MARKER = "[Deep Agent restored task plan]"
ARTIFACT_CONTEXT_MARKER = "[Deep Agent restored recent artifacts]"
SESSION_NOTES_MARKER = "[Deep Agent restored Session Notes]"
_RESTORED_CONTEXT_MARKERS = (
    PROJECT_CONTEXT_MARKER,
    PLAN_CONTEXT_MARKER,
    ARTIFACT_CONTEXT_MARKER,
    SESSION_NOTES_MARKER,
)
_NOTES_METADATA_PREFIX = "<!-- deep-agent-session-notes:"
_NOTES_METADATA_SUFFIX = " -->"


class IntentJournalReader(Protocol):
    def read(self, session_id: str, *, limit: int = 100) -> list[dict[str, Any]]: ...


class SessionMemoryBuilder:
    """Atomically project one Session note from validated Intent Journal entries.

    The Builder deliberately accepts only a Journal reader and a destination.
    It never opens or parses the Journal file itself, preserving IntentJournal
    as the single integrity-checking source of truth.
    """

    SCHEMA_VERSION = 1
    MAX_NOTES_BYTES = 64 * 1024
    MAX_JOURNAL_ENTRIES = 100
    MAX_PLAN_ITEMS = 128
    MAX_ARTIFACT_ITEMS = 32
    MAX_TRANSITIONS = 16
    MAX_ERRORS = 8

    def __init__(self, journal: IntentJournalReader, output_path: Path) -> None:
        if not callable(getattr(journal, "read", None)):
            raise TypeError("SessionMemoryBuilder requires an IntentJournal reader")
        self.journal = journal
        self.output_path = Path(output_path)
        self._loaded_cache: tuple[int, int, int, int, str] | None = None

    def refresh(self, session_id: str) -> Path | None:
        """Refresh the projection from ``IntentJournal.read()`` only."""

        entries = self.journal.read(session_id, limit=self.MAX_JOURNAL_ENTRIES)
        checkpoints = [item for item in entries if isinstance(item, dict) and item.get("event") != "projection"]
        if not checkpoints:
            return None
        content = self._render(session_id, checkpoints)
        encoded = content.encode("utf-8")
        if len(encoded) > self.MAX_NOTES_BYTES:
            raise ValueError("Session Notes projection exceeds its bounded size")
        self._atomic_replace(encoded)
        self._loaded_cache = None
        return self.output_path

    def load(self, session_id: str) -> str:
        """Load only a bounded projection belonging to ``session_id``."""

        content = self._read_projection()
        if not content:
            return ""
        first_line, _separator, _remainder = content.partition("\n")
        metadata = self._metadata(first_line)
        if metadata.get("schema_version") != self.SCHEMA_VERSION:
            return ""
        if metadata.get("session_id") != session_id:
            return ""
        return content

    def _render(self, session_id: str, entries: list[dict[str, Any]]) -> str:
        latest = entries[-1]
        objective = next(
            (self._inline(item.get("objective"), 2_000) for item in reversed(entries) if item.get("objective")),
            "No objective was journaled.",
        )
        plan = latest.get("plan") if isinstance(latest.get("plan"), list) else []
        plan = [item for item in plan[: self.MAX_PLAN_ITEMS] if isinstance(item, dict)]
        completed = [item for item in plan if item.get("status") in {"completed", "skipped"}]
        remaining = [item for item in plan if item.get("status") in {"pending", "in_progress", "failed"}]
        decisions = self._decisions(entries)
        errors = self._errors(entries)
        artifacts = self._artifacts(entries)
        sequence = self._bounded_non_negative_int(latest.get("sequence"))
        metadata = json.dumps(
            {
                "schema_version": self.SCHEMA_VERSION,
                "session_id": session_id,
                "sequence": sequence,
                "entry_hash": self._inline(latest.get("entry_hash"), 64),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        projection = latest.get("plan_projection") if isinstance(latest.get("plan_projection"), dict) else {}
        omitted_plan = self._bounded_non_negative_int(projection.get("omitted"))
        lines = [
            f"{_NOTES_METADATA_PREFIX}{metadata}{_NOTES_METADATA_SUFFIX}",
            "# Session Notes",
            "",
            f"- Session: `{session_id}`",
            f"- Journal sequence: `{sequence}`",
            f"- Turn: `{self._bounded_non_negative_int(latest.get('turn'))}`",
            f"- Round: `{self._bounded_non_negative_int(latest.get('round'))}`",
            f"- Status: `{self._inline(latest.get('status') or 'unknown', 32)}`",
            "",
            "## Objective",
            "",
            objective,
            "",
            "## Completed Steps",
            "",
            *(self._plan_lines(completed) or ["- None journaled."]),
            "",
            "## Journaled Decisions and State Transitions",
            "",
            *(decisions or ["- None journaled."]),
            "",
            "## Recent Errors",
            "",
            *(errors or ["- None journaled."]),
            "",
            "## Recent Artifacts",
            "",
            *(artifacts or ["- None journaled."]),
            "",
            "## Next Steps",
            "",
            *(self._plan_lines(remaining) or ["- No remaining step was journaled."]),
        ]
        if omitted_plan:
            lines.extend(["", f"- Journal projection omitted `{omitted_plan}` additional plan item(s)."])
        return "\n".join(lines).rstrip() + "\n"

    @classmethod
    def _plan_lines(cls, items: Sequence[dict[str, Any]]) -> list[str]:
        return [
            f"- `{cls._inline(item.get('id') or 'unknown', 160)}` "
            f"status=`{cls._inline(item.get('status') or 'unknown', 32)}` "
            f"type=`{cls._inline(item.get('step_type') or 'generic', 32)}`"
            for item in items
        ]

    @classmethod
    def _decisions(cls, entries: Sequence[dict[str, Any]]) -> list[str]:
        decisions: list[str] = []
        seen: set[str] = set()
        for item in reversed(entries):
            decision = item.get("execution_decision")
            if isinstance(decision, dict):
                rendered = ", ".join(
                    f"{key}={cls._inline(value, 160)}"
                    for key, value in sorted(decision.items())
                    if cls._inline(value, 160)
                )
            else:
                rendered = ""
            impacts = item.get("external_impacts")
            workflow_transitions: list[str] = []
            if isinstance(impacts, list):
                for impact in impacts[: cls.MAX_ARTIFACT_ITEMS]:
                    if not isinstance(impact, dict) or not impact.get("workflow_id"):
                        continue
                    event = cls._inline(impact.get("workflow_event"), 64)
                    if not event:
                        continue
                    path = cls._inline(impact.get("path"), 256)
                    completed = cls._bounded_non_negative_int(impact.get("completed_chapters"))
                    total = cls._bounded_non_negative_int(impact.get("total_chapters"))
                    workflow_transitions.append(f"document_event={event}, path={path}, chapters={completed}/{total}")
                    if len(workflow_transitions) >= 4:
                        break
            if workflow_transitions:
                workflow_text = "; ".join(workflow_transitions)
                rendered = ", ".join(part for part in (rendered, workflow_text) if part)
            if not rendered:
                current_step = cls._inline(item.get("current_step"), 160)
                status = cls._inline(item.get("status"), 32)
                event = cls._inline(item.get("event"), 64)
                rendered = ", ".join(
                    value
                    for value in (
                        f"event={event}" if event else "",
                        f"status={status}" if status else "",
                        f"current_step={current_step}" if current_step else "",
                    )
                    if value
                )
            if not rendered or rendered in seen:
                continue
            seen.add(rendered)
            decisions.append(f"- sequence `{cls._bounded_non_negative_int(item.get('sequence'))}`: {rendered}")
            if len(decisions) >= cls.MAX_TRANSITIONS:
                break
        decisions.reverse()
        return decisions

    @classmethod
    def _errors(cls, entries: Sequence[dict[str, Any]]) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()
        for item in reversed(entries):
            for key in ("error", "recent_error"):
                value = cls._inline(item.get(key), 1_000)
                if not value or value in seen:
                    continue
                seen.add(value)
                values.append(f"- sequence `{cls._bounded_non_negative_int(item.get('sequence'))}`: {value}")
                if len(values) >= cls.MAX_ERRORS:
                    return list(reversed(values))
        return list(reversed(values))

    @classmethod
    def _artifacts(cls, entries: Sequence[dict[str, Any]]) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()
        for item in reversed(entries):
            impacts = item.get("external_impacts")
            if not isinstance(impacts, list):
                continue
            for impact in reversed(impacts):
                if not isinstance(impact, dict):
                    continue
                path = cls._inline(impact.get("path"), 512)
                if not path or path in seen:
                    continue
                seen.add(path)
                values.append(
                    f"- `{path}` state=`{cls._inline(impact.get('state') or 'unknown', 32)}` "
                    f"kind=`{cls._inline(impact.get('kind') or 'file', 32)}`"
                    + (
                        " workflow="
                        f"`{cls._inline(impact.get('workflow_status') or 'unknown', 64)}` "
                        f"chapters=`{cls._bounded_non_negative_int(impact.get('completed_chapters'))}/"
                        f"{cls._bounded_non_negative_int(impact.get('total_chapters'))}`"
                        if impact.get("workflow_id")
                        else ""
                    )
                )
                if len(values) >= cls.MAX_ARTIFACT_ITEMS:
                    return list(reversed(values))
        return list(reversed(values))

    def _atomic_replace(self, content: bytes) -> None:
        parent = self.output_path.parent
        metadata = parent.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise OSError("Session Notes parent must be a real directory")
        if self.output_path.exists() or self.output_path.is_symlink():
            target_metadata = self.output_path.lstat()
            if not stat.S_ISREG(target_metadata.st_mode):
                raise OSError("Session Notes path must be a regular file")
        temporary = self.output_path.with_name(f".{self.output_path.name}.{uuid4().hex}.tmp")
        descriptor: int | None = None
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(temporary, flags, 0o600)
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                descriptor = None
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            if self.output_path.exists() or self.output_path.is_symlink():
                target_metadata = self.output_path.lstat()
                if not stat.S_ISREG(target_metadata.st_mode):
                    raise OSError("Session Notes path must be a regular file")
            os.replace(temporary, self.output_path)
            os.chmod(self.output_path, 0o600)
            directory_descriptor = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def _read_projection(self) -> str:
        path = self.output_path
        try:
            before = path.lstat()
        except FileNotFoundError:
            self._loaded_cache = None
            return ""
        if not stat.S_ISREG(before.st_mode) or before.st_size > self.MAX_NOTES_BYTES:
            return ""
        cache_key = (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_size)
        if self._loaded_cache is not None and self._loaded_cache[:4] == cache_key:
            return self._loaded_cache[4]
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            opened = os.fstat(handle.fileno())
            opened_key = (opened.st_dev, opened.st_ino, opened.st_mtime_ns, opened.st_size)
            if not stat.S_ISREG(opened.st_mode) or opened_key != cache_key:
                return ""
            raw = handle.read(self.MAX_NOTES_BYTES + 1)
        if len(raw) > self.MAX_NOTES_BYTES:
            return ""
        content = raw.decode("utf-8", errors="replace")
        self._loaded_cache = (*cache_key, content)
        return content

    @staticmethod
    def _metadata(first_line: str) -> dict[str, Any]:
        if not first_line.startswith(_NOTES_METADATA_PREFIX) or not first_line.endswith(_NOTES_METADATA_SUFFIX):
            return {}
        raw = first_line[len(_NOTES_METADATA_PREFIX) : -len(_NOTES_METADATA_SUFFIX)]
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _inline(value: object, maximum: int) -> str:
        text = redact_sensitive_text(str(value or ""), maximum=maximum)
        return " ".join(text.split())[:maximum]

    @staticmethod
    def _bounded_non_negative_int(value: object) -> int:
        return max(0, min(value, 10_000_000)) if isinstance(value, int) and not isinstance(value, bool) else 0


class StaticContextRestorer:
    """Rebuild bounded, deduplicated static context after any history rewrite."""

    MAX_PLAN_CHARS = 16_000
    MAX_ARTIFACT_CHARS = 12_000
    MAX_SESSION_NOTES_CHARS = 16_000

    def __init__(
        self,
        context_path: Path,
        *,
        max_context_chars: int = 8_000,
        recent_artifact_limit: int = 20,
    ) -> None:
        self.context_path = Path(context_path)
        self.max_context_chars = max(256, min(int(max_context_chars), 64_000))
        self.recent_artifact_limit = max(1, min(int(recent_artifact_limit), 100))
        self._context_cache: tuple[int, int, int, int, str] | None = None

    def restore(
        self,
        state: AgentState,
        messages: list[dict[str, Any]],
        *,
        session_notes: str = "",
    ) -> bool:
        retained = [item for item in messages if not self._is_restored_message(item)]
        restored = self._messages(state, session_notes=session_notes)
        insertion_index = 1 if retained and retained[0].get("role") == "system" else 0
        candidate = [*retained[:insertion_index], *restored, *retained[insertion_index:]]
        changed = candidate != messages
        if changed:
            messages[:] = candidate
        return changed

    def project_context(self) -> str:
        """Read the real ``.project-agent/context.md`` through an mtime cache."""

        path = self.context_path
        try:
            before = path.lstat()
        except FileNotFoundError:
            self._context_cache = None
            return ""
        if not stat.S_ISREG(before.st_mode):
            self._context_cache = None
            return ""
        cache_key = (before.st_dev, before.st_ino, before.st_mtime_ns, before.st_size)
        if self._context_cache is not None and self._context_cache[:4] == cache_key:
            return self._context_cache[4]
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
            with os.fdopen(descriptor, "rb", closefd=True) as handle:
                opened = os.fstat(handle.fileno())
                opened_key = (opened.st_dev, opened.st_ino, opened.st_mtime_ns, opened.st_size)
                if not stat.S_ISREG(opened.st_mode) or opened_key != cache_key:
                    return ""
                raw = handle.read(self.max_context_chars + 1)
        except OSError:
            return ""
        content = raw[: self.max_context_chars].decode("utf-8", errors="replace").strip()
        if len(raw) > self.max_context_chars:
            content = content.rstrip() + "\n...[truncated]"
        self._context_cache = (*cache_key, content)
        return content

    def _messages(self, state: AgentState, *, session_notes: str) -> list[dict[str, Any]]:
        project_context = self.project_context()
        result: list[dict[str, Any]] = []
        if project_context:
            result.append(
                {
                    "role": "system",
                    "content": f"{PROJECT_CONTEXT_MARKER}\n{project_context}",
                }
            )
        result.extend(
            [
                {
                    "role": "system",
                    "content": f"{PLAN_CONTEXT_MARKER}\n{self._plan_projection(state)}",
                },
                {
                    "role": "system",
                    "content": f"{ARTIFACT_CONTEXT_MARKER}\n{self._artifact_projection(state)}",
                },
            ]
        )
        bounded_notes = self._head_tail(session_notes.strip(), self.MAX_SESSION_NOTES_CHARS)
        if bounded_notes:
            result.append(
                {
                    "role": "system",
                    "content": f"{SESSION_NOTES_MARKER}\n{bounded_notes}",
                }
            )
        return result

    @classmethod
    def _plan_projection(cls, state: AgentState) -> str:
        lines = ["Current Task Graph (authoritative AgentState projection):"]
        for step in state.plan[:128]:
            value = {
                "id": cls._inline(step.id, 160),
                "title": cls._inline(step.title, 500),
                "status": cls._inline(step.status, 32),
                "description": cls._inline(step.description, 1_000),
                "artifact_ids": [cls._inline(item, 500) for item in step.artifact_ids[:32]],
                "step_type": cls._inline(step.step_type, 32),
            }
            line = "- " + json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if len("\n".join([*lines, line])) > cls.MAX_PLAN_CHARS:
                lines.append(f"- ...[{len(state.plan) - len(lines) + 1} additional plan item(s) omitted]")
                break
            lines.append(line)
        if len(lines) == 1:
            lines.append("- No task graph has been published.")
        return "\n".join(lines)

    def _artifact_projection(self, state: AgentState) -> str:
        registry = state.artifact_registry if isinstance(state.artifact_registry, dict) else {}
        artifacts = registry.get("artifacts") if isinstance(registry.get("artifacts"), dict) else {}
        candidates = [(str(path), value) for path, value in artifacts.items() if isinstance(value, dict)]
        candidates.sort(
            key=lambda item: (
                self._bounded_int(item[1].get("turn")),
                self._bounded_int(item[1].get("round")),
                str(item[1].get("updated_at") or ""),
                item[0],
            )
        )
        selected = candidates[-self.recent_artifact_limit :]
        lines = [f"Most recent {self.recent_artifact_limit} ArtifactRegistry item(s):"]
        for path, value in selected:
            artifact = {
                "path": self._inline(value.get("path") or path, 512),
                "kind": self._inline(value.get("kind") or "file", 32),
                "state": self._inline(value.get("state") or "unknown", 32),
                "last_capability": self._inline(value.get("last_capability"), 160),
                "step_ids": [self._inline(item, 160) for item in list(value.get("step_ids") or [])[:32]],
            }
            line = "- " + json.dumps(artifact, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if len("\n".join([*lines, line])) > self.MAX_ARTIFACT_CHARS:
                lines.append("- ...[additional artifact item(s) omitted]")
                break
            lines.append(line)
        if len(lines) == 1:
            lines.append("- No artifact has been registered.")
        return "\n".join(lines)

    @staticmethod
    def _is_restored_message(item: dict[str, Any]) -> bool:
        if item.get("role") != "system":
            return False
        content = item.get("content")
        return isinstance(content, str) and content.startswith(_RESTORED_CONTEXT_MARKERS)

    @staticmethod
    def _inline(value: object, maximum: int) -> str:
        text = redact_sensitive_text(str(value or ""), maximum=maximum)
        return " ".join(text.split())[:maximum]

    @staticmethod
    def _bounded_int(value: object) -> int:
        return max(0, min(value, 10_000_000)) if isinstance(value, int) and not isinstance(value, bool) else 0

    @staticmethod
    def _head_tail(value: str, maximum: int) -> str:
        if len(value) <= maximum:
            return value
        marker = "\n...[middle omitted]...\n"
        available = max(0, maximum - len(marker))
        head = available // 2
        return value[:head] + marker + value[-(available - head) :]


__all__ = [
    "ARTIFACT_CONTEXT_MARKER",
    "PLAN_CONTEXT_MARKER",
    "PROJECT_CONTEXT_MARKER",
    "SESSION_NOTES_MARKER",
    "SessionMemoryBuilder",
    "StaticContextRestorer",
]
