from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..artifact import (
    ARTIFACT_VERIFICATION_METADATA_KEY,
    MANAGED_DOCUMENT_ARTIFACT_ID,
    MAX_ARTIFACT_BYTES_HARD_LIMIT,
    ArtifactSpec,
    ArtifactVerifier,
)
from ..project import Project
from ..timeutil import utc_now_iso
from .base import ToolResult
from .document import DocumentTool
from .file_edit import FileEditTool
from .pathsafe import resolve_project_path


WORKFLOW_ID_RE = re.compile(r"^[0-9a-f]{32}$")
CHAPTER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DATE_LITERAL_RE = re.compile(r"(?<!\d)20\d{2}(?:年\s*\d{1,2}月(?:\s*\d{1,2}日)?|[-/.]\d{1,2}(?:[-/.]\d{1,2})?)(?!\d)")


class DocumentGeneratorTool:
    """Persist a bounded, model-free chapter workflow for large documents.

    The Runtime remains the only component that asks DeepSeek to produce text.
    This managed tool records an approved outline and one chapter at a time,
    then creates a normal snapshot-backed preview through ``FileEditTool``.
    """

    SCHEMA_VERSION = 1
    MAX_CHAPTERS = 64
    MAX_TITLE_CHARS = 500
    MAX_CHAPTER_TITLE_CHARS = 300
    MAX_CHAPTER_CHARS = 100_000
    MAX_CHAPTER_BYTES = 400_000
    MAX_SUMMARY_CHARS = 2_000
    MAX_TOTAL_ESTIMATED_TOKENS = 1_000_000
    MAX_STATE_BYTES = 256 * 1024

    def __init__(
        self,
        project: Project,
        document: DocumentTool,
        file_edit: FileEditTool,
    ) -> None:
        self.project = project
        self.document = document
        self.file_edit = file_edit
        self.root = project.agent_dir / "document-workflows"
        self.root.mkdir(parents=True, exist_ok=True)
        metadata = self.root.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("document workflow root must be a regular directory")
        try:
            self.root.chmod(0o700)
        except OSError:
            pass

    def create_outline(
        self,
        *,
        session_id: str,
        title: str,
        output_path: str,
        chapters: list[dict[str, Any]],
    ) -> ToolResult:
        try:
            normalized_session = self._session_id(session_id)
            normalized_title = self._bounded_required(title, "title", self.MAX_TITLE_CHARS)
            normalized_output = self._output_path(output_path)
            normalized_chapters = self._normalize_chapters(chapters)
        except (TypeError, ValueError) as exc:
            return ToolResult(False, "", str(exc))
        workflow_id = uuid4().hex
        outline = {
            "title": normalized_title,
            "output_path": normalized_output,
            "chapters": [
                {
                    "id": item["id"],
                    "title": item["title"],
                    "estimated_tokens": item["estimated_tokens"],
                }
                for item in normalized_chapters
            ],
        }
        outline_hash = self._digest(outline)
        now = utc_now_iso()
        state = {
            "schema_version": self.SCHEMA_VERSION,
            "workflow_id": workflow_id,
            "session_id": normalized_session,
            "status": "awaiting_confirmation",
            "title": normalized_title,
            "output_path": normalized_output,
            "outline_hash": outline_hash,
            "chapters": normalized_chapters,
            "created_at": now,
            "updated_at": now,
        }
        try:
            self._write_state(state)
        except OSError as exc:
            return ToolResult(False, "", f"could not persist document outline: {type(exc).__name__}")
        total_tokens = sum(int(item["estimated_tokens"]) for item in normalized_chapters)
        return ToolResult(
            True,
            self._render_outline(state),
            data=self._result_data(
                state,
                event="outline_created",
                outline_hash=outline_hash,
                total_estimated_tokens=total_tokens,
                requires_outline_confirmation=True,
            ),
        )

    def confirm_outline(self, *, session_id: str, workflow_id: str, outline_hash: str) -> ToolResult:
        try:
            state = self._load_state(workflow_id, session_id=session_id)
        except (OSError, ValueError) as exc:
            return ToolResult(False, "", str(exc))
        if state["status"] != "awaiting_confirmation":
            return ToolResult(False, "", f"document outline cannot be confirmed from {state['status']} state")
        if not isinstance(outline_hash, str) or outline_hash != state["outline_hash"]:
            return ToolResult(False, "", "document outline hash changed; inspect the current outline before confirming")
        state["status"] = "drafting"
        state["confirmed_at"] = utc_now_iso()
        state["updated_at"] = state["confirmed_at"]
        self._write_state(state)
        return ToolResult(
            True,
            "outline confirmed; request the next chapter checkpoint",
            data=self._result_data(state, event="outline_confirmed"),
        )

    def next_chapter(self, *, session_id: str, workflow_id: str) -> ToolResult:
        try:
            state = self._load_state(workflow_id, session_id=session_id)
        except (OSError, ValueError) as exc:
            return ToolResult(False, "", str(exc))
        if state["status"] == "awaiting_confirmation":
            return ToolResult(False, "", "confirm the exact outline before generating chapter text")
        if state["status"] in {"ready_to_render", "awaiting_apply", "completed"}:
            return ToolResult(False, "", f"document workflow has no pending chapter ({state['status']})")
        chapter = next((item for item in state["chapters"] if item["status"] == "in_progress"), None)
        resumed = chapter is not None
        if chapter is None:
            chapter = next((item for item in state["chapters"] if item["status"] == "pending"), None)
        if chapter is None:
            state["status"] = "ready_to_render"
            state["updated_at"] = utc_now_iso()
            self._write_state(state)
            return ToolResult(True, "all chapters are checkpointed", data=self._result_data(state, event="draft_ready"))
        if not resumed:
            chapter["status"] = "in_progress"
            chapter["started_at"] = utc_now_iso()
            state["current_chapter_id"] = chapter["id"]
            state["updated_at"] = chapter["started_at"]
            self._write_state(state)
        prior_summaries = [
            {"id": item["id"], "title": item["title"], "summary": item.get("summary", "")}
            for item in state["chapters"]
            if item["status"] == "completed"
        ][-16:]
        data = self._result_data(
            state,
            event="chapter_resumed" if resumed else "chapter_started",
            chapter={
                "id": chapter["id"],
                "title": chapter["title"],
                "estimated_tokens": chapter["estimated_tokens"],
            },
            prior_chapter_summaries=prior_summaries,
            independent_context=True,
        )
        return ToolResult(True, json.dumps(data, ensure_ascii=False, indent=2), data=data)

    def save_chapter(
        self,
        *,
        session_id: str,
        workflow_id: str,
        chapter_id: str,
        markdown: str,
        summary: str,
    ) -> ToolResult:
        try:
            state = self._load_state(workflow_id, session_id=session_id)
            chapter = self._chapter(state, chapter_id)
            content = self._bounded_required(markdown, "chapter markdown", self.MAX_CHAPTER_CHARS)
            encoded = content.encode("utf-8")
            if len(encoded) > self.MAX_CHAPTER_BYTES:
                raise ValueError(f"chapter markdown exceeds {self.MAX_CHAPTER_BYTES} UTF-8 bytes")
            normalized_summary = self._bounded_required(summary, "chapter summary", self.MAX_SUMMARY_CHARS)
        except (OSError, TypeError, ValueError) as exc:
            return ToolResult(False, "", str(exc))
        if chapter["status"] != "in_progress" or state.get("current_chapter_id") != chapter["id"]:
            return ToolResult(False, "", "only the current in-progress chapter can be checkpointed")
        chapter_path = self._chapter_path(state["workflow_id"], chapter["id"])
        try:
            self._atomic_write_bytes(chapter_path, encoded)
            chapter.update(
                {
                    "status": "completed",
                    "summary": normalized_summary,
                    "content_bytes": len(encoded),
                    "content_sha256": hashlib.sha256(encoded).hexdigest(),
                    "completed_at": utc_now_iso(),
                }
            )
            state.pop("current_chapter_id", None)
            state["status"] = (
                "ready_to_render" if all(item["status"] == "completed" for item in state["chapters"]) else "drafting"
            )
            state["updated_at"] = chapter["completed_at"]
            self._write_state(state)
        except OSError as exc:
            return ToolResult(False, "", f"could not checkpoint current chapter: {type(exc).__name__}")
        return ToolResult(
            True,
            f"checkpointed chapter {chapter['id']} ({len(encoded)} bytes)",
            data=self._result_data(
                state,
                event="chapter_completed",
                chapter_id=chapter["id"],
                chapter_bytes=len(encoded),
                chapter_sha256=chapter["content_sha256"],
            ),
        )

    def rollback_chapter(self, *, session_id: str, workflow_id: str, chapter_id: str) -> ToolResult:
        try:
            state = self._load_state(workflow_id, session_id=session_id)
            chapter = self._chapter(state, chapter_id)
        except (OSError, ValueError) as exc:
            return ToolResult(False, "", str(exc))
        if chapter["status"] != "in_progress" or state.get("current_chapter_id") != chapter["id"]:
            return ToolResult(False, "", "only the current in-progress chapter can be rolled back")
        chapter["status"] = "pending"
        chapter.pop("started_at", None)
        state.pop("current_chapter_id", None)
        state["status"] = "drafting"
        state["updated_at"] = utc_now_iso()
        self._write_state(state)
        return ToolResult(
            True,
            f"rolled back current chapter {chapter['id']}; completed chapters were not changed",
            data=self._result_data(state, event="chapter_rolled_back", chapter_id=chapter["id"]),
        )

    def status(self, *, session_id: str, workflow_id: str) -> ToolResult:
        try:
            state = self._load_state(workflow_id, session_id=session_id)
        except (OSError, ValueError) as exc:
            return ToolResult(False, "", str(exc))
        data = self._result_data(
            state,
            event="status_read",
            outline_hash=state["outline_hash"],
            render_preview_id=state.get("render_preview_id"),
            apply_verified=self._apply_matches(state),
            chapters=[
                {
                    "id": item["id"],
                    "title": item["title"],
                    "estimated_tokens": item["estimated_tokens"],
                    "status": item["status"],
                    "summary": item.get("summary", ""),
                }
                for item in state["chapters"]
            ],
        )
        return ToolResult(True, json.dumps(data, ensure_ascii=False, indent=2), data=data)

    def render(self, *, session_id: str, workflow_id: str) -> ToolResult:
        try:
            state = self._load_state(workflow_id, session_id=session_id)
            if state["status"] not in {"ready_to_render", "awaiting_apply"}:
                raise ValueError("all chapters must be checkpointed before rendering")
            output_path = str(state["output_path"])
            is_docx = Path(output_path).suffix.lower() == ".docx"
            markdown = self._assembled_markdown(state, include_title=not is_docx)
            if is_docx:
                content, metadata = self.document.render_docx(title=str(state["title"]), markdown=markdown)
                preview = self.file_edit.preview_binary(
                    path=output_path,
                    session_id=session_id,
                    content=content,
                    source="document_generator.render",
                )
            else:
                metadata = {"format": "markdown", "bytes": len(markdown.encode("utf-8"))}
                preview = self.file_edit.preview(path=output_path, session_id=session_id, content=markdown)
            metadata["date_literals"] = sorted(set(DATE_LITERAL_RE.findall(markdown)))[:100]
            metadata["generated_metadata_dates"] = sorted(
                {
                    date
                    for line in markdown.splitlines()
                    if ("生成" in line or "汇总" in line or "报告" in line) and ("时间" in line or "日期" in line)
                    for date in DATE_LITERAL_RE.findall(line)
                }
            )[:100]
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return ToolResult(False, "", str(exc))
        if not preview.success:
            return preview
        preview_id = str((preview.data or {}).get("preview_id") or "")
        result_hash = str((preview.data or {}).get("result_hash") or "")
        if not WORKFLOW_ID_RE.fullmatch(preview_id) or not SHA256_RE.fullmatch(result_hash):
            return ToolResult(False, "", "document render returned invalid FileEdit preview metadata")
        state["status"] = "awaiting_apply"
        state["render_preview_id"] = preview_id
        state["render_result_hash"] = result_hash
        state["updated_at"] = utc_now_iso()
        self._write_state(state)
        data = dict(preview.data or {})
        data.update(metadata)
        data.update(self._result_data(state, event="render_preview_created"))
        return ToolResult(preview.success, preview.stdout, preview.stderr, data=data)

    def finalize(
        self,
        *,
        session_id: str,
        workflow_id: str,
        verification_result: ToolResult | Mapping[str, Any] | None = None,
    ) -> ToolResult:
        """Verify the exact managed apply and Word parse receipts before completing."""

        try:
            state = self._load_state(workflow_id, session_id=session_id)
        except (OSError, ValueError) as exc:
            return ToolResult(False, "", str(exc))
        if state["status"] not in {"awaiting_apply", "completed"}:
            return ToolResult(False, "", "render and apply the document preview before finalizing")
        apply_receipt = self._apply_receipt(state)
        if apply_receipt is None:
            return ToolResult(
                False,
                "",
                "file_apply is not verified for this Session, preview, path, snapshot, and rendered hash",
            )
        verification_receipt: dict[str, Any] | None = None
        if Path(str(state["output_path"])).suffix.lower() == ".docx":
            verification_receipt, verification_error = self._docx_verification_receipt(
                state,
                verification_result,
            )
            if verification_error:
                return ToolResult(False, "", verification_error)
        if state["status"] != "completed":
            state["status"] = "completed"
            if verification_receipt is not None:
                state["docx_verification"] = verification_receipt
            state["completed_at"] = utc_now_iso()
            state["updated_at"] = state["completed_at"]
            self._write_state(state)
        return ToolResult(
            True,
            f"verified applied document {state['output_path']}",
            data=self._result_data(
                state,
                event="document_completed",
                render_preview_id=state.get("render_preview_id"),
                apply_snapshot_id=apply_receipt["snapshot_id"],
                apply_result_hash=apply_receipt["result_hash"],
                apply_verified=True,
                parse_verified=verification_receipt is not None,
            ),
        )

    def _normalize_chapters(self, chapters: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not isinstance(chapters, list) or not 1 <= len(chapters) <= self.MAX_CHAPTERS:
            raise ValueError(f"chapters must contain 1 to {self.MAX_CHAPTERS} outline entries")
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        total_tokens = 0
        for index, item in enumerate(chapters, start=1):
            if not isinstance(item, dict):
                raise TypeError("every chapter outline entry must be an object")
            chapter_id = str(item.get("id") or f"chapter-{index}").strip()
            if not CHAPTER_ID_RE.fullmatch(chapter_id) or chapter_id in seen:
                raise ValueError("chapter ids must be unique bounded ASCII identifiers")
            seen.add(chapter_id)
            chapter_title = self._bounded_required(
                item.get("title"),
                f"chapter {chapter_id} title",
                self.MAX_CHAPTER_TITLE_CHARS,
            )
            raw_tokens = item.get("estimated_tokens")
            if isinstance(raw_tokens, bool):
                raise ValueError("chapter estimated_tokens must be a positive integer")
            try:
                estimated_tokens = int(raw_tokens)
            except (TypeError, ValueError) as exc:
                raise ValueError("chapter estimated_tokens must be a positive integer") from exc
            if not 1 <= estimated_tokens <= 100_000:
                raise ValueError("chapter estimated_tokens must be between 1 and 100000")
            total_tokens += estimated_tokens
            normalized.append(
                {
                    "id": chapter_id,
                    "title": chapter_title,
                    "estimated_tokens": estimated_tokens,
                    "status": "pending",
                }
            )
        if total_tokens > self.MAX_TOTAL_ESTIMATED_TOKENS:
            raise ValueError(f"outline exceeds {self.MAX_TOTAL_ESTIMATED_TOKENS} estimated tokens")
        return normalized

    def _output_path(self, value: str) -> str:
        target = resolve_project_path(self.project.root, value, require_file=True)
        suffix = target.suffix.lower()
        if suffix not in {".docx", ".md", ".markdown"}:
            raise ValueError("document workflow output must end with .docx, .md, or .markdown")
        return target.relative_to(self.project.root.resolve()).as_posix()

    def _load_state(self, workflow_id: str, *, session_id: str) -> dict[str, Any]:
        path = self._state_path(workflow_id)
        content = self._read_bounded_regular_file(
            path,
            maximum=self.MAX_STATE_BYTES,
            label="document workflow state",
        )
        value = json.loads(content.decode("utf-8"))
        normalized_session = self._session_id(session_id)
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != self.SCHEMA_VERSION
            or value.get("workflow_id") != workflow_id
            or value.get("session_id") != normalized_session
            or not isinstance(value.get("chapters"), list)
        ):
            raise ValueError("document workflow state is invalid or belongs to another Session")
        try:
            outline_hash = self._digest(self._outline_payload(value))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("document workflow state contains an invalid outline") from exc
        if outline_hash != value.get("outline_hash"):
            raise ValueError("document workflow outline hash changed")
        return value

    def _write_state(self, state: dict[str, Any]) -> None:
        content = (json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        if len(content) > self.MAX_STATE_BYTES:
            raise ValueError("document workflow state exceeds its bounded size")
        self._atomic_write_bytes(self._state_path(str(state["workflow_id"])), content)

    def _assembled_markdown(self, state: dict[str, Any], *, include_title: bool) -> str:
        sections: list[str] = [f"# {state['title']}"] if include_title else []
        for chapter in state["chapters"]:
            if chapter["status"] != "completed":
                raise ValueError("document workflow contains an incomplete chapter")
            path = self._chapter_path(state["workflow_id"], chapter["id"])
            content = self._read_bounded_regular_file(
                path,
                maximum=self.MAX_CHAPTER_BYTES,
                label=f"chapter checkpoint {chapter['id']}",
            )
            if hashlib.sha256(content).hexdigest() != chapter.get("content_sha256"):
                raise ValueError(f"chapter checkpoint hash changed: {chapter['id']}")
            text = content.decode("utf-8")
            sections.append(f"## {chapter['title']}\n\n{text.strip()}")
        return "\n\n".join(sections).strip() + "\n"

    def _state_path(self, workflow_id: str) -> Path:
        if not WORKFLOW_ID_RE.fullmatch(str(workflow_id)):
            raise ValueError("invalid document workflow id")
        return self.root / f"{workflow_id}.json"

    def _chapter_path(self, workflow_id: str, chapter_id: str) -> Path:
        if not WORKFLOW_ID_RE.fullmatch(str(workflow_id)) or not CHAPTER_ID_RE.fullmatch(str(chapter_id)):
            raise ValueError("invalid document workflow chapter path")
        directory = self.root / workflow_id
        directory.mkdir(parents=True, exist_ok=True)
        metadata = directory.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("document workflow chapter root must be a regular directory")
        return directory / f"{chapter_id}.md"

    @staticmethod
    def _chapter(state: dict[str, Any], chapter_id: str) -> dict[str, Any]:
        chapter = next((item for item in state["chapters"] if item.get("id") == chapter_id), None)
        if not isinstance(chapter, dict):
            raise ValueError(f"unknown document workflow chapter: {chapter_id}")
        return chapter

    @staticmethod
    def _bounded_required(value: Any, label: str, maximum: int) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError(f"{label} must not be empty")
        if len(text) > maximum:
            raise ValueError(f"{label} exceeds {maximum} characters")
        return text

    @staticmethod
    def _session_id(value: Any) -> str:
        session_id = str(value or "").strip()
        if not SESSION_ID_RE.fullmatch(session_id):
            raise ValueError("document workflow Session id is invalid")
        return session_id

    @staticmethod
    def _outline_payload(state: dict[str, Any]) -> dict[str, Any]:
        chapters = state["chapters"]
        if not isinstance(chapters, list):
            raise ValueError("document workflow chapters are invalid")
        return {
            "title": state["title"],
            "output_path": state["output_path"],
            "chapters": [
                {
                    "id": item["id"],
                    "title": item["title"],
                    "estimated_tokens": item["estimated_tokens"],
                }
                for item in chapters
            ],
        }

    def _apply_matches(self, state: dict[str, Any]) -> bool:
        return self._apply_receipt(state) is not None

    def _apply_receipt(self, state: dict[str, Any]) -> dict[str, str] | None:
        preview_id = state.get("render_preview_id")
        expected = state.get("render_result_hash")
        session_id = state.get("session_id")
        if (
            not isinstance(preview_id, str)
            or not WORKFLOW_ID_RE.fullmatch(preview_id)
            or not isinstance(expected, str)
            or not SHA256_RE.fullmatch(expected)
            or not isinstance(session_id, str)
        ):
            return None
        return self.file_edit.applied_preview_receipt(
            preview_id=preview_id,
            session_id=session_id,
            path=str(state.get("output_path") or ""),
            result_hash=expected,
        )

    @staticmethod
    def _docx_verification_receipt(
        state: dict[str, Any],
        verification_result: ToolResult | Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any] | None, str]:
        evidence: ToolResult | Mapping[str, Any]
        if verification_result is None:
            stored = state.get("docx_verification")
            if not isinstance(stored, Mapping):
                return None, "re-open the applied Word output with document_parse before finalizing"
            evidence = {
                "success": True,
                "data": {ARTIFACT_VERIFICATION_METADATA_KEY: dict(stored)},
            }
        else:
            evidence = verification_result
        try:
            spec = ArtifactSpec(
                MANAGED_DOCUMENT_ARTIFACT_ID,
                str(state["output_path"]),
                format="docx",
                max_bytes=MAX_ARTIFACT_BYTES_HARD_LIMIT,
            )
            verification = ArtifactVerifier.verify_receipt(spec, evidence)
        except (KeyError, TypeError, ValueError):
            return None, "document_parse returned invalid managed Word verification evidence"
        if not verification.passed:
            detail = f": {verification.errors[0]}" if verification.errors else ""
            return None, f"document_parse did not verify the applied Word output{detail}"
        data = (
            evidence.data
            if isinstance(evidence, ToolResult) and isinstance(evidence.data, Mapping)
            else evidence.get("data")
            if isinstance(evidence, Mapping) and isinstance(evidence.get("data"), Mapping)
            else {}
        )
        receipt = data.get(ARTIFACT_VERIFICATION_METADATA_KEY)
        if not isinstance(receipt, Mapping):
            return None, "document_parse returned invalid managed Word verification evidence"
        expected_hash = state.get("render_result_hash")
        if receipt.get("content_sha256") != expected_hash:
            return None, "document_parse verification does not match the active rendered Word output"
        allowed = {
            "schema_version",
            "artifact_id",
            "path",
            "format",
            "passed",
            "content_complete",
            "size_bytes",
            "content_sha256",
            "checks_run",
            "errors",
        }
        return {str(key): value for key, value in receipt.items() if key in allowed}, ""

    @staticmethod
    def _digest(value: Any) -> str:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _result_data(state: dict[str, Any], *, event: str, **extra: Any) -> dict[str, Any]:
        completed = sum(1 for item in state["chapters"] if item["status"] == "completed")
        return {
            "document_workflow_event": event,
            "workflow_id": state["workflow_id"],
            "workflow_status": state["status"],
            "output_path": state["output_path"],
            "completed_chapters": completed,
            "total_chapters": len(state["chapters"]),
            **extra,
        }

    @staticmethod
    def _render_outline(state: dict[str, Any]) -> str:
        lines = [f"# {state['title']}", "", f"Output: `{state['output_path']}`", "", "## Outline", ""]
        lines.extend(
            f"{index}. {item['title']} (`{item['id']}`, estimated {item['estimated_tokens']} tokens)"
            for index, item in enumerate(state["chapters"], start=1)
        )
        lines.extend(["", f"Outline hash: `{state['outline_hash']}`", "Confirm this exact hash before drafting."])
        return "\n".join(lines)

    @staticmethod
    def _atomic_write_bytes(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        descriptor = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
        finally:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _read_bounded_regular_file(path: Path, *, maximum: int, label: str) -> bytes:
        try:
            metadata = path.lstat()
        except FileNotFoundError as exc:
            raise ValueError(f"{label} is missing") from exc
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > maximum:
            raise ValueError(f"{label} is not a bounded regular file")
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise ValueError(f"{label} cannot be opened safely") from exc
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
                metadata.st_dev,
                metadata.st_ino,
            ):
                raise ValueError(f"{label} changed during secure open")
            with os.fdopen(descriptor, "rb") as handle:
                descriptor = -1
                content = handle.read(maximum + 1)
                completed = os.fstat(handle.fileno())
                if (
                    completed.st_dev,
                    completed.st_ino,
                    completed.st_size,
                    completed.st_mtime_ns,
                ) != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns):
                    raise ValueError(f"{label} changed while it was being read")
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        if len(content) > maximum:
            raise ValueError(f"{label} exceeds its bounded size")
        return content


__all__ = ["DocumentGeneratorTool"]
