from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..project import Project
from ..timeutil import utc_now_iso
from .base import ToolResult, truncate_text
from .pathsafe import resolve_project_path


ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
MISSING_HASH = "missing"


class FileEditTool:
    """Preview-first file editing with non-destructive, session-scoped snapshots."""

    def __init__(self, project: Project, max_file_bytes: int = 2_000_000) -> None:
        self.project = project
        self.max_file_bytes = max_file_bytes
        self.preview_dir = project.agent_dir / "cache" / "file-previews"
        self.snapshot_root = project.agent_dir / "snapshots"
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_root.mkdir(parents=True, exist_ok=True)
        for path in (self.preview_dir, self.snapshot_root):
            try:
                path.chmod(0o700)
            except OSError:
                pass

    def preview(
        self,
        *,
        path: str,
        session_id: str,
        content: str | None = None,
        old_text: str | None = None,
        new_text: str | None = None,
        replace_all: bool = False,
        delete: bool = False,
    ) -> ToolResult:
        target = resolve_project_path(self.project.root, path, require_file=True)
        before_bytes = self._read_bytes(target)
        before_text = self._decode_text(before_bytes, target) if before_bytes is not None else ""

        if delete:
            if content is not None or old_text is not None or new_text is not None:
                return ToolResult(False, "", "delete cannot be combined with content or text replacement")
            if before_bytes is None:
                return ToolResult(False, "", f"file does not exist: {self._relative(target)}")
            after_text: str | None = None
        elif content is not None:
            if old_text is not None or new_text is not None:
                return ToolResult(False, "", "content cannot be combined with old_text/new_text")
            after_text = content
        elif old_text is not None:
            if new_text is None:
                return ToolResult(False, "", "new_text is required when old_text is provided")
            occurrences = before_text.count(old_text)
            if occurrences == 0:
                return ToolResult(False, "", "old_text was not found; refresh the file and create a new preview")
            if occurrences > 1 and not replace_all:
                return ToolResult(
                    False,
                    "",
                    f"old_text matched {occurrences} locations; use a larger unique block or set replace_all=true",
                )
            after_text = before_text.replace(old_text, new_text, -1 if replace_all else 1)
        else:
            return ToolResult(False, "", "provide content, old_text/new_text, or delete=true")

        after_bytes = None if after_text is None else after_text.encode("utf-8")
        if after_bytes is not None and len(after_bytes) > self.max_file_bytes:
            return ToolResult(False, "", f"result exceeds file-edit limit: {len(after_bytes)} bytes")
        if before_bytes == after_bytes:
            return ToolResult(False, "", "preview contains no changes")

        relative = self._relative(target)
        diff = self._diff(relative, before_text if before_bytes is not None else None, after_text)
        preview_id = uuid4().hex
        record = {
            "schema_version": 1,
            "preview_id": preview_id,
            "session_id": session_id,
            "path": relative,
            "base_hash": self._hash(before_bytes),
            "result_hash": self._hash(after_bytes),
            "before_exists": before_bytes is not None,
            "after_exists": after_bytes is not None,
            "content": after_text,
            "diff": diff,
            "status": "pending",
            "created_at": utc_now_iso(),
        }
        self._write_json(self._preview_path(preview_id), record)
        return ToolResult(
            True,
            truncate_text(diff),
            data={
                "preview_id": preview_id,
                "path": relative,
                "base_hash": record["base_hash"],
                "result_hash": record["result_hash"],
                "requires_apply": True,
            },
        )

    def apply(self, *, preview_id: str, session_id: str) -> ToolResult:
        record = self._load_preview(preview_id)
        self._check_session(record, session_id)
        if record.get("status") != "pending":
            return ToolResult(False, "", f"preview is not pending: {preview_id} ({record.get('status')})")
        target = resolve_project_path(self.project.root, str(record["path"]), require_file=True)
        current = self._read_bytes(target)
        if self._hash(current) != record["base_hash"]:
            return ToolResult(False, "", "file changed after preview; create a fresh file_diff before applying")

        after_text = record.get("content") if record.get("after_exists") else None
        after_bytes = after_text.encode("utf-8") if isinstance(after_text, str) else None
        snapshot_id = uuid4().hex
        snapshot_dir = self._session_snapshot_dir(session_id) / snapshot_id
        snapshot_dir.mkdir(parents=True, exist_ok=False)
        try:
            snapshot_dir.chmod(0o700)
        except OSError:
            pass
        if current is not None:
            before_path = snapshot_dir / "before.bin"
            before_path.write_bytes(current)
            try:
                before_path.chmod(0o600)
            except OSError:
                pass
        manifest = {
            "schema_version": 1,
            "snapshot_id": snapshot_id,
            "session_id": session_id,
            "preview_id": preview_id,
            "path": str(record["path"]),
            "before_exists": current is not None,
            "before_hash": self._hash(current),
            "after_exists": after_bytes is not None,
            "after_hash": self._hash(after_bytes),
            "before_mode": (target.stat().st_mode & 0o777) if target.exists() else 0o644,
            "git": self._git_metadata(target),
            "status": "prepared",
            "created_at": utc_now_iso(),
        }
        self._write_json(snapshot_dir / "manifest.json", manifest)
        try:
            self._replace_target(target, after_bytes, int(manifest["before_mode"]))
            if self._hash(self._read_bytes(target)) != manifest["after_hash"]:
                raise RuntimeError("post-write hash verification failed")
        except Exception:
            self._restore(snapshot_dir, manifest, verify_after=False)
            manifest["status"] = "rolled_back"
            self._write_json(snapshot_dir / "manifest.json", manifest)
            raise

        manifest["status"] = "applied"
        manifest["applied_at"] = utc_now_iso()
        self._write_json(snapshot_dir / "manifest.json", manifest)
        self._append_stack(session_id, snapshot_id)
        record["status"] = "applied"
        record["snapshot_id"] = snapshot_id
        record["applied_at"] = utc_now_iso()
        self._write_json(self._preview_path(preview_id), record)
        return ToolResult(
            True,
            f"applied {record['path']} (snapshot {snapshot_id})",
            data={"snapshot_id": snapshot_id, "preview_id": preview_id, "path": record["path"]},
        )

    def undo(self, *, session_id: str, snapshot_id: str | None = None) -> ToolResult:
        stack = self._load_stack(session_id)
        selected = snapshot_id or next(
            (item["snapshot_id"] for item in reversed(stack) if item.get("status") == "applied"),
            None,
        )
        if not selected:
            return ToolResult(False, "", "no applied snapshot is available for this session")
        self._validate_id(selected, "snapshot")
        snapshot_dir = self._session_snapshot_dir(session_id) / selected
        manifest = self._read_json(snapshot_dir / "manifest.json")
        self._check_session(manifest, session_id)
        if manifest.get("status") != "applied":
            return ToolResult(False, "", f"snapshot is not active: {selected} ({manifest.get('status')})")
        target = resolve_project_path(self.project.root, str(manifest["path"]), require_file=True)
        if self._hash(self._read_bytes(target)) != manifest["after_hash"]:
            return ToolResult(False, "", "file changed after the snapshot; refusing to overwrite newer work")

        self._restore(snapshot_dir, manifest, verify_after=True)
        manifest["status"] = "undone"
        manifest["undone_at"] = utc_now_iso()
        self._write_json(snapshot_dir / "manifest.json", manifest)
        for item in stack:
            if item.get("snapshot_id") == selected:
                item["status"] = "undone"
                item["undone_at"] = manifest["undone_at"]
        self._write_stack(session_id, stack)
        return ToolResult(
            True,
            f"restored {manifest['path']} from snapshot {selected}",
            data={"snapshot_id": selected, "path": manifest["path"]},
        )

    def status(self, session_id: str | None = None) -> dict[str, Any]:
        pending = 0
        for path in self.preview_dir.glob("*.json"):
            try:
                record = self._read_json(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if record.get("status") == "pending" and (not session_id or record.get("session_id") == session_id):
                pending += 1
        stack = self._load_stack(session_id) if session_id else []
        active = [item for item in stack if item.get("status") == "applied"]
        return {"pending_previews": pending, "active_snapshots": len(active), "snapshots": stack}

    def approval_summary(self, preview_id: str) -> str:
        record = self._load_preview(preview_id)
        return f"Apply file preview {preview_id} to {record['path']}?\n\n{truncate_text(str(record['diff']), 8000)}"

    def _restore(self, snapshot_dir: Path, manifest: dict[str, Any], *, verify_after: bool) -> None:
        target = resolve_project_path(self.project.root, str(manifest["path"]), require_file=True)
        if verify_after and self._hash(self._read_bytes(target)) != manifest["after_hash"]:
            raise RuntimeError("snapshot target no longer matches the applied version")
        if manifest.get("before_exists"):
            self._replace_target(target, (snapshot_dir / "before.bin").read_bytes(), int(manifest["before_mode"]))
        elif target.exists():
            target.unlink()
        if self._hash(self._read_bytes(target)) != manifest["before_hash"]:
            raise RuntimeError("snapshot restore hash verification failed")

    def _replace_target(self, target: Path, content: bytes | None, mode: int) -> None:
        if content is None:
            if target.exists():
                target.unlink()
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.parent / f".{target.name}.deep-agent-{uuid4().hex}.tmp"
        try:
            temp.write_bytes(content)
            temp.chmod(mode)
            os.replace(temp, target)
        finally:
            if temp.exists():
                temp.unlink()

    def _git_metadata(self, target: Path) -> dict[str, Any]:
        def run(*args: str) -> str:
            try:
                completed = subprocess.run(
                    ["git", *args],
                    cwd=self.project.root,
                    text=True,
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                return ""
            return completed.stdout.strip() if completed.returncode == 0 else ""

        relative = self._relative(target)
        top = run("rev-parse", "--show-toplevel")
        if not top:
            return {"repository": False}
        return {
            "repository": True,
            "root": top,
            "head": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "path_status": run("status", "--short", "--", relative),
        }

    @staticmethod
    def _diff(path: str, before: str | None, after: str | None) -> str:
        before_lines = [] if before is None else before.splitlines(keepends=True)
        after_lines = [] if after is None else after.splitlines(keepends=True)
        from_name = "/dev/null" if before is None else f"a/{path}"
        to_name = "/dev/null" if after is None else f"b/{path}"
        return "".join(difflib.unified_diff(before_lines, after_lines, fromfile=from_name, tofile=to_name))

    def _read_bytes(self, path: Path) -> bytes | None:
        if not path.exists():
            return None
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"refusing to edit non-regular file: {path}")
        size = path.stat().st_size
        if size > self.max_file_bytes:
            raise ValueError(f"file exceeds file-edit limit: {size} bytes")
        return path.read_bytes()

    @staticmethod
    def _decode_text(content: bytes, path: Path) -> str:
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"file is not UTF-8 text: {path}") from exc

    @staticmethod
    def _hash(content: bytes | None) -> str:
        return MISSING_HASH if content is None else hashlib.sha256(content).hexdigest()

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.project.root.resolve()).as_posix()

    def _preview_path(self, preview_id: str) -> Path:
        self._validate_id(preview_id, "preview")
        return self.preview_dir / f"{preview_id}.json"

    def _load_preview(self, preview_id: str) -> dict[str, Any]:
        path = self._preview_path(preview_id)
        if not path.exists():
            raise FileNotFoundError(f"file preview not found: {preview_id}")
        return self._read_json(path)

    def _session_snapshot_dir(self, session_id: str) -> Path:
        self._validate_id(session_id, "session")
        path = self.snapshot_root / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _load_stack(self, session_id: str | None) -> list[dict[str, Any]]:
        if not session_id:
            return []
        path = self._session_snapshot_dir(session_id) / "stack.json"
        if not path.exists():
            return []
        data = self._read_json(path)
        return list(data.get("snapshots") or [])

    def _append_stack(self, session_id: str, snapshot_id: str) -> None:
        stack = self._load_stack(session_id)
        stack.append({"snapshot_id": snapshot_id, "status": "applied", "created_at": utc_now_iso()})
        self._write_stack(session_id, stack)

    def _write_stack(self, session_id: str, stack: list[dict[str, Any]]) -> None:
        self._write_json(
            self._session_snapshot_dir(session_id) / "stack.json",
            {"schema_version": 1, "session_id": session_id, "snapshots": stack},
        )

    @staticmethod
    def _check_session(record: dict[str, Any], session_id: str) -> None:
        if record.get("session_id") != session_id:
            raise ValueError("preview or snapshot belongs to a different session")

    @staticmethod
    def _validate_id(value: str, label: str) -> None:
        if not ID_RE.fullmatch(value):
            raise ValueError(f"invalid {label} id")

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"invalid JSON record: {path}")
        return value

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(path)
        try:
            path.chmod(0o600)
        except OSError:
            pass
