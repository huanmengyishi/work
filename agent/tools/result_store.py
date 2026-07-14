from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from .base import ToolResult, _bounded_result_data, _head_tail, bounded_result_source_bytes


ATTACHMENT_DATA_KEY = "attachment"
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,160}$")
_MAX_REQUEST_ID_CHARS = 512


class ToolResultStoreError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolResultChunk:
    content: str
    request_id: str
    offset: int
    next_offset: int
    total_chars: int
    total_bytes: int
    sha256: str

    @property
    def eof(self) -> bool:
        return self.next_offset >= self.total_chars


class ToolResultStore:
    """Session-private, bounded storage for complete ToolResult payloads."""

    def __init__(
        self,
        agent_dir: Path,
        *,
        max_attachment_bytes: int,
        persist_threshold_bytes: int,
        preview_chars: int,
        max_read_chars: int,
        max_attachments_per_session: int,
        max_session_bytes: int,
    ) -> None:
        self.agent_dir = agent_dir
        self.root = agent_dir / "tool-results"
        self.max_attachment_bytes = bounded_result_source_bytes(max_attachment_bytes)
        self.persist_threshold_bytes = max(
            512,
            min(int(persist_threshold_bytes), self.max_attachment_bytes),
        )
        self.preview_chars = max(512, min(int(preview_chars), 100_000))
        self.max_read_chars = max(256, min(int(max_read_chars), 100_000))
        self.max_attachments_per_session = max(1, min(int(max_attachments_per_session), 10_000))
        self.max_session_bytes = max(
            self.max_attachment_bytes,
            min(int(max_session_bytes), 1024 * 1024 * 1024),
        )
        self._lock = threading.RLock()

    def persist(self, result: ToolResult, *, session_id: str, request_id: str) -> ToolResult:
        normalized_session = self._validate_session_id(session_id)
        normalized_request = self._validate_request_id(request_id)
        raw_payload = _canonical_bytes(result.to_dict())
        original_serialized_bytes = len(raw_payload)
        if original_serialized_bytes <= self.persist_threshold_bytes:
            return result
        serialization_truncated = original_serialized_bytes > self.max_attachment_bytes
        payload = (
            _bounded_attachment_payload(result, self.max_attachment_bytes, original_serialized_bytes)
            if serialization_truncated
            else raw_payload
        )
        if len(payload) > self.max_attachment_bytes:
            raise ToolResultStoreError("bounded tool result serialization exceeds its hard attachment limit")

        with self._lock:
            session_dir = self._session_dir(normalized_session, create=True)
            path = self._result_path(session_dir, normalized_request)
            self._write_once(path, payload)

        digest = hashlib.sha256(payload).hexdigest()
        source = result.data if isinstance(result.data, dict) else {}
        upstream_truncated = bool(source.get("source_truncated"))
        upstream_original = source.get("source_original_bytes")
        source_original_bytes = (
            int(upstream_original)
            if isinstance(upstream_original, int) and not isinstance(upstream_original, bool) and upstream_original >= 0
            else original_serialized_bytes
        )
        metadata = {
            "request_id": normalized_request,
            "bytes": len(payload),
            "sha256": digest,
            "content_type": "application/json",
            "original_serialized_bytes": original_serialized_bytes,
            "source_truncated": upstream_truncated or serialization_truncated,
            "source_original_bytes": source_original_bytes,
            "source_original_bytes_known": bool(source.get("source_original_bytes_known", True)),
        }
        return self._preview(result, metadata)

    def read_chunk(
        self,
        *,
        session_id: str,
        request_id: str,
        offset: int = 0,
        max_chars: int | None = None,
    ) -> ToolResultChunk:
        normalized_session = self._validate_session_id(session_id)
        normalized_request = self._validate_request_id(request_id)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ToolResultStoreError("tool result offset must be a non-negative integer")
        limit = self.max_read_chars if max_chars is None else max_chars
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= self.max_read_chars:
            raise ToolResultStoreError(f"tool result max_chars must be between 1 and {self.max_read_chars}")

        with self._lock:
            session_dir = self._session_dir(normalized_session, create=False)
            path = self._result_path(session_dir, normalized_request)
            payload = self._read_regular_file(path)
        try:
            text = payload.decode("utf-8", errors="strict")
            decoded = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ToolResultStoreError("stored tool result is not valid UTF-8 JSON") from exc
        if not isinstance(decoded, dict) or str(decoded.get("request_id") or "") != normalized_request:
            raise ToolResultStoreError("stored tool result request_id does not match the requested attachment")
        if offset > len(text):
            raise ToolResultStoreError(f"tool result offset exceeds total characters: {offset} > {len(text)}")
        content = text[offset : offset + limit]
        next_offset = offset + len(content)
        return ToolResultChunk(
            content=content,
            request_id=normalized_request,
            offset=offset,
            next_offset=next_offset,
            total_chars=len(text),
            total_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )

    def path_for_test(self, *, session_id: str, request_id: str) -> Path:
        """Return the deterministic private path without creating it."""

        session = self._validate_session_id(session_id)
        request = self._validate_request_id(request_id)
        return self.root / session / self._filename(request)

    def _preview(self, result: ToolResult, attachment: dict[str, Any]) -> ToolResult:
        if result.success:
            stdout_chars = self.preview_chars * 3 // 4
            stderr_chars = self.preview_chars - stdout_chars
        else:
            stderr_chars = self.preview_chars * 3 // 4
            stdout_chars = self.preview_chars - stderr_chars
        bounded_data = _bounded_result_data(result.data or {})
        data = dict(bounded_data) if isinstance(bounded_data, dict) else {"value": bounded_data}
        data[ATTACHMENT_DATA_KEY] = attachment
        return ToolResult(
            result.success,
            _head_tail(result.stdout, stdout_chars),
            _head_tail(result.stderr, stderr_chars),
            data=data,
            duration_ms=result.duration_ms,
            request_id=result.request_id,
        )

    def _session_dir(self, session_id: str, *, create: bool) -> Path:
        self._validate_private_parent()
        _ensure_private_directory(self.root, create=create)
        session_dir = self.root / session_id
        _ensure_private_directory(session_dir, create=create)
        return session_dir

    def _validate_private_parent(self) -> None:
        if self.agent_dir.is_symlink():
            raise ToolResultStoreError(f"project Agent directory must not be a symbolic link: {self.agent_dir}")
        try:
            mode = self.agent_dir.stat().st_mode
        except FileNotFoundError as exc:
            raise ToolResultStoreError(f"project Agent directory does not exist: {self.agent_dir}") from exc
        if not stat.S_ISDIR(mode):
            raise ToolResultStoreError(f"project Agent path is not a directory: {self.agent_dir}")

    def _write_once(self, path: Path, payload: bytes) -> None:
        if path.exists() or path.is_symlink():
            existing = self._read_regular_file(path)
            if existing != payload:
                raise ToolResultStoreError("request_id already has a different stored tool result")
            return

        attachment_count, session_bytes = self._session_usage(path.parent)
        if attachment_count >= self.max_attachments_per_session:
            raise ToolResultStoreError(
                f"Session tool result attachment count exceeds {self.max_attachments_per_session}"
            )
        if session_bytes + len(payload) > self.max_session_bytes:
            raise ToolResultStoreError(f"Session tool result attachments exceed {self.max_session_bytes} bytes")

        temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = -1
        try:
            fd = os.open(temp, flags, 0o600)
            with os.fdopen(fd, "wb", closefd=True) as handle:
                fd = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp, 0o600, follow_symlinks=False)
            try:
                os.link(temp, path, follow_symlinks=False)
            except FileExistsError:
                existing = self._read_regular_file(path)
                if existing != payload:
                    raise ToolResultStoreError("request_id already has a different stored tool result") from None
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError as exc:
            raise ToolResultStoreError(f"could not persist tool result securely: {exc}") from exc
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                temp.unlink()
            except FileNotFoundError:
                pass

    def _session_usage(self, session_dir: Path) -> tuple[int, int]:
        count = 0
        total_bytes = 0
        try:
            entries = os.scandir(session_dir)
        except OSError as exc:
            raise ToolResultStoreError(f"could not inspect Session tool result attachments: {exc}") from exc
        with entries:
            for entry in entries:
                if entry.name.startswith(".") and entry.name.endswith(".tmp"):
                    continue
                try:
                    info = entry.stat(follow_symlinks=False)
                except OSError as exc:
                    raise ToolResultStoreError(f"could not inspect tool result attachment: {exc}") from exc
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                    raise ToolResultStoreError("Session tool result directory contains a non-regular entry")
                count += 1
                total_bytes += info.st_size
                if count > self.max_attachments_per_session or total_bytes > self.max_session_bytes:
                    break
        return count, total_bytes

    def _read_regular_file(self, path: Path) -> bytes:
        try:
            info = path.lstat()
        except FileNotFoundError as exc:
            raise ToolResultStoreError(f"tool result attachment does not exist for request_id: {path.stem}") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ToolResultStoreError("tool result attachment must be a regular non-symlink file")
        if info.st_size > self.max_attachment_bytes:
            raise ToolResultStoreError("stored tool result exceeds its configured attachment limit")
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(path, flags)
        except OSError as exc:
            raise ToolResultStoreError(f"could not open tool result attachment securely: {exc}") from exc
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                raise ToolResultStoreError("tool result attachment changed during secure open")
            chunks: list[bytes] = []
            remaining = self.max_attachment_bytes + 1
            while remaining > 0:
                chunk = os.read(fd, min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
            if len(payload) > self.max_attachment_bytes:
                raise ToolResultStoreError("stored tool result exceeds its configured attachment limit")
            return payload
        finally:
            os.close(fd)

    @staticmethod
    def _filename(request_id: str) -> str:
        return hashlib.sha256(request_id.encode("utf-8")).hexdigest() + ".json"

    @classmethod
    def _result_path(cls, session_dir: Path, request_id: str) -> Path:
        return session_dir / cls._filename(request_id)

    @staticmethod
    def _validate_session_id(value: str) -> str:
        normalized = str(value)
        if not _SESSION_ID_RE.fullmatch(normalized):
            raise ToolResultStoreError("invalid session_id for tool result attachment")
        return normalized

    @staticmethod
    def _validate_request_id(value: str) -> str:
        normalized = str(value)
        if not normalized or len(normalized) > _MAX_REQUEST_ID_CHARS or "\x00" in normalized:
            raise ToolResultStoreError(f"request_id must contain 1-{_MAX_REQUEST_ID_CHARS} non-NUL characters")
        return normalized


def _ensure_private_directory(path: Path, *, create: bool) -> None:
    if path.is_symlink():
        raise ToolResultStoreError(f"private tool result directory must not be a symbolic link: {path}")
    if path.exists():
        if not path.is_dir():
            raise ToolResultStoreError(f"private tool result path is not a directory: {path}")
    elif create:
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
        if path.is_symlink() or not path.is_dir():
            raise ToolResultStoreError(f"private tool result path is not a safe directory: {path}")
    else:
        raise ToolResultStoreError(f"private tool result directory does not exist: {path}")
    try:
        path.chmod(0o700)
    except OSError as exc:
        raise ToolResultStoreError(f"could not secure private tool result directory: {path}: {exc}") from exc


def _canonical_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=lambda item: str(item),
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise ToolResultStoreError(f"tool result is not safely serializable: {exc}") from exc
    return rendered.encode("utf-8")


def _bounded_attachment_payload(result: ToolResult, max_bytes: int, original_bytes: int) -> bytes:
    data = _bounded_result_data(result.data or {})
    payload: dict[str, Any] = {
        "success": result.success,
        "stdout": "",
        "stderr": "",
        "duration_ms": result.duration_ms,
        "request_id": result.request_id,
        "data": data,
        "source_truncated": True,
        "original_serialized_bytes": original_bytes,
    }

    def render(excerpt_chars: int) -> bytes:
        if result.success:
            stdout_chars = excerpt_chars * 3 // 4
            stderr_chars = excerpt_chars - stdout_chars
        else:
            stderr_chars = excerpt_chars * 3 // 4
            stdout_chars = excerpt_chars - stderr_chars
        payload["stdout"] = _head_tail(result.stdout, stdout_chars)
        payload["stderr"] = _head_tail(result.stderr, stderr_chars)
        return _canonical_bytes(payload)

    base = render(0)
    if len(base) > max_bytes:
        payload["data"] = {"keys": sorted(str(key) for key in (result.data or {}))[:30]}
        base = render(0)
    if len(base) > max_bytes:
        payload["data"] = {}
        base = render(0)
    if len(base) > max_bytes:
        raise ToolResultStoreError("attachment limit is too small for bounded tool result metadata")

    low = 0
    high = len(result.stdout) + len(result.stderr)
    best = base
    while low <= high:
        middle = (low + high) // 2
        candidate = render(middle)
        if len(candidate) <= max_bytes:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best
