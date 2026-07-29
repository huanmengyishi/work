from __future__ import annotations

import json
import math
import os
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .memory import GLOBAL_KNOWLEDGE_KINDS, MemoryItem, MemoryKind, MemoryStore
from .timeutil import utc_now_iso


MEMORY_TRANSFER_FORMAT = "deep-agent-memory"
MEMORY_TRANSFER_VERSION = 1


class MemoryTransferError(ValueError):
    """A bounded portable Memory document failed validation or I/O policy."""


@dataclass(frozen=True)
class MemoryTransferLimits:
    max_file_bytes: int
    max_records: int
    max_path_chars: int
    max_title_chars: int
    max_title_bytes: int
    max_content_chars: int
    max_content_bytes: int
    max_tags_per_record: int
    max_tag_chars: int
    max_tag_bytes: int

    @classmethod
    def from_store(cls, store: MemoryStore) -> MemoryTransferLimits:
        get = store.config.get
        return cls(
            max_file_bytes=_bounded_int(
                get("memory.transfer.max_file_bytes", 8_388_608),
                default=8_388_608,
                minimum=1_024,
                maximum=67_108_864,
            ),
            max_records=_bounded_int(
                get("memory.transfer.max_records", 5_000),
                default=5_000,
                minimum=1,
                maximum=100_000,
            ),
            max_path_chars=_bounded_int(
                get("memory.transfer.max_path_chars", 4_096),
                default=4_096,
                minimum=64,
                maximum=32_768,
            ),
            max_title_chars=_bounded_int(
                get("memory.transfer.max_title_chars", 500),
                default=500,
                minimum=1,
                maximum=10_000,
            ),
            max_title_bytes=_bounded_int(
                get("memory.transfer.max_title_bytes", 2_000),
                default=2_000,
                minimum=4,
                maximum=40_000,
            ),
            max_content_chars=_bounded_int(
                get("memory.transfer.max_content_chars", 50_000),
                default=50_000,
                minimum=1,
                maximum=1_000_000,
            ),
            max_content_bytes=_bounded_int(
                get("memory.transfer.max_content_bytes", 200_000),
                default=200_000,
                minimum=4,
                maximum=4_000_000,
            ),
            max_tags_per_record=_bounded_int(
                get("memory.transfer.max_tags_per_record", 32),
                default=32,
                minimum=0,
                maximum=256,
            ),
            max_tag_chars=_bounded_int(
                get("memory.transfer.max_tag_chars", 100),
                default=100,
                minimum=1,
                maximum=1_000,
            ),
            max_tag_bytes=_bounded_int(
                get("memory.transfer.max_tag_bytes", 400),
                default=400,
                minimum=4,
                maximum=4_000,
            ),
        )


def export_memory(
    store: MemoryStore,
    destination: str | Path,
    *,
    project_id: str,
    scope: str = "project",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Atomically export the current project, global Memory, or both.

    ``project`` never includes another project's data. ``both`` means the
    current project plus global data, not every project in the shared database.
    """

    if scope not in {"project", "global", "both"}:
        raise MemoryTransferError("memory export scope must be project, global, or both")
    limits = MemoryTransferLimits.from_store(store)
    target = _validate_export_path(destination, limits, overwrite=overwrite)
    normalized_project_id = _validate_project_id(project_id, required=scope != "global")
    try:
        items = store.select_transfer_memories(
            project_id=normalized_project_id or "",
            scope=scope,
            limit=limits.max_records,
            max_payload_bytes=limits.max_file_bytes,
        )
    except ValueError as exc:
        raise MemoryTransferError(str(exc)) from exc
    records = [_record_from_item(item, limits) for item in items]
    document = {
        "format": MEMORY_TRANSFER_FORMAT,
        "version": MEMORY_TRANSFER_VERSION,
        "exported_at": utc_now_iso(),
        "selection": {
            "scope": scope,
            "project_id": normalized_project_id if scope != "global" else None,
        },
        "records": records,
    }
    encoded = (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if len(encoded) > limits.max_file_bytes:
        raise MemoryTransferError(
            f"memory export is {len(encoded)} bytes; configured limit is {limits.max_file_bytes} bytes"
        )
    _atomic_write_private(target, encoded, overwrite=overwrite)
    return {
        "format": MEMORY_TRANSFER_FORMAT,
        "version": MEMORY_TRANSFER_VERSION,
        "path": str(target),
        "scope": scope,
        "record_count": len(records),
        "bytes": len(encoded),
    }


def import_memory(
    store: MemoryStore,
    source: str | Path,
    *,
    project_id: str,
    target_scope: str = "preserve",
    conflict: str = "skip",
) -> dict[str, Any]:
    """Validate a complete versioned document, then apply it in one transaction.

    With ``preserve``, project records map to the current project while global
    records stay global. ``project`` and ``global`` explicitly remap all input.
    Exact payload duplicates are always skipped. Title conflicts use the
    requested ``skip`` (default) or ``replace`` strategy.
    """

    if target_scope not in {"preserve", "project", "global"}:
        raise MemoryTransferError("memory import target scope must be preserve, project, or global")
    if conflict not in {"skip", "replace"}:
        raise MemoryTransferError("memory import conflict strategy must be skip or replace")
    limits = MemoryTransferLimits.from_store(store)
    source_path = _validate_import_path(source, limits)
    normalized_project_id = _validate_project_id(project_id, required=target_scope != "global")
    encoded = _read_bounded_regular_file(source_path, limits.max_file_bytes)
    document = _decode_document(encoded)
    records, selection_scope = _validate_document(document, limits)
    for index, record in enumerate(records):
        destination_scope = record["scope"] if target_scope == "preserve" else target_scope
        parsed_kind = MemoryKind.parse(record["kind"])
        if destination_scope == "global" and parsed_kind not in GLOBAL_KNOWLEDGE_KINDS:
            allowed = ", ".join(sorted(item.value for item in GLOBAL_KNOWLEDGE_KINDS))
            raise MemoryTransferError(f"memory record {index} global knowledge kind must be one of: {allowed}")
    applied = store.apply_transfer_records(
        records,
        project_id=normalized_project_id or "",
        target_scope=target_scope,
        conflict=conflict,
    )
    return {
        "format": MEMORY_TRANSFER_FORMAT,
        "version": MEMORY_TRANSFER_VERSION,
        "path": str(source_path),
        "source_scope": selection_scope,
        "target_scope": target_scope,
        "conflict_strategy": conflict,
        **applied,
    }


def _record_from_item(item: MemoryItem, limits: MemoryTransferLimits) -> dict[str, Any]:
    if not isinstance(item.kind, MemoryKind):
        raise MemoryTransferError(f"memory {item.id} has unsupported legacy kind {item.kind!r}")
    record = {
        "scope": "global" if item.project_id is None else "project",
        "kind": item.kind.value,
        "title": item.title,
        "content": item.content,
        "tags": list(item.tags),
        "confidence": item.confidence,
        "expires_at": item.expires_at,
    }
    return _validate_record(record, limits, index=item.id)


def _validate_document(document: Any, limits: MemoryTransferLimits) -> tuple[list[dict[str, Any]], str]:
    if not isinstance(document, dict):
        raise MemoryTransferError("memory import root must be a JSON object")
    expected_root = {"format", "version", "exported_at", "selection", "records"}
    if set(document) != expected_root:
        raise MemoryTransferError("memory import root fields do not match version 1 schema")
    if document["format"] != MEMORY_TRANSFER_FORMAT:
        raise MemoryTransferError(f"unsupported memory import format: {document['format']!r}")
    if document["version"] != MEMORY_TRANSFER_VERSION or isinstance(document["version"], bool):
        raise MemoryTransferError(f"unsupported memory import version: {document['version']!r}")
    _validate_timestamp(document["exported_at"], field="exported_at", allow_none=False)
    selection = document["selection"]
    if not isinstance(selection, dict) or set(selection) != {"scope", "project_id"}:
        raise MemoryTransferError("memory import selection must contain only scope and project_id")
    selection_scope = selection["scope"]
    if selection_scope not in {"project", "global", "both"}:
        raise MemoryTransferError("memory import selection scope must be project, global, or both")
    source_project_id = selection["project_id"]
    if selection_scope == "global":
        if source_project_id is not None:
            raise MemoryTransferError("global memory export must not carry a project_id")
    else:
        _validate_project_id(source_project_id, required=True)
    raw_records = document["records"]
    if not isinstance(raw_records, list):
        raise MemoryTransferError("memory import records must be a JSON array")
    if len(raw_records) > limits.max_records:
        raise MemoryTransferError(
            f"memory import contains {len(raw_records)} records; configured limit is {limits.max_records}"
        )
    records = [_validate_record(record, limits, index=index) for index, record in enumerate(raw_records)]
    allowed_scopes = {
        "project": {"project"},
        "global": {"global"},
        "both": {"project", "global"},
    }[selection_scope]
    if any(record["scope"] not in allowed_scopes for record in records):
        raise MemoryTransferError("memory record scope is outside the document selection")
    return records, str(selection_scope)


def _validate_record(record: Any, limits: MemoryTransferLimits, *, index: int) -> dict[str, Any]:
    label = f"memory record {index}"
    if not isinstance(record, dict):
        raise MemoryTransferError(f"{label} must be a JSON object")
    expected = {"scope", "kind", "title", "content", "tags", "confidence", "expires_at"}
    if set(record) != expected:
        raise MemoryTransferError(f"{label} fields do not match version 1 schema")
    scope = record["scope"]
    if scope not in {"project", "global"}:
        raise MemoryTransferError(f"{label} scope must be project or global")
    try:
        kind = MemoryKind.parse(record["kind"])
    except ValueError as exc:
        raise MemoryTransferError(f"{label} has an invalid kind: {exc}") from exc
    if not isinstance(kind, MemoryKind):
        raise MemoryTransferError(f"{label} has an invalid kind")
    title = _bounded_text(
        record["title"],
        label=f"{label} title",
        max_chars=limits.max_title_chars,
        max_bytes=limits.max_title_bytes,
    )
    content = _bounded_text(
        record["content"],
        label=f"{label} content",
        max_chars=limits.max_content_chars,
        max_bytes=limits.max_content_bytes,
        preserve_whitespace=True,
    )
    raw_tags = record["tags"]
    if not isinstance(raw_tags, list):
        raise MemoryTransferError(f"{label} tags must be a JSON array")
    if len(raw_tags) > limits.max_tags_per_record:
        raise MemoryTransferError(f"{label} has more than {limits.max_tags_per_record} tags")
    tags: list[str] = []
    seen_tags: set[str] = set()
    for tag_index, raw_tag in enumerate(raw_tags):
        tag = _bounded_text(
            raw_tag,
            label=f"{label} tag {tag_index}",
            max_chars=limits.max_tag_chars,
            max_bytes=limits.max_tag_bytes,
        )
        if tag not in seen_tags:
            tags.append(tag)
            seen_tags.add(tag)
    confidence = record["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise MemoryTransferError(f"{label} confidence must be a number from 0 to 1")
    confidence_value = float(confidence)
    if not math.isfinite(confidence_value) or not 0.0 <= confidence_value <= 1.0:
        raise MemoryTransferError(f"{label} confidence must be a finite number from 0 to 1")
    expires_at = _validate_timestamp(record["expires_at"], field=f"{label} expires_at", allow_none=True)
    return {
        "scope": scope,
        "kind": kind.value,
        "title": title,
        "content": content,
        "tags": tags,
        "confidence": confidence_value,
        "expires_at": expires_at,
    }


def _decode_document(encoded: bytes) -> Any:
    try:
        text = encoded.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise MemoryTransferError("memory import must be valid UTF-8 JSON") from exc

    def reject_constant(value: str) -> None:
        raise MemoryTransferError(f"memory import contains invalid JSON number {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise MemoryTransferError(f"memory import contains duplicate JSON field {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except MemoryTransferError:
        raise
    except (ValueError, RecursionError) as exc:
        raise MemoryTransferError(f"memory import is not valid bounded JSON: {exc}") from exc


def _bounded_text(
    value: Any,
    *,
    label: str,
    max_chars: int,
    max_bytes: int,
    preserve_whitespace: bool = False,
) -> str:
    if not isinstance(value, str):
        raise MemoryTransferError(f"{label} must be text")
    if not value.strip():
        raise MemoryTransferError(f"{label} must not be empty")
    if len(value) > max_chars or _utf8_length(value, label=label) > max_bytes:
        raise MemoryTransferError(f"{label} exceeds its configured character or UTF-8 byte limit")
    return value if preserve_whitespace else value.strip()


def _validate_timestamp(value: Any, *, field: str, allow_none: bool) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value or len(value) > 64:
        raise MemoryTransferError(f"{field} must be a bounded ISO-8601 timestamp")
    if _utf8_length(value, label=field) > 256:
        raise MemoryTransferError(f"{field} must be a bounded ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("timezone required")
        parsed.astimezone(UTC)
    except (ValueError, OverflowError) as exc:
        raise MemoryTransferError(f"{field} must be a valid ISO-8601 timestamp") from exc
    return value


def _validate_project_id(value: Any, *, required: bool) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str):
        raise MemoryTransferError("memory project_id must be text")
    normalized = value.strip()
    if not normalized and not required:
        return None
    if not normalized or len(normalized) > 500 or _utf8_length(normalized, label="memory project_id") > 2_000:
        raise MemoryTransferError("memory project_id must contain 1 to 500 characters and at most 2000 UTF-8 bytes")
    return normalized


def _validate_export_path(
    value: str | Path,
    limits: MemoryTransferLimits,
    *,
    overwrite: bool,
) -> Path:
    path = _bounded_path(value, limits)
    parent = path.parent
    _reject_symlink_components(parent, label="memory export parent")
    if not parent.exists() or not parent.is_dir():
        raise MemoryTransferError(f"memory export parent directory does not exist: {parent}")
    if path.exists() or path.is_symlink():
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            raise MemoryTransferError(f"cannot inspect memory export path: {exc}") from exc
        if stat.S_ISLNK(mode):
            raise MemoryTransferError("memory export refuses a symbolic-link destination")
        if not stat.S_ISREG(mode):
            raise MemoryTransferError("memory export destination must be a regular file")
        if not overwrite:
            raise MemoryTransferError("memory export destination exists; pass --force to replace it atomically")
    return path


def _validate_import_path(value: str | Path, limits: MemoryTransferLimits) -> Path:
    path = _bounded_path(value, limits)
    _reject_symlink_components(path.parent, label="memory import parent")
    try:
        info = path.lstat()
    except OSError as exc:
        raise MemoryTransferError(f"cannot inspect memory import path: {exc}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise MemoryTransferError("memory import refuses a symbolic-link source")
    if not stat.S_ISREG(info.st_mode):
        raise MemoryTransferError("memory import source must be a regular file")
    if info.st_size > limits.max_file_bytes:
        raise MemoryTransferError(
            f"memory import is {info.st_size} bytes; configured limit is {limits.max_file_bytes} bytes"
        )
    return path


def _bounded_path(value: str | Path, limits: MemoryTransferLimits) -> Path:
    rendered = os.fspath(value)
    if not isinstance(rendered, str):
        raise MemoryTransferError("memory transfer path must be text")
    if (
        not rendered
        or "\x00" in rendered
        or len(rendered) > limits.max_path_chars
        or _utf8_length(rendered, label="memory transfer path") > limits.max_path_chars * 4
    ):
        raise MemoryTransferError("memory transfer path is empty or exceeds its configured limit")
    return Path(rendered).expanduser()


def _atomic_write_private(path: Path, content: bytes, *, overwrite: bool) -> None:
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    descriptor: int | None = None
    try:
        descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            try:
                destination_info = path.lstat()
            except FileNotFoundError:
                destination_info = None
            if destination_info is not None:
                if stat.S_ISLNK(destination_info.st_mode):
                    raise MemoryTransferError("memory export destination became a symbolic link")
                if not stat.S_ISREG(destination_info.st_mode):
                    raise MemoryTransferError("memory export destination must remain a regular file")
            os.replace(temp, path)
        else:
            try:
                os.link(temp, path, follow_symlinks=False)
            except FileExistsError as exc:
                raise MemoryTransferError("memory export destination appeared during export") from exc
            temp.unlink()
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except MemoryTransferError:
        raise
    except OSError as exc:
        raise MemoryTransferError(f"memory export failed atomically: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass


def _read_bounded_regular_file(path: Path, max_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise MemoryTransferError("memory import source must remain a regular file")
        _assert_open_file_identity(path, info)
        if info.st_size > max_bytes:
            raise MemoryTransferError(f"memory import is {info.st_size} bytes; configured limit is {max_bytes} bytes")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            content = handle.read(max_bytes + 1)
            after = os.fstat(handle.fileno())
        if len(content) > max_bytes:
            raise MemoryTransferError(f"memory import exceeds the configured {max_bytes} byte limit while being read")
        if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        ):
            raise MemoryTransferError("memory import source changed while it was being read")
        _assert_open_file_identity(path, info)
        return content
    except MemoryTransferError:
        raise
    except OSError as exc:
        raise MemoryTransferError(f"memory import could not be read safely: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return default
    return min(maximum, max(minimum, parsed))


def _utf8_length(value: str, *, label: str) -> int:
    try:
        return len(value.encode("utf-8", errors="strict"))
    except UnicodeEncodeError as exc:
        raise MemoryTransferError(f"{label} must be valid Unicode text") from exc


def _reject_symlink_components(path: Path, *, label: str) -> None:
    """Reject existing symbolic-link components without resolving the path."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            info = current.lstat()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise MemoryTransferError(f"cannot inspect {label}: {exc}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise MemoryTransferError(f"{label} must not contain symbolic links")


def _assert_open_file_identity(path: Path, opened: os.stat_result) -> None:
    try:
        current = path.lstat()
    except OSError as exc:
        raise MemoryTransferError(f"memory import source changed before it could be validated: {exc}") from exc
    if stat.S_ISLNK(current.st_mode) or not stat.S_ISREG(current.st_mode):
        raise MemoryTransferError("memory import source must remain a non-symbolic regular file")
    if not os.path.samestat(opened, current):
        raise MemoryTransferError("memory import source changed before it could be validated")
