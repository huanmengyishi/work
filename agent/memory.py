from __future__ import annotations

import json
import difflib
import hashlib
import math
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable

from . import paths
from .config import AppConfig
from .project import Project
from .timeutil import utc_now_iso
from .vector import OptionalChromaStore


_MEMORY_PAYLOAD_BYTES_SQL = """(
    length(cast(coalesce(kind, '') as blob))
    + length(cast(coalesce(title, '') as blob))
    + length(cast(coalesce(content, '') as blob))
    + length(cast(coalesce(tags, '') as blob))
)"""


class MemoryKind(StrEnum):
    """Canonical kinds accepted for new Memory records.

    ``allow_unknown`` exists only for loading legacy databases.  Write paths
    must use the strict default so a typo cannot silently create a new kind.
    """

    LESSON = "Lesson"
    CORRECTION = "Correction"
    REFLECTION = "Reflection"
    BUG = "Bug"
    DECISION = "Decision"
    KNOWLEDGE = "Knowledge"
    SUMMARY = "Summary"

    @classmethod
    def parse(cls, value: object, *, allow_unknown: bool = False) -> MemoryKind | str:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.strip()
            by_value = {item.value.casefold(): item for item in cls}
            parsed = by_value.get(normalized.casefold())
            if parsed is not None:
                return parsed
            if allow_unknown:
                return normalized
        expected = ", ".join(item.value for item in cls)
        rendered = repr(value)
        if len(rendered) > 100:
            rendered = rendered[:97] + "..."
        raise ValueError(f"unknown memory kind {rendered}; expected one of: {expected}")


@dataclass(frozen=True)
class MemoryItem:
    id: int
    project_id: str | None
    kind: MemoryKind | str
    title: str
    content: str
    tags: list[str]
    updated_at: str
    confidence: float = 0.7
    use_count: int = 0
    last_used_at: str | None = None
    expires_at: str | None = None
    merged_into: int | None = None


@dataclass(frozen=True)
class MemoryStats:
    total: int
    by_scope: dict[str, int]
    by_kind: dict[str, int]
    by_tag: dict[str, int]


class MemoryStore:
    def __init__(self, config: AppConfig, db_path: Path | None = None) -> None:
        self.config = config
        self.data_dir = config.data_dir
        configured_db = Path(str(config.get("memory.sqlite_path", paths.memory_db_path()))).expanduser()
        configured_vector = Path(str(config.get("memory.vector_path", paths.vector_dir()))).expanduser()
        self.db_path = db_path or configured_db
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector = OptionalChromaStore(
            configured_vector,
            enabled=bool(config.get("memory.vector_enabled", False)),
        )
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("pragma busy_timeout = 30000")
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute("pragma journal_mode = wal")
            con.executescript(
                """
                create table if not exists projects (
                    project_id text primary key,
                    name text not null,
                    root_path text not null,
                    language text,
                    updated_at text not null
                );

                create table if not exists documents (
                    id integer primary key autoincrement,
                    project_id text,
                    path text not null,
                    content text not null,
                    summary text,
                    tags text not null default '[]',
                    updated_at text not null
                );

                create table if not exists memories (
                    id integer primary key autoincrement,
                    project_id text,
                    kind text not null,
                    title text not null,
                    content text not null,
                    tags text not null default '[]',
                    created_at text not null,
                    updated_at text not null
                );

                create table if not exists summaries (
                    id integer primary key autoincrement,
                    project_id text,
                    scope text not null,
                    content text not null,
                    updated_at text not null
                );

                create table if not exists embeddings (
                    id integer primary key autoincrement,
                    memory_id integer,
                    document_id integer,
                    provider text,
                    vector_id text,
                    updated_at text not null
                );

                create table if not exists pipeline_runs (
                    run_id text primary key,
                    project_id text,
                    summary_memory_id integer,
                    experience_memory_id integer,
                    processed_at text not null
                );

                create table if not exists memory_usage_events (
                    usage_id text primary key,
                    run_id text not null,
                    project_id text,
                    memory_ids text not null,
                    recorded_at text not null
                );

                create table if not exists memory_feedback_events (
                    feedback_id text primary key,
                    memory_id integer not null,
                    helpful integer not null check (helpful in (0, 1)),
                    confidence_before real not null,
                    confidence_after real not null,
                    recorded_at text not null
                );
                """
            )
            if self._fts_available(con):
                con.executescript(
                    """
                    create virtual table if not exists memory_fts using fts5(
                        title,
                        content,
                        tags,
                        content='memories',
                        content_rowid='id'
                    );

                    create trigger if not exists memories_ai after insert on memories begin
                        insert into memory_fts(rowid, title, content, tags)
                        values (new.id, new.title, new.content, new.tags);
                    end;

                    create trigger if not exists memories_ad after delete on memories begin
                        insert into memory_fts(memory_fts, rowid, title, content, tags)
                        values ('delete', old.id, old.title, old.content, old.tags);
                    end;

                    create trigger if not exists memories_au after update on memories begin
                        insert into memory_fts(memory_fts, rowid, title, content, tags)
                        values ('delete', old.id, old.title, old.content, old.tags);
                        insert into memory_fts(rowid, title, content, tags)
                        values (new.id, new.title, new.content, new.tags);
                    end;
                    """
                )
            self._ensure_memory_columns(con)
            if self._table_exists(con, "memory_fts"):
                memory_count = int(con.execute("select count(*) from memories").fetchone()[0])
                fts_count = int(con.execute("select count(*) from memory_fts").fetchone()[0])
                if memory_count != fts_count:
                    con.execute("insert into memory_fts(memory_fts) values ('rebuild')")

    @staticmethod
    def _ensure_memory_columns(con: sqlite3.Connection) -> None:
        existing = {str(row[1]) for row in con.execute("pragma table_info(memories)").fetchall()}
        columns = {
            "confidence": "real not null default 0.7",
            "use_count": "integer not null default 0",
            "last_used_at": "text",
            "expires_at": "text",
            "merged_into": "integer",
        }
        for name, definition in columns.items():
            if name not in existing:
                con.execute(f"alter table memories add column {name} {definition}")

    @staticmethod
    def _fts_available(con: sqlite3.Connection) -> bool:
        try:
            con.execute("create virtual table if not exists _fts_probe using fts5(x)")
            con.execute("drop table if exists _fts_probe")
            return True
        except sqlite3.Error:
            return False

    def sync_project(self, project: Project) -> None:
        with self._connect() as con:
            con.execute(
                """
                insert into projects(project_id, name, root_path, language, updated_at)
                values (?, ?, ?, ?, ?)
                on conflict(project_id) do update set
                    name = excluded.name,
                    root_path = excluded.root_path,
                    language = excluded.language,
                    updated_at = excluded.updated_at
                """,
                (project.id, project.name, str(project.root), project.language, utc_now_iso()),
            )

    def add_memory(
        self,
        *,
        kind: str | MemoryKind,
        title: str,
        content: str,
        tags: Iterable[str] = (),
        project_id: str | None = None,
        confidence: float | None = None,
        expires_at: str | None = None,
    ) -> int:
        parsed_kind = MemoryKind.parse(kind)
        if not isinstance(parsed_kind, MemoryKind):  # strict parse is intentionally fail closed
            raise ValueError(f"invalid memory kind: {kind!r}")
        kind_value = parsed_kind.value
        now = utc_now_iso()
        tag_list = list(tags)
        tags_json = json.dumps(tag_list, ensure_ascii=False)
        confidence_value = min(
            1.0,
            max(
                0.0, float(confidence if confidence is not None else self.config.get("memory.default_confidence", 0.7))
            ),
        )
        protected = self._protected_kinds()
        if expires_at is None and kind_value not in protected:
            expiry_days = max(0, int(self.config.get("memory.expiry_days", 365)))
            if expiry_days:
                expires_at = (datetime.now(UTC) + timedelta(days=expiry_days)).replace(microsecond=0).isoformat()
        with self._connect() as con:
            cur = con.execute(
                """
                insert into memories(
                    project_id, kind, title, content, tags, created_at, updated_at, confidence, expires_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (project_id, kind_value, title, content, tags_json, now, now, confidence_value, expires_at),
            )
            memory_id = int(cur.lastrowid)
        if self.vector.is_enabled():
            self.vector.upsert_memory(
                memory_id=memory_id,
                project_id=project_id,
                kind=kind_value,
                title=title,
                content=content,
                tags=tag_list,
            )
        return memory_id

    def get_memory(self, memory_id: int) -> MemoryItem | None:
        with self._connect() as con:
            row = con.execute("select * from memories where id = ?", (memory_id,)).fetchone()
        return self._row_to_memory(row) if row else None

    def list_memories(
        self,
        *,
        project_id: str | None = None,
        limit: int = 50,
        kind: str | None = None,
        tag: str | None = None,
        global_only: bool = False,
    ) -> list[MemoryItem]:
        clauses: list[str] = ["merged_into is null"]
        params: list[object] = []
        if global_only:
            clauses.append("project_id is null")
        elif project_id is not None:
            clauses.append("(project_id = ? or project_id is null)")
            params.append(project_id)
        if kind:
            clauses.append("lower(kind) = lower(?)")
            params.append(kind)
        if tag:
            clauses.append("tags like ?")
            params.append(f'%"{tag}"%')
        where = " where " + " and ".join(clauses) if clauses else ""
        params.append(max(1, min(limit, 1000)))
        with self._connect() as con:
            rows = con.execute(
                f"select * from memories{where} order by updated_at desc, id desc limit ?",
                params,
            ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def update_memory(
        self,
        memory_id: int,
        *,
        title: str | None = None,
        content: str | None = None,
        tags: Iterable[str] | None = None,
        confidence: float | None = None,
        expires_at: str | None = None,
    ) -> MemoryItem:
        current = self.get_memory(memory_id)
        if current is None:
            raise KeyError(f"memory not found: {memory_id}")
        if current.merged_into is not None:
            raise ValueError(f"memory {memory_id} was merged into memory {current.merged_into}")
        next_title = current.title if title is None else title.strip()
        next_content = current.content if content is None else content.strip()
        next_tags = current.tags if tags is None else self._normalize_tags(tags)
        next_confidence = current.confidence if confidence is None else min(1.0, max(0.0, float(confidence)))
        next_expires = current.expires_at if expires_at is None else expires_at
        if not next_title or not next_content:
            raise ValueError("memory title and content must not be empty")
        with self._connect() as con:
            con.execute(
                """
                update memories
                set title = ?, content = ?, tags = ?, confidence = ?, expires_at = ?, updated_at = ?
                where id = ?
                """,
                (
                    next_title,
                    next_content,
                    json.dumps(next_tags, ensure_ascii=False),
                    next_confidence,
                    next_expires,
                    utc_now_iso(),
                    memory_id,
                ),
            )
        updated = self.get_memory(memory_id)
        if updated is None:
            raise RuntimeError(f"memory disappeared during update: {memory_id}")
        if self.vector.is_enabled():
            self.vector.upsert_memory(
                memory_id=updated.id,
                project_id=updated.project_id,
                kind=updated.kind,
                title=updated.title,
                content=updated.content,
                tags=updated.tags,
            )
        return updated

    def maintain(self, *, project_id: str | None, apply: bool = False) -> dict[str, Any]:
        items = self.list_memories(project_id=project_id, limit=1000)
        candidates = [item for item in items if item.kind in {"Correction", "Lesson", "Reflection"}]
        threshold = float(self.config.get("memory.dedupe_similarity", 0.94))
        merges: list[dict[str, Any]] = []
        groups = self._duplicate_groups(candidates, threshold)
        for group in groups:
            preferred = max(group, key=self._memory_preference)
            for duplicate in group:
                if duplicate.id == preferred.id:
                    continue
                merges.append(
                    {
                        "keep": preferred.id,
                        "merge": duplicate.id,
                        "similarity": round(self._memory_similarity(preferred, duplicate), 4),
                        "kind": preferred.kind,
                    }
                )
                if apply:
                    self._merge_memory(preferred, duplicate)
                    preferred = self.get_memory(preferred.id) or preferred

        expired: list[int] = []
        protected = self._protected_kinds()
        now = datetime.now(UTC)
        for item in items:
            expires_at = self._parse_timestamp(item.expires_at)
            if expires_at and expires_at <= now and item.kind not in protected and item.confidence < 0.5:
                expired.append(item.id)
                if apply:
                    self.delete_memory(item.id)
        # Capacity eviction is deliberately preview-only here.  Existing daemon
        # callers use maintain(apply=True) for duplicate/expiry cleanup; silently
        # extending that flag to a new deletion policy would not be an explicit
        # opt-in.  Call maintain_capacity(..., apply=True) to enforce the plan.
        capacity = self.maintain_capacity(project_id=project_id)
        return {
            "apply": apply,
            "scanned": len(items),
            "merges": merges,
            "expired": expired,
            "merge_count": len(merges),
            "expired_count": len(expired),
            "capacity": capacity,
        }

    def maintain_capacity(self, *, project_id: str | None = None, apply: bool = False) -> dict[str, Any]:
        """Plan or explicitly apply bounded Memory capacity eviction.

        Capacity is measured from canonical text payload columns using SQLite's
        UTF-8 byte representation.  The aggregate count does not materialize
        records in Python, while eviction candidates are always scan-bounded.
        Legacy unknown kinds, Corrections, Decisions, and configured protected
        kinds are never candidates.
        """

        max_items = self._bounded_int(
            self.config.get("memory.max_items", 5000),
            default=5000,
            minimum=1,
            maximum=1_000_000,
        )
        max_storage_mb = self._bounded_float(
            self.config.get("memory.max_storage_mb", 100),
            default=100.0,
            minimum=0.000_001,
            maximum=10_240.0,
        )
        max_payload_bytes = max(1, int(max_storage_mb * 1024 * 1024))
        scan_limit = self._bounded_int(
            self.config.get("memory.capacity_scan_limit", 5000),
            default=5000,
            minimum=1,
            maximum=10_000,
        )
        report_limit = self._bounded_int(
            self.config.get("memory.capacity_report_limit", 100),
            default=100,
            minimum=1,
            maximum=1000,
        )
        protected = self._protected_kinds()
        scope_clause = "" if project_id is None else " where (project_id = ? or project_id is null)"
        scope_params: list[object] = [] if project_id is None else [project_id]

        with self._connect() as con:
            before_row = con.execute(
                f"""
                select count(*) as item_count,
                       coalesce(sum({_MEMORY_PAYLOAD_BYTES_SQL}), 0) as payload_bytes
                from memories{scope_clause}
                """,
                scope_params,
            ).fetchone()
            before_count = int(before_row["item_count"] if before_row else 0)
            before_payload_bytes = int(before_row["payload_bytes"] if before_row else 0)

            candidates: list[dict[str, Any]] = []
            if before_count > max_items or before_payload_bytes > max_payload_bytes:
                evictable_kinds = [item.value for item in MemoryKind if item.value not in protected]
                if evictable_kinds:
                    clauses = [f"m.kind in ({','.join('?' for _ in evictable_kinds)})"]
                    params = [*scope_params, *evictable_kinds, scan_limit]
                    if project_id is not None:
                        clauses.insert(0, "(m.project_id = ? or m.project_id is null)")
                    where = " where " + " and ".join(clauses)
                    rows = con.execute(
                        f"""
                        select m.id, m.kind, m.updated_at,
                               {_MEMORY_PAYLOAD_BYTES_SQL} as payload_bytes
                        from memories m
                        {where}
                          and not exists (select 1 from memories child where child.merged_into = m.id)
                        order by coalesce(m.confidence, 0.7) asc,
                                 coalesce(m.use_count, 0) asc,
                                 coalesce(m.last_used_at, m.updated_at, '') asc,
                                 m.id asc
                        limit ?
                        """,
                        params,
                    ).fetchall()
                    candidates = [
                        {
                            "id": int(row["id"]),
                            "kind": str(row["kind"]),
                            "updated_at": str(row["updated_at"]),
                            "payload_bytes": int(row["payload_bytes"] or 0),
                        }
                        for row in rows
                    ]

        planned: list[dict[str, Any]] = []
        projected_count = before_count
        projected_payload_bytes = before_payload_bytes
        for candidate in candidates:
            if projected_count <= max_items and projected_payload_bytes <= max_payload_bytes:
                break
            planned.append(candidate)
            projected_count -= 1
            projected_payload_bytes = max(0, projected_payload_bytes - int(candidate["payload_bytes"]))

        deleted_ids: list[int] = []
        if apply and planned:
            # Revalidate immutable plan evidence inside one transaction.  A row
            # edited since planning, newly protected, or used as a merge keeper
            # is skipped rather than force-deleted.
            with self._connect() as con:
                for candidate in planned:
                    kind = str(candidate["kind"])
                    parsed_kind = MemoryKind.parse(kind, allow_unknown=True)
                    if not isinstance(parsed_kind, MemoryKind) or parsed_kind.value in protected:
                        continue
                    deleted = con.execute(
                        """
                        delete from memories
                        where id = ? and kind = ? and updated_at = ?
                          and not exists (select 1 from memories child where child.merged_into = memories.id)
                        """,
                        (candidate["id"], kind, candidate["updated_at"]),
                    )
                    if deleted.rowcount == 1:
                        deleted_ids.append(int(candidate["id"]))
            if self.vector.is_enabled():
                for memory_id in deleted_ids:
                    self.vector.delete_memory(memory_id)

        if apply:
            after_count, after_payload_bytes = self._capacity_totals(project_id)
        else:
            after_count, after_payload_bytes = projected_count, projected_payload_bytes
        unresolved_count = max(0, after_count - max_items)
        unresolved_payload_bytes = max(0, after_payload_bytes - max_payload_bytes)
        eviction_ids = [int(candidate["id"]) for candidate in planned]
        return {
            "apply": apply,
            "scope": project_id or "all",
            "max_items": max_items,
            "max_payload_bytes": max_payload_bytes,
            "current_count": before_count,
            "current_payload_bytes": before_payload_bytes,
            "payload_columns": ["kind", "title", "content", "tags"],
            "scan_limit": scan_limit,
            "scanned": len(candidates),
            "protected_kinds": sorted(protected),
            "eviction_count": len(planned),
            "eviction_payload_bytes": sum(int(candidate["payload_bytes"]) for candidate in planned),
            "eviction_ids": eviction_ids[:report_limit],
            "eviction_ids_truncated": len(eviction_ids) > report_limit,
            "deleted_count": len(deleted_ids),
            "deleted_ids": deleted_ids[:report_limit],
            "deleted_ids_truncated": len(deleted_ids) > report_limit,
            "projected_count": after_count,
            "projected_payload_bytes": after_payload_bytes,
            "unresolved_count": unresolved_count,
            "unresolved_payload_bytes": unresolved_payload_bytes,
            "complete": unresolved_count == 0 and unresolved_payload_bytes == 0,
        }

    def _capacity_totals(self, project_id: str | None) -> tuple[int, int]:
        scope_clause = "" if project_id is None else " where (project_id = ? or project_id is null)"
        params: tuple[object, ...] = () if project_id is None else (project_id,)
        with self._connect() as con:
            row = con.execute(
                f"""
                select count(*) as item_count,
                       coalesce(sum({_MEMORY_PAYLOAD_BYTES_SQL}), 0) as payload_bytes
                from memories{scope_clause}
                """,
                params,
            ).fetchone()
        return (int(row["item_count"] if row else 0), int(row["payload_bytes"] if row else 0))

    def _merge_memory(self, keeper: MemoryItem, duplicate: MemoryItem) -> None:
        tags = self._normalize_tags([*keeper.tags, *duplicate.tags])
        content = keeper.content if len(keeper.content) >= len(duplicate.content) else duplicate.content
        confidence = max(keeper.confidence, duplicate.confidence)
        with self._connect() as con:
            con.execute(
                """
                update memories
                set content = ?, tags = ?, confidence = ?, use_count = ?, last_used_at = ?, updated_at = ?
                where id = ?
                """,
                (
                    content,
                    json.dumps(tags, ensure_ascii=False),
                    confidence,
                    keeper.use_count + duplicate.use_count,
                    max(keeper.last_used_at or "", duplicate.last_used_at or "") or None,
                    utc_now_iso(),
                    keeper.id,
                ),
            )
            con.execute(
                "update memories set merged_into = ?, updated_at = ? where id = ?",
                (keeper.id, utc_now_iso(), duplicate.id),
            )
        updated = self.get_memory(keeper.id)
        if updated and self.vector.is_enabled():
            self.vector.upsert_memory(
                memory_id=updated.id,
                project_id=updated.project_id,
                kind=updated.kind,
                title=updated.title,
                content=updated.content,
                tags=updated.tags,
            )
            self.vector.delete_memory(duplicate.id)

    @staticmethod
    def _preferred_memory(first: MemoryItem, second: MemoryItem) -> tuple[MemoryItem, MemoryItem]:
        score_first = MemoryStore._memory_preference(first)
        score_second = MemoryStore._memory_preference(second)
        return (first, second) if score_first >= score_second else (second, first)

    @staticmethod
    def _memory_preference(item: MemoryItem) -> tuple[float, int, int, int]:
        return (item.confidence, item.use_count, len(item.content), -item.id)

    @classmethod
    def _duplicate_groups(cls, items: list[MemoryItem], threshold: float) -> list[list[MemoryItem]]:
        partitions: dict[tuple[str, str | None], list[MemoryItem]] = {}
        for item in items:
            partitions.setdefault((item.kind, item.project_id), []).append(item)
        groups: list[list[MemoryItem]] = []
        for candidates in partitions.values():
            ordered = sorted(candidates, key=cls._memory_preference, reverse=True)
            item_by_id = {item.id: item for item in ordered}
            # Small partitions are cheap enough to compare exactly. Larger ones use
            # character shingles instead of whole-token buckets: this preserves
            # recall for near-identical identifiers/typos while common prose words
            # no longer make every record a candidate for every other record.
            exact_partition = len(ordered) <= 200
            shingle_sets = {item.id: cls._memory_shingles(item) for item in ordered}
            inverted: dict[str, set[int]] = {}
            if not exact_partition:
                for item in ordered:
                    for shingle in shingle_sets[item.id]:
                        inverted.setdefault(shingle, set()).add(item.id)
            remaining_ids = {item.id for item in ordered}
            for keeper in ordered:
                if keeper.id not in remaining_ids:
                    continue
                if exact_partition:
                    candidates_ids = set(remaining_ids)
                else:
                    overlap_counts: dict[int, int] = {}
                    for shingle in shingle_sets[keeper.id]:
                        for item_id in inverted.get(shingle, ()):
                            if item_id in remaining_ids and item_id != keeper.id:
                                overlap_counts[item_id] = overlap_counts.get(item_id, 0) + 1
                    candidates_ids = {
                        item_id
                        for item_id, overlap in overlap_counts.items()
                        if cls._shingle_candidate(
                            overlap,
                            len(shingle_sets[keeper.id]),
                            len(shingle_sets[item_id]),
                            threshold,
                        )
                    }
                candidates_ids.discard(keeper.id)
                duplicates = [
                    item_by_id[item_id]
                    for item_id in candidates_ids & remaining_ids
                    if cls._memory_similarity(keeper, item_by_id[item_id]) >= threshold
                ]
                if duplicates:
                    duplicates.sort(key=cls._memory_preference, reverse=True)
                    groups.append([keeper, *duplicates])
                    remaining_ids.difference_update(item.id for item in duplicates)
                remaining_ids.discard(keeper.id)
        return groups

    @staticmethod
    def _memory_shingles(item: MemoryItem, size: int = 8) -> set[str]:
        value = f"{item.title}\n{item.content}".lower()
        normalized = " ".join("".join(ch if ch.isalnum() else " " for ch in value).split())
        if len(normalized) <= size:
            return {normalized} if normalized else set()
        return {normalized[index : index + size] for index in range(len(normalized) - size + 1)}

    @staticmethod
    def _shingle_candidate(overlap: int, left_size: int, right_size: int, threshold: float) -> bool:
        if overlap <= 0 or not left_size or not right_size:
            return False
        # SequenceMatcher ratios near the configured duplicate threshold imply
        # substantial common substrings. A conservative overlap floor keeps those
        # pairs for exact comparison while discarding incidental shared prose.
        minimum_ratio = max(0.15, threshold - 0.4)
        return overlap / min(left_size, right_size) >= minimum_ratio

    @staticmethod
    def _parse_timestamp(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)

    @staticmethod
    def _bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
        if isinstance(value, bool):
            return default
        try:
            parsed = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError, OverflowError):
            return default
        return min(maximum, max(minimum, parsed))

    @staticmethod
    def _bounded_float(value: object, *, default: float, minimum: float, maximum: float) -> float:
        if isinstance(value, bool):
            return default
        try:
            parsed = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError, OverflowError):
            return default
        if not math.isfinite(parsed):
            return default
        return min(maximum, max(minimum, parsed))

    @staticmethod
    def _memory_similarity(first: MemoryItem, second: MemoryItem) -> float:
        def normalized(item: MemoryItem) -> str:
            value = f"{item.title}\n{item.content}".lower()
            return " ".join("".join(ch if ch.isalnum() else " " for ch in value).split())

        left, right = normalized(first), normalized(second)
        if hashlib.sha256(left.encode()).digest() == hashlib.sha256(right.encode()).digest():
            return 1.0
        return difflib.SequenceMatcher(None, left, right, autojunk=False).ratio()

    def delete_memory(self, memory_id: int) -> bool:
        with self._connect() as con:
            ids = [
                int(row[0])
                for row in con.execute(
                    "select id from memories where id = ? or merged_into = ?", (memory_id, memory_id)
                ).fetchall()
            ]
            deleted = (
                con.execute("delete from memories where id = ? or merged_into = ?", (memory_id, memory_id)).rowcount > 0
            )
        if deleted and self.vector.is_enabled():
            for item_id in ids:
                self.vector.delete_memory(item_id)
        return deleted

    def stats(self, *, project_id: str | None = None) -> MemoryStats:
        clauses = (
            " where merged_into is null and (project_id = ? or project_id is null)"
            if project_id is not None
            else " where merged_into is null"
        )
        params = (project_id,) if project_id is not None else ()
        with self._connect() as con:
            rows = con.execute(f"select project_id, kind, tags from memories{clauses}", params).fetchall()
        by_scope: dict[str, int] = {}
        by_kind: dict[str, int] = {}
        by_tag: dict[str, int] = {}
        for row in rows:
            scope = "global" if row["project_id"] is None else "project"
            by_scope[scope] = by_scope.get(scope, 0) + 1
            kind = str(row["kind"])
            by_kind[kind] = by_kind.get(kind, 0) + 1
            try:
                tags = json.loads(row["tags"] or "[]")
            except json.JSONDecodeError:
                tags = []
            for tag in tags if isinstance(tags, list) else []:
                value = str(tag)
                by_tag[value] = by_tag.get(value, 0) + 1
        return MemoryStats(len(rows), by_scope, by_kind, by_tag)

    def search_recovery(self, error_text: str, project_id: str, limit: int = 4) -> list[MemoryItem]:
        tokens = self._recovery_tokens(error_text)
        if not tokens:
            return []
        clauses = " or ".join("lower(title || ' ' || content || ' ' || tags) like ?" for _ in tokens)
        params: list[object] = [project_id, *[f"%{token.lower()}%" for token in tokens], max(1, limit)]
        with self._connect() as con:
            rows = con.execute(
                f"""
                select * from memories
                where (project_id = ? or project_id is null)
                  and kind in ('Correction', 'Lesson')
                  and merged_into is null
                  and ({clauses})
                order by case kind when 'Correction' then 0 else 1 end, updated_at desc
                limit ?
                """,
                params,
            ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def add_document(
        self,
        *,
        path: str,
        content: str,
        summary: str | None = None,
        tags: Iterable[str] = (),
        project_id: str | None = None,
    ) -> int:
        now = utc_now_iso()
        with self._connect() as con:
            cur = con.execute(
                """
                insert into documents(project_id, path, content, summary, tags, updated_at)
                values (?, ?, ?, ?, ?, ?)
                """,
                (project_id, path, content, summary, json.dumps(list(tags), ensure_ascii=False), now),
            )
            return int(cur.lastrowid)

    def update_summary(self, *, scope: str, content: str, project_id: str | None = None) -> None:
        now = utc_now_iso()
        with self._connect() as con:
            row = con.execute(
                "select id from summaries where project_id is ? and scope = ? order by id desc limit 1",
                (project_id, scope),
            ).fetchone()
            if row:
                con.execute(
                    "update summaries set content = ?, updated_at = ? where id = ?",
                    (content, now, row["id"]),
                )
            else:
                con.execute(
                    "insert into summaries(project_id, scope, content, updated_at) values (?, ?, ?, ?)",
                    (project_id, scope, content, now),
                )

    def is_pipeline_run_processed(self, run_id: str) -> bool:
        with self._connect() as con:
            return con.execute("select 1 from pipeline_runs where run_id = ?", (run_id,)).fetchone() is not None

    def mark_pipeline_run_processed(
        self,
        run_id: str,
        project_id: str | None,
        summary_memory_id: int | None,
        experience_memory_id: int | None,
    ) -> None:
        with self._connect() as con:
            con.execute(
                """
                insert into pipeline_runs(
                    run_id, project_id, summary_memory_id, experience_memory_id, processed_at
                ) values (?, ?, ?, ?, ?)
                on conflict(run_id) do update set
                    project_id = excluded.project_id,
                    summary_memory_id = excluded.summary_memory_id,
                    experience_memory_id = excluded.experience_memory_id,
                    processed_at = excluded.processed_at
                """,
                (run_id, project_id, summary_memory_id, experience_memory_id, utc_now_iso()),
            )

    def search(
        self,
        query: str,
        project_id: str | None = None,
        limit: int | None = None,
        *,
        global_only: bool = False,
        record_usage: bool = True,
    ) -> list[MemoryItem]:
        limit = limit or int(self.config.get("memory.retrieval_limit", 8))
        with self._connect() as con:
            if self._table_exists(con, "memory_fts") and query.strip():
                if global_only:
                    rows = con.execute(
                        """
                        select m.*
                        from memory_fts f
                        join memories m on m.id = f.rowid
                        where memory_fts match ? and m.project_id is null and m.merged_into is null
                        order by bm25(memory_fts)
                        limit ?
                        """,
                        (self._safe_fts_query(query), limit),
                    ).fetchall()
                else:
                    rows = con.execute(
                        """
                        select m.*
                        from memory_fts f
                        join memories m on m.id = f.rowid
                        where memory_fts match ?
                          and (? is null or m.project_id = ? or m.project_id is null)
                          and m.merged_into is null
                        order by bm25(memory_fts)
                        limit ?
                        """,
                        (self._safe_fts_query(query), project_id, project_id, limit),
                    ).fetchall()
            else:
                like = f"%{query}%"
                if global_only:
                    rows = con.execute(
                        """
                        select *
                        from memories
                        where (? = '' or title like ? or content like ? or tags like ?)
                          and project_id is null
                          and merged_into is null
                        order by updated_at desc
                        limit ?
                        """,
                        (query, like, like, like, limit),
                    ).fetchall()
                else:
                    rows = con.execute(
                        """
                        select *
                        from memories
                        where (? = '' or title like ? or content like ? or tags like ?)
                          and (? is null or project_id = ? or project_id is null)
                          and merged_into is null
                        order by updated_at desc
                        limit ?
                        """,
                        (query, like, like, like, project_id, project_id, limit),
                    ).fetchall()
            items = [self._row_to_memory(row) for row in rows]
            seen = {item.id for item in items}
            if len(items) < limit and self.vector.is_enabled() and query.strip() and not global_only:
                vector_ids = self.vector.query_memory_ids(query=query, project_id=project_id, limit=limit)
                missing_ids = [memory_id for memory_id in vector_ids if memory_id not in seen]
                if missing_ids:
                    placeholders = ",".join("?" for _ in missing_ids)
                    extra_rows = con.execute(
                        f"select * from memories where merged_into is null and id in ({placeholders})",
                        missing_ids,
                    ).fetchall()
                    for row in extra_rows:
                        item = self._row_to_memory(row)
                        if item.id not in seen:
                            items.append(item)
                            seen.add(item.id)
                        if len(items) >= limit:
                            break
        selected = items[:limit]
        if record_usage:
            self._record_usage([item.id for item in selected])
        return selected

    def record_usage(self, memory_ids: Iterable[int]) -> None:
        """Reinforce only Memory entries that were actually included in model context."""
        self._record_usage([int(memory_id) for memory_id in memory_ids])

    def record_usage_once(
        self,
        usage_id: str,
        memory_ids: Iterable[int],
        *,
        run_id: str,
        project_id: str | None,
    ) -> bool:
        """Atomically record one context-inclusion batch exactly once per usage ID."""

        normalized_usage_id = str(usage_id).strip()
        normalized_run_id = str(run_id).strip()
        normalized_ids = sorted(set(int(memory_id) for memory_id in memory_ids))
        if not normalized_usage_id or len(normalized_usage_id) > 500:
            raise ValueError("memory usage_id must contain 1 to 500 characters")
        if not normalized_run_id or len(normalized_run_id) > 500:
            raise ValueError("memory run_id must contain 1 to 500 characters")
        if not normalized_ids or len(normalized_ids) > 1000 or any(memory_id <= 0 for memory_id in normalized_ids):
            raise ValueError("memory usage requires 1 to 1000 positive IDs")
        serialized_ids = json.dumps(normalized_ids, separators=(",", ":"))
        now = utc_now_iso()
        with self._connect() as con:
            inserted = con.execute(
                """
                insert into memory_usage_events(usage_id, run_id, project_id, memory_ids, recorded_at)
                values (?, ?, ?, ?, ?)
                on conflict(usage_id) do nothing
                """,
                (normalized_usage_id, normalized_run_id, project_id, serialized_ids, now),
            )
            if inserted.rowcount == 0:
                existing = con.execute(
                    "select run_id, project_id, memory_ids from memory_usage_events where usage_id = ?",
                    (normalized_usage_id,),
                ).fetchone()
                if existing is None:
                    raise RuntimeError("memory usage journal conflict could not be resolved")
                if (
                    str(existing["run_id"]) != normalized_run_id
                    or existing["project_id"] != project_id
                    or str(existing["memory_ids"]) != serialized_ids
                ):
                    raise ValueError("memory usage_id was replayed with different evidence")
                return False
            placeholders = ",".join("?" for _ in normalized_ids)
            updated = con.execute(
                f"""
                update memories
                set use_count = use_count + 1, last_used_at = ?
                where id in ({placeholders}) and merged_into is null
                """,
                [now, *normalized_ids],
            )
            if updated.rowcount != len(normalized_ids):
                raise ValueError("memory usage evidence contains missing or merged IDs")
        return True

    def record_feedback(self, feedback_id: str, memory_id: int, *, helpful: bool) -> bool:
        """Apply explicit confidence feedback exactly once per feedback ID.

        Retrieval and context inclusion are intentionally not treated as proof
        of correctness; ``record_usage`` only updates usage metadata.  A caller
        must provide an explicit helpful/not-helpful outcome here.
        """

        normalized_feedback_id = str(feedback_id).strip()
        normalized_memory_id = int(memory_id)
        if (
            not normalized_feedback_id
            or len(normalized_feedback_id) > 500
            or len(normalized_feedback_id.encode("utf-8")) > 2000
        ):
            raise ValueError("memory feedback_id must contain 1 to 500 characters and at most 2000 UTF-8 bytes")
        if normalized_memory_id <= 0:
            raise ValueError("memory feedback requires a positive memory ID")
        if not isinstance(helpful, bool):
            raise ValueError("memory feedback helpful must be a boolean")

        bonus = self._bounded_float(
            self.config.get("memory.confidence.use_bonus", 0.02),
            default=0.02,
            minimum=0.0,
            maximum=0.25,
        )
        penalty = self._bounded_float(
            self.config.get("memory.confidence.contradiction_penalty", 0.15),
            default=0.15,
            minimum=0.0,
            maximum=0.5,
        )
        lower_bound = self._bounded_float(
            self.config.get("memory.confidence.lower_bound", 0.1),
            default=0.1,
            minimum=0.0,
            maximum=1.0,
        )
        upper_bound = self._bounded_float(
            self.config.get("memory.confidence.upper_bound", 0.95),
            default=0.95,
            minimum=0.0,
            maximum=1.0,
        )
        if lower_bound > upper_bound:
            lower_bound, upper_bound = 0.1, 0.95
        now = utc_now_iso()
        with self._connect() as con:
            existing = con.execute(
                "select memory_id, helpful from memory_feedback_events where feedback_id = ?",
                (normalized_feedback_id,),
            ).fetchone()
            if existing is not None:
                if int(existing["memory_id"]) != normalized_memory_id or bool(existing["helpful"]) != helpful:
                    raise ValueError("memory feedback_id was replayed with different evidence")
                return False
            row = con.execute(
                "select confidence, merged_into from memories where id = ?",
                (normalized_memory_id,),
            ).fetchone()
            if row is None:
                raise ValueError("memory feedback contains a missing memory ID")
            if row["merged_into"] is not None:
                raise ValueError("memory feedback contains a merged memory ID")

            before = min(1.0, max(0.0, float(row["confidence"] if row["confidence"] is not None else 0.7)))
            if helpful:
                after = max(before, min(upper_bound, before + bonus))
            else:
                after = min(before, max(lower_bound, before - penalty))
            inserted = con.execute(
                """
                insert into memory_feedback_events(
                    feedback_id, memory_id, helpful, confidence_before, confidence_after, recorded_at
                ) values (?, ?, ?, ?, ?, ?)
                on conflict(feedback_id) do nothing
                """,
                (normalized_feedback_id, normalized_memory_id, int(helpful), before, after, now),
            )
            if inserted.rowcount == 0:
                existing = con.execute(
                    """
                    select memory_id, helpful
                    from memory_feedback_events
                    where feedback_id = ?
                    """,
                    (normalized_feedback_id,),
                ).fetchone()
                if existing is None:
                    raise RuntimeError("memory feedback journal conflict could not be resolved")
                if int(existing["memory_id"]) != normalized_memory_id or bool(existing["helpful"]) != helpful:
                    raise ValueError("memory feedback_id was replayed with different evidence")
                return False
            updated = con.execute(
                """
                update memories
                set confidence = ?, updated_at = ?
                where id = ? and merged_into is null and confidence = ?
                """,
                (after, now, normalized_memory_id, row["confidence"]),
            )
            if updated.rowcount != 1:
                raise RuntimeError("memory changed while confidence feedback was being recorded")
        return True

    def _record_usage(self, memory_ids: list[int]) -> None:
        if not memory_ids:
            return
        placeholders = ",".join("?" for _ in memory_ids)
        with self._connect() as con:
            con.execute(
                f"update memories set use_count = use_count + 1, last_used_at = ? where id in ({placeholders})",
                [utc_now_iso(), *memory_ids],
            )

    def recent(self, project_id: str | None = None, limit: int = 10) -> list[MemoryItem]:
        with self._connect() as con:
            rows = con.execute(
                """
                select *
                from memories
                where (? is null or project_id = ? or project_id is null)
                  and merged_into is null
                order by updated_at desc
                limit ?
                """,
                (project_id, project_id, limit),
            ).fetchall()
        return [self._row_to_memory(row) for row in rows]

    def context_block(self, project: Project, query: str) -> str:
        items = self.search(query, project.id)
        return self.context_block_from_items(items)

    @staticmethod
    def context_block_from_items(items: Iterable[MemoryItem]) -> str:
        item_list = list(items)
        if not item_list:
            return "No relevant long-term memory found."
        parts = []
        for item in item_list:
            scope = "global" if item.project_id is None else "project"
            tags = ", ".join(item.tags)
            parts.append(
                f"- [{scope}/{item.kind}] {item.title}\n  tags: {tags or '-'}\n  {item.content.strip()[:1200]}"
            )
        return "\n".join(parts)

    def persist_lesson_file(
        self,
        *,
        kind: str | MemoryKind,
        title: str,
        content: str,
        project: Project | None,
        global_memory: bool = True,
    ) -> None:
        parsed_kind = MemoryKind.parse(kind)
        if not isinstance(parsed_kind, MemoryKind):  # strict parse is intentionally fail closed
            raise ValueError(f"invalid memory kind: {kind!r}")
        kind_value = parsed_kind.value
        base = self.data_dir / "memory" / kind_value.lower()
        if project and not global_memory:
            base = project.agent_dir / "memory" / kind_value.lower()
        base.mkdir(parents=True, exist_ok=True)
        stamp = utc_now_iso().replace(":", "-")
        filename = f"{stamp}-{slugify(title)}.md"
        (base / filename).write_text(content.strip() + "\n", encoding="utf-8")

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> MemoryItem:
        try:
            tags = json.loads(row["tags"] or "[]")
        except json.JSONDecodeError:
            tags = []
        return MemoryItem(
            id=int(row["id"]),
            project_id=row["project_id"],
            kind=MemoryKind.parse(row["kind"], allow_unknown=True),
            title=row["title"],
            content=row["content"],
            tags=tags if isinstance(tags, list) else [],
            updated_at=row["updated_at"],
            confidence=float(row["confidence"] if row["confidence"] is not None else 0.7),
            use_count=int(row["use_count"] or 0),
            last_used_at=row["last_used_at"],
            expires_at=row["expires_at"],
            merged_into=int(row["merged_into"]) if row["merged_into"] is not None else None,
        )

    @staticmethod
    def _table_exists(con: sqlite3.Connection, table: str) -> bool:
        return (
            con.execute("select 1 from sqlite_master where type='table' and name = ?", (table,)).fetchone() is not None
        )

    @staticmethod
    def _safe_fts_query(query: str) -> str:
        tokens = [token.strip('"*:()') for token in query.split() if token.strip('"*:()')]
        return " OR ".join(f'"{token}"' for token in tokens) if tokens else '""'

    @staticmethod
    def _normalize_tags(tags: Iterable[str]) -> list[str]:
        values: list[str] = []
        seen: set[str] = set()
        for item in tags:
            tag = str(item).strip()
            if tag and tag not in seen:
                values.append(tag)
                seen.add(tag)
        return values

    def _protected_kinds(self) -> set[str]:
        # Corrections and Decisions remain protected even when an older or
        # incomplete configuration omits memory.protect_kinds.
        protected = {MemoryKind.CORRECTION.value, MemoryKind.DECISION.value}
        configured = self.config.get("memory.protect_kinds", ())
        if not isinstance(configured, (list, tuple, set, frozenset)):
            return protected
        for value in configured:
            raw = str(value).strip()
            if not raw:
                continue
            parsed = MemoryKind.parse(raw, allow_unknown=True)
            protected.add(parsed.value if isinstance(parsed, MemoryKind) else parsed)
        return protected

    @staticmethod
    def _recovery_tokens(value: str) -> list[str]:
        cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else " " for ch in value)
        ignored = {"error", "failed", "failure", "false", "tool", "command", "with", "from", "this", "that"}
        tokens: list[str] = []
        for token in cleaned.split():
            normalized = token.strip().lower()
            if len(normalized) < 3 or normalized in ignored or normalized in tokens:
                continue
            tokens.append(normalized)
            if len(tokens) >= 8:
                break
        return tokens


def slugify(value: str) -> str:
    chars = []
    for ch in value.lower().strip():
        if ch.isalnum():
            chars.append(ch)
        elif ch in {" ", "-", "_", "."}:
            chars.append("-")
    slug = "".join(chars).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug[:80] or "memory"
