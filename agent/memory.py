from __future__ import annotations

import json
import difflib
import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from . import paths
from .config import AppConfig
from .project import Project
from .timeutil import utc_now_iso
from .vector import OptionalChromaStore


@dataclass(frozen=True)
class MemoryItem:
    id: int
    project_id: str | None
    kind: str
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
            enabled=bool(config.get("memory.vector_enabled", True)),
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
        kind: str,
        title: str,
        content: str,
        tags: Iterable[str] = (),
        project_id: str | None = None,
        confidence: float | None = None,
        expires_at: str | None = None,
    ) -> int:
        now = utc_now_iso()
        tag_list = list(tags)
        tags_json = json.dumps(tag_list, ensure_ascii=False)
        confidence_value = min(
            1.0,
            max(
                0.0, float(confidence if confidence is not None else self.config.get("memory.default_confidence", 0.7))
            ),
        )
        protected = {str(item) for item in self.config.get("memory.protect_kinds", ["Correction", "Decision"])}
        if expires_at is None and kind not in protected:
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
                (project_id, kind, title, content, tags_json, now, now, confidence_value, expires_at),
            )
            memory_id = int(cur.lastrowid)
        if self.vector.is_enabled():
            self.vector.upsert_memory(
                memory_id=memory_id,
                project_id=project_id,
                kind=kind,
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
        protected = {str(item) for item in self.config.get("memory.protect_kinds", ["Correction", "Decision"])}
        now = datetime.now(UTC)
        for item in items:
            expires_at = self._parse_timestamp(item.expires_at)
            if expires_at and expires_at <= now and item.kind not in protected and item.confidence < 0.5:
                expired.append(item.id)
                if apply:
                    self.delete_memory(item.id)
        return {
            "apply": apply,
            "scanned": len(items),
            "merges": merges,
            "expired": expired,
            "merge_count": len(merges),
            "expired_count": len(expired),
        }

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
        kind: str,
        title: str,
        content: str,
        project: Project | None,
        global_memory: bool = True,
    ) -> None:
        base = self.data_dir / "memory" / kind.lower()
        if project and not global_memory:
            base = project.agent_dir / "memory" / kind.lower()
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
            kind=row["kind"],
            title=row["title"],
            content=row["content"],
            tags=tags if isinstance(tags, list) else [],
            updated_at=row["updated_at"],
            confidence=float(row["confidence"] or 0.7),
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
