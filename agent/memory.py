from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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


@dataclass(frozen=True)
class MemoryStats:
    total: int
    by_scope: dict[str, int]
    by_kind: dict[str, int]
    by_tag: dict[str, int]


class MemoryStore:
    def __init__(self, config: AppConfig, db_path: Path | None = None) -> None:
        self.config = config
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
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
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
    ) -> int:
        now = utc_now_iso()
        tag_list = list(tags)
        tags_json = json.dumps(tag_list, ensure_ascii=False)
        with self._connect() as con:
            cur = con.execute(
                """
                insert into memories(project_id, kind, title, content, tags, created_at, updated_at)
                values (?, ?, ?, ?, ?, ?, ?)
                """,
                (project_id, kind, title, content, tags_json, now, now),
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
        clauses: list[str] = []
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
    ) -> MemoryItem:
        current = self.get_memory(memory_id)
        if current is None:
            raise KeyError(f"memory not found: {memory_id}")
        next_title = current.title if title is None else title.strip()
        next_content = current.content if content is None else content.strip()
        next_tags = current.tags if tags is None else self._normalize_tags(tags)
        if not next_title or not next_content:
            raise ValueError("memory title and content must not be empty")
        with self._connect() as con:
            con.execute(
                "update memories set title = ?, content = ?, tags = ?, updated_at = ? where id = ?",
                (
                    next_title,
                    next_content,
                    json.dumps(next_tags, ensure_ascii=False),
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

    def delete_memory(self, memory_id: int) -> bool:
        with self._connect() as con:
            deleted = con.execute("delete from memories where id = ?", (memory_id,)).rowcount > 0
        if deleted and self.vector.is_enabled():
            self.vector.delete_memory(memory_id)
        return deleted

    def stats(self, *, project_id: str | None = None) -> MemoryStats:
        clauses = " where project_id = ? or project_id is null" if project_id is not None else ""
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
                        where memory_fts match ? and m.project_id is null
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
                        f"select * from memories where id in ({placeholders})",
                        missing_ids,
                    ).fetchall()
                    for row in extra_rows:
                        item = self._row_to_memory(row)
                        if item.id not in seen:
                            items.append(item)
                            seen.add(item.id)
                        if len(items) >= limit:
                            break
        return items[:limit]

    def recent(self, project_id: str | None = None, limit: int = 10) -> list[MemoryItem]:
        with self._connect() as con:
            rows = con.execute(
                """
                select *
                from memories
                where ? is null or project_id = ? or project_id is null
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
        base = paths.memory_dir() / kind.lower()
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
