from __future__ import annotations

import fcntl
import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from . import paths
from .config import AppConfig
from .timeutil import utc_now_iso


PROJECT_AGENT_DIR = ".project-agent"


@dataclass(frozen=True)
class Project:
    id: str
    name: str
    root: Path
    agent_dir: Path
    config_path: Path
    context_path: Path
    language: str


class ProjectRegistry:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or paths.projects_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                create table if not exists projects (
                    project_id text primary key,
                    name text not null,
                    root_path text not null,
                    language text,
                    created_at text not null,
                    last_opened text not null,
                    tags text not null default '[]',
                    context_path text not null
                )
                """
            )
            con.execute("create unique index if not exists idx_projects_root on projects(root_path)")

    def upsert(self, project: Project, tags: list[str] | None = None) -> None:
        now = utc_now_iso()
        with self._connect() as con:
            row = con.execute(
                "select created_at, tags from projects where project_id = ?",
                (project.id,),
            ).fetchone()
            created_at = row["created_at"] if row else now
            if tags is None and row:
                tags_json = row["tags"]
            else:
                tags_json = json.dumps(tags or [], ensure_ascii=False)
            con.execute(
                """
                insert into projects (
                    project_id, name, root_path, language, created_at, last_opened, tags, context_path
                ) values (?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(project_id) do update set
                    name = excluded.name,
                    root_path = excluded.root_path,
                    language = excluded.language,
                    last_opened = excluded.last_opened,
                    tags = excluded.tags,
                    context_path = excluded.context_path
                """,
                (
                    project.id,
                    project.name,
                    str(project.root),
                    project.language,
                    created_at,
                    now,
                    tags_json,
                    str(project.context_path),
                ),
            )

    def list_projects(self, limit: int = 50) -> list[sqlite3.Row]:
        with self._connect() as con:
            return list(
                con.execute(
                    """
                    select project_id, name, root_path, language, created_at, last_opened, tags, context_path
                    from projects
                    order by last_opened desc
                    limit ?
                    """,
                    (limit,),
                )
            )

    def get_by_root(self, root: Path) -> sqlite3.Row | None:
        with self._connect() as con:
            return con.execute(
                "select * from projects where root_path = ?",
                (str(root.resolve()),),
            ).fetchone()


class ProjectManager:
    def __init__(self, config: AppConfig, registry: ProjectRegistry | None = None) -> None:
        self.config = config
        self.registry = registry or ProjectRegistry(config.data_dir / "projects.db")
        self.agent_dir_name = str(config.get("project.agent_dir", PROJECT_AGENT_DIR))

    def resolve_project(self, start: Path | None = None) -> Project:
        cwd = (start or Path.cwd()).resolve()
        root = self._find_existing_root(cwd) or self._infer_new_root(cwd)
        agent_dir = root / self.agent_dir_name
        if agent_dir.is_symlink():
            raise ValueError(f"project Agent directory must not be a symbolic link: {agent_dir}")
        agent_dir.mkdir(parents=True, exist_ok=True)
        lock_handle = (agent_dir / ".project.lock").open("a+")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            project = self._ensure_project(root)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()
        self.registry.upsert(project)
        return project

    def _find_existing_root(self, start: Path) -> Path | None:
        for current in (start, *start.parents):
            marker = current / self.agent_dir_name
            if marker.is_symlink():
                continue
            if marker.is_dir():
                return current
        return None

    def _infer_new_root(self, start: Path) -> Path:
        for current in (start, *start.parents):
            if self._is_git_root(current):
                return current
        return start

    @staticmethod
    def _is_git_root(path: Path) -> bool:
        """Recognize actual Git metadata, not an arbitrary directory named .git."""
        marker = path / ".git"
        if marker.is_file():
            try:
                return marker.read_text(encoding="utf-8", errors="replace").lstrip().startswith("gitdir:")
            except OSError:
                return False
        return marker.is_dir() and (marker / "HEAD").is_file()

    def _ensure_project(self, root: Path) -> Project:
        agent_dir = root / self.agent_dir_name
        if agent_dir.is_symlink():
            raise ValueError(f"project Agent directory must not be a symbolic link: {agent_dir}")
        if agent_dir.exists() and not agent_dir.is_dir():
            raise ValueError(f"project Agent path is not a directory: {agent_dir}")
        config_path = agent_dir / "project.yaml"
        context_path = agent_dir / "context.md"
        created = not config_path.exists()
        agent_dir.mkdir(parents=True, exist_ok=True)
        for private_name in (
            "sessions",
            "cache",
            "snapshots",
            "browser-sessions",
            "downloads",
            "queues",
            "parallel",
        ):
            private_dir = agent_dir / private_name
            if private_dir.is_symlink():
                raise ValueError(f"private Agent directory must not be a symbolic link: {private_dir}")
            if private_dir.exists() and not private_dir.is_dir():
                raise ValueError(f"private Agent path is not a directory: {private_dir}")
            private_dir.mkdir(exist_ok=True)
            try:
                private_dir.chmod(0o700)
            except OSError:
                pass

        if config_path.exists():
            metadata = self._read_project_yaml(config_path)
            project_id = str(metadata.get("project_id") or metadata.get("id") or self._stable_id(root))
        else:
            project_id = self._new_project_id(root)
            metadata = {}

        language = str(metadata.get("language") or detect_language(root))
        previous_root = Path(str(metadata.get("root_path") or root))
        stored_name = str(metadata.get("name") or "")
        name_source = str(metadata.get("name_source") or "")
        if name_source == "custom":
            name = stored_name or root.name or "project"
        elif stored_name and stored_name != previous_root.name:
            name = stored_name
            name_source = "custom"
        else:
            name = root.name or stored_name or "project"
            name_source = "auto"
        now = utc_now_iso()
        metadata = {
            "project_id": project_id,
            "name": name,
            "name_source": name_source,
            "root_path": str(root),
            "language": language,
            "created_at": metadata.get("created_at") or now,
            "updated_at": now,
        }
        self._write_project_yaml(config_path, metadata)
        self._write_default_files(agent_dir, context_path, root, project_id, created)
        return Project(
            id=project_id,
            name=name,
            root=root,
            agent_dir=agent_dir,
            config_path=config_path,
            context_path=context_path,
            language=language,
        )

    def _new_project_id(self, root: Path) -> str:
        strategy = str(self.config.get("project.id_strategy", "uuid"))
        if strategy == "sha256":
            return self._stable_id(root)
        return str(uuid.uuid4())

    @staticmethod
    def _stable_id(root: Path) -> str:
        return hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()

    @staticmethod
    def _read_project_yaml(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            return {}
        return data

    @staticmethod
    def _write_project_yaml(path: Path, metadata: dict[str, Any]) -> None:
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with temp.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(metadata, fh, sort_keys=False, allow_unicode=True)
        temp.replace(path)

    @staticmethod
    def _write_default_files(
        agent_dir: Path,
        context_path: Path,
        root: Path,
        project_id: str,
        created: bool,
    ) -> None:
        if not context_path.exists():
            context_path.write_text(
                "\n".join(
                    [
                        f"# {root.name or 'Project'} Context",
                        "",
                        f"- Project ID: `{project_id}`",
                        f"- Root: `{root}`",
                        "",
                        "## Overview",
                        "",
                        "Add durable project facts, architecture notes, conventions, and constraints here.",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
        todo_path = agent_dir / "todo.md"
        if not todo_path.exists():
            todo_path.write_text("# TODO\n\n", encoding="utf-8")
        architecture_path = agent_dir / "architecture.md"
        if not architecture_path.exists():
            architecture_path.write_text("# Architecture\n\n", encoding="utf-8")
        ignore_path = agent_dir / "ignore"
        if not ignore_path.exists():
            ignore_path.write_text(
                "\n".join(
                    [
                        ".git/",
                        "node_modules/",
                        ".venv/",
                        "venv/",
                        "__pycache__/",
                        "dist/",
                        "build/",
                        ".project-agent/cache/",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
        gitignore_path = agent_dir / ".gitignore"
        gitignore_entries = (
            ".project.lock",
            "cache/",
            "sessions/",
            "snapshots/",
            "browser-sessions/",
            "downloads/",
            "queues/",
            "parallel/",
            "memory/",
            "index.json",
            "index.semantic.json",
            "workspace_memory.json",
        )
        existing = gitignore_path.read_text(encoding="utf-8", errors="replace") if gitignore_path.exists() else ""
        merged_lines: list[str] = []
        seen_lines: set[str] = set()
        for raw in [*existing.splitlines(), *gitignore_entries]:
            line = raw.strip()
            if line and line not in seen_lines:
                merged_lines.append(line)
                seen_lines.add(line)
        normalized = "\n".join(merged_lines) + "\n"
        if normalized != existing:
            temp = gitignore_path.with_name(f"{gitignore_path.name}.{uuid.uuid4().hex}.tmp")
            temp.write_text(normalized, encoding="utf-8")
            temp.replace(gitignore_path)
        if created:
            readme_path = agent_dir / "README.md"
            readme_path.write_text(
                "This directory stores project-local context for Deep Agent.\n",
                encoding="utf-8",
            )


def detect_language(root: Path) -> str:
    checks = [
        ("Python", ["pyproject.toml", "requirements.txt", "setup.py"]),
        ("JavaScript", ["package.json"]),
        ("Rust", ["Cargo.toml"]),
        ("Go", ["go.mod"]),
        ("Java", ["pom.xml", "build.gradle", "settings.gradle"]),
        ("Godot", ["project.godot"]),
        ("CSharp", ["*.csproj", "*.sln"]),
    ]
    for language, patterns in checks:
        for pattern in patterns:
            if "*" in pattern:
                if next(root.glob(pattern), None):
                    return language
            elif (root / pattern).exists():
                return language
    return "Unknown"
