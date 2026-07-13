from __future__ import annotations

import ast
import fnmatch
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AppConfig
from .project import Project
from .timeutil import utc_now_iso


CONTEXT_FILENAMES = (
    "README.md",
    "README.rst",
    "README.txt",
    "CLAUDE.md",
    "AGENTS.md",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "project.godot",
    ".gitignore",
)
SOURCE_SUFFIXES = {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".gd", ".cs"}
SEMANTIC_LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
}
DEFAULT_IGNORED_DIRS = {
    ".git",
    ".project-agent",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "dist",
    "build",
    ".next",
    ".idea",
    ".vscode",
}


@dataclass(frozen=True)
class ContextSnapshot:
    rendered: str
    index: dict[str, Any]
    index_path: Path
    generated_path: Path
    loaded_files: list[str]
    git_branch: str | None


class ContextBuilder:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def build(self, project: Project, *, refresh: bool = False) -> ContextSnapshot:
        records = self._scan_files(project)
        fingerprint = self._fingerprint(records)
        index_path = project.agent_dir / "index.json"
        old_index = self._read_json(index_path)
        if not refresh and old_index.get("fingerprint") == fingerprint:
            index = old_index
            index["cache_hit"] = True
        else:
            index = self._build_index(project, records, fingerprint)
            self._write_json(index_path, index)

        semantic_index = None
        if bool(self.config.get("context.semantic_index_enabled", False)):
            semantic_index = self._build_semantic_index(project, records, fingerprint, refresh=refresh)

        rendered, loaded_files = self._render_context(project, index, semantic_index)
        generated_path = project.agent_dir / "cache" / "context.generated.md"
        generated_path.parent.mkdir(parents=True, exist_ok=True)
        generated_path.write_text(rendered.rstrip() + "\n", encoding="utf-8")
        return ContextSnapshot(
            rendered=rendered,
            index=index,
            index_path=index_path,
            generated_path=generated_path,
            loaded_files=loaded_files,
            git_branch=read_git_branch(project.root),
        )

    def _scan_files(self, project: Project) -> list[dict[str, Any]]:
        max_files = int(self.config.get("context.max_files", 5000))
        max_file_size = int(self.config.get("context.max_index_file_bytes", 1_000_000))
        patterns = self._ignore_patterns(project)
        records: list[dict[str, Any]] = []
        for current, dirs, files in os.walk(project.root, followlinks=False):
            current_path = Path(current)
            rel_dir = current_path.relative_to(project.root).as_posix()
            dirs[:] = sorted(
                name
                for name in dirs
                if name not in DEFAULT_IGNORED_DIRS
                and not self._ignored(self._join_rel(rel_dir, name), patterns, is_dir=True)
            )
            for name in sorted(files):
                rel = self._join_rel(rel_dir, name)
                if self._ignored(rel, patterns, is_dir=False):
                    continue
                path = current_path / name
                try:
                    stat = path.stat()
                except OSError:
                    continue
                if stat.st_size > max_file_size:
                    continue
                records.append(
                    {
                        "path": rel,
                        "size": stat.st_size,
                        "mtime_ns": stat.st_mtime_ns,
                        "suffix": path.suffix.lower(),
                    }
                )
                if len(records) >= max_files:
                    return records
        return records

    def _build_index(
        self,
        project: Project,
        records: list[dict[str, Any]],
        fingerprint: str,
    ) -> dict[str, Any]:
        source_records = [item for item in records if item["suffix"] in SOURCE_SUFFIXES]
        max_symbol_files = int(self.config.get("context.max_symbol_files", 500))
        symbols: list[dict[str, Any]] = []
        for item in source_records[:max_symbol_files]:
            symbols.extend(self._extract_symbols(project.root / item["path"], item["path"]))
        entries = self._detect_entries(records)
        return {
            "schema_version": 1,
            "project_id": project.id,
            "language": project.language,
            "generated_at": utc_now_iso(),
            "fingerprint": fingerprint,
            "cache_hit": False,
            "entry": entries[0] if entries else None,
            "entries": entries,
            "file_count": len(records),
            "source_file_count": len(source_records),
            "files": records,
            "symbols": symbols[:5000],
        }

    def _render_context(
        self,
        project: Project,
        index: dict[str, Any],
        semantic_index: dict[str, Any] | None = None,
    ) -> tuple[str, list[str]]:
        max_total = int(self.config.get("context.max_prompt_chars", 32_000))
        max_per_file = int(self.config.get("context.max_context_file_chars", 8_000))
        sections = [
            "# Runtime Project Context",
            "",
            f"- Project ID: `{project.id}`",
            f"- Name: `{project.name}`",
            f"- Root: `{project.root}`",
            f"- Language: `{project.language}`",
            f"- Indexed files: `{index.get('file_count', 0)}`",
            f"- Entry points: `{', '.join(index.get('entries') or []) or 'unknown'}`",
            "",
        ]
        loaded_files: list[str] = []
        candidates = [
            project.context_path,
            project.agent_dir / "architecture.md",
            project.agent_dir / "todo.md",
            *(project.root / name for name in CONTEXT_FILENAMES),
        ]
        seen: set[Path] = set()
        for path in candidates:
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = path.relative_to(project.root).as_posix()
            remaining = max_total - len("\n".join(sections))
            if remaining <= 500:
                break
            limit = min(max_per_file, remaining)
            if len(content) > limit:
                content = content[:limit] + "\n...[truncated]"
            sections.extend([f"## {rel}", "", content.strip(), ""])
            loaded_files.append(rel)

        symbols = index.get("symbols") or []
        symbol_lines = [
            f"- `{item.get('path')}:{item.get('line')}` {item.get('kind')} `{item.get('name')}`"
            for item in symbols[:100]
        ]
        sections.extend(
            [
                "## Source Index Summary",
                "",
                *(symbol_lines or ["No source symbols were indexed."]),
                "",
                f"Full index: `{project.agent_dir / 'index.json'}`",
            ]
        )
        if semantic_index and semantic_index.get("enabled"):
            semantic_lines: list[str] = []
            import_lines: list[str] = []
            for file_item in semantic_index.get("files", []):
                path = str(file_item.get("path") or "")
                semantic_lines.extend(self._render_semantic_items(path, file_item.get("structures", [])))
                import_lines.extend(
                    f"- `{path}:{item.get('line')}` imports `{item.get('source')}`"
                    for item in file_item.get("imports", [])
                )
            sections.extend(
                [
                    "",
                    "## Optional Semantic Index Summary",
                    "",
                    *(semantic_lines[:150] or ["No semantic structures were indexed."]),
                    *(import_lines[:100] or []),
                    "",
                    f"Full semantic index: `{project.agent_dir / 'index.semantic.json'}`",
                ]
            )
        return "\n".join(sections).strip(), loaded_files

    @classmethod
    def _render_semantic_items(
        cls,
        path: str,
        items: list[dict[str, Any]],
        *,
        parent: str = "",
    ) -> list[str]:
        lines: list[str] = []
        for item in items:
            name = str(item.get("name") or "")
            qualified = f"{parent}.{name}" if parent else name
            signature = f" `{item.get('signature')}`" if item.get("signature") else ""
            lines.append(f"- `{path}:{item.get('line')}` {item.get('kind')} `{qualified}`{signature}")
            children = item.get("children")
            if isinstance(children, list):
                lines.extend(cls._render_semantic_items(path, children, parent=qualified))
        return lines

    def _build_semantic_index(
        self,
        project: Project,
        records: list[dict[str, Any]],
        fingerprint: str,
        *,
        refresh: bool,
    ) -> dict[str, Any]:
        path = project.agent_dir / "index.semantic.json"
        old = self._read_json(path)
        if not refresh and old.get("fingerprint") == fingerprint:
            old["cache_hit"] = True
            return old
        try:
            from tree_sitter_language_pack import ProcessConfig, process
        except Exception as exc:
            value = {
                "schema_version": 1,
                "enabled": False,
                "reason": f"tree-sitter language pack unavailable: {exc}",
                "fingerprint": fingerprint,
                "generated_at": utc_now_iso(),
            }
            self._write_json(path, value)
            return value

        allowed = self.config.get("context.semantic_languages", [])
        allowed_languages = {str(item) for item in allowed} if isinstance(allowed, list) else set()
        max_files = int(self.config.get("context.max_symbol_files", 500))
        files: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        for record in records:
            language = SEMANTIC_LANGUAGE_BY_SUFFIX.get(str(record.get("suffix") or ""))
            if not language or (allowed_languages and language not in allowed_languages):
                continue
            try:
                source = (project.root / record["path"]).read_text(encoding="utf-8", errors="replace")
                result = process(source, ProcessConfig(language=language, structure=True, imports=True, symbols=True))
                structures = self._semantic_structures(result.structure)
                imports = [
                    {
                        "source": str(item.source),
                        "alias": str(item.alias) if item.alias else None,
                        "line": int(item.span.start_line) + 1,
                    }
                    for item in result.imports
                ]
                files.append(
                    {
                        "path": record["path"],
                        "language": language,
                        "structures": structures,
                        "imports": imports,
                    }
                )
            except Exception as exc:
                failures.append({"path": str(record["path"]), "error": str(exc)[:300]})
            if len(files) >= max_files:
                break
        value = {
            "schema_version": 1,
            "enabled": True,
            "project_id": project.id,
            "generated_at": utc_now_iso(),
            "fingerprint": fingerprint,
            "cache_hit": False,
            "file_count": len(files),
            "files": files,
            "failures": failures[:100],
        }
        self._write_json(path, value)
        return value

    @classmethod
    def _semantic_structures(cls, items: Any) -> list[dict[str, Any]]:
        structures: list[dict[str, Any]] = []
        for item in items:
            structures.append(
                {
                    "kind": str(item.kind),
                    "name": str(item.name),
                    "signature": str(item.signature) if item.signature else None,
                    "line": int(item.span.start_line) + 1,
                    "end_line": int(item.span.end_line) + 1,
                    "children": cls._semantic_structures(item.children),
                }
            )
        return structures

    def _extract_symbols(self, path: Path, relative: str) -> list[dict[str, Any]]:
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        if path.suffix.lower() == ".py":
            try:
                tree = ast.parse(source)
            except SyntaxError:
                return []
            symbols = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    kind = "class" if isinstance(node, ast.ClassDef) else "function"
                    symbols.append({"path": relative, "kind": kind, "name": node.name, "line": node.lineno})
            return symbols

        patterns = (
            ("class", re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)),
            ("function", re.compile(r"^\s*(?:def|func|function)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)),
        )
        symbols = []
        for kind, pattern in patterns:
            for match in pattern.finditer(source):
                symbols.append(
                    {
                        "path": relative,
                        "kind": kind,
                        "name": match.group(1),
                        "line": source.count("\n", 0, match.start()) + 1,
                    }
                )
        return symbols

    @staticmethod
    def _detect_entries(records: list[dict[str, Any]]) -> list[str]:
        paths = {item["path"] for item in records}
        candidates = (
            "main.py",
            "app.py",
            "manage.py",
            "src/main.py",
            "index.js",
            "src/index.js",
            "src/main.ts",
            "main.go",
            "src/main.rs",
            "project.godot",
        )
        return [candidate for candidate in candidates if candidate in paths]

    def _ignore_patterns(self, project: Project) -> list[str]:
        patterns: list[str] = []
        for path in (project.agent_dir / "ignore", project.root / ".gitignore"):
            if not path.exists():
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            patterns.extend(line.strip() for line in lines if line.strip() and not line.lstrip().startswith(("#", "!")))
        return patterns

    @staticmethod
    def _ignored(relative: str, patterns: list[str], *, is_dir: bool) -> bool:
        normalized = relative.strip("/")
        name = Path(normalized).name
        for raw in patterns:
            pattern = raw.strip().lstrip("/").rstrip("/")
            if not pattern:
                continue
            if fnmatch.fnmatch(normalized, pattern) or fnmatch.fnmatch(name, pattern):
                return True
            if is_dir and (normalized == pattern or normalized.startswith(pattern + "/")):
                return True
        return False

    @staticmethod
    def _join_rel(parent: str, name: str) -> str:
        return name if parent == "." else f"{parent}/{name}"

    @staticmethod
    def _fingerprint(records: list[dict[str, Any]]) -> str:
        compact = [(item["path"], item["size"], item["mtime_ns"]) for item in records]
        raw = json.dumps(compact, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(path)


def read_git_branch(root: Path) -> str | None:
    head = root / ".git" / "HEAD"
    if not head.is_file():
        return None
    try:
        value = head.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    prefix = "ref: refs/heads/"
    return value[len(prefix) :] if value.startswith(prefix) else value[:12] or None
