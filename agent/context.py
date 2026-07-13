from __future__ import annotations

import ast
import fnmatch
import hashlib
import json
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence
from uuid import uuid4

from .config import AppConfig
from .project import Project
from .timeutil import utc_now_iso
from .workspace_memory import WorkspaceMemoryManager

if TYPE_CHECKING:
    from .memory import MemoryItem
    from .state import AgentState


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
class ContextSection:
    key: str
    title: str
    content: str
    priority: int
    source_ids: tuple[str, ...] = ()
    original_chars: int = 0
    included_chars: int = 0
    truncated: bool = False

    @property
    def rendered(self) -> str:
        return f"## {self.title}\n\n{self.content.strip()}"


@dataclass(frozen=True)
class ContextSnapshot:
    rendered: str
    index: dict[str, Any]
    index_path: Path
    generated_path: Path
    loaded_files: list[str]
    git_branch: str | None
    sections: tuple[ContextSection, ...] = ()


@dataclass(frozen=True)
class ContextBuildRequest:
    snapshot: ContextSnapshot
    state: AgentState
    memory_items: Sequence[MemoryItem] = ()
    memory_context: str = ""
    capability_summary: str = ""
    prior_messages: Sequence[dict[str, Any]] = ()
    recovery_context: str = ""
    recovery_memory_ids: Sequence[int] = ()
    phase: str = "initial"
    max_chars: int = 32_000


@dataclass(frozen=True)
class ContextPackage:
    schema_version: int
    phase: str
    project_id: str
    session_id: str
    turn: int
    user_request: str
    sections: tuple[ContextSection, ...]
    fingerprint: str
    file_count: int
    source_file_count: int
    git_branch: str | None
    index_path: Path
    generated_path: Path
    loaded_files: tuple[str, ...]
    included_memory_ids: tuple[int, ...]
    max_chars: int
    used_chars: int
    rendered_chars: int
    original_user_request_chars: int
    user_request_truncated: bool
    omitted_sections: tuple[str, ...] = ()
    truncated_sections: tuple[str, ...] = ()

    @property
    def rendered(self) -> str:
        return "\n\n".join(section.rendered for section in self.sections)


class ContextBuilder:
    SECTION_LIMITS = {
        "task": 8_000,
        "project_instructions": 12_000,
        "execution": 5_000,
        "session": 6_000,
        "project_summary": 6_000,
        "workspace": 4_000,
        "project_docs": 12_000,
        "memory": 8_000,
        "semantic": 12_000,
        "capabilities": 8_000,
        "recovery": 6_000,
    }
    SECTION_ORDER = {
        "task": 0,
        "project_instructions": 1,
        "execution": 2,
        "session": 3,
        "project_summary": 4,
        "workspace": 5,
        "memory": 6,
        "semantic": 7,
        "capabilities": 8,
        "project_docs": 9,
        "recovery": 2,
    }

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

        workspace_memory = WorkspaceMemoryManager(project).refresh(records, fingerprint=fingerprint)

        rendered, loaded_files, sections = self._render_context_details(
            project,
            index,
            semantic_index,
            workspace_memory,
        )
        generated_path = project.agent_dir / "cache" / "context.generated.md"
        generated_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_text(generated_path, rendered.rstrip() + "\n")
        return ContextSnapshot(
            rendered=rendered,
            index=index,
            index_path=index_path,
            generated_path=generated_path,
            loaded_files=loaded_files,
            git_branch=read_git_branch(project.root),
            sections=sections,
        )

    def build_package(self, request: ContextBuildRequest) -> ContextPackage:
        """Build one bounded, in-memory package for Prompt rendering.

        The public project snapshot remains independently cached for CLI and
        daemon use. Private Memory, Session, and recovery text is never written
        to ``context.generated.md``.
        """
        if request.phase not in {"initial", "resume", "recovery"}:
            raise ValueError(f"unsupported context phase: {request.phase}")
        max_chars = int(request.max_chars)
        if max_chars <= 0:
            raise ValueError("context package max_chars must be positive")

        original_user_request = str(request.state.user_request)
        request_limit = self._positive_limit(
            self.config.get("context.max_user_request_chars", 32_000),
            default=32_000,
        )
        request_budget = min(request_limit, max(1, max_chars // 2))
        bounded_user_request = self._head_tail(original_user_request, request_budget)
        context_budget = max(0, max_chars - len(bounded_user_request))

        candidates = [self._task_section(request.state)]
        if request.phase != "recovery":
            candidates.extend(request.snapshot.sections)
            execution = self._execution_section(request.state)
            if execution:
                candidates.append(execution)
        if request.phase == "resume":
            candidates.append(self._resume_section(request.prior_messages))
        for memory_item in request.memory_items:
            candidates.append(self._memory_section(memory_item))
        memory_context = request.memory_context.strip()
        if memory_context:
            candidates.append(self._section("memory", "Relevant Long-Term Memory", memory_context, priority=60))
        capability_summary = request.capability_summary.strip()
        if capability_summary:
            candidates.append(
                self._section("capabilities", "Registered Tool Capabilities", capability_summary, priority=80)
            )
        recovery_context = request.recovery_context.strip()
        if recovery_context:
            candidates.append(
                self._section(
                    "recovery",
                    "Failure Recovery Memory",
                    recovery_context,
                    priority=25,
                    source_ids=tuple(str(item) for item in request.recovery_memory_ids),
                )
            )

        selected, omitted, truncated = self._select_sections(candidates, max_chars=context_budget)
        rendered = "\n\n".join(section.rendered for section in selected)
        snapshot = request.snapshot
        state = request.state
        return ContextPackage(
            schema_version=1,
            phase=request.phase,
            project_id=str(state.project.get("id") or snapshot.index.get("project_id") or ""),
            session_id=str(state.session_id),
            turn=int(state.turn),
            user_request=bounded_user_request,
            sections=selected,
            fingerprint=str(snapshot.index.get("fingerprint") or ""),
            file_count=max(0, int(snapshot.index.get("file_count") or 0)),
            source_file_count=max(0, int(snapshot.index.get("source_file_count") or 0)),
            git_branch=snapshot.git_branch,
            index_path=snapshot.index_path,
            generated_path=snapshot.generated_path,
            loaded_files=tuple(snapshot.loaded_files),
            included_memory_ids=tuple(
                dict.fromkeys(
                    int(source_id)
                    for section in selected
                    if section.key in {"memory", "recovery"}
                    for source_id in section.source_ids
                    if source_id.isdigit()
                )
            ),
            max_chars=max_chars,
            used_chars=len(rendered) + len(bounded_user_request),
            rendered_chars=len(rendered),
            original_user_request_chars=len(original_user_request),
            user_request_truncated=len(bounded_user_request) < len(original_user_request),
            omitted_sections=omitted,
            truncated_sections=truncated,
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
                if path.is_symlink():
                    continue
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
        workspace_memory: dict[str, Any] | None = None,
    ) -> tuple[str, list[str]]:
        rendered, loaded_files, _sections = self._render_context_details(
            project,
            index,
            semantic_index,
            workspace_memory,
        )
        return rendered, loaded_files

    def _render_context_details(
        self,
        project: Project,
        index: dict[str, Any],
        semantic_index: dict[str, Any] | None = None,
        workspace_memory: dict[str, Any] | None = None,
    ) -> tuple[str, list[str], tuple[ContextSection, ...]]:
        max_total = int(self.config.get("context.max_prompt_chars", 32_000))
        max_per_file = int(self.config.get("context.max_context_file_chars", 8_000))
        summary_content = "\n".join(
            [
                f"- Project ID: `{project.id}`",
                f"- Name: `{project.name}`",
                f"- Root: `{project.root}`",
                f"- Language: `{project.language}`",
                f"- Indexed files: `{index.get('file_count', 0)}`",
                f"- Entry points: `{', '.join(index.get('entries') or []) or 'unknown'}`",
            ]
        )
        context_sections = [
            self._section(
                "project_summary",
                "Runtime Project Context",
                summary_content,
                priority=40,
                source_ids=(str(project.root),),
            )
        ]
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
        if workspace_memory:
            workspace_rendered = WorkspaceMemoryManager.render(workspace_memory)
            sections.extend([workspace_rendered, ""])
            context_sections.append(
                self._section(
                    "workspace",
                    "Workspace Memory",
                    self._without_heading(workspace_rendered, "## Workspace Memory"),
                    priority=50,
                    source_ids=("workspace_memory.json",),
                )
            )
        loaded_files: list[str] = []
        candidates = [
            project.root / "AGENTS.md",
            project.root / "CLAUDE.md",
            project.context_path,
            project.agent_dir / "architecture.md",
            project.agent_dir / "todo.md",
            *(project.root / name for name in CONTEXT_FILENAMES if name not in {"AGENTS.md", "CLAUDE.md"}),
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
            original_chars = len(content)
            key = (
                "project_instructions"
                if path in {project.root / "AGENTS.md", project.root / "CLAUDE.md", project.context_path}
                else "project_docs"
            )
            remaining = max_total - len("\n".join(sections))
            if remaining <= 500:
                break
            limit = min(max_per_file, remaining)
            was_truncated = len(content) > limit
            if was_truncated:
                content = self._truncate_units(content, max_chars=limit, key=key)
            sections.extend([f"## {rel}", "", content.strip(), ""])
            loaded_files.append(rel)
            context_sections.append(
                ContextSection(
                    key=key,
                    title=rel,
                    content=content.strip(),
                    priority=10 if key == "project_instructions" else 90,
                    source_ids=(rel,),
                    original_chars=original_chars,
                    included_chars=len(content.strip()),
                    truncated=was_truncated,
                )
            )

        symbols = index.get("symbols") or []
        symbol_lines = [
            f"- `{item.get('path')}:{item.get('line')}` {item.get('kind')} `{item.get('name')}`"
            for item in symbols[:100]
            if isinstance(item, dict) and item.get("path") and item.get("kind") and item.get("name")
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
        source_summary = "\n".join(
            [
                *(symbol_lines or ["No source symbols were indexed."]),
                "",
                f"Full index: `{project.agent_dir / 'index.json'}`",
            ]
        ).strip()
        context_sections.append(
            self._section(
                "project_summary",
                "Source Index Summary",
                source_summary,
                priority=45,
                source_ids=(str(project.agent_dir / "index.json"),),
            )
        )
        if semantic_index and semantic_index.get("enabled"):
            semantic_lines: list[str] = []
            import_lines: list[str] = []
            module_lines: list[str] = []
            for file_item in semantic_index.get("files", []):
                path = str(file_item.get("path") or "")
                semantic_lines.extend(self._render_semantic_items(path, file_item.get("structures", [])))
                summary = file_item.get("module_summary") or {}
                module_lines.append(
                    f"- `{path}` structures={summary.get('structure_count', 0)} "
                    f"symbols={summary.get('symbol_count', 0)} imports={summary.get('import_count', 0)} "
                    f"exports={','.join(file_item.get('exports', [])) or '-'}"
                )
                import_lines.extend(
                    f"- `{path}:{item.get('line')}` imports `{item.get('source')}`"
                    for item in file_item.get("imports", [])
                )
            sections.extend(
                [
                    "",
                    "## Optional Semantic Index Summary",
                    "",
                    *module_lines[:100],
                    *(semantic_lines[:150] or ["No semantic structures were indexed."]),
                    *(import_lines[:100] or []),
                    *[
                        f"- relation `{item.get('source')}` -> `{item.get('target')}` via `{item.get('import')}`"
                        for item in semantic_index.get("relationships", [])[:100]
                    ],
                    "",
                    f"Full semantic index: `{project.agent_dir / 'index.semantic.json'}`",
                ]
            )
            semantic_content = "\n".join(
                [
                    *module_lines[:100],
                    *(semantic_lines[:150] or ["No semantic structures were indexed."]),
                    *(import_lines[:100] or []),
                    *[
                        f"- relation `{item.get('source')}` -> `{item.get('target')}` via `{item.get('import')}`"
                        for item in semantic_index.get("relationships", [])[:100]
                    ],
                    "",
                    f"Full semantic index: `{project.agent_dir / 'index.semantic.json'}`",
                ]
            ).strip()
            context_sections.append(
                self._section(
                    "semantic",
                    "Optional Semantic Index Summary",
                    semantic_content,
                    priority=70,
                    source_ids=(str(project.agent_dir / "index.semantic.json"),),
                )
            )
        rendered = "\n".join(sections).strip()
        if len(rendered) > max_total:
            rendered = rendered[: max(0, max_total - 16)].rstrip() + "\n...[truncated]"
        return rendered, loaded_files, tuple(context_sections)

    @staticmethod
    def _section(
        key: str,
        title: str,
        content: str,
        *,
        priority: int,
        source_ids: tuple[str, ...] = (),
    ) -> ContextSection:
        normalized = content.strip()
        return ContextSection(
            key=key,
            title=title,
            content=normalized,
            priority=priority,
            source_ids=source_ids,
            original_chars=len(normalized),
            included_chars=len(normalized),
        )

    @staticmethod
    def _without_heading(content: str, heading: str) -> str:
        normalized = content.strip()
        if normalized.startswith(heading):
            return normalized[len(heading) :].lstrip()
        return normalized

    def _task_section(self, state: AgentState) -> ContextSection:
        strategy = state.task_strategy or {}
        task_route = state.task_route or {}
        model_route = state.model_route or {}
        plan_lines = []
        for step in state.plan:
            plan_lines.append(
                f"- `{self._inline(step.id, 100)}` {self._inline(step.title, 300)}; "
                f"status={self._inline(step.status, 40)}; "
                f"deps={','.join(self._inline(item, 100) for item in step.dependencies) or '-'}; "
                f"retries={step.retry_count}/{step.max_retries}; "
                f"parallel={str(step.allow_parallel).lower()}; "
                f"done_when={self._inline(step.completion_criteria, 300) or '-'}"
            )
        content = "\n".join(
            [
                f"- Session: `{self._inline(state.session_id, 200)}`",
                f"- Turn: `{state.turn}`",
                f"- Working directory: `{self._inline(state.working_directory, 1000)}`",
                f"- Git branch: `{self._inline(state.git_branch or 'not detected', 300)}`",
                f"- Execution mode: `{self._inline(strategy.get('mode', 'standard'), 40)}`",
                f"- Thinking: `{str(bool(strategy.get('thinking_enabled', False))).lower()}`",
                f"- Reasoning effort: `{self._inline(strategy.get('reasoning_effort') or 'default', 40)}`",
                f"- Chunked context: `{str(bool(strategy.get('chunked_context', False))).lower()}`",
                f"- Plan required: `{str(bool(strategy.get('require_plan', False))).lower()}`",
                f"- Task type: `{self._inline(task_route.get('task_type') or 'unspecified', 80)}`",
                f"- Task scale: `{self._inline(task_route.get('scale') or 'unspecified', 40)}`",
                f"- Task risk: `{self._inline(task_route.get('risk') or 'unspecified', 40)}`",
                f"- DeepSeek tier: `{self._inline(model_route.get('tier') or 'standard', 40)}`",
                f"- DeepSeek model: `{self._inline(model_route.get('model') or 'configured default', 200)}`",
                "",
                "### Task Graph",
                "",
                *(plan_lines or ["No task graph has been published."]),
            ]
        )
        return self._section("task", "Agent State and Task", content, priority=0)

    def _memory_section(self, item: MemoryItem) -> ContextSection:
        scope = "global" if item.project_id is None else "project"
        tags = ", ".join(item.tags) or "-"
        content = (
            f"- [{scope}/{self._inline(item.kind, 80)}] {self._inline(item.title, 500)}\n"
            f"  tags: {self._inline(tags, 500)}\n"
            f"  {str(item.content).strip()[:1200]}"
        )
        return self._section(
            "memory",
            "Relevant Long-Term Memory",
            content,
            priority=60,
            source_ids=(str(item.id),),
        )

    def _execution_section(self, state: AgentState) -> ContextSection | None:
        execution = state.execution_context
        if execution is None:
            return None
        content = "\n".join(
            [
                f"- Current directory: `{self._inline(execution.current_directory, 1000)}`",
                f"- Git branch: `{self._inline(execution.git_branch or 'not detected', 300)}`",
                f"- Current plan step: `{self._inline(execution.current_plan_id or 'none', 200)}`",
                f"- Current queue: `{self._inline(execution.current_queue_id or 'none', 200)}`",
                f"- Recent tool: `{self._inline(execution.recent_tool or 'none', 200)}`",
                f"- Recent error: `{self._inline(execution.recent_error or 'none', 500)}`",
                f"- Current snapshot: `{self._inline(execution.current_snapshot or 'none', 200)}`",
                f"- Modified files: `{', '.join(self._inline(item, 300) for item in execution.modified_files[-20:]) or 'none'}`",
                f"- Prompt phase: `{self._inline(execution.prompt_phase, 80)}`",
            ]
        )
        return self._section("execution", "Execution Context", content, priority=30)

    def _resume_section(self, messages: Sequence[dict[str, Any]]) -> ContextSection:
        previous = self.previous_outcome(messages)
        content = (
            "Session resumed from a compact checkpoint. The previous raw tool transcript was removed. "
            "Use Agent State and Execution Context as the source of truth."
        )
        if previous:
            content += "\n\n### Previous Outcome\n\n" + previous
        return self._section("session", "Resume Context", content, priority=35)

    @classmethod
    def previous_outcome(cls, messages: Sequence[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") != "assistant" or message.get("tool_calls"):
                continue
            content = str(message.get("content") or "").strip()
            if content:
                return cls._head_tail(content, min(4_000, cls.SECTION_LIMITS["session"] - 1_000))
        return ""

    def _select_sections(
        self,
        candidates: Sequence[ContextSection],
        *,
        max_chars: int,
    ) -> tuple[tuple[ContextSection, ...], tuple[str, ...], tuple[str, ...]]:
        indexed = list(enumerate(candidates))
        indexed.sort(
            key=lambda item: (
                self.SECTION_ORDER.get(item[1].key, 100),
                item[1].priority,
                item[0],
            )
        )
        selected: list[ContextSection] = []
        omitted: list[str] = []
        truncated: list[str] = []
        source_used: dict[str, int] = {}
        used = 0

        for _index, section in indexed:
            identifier = f"{section.key}:{section.title}"
            prefix_chars = 2 if selected else 0
            header_chars = len(f"## {section.title}\n\n")
            remaining = max_chars - used - prefix_chars - header_chars
            source_limit = self._section_limit(section.key, max_chars)
            if section.key == "task":
                source_limit = min(source_limit, max(1, max_chars // 2))
            source_remaining = source_limit - source_used.get(section.key, 0)
            content_budget = min(remaining, source_remaining)
            if content_budget <= 0:
                omitted.append(identifier)
                continue
            fitted = self._fit_section(section, content_budget)
            if not fitted.content:
                omitted.append(identifier)
                continue
            rendered_chars = prefix_chars + len(fitted.rendered)
            if used + rendered_chars > max_chars:
                omitted.append(identifier)
                continue
            selected.append(fitted)
            used += rendered_chars
            source_used[section.key] = source_used.get(section.key, 0) + fitted.included_chars
            if fitted.truncated:
                truncated.append(identifier)

        return tuple(selected), tuple(omitted), tuple(truncated)

    def _section_limit(self, key: str, fallback: int) -> int:
        config_keys = {
            "task": "context.max_task_context_chars",
            "session": "context.max_session_context_chars",
            "memory": "context.max_memory_context_chars",
            "capabilities": "context.max_capability_context_chars",
            "recovery": "context.max_recovery_context_chars",
        }
        default = self.SECTION_LIMITS.get(key, fallback)
        configured = self.config.get(config_keys[key], default) if key in config_keys else default
        try:
            return max(0, int(configured))
        except (TypeError, ValueError, OverflowError):
            return default

    @staticmethod
    def _positive_limit(value: Any, *, default: int) -> int:
        try:
            return max(1, int(value))
        except (TypeError, ValueError, OverflowError):
            return default

    @classmethod
    def _fit_section(cls, section: ContextSection, max_chars: int) -> ContextSection:
        content = section.content.strip()
        if len(content) <= max_chars:
            return replace(section, content=content, included_chars=len(content))
        fitted = cls._truncate_units(content, max_chars=max_chars, key=section.key)
        return replace(
            section,
            content=fitted,
            included_chars=len(fitted),
            truncated=True,
        )

    @staticmethod
    def _truncate_units(content: str, *, max_chars: int, key: str) -> str:
        marker = "...[truncated]"
        if max_chars < len(marker):
            return ""
        if key == "memory":
            units = [item.strip() for item in re.split(r"(?=^- \[)", content, flags=re.MULTILINE) if item.strip()]
            separator = "\n"
        elif key in {"project_instructions", "project_docs", "session", "recovery", "workspace"}:
            units = [item.strip() for item in re.split(r"\n\s*\n", content) if item.strip()]
            separator = "\n\n"
        else:
            units = [item.rstrip() for item in content.splitlines() if item.strip()]
            separator = "\n"

        included: list[str] = []
        for unit in units:
            candidate = separator.join([*included, unit])
            suffix = separator + marker
            if len(candidate) + len(suffix) > max_chars:
                break
            included.append(unit)
        if not included and key == "memory":
            return ""
        if not included and key in {"project_instructions", "project_docs"}:
            return ContextBuilder._head_tail(content, max_chars)
        if not included:
            return marker
        return separator.join(included) + separator + marker

    @staticmethod
    def _head_tail(content: str, max_chars: int) -> str:
        if len(content) <= max_chars:
            return content
        marker = "\n...[middle truncated]...\n"
        if max_chars <= len(marker):
            return marker[:max_chars]
        available = max(0, max_chars - len(marker))
        head = available // 2
        tail_length = available - head
        tail = content[-tail_length:].lstrip() if tail_length else ""
        return content[:head].rstrip() + marker + tail

    @staticmethod
    def _inline(value: Any, max_chars: int) -> str:
        normalized = " ".join(str(value).split())
        if len(normalized) <= max_chars:
            return normalized
        return normalized[: max(0, max_chars - 15)].rstrip() + "...[truncated]"

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
                        "exports": [
                            str(item.get("name"))
                            for item in structures
                            if item.get("name") and not str(item.get("name")).startswith("_")
                        ],
                        "module_summary": {
                            "structure_count": self._count_semantic_structures(structures),
                            "symbol_count": len(result.symbols),
                            "import_count": len(imports),
                        },
                    }
                )
            except Exception as exc:
                failures.append({"path": str(record["path"]), "error": str(exc)[:300]})
            if len(files) >= max_files:
                break
        relationships = self._semantic_relationships(files)
        value = {
            "schema_version": 1,
            "enabled": True,
            "project_id": project.id,
            "generated_at": utc_now_iso(),
            "fingerprint": fingerprint,
            "cache_hit": False,
            "file_count": len(files),
            "files": files,
            "relationships": relationships,
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

    @classmethod
    def _count_semantic_structures(cls, items: list[dict[str, Any]]) -> int:
        return sum(1 + cls._count_semantic_structures(item.get("children", [])) for item in items)

    @staticmethod
    def _semantic_relationships(files: list[dict[str, Any]]) -> list[dict[str, str]]:
        paths = {str(item.get("path") or "") for item in files}
        stems: dict[str, str] = {}
        for path in paths:
            pure = Path(path)
            without_suffix = pure.with_suffix("").as_posix()
            stems[without_suffix] = path
            stems[without_suffix.replace("/", ".")] = path
            if pure.name == "__init__.py":
                stems[pure.parent.as_posix().replace("/", ".")] = path
        relationships: list[dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for file_item in files:
            source_path = str(file_item.get("path") or "")
            for imported in file_item.get("imports", []):
                raw = str(imported.get("source") or "")
                candidates = ContextBuilder._import_candidates(source_path, raw)
                target = next((stems[candidate] for candidate in candidates if candidate in stems), None)
                if not target or target == source_path:
                    continue
                key = (source_path, target, raw)
                if key in seen:
                    continue
                seen.add(key)
                relationships.append({"source": source_path, "target": target, "import": raw})
        return relationships[:5000]

    @staticmethod
    def _import_candidates(source_path: str, raw: str) -> list[str]:
        value = raw.strip().rstrip(";")
        candidates: list[str] = []
        python_match = re.search(r"(?:from|import)\s+([.A-Za-z0-9_]+)", value)
        if python_match:
            module = python_match.group(1)
            if module.startswith("."):
                parent_parts = list(Path(source_path).parent.parts)
                dots = len(module) - len(module.lstrip("."))
                base = parent_parts[: max(0, len(parent_parts) - dots + 1)]
                module = ".".join([*base, module.lstrip(".")]).strip(".")
            candidates.extend([module, module.replace(".", "/")])
        js_match = re.search(r"(?:from\s+|require\s*\(\s*)['\"]([^'\"]+)['\"]", value)
        if js_match:
            module = js_match.group(1)
            if module.startswith("."):
                module = (Path(source_path).parent / module).as_posix()
            candidates.append(module.removesuffix("/index"))
            candidates.append(module)
        go_match = re.search(r"['\"]([^'\"]+)['\"]", value)
        if go_match:
            candidates.append(go_match.group(1))
        expanded: list[str] = []
        for candidate in candidates:
            normalized = candidate.strip("./")
            expanded.extend([normalized, f"{normalized}/index"])
        return list(dict.fromkeys(expanded))

    def _extract_symbols(self, path: Path, relative: str) -> list[dict[str, Any]]:
        if path.is_symlink():
            return []
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
        temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(path)

    @staticmethod
    def _write_text(path: Path, value: str) -> None:
        temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temp.write_text(value, encoding="utf-8")
        temp.replace(path)


def read_git_branch(root: Path) -> str | None:
    marker = root / ".git"
    git_dir = marker
    if marker.is_file():
        try:
            value = marker.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return None
        if not value.lower().startswith("gitdir:"):
            return None
        target = value.split(":", 1)[1].strip()
        git_dir = (root / target).resolve() if not Path(target).is_absolute() else Path(target)
    head = git_dir / "HEAD"
    if not head.is_file():
        return None
    try:
        value = head.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    if not value:
        return None
    prefix = "ref: "
    if value.startswith(prefix):
        ref = value[len(prefix) :]
        return ref.removeprefix("refs/heads/")
    return value[:12]
