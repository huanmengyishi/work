"""Declarative built-in capability catalog.

The catalog contains schemas and *handler names*, never bound tool instances.
This keeps capability discovery cheap and lets ``ToolManager`` register every
built-in before any optional or filesystem-writing tool object is created.
"""

from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass
from typing import Any

from ..config import AppConfig
from .registry import ToolCapability


@dataclass(frozen=True)
class CapabilityDeclaration:
    capability: ToolCapability
    handler_name: str


def _declare(
    handler_name: str,
    tool: str,
    action: str,
    model_name: str,
    description: str,
    properties: dict[str, Any],
    required: tuple[str, ...] = (),
    permissions: tuple[str, ...] = ("read",),
    timeout_seconds: int = 120,
    *,
    supports_stream: bool = False,
    enabled: bool = True,
    input_formats: tuple[str, ...] = (),
    output_formats: tuple[str, ...] = (),
    available: bool = True,
    unavailable_reason: str = "",
    requires_confirmation: bool = False,
    concurrency_safe: bool = False,
) -> CapabilityDeclaration:
    return CapabilityDeclaration(
        ToolCapability(
            tool,
            action,
            model_name,
            description,
            properties,
            required,
            permissions,
            timeout_seconds,
            supports_stream,
            enabled,
            input_formats,
            output_formats,
            available,
            unavailable_reason,
            requires_confirmation,
            concurrency_safe,
        ),
        handler_name,
    )


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _lsp_availability() -> tuple[bool, str]:
    pyright = shutil.which("pyright")
    tsc = shutil.which("tsc")
    if pyright or tsc:
        engines = [name for name, path in (("Pyright", pyright), ("TypeScript", tsc)) if path]
        return True, f"available diagnostics engines: {', '.join(engines)}"
    return False, "missing diagnostics engine: pyright, tsc"


def builtin_capability_declarations(
    config: AppConfig,
    *,
    max_result_read_chars: int,
) -> tuple[CapabilityDeclaration, ...]:
    """Build the configured-independent catalog without constructing tools."""

    cwd_property = {
        "type": "string",
        "description": "Optional working directory inside the current project. Defaults to project root.",
    }
    timeout_property = {"type": "integer", "minimum": 1, "description": "Optional timeout seconds."}
    playwright_available = _module_available("playwright")
    lsp_available, lsp_reason = _lsp_availability()
    return (
        _declare(
            "_file_diff",
            "file",
            "diff",
            "file_diff",
            (
                "Create and store a unified-diff preview for one UTF-8 file. This never modifies the file. "
                "When copying source from read_file, exclude its line-number and → prefix; only text after "
                "the → is file content."
            ),
            {
                "path": {"type": "string"},
                "content": {"type": "string", "description": "Complete replacement content."},
                "old_text": {
                    "type": "string",
                    "description": "Exact file text to replace, excluding every read_file line-number/→ prefix.",
                },
                "new_text": {"type": "string", "description": "Replacement for old_text."},
                "replace_all": {"type": "boolean"},
                "delete": {"type": "boolean"},
            },
            ("path",),
            ("read",),
        ),
        _declare(
            "_file_apply",
            "file",
            "apply",
            "file_apply",
            "Apply a previously created file_diff preview atomically after snapshotting the original file.",
            {"preview_id": {"type": "string"}},
            ("preview_id",),
            ("write",),
            requires_confirmation=True,
        ),
        _declare(
            "_file_undo",
            "file",
            "undo",
            "file_undo",
            "Undo the latest active file snapshot in this session, or a selected snapshot.",
            {"snapshot_id": {"type": "string"}},
            permissions=("write",),
            requires_confirmation=True,
        ),
        _declare(
            "_template_list_dir",
            "template",
            "list_dir",
            "list_dir",
            "List a project directory with bounded depth and result count without invoking a shell.",
            {
                "path": {"type": "string"},
                "depth": {"type": "integer", "minimum": 0, "maximum": 8},
                "max_entries": {"type": "integer", "minimum": 1, "maximum": 5000},
            },
            permissions=("read",),
            concurrency_safe=True,
        ),
        _declare(
            "_make_dir",
            "template",
            "make_dir",
            "make_dir",
            "Create a project directory and any missing parents without invoking a shell.",
            {"path": {"type": "string"}},
            ("path",),
            ("write",),
        ),
        _declare(
            "_template_search_code",
            "template",
            "search_code",
            "search_code",
            "Search project text with ripgrep using separated arguments and bounded output.",
            {
                "query": {"type": "string"},
                "path": {"type": "string"},
                "glob": {"type": "string"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 1000},
            },
            ("query",),
            ("read",),
            concurrency_safe=True,
        ),
        _declare(
            "_template_read_file",
            "template",
            "read_file",
            "read_file",
            (
                "Read a bounded line range from one UTF-8 project file. Each output line is "
                "`padded line number→exact source`; the prefix through → is display metadata, not file content."
            ),
            {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
            },
            ("path",),
            ("read",),
            concurrency_safe=True,
        ),
        _declare(
            "_template_find_files",
            "template",
            "find_files",
            "find_files",
            "Find project files by glob with bounded output.",
            {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 5000},
            },
            permissions=("read",),
            concurrency_safe=True,
        ),
        _declare(
            "_template_git_diff_staged",
            "template",
            "git_diff_staged",
            "git_diff_staged",
            "Show staged Git changes for the project or one project path.",
            {"path": {"type": "string"}},
            permissions=("read",),
            available=shutil.which("git") is not None,
            unavailable_reason="git is not installed",
        ),
        _declare(
            "_template_run_tests",
            "template",
            "run_tests",
            "run_tests",
            "Run the detected project test command from a project directory without shell interpolation.",
            {
                "framework": {
                    "type": "string",
                    "enum": [
                        "auto",
                        "pytest",
                        "npm",
                        "npm:test",
                        "npm:typecheck",
                        "npm:check",
                        "npm:lint",
                        "npm:build",
                        "cargo",
                        "go",
                        "gradle",
                        "maven",
                    ],
                },
                "path": {"type": "string"},
            },
            permissions=("read", "execute"),
            timeout_seconds=int(config.get("tools.template.timeout_seconds", 300)),
        ),
        _declare(
            "_shell_run",
            "shell",
            "run",
            "shell_run",
            "Run a shell command in the current project for inspection, tests, builds, and file operations.",
            {"command": {"type": "string"}, "cwd": cwd_property, "timeout": timeout_property},
            ("command",),
            ("read", "write", "execute"),
            int(config.get("tools.shell.timeout_seconds", 120)),
            available=shutil.which("bash") is not None,
            unavailable_reason="bash is not installed",
            requires_confirmation=True,
        ),
        _declare(
            "_python_run",
            "python",
            "run",
            "python_run",
            "Run a short Python snippet in the current project.",
            {"code": {"type": "string"}, "cwd": cwd_property, "timeout": timeout_property},
            ("code",),
            ("read", "write", "execute"),
            int(config.get("tools.python.timeout_seconds", 120)),
            requires_confirmation=True,
        ),
        _declare(
            "_git_status",
            "git",
            "status",
            "git_status",
            "Show current Git status.",
            {},
            permissions=("read",),
            available=shutil.which("git") is not None,
            unavailable_reason="git is not installed",
        ),
        _declare(
            "_git_diff",
            "git",
            "diff",
            "git_diff",
            "Show unstaged Git changes for the project or one path.",
            {"path": {"type": "string"}},
            permissions=("read",),
        ),
        _declare(
            "_git_log",
            "git",
            "log",
            "git_log",
            "Show recent Git commits.",
            {"limit": {"type": "integer", "minimum": 1, "maximum": 100}},
            permissions=("read",),
        ),
        _declare(
            "_git_add",
            "git",
            "add",
            "git_add",
            "Stage paths in Git.",
            {"paths": {"type": "array", "items": {"type": "string"}}},
            ("paths",),
            ("write",),
            requires_confirmation=True,
        ),
        _declare(
            "_git_commit",
            "git",
            "commit",
            "git_commit",
            "Create a Git commit with the provided message.",
            {"message": {"type": "string"}},
            ("message",),
            ("write",),
            requires_confirmation=True,
        ),
        _declare(
            "_document_parse",
            "document",
            "parse",
            "document_parse",
            "Parse text, PDF, image, or Word content into Markdown, using local OCR when needed.",
            {"path": {"type": "string"}, "ocr": {"type": "boolean"}},
            ("path",),
            ("read",),
            int(config.get("tools.document.timeout_seconds", 180)),
            input_formats=("text", "pdf", "image", "word"),
            output_formats=("markdown",),
        ),
        _declare(
            "_document_render_docx",
            "document",
            "render_docx",
            "document_render_docx",
            "Render bounded Markdown into a Word .docx binary preview. Apply the returned preview_id with file_apply.",
            {
                "path": {"type": "string"},
                "title": {"type": "string"},
                "markdown": {"type": "string"},
            },
            ("path", "title", "markdown"),
            ("write",),
            int(config.get("tools.document.timeout_seconds", 180)),
            input_formats=("markdown",),
            output_formats=("docx-preview",),
        ),
        _declare(
            "_document_parse",
            "ocr",
            "parse",
            "ocr_parse",
            "OCR an image or scanned PDF and return Markdown.",
            {"path": {"type": "string"}, "ocr": {"type": "boolean"}},
            ("path",),
            ("read",),
            int(config.get("tools.ocr.timeout_seconds", 180)),
            input_formats=("pdf", "png", "jpg", "jpeg", "tiff", "webp"),
            output_formats=("markdown",),
        ),
        _declare(
            "_docker_run",
            "docker",
            "run",
            "docker_run",
            "Run Docker CLI arguments without the leading docker word.",
            {"args": {"type": "array", "items": {"type": "string"}}},
            ("args",),
            ("read", "write", "execute"),
            int(config.get("tools.docker.timeout_seconds", 180)),
            available=shutil.which("docker") is not None,
            unavailable_reason="docker CLI/engine is not installed or not on PATH",
            requires_confirmation=True,
        ),
        _declare(
            "_browser_open_url",
            "browser",
            "open_url",
            "browser_open_url",
            "Open an HTTP(S) URL with Playwright and optionally reuse a named persistent session.",
            {"url": {"type": "string"}, "session_name": {"type": "string"}},
            ("url",),
            ("network", "read"),
            int(config.get("tools.browser.timeout_seconds", 180)),
            available=playwright_available,
            unavailable_reason="the Playwright Python package is not installed",
        ),
        _declare(
            "_browser_download",
            "browser",
            "download",
            "browser_download",
            "Open an HTTP(S) page, click a selector that triggers a download, and save it in the project.",
            {
                "url": {"type": "string"},
                "selector": {"type": "string"},
                "session_name": {"type": "string"},
                "filename": {"type": "string"},
            },
            ("url", "selector"),
            ("network", "write"),
            int(config.get("tools.browser.timeout_seconds", 180)),
            available=playwright_available,
            unavailable_reason="the Playwright Python package is not installed",
        ),
        _declare(
            "_browser_list_sessions",
            "browser",
            "list_sessions",
            "browser_list_sessions",
            "List project-local persistent browser sessions and their disk usage.",
            {},
            permissions=("read",),
            available=playwright_available,
            unavailable_reason="the Playwright Python package is not installed",
            concurrency_safe=True,
        ),
        _declare(
            "_browser_close_session",
            "browser",
            "close_session",
            "browser_close_session",
            "Report a named browser session as closed, or permanently clear its cookies and storage.",
            {"session_name": {"type": "string"}, "clear_data": {"type": "boolean"}},
            ("session_name",),
            ("write",),
            available=playwright_available,
            unavailable_reason="the Playwright Python package is not installed",
            requires_confirmation=True,
        ),
        _declare(
            "_http_request",
            "http",
            "request",
            "http_request",
            "Send a bounded GET or POST JSON request to a configured allowlisted domain.",
            {
                "url": {"type": "string"},
                "method": {"type": "string", "enum": ["GET", "POST"]},
                "json_body": {},
                "headers": {"type": "object", "additionalProperties": {"type": "string"}},
                "timeout": {"type": "integer", "minimum": 1, "maximum": 30},
            },
            ("url",),
            ("network", "read", "write"),
            min(int(config.get("tools.http.timeout_seconds", 30)), 30),
            requires_confirmation=True,
        ),
        _declare(
            "_lsp_diagnostics",
            "lsp",
            "diagnostics",
            "lsp_diagnostics",
            "Run bounded Python, JavaScript, or TypeScript diagnostics and return file/line messages.",
            {"path": {"type": "string"}},
            permissions=("read", "execute"),
            timeout_seconds=int(config.get("tools.lsp.timeout_seconds", 60)),
            available=lsp_available,
            unavailable_reason=lsp_reason,
        ),
        _declare(
            "_tool_result_read",
            "tool_result",
            "read",
            "tool_result_read",
            "Read one bounded character chunk from a tool result attachment in the current Session.",
            {
                "request_id": {"type": "string", "maxLength": 512},
                "offset": {"type": "integer", "minimum": 0},
                "max_chars": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": max_result_read_chars,
                },
            },
            ("request_id",),
            ("read",),
            concurrency_safe=True,
        ),
        _declare(
            "_memory_search",
            "memory",
            "search",
            "memory_search",
            "Search project and global long-term memory.",
            {"query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 50}},
            ("query",),
            ("read",),
        ),
        _declare(
            "_memory_add",
            "memory",
            "add",
            "memory_add",
            "Store a durable lesson, correction, reflection, bug, decision, knowledge item, or summary.",
            {
                "kind": {
                    "type": "string",
                    "enum": ["Lesson", "Correction", "Reflection", "Bug", "Decision", "Knowledge", "Summary"],
                },
                "title": {"type": "string"},
                "content": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "global_memory": {"type": "boolean"},
            },
            ("kind", "title", "content"),
            ("write",),
        ),
        _declare(
            "_project_read_context",
            "project",
            "read_context",
            "project_read_context",
            "Read project durable context.md.",
            {},
            permissions=("read",),
            concurrency_safe=True,
        ),
        _declare(
            "_project_write_context",
            "project",
            "write_context",
            "project_write_context",
            "Overwrite project durable context.md with durable facts only.",
            {"content": {"type": "string"}},
            ("content",),
            ("write",),
            requires_confirmation=True,
        ),
        _declare(
            "_agent_update_plan",
            "agent",
            "update_plan",
            "agent_update_plan",
            "Replace the current task plan with concise ordered steps.",
            {
                "steps": {
                    "type": "array",
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "dependencies": {"type": "array", "items": {"type": "string"}},
                            "retry_count": {"type": "integer", "minimum": 0},
                            "max_retries": {"type": "integer", "minimum": 0, "maximum": 10},
                            "allow_parallel": {"type": "boolean"},
                            "completion_criteria": {"type": "string"},
                            "parent_id": {"type": "string"},
                            "step_type": {
                                "type": "string",
                                "enum": [
                                    "scope",
                                    "inspect",
                                    "implement",
                                    "synthesize",
                                    "generate",
                                    "render",
                                    "verify",
                                    "review",
                                    "generic",
                                ],
                            },
                            "estimated_tool_rounds": {"type": "integer", "minimum": 0, "maximum": 64},
                            "artifact_ids": {
                                "type": "array",
                                "maxItems": 32,
                                "items": {"type": "string"},
                            },
                            "validation_rules": {
                                "type": "array",
                                "maxItems": 32,
                                "items": {"type": "string"},
                            },
                            "progress_weight": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                                "maximum": 100,
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress"],
                            },
                        },
                        "required": ["title"],
                        "additionalProperties": False,
                    },
                }
            },
            ("steps",),
            ("state",),
        ),
        _declare(
            "_agent_update_step",
            "agent",
            "update_step",
            "agent_update_step",
            "Update one task-plan step status.",
            {
                "step_id": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "completed", "failed", "skipped"],
                },
            },
            ("step_id", "status"),
            ("state",),
        ),
    )


__all__ = ["CapabilityDeclaration", "builtin_capability_declarations"]
