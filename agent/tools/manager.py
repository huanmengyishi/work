from __future__ import annotations

import hashlib
import inspect
import json
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..config import AppConfig
from ..events import EventBus, sanitize_for_log
from ..memory import MemoryStore
from ..planner import PlanManager
from ..project import Project
from ..state import AgentState
from .base import ToolRequest, ToolResult, _bounded_result_data, _head_tail, elapsed_ms
from .browser import BrowserTool
from .docker import DockerTool
from .document import DocumentTool
from .git import GitTool
from .http import HttpTool
from .file_edit import FileEditTool
from .lsp import LSPManager, SUPPORTED_SUFFIXES
from .mcp import MCPManager
from .pathsafe import resolve_project_path
from .permission import PermissionManager
from .python import PythonTool
from .registry import ToolCapability, ToolCapabilityRegistry
from .result_store import ToolResultStore, ToolResultStoreError
from .shell import ShellTool
from .templates import SafeTemplateTool


ApprovalHandler = Callable[[ToolRequest, ToolCapability, str], bool]


_HEALTH_ERROR_MAX_CHARS = 1000
_EVENT_LABEL_MAX_CHARS = 200
_EVENT_COUNT_MAX = 1000
_EVENT_DURATION_MAX_MS = 86_400_000
_DATE_LITERAL_RE = re.compile(r"(?<!\d)20\d{2}(?:年\s*\d{1,2}月(?:\s*\d{1,2}日)?|[-/.]\d{1,2}(?:[-/.]\d{1,2})?)(?!\d)")
_SENSITIVE_ERROR_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|authorization|cookie|password|passwd|secret|token)"
    r"(\s*[=:]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_URL_CREDENTIALS = re.compile(r"(?i)\b(https?://)[^\s/@:]+:[^\s/@]+@")


@dataclass(frozen=True)
class _ToolExecutionOwnership:
    """Immutable attribution captured before one ToolRequest is evaluated."""

    state: AgentState | None
    session_id: str | None
    run_id: str | None
    events: EventBus | None


class ToolManager:
    def __init__(
        self,
        config: AppConfig,
        project: Project,
        memory: MemoryStore,
        *,
        events: EventBus | None = None,
        approval_handler: ApprovalHandler | None = None,
        auto_approve: bool = False,
        yolo: bool = False,
        super_yolo: bool = False,
    ) -> None:
        # capability_health depends on tools.registry.  Resolve it only after
        # this module is initialized so event_pipelines can be imported first.
        from ..capability_health import CapabilityHealthManager

        self.config = config
        self.project = project
        self.memory = memory
        self.cwd = project.root
        self.events = events
        self.approval_handler = approval_handler
        self.auto_approve = auto_approve
        self.yolo = yolo
        self.super_yolo = super_yolo
        self.state: AgentState | None = None
        self._execution_local = threading.local()
        self.plan_manager = PlanManager()
        self.permission = PermissionManager(config, project.root)
        self.registry = ToolCapabilityRegistry(config)
        self.health = CapabilityHealthManager(config, project.id)
        max_result_bytes = int(config.get("tools.tool_result.max_attachment_bytes", 8_388_608))
        self.result_store = ToolResultStore(
            project.agent_dir,
            max_attachment_bytes=max_result_bytes,
            persist_threshold_bytes=int(config.get("tools.tool_result.persist_threshold_bytes", 12_000)),
            preview_chars=int(config.get("tools.tool_result.preview_chars", 12_000)),
            max_read_chars=int(config.get("tools.tool_result.max_read_chars", 32_000)),
            max_attachments_per_session=int(config.get("tools.tool_result.max_attachments_per_session", 512)),
            max_session_bytes=int(config.get("tools.tool_result.max_session_bytes", 268_435_456)),
        )
        self.shell = ShellTool(
            self.cwd,
            int(config.get("tools.shell.timeout_seconds", 120)),
            max_output_bytes=max_result_bytes,
        )
        self.python = PythonTool(
            self.cwd,
            int(config.get("tools.python.timeout_seconds", 120)),
            max_output_bytes=max_result_bytes,
        )
        self.git = GitTool(
            self.cwd,
            int(config.get("tools.git.timeout_seconds", 120)),
            max_output_bytes=max_result_bytes,
        )
        self.document = DocumentTool(
            self.cwd,
            int(config.get("tools.document.timeout_seconds", 180)),
            max_input_bytes=int(config.get("tools.document.max_input_bytes", 25_000_000)),
            max_result_bytes=max_result_bytes,
        )
        self.docker = DockerTool(
            self.cwd,
            int(config.get("tools.docker.timeout_seconds", 180)),
            max_output_bytes=max_result_bytes,
        )
        self.browser = BrowserTool(
            self.cwd,
            int(config.get("tools.browser.timeout_seconds", 180)),
            max_download_bytes=int(config.get("tools.browser.max_download_bytes", 100_000_000)),
        )
        allowed_domains = config.get("tools.http.allowed_domains", [])
        self.http = HttpTool(
            self.cwd,
            allowed_domains=[str(item) for item in allowed_domains] if isinstance(allowed_domains, list) else [],
            timeout=int(config.get("tools.http.timeout_seconds", 30)),
            max_response_bytes=int(config.get("tools.http.max_response_bytes", 1_048_576)),
        )
        self.lsp = LSPManager(
            self.cwd,
            timeout=int(config.get("tools.lsp.timeout_seconds", 60)),
            max_diagnostics=int(config.get("tools.lsp.max_diagnostics", 200)),
        )
        self.file_edit = FileEditTool(
            project,
            int(config.get("tools.file.max_file_bytes", 2_000_000)),
        )
        self.templates = SafeTemplateTool(
            self.cwd,
            int(config.get("tools.template.timeout_seconds", 300)),
            max_input_bytes=int(config.get("tools.template.max_input_bytes", 67_108_864)),
            max_result_bytes=max_result_bytes,
        )
        self.mcp = MCPManager(config, self.cwd)
        self._register_capabilities()
        self._register_mcp_capabilities()
        self._apply_capability_timeouts()

    def bind_state(self, state: AgentState) -> None:
        self.state = state

    def set_event_bus(self, events: EventBus) -> None:
        self.events = events

    def schemas(self) -> list[dict[str, Any]]:
        return [
            item.schema()
            for item in self.capabilities(enabled_only=True)
            if self.health.evaluate(item).status == "Available"
        ]

    def capabilities(self, *, enabled_only: bool = False) -> list[ToolCapability]:
        return self.registry.capabilities(enabled_only=enabled_only)

    def model_function_name(self, name: str) -> str:
        """Return the advertised model function name for any accepted alias."""

        capability, _handler = self.registry.resolve(str(name))
        return capability.model_name if capability is not None else str(name)

    def health_report(self) -> list:
        return self.health.report(self.capabilities(enabled_only=False))

    def capability_summary(self) -> str:
        lines = []
        for item in self.capabilities(enabled_only=True):
            if self.health.evaluate(item).status != "Available":
                continue
            permissions = ", ".join(item.permissions) or "none"
            formats = ""
            if item.input_formats or item.output_formats:
                formats = (
                    f"; input={','.join(item.input_formats) or '-'}; output={','.join(item.output_formats) or '-'}"
                )
            lines.append(
                f"- `{item.name}` as `{item.model_name}`: permissions={permissions}; "
                f"timeout={item.timeout_seconds}s; stream={str(item.supports_stream).lower()}; "
                f"confirm={str(item.requires_confirmation).lower()}{formats}"
            )
        return "\n".join(lines) or "No tool capabilities are enabled."

    def execute_model_call(
        self,
        name: str,
        arguments: str | dict[str, Any] | None,
        *,
        request_id: str | None = None,
        runtime_denied_reason: str | None = None,
    ) -> tuple[ToolRequest, ToolResult]:
        try:
            args = parse_arguments(arguments)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            request = self.registry.request(name, {}, request_id=request_id)
            return request, self.execute(
                request,
                runtime_denied_reason=runtime_denied_reason,
                argument_error=f"invalid arguments for {name}: {exc}",
            )
        request = self.registry.request(name, args, request_id=request_id)
        return request, self.execute(request, runtime_denied_reason=runtime_denied_reason)

    def capture_model_call_context(self) -> object:
        """Capture batch ownership before work can move to a worker thread."""

        return self._capture_execution_ownership()

    def execute_model_call_in_context(
        self,
        context: object,
        name: str,
        arguments: str | dict[str, Any] | None,
        *,
        request_id: str | None = None,
        runtime_denied_reason: str | None = None,
    ) -> tuple[ToolRequest, ToolResult]:
        """Execute one model call under ownership captured by its batch."""

        if not isinstance(context, _ToolExecutionOwnership):
            raise TypeError("invalid model tool execution context")
        previous = getattr(self._execution_local, "ownership", None)
        self._execution_local.ownership = context
        try:
            return self.execute_model_call(
                name,
                arguments,
                request_id=request_id,
                runtime_denied_reason=runtime_denied_reason,
            )
        finally:
            if previous is None:
                try:
                    del self._execution_local.ownership
                except AttributeError:
                    pass
            else:
                self._execution_local.ownership = previous

    def execute(
        self,
        request: ToolRequest,
        *,
        runtime_denied_reason: str | None = None,
        argument_error: str | None = None,
    ) -> ToolResult:
        ownership = self._capture_execution_ownership()
        previous = getattr(self._execution_local, "ownership", None)
        self._execution_local.ownership = ownership
        try:
            return self._execute_owned(
                request,
                runtime_denied_reason=runtime_denied_reason,
                argument_error=argument_error,
                ownership=ownership,
            )
        finally:
            if previous is None:
                try:
                    del self._execution_local.ownership
                except AttributeError:
                    pass
            else:
                self._execution_local.ownership = previous

    def _execute_owned(
        self,
        request: ToolRequest,
        *,
        runtime_denied_reason: str | None,
        argument_error: str | None,
        ownership: _ToolExecutionOwnership,
    ) -> ToolResult:
        capability, handler = self.registry.resolve(request.capability)
        if capability is None or handler is None:
            if runtime_denied_reason:
                result = self._runtime_denied_result(request, runtime_denied_reason)
                self._publish("tool.denied", request, result, ownership=ownership)
                return result
            return self._not_executed_result(
                request,
                f"unknown tool capability: {request.capability}",
            )
        decision = self.permission.evaluate(request, capability, super_yolo=self.super_yolo)
        if runtime_denied_reason:
            result = self._runtime_denied_result(request, runtime_denied_reason)
            self._publish("tool.denied", request, result, ownership=ownership)
            return result
        if argument_error:
            result = self._not_executed_result(request, argument_error)
            self._publish("tool.denied", request, result, ownership=ownership)
            return result
        if not decision.allowed:
            result = self._not_executed_result(request, decision.reason)
            self._publish("tool.denied", request, result, ownership=ownership)
            return result

        handler_argument_error = self._handler_argument_error(handler, request.args, capability.name)
        if handler_argument_error:
            result = self._not_executed_result(request, handler_argument_error)
            self._publish("tool.denied", request, result, ownership=ownership)
            return result

        auto_approved = (
            self.super_yolo or self.yolo or (self.auto_approve and capability.name in self._auto_approve_capabilities())
        )
        if capability.requires_confirmation and not auto_approved:
            try:
                summary = self._approval_summary(request, capability)
            except Exception as exc:
                result = self._not_executed_result(request, f"could not prepare approval: {exc}")
                self._publish("tool.denied", request, result, ownership=ownership)
                return result
            if self.approval_handler is None:
                result = self._not_executed_result(
                    request,
                    "operation requires user confirmation; use interactive mode or --yolo",
                )
                self._publish("tool.denied", request, result, ownership=ownership)
                return result
            try:
                approved = self.approval_handler(request, capability, summary)
            except Exception as exc:
                result = self._not_executed_result(request, f"approval failed: {exc}")
                self._publish("tool.denied", request, result, ownership=ownership)
                return result
            if not approved:
                result = self._not_executed_result(request, "operation denied by user")
                self._publish("tool.denied", request, result, ownership=ownership)
                return result

        self._publish("tool.started", request, None, ownership=ownership)
        started = time.monotonic()
        try:
            result = handler(**request.args)
            if not isinstance(result, ToolResult):
                result = ToolResult(True, str(result))
        except Exception as exc:
            result = ToolResult(False, "", str(exc))
        handler_result = result.with_execution(request_id=request.request_id, duration_ms=result.duration_ms)
        if capability.name != "tool_result.read" and ownership.session_id is not None:
            try:
                result = self.result_store.persist(
                    handler_result,
                    session_id=ownership.session_id,
                    request_id=request.request_id,
                )
            except ToolResultStoreError as exc:
                # The handler has already completed and may have performed an
                # approved side effect. Attachment failure must not rewrite
                # that fact as a failed tool execution, which could cause the
                # model to repeat a write, command, or network request.
                result = self._attachment_persistence_fallback(handler_result, exc)
        else:
            result = handler_result
        result = result.with_execution(request_id=request.request_id, duration_ms=elapsed_ms(started))
        self._publish("tool.finished", request, result, ownership=ownership)
        return result

    def _capture_execution_ownership(self) -> _ToolExecutionOwnership:
        current = getattr(self._execution_local, "ownership", None)
        if isinstance(current, _ToolExecutionOwnership):
            return current
        state = self.state
        return _ToolExecutionOwnership(
            state=state,
            session_id=state.session_id if state is not None else None,
            run_id=state.run_id if state is not None else None,
            events=self.events,
        )

    def _attachment_persistence_fallback(
        self,
        result: ToolResult,
        error: ToolResultStoreError,
    ) -> ToolResult:
        """Return a bounded truthful result when its private attachment fails."""

        if result.success:
            stdout_chars = self.result_store.preview_chars * 3 // 4
            stderr_chars = self.result_store.preview_chars - stdout_chars
        else:
            stderr_chars = self.result_store.preview_chars * 3 // 4
            stdout_chars = self.result_store.preview_chars - stderr_chars
        bounded_data = _bounded_result_data(result.data or {})
        data = dict(bounded_data) if isinstance(bounded_data, dict) else {"value": bounded_data}
        data["attachment_persistence_error"] = {
            "type": type(error).__name__,
            "message": self._health_error_summary(ToolResult(False, "", str(error))),
            "result_preserved": True,
            "full_body_available": False,
            "stdout_chars": len(result.stdout),
            "stderr_chars": len(result.stderr),
            "stdout_sha256": hashlib.sha256(result.stdout.encode("utf-8", errors="replace")).hexdigest(),
            "stderr_sha256": hashlib.sha256(result.stderr.encode("utf-8", errors="replace")).hexdigest(),
        }
        return ToolResult(
            result.success,
            _head_tail(result.stdout, stdout_chars),
            _head_tail(result.stderr, stderr_chars),
            data=data,
            duration_ms=result.duration_ms,
            request_id=result.request_id,
        )

    @staticmethod
    def _runtime_denied_result(request: ToolRequest, reason: str) -> ToolResult:
        return ToolManager._not_executed_result(
            request,
            str(reason)[:2_000],
            data={"runtime_denied": True},
        )

    @staticmethod
    def _not_executed_result(
        request: ToolRequest,
        error: str,
        *,
        data: dict[str, Any] | None = None,
    ) -> ToolResult:
        result_data = dict(data or {})
        result_data["not_executed"] = True
        return ToolResult(
            False,
            "",
            str(error),
            data=result_data,
            request_id=request.request_id,
        )

    @staticmethod
    def _handler_argument_error(
        handler: Callable[..., ToolResult],
        arguments: dict[str, Any],
        capability_name: str,
    ) -> str | None:
        """Return a binding error without entering the handler body."""

        try:
            signature = inspect.signature(handler)
        except (TypeError, ValueError):
            return None
        try:
            signature.bind(**arguments)
        except TypeError as exc:
            return f"invalid arguments for {capability_name}: {exc}"
        return None

    def call(self, name: str, arguments: str | dict[str, Any] | None) -> ToolResult:
        _, result = self.execute_model_call(name, arguments)
        return result

    def _register_capabilities(self) -> None:
        cwd_property = {
            "type": "string",
            "description": "Optional working directory inside the current project. Defaults to project root.",
        }
        timeout_property = {"type": "integer", "minimum": 1, "description": "Optional timeout seconds."}
        registrations = [
            (
                ToolCapability(
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
                self._file_diff,
            ),
            (
                ToolCapability(
                    "file",
                    "apply",
                    "file_apply",
                    "Apply a previously created file_diff preview atomically after snapshotting the original file.",
                    {"preview_id": {"type": "string"}},
                    ("preview_id",),
                    ("write",),
                    requires_confirmation=True,
                ),
                self._file_apply,
            ),
            (
                ToolCapability(
                    "file",
                    "undo",
                    "file_undo",
                    "Undo the latest active file snapshot in this session, or a selected snapshot.",
                    {"snapshot_id": {"type": "string"}},
                    permissions=("write",),
                    requires_confirmation=True,
                ),
                self._file_undo,
            ),
            (
                ToolCapability(
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
                self.templates.list_dir,
            ),
            (
                ToolCapability(
                    "template",
                    "make_dir",
                    "make_dir",
                    "Create a project directory and any missing parents without invoking a shell.",
                    {"path": {"type": "string"}},
                    ("path",),
                    ("write",),
                ),
                self._make_dir,
            ),
            (
                ToolCapability(
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
                self.templates.search_code,
            ),
            (
                ToolCapability(
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
                self.templates.read_file,
            ),
            (
                ToolCapability(
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
                self.templates.find_files,
            ),
            (
                ToolCapability(
                    "template",
                    "git_diff_staged",
                    "git_diff_staged",
                    "Show staged Git changes for the project or one project path.",
                    {"path": {"type": "string"}},
                    permissions=("read",),
                    available=shutil.which("git") is not None,
                    unavailable_reason="git is not installed",
                ),
                self.templates.git_diff_staged,
            ),
            (
                ToolCapability(
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
                    timeout_seconds=int(self.config.get("tools.template.timeout_seconds", 300)),
                ),
                self.templates.run_tests,
            ),
            (
                ToolCapability(
                    "shell",
                    "run",
                    "shell_run",
                    "Run a shell command in the current project for inspection, tests, builds, and file operations.",
                    {"command": {"type": "string"}, "cwd": cwd_property, "timeout": timeout_property},
                    ("command",),
                    ("read", "write", "execute"),
                    int(self.config.get("tools.shell.timeout_seconds", 120)),
                    available=shutil.which("bash") is not None,
                    unavailable_reason="bash is not installed",
                    requires_confirmation=True,
                ),
                self._shell_run,
            ),
            (
                ToolCapability(
                    "python",
                    "run",
                    "python_run",
                    "Run a short Python snippet in the current project.",
                    {"code": {"type": "string"}, "cwd": cwd_property, "timeout": timeout_property},
                    ("code",),
                    ("read", "write", "execute"),
                    int(self.config.get("tools.python.timeout_seconds", 120)),
                    requires_confirmation=True,
                ),
                self._python_run,
            ),
            (
                ToolCapability(
                    "git",
                    "status",
                    "git_status",
                    "Show current Git status.",
                    {},
                    permissions=("read",),
                    available=shutil.which("git") is not None,
                    unavailable_reason="git is not installed",
                ),
                self.git.status,
            ),
            (
                ToolCapability(
                    "git",
                    "diff",
                    "git_diff",
                    "Show unstaged Git changes for the project or one path.",
                    {"path": {"type": "string"}},
                    permissions=("read",),
                ),
                self._git_diff,
            ),
            (
                ToolCapability(
                    "git",
                    "log",
                    "git_log",
                    "Show recent Git commits.",
                    {"limit": {"type": "integer", "minimum": 1, "maximum": 100}},
                    permissions=("read",),
                ),
                self._git_log,
            ),
            (
                ToolCapability(
                    "git",
                    "add",
                    "git_add",
                    "Stage paths in Git.",
                    {"paths": {"type": "array", "items": {"type": "string"}}},
                    ("paths",),
                    ("write",),
                    requires_confirmation=True,
                ),
                self._git_add,
            ),
            (
                ToolCapability(
                    "git",
                    "commit",
                    "git_commit",
                    "Create a Git commit with the provided message.",
                    {"message": {"type": "string"}},
                    ("message",),
                    ("write",),
                    requires_confirmation=True,
                ),
                self._git_commit,
            ),
            (
                ToolCapability(
                    "document",
                    "parse",
                    "document_parse",
                    "Parse text, PDF, image, or Word content into Markdown, using local OCR when needed.",
                    {"path": {"type": "string"}, "ocr": {"type": "boolean"}},
                    ("path",),
                    ("read",),
                    int(self.config.get("tools.document.timeout_seconds", 180)),
                    input_formats=("text", "pdf", "image", "word"),
                    output_formats=("markdown",),
                ),
                self._document_parse,
            ),
            (
                ToolCapability(
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
                    int(self.config.get("tools.document.timeout_seconds", 180)),
                    input_formats=("markdown",),
                    output_formats=("docx-preview",),
                ),
                self._document_render_docx,
            ),
            (
                ToolCapability(
                    "ocr",
                    "parse",
                    "ocr_parse",
                    "OCR an image or scanned PDF and return Markdown.",
                    {"path": {"type": "string"}, "ocr": {"type": "boolean"}},
                    ("path",),
                    ("read",),
                    int(self.config.get("tools.ocr.timeout_seconds", 180)),
                    input_formats=("pdf", "png", "jpg", "jpeg", "tiff", "webp"),
                    output_formats=("markdown",),
                ),
                self._document_parse,
            ),
            (
                ToolCapability(
                    "docker",
                    "run",
                    "docker_run",
                    "Run Docker CLI arguments without the leading docker word.",
                    {"args": {"type": "array", "items": {"type": "string"}}},
                    ("args",),
                    ("read", "write", "execute"),
                    int(self.config.get("tools.docker.timeout_seconds", 180)),
                    available=shutil.which("docker") is not None,
                    unavailable_reason="docker CLI/engine is not installed or not on PATH",
                    requires_confirmation=True,
                ),
                self._docker_run,
            ),
            (
                ToolCapability(
                    "browser",
                    "open_url",
                    "browser_open_url",
                    "Open an HTTP(S) URL with Playwright and optionally reuse a named persistent session.",
                    {"url": {"type": "string"}, "session_name": {"type": "string"}},
                    ("url",),
                    ("network", "read"),
                    int(self.config.get("tools.browser.timeout_seconds", 180)),
                    available=self._module_available("playwright"),
                    unavailable_reason="the Playwright Python package is not installed",
                ),
                self._browser_open_url,
            ),
            (
                ToolCapability(
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
                    int(self.config.get("tools.browser.timeout_seconds", 180)),
                    available=self._module_available("playwright"),
                    unavailable_reason="the Playwright Python package is not installed",
                ),
                self._browser_download,
            ),
            (
                ToolCapability(
                    "browser",
                    "list_sessions",
                    "browser_list_sessions",
                    "List project-local persistent browser sessions and their disk usage.",
                    {},
                    permissions=("read",),
                    available=self._module_available("playwright"),
                    unavailable_reason="the Playwright Python package is not installed",
                    concurrency_safe=True,
                ),
                self.browser.list_sessions,
            ),
            (
                ToolCapability(
                    "browser",
                    "close_session",
                    "browser_close_session",
                    "Report a named browser session as closed, or permanently clear its cookies and storage.",
                    {"session_name": {"type": "string"}, "clear_data": {"type": "boolean"}},
                    ("session_name",),
                    ("write",),
                    available=self._module_available("playwright"),
                    unavailable_reason="the Playwright Python package is not installed",
                    requires_confirmation=True,
                ),
                self._browser_close_session,
            ),
            (
                ToolCapability(
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
                    min(int(self.config.get("tools.http.timeout_seconds", 30)), 30),
                    requires_confirmation=True,
                ),
                self.http.request,
            ),
            (
                ToolCapability(
                    "lsp",
                    "diagnostics",
                    "lsp_diagnostics",
                    "Run bounded Python, JavaScript, or TypeScript diagnostics and return file/line messages.",
                    {"path": {"type": "string"}},
                    permissions=("read", "execute"),
                    timeout_seconds=int(self.config.get("tools.lsp.timeout_seconds", 60)),
                    available=self.lsp.available()[0],
                    unavailable_reason=self.lsp.available()[1],
                ),
                self.lsp.diagnostics,
            ),
            (
                ToolCapability(
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
                            "maximum": self.result_store.max_read_chars,
                        },
                    },
                    ("request_id",),
                    ("read",),
                    concurrency_safe=True,
                ),
                self._tool_result_read,
            ),
            (
                ToolCapability(
                    "memory",
                    "search",
                    "memory_search",
                    "Search project and global long-term memory.",
                    {"query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 50}},
                    ("query",),
                    ("read",),
                ),
                self._memory_search,
            ),
            (
                ToolCapability(
                    "memory",
                    "add",
                    "memory_add",
                    "Store a durable lesson, correction, reflection, bug, decision, knowledge item, or summary.",
                    {
                        "kind": {
                            "type": "string",
                            "enum": [
                                "Lesson",
                                "Correction",
                                "Reflection",
                                "Bug",
                                "Decision",
                                "Knowledge",
                                "Summary",
                            ],
                        },
                        "title": {"type": "string"},
                        "content": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "global_memory": {"type": "boolean"},
                    },
                    ("kind", "title", "content"),
                    ("write",),
                ),
                self._memory_add,
            ),
            (
                ToolCapability(
                    "project",
                    "read_context",
                    "project_read_context",
                    "Read project durable context.md.",
                    {},
                    permissions=("read",),
                    concurrency_safe=True,
                ),
                self._project_read_context,
            ),
            (
                ToolCapability(
                    "project",
                    "write_context",
                    "project_write_context",
                    "Overwrite project durable context.md with durable facts only.",
                    {"content": {"type": "string"}},
                    ("content",),
                    ("write",),
                    requires_confirmation=True,
                ),
                self._project_write_context,
            ),
            (
                ToolCapability(
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
                                    "status": {
                                        "type": "string",
                                        "enum": ["pending", "in_progress", "completed", "failed", "skipped"],
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
                self._agent_update_plan,
            ),
            (
                ToolCapability(
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
                self._agent_update_step,
            ),
        ]
        for capability, handler in registrations:
            self.registry.register(capability, handler)

    def _register_mcp_capabilities(self) -> None:
        for capability, handler in self.mcp.discover():
            self.registry.register(capability, handler)

    def close(self) -> None:
        self.mcp.close()

    def _apply_capability_timeouts(self) -> None:
        mappings = (
            (self.shell, "shell.run"),
            (self.python, "python.run"),
            (self.git, "git.status"),
            (self.document, "document.parse"),
            (self.docker, "docker.run"),
            (self.browser, "browser.open_url"),
            (self.templates, "template.run_tests"),
            (self.lsp, "lsp.diagnostics"),
        )
        for tool, capability_name in mappings:
            capability, _ = self.registry.resolve(capability_name)
            if capability is not None:
                tool.timeout = capability.timeout_seconds

    @staticmethod
    def _module_available(name: str) -> bool:
        try:
            __import__(name)
        except Exception:
            return False
        return True

    def _shell_run(self, command: str, cwd: str | None = None, timeout: int | None = None) -> ToolResult:
        return self.shell.run(command, cwd=self._resolve_cwd(cwd), timeout=timeout)

    def _python_run(self, code: str, cwd: str | None = None, timeout: int | None = None) -> ToolResult:
        return self.python.run(code, cwd=self._resolve_cwd(cwd), timeout=timeout)

    def _file_diff(
        self,
        path: str,
        content: str | None = None,
        old_text: str | None = None,
        new_text: str | None = None,
        replace_all: bool = False,
        delete: bool = False,
    ) -> ToolResult:
        return self.file_edit.preview(
            path=path,
            session_id=self._require_state().session_id,
            content=content,
            old_text=old_text,
            new_text=new_text,
            replace_all=replace_all,
            delete=delete,
        )

    def _file_apply(self, preview_id: str) -> ToolResult:
        result = self.file_edit.apply(preview_id=preview_id, session_id=self._require_state().session_id)
        path = str((result.data or {}).get("path") or "")
        if (
            result.success
            and bool(self.config.get("tools.lsp.auto_after_file_apply", True))
            and Path(path).suffix.lower() in SUPPORTED_SUFFIXES
        ):
            diagnostics = self.lsp.diagnostics(path)
            data = dict(result.data or {})
            data["lsp"] = diagnostics.data or {
                "success": diagnostics.success,
                "stdout": diagnostics.stdout,
                "stderr": diagnostics.stderr,
            }
            diagnostic_output = diagnostics.stdout or diagnostics.stderr
            return ToolResult(
                True,
                "\n".join(part for part in (result.stdout, diagnostic_output) if part),
                "",
                data=data,
            )
        return result

    def _file_undo(self, snapshot_id: str | None = None) -> ToolResult:
        return self.file_edit.undo(session_id=self._require_state().session_id, snapshot_id=snapshot_id)

    def _make_dir(self, path: str) -> ToolResult:
        try:
            target = resolve_project_path(self.project.root, path)
            target.mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError) as exc:
            return ToolResult(False, "", str(exc))
        return ToolResult(
            True,
            f"created directory {target.relative_to(self.project.root)}",
            data={"path": target.relative_to(self.project.root).as_posix()},
        )

    def _resolve_cwd(self, cwd: str | None) -> str | None:
        if not cwd:
            return None
        path = Path(cwd).expanduser()
        return str(path if path.is_absolute() else self.project.root / path)

    def _git_diff(self, path: str | None = None) -> ToolResult:
        return self.git.diff(path)

    def _git_log(self, limit: int = 10) -> ToolResult:
        return self.git.log(limit)

    def _git_add(self, paths: list[str]) -> ToolResult:
        return self.git.add(paths)

    def _git_commit(self, message: str) -> ToolResult:
        return self.git.commit(message)

    def _document_parse(self, path: str, ocr: bool = True) -> ToolResult:
        result = self.document.parse(path, ocr=ocr)
        data = dict(result.data or {})
        data["date_literals"] = sorted(set(_DATE_LITERAL_RE.findall(result.stdout)))[:100]
        return ToolResult(
            result.success,
            result.stdout,
            result.stderr,
            data=data,
            duration_ms=result.duration_ms,
            request_id=result.request_id,
        )

    def _document_render_docx(self, path: str, title: str, markdown: str) -> ToolResult:
        if Path(path).suffix.lower() != ".docx":
            return ToolResult(False, "", "document_render_docx path must end with .docx")
        markdown_limit = max(
            1,
            min(int(self.config.get("tools.document.max_render_chars", 250_000)), 1_000_000),
        )
        if len(markdown) > markdown_limit:
            return ToolResult(False, "", f"document_render_docx markdown exceeds {markdown_limit} characters")
        try:
            content, metadata = self.document.render_docx(title=title, markdown=markdown)
            preview = self.file_edit.preview_binary(
                path=path,
                session_id=self._require_state().session_id,
                content=content,
                source="document.render_docx",
            )
        except Exception as exc:
            return ToolResult(False, "", str(exc))
        data = dict(preview.data or {})
        data.update(metadata)
        data["date_literals"] = sorted(set(_DATE_LITERAL_RE.findall(markdown)))[:100]
        data["generated_metadata_dates"] = sorted(
            {
                date
                for line in markdown.splitlines()
                if ("生成" in line or "汇总" in line or "报告" in line) and ("时间" in line or "日期" in line)
                for date in _DATE_LITERAL_RE.findall(line)
            }
        )[:100]
        return ToolResult(preview.success, preview.stdout, preview.stderr, data=data)

    def _docker_run(self, args: list[str]) -> ToolResult:
        return self.docker.run(args)

    def _browser_open_url(self, url: str, session_name: str | None = None) -> ToolResult:
        return self.browser.open_url(url, session_name=session_name)

    def _browser_download(
        self,
        url: str,
        selector: str,
        session_name: str | None = None,
        filename: str | None = None,
    ) -> ToolResult:
        return self.browser.download(url, selector, session_name=session_name, filename=filename)

    def _browser_close_session(self, session_name: str, clear_data: bool = False) -> ToolResult:
        return self.browser.close_session(session_name, clear_data=clear_data)

    def _memory_search(self, query: str, limit: int | None = None) -> ToolResult:
        items = self.memory.search(query, self.project.id, limit=limit)
        data = [
            {
                "id": item.id,
                "project_id": item.project_id,
                "kind": item.kind,
                "title": item.title,
                "content": item.content,
                "tags": item.tags,
                "updated_at": item.updated_at,
            }
            for item in items
        ]
        return ToolResult(True, json.dumps(data, ensure_ascii=False, indent=2), data={"items": data})

    def _memory_add(
        self,
        kind: str,
        title: str,
        content: str,
        tags: list[str] | None = None,
        global_memory: bool = False,
    ) -> ToolResult:
        normalized_tags = [str(tag).strip() for tag in (tags or []) if str(tag).strip()]
        if kind == "Correction":
            if not any(tag.startswith("correction:") for tag in normalized_tags):
                return ToolResult(False, "", "Correction memory requires a correction:<topic> tag")
            if self.project.name not in normalized_tags:
                normalized_tags.append(self.project.name)
        project_id = None if global_memory else self.project.id
        memory_id = self.memory.add_memory(
            kind=kind,
            title=title,
            content=content,
            tags=normalized_tags,
            project_id=project_id,
        )
        self.memory.persist_lesson_file(
            kind=kind,
            title=title,
            content=content,
            project=self.project,
            global_memory=global_memory,
        )
        return ToolResult(True, f"memory added: {memory_id}", data={"id": memory_id})

    def _project_read_context(self) -> ToolResult:
        return ToolResult(True, self.project.context_path.read_text(encoding="utf-8"))

    def _tool_result_read(
        self,
        request_id: str,
        offset: int = 0,
        max_chars: int | None = None,
    ) -> ToolResult:
        try:
            chunk = self.result_store.read_chunk(
                session_id=self._require_state().session_id,
                request_id=request_id,
                offset=offset,
                max_chars=max_chars,
            )
        except ToolResultStoreError as exc:
            return ToolResult(False, "", str(exc))
        return ToolResult(
            True,
            chunk.content,
            data={
                "request_id": chunk.request_id,
                "offset": chunk.offset,
                "returned_chars": len(chunk.content),
                "next_offset": chunk.next_offset,
                "total_chars": chunk.total_chars,
                "bytes": chunk.total_bytes,
                "sha256": chunk.sha256,
                "eof": chunk.eof,
            },
        )

    def _project_write_context(self, content: str) -> ToolResult:
        temp = self.project.context_path.with_suffix(".md.tmp")
        temp.write_text(content.rstrip() + "\n", encoding="utf-8")
        temp.replace(self.project.context_path)
        return ToolResult(True, f"wrote {self.project.context_path}")

    def _agent_update_plan(self, steps: list[str | dict[str, Any]]) -> ToolResult:
        state = self._require_state()
        if state.plan and any(not state.plan_step_satisfied(step) for step in state.plan):
            return ToolResult(
                False,
                "",
                "a Task Graph already exists; complete its steps with agent_update_step instead of replacing it",
            )
        if len(steps) > 8:
            return ToolResult(False, "", "agent_update_plan accepts at most 8 bounded steps")
        try:
            plan = self.plan_manager.replace(state, steps)
        except (TypeError, ValueError) as exc:
            return ToolResult(False, "", str(exc))
        data = {
            "steps": [item.__dict__ for item in plan],
            "ready_steps": [item.id for item in self.plan_manager.ready_steps(state)],
        }
        return ToolResult(True, json.dumps(data, ensure_ascii=False, indent=2), data=data)

    def _agent_update_step(self, step_id: str, status: str) -> ToolResult:
        state = self._require_state()
        try:
            step = self.plan_manager.update_step(state, step_id, status)
        except ValueError as exc:
            return ToolResult(False, "", str(exc))
        data = {
            "step": step.__dict__,
            "ready_steps": [item.id for item in self.plan_manager.ready_steps(state)],
        }
        return ToolResult(True, json.dumps(data, ensure_ascii=False, indent=2), data=data)

    def _require_state(self) -> AgentState:
        ownership = getattr(self._execution_local, "ownership", None)
        if isinstance(ownership, _ToolExecutionOwnership) and ownership.state is not None:
            return ownership.state
        if self.state is None:
            raise RuntimeError("agent state is not bound to ToolManager")
        return self.state

    def _approval_summary(self, request: ToolRequest, capability: ToolCapability) -> str:
        if capability.name == "file.apply":
            return self.file_edit.approval_summary(str(request.args.get("preview_id") or ""))
        if capability.name == "file.undo":
            snapshot = str(request.args.get("snapshot_id") or "latest")
            return f"Undo file snapshot {snapshot} for the active session?"
        if capability.name == "shell.run":
            return f"Run unstructured shell command without an automatic file snapshot?\n\n{request.args.get('command', '')}"
        if capability.name == "python.run":
            return f"Run unstructured Python code without an automatic file snapshot?\n\n{request.args.get('code', '')}"
        if capability.name == "docker.run":
            return f"Run Docker arguments without an automatic project snapshot?\n\n{request.args.get('args', [])}"
        if capability.name.startswith("mcp."):
            keys = ", ".join(sorted(str(key) for key in request.args)) or "none"
            return f"Call external capability {capability.name}? Argument names: {keys}. Values are hidden."
        return f"Allow {capability.name}?"

    def _auto_approve_capabilities(self) -> set[str]:
        values = self.config.get("permissions.auto_approve_capabilities", ["file.apply", "file.undo"])
        if not isinstance(values, list):
            return {"file.apply", "file.undo"}
        return {str(item) for item in values}

    @staticmethod
    def _is_health_failure(result: ToolResult) -> bool:
        error = str(result.stderr or "").lower()
        markers = (
            "timeout",
            "timed out",
            "command not found",
            "dependency",
            "unavailable",
            "could not start",
            "connection refused",
            "connection reset",
            "not installed",
            "closed its input",
            "broken pipe",
        )
        return any(marker in error for marker in markers)

    @staticmethod
    def _health_error_summary(result: ToolResult) -> str:
        value = str(sanitize_for_log(str(result.stderr or "")))
        value = _URL_CREDENTIALS.sub(r"\1[redacted]@", value)
        value = _SENSITIVE_ERROR_ASSIGNMENT.sub(
            lambda match: f"{match.group(1)}{match.group(2)}[redacted]",
            value,
        )
        value = " ".join(value.replace("\x00", " ").split())
        if len(value) <= _HEALTH_ERROR_MAX_CHARS:
            return value
        suffix = "...[truncated]"
        return value[: _HEALTH_ERROR_MAX_CHARS - len(suffix)] + suffix

    @staticmethod
    def _event_label(value: Any) -> str | None:
        if value is None:
            return None
        label = str(sanitize_for_log(str(value))).strip()
        if len(label) <= _EVENT_LABEL_MAX_CHARS:
            return label
        suffix = "...[truncated]"
        return label[: _EVENT_LABEL_MAX_CHARS - len(suffix)] + suffix

    @staticmethod
    def _event_count(value: Any) -> int:
        try:
            count = len(value)
        except (TypeError, ValueError, OverflowError):
            return 0
        return max(0, min(count, _EVENT_COUNT_MAX))

    @staticmethod
    def _event_duration_ms(value: Any) -> int:
        try:
            duration = int(value or 0)
        except (TypeError, ValueError, OverflowError):
            return 0
        return max(0, min(duration, _EVENT_DURATION_MAX_MS))

    def _publish(
        self,
        event_name: str,
        request: ToolRequest,
        result: ToolResult | None,
        *,
        ownership: _ToolExecutionOwnership,
    ) -> None:
        if not ownership.events:
            return
        payload: dict[str, Any] = {
            "request": {
                "tool": self._event_label(request.tool),
                "action": self._event_label(request.action),
                "capability": self._event_label(request.capability),
                "request_id": self._event_label(request.request_id),
                "argument_count": self._event_count(request.args),
            }
        }
        if result is not None:
            health_failure = event_name == "tool.finished" and not result.success and self._is_health_failure(result)
            payload["result"] = {
                "success": result.success,
                "duration_ms": self._event_duration_ms(result.duration_ms),
                "request_id": self._event_label(result.request_id),
                "data_field_count": self._event_count(result.data) if isinstance(result.data, dict) else 0,
                "health_failure": health_failure,
                "error": self._health_error_summary(result) if health_failure else "",
            }
        ownership.events.publish(
            event_name,
            payload,
            project_id=self.project.id,
            session_id=ownership.session_id,
            run_id=ownership.run_id,
        )


def parse_arguments(arguments: str | dict[str, Any] | None) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        raise TypeError("tool arguments must be a JSON object or string")
    if not arguments.strip():
        return {}
    parsed = json.loads(arguments)
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must decode to an object")
    return parsed
