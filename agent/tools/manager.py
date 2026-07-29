from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

from ..capability_health import CapabilityHealthManager
from ..config import AppConfig
from ..events import EventBus
from ..global_knowledge import GlobalKnowledgeBase
from ..memory import MemoryStore
from ..planner import PlanManager
from ..project import Project
from ..state import AgentState
from .base import ToolRequest, ToolResult
from .browser import BrowserTool
from .capabilities import builtin_capability_declarations
from .docker import DockerTool
from .document import DocumentTool
from .executor import ApprovalHandler, ToolExecutionOwnership, ToolExecutor
from .file_edit import FileEditTool
from .git import GitTool
from .http import HttpTool
from .lsp import LSPManager, SUPPORTED_SUFFIXES
from .mcp import MCPManager, MCPServerStatus
from .pathsafe import resolve_project_path
from .permission import PermissionManager
from .python import PythonTool
from .registry import ToolCapability, ToolCapabilityRegistry
from .result_store import ToolResultStore, ToolResultStoreError
from .shell import ShellTool
from .templates import SafeTemplateTool

_DATE_LITERAL_RE = re.compile(r"(?<!\d)20\d{2}(?:年\s*\d{1,2}月(?:\s*\d{1,2}日)?|[-/.]\d{1,2}(?:[-/.]\d{1,2})?)(?!\d)")


class _DisabledMCPFacade:
    """Compatibility surface for disabled MCP without clients or atexit hooks."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        servers = config.get("mcp.servers", [])
        server_values = servers if isinstance(servers, list) else []
        self.statuses = [
            MCPServerStatus(str(spec.get("name") or "unnamed"), False, False)
            for spec in server_values
            if isinstance(spec, dict)
        ]

    def discover(self) -> list[tuple[ToolCapability, Any]]:
        return []

    def close(self) -> None:
        return None

    def summary(self) -> str:
        if not self.statuses:
            return "disabled"
        connected = sum(1 for item in self.statuses if item.connected)
        enabled = sum(1 for item in self.statuses if item.enabled)
        tools = sum(item.tool_count for item in self.statuses)
        return f"{connected}/{enabled} servers connected ({tools} tools)"


class ToolManager:
    _LAZY_TOOL_NAMES = frozenset(
        {
            "browser",
            "docker",
            "document",
            "file_edit",
            "git",
            "http",
            "lsp",
            "python",
            "shell",
            "templates",
        }
    )

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
        health: CapabilityHealthManager | None = None,
    ) -> None:
        self.config = config
        self.project = project
        self.memory = memory
        self.global_knowledge = GlobalKnowledgeBase(memory)
        self.cwd = project.root
        self.events = events
        self.approval_handler = approval_handler
        self.auto_approve = auto_approve
        self.yolo = yolo
        self.super_yolo = super_yolo
        self.state: AgentState | None = None
        self._execution_local = threading.local()
        self._lazy_tool_lock = threading.RLock()
        self._lazy_tools: dict[str, Any] = {}
        self.plan_manager = PlanManager()
        self._permission = PermissionManager(config, project.root)
        self.registry = ToolCapabilityRegistry(config)
        self.health = health or CapabilityHealthManager(config, project.id)
        self._max_result_bytes = int(config.get("tools.tool_result.max_attachment_bytes", 8_388_608))
        self.result_store = ToolResultStore(
            project.agent_dir,
            max_attachment_bytes=self._max_result_bytes,
            persist_threshold_bytes=int(config.get("tools.tool_result.persist_threshold_bytes", 12_000)),
            preview_chars=int(config.get("tools.tool_result.preview_chars", 12_000)),
            max_read_chars=int(config.get("tools.tool_result.max_read_chars", 32_000)),
            max_attachments_per_session=int(config.get("tools.tool_result.max_attachments_per_session", 512)),
            max_session_bytes=int(config.get("tools.tool_result.max_session_bytes", 268_435_456)),
        )
        self._register_capabilities()
        self._mcp_instance: MCPManager | None = None
        self._disabled_mcp = _DisabledMCPFacade(config)
        if bool(config.get("mcp.enabled", False)):
            self._mcp_instance = MCPManager(config, self.cwd)
            self._register_mcp_capabilities()
        self.executor = ToolExecutor(
            self.registry,
            self.permission,
            project_id=project.id,
            result_store=self.result_store,
            approval_handler=approval_handler,
            approval_summary=self._approval_summary,
            auto_approve_capabilities=self._auto_approve_capabilities,
            auto_approve=auto_approve,
            yolo=yolo,
            super_yolo=super_yolo,
        )

    @property
    def permission(self) -> PermissionManager:
        return self._permission

    @permission.setter
    def permission(self, value: PermissionManager) -> None:
        self._permission = value
        executor = getattr(self, "executor", None)
        if executor is not None:
            executor.permission = value

    def bind_state(self, state: AgentState) -> None:
        self.state = state

    def set_event_bus(self, events: EventBus) -> None:
        self.events = events

    def configure_permissions(self, *, yolo: bool | None = None, super_yolo: bool | None = None) -> None:
        """Update interactive approval modes through one explicit boundary."""

        if yolo is not None:
            self.yolo = bool(yolo)
        if super_yolo is not None:
            self.super_yolo = bool(super_yolo)
        self.executor.configure_permissions(yolo=yolo, super_yolo=super_yolo)

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

    def canonical_capability_name(self, name: str) -> str:
        capability, _handler = self.registry.resolve(str(name))
        return capability.name if capability is not None else str(name)

    def capability_health_status(self, name: str) -> str:
        capability, _handler = self.registry.resolve(str(name))
        return self.health.evaluate(capability).status if capability is not None else "Unavailable"

    @classmethod
    def result_is_health_failure(cls, result: ToolResult) -> bool:
        return ToolExecutor.result_is_health_failure(result)

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

        if not isinstance(context, ToolExecutionOwnership):
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
        ownership: ToolExecutionOwnership,
    ) -> ToolResult:
        # Preserve the historical mutability of ToolManager's public approval
        # attributes while keeping the lifecycle implementation independent.
        self.executor.registry = self.registry
        self.executor.permission = self.permission
        self.executor.result_store = self.result_store
        self.executor.approval_handler = self.approval_handler
        self.executor.approval_summary = self._approval_summary
        self.executor.auto_approve_capabilities = self._auto_approve_capabilities
        self.executor.auto_approve = self.auto_approve
        self.executor.yolo = self.yolo
        self.executor.super_yolo = self.super_yolo
        return self.executor.execute(
            request,
            ownership=ownership,
            runtime_denied_reason=runtime_denied_reason,
            argument_error=argument_error,
        )

    def _capture_execution_ownership(self) -> ToolExecutionOwnership:
        current = getattr(self._execution_local, "ownership", None)
        if isinstance(current, ToolExecutionOwnership):
            return current
        state = self.state
        return ToolExecutionOwnership(
            state=state,
            session_id=state.session_id if state is not None else None,
            run_id=state.run_id if state is not None else None,
            events=self.events,
        )

    def call(self, name: str, arguments: str | dict[str, Any] | None) -> ToolResult:
        _, result = self.execute_model_call(name, arguments)
        return result

    def _register_capabilities(self) -> None:
        declarations = builtin_capability_declarations(
            self.config,
            max_result_read_chars=self.result_store.max_read_chars,
        )
        for declaration in declarations:
            handler = getattr(self, declaration.handler_name)
            self.registry.register(declaration.capability, handler)

    def _register_mcp_capabilities(self) -> None:
        if self._mcp_instance is None:
            return
        for capability, handler in self._mcp_instance.discover():
            self.registry.register(capability, handler)

    def close(self) -> None:
        if self._mcp_instance is not None:
            self._mcp_instance.close()

    def __getattr__(self, name: str) -> Any:
        """Create compatibility tool attributes on first use only."""

        if name in self._LAZY_TOOL_NAMES:
            return self._get_lazy_tool(name)
        if name == "mcp":
            return self._mcp_instance or self._disabled_mcp
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def _get_lazy_tool(self, name: str) -> Any:
        with self._lazy_tool_lock:
            instance = self._lazy_tools.get(name)
            if instance is None:
                instance = self._build_lazy_tool(name)
                self._lazy_tools[name] = instance
            return instance

    def _build_lazy_tool(self, name: str) -> Any:
        if name == "shell":
            return ShellTool(
                self.cwd,
                self._capability_timeout("shell.run", 120),
                max_output_bytes=self._max_result_bytes,
            )
        if name == "python":
            return PythonTool(
                self.cwd,
                self._capability_timeout("python.run", 120),
                max_output_bytes=self._max_result_bytes,
            )
        if name == "git":
            return GitTool(
                self.cwd,
                self._capability_timeout("git.status", 120),
                max_output_bytes=self._max_result_bytes,
            )
        if name == "document":
            return DocumentTool(
                self.cwd,
                self._capability_timeout("document.parse", 180),
                max_input_bytes=int(self.config.get("tools.document.max_input_bytes", 25_000_000)),
                max_result_bytes=self._max_result_bytes,
            )
        if name == "docker":
            return DockerTool(
                self.cwd,
                self._capability_timeout("docker.run", 180),
                max_output_bytes=self._max_result_bytes,
            )
        if name == "browser":
            return BrowserTool(
                self.cwd,
                self._capability_timeout("browser.open_url", 180),
                max_download_bytes=int(self.config.get("tools.browser.max_download_bytes", 100_000_000)),
            )
        if name == "http":
            allowed_domains = self.config.get("tools.http.allowed_domains", [])
            return HttpTool(
                self.cwd,
                allowed_domains=([str(item) for item in allowed_domains] if isinstance(allowed_domains, list) else []),
                timeout=self._capability_timeout("http.request", 30),
                max_response_bytes=int(self.config.get("tools.http.max_response_bytes", 1_048_576)),
            )
        if name == "lsp":
            return LSPManager(
                self.cwd,
                timeout=self._capability_timeout("lsp.diagnostics", 60),
                max_diagnostics=int(self.config.get("tools.lsp.max_diagnostics", 200)),
            )
        if name == "file_edit":
            return FileEditTool(
                self.project,
                int(self.config.get("tools.file.max_file_bytes", 2_000_000)),
            )
        if name == "templates":
            return SafeTemplateTool(
                self.cwd,
                self._capability_timeout("template.run_tests", 300),
                max_input_bytes=int(self.config.get("tools.template.max_input_bytes", 67_108_864)),
                max_result_bytes=self._max_result_bytes,
            )
        raise AttributeError(f"unknown lazy tool: {name}")

    def _capability_timeout(self, capability_name: str, default: int) -> int:
        capability, _handler = self.registry.resolve(capability_name)
        return capability.timeout_seconds if capability is not None else int(default)

    def _template_list_dir(self, path: str = ".", depth: int = 2, max_entries: int = 500) -> ToolResult:
        return self.templates.list_dir(path=path, depth=depth, max_entries=max_entries)

    def _template_search_code(
        self,
        query: str,
        path: str = ".",
        glob: str | None = None,
        max_results: int = 200,
    ) -> ToolResult:
        return self.templates.search_code(
            query=query,
            path=path,
            glob=glob,
            max_results=max_results,
        )

    def _template_read_file(
        self,
        path: str,
        start_line: int = 1,
        end_line: int = 240,
    ) -> ToolResult:
        return self.templates.read_file(path=path, start_line=start_line, end_line=end_line)

    def _template_find_files(
        self,
        pattern: str = "*",
        path: str = ".",
        max_results: int = 500,
    ) -> ToolResult:
        return self.templates.find_files(pattern=pattern, path=path, max_results=max_results)

    def _template_git_diff_staged(self, path: str | None = None) -> ToolResult:
        return self.templates.git_diff_staged(path=path)

    def _template_run_tests(self, framework: str = "auto", path: str = ".") -> ToolResult:
        return self.templates.run_tests(framework=framework, path=path)

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
            and self.capability_health_status("lsp.diagnostics") == "Available"
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

    def _git_status(self) -> ToolResult:
        return self.git.status()

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

    def _browser_list_sessions(self) -> ToolResult:
        return self.browser.list_sessions()

    def _http_request(
        self,
        url: str,
        method: str = "GET",
        json_body: Any = None,
        headers: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> ToolResult:
        return self.http.request(
            url=url,
            method=method,
            json_body=json_body,
            headers=headers,
            timeout=timeout,
        )

    def _lsp_diagnostics(self, path: str | None = None) -> ToolResult:
        return self.lsp.diagnostics(path)

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
        if kind == "Correction" and not global_memory:
            if not any(tag.startswith("correction:") for tag in normalized_tags):
                return ToolResult(False, "", "Correction memory requires a correction:<topic> tag")
            if self.project.name not in normalized_tags:
                normalized_tags.append(self.project.name)
        try:
            memory_id = (
                self.global_knowledge.add(
                    kind=kind,
                    title=title,
                    content=content,
                    tags=normalized_tags,
                )
                if global_memory
                else self.memory.add_memory(
                    kind=kind,
                    title=title,
                    content=content,
                    tags=normalized_tags,
                    project_id=self.project.id,
                )
            )
        except ValueError as exc:
            return ToolResult(False, "", str(exc))
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
        model_statuses = [str(item.get("status") or "pending") for item in steps if isinstance(item, dict)]
        if any(status not in {"pending", "in_progress"} for status in model_statuses):
            return ToolResult(
                False,
                "",
                "a new Task Graph cannot pre-claim completed, failed, or skipped steps; only the implement step "
                "of an existing conditional-mutation graph may later be skipped",
            )
        if model_statuses.count("in_progress") > 1:
            return ToolResult(False, "", "a new Task Graph can have at most one in-progress step")
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
        if isinstance(ownership, ToolExecutionOwnership) and ownership.state is not None:
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

    _is_health_failure = staticmethod(ToolExecutor.result_is_health_failure)
    _health_error_summary = staticmethod(ToolExecutor.health_error_summary)

    def _publish(
        self,
        event_name: str,
        request: ToolRequest,
        result: ToolResult | None,
        *,
        ownership: ToolExecutionOwnership,
    ) -> None:
        self.executor.publish(event_name, request, result, ownership=ownership)


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
