from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import agent.tools.manager as manager_module
from agent.contracts import RuntimeToolsProtocol, ToolExecutorProtocol
from agent.memory import MemoryStore
from agent.project import ProjectManager
from agent.state import AgentState
from agent.tools import ToolExecutionOwnership, ToolExecutor, ToolManager
from agent.tools.base import ToolRequest, ToolResult
from agent.tools.capabilities import builtin_capability_declarations
from agent.tools.permission import PermissionDecision
from agent.tools.registry import ToolCapability, ToolCapabilityRegistry


def build_manager(
    root: Path,
    make_config,
    overrides: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ToolManager:
    config = make_config(overrides)
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    memory.sync_project(project)
    return ToolManager(config, project, memory, **kwargs)


def test_tool_executor_independently_enforces_permission_before_handler(
    tmp_path: Path,
    make_config,
) -> None:
    config = make_config()
    registry = ToolCapabilityRegistry(config)
    calls: list[str] = []

    class RecordingPermission:
        allowed = True

        def evaluate(self, request, capability, *, super_yolo=False):
            calls.append(f"permission:{request.capability}:{capability.name}:{super_yolo}")
            return PermissionDecision(self.allowed, "blocked by test policy")

    permission = RecordingPermission()

    def handler(value: str) -> ToolResult:
        calls.append(f"handler:{value}")
        return ToolResult(True, value)

    registry.register(
        ToolCapability(
            "probe",
            "run",
            "probe_run",
            "Independent executor probe.",
            {"value": {"type": "string"}},
            ("value",),
        ),
        handler,
    )
    executor = ToolExecutor(
        registry,
        permission,  # type: ignore[arg-type]
        project_id="project-id",
        yolo=True,
    )
    assert isinstance(executor, ToolExecutorProtocol)
    assert not isinstance(executor, RuntimeToolsProtocol)
    ownership = ToolExecutionOwnership(None, None, None, None)

    result = executor.execute(
        ToolRequest("probe", "run", {"value": "ok"}, request_id="allowed"),
        ownership=ownership,
    )

    assert result.success is True
    assert result.stdout == "ok"
    assert result.request_id == "allowed"
    assert calls == ["permission:probe.run:probe.run:False", "handler:ok"]

    permission.allowed = False
    calls.clear()
    denied = executor.execute(
        ToolRequest("probe", "run", {"value": "never"}, request_id="denied"),
        ownership=ownership,
    )

    assert denied.success is False
    assert denied.data == {"not_executed": True}
    assert denied.stderr == "blocked by test policy"
    assert calls == ["permission:probe.run:probe.run:False"]


def test_capability_catalog_uses_unique_handler_names_without_tool_instances(
    tmp_path: Path,
    make_config,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    declarations = builtin_capability_declarations(config, max_result_read_chars=32_000)

    assert len(declarations) == 34
    assert len({item.capability.name for item in declarations}) == len(declarations)
    assert len({item.capability.model_name for item in declarations}) == len(declarations)
    assert all(item.handler_name.startswith("_") for item in declarations)

    tools = build_manager(root, make_config)
    try:
        assert tools._lazy_tools == {}
        for declaration in declarations:
            capability, handler = tools.registry.resolve(declaration.capability.name)
            assert capability is not None
            assert handler is not None
            assert handler.__self__ is tools
            assert handler.__name__ == declaration.handler_name
        assert tools._lazy_tools == {}
    finally:
        tools.close()


def test_manager_construction_and_capability_queries_do_not_construct_heavy_tools(
    tmp_path: Path,
    make_config,
    monkeypatch,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    constructed: list[str] = []
    names = (
        "BrowserTool",
        "DockerTool",
        "DocumentTool",
        "FileEditTool",
        "GitTool",
        "HttpTool",
        "LSPManager",
        "MCPManager",
        "PythonTool",
        "SafeTemplateTool",
        "ShellTool",
    )

    def forbidden(name: str):
        def factory(*_args, **_kwargs):
            constructed.append(name)
            raise AssertionError(f"constructed too early: {name}")

        return factory

    for name in names:
        monkeypatch.setattr(manager_module, name, forbidden(name))

    tools = build_manager(root, make_config)
    try:
        assert tools._lazy_tools == {}
        assert tools.schemas()
        assert tools.capabilities(enabled_only=False)
        assert tools.capability_summary()
        assert tools.mcp.summary() == "disabled"
        assert constructed == []
        assert tools._lazy_tools == {}
    finally:
        tools.close()
    assert constructed == []


def test_disabled_capability_is_denied_without_constructing_its_tool(
    tmp_path: Path,
    make_config,
    monkeypatch,
) -> None:
    root = tmp_path / "project"
    root.mkdir()

    def forbidden_shell(*_args, **_kwargs):
        raise AssertionError("disabled ShellTool must remain unconstructed")

    monkeypatch.setattr(manager_module, "ShellTool", forbidden_shell)
    tools = build_manager(
        root,
        make_config,
        {"tools": {"capabilities": {"shell": {"run": {"enabled": False}}}}},
        yolo=True,
    )
    try:
        _, denied = tools.execute_model_call("shell_run", {"command": "printf never"})
        assert denied.success is False
        assert denied.data == {"not_executed": True}
        assert "disabled" in denied.stderr
        assert "shell" not in tools._lazy_tools
    finally:
        tools.close()


def test_permission_denial_precedes_lazy_tool_construction(
    tmp_path: Path,
    make_config,
    monkeypatch,
) -> None:
    root = tmp_path / "project"
    root.mkdir()

    def forbidden_shell(*_args, **_kwargs):
        raise AssertionError("permission-denied ShellTool must remain unconstructed")

    monkeypatch.setattr(manager_module, "ShellTool", forbidden_shell)
    tools = build_manager(
        root,
        make_config,
        {"permissions": {"deny_capabilities": ["shell.run"]}},
        yolo=True,
    )
    try:
        _, denied = tools.execute_model_call("shell_run", {"command": "printf never"})
        assert denied.success is False
        assert denied.data == {"not_executed": True}
        assert "denied by policy" in denied.stderr
        assert "shell" not in tools._lazy_tools
    finally:
        tools.close()


def test_disabled_lsp_is_not_initialized_by_file_apply(
    tmp_path: Path,
    make_config,
    monkeypatch,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "sample.py").write_text("value = 1\n", encoding="utf-8")

    def forbidden_lsp(*_args, **_kwargs):
        raise AssertionError("disabled LSPManager must remain unconstructed")

    monkeypatch.setattr(manager_module, "LSPManager", forbidden_lsp)
    tools = build_manager(
        root,
        make_config,
        {"tools": {"capabilities": {"lsp": {"diagnostics": {"enabled": False}}}}},
        yolo=True,
    )
    state = AgentState.create(
        session_id="lazy-lsp",
        project=tools.project,
        user_request="edit without disabled diagnostics",
        loaded_memories=[],
        loaded_tools=[],
        git_branch=None,
        context_index_path=str(tools.project.agent_dir / "index.json"),
    )
    tools.bind_state(state)
    try:
        _, preview = tools.execute_model_call(
            "file_diff",
            {"path": "sample.py", "content": "value = 2\n"},
        )
        _, applied = tools.execute_model_call(
            "file_apply",
            {"preview_id": preview.data["preview_id"]},
        )
        assert applied.success is True
        assert (root / "sample.py").read_text(encoding="utf-8") == "value = 2\n"
        assert "lsp" not in tools._lazy_tools
    finally:
        tools.close()


def test_compatibility_attribute_and_first_call_initialize_once_thread_safely(
    tmp_path: Path,
    make_config,
    monkeypatch,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    created = 0
    created_lock = threading.Lock()

    class FakeShell:
        def __init__(self, _cwd, timeout, *, max_output_bytes):
            nonlocal created
            time.sleep(0.01)
            with created_lock:
                created += 1
            self.timeout = timeout
            self.max_output_bytes = max_output_bytes

        def run(self, command, cwd=None, timeout=None):
            return ToolResult(True, f"{command}:{cwd}:{timeout}")

    monkeypatch.setattr(manager_module, "ShellTool", FakeShell)
    tools = build_manager(
        root,
        make_config,
        {"tools": {"capabilities": {"shell": {"run": {"timeout_seconds": 7}}}}},
        yolo=True,
    )
    try:
        assert tools._lazy_tools == {}

        def execute(index: int) -> ToolResult:
            _, result = tools.execute_model_call(
                "shell_run",
                {"command": f"probe-{index}", "cwd": ".", "timeout": 3},
            )
            return result

        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(execute, range(64)))
        assert all(result.success for result in results)
        assert {result.stdout.split(":", 1)[0] for result in results} == {f"probe-{index}" for index in range(64)}
        assert len({result.request_id for result in results}) == 64
        assert created == 1

        with ThreadPoolExecutor(max_workers=16) as pool:
            identities = list(pool.map(lambda _index: id(tools.shell), range(64)))
        assert len(set(identities)) == 1
        assert created == 1
        assert tools.shell.timeout == 7

        _, result = tools.execute_model_call(
            "shell_run",
            {"command": "probe", "cwd": ".", "timeout": 3},
        )
        assert result.success is True
        assert result.stdout.endswith(":3")
        assert created == 1
        assert set(tools._lazy_tools) == {"shell"}
    finally:
        tools.close()


def test_first_template_call_initializes_only_template_tool(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "sample.txt").write_text("sample\n", encoding="utf-8")
    tools = build_manager(root, make_config, yolo=True)
    try:
        assert tools._lazy_tools == {}
        _, result = tools.execute_model_call("list_dir", {"path": ".", "depth": 1})
        assert result.success is True
        assert "sample.txt" in result.stdout
        assert set(tools._lazy_tools) == {"templates"}
    finally:
        tools.close()
