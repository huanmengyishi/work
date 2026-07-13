from __future__ import annotations

import ast
import inspect
from dataclasses import fields
from pathlib import Path

from agent import prompt as prompt_module
from agent.cli import build_runtime
from agent.context import ContextPackage
from agent.contracts import (
    CONTEXT_INTERFACE_CHAIN,
    CORE_INTERFACE_CHAIN,
    CORE_INTERFACE_CONTRACT_VERSION,
)
from agent.memory import MemoryStore
from agent.model_router import ModelRoute
from agent.project import ProjectManager
from agent.prompt import PromptBuilder
from agent.runtime import AgentRuntime
from agent.state import AgentState
from agent.task_plan import TaskPlanFactory
from agent.task_router import TaskRoute
from agent.tools import ToolManager
from agent.tools.base import ToolRequest, ToolResult
from agent.tools.permission import PermissionDecision
from agent.tools.registry import ToolCapability


def _parameter_names(callable_object) -> tuple[str, ...]:
    return tuple(inspect.signature(callable_object).parameters)


def test_core_interface_chain_and_runtime_entrypoints_are_versioned() -> None:
    assert CORE_INTERFACE_CONTRACT_VERSION == 1
    assert CORE_INTERFACE_CHAIN == (
        "CLI",
        "Runtime",
        "AgentState",
        "Prompt",
        "Capability",
        "Permission",
    )
    assert CONTEXT_INTERFACE_CHAIN == ("ContextBuilder", "ContextPackage", "PromptBuilder")
    assert inspect.signature(build_runtime).return_annotation in {AgentRuntime, "AgentRuntime"}
    assert _parameter_names(AgentRuntime.run) == ("self", "prompt", "initial_plan", "queue_id")
    assert _parameter_names(AgentRuntime.resume) == ("self", "prompt", "session_id")
    assert _parameter_names(TaskPlanFactory.build) == ("self", "route")
    assert inspect.signature(TaskPlanFactory.build).parameters["route"].annotation in {TaskRoute, "TaskRoute"}
    assert not hasattr(AgentRuntime, "strategy_selector")


def test_prompt_builder_accepts_only_context_package_and_has_no_file_access() -> None:
    assert _parameter_names(PromptBuilder.build_initial) == ("self", "package")
    assert _parameter_names(PromptBuilder.build_resume) == ("self", "package")
    assert not hasattr(PromptBuilder, "append_resume")

    for method in (PromptBuilder.build_initial, PromptBuilder.build_resume):
        annotation = inspect.signature(method).parameters["package"].annotation
        assert annotation in {ContextPackage, "ContextPackage"}

    source = inspect.getsource(prompt_module)
    tree = ast.parse(source)
    forbidden_imports = {"AgentState", "ContextBuilder", "ContextSnapshot", "Path"}
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imported_names.isdisjoint(forbidden_imports)

    forbidden_calls = {"open", "read_text", "read_bytes", "write_text", "write_bytes"}
    called_names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            called_names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            called_names.add(node.func.attr)
    assert called_names.isdisjoint(forbidden_calls)


def test_context_package_and_tool_protocol_serialized_fields_are_frozen() -> None:
    assert tuple(item.name for item in fields(ContextPackage)) == (
        "schema_version",
        "phase",
        "project_id",
        "session_id",
        "turn",
        "user_request",
        "sections",
        "fingerprint",
        "file_count",
        "source_file_count",
        "git_branch",
        "index_path",
        "generated_path",
        "loaded_files",
        "included_memory_ids",
        "max_chars",
        "used_chars",
        "rendered_chars",
        "original_user_request_chars",
        "user_request_truncated",
        "omitted_sections",
        "truncated_sections",
    )
    request = ToolRequest(tool="contract", action="probe", args={"value": 1}, request_id="request-1")
    result = ToolResult(True, "ok", request_id="request-1")
    assert tuple(request.to_dict()) == ("tool", "action", "args", "request_id", "model_name")
    assert tuple(result.to_dict()) == (
        "success",
        "stdout",
        "stderr",
        "duration_ms",
        "request_id",
        "data",
    )
    model_route = ModelRoute(
        provider="deepseek",
        tier="standard",
        model="deepseek-base",
        thinking_enabled=True,
        reasoning_effort="high",
        max_tokens=4096,
        cost_class="balanced",
        reasons=("contract",),
    )
    assert tuple(model_route.to_dict()) == (
        "schema_version",
        "provider",
        "tier",
        "model",
        "thinking_enabled",
        "reasoning_effort",
        "max_tokens",
        "cost_class",
        "reasons",
    )


def test_agent_state_declares_and_validates_its_serialization_contract(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    project = ProjectManager(make_config()).resolve_project(root)
    state = AgentState.create(
        session_id="contract-session",
        project=project,
        user_request="verify the interface contract",
        loaded_memories=[],
        loaded_tools=[],
        git_branch=None,
        context_index_path=str(project.agent_dir / "index.json"),
    )

    assert tuple(state.to_dict()) == AgentState.SERIALIZED_FIELDS
    assert {item.split(".", 1)[0] for item in AgentState.FROZEN_FIELDS} <= set(AgentState.SERIALIZED_FIELDS)
    assert state.validate() is state


def test_capability_execution_cannot_bypass_permission_manager(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    tools = ToolManager(config, project, memory, yolo=True)
    calls: list[str] = []

    capability = ToolCapability(
        "contract",
        "probe",
        "contract_probe",
        "Interface contract probe.",
        {},
        permissions=("read",),
    )

    def handler() -> ToolResult:
        calls.append("handler")
        return ToolResult(True, "ok")

    class RecordingPermissionManager:
        def evaluate(self, request, resolved_capability, *, super_yolo=False):
            calls.append("permission")
            assert request.capability == "contract.probe"
            assert resolved_capability.name == "contract.probe"
            return PermissionDecision(True)

    tools.registry.register(capability, handler)
    tools.permission = RecordingPermissionManager()
    request = ToolRequest("contract", "probe", request_id="contract-request")

    result = tools.execute(request)

    assert result.success is True
    assert result.request_id == "contract-request"
    assert calls == ["permission", "handler"]
    tools.close()
