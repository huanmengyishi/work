from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

import pytest

from agent.deepseek import ChatResponse
from agent.memory import MemoryStore
from agent.project import ProjectManager
from agent.runtime import AgentRuntime
from agent.tools import ToolManager
from agent.tools.base import ToolResult


class _RecordingClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.schemas: list[list[dict[str, Any]] | None] = []

    def chat(self, *, messages, tools=None, **_kwargs) -> ChatResponse:
        del messages
        self.schemas.append(tools)
        if not self.responses:
            raise AssertionError("fake response queue exhausted")
        return ChatResponse(message=self.responses.pop(0), raw={})


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    return call_id, name, arguments


def _tool_calls(*calls: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
            for call_id, name, arguments in calls
        ],
    }


def _manager(root: Path, make_config, overrides: dict[str, Any] | None = None) -> ToolManager:
    config = make_config(overrides)
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    memory.sync_project(project)
    return ToolManager(config, project, memory, yolo=True)


@pytest.mark.skipif(shutil.which("npm") is None, reason="npm is required for the real hard-phase integration")
def test_runtime_hard_phase_runs_validation_but_never_executes_file_exploration(
    tmp_path: Path,
    make_config,
    monkeypatch,
) -> None:
    root = tmp_path / "typescript-project"
    root.mkdir()
    (root / "package.json").write_text(
        json.dumps(
            {"scripts": {"typecheck": ("node -e \"require('node:fs').writeFileSync('typecheck.ok', 'validated')\"")}}
        ),
        encoding="utf-8",
    )
    config = make_config(
        {
            "runtime": {
                "task_mode": "deep",
                "max_tool_rounds_hard_limit": 2,
                "convergence": {"reserved_tool_rounds": 1},
            }
        }
    )
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    tools = ToolManager(config, project, memory, yolo=True)

    permission_checks: list[str] = []
    original_evaluate = tools.permission.evaluate

    def tracking_evaluate(request, capability, *, super_yolo=False):
        permission_checks.append(request.capability)
        return original_evaluate(request, capability, super_yolo=super_yolo)

    executed_shell_commands: list[str] = []
    original_shell_run = tools.shell.run

    def tracking_shell_run(command: str, cwd: str | None = None, timeout: int | None = None):
        executed_shell_commands.append(command)
        return original_shell_run(command, cwd=cwd, timeout=timeout)

    monkeypatch.setattr(tools.permission, "evaluate", tracking_evaluate)
    monkeypatch.setattr(tools.shell, "run", tracking_shell_run)

    plan_transition = _tool_calls(
        _tool_call("scope-done", "agent_update_step", {"step_id": "scope", "status": "completed"}),
        _tool_call(
            "inspect-start",
            "agent_update_step",
            {"step_id": "inspect-chunks", "status": "in_progress"},
        ),
        _tool_call(
            "inspect-done",
            "agent_update_step",
            {"step_id": "inspect-chunks", "status": "completed"},
        ),
        _tool_call(
            "synthesize-start",
            "agent_update_step",
            {"step_id": "synthesize", "status": "in_progress"},
        ),
        _tool_call(
            "synthesize-done",
            "agent_update_step",
            {"step_id": "synthesize", "status": "completed"},
        ),
        _tool_call(
            "verify-start",
            "agent_update_step",
            {"step_id": "verify", "status": "in_progress"},
        ),
    )
    denied_commands = [
        "cat package.json",
        "rg scripts .",
        "sed -n '1,5p' package.json",
        "head -1 package.json",
        "git show HEAD:package.json",
        "dd if=package.json bs=1 count=8",
        "source package.json",
    ]
    hard_phase = _tool_calls(
        _tool_call("typecheck", "shell_run", {"command": "npm run typecheck"}),
        *(
            _tool_call(f"blocked-{index}", "shell_run", {"command": command})
            for index, command in enumerate(denied_commands, start=1)
        ),
        _tool_call(
            "verify-done",
            "agent_update_step",
            {"step_id": "verify", "status": "completed"},
        ),
    )
    client = _RecordingClient(
        [
            plan_transition,
            hard_phase,
            {"role": "assistant", "content": "验证命令已通过，文件探索命令均被 hard phase 拒绝。"},
        ]
    )
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=tools,
        client=client,
    )

    answer = runtime.run("全面审计整个项目并验证现有 TypeScript 代码")

    assert answer == "验证命令已通过，文件探索命令均被 hard phase 拒绝。"
    assert (root / "typecheck.ok").read_text(encoding="utf-8") == "validated"
    assert executed_shell_commands == ["npm run typecheck"]
    assert permission_checks.count("shell.run") == 1 + len(denied_commands)

    second_schema_names = {str((item.get("function") or {}).get("name") or "") for item in client.schemas[1] or []}
    assert "shell_run" in second_schema_names
    state = runtime.sessions.load(runtime.last_session_id).state
    shell_results = [item["result"] for item in state.tool_calls if (item.get("request") or {}).get("tool") == "shell"]
    assert shell_results[0]["success"] is True
    assert all(result["success"] is False for result in shell_results[1:])
    assert all(result["data"] == {"runtime_denied": True, "not_executed": True} for result in shell_results[1:])


@pytest.mark.skipif(shutil.which("npm") is None, reason="npm is required for the real validation command selection")
def test_runtime_hard_phase_can_read_bounded_attachment_from_current_validation(
    tmp_path: Path,
    make_config,
    monkeypatch,
) -> None:
    root = tmp_path / "typescript-project"
    root.mkdir()
    (root / "package.json").write_text(
        json.dumps({"scripts": {"typecheck": "tsc --noEmit"}}),
        encoding="utf-8",
    )
    config = make_config(
        {
            "runtime": {
                "task_mode": "deep",
                "max_tool_rounds_hard_limit": 2,
                "convergence": {
                    "reserved_tool_rounds": 1,
                    "max_implementation_evidence_reads": 0,
                    "max_validation_attachment_reads": 2,
                },
            },
            "tools": {"tool_result": {"persist_threshold_bytes": 512, "preview_chars": 256}},
        }
    )
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    tools = ToolManager(config, project, memory, yolo=True)

    def large_failed_validation(_args, **_kwargs):
        return ToolResult(False, "TYPECHECK-EVIDENCE\n" + "E" * 8_000, data={"returncode": 2})

    monkeypatch.setattr("agent.tools.templates.run_command", large_failed_validation)
    client = _RecordingClient(
        [
            _tool_calls(
                _tool_call("scope-done", "agent_update_step", {"step_id": "scope", "status": "completed"}),
                _tool_call(
                    "inspect-start",
                    "agent_update_step",
                    {"step_id": "inspect-chunks", "status": "in_progress"},
                ),
                _tool_call(
                    "inspect-done",
                    "agent_update_step",
                    {"step_id": "inspect-chunks", "status": "completed"},
                ),
                _tool_call(
                    "implement-start",
                    "agent_update_step",
                    {"step_id": "implement", "status": "in_progress"},
                ),
                _tool_call("validation-result", "run_tests", {"framework": "npm:typecheck", "path": "."}),
            ),
            _tool_calls(
                _tool_call(
                    "validation-chunk",
                    "tool_result_read",
                    {"request_id": "validation-result", "offset": 0, "max_chars": 1_000},
                )
            ),
            {"role": "assistant", "content": "验证附件已按受限范围读取。"},
        ]
    )
    runtime = AgentRuntime(config=config, project=project, memory=memory, tools=tools, client=client)

    answer = runtime.run("全面审计整个项目；若找到真实缺陷则修复并验证，否则不要修改")

    assert "任务尚未完成" in answer
    second_schema_names = {str((item.get("function") or {}).get("name") or "") for item in client.schemas[1] or []}
    assert "tool_result_read" in second_schema_names
    assert "read_file" not in second_schema_names
    state = runtime.sessions.load(runtime.last_session_id).state
    validation = next(
        item for item in state.tool_calls if (item.get("request") or {}).get("request_id") == "validation-result"
    )
    attachment_read = next(
        item for item in state.tool_calls if (item.get("request") or {}).get("request_id") == "validation-chunk"
    )
    assert validation["result"]["success"] is False
    assert validation["result"]["data"]["attachment"]["request_id"] == "validation-result"
    assert attachment_read["result"]["success"] is True
    assert "TYPECHECK-EVIDENCE" in attachment_read["result"]["stdout"]
    assert state.convergence["validation_attachment_reads_used"] == 1


@pytest.mark.parametrize(
    ("scripts", "expected"),
    [
        ({"build": "build", "lint": "lint", "check": "check", "typecheck": "tsc", "test": "test"}, "test"),
        ({"build": "build", "lint": "lint", "check": "check", "typecheck": "tsc"}, "typecheck"),
        ({"build": "build", "lint": "lint", "check": "check"}, "check"),
        ({"build": "build", "lint": "lint"}, "lint"),
        ({"build": "build"}, "build"),
    ],
)
def test_run_tests_uses_first_existing_package_script_with_argv(
    tmp_path: Path,
    make_config,
    monkeypatch,
    scripts: dict[str, str],
    expected: str,
) -> None:
    root = tmp_path / expected
    root.mkdir()
    (root / "package.json").write_text(json.dumps({"scripts": scripts}), encoding="utf-8")
    tools = _manager(root, make_config)
    calls: list[list[str]] = []

    def fake_run_command(args, *, cwd, timeout, **_kwargs):
        assert cwd == root
        assert timeout > 0
        calls.append(args)
        return ToolResult(True, "validation passed")

    monkeypatch.setattr("agent.tools.templates.run_command", fake_run_command)

    _, result = tools.execute_model_call("run_tests", {"framework": "auto", "path": "."})

    assert result.success is True
    assert calls == [["npm", "run", expected]]


def test_run_tests_does_not_misclassify_package_project_with_tests_directory(
    tmp_path: Path,
    make_config,
    monkeypatch,
) -> None:
    root = tmp_path / "typescript-with-tests"
    root.mkdir()
    (root / "tests").mkdir()
    (root / "package.json").write_text(
        json.dumps({"scripts": {"typecheck": "tsc --noEmit"}}),
        encoding="utf-8",
    )
    tools = _manager(root, make_config)
    calls: list[list[str]] = []

    def fake_run_command(args, *, cwd, timeout, **_kwargs):
        assert cwd == root
        assert timeout > 0
        calls.append(args)
        return ToolResult(True, "validation passed")

    monkeypatch.setattr("agent.tools.templates.run_command", fake_run_command)

    _, result = tools.execute_model_call("run_tests", {"framework": "auto", "path": "."})

    assert result.success is True
    assert calls == [["npm", "run", "typecheck"]]


def test_run_tests_still_obeys_permission_manager(
    tmp_path: Path,
    make_config,
    monkeypatch,
) -> None:
    root = tmp_path / "permission-denied"
    root.mkdir()
    (root / "package.json").write_text(
        json.dumps({"scripts": {"typecheck": "tsc --noEmit"}}),
        encoding="utf-8",
    )
    tools = _manager(
        root,
        make_config,
        {"permissions": {"deny_capabilities": ["template.run_tests"]}},
    )

    def forbidden_run_command(*_args, **_kwargs):
        raise AssertionError("permission-denied validation must not reach the process runner")

    monkeypatch.setattr("agent.tools.templates.run_command", forbidden_run_command)

    _, result = tools.execute_model_call("run_tests", {"framework": "auto", "path": "."})

    assert result.success is False
    assert "capability denied by policy: template.run_tests" in result.stderr
