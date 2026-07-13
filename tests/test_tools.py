from __future__ import annotations

from pathlib import Path
import subprocess
import time

from agent.memory import MemoryStore
from agent.project import ProjectManager
from agent.state import AgentState
from agent.tools import ToolManager


def build_manager(
    root: Path,
    make_config,
    overrides=None,
    *,
    auto_approve: bool = False,
    yolo: bool = False,
    super_yolo: bool = False,
):
    config = make_config(overrides)
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    memory.sync_project(project)
    return (
        config,
        project,
        memory,
        ToolManager(
            config,
            project,
            memory,
            auto_approve=auto_approve,
            yolo=yolo,
            super_yolo=super_yolo,
        ),
    )


def test_tool_request_result_and_permission_policy(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _, _, _, tools = build_manager(root, make_config, yolo=True)

    request, result = tools.execute_model_call("shell_run", {"command": "printf ok"})
    assert request.capability == "shell.run"
    assert result.success is True
    assert result.stdout == "ok"
    assert result.request_id == request.request_id
    assert result.duration_ms >= 0

    _, denied_sudo = tools.execute_model_call("shell_run", {"command": "sudo true"})
    _, denied_cwd = tools.execute_model_call("shell_run", {"command": "pwd", "cwd": str(tmp_path.parent)})
    _, denied_timeout = tools.execute_model_call("shell_run", {"command": "true", "timeout": 999})
    assert denied_sudo.success is False
    assert "denied" in denied_sudo.stderr
    assert denied_cwd.success is False
    assert "outside" in denied_cwd.stderr
    assert denied_timeout.success is False
    assert "exceeds" in denied_timeout.stderr

    _, _, _, guarded = build_manager(root, make_config)
    _, confirmation_denied = guarded.execute_model_call("shell_run", {"command": "printf blocked"})
    assert confirmation_denied.success is False
    assert "requires user confirmation" in confirmation_denied.stderr


def test_yolo_and_super_yolo_permission_levels(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _, _, _, yolo = build_manager(root, make_config, yolo=True)
    capability, _ = yolo.registry.resolve("shell.run")
    request = yolo.registry.request("shell.run", {"command": "sudo true"})
    assert capability is not None
    assert yolo.permission.evaluate(request, capability, super_yolo=False).allowed is False

    _, _, _, super_yolo = build_manager(root, make_config, super_yolo=True)
    capability, _ = super_yolo.registry.resolve("shell.run")
    request = super_yolo.registry.request("shell.run", {"command": "sudo true", "cwd": str(tmp_path.parent)})
    assert capability is not None
    decision = super_yolo.permission.evaluate(request, capability, super_yolo=True)
    assert decision.allowed is True
    assert "SUPER YOLO" in decision.reason


def test_yolo_denies_docker_host_escape_variants(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _, _, _, tools = build_manager(root, make_config, yolo=True)
    capability, _ = tools.registry.resolve("docker.run")
    assert capability is not None
    denied = (
        ["run", "--mount", "type=bind,src=/,dst=/host", "alpine"],
        ["run", "--pid=host", "alpine"],
        ["run", "-v=/var/run/docker.sock:/var/run/docker.sock", "alpine"],
        ["run", "--device=/dev/sda", "alpine"],
    )
    for args in denied:
        request = tools.registry.request("docker.run", {"args": args})
        assert tools.permission.evaluate(request, capability).allowed is False


def test_auto_approve_does_not_approve_shell(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _, _, _, tools = build_manager(root, make_config, auto_approve=True)
    _, result = tools.execute_model_call("shell_run", {"command": "printf blocked"})
    assert result.success is False
    assert "requires user confirmation" in result.stderr


def test_correction_memory_requires_topic_and_adds_project_tag(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _, project, memory, tools = build_manager(root, make_config, yolo=True)

    _, denied = tools.execute_model_call(
        "memory_add",
        {"kind": "Correction", "title": "Port", "content": "Use 8080", "tags": []},
    )
    assert denied.success is False
    _, added = tools.execute_model_call(
        "memory_add",
        {
            "kind": "Correction",
            "title": "Port",
            "content": "Use 8080",
            "tags": ["correction:port"],
        },
    )
    assert added.success is True
    item = memory.get_memory(added.data["id"])
    assert item is not None
    assert project.name in item.tags


def test_capability_config_and_plan_state(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    overrides = {
        "tools": {
            "capabilities": {
                "browser": {"open_url": {"enabled": False}},
                "shell": {"run": {"timeout_seconds": 7}},
            }
        }
    }
    _, project, _, tools = build_manager(root, make_config, overrides)
    names = {item["function"]["name"] for item in tools.schemas()}
    shell_capability = next(item for item in tools.capabilities() if item.name == "shell.run")
    assert "browser_open_url" not in names
    assert shell_capability.timeout_seconds == 7
    assert tools.shell.timeout == 7

    state = AgentState.create(
        session_id="session-1",
        project=project,
        user_request="test plan",
        loaded_memories=[],
        loaded_tools=[],
        git_branch=None,
        context_index_path=str(project.agent_dir / "index.json"),
    )
    tools.bind_state(state)
    _, plan_result = tools.execute_model_call(
        "agent_update_plan",
        {"steps": [{"id": "inspect", "title": "Inspect files", "status": "in_progress"}]},
    )
    _, step_result = tools.execute_model_call(
        "agent_update_step",
        {"step_id": "inspect", "status": "completed"},
    )
    assert plan_result.success is True
    assert step_result.success is True
    assert state.completed_steps == ["inspect"]
    assert state.current_step is None


def test_git_tool_in_repository(tmp_path: Path, make_config) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "example.txt").write_text("content\n", encoding="utf-8")
    _, _, _, tools = build_manager(root, make_config)

    _, result = tools.execute_model_call("git_status", {})

    assert result.success is True
    assert "example.txt" in result.stdout


def test_shell_timeout_kills_child_process_group(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _, _, _, tools = build_manager(root, make_config, yolo=True)
    marker = root / "orphan.txt"
    command = f"(sleep 2; printf orphan > {marker}) & wait"

    _, result = tools.execute_model_call("shell_run", {"command": command, "timeout": 1})
    time.sleep(2.2)

    assert result.success is False
    assert "timeout" in result.stderr
    assert not marker.exists()
