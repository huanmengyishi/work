from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.context import ContextBuilder
from agent.memory import MemoryStore
from agent.planner import PlanManager
from agent.project import ProjectManager
from agent.reflection import ReflectionEngine
from agent.state import AgentState
from agent.tools import ToolManager


def make_state(project) -> AgentState:
    return AgentState.create(
        session_id="session-v06",
        project=project,
        user_request="test graph",
        loaded_memories=[],
        loaded_tools=[],
        git_branch="main",
        context_index_path=str(project.agent_dir / "index.json"),
    )


def test_task_graph_dependencies_retries_and_old_state(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    project = ProjectManager(make_config()).resolve_project(root)
    state = make_state(project)
    planner = PlanManager()
    plan = planner.replace(
        state,
        [
            {"id": "inspect", "title": "Inspect", "completion_criteria": "Files identified"},
            {
                "id": "change",
                "title": "Change",
                "dependencies": ["inspect"],
                "max_retries": 2,
                "allow_parallel": False,
            },
        ],
    )
    assert [step.id for step in planner.ready_steps(state)] == ["inspect"]
    with pytest.raises(ValueError, match="dependencies"):
        planner.update_step(state, "change", "in_progress")
    planner.update_step(state, "inspect", "completed")
    assert [step.id for step in planner.ready_steps(state)] == ["change"]
    assert state.current_step == "change"
    planner.update_step(state, "change", "failed")
    assert plan[1].status == "pending"
    assert plan[1].retry_count == 1
    with pytest.raises(ValueError, match="cycle"):
        planner.replace(
            state,
            [
                {"id": "a", "title": "A", "dependencies": ["b"]},
                {"id": "b", "title": "B", "dependencies": ["a"]},
            ],
        )

    old = state.to_dict()
    old["plan"] = [{"id": "legacy", "title": "Legacy", "status": "pending"}]
    old.pop("execution_context", None)
    restored = AgentState.from_dict(old)
    assert restored.plan[0].dependencies == []
    assert restored.execution_context is not None


def test_workspace_memory_detects_project_and_preserves_manual(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\nname='sample'\ndependencies=['fastapi','sqlalchemy']\n[tool.pytest.ini_options]\n",
        encoding="utf-8",
    )
    (root / "main.py").write_text("app = object()\n", encoding="utf-8")
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    first = ContextBuilder(config).build(project, refresh=True)
    workspace_path = project.agent_dir / "workspace_memory.json"
    workspace = json.loads(workspace_path.read_text(encoding="utf-8"))
    assert "FastAPI" in workspace["detected"]["frameworks"]
    assert "SQLite" not in workspace["detected"]["databases"]
    workspace["manual"] = {"run_commands": ["uvicorn main:app"]}
    workspace_path.write_text(json.dumps(workspace), encoding="utf-8")
    second = ContextBuilder(config).build(project, refresh=True)
    refreshed = json.loads(workspace_path.read_text(encoding="utf-8"))
    assert refreshed["manual"]["run_commands"] == ["uvicorn main:app"]
    assert "Workspace Memory" in first.rendered
    assert "uvicorn main:app" in second.rendered


def test_reflection_rules_and_execution_context(tmp_path: Path, make_config) -> None:
    config = make_config()
    reflection = ReflectionEngine(config).reflect(
        prompt="run tests",
        final="",
        error="timeout",
        success=False,
        tool_calls=[
            {
                "request": {"tool": "shell", "action": "run"},
                "result": {"success": False, "stderr": "timeout after 30s", "duration_ms": 30_000},
            }
            for _ in range(3)
        ],
    )
    assert reflection is not None
    assert "Three or more timeouts" in reflection

    root = tmp_path / "project"
    root.mkdir()
    project = ProjectManager(config).resolve_project(root)
    state = make_state(project)
    state.record_tool_call(
        {"tool": "file", "action": "apply", "args": {"path": "main.py"}},
        {"success": True, "stderr": "", "data": {"path": "main.py", "snapshot_id": "snap-1"}},
    )
    assert state.execution_context is not None
    assert state.execution_context.modified_files == ["main.py"]
    assert state.execution_context.current_snapshot == "snap-1"
    restored = AgentState.from_dict(state.to_dict())
    assert restored.execution_context.current_snapshot == "snap-1"


def test_capability_health_breaks_and_resets_tool(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config({"runtime": {"capability_failure_threshold": 2}})
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    tools = ToolManager(config, project, memory, yolo=True)
    tools.health.record("shell.run", success=False, error="broken one")
    tools.health.record("shell.run", success=False, error="broken two")
    shell = next(item for item in tools.capabilities() if item.name == "shell.run")
    assert tools.health.evaluate(shell).status == "Broken"
    assert "shell_run" not in {item["function"]["name"] for item in tools.schemas()}
    tools.health.reset("shell.run")
    assert tools.health.evaluate(shell).status == "Available"
