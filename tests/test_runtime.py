from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from agent.deepseek import ChatResponse
from agent.memory import MemoryStore
from agent.project import ProjectManager
from agent.runtime import AgentRuntime
from agent.tools import ToolManager


class FakeClient:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)

    def chat(self, *, messages, tools=None, tool_choice="auto", max_tokens=None) -> ChatResponse:
        if not self.responses:
            raise AssertionError("fake response queue exhausted")
        return ChatResponse(message=self.responses.pop(0), raw={})


class RecordingClient(FakeClient):
    def __init__(self, responses: list[dict]) -> None:
        super().__init__(responses)
        self.requests: list[list[dict]] = []

    def chat(self, *, messages, tools=None, tool_choice="auto", max_tokens=None) -> ChatResponse:
        self.requests.append(list(messages))
        return super().chat(messages=messages, tools=tools, tool_choice=tool_choice, max_tokens=max_tokens)


def tool_message(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def test_runtime_checkpoint_resume_events_and_memory_pipeline(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "main.py").write_text("def main():\n    return 'ok'\n", encoding="utf-8")
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    memory.sync_project(project)
    tools = ToolManager(config, project, memory, yolo=True)
    client = FakeClient(
        [
            tool_message(
                "plan-1",
                "agent_update_plan",
                {"steps": [{"id": "inspect", "title": "Inspect", "status": "in_progress"}]},
            ),
            tool_message("shell-1", "shell_run", {"command": "printf first-turn"}),
            {"role": "assistant", "content": "first complete"},
            tool_message("shell-2", "shell_run", {"command": "printf second-turn"}),
            {"role": "assistant", "content": "second complete"},
        ]
    )
    runtime = AgentRuntime(config=config, project=project, memory=memory, tools=tools, client=client)

    assert runtime.run("inspect the project") == "first complete"
    session_id = runtime.last_session_id
    assert session_id
    assert runtime.resume("continue verification", session_id) == "second complete"

    session_path = project.agent_dir / "sessions" / f"{session_id}.json"
    payload = json.loads(session_path.read_text(encoding="utf-8"))
    state = payload["state"]
    assert state["status"] == "completed"
    assert state["turn"] == 2
    assert [item["turn"] for item in state["tool_calls"]] == [1, 1, 2]
    assert state["tool_calls"][-1]["result"]["stdout"] == "second-turn"
    assert (project.agent_dir / "sessions" / f"{session_id}.md").exists()
    assert (project.agent_dir / "index.json").exists()

    memories = memory.recent(project.id, limit=20)
    assert [item.kind for item in memories].count("Summary") == 2
    assert [item.kind for item in memories].count("Lesson") == 2
    with sqlite3.connect(memory.db_path) as con:
        assert con.execute("select count(*) from pipeline_runs").fetchone()[0] == 2

    before = len(memory.recent(project.id, limit=20))
    runtime._publish_terminal("task.finished", runtime.sessions.load(session_id).state, final="duplicate")
    after = len(memory.recent(project.id, limit=20))
    assert after == before


def test_runtime_injects_recovery_memory_after_tool_failure(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    memory.sync_project(project)
    memory.add_memory(
        kind="Correction",
        title="Missing command",
        content="Install missing-command before retrying.",
        tags=["correction:dependency", project.name],
        project_id=project.id,
    )
    tools = ToolManager(config, project, memory, yolo=True)
    client = RecordingClient(
        [
            tool_message("shell-1", "shell_run", {"command": "missing-command --version"}),
            {"role": "assistant", "content": "diagnosed"},
        ]
    )
    runtime = AgentRuntime(config=config, project=project, memory=memory, tools=tools, client=client)
    assert runtime.run("run the missing command") == "diagnosed"
    assert any("Failure Recovery Memory" in str(message.get("content")) for message in client.requests[1])
