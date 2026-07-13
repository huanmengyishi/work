from __future__ import annotations

import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from agent.deepseek import ChatResponse
from agent.memory import MemoryStore
from agent.project import ProjectManager
from agent.runtime import AgentRuntime
from agent.tools import ToolManager


class FakeClient:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)

    def chat(
        self,
        *,
        messages,
        tools=None,
        tool_choice="auto",
        max_tokens=None,
        thinking=None,
        reasoning_effort=None,
        model=None,
    ) -> ChatResponse:
        if not self.responses:
            raise AssertionError("fake response queue exhausted")
        return ChatResponse(message=self.responses.pop(0), raw={})


class RecordingClient(FakeClient):
    def __init__(self, responses: list[dict]) -> None:
        super().__init__(responses)
        self.requests: list[list[dict]] = []
        self.options: list[dict] = []
        self.models: list[str | None] = []

    def chat(
        self,
        *,
        messages,
        tools=None,
        tool_choice="auto",
        max_tokens=None,
        thinking=None,
        reasoning_effort=None,
        model=None,
    ) -> ChatResponse:
        self.requests.append(list(messages))
        self.options.append({"thinking": thinking, "reasoning_effort": reasoning_effort})
        self.models.append(model)
        return super().chat(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            model=model,
        )


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
    assert state["schema_version"] == 2
    assert state["task_route"]["mode"] == "standard"
    assert state["model_route"]["provider"] == "deepseek"
    assert state["context_manifest"]["used_chars"] <= state["context_manifest"]["max_chars"]
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


def test_runtime_adapts_deep_task_and_reports_reasoning_progress(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "main.py").write_text("print('ok')\n", encoding="utf-8")
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    tools = ToolManager(config, project, memory, yolo=True)
    client = RecordingClient(
        [
            {
                "role": "assistant",
                "content": "complete",
                "reasoning_content": "inspect bounded chunks first",
            }
        ]
    )
    progress: list[dict] = []
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=tools,
        client=client,
        progress_handler=progress.append,
    )

    assert runtime.run("全面审计整个代码库的所有安全问题并完成大规模重构") == "complete"
    session = runtime.sessions.load(runtime.last_session_id)

    assert session.state.task_strategy["mode"] == "deep"
    assert session.state.task_route["task_type"] == "refactor"
    assert session.state.task_route["risk"] == "high"
    assert session.state.model_route["provider"] == "deepseek"
    assert session.state.model_route["tier"] == "deep"
    assert session.state.task_strategy["max_tool_rounds"] == 24
    assert [step.id for step in session.state.plan] == ["scope", "inspect-chunks", "implement", "verify"]
    assert client.options == [{"thinking": True, "reasoning_effort": "max"}]
    assert client.models == ["deepseek-v4-pro"]
    assert any(item["event"] == "thinking.content" for item in progress)


def test_short_resume_keeps_deep_strategy_and_plan(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    tools = ToolManager(config, project, memory, yolo=True)
    client = RecordingClient(
        [
            {"role": "assistant", "content": "checkpoint"},
            {"role": "assistant", "content": "continued"},
        ]
    )
    runtime = AgentRuntime(config=config, project=project, memory=memory, tools=tools, client=client)

    assert runtime.run("全面审计整个代码库并深度重构所有安全问题") == "checkpoint"
    session_id = runtime.last_session_id
    original_state = runtime.sessions.load(session_id).state
    original_plan = [step.id for step in original_state.plan]
    original_model = original_state.model_route["model"]
    assert runtime.resume("继续", session_id) == "continued"
    resumed = runtime.sessions.load(session_id).state

    assert resumed.task_strategy["mode"] == "deep"
    assert resumed.task_route["mode"] == "deep"
    assert resumed.model_route["tier"] == "deep"
    assert resumed.model_route["model"] == original_model
    assert resumed.context_manifest["phase"] == "resume"
    assert [step.id for step in resumed.plan] == original_plan
    assert client.options[-1] == {"thinking": True, "reasoning_effort": "max"}
    assert client.models == [original_model, original_model]


def test_concurrent_resume_rejects_second_session_turn(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    started = threading.Event()
    release = threading.Event()

    class BlockingClient(FakeClient):
        def chat(self, **kwargs) -> ChatResponse:
            started.set()
            assert release.wait(timeout=5)
            return super().chat(**kwargs)

    initial_runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=FakeClient([{"role": "assistant", "content": "checkpoint"}]),
    )
    assert initial_runtime.run("create resumable session") == "checkpoint"
    session_id = initial_runtime.last_session_id
    first = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=BlockingClient([{"role": "assistant", "content": "first"}]),
    )
    second = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=FakeClient([{"role": "assistant", "content": "second"}]),
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(first.resume, "first continuation", session_id)
        assert started.wait(timeout=5)
        with pytest.raises(RuntimeError, match="already being resumed"):
            second.resume("second continuation", session_id)
        release.set()
        assert future.result(timeout=5) == "first"

    state = first.sessions.load(session_id).state
    assert state.turn == 2
    assert state.final_answer == "first"


def test_runtime_respects_small_context_hard_limit_and_input_limit(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config(
        {
            "context": {
                "package_limits": {"simple": 900},
                "max_package_chars_hard_limit": 500,
            },
            "runtime": {"max_user_request_chars": 20},
        }
    )
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    client = RecordingClient([{"role": "assistant", "content": "bounded"}])
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )

    assert runtime.run("什么是 Python？") == "bounded"
    state = runtime.sessions.load(runtime.last_session_id).state
    assert state.context_manifest["max_chars"] == 500
    assert state.context_manifest["used_chars"] <= 500
    assert len(client.requests[0][1]["content"]) + len(client.requests[0][2]["content"]) <= 500
    with pytest.raises(ValueError, match="save large text/code"):
        runtime.run("x" * 21)


def test_resume_repeated_failure_escalates_model_without_losing_task_type(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    client = RecordingClient(
        [
            {"role": "assistant", "content": "checkpoint"},
            {"role": "assistant", "content": "recovered"},
        ]
    )
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )
    prompt = "修复这个错误并运行测试 " + "详细范围 " * 150
    assert runtime.run(prompt) == "checkpoint"
    session_id = runtime.last_session_id
    record = runtime.sessions.load(session_id)
    original_type = record.state.task_route["task_type"]
    assert record.state.task_route["score"] == 2
    record.state.tool_calls.extend(
        [
            {"result": {"success": False}},
            {"result": {"success": False}},
        ]
    )
    runtime.sessions.checkpoint(record.state, record.messages)

    assert runtime.resume("继续", session_id) == "recovered"
    resumed = runtime.sessions.load(session_id).state
    assert resumed.task_route["task_type"] == original_type
    assert resumed.task_route["failure_count"] == 2
    assert resumed.model_route["tier"] == "deep"
    assert client.models[-1] == resumed.model_route["model"]


def test_runtime_falls_back_from_malformed_context_limits(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config(
        {
            "context": {
                "package_limits": {"simple": None},
                "max_package_chars_hard_limit": "invalid",
                "max_recovery_context_chars": None,
            },
            "runtime": {"max_user_request_chars": "invalid"},
        }
    )
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=FakeClient([{"role": "assistant", "content": "ok"}]),
    )

    assert runtime.run("什么是 Python？") == "ok"
    state = runtime.sessions.load(runtime.last_session_id).state
    assert state.context_manifest["max_chars"] == 12_000
    assert (
        runtime._bounded_config_int("context.max_recovery_context_chars", 6_000, minimum=0, maximum=1_000_000) == 6_000
    )


def test_resume_keeps_large_scope_but_upgrades_architecture_model(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    client = RecordingClient(
        [
            {"role": "assistant", "content": "large checkpoint"},
            {"role": "assistant", "content": "architecture complete"},
        ]
    )
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )

    assert runtime.run("分析整个代码库的所有文件并总结") == "large checkpoint"
    session_id = runtime.last_session_id
    first = runtime.sessions.load(session_id).state
    assert first.task_route["mode"] == "large"
    assert first.model_route["tier"] == "standard"

    assert runtime.resume("请解释并设计系统架构", session_id) == "architecture complete"
    resumed = runtime.sessions.load(session_id).state
    assert resumed.task_route["mode"] == "large"
    assert resumed.model_route["tier"] == "deep"
    assert client.options[-1]["reasoning_effort"] == "max"
