from __future__ import annotations

import ast
import sqlite3
from pathlib import Path
from types import MethodType

import pytest

from agent.event_pipelines import (
    MEMORY_USAGE_RECORDED,
    PROGRESS_UPDATED,
    SESSION_CHECKPOINT_REQUESTED,
    SESSION_FINALIZE_REQUESTED,
)
from agent.events import EventBus, EventDispatchError
from agent.memory import MemoryStore
from agent.project import ProjectManager
from agent.runtime import AgentRuntime
from agent.session import SessionManager
from agent.state import AgentState
from agent.tools import ToolManager


class _FinalClient:
    def __init__(self, content: str = "complete") -> None:
        self.content = content

    def chat(self, **_kwargs):
        from agent.deepseek import ChatResponse

        return ChatResponse(message={"role": "assistant", "content": self.content}, raw={})


class _RecordingSessions(SessionManager):
    def __init__(self, project, calls: list[str]) -> None:
        super().__init__(project)
        self.calls = calls

    def checkpoint(self, state, messages):
        self.calls.append("session.checkpoint")
        return super().checkpoint(state, messages)

    def finalize(self, state, messages):
        self.calls.append("session.finalize")
        return super().finalize(state, messages)


class _FailingSessions(SessionManager):
    def __init__(self, project, *, fail_checkpoint: bool = False, fail_finalize: bool = False) -> None:
        super().__init__(project)
        self.fail_checkpoint = fail_checkpoint
        self.fail_finalize = fail_finalize
        self.checkpoint_calls = 0
        self.finalize_calls = 0

    def checkpoint(self, state, messages):
        self.checkpoint_calls += 1
        if self.fail_checkpoint:
            raise OSError("checkpoint disk unavailable")
        return super().checkpoint(state, messages)

    def finalize(self, state, messages):
        self.finalize_calls += 1
        if self.fail_finalize:
            raise OSError("finalize disk unavailable")
        return super().finalize(state, messages)


def _runtime(tmp_path: Path, make_config, *, events=None, sessions=None, memory=None) -> AgentRuntime:
    root = tmp_path / "project"
    root.mkdir(exist_ok=True)
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    store = memory or MemoryStore(config)
    tools = ToolManager(config, project, store, yolo=True)
    return AgentRuntime(
        config=config,
        project=project,
        memory=store,
        tools=tools,
        client=_FinalClient(),
        events=events,
        sessions=sessions,
    )


def test_runtime_registers_event_pipelines_and_has_no_legacy_session_or_memory_writes() -> None:
    source_path = Path(__file__).parents[1] / "agent" / "runtime.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names
    }
    direct_calls = {
        (node.func.value.attr, node.func.attr)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Attribute)
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "self"
    }

    assert "JsonlEventLogger" not in imports
    assert "MemoryPipeline" not in imports
    assert ("sessions", "checkpoint") not in direct_calls
    assert ("sessions", "finalize") not in direct_calls
    assert ("memory", "record_usage") not in direct_calls


def test_progress_is_forwarded_by_best_effort_event_and_not_audited(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    events = EventBus()
    seen: list[dict] = []
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=_FinalClient(),
        events=events,
        progress_handler=seen.append,
    )

    captured = []
    events.subscribe(PROGRESS_UPDATED, captured.append, name="test.progress-capture")
    assert runtime.run("explain this project") == "complete"

    assert any(item["event"] == "strategy.selected" for item in seen)
    assert any(item["event"] == "model.requested" for item in seen)
    assert all(event.payload.get("value") == item for event, item in zip(captured, seen, strict=True))

    failing_events = EventBus()
    failed_progress_calls = 0

    def broken_progress(_value) -> None:
        nonlocal failed_progress_calls
        failed_progress_calls += 1
        raise RuntimeError("terminal unavailable")

    failing_runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=_FinalClient("still complete"),
        events=failing_events,
        progress_handler=broken_progress,
    )
    assert failing_runtime.run("explain another part") == "still complete"
    assert failed_progress_calls >= 2


def test_required_session_events_are_owned_and_persist_before_terminal(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    calls: list[str] = []
    events = EventBus()
    sessions = _RecordingSessions(project, calls)
    events.subscribe("task.finished", lambda _event: calls.append("task.finished"), name="order-observer")
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=_FinalClient(),
        events=events,
        sessions=sessions,
    )

    assert events.subscriber_count(SESSION_CHECKPOINT_REQUESTED) == 1
    assert events.subscriber_count(SESSION_FINALIZE_REQUESTED) == 1
    assert events.subscriber_count(MEMORY_USAGE_RECORDED) == 1
    assert runtime.run("explain this project") == "complete"
    assert calls == ["session.checkpoint", "session.finalize", "session.checkpoint", "task.finished"]


def test_missing_required_owner_and_session_write_failure_fail_closed(tmp_path: Path, make_config) -> None:
    runtime = _runtime(tmp_path, make_config)
    checkpoint_handler = runtime.event_pipelines.session.checkpoint
    assert runtime.events.unsubscribe(SESSION_CHECKPOINT_REQUESTED, checkpoint_handler) is True

    with pytest.raises(EventDispatchError, match="no required subscribers"):
        runtime.run("explain this project")
    assert runtime.last_session_id is not None

    root = tmp_path / "failing-project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    sessions = _FailingSessions(project, fail_checkpoint=True)
    failing_runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=_FinalClient(),
        sessions=sessions,
    )

    with pytest.raises(EventDispatchError, match="checkpoint disk unavailable"):
        failing_runtime.run("explain this project")
    assert sessions.checkpoint_calls == 1
    assert sessions.finalize_calls == 0


def test_finalize_failure_does_not_publish_terminal_or_recurse(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    sessions = _FailingSessions(project, fail_finalize=True)
    events = EventBus()
    terminals: list[str] = []
    events.subscribe("task.finished", lambda event: terminals.append(event.name), name="finished-observer")
    events.subscribe("task.failed", lambda event: terminals.append(event.name), name="failed-observer")
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=_FinalClient(),
        sessions=sessions,
        events=events,
    )

    with pytest.raises(EventDispatchError, match="finalize disk unavailable"):
        runtime.run("explain this project")
    assert sessions.finalize_calls == 1
    assert terminals == []


def test_tool_checkpoint_failure_finalizes_once_before_failed_terminal(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)

    class _ToolClient:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, **_kwargs):
            from agent.deepseek import ChatResponse

            self.calls += 1
            if self.calls == 1:
                return ChatResponse(
                    message={
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "tool-1",
                                "type": "function",
                                "function": {"name": "shell_run", "arguments": '{"command":"printf ok"}'},
                            }
                        ],
                    },
                    raw={},
                )
            return ChatResponse(message={"role": "assistant", "content": "unexpected"}, raw={})

    class _SecondCheckpointFails(_FailingSessions):
        def checkpoint(self, state, messages):
            self.checkpoint_calls += 1
            if self.checkpoint_calls == 2:
                raise OSError("tool checkpoint unavailable")
            return SessionManager.checkpoint(self, state, messages)

    sessions = _SecondCheckpointFails(project)
    events = EventBus()
    terminals: list[str] = []
    events.subscribe("task.finished", lambda event: terminals.append(event.name), name="finished-observer")
    events.subscribe("task.failed", lambda event: terminals.append(event.name), name="failed-observer")
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=_ToolClient(),
        sessions=sessions,
        events=events,
    )

    with pytest.raises(EventDispatchError, match="tool checkpoint unavailable"):
        runtime.run("run one tool")
    assert sessions.checkpoint_calls == 2
    assert sessions.finalize_calls == 0
    assert terminals == []
    saved = sessions.load(runtime.last_session_id).state
    assert saved.status == "running"


def test_checkpoint_observer_failure_after_commit_preserves_recoverable_terminal(
    tmp_path: Path,
    make_config,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)

    class _ToolClient:
        def chat(self, **_kwargs):
            from agent.deepseek import ChatResponse

            return ChatResponse(
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tool-1",
                            "type": "function",
                            "function": {"name": "shell_run", "arguments": '{"command":"printf ok"}'},
                        }
                    ],
                },
                raw={},
            )

    events = EventBus()
    terminal_names: list[str] = []
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=_ToolClient(),
        events=events,
    )
    checkpoint_calls = 0

    def fail_second_observation(_event) -> None:
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        if checkpoint_calls == 2:
            raise RuntimeError("required checkpoint observer unavailable")

    events.subscribe(
        SESSION_CHECKPOINT_REQUESTED,
        fail_second_observation,
        required=True,
        name="test.required-observer",
    )
    events.subscribe("task.failed", lambda event: terminal_names.append(event.name), name="test.terminal")

    with pytest.raises(EventDispatchError, match="required checkpoint observer unavailable"):
        runtime.run("run one tool")

    saved = runtime.sessions.load(runtime.last_session_id).state
    assert saved.status == "failed"
    assert terminal_names == ["task.failed"]


def test_tool_checkpoint_and_recovery_finalize_failure_do_not_publish_terminal(
    tmp_path: Path,
    make_config,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)

    class _ToolClient:
        def chat(self, **_kwargs):
            from agent.deepseek import ChatResponse

            return ChatResponse(
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tool-1",
                            "type": "function",
                            "function": {"name": "shell_run", "arguments": '{"command":"printf ok"}'},
                        }
                    ],
                },
                raw={},
            )

    class _CheckpointAndFinalizeFail(_FailingSessions):
        def checkpoint(self, state, messages):
            self.checkpoint_calls += 1
            if self.checkpoint_calls >= 2:
                raise OSError("all session writes unavailable")
            return SessionManager.checkpoint(self, state, messages)

        def finalize(self, state, messages):
            self.finalize_calls += 1
            raise OSError("all session writes unavailable")

    sessions = _CheckpointAndFinalizeFail(project)
    events = EventBus()
    terminals: list[str] = []
    events.subscribe("task.failed", lambda event: terminals.append(event.name), name="failed-observer")
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=_ToolClient(),
        sessions=sessions,
        events=events,
    )

    with pytest.raises(EventDispatchError) as raised:
        runtime.run("run one tool")
    assert raised.value.event_name == SESSION_CHECKPOINT_REQUESTED
    assert sessions.checkpoint_calls == 2
    assert sessions.finalize_calls == 0
    assert terminals == []


def test_memory_usage_event_is_idempotent_and_updates_state_only_after_success(tmp_path: Path, make_config) -> None:
    runtime = _runtime(tmp_path, make_config)
    memory_id = runtime.memory.add_memory(
        kind="Lesson",
        title="included context",
        content="Only count this memory after required event persistence succeeds.",
        project_id=runtime.project.id,
    )
    snapshot = runtime.context_builder.build(runtime.project)
    agent_state = AgentState.create(
        session_id=runtime.sessions.new_session_id(),
        project=runtime.project,
        user_request="use memory",
        loaded_memories=[],
        loaded_tools=[],
        git_branch=snapshot.git_branch,
        context_index_path=str(snapshot.index_path),
    )

    runtime._record_included_memories(agent_state, (memory_id, memory_id))
    runtime._record_included_memories(agent_state, (memory_id,))
    assert agent_state.loaded_memories == [memory_id]
    assert runtime.memory.get_memory(memory_id).use_count == 1

    usage_id = runtime._memory_usage_id(agent_state, [memory_id])
    runtime.events.dispatch_required(
        MEMORY_USAGE_RECORDED,
        {"memory_ids": [memory_id], "usage_id": usage_id},
        project_id=runtime.project.id,
        session_id=agent_state.session_id,
        run_id=agent_state.run_id,
    )
    assert runtime.memory.get_memory(memory_id).use_count == 1
    with sqlite3.connect(runtime.memory.db_path) as con:
        assert con.execute("select count(*) from memory_usage_events").fetchone()[0] == 1

    memory_handler = runtime.event_pipelines.memory_usage.handle
    assert runtime.events.unsubscribe(MEMORY_USAGE_RECORDED, memory_handler) is True
    second_id = runtime.memory.add_memory(
        kind="Lesson",
        title="failed context usage",
        content="This ID must stay absent from state when persistence has no owner.",
        project_id=runtime.project.id,
    )
    with pytest.raises(EventDispatchError, match="no required subscribers"):
        runtime._record_included_memories(agent_state, (second_id,))
    assert second_id not in agent_state.loaded_memories
    assert runtime.memory.get_memory(second_id).use_count == 0


def test_memory_usage_write_failure_keeps_state_retryable(tmp_path: Path, make_config) -> None:
    runtime = _runtime(tmp_path, make_config)
    memory_id = runtime.memory.add_memory(
        kind="Lesson",
        title="retryable usage",
        content="A failed usage write must not mark AgentState as already recorded.",
        project_id=runtime.project.id,
    )
    snapshot = runtime.context_builder.build(runtime.project)
    state = AgentState.create(
        session_id=runtime.sessions.new_session_id(),
        project=runtime.project,
        user_request="use retryable memory",
        loaded_memories=[],
        loaded_tools=[],
        git_branch=snapshot.git_branch,
        context_index_path=str(snapshot.index_path),
    )
    original = runtime.memory.record_usage_once

    def fail_usage(self, *_args, **_kwargs):
        raise sqlite3.OperationalError("memory database unavailable")

    runtime.memory.record_usage_once = MethodType(fail_usage, runtime.memory)
    with pytest.raises(EventDispatchError, match="memory database unavailable"):
        runtime._record_included_memories(state, (memory_id,))
    assert state.loaded_memories == []
    assert runtime.memory.get_memory(memory_id).use_count == 0

    runtime.memory.record_usage_once = original
    runtime._record_included_memories(state, (memory_id,))
    assert state.loaded_memories == [memory_id]
    assert runtime.memory.get_memory(memory_id).use_count == 1


def test_memory_usage_rejects_same_id_with_different_evidence(tmp_path: Path, make_config) -> None:
    runtime = _runtime(tmp_path, make_config)
    first_id = runtime.memory.add_memory(
        kind="Lesson",
        title="first evidence",
        content="First immutable usage evidence.",
        project_id=runtime.project.id,
    )
    second_id = runtime.memory.add_memory(
        kind="Lesson",
        title="second evidence",
        content="Second conflicting usage evidence.",
        project_id=runtime.project.id,
    )
    usage_id = "session-1:turn:1:memory:batch-1"
    runtime.events.dispatch_required(
        MEMORY_USAGE_RECORDED,
        {"memory_ids": [first_id], "usage_id": usage_id},
        project_id=runtime.project.id,
        session_id="session-1",
        run_id="session-1:turn:1",
    )

    with pytest.raises(EventDispatchError, match="different evidence"):
        runtime.events.dispatch_required(
            MEMORY_USAGE_RECORDED,
            {"memory_ids": [second_id], "usage_id": usage_id},
            project_id=runtime.project.id,
            session_id="session-1",
            run_id="session-1:turn:1",
        )

    assert runtime.memory.get_memory(first_id).use_count == 1
    assert runtime.memory.get_memory(second_id).use_count == 0


def test_memory_usage_batch_rolls_back_when_journal_insert_fails(tmp_path: Path, make_config) -> None:
    runtime = _runtime(tmp_path, make_config)
    memory_id = runtime.memory.add_memory(
        kind="Lesson",
        title="transaction evidence",
        content="Usage and its idempotency journal must commit atomically.",
        project_id=runtime.project.id,
    )
    with sqlite3.connect(runtime.memory.db_path) as con:
        con.execute(
            """
            create trigger fail_memory_usage_journal before insert on memory_usage_events
            begin
                select raise(abort, 'journal unavailable');
            end
            """
        )

    with pytest.raises(EventDispatchError, match="journal unavailable"):
        runtime.events.dispatch_required(
            MEMORY_USAGE_RECORDED,
            {"memory_ids": [memory_id], "usage_id": "transaction-batch-1"},
            project_id=runtime.project.id,
            session_id="session-1",
            run_id="session-1:turn:1",
        )

    assert runtime.memory.get_memory(memory_id).use_count == 0
    with sqlite3.connect(runtime.memory.db_path) as con:
        assert con.execute("select count(*) from memory_usage_events").fetchone()[0] == 0


def test_memory_usage_missing_id_rolls_back_the_entire_batch(tmp_path: Path, make_config) -> None:
    runtime = _runtime(tmp_path, make_config)
    memory_id = runtime.memory.add_memory(
        kind="Lesson",
        title="existing evidence",
        content="A mixed valid/missing batch must not partially increment usage.",
        project_id=runtime.project.id,
    )

    with pytest.raises(EventDispatchError, match="missing or merged IDs"):
        runtime.events.dispatch_required(
            MEMORY_USAGE_RECORDED,
            {"memory_ids": [memory_id, 999_999], "usage_id": "missing-id-batch"},
            project_id=runtime.project.id,
            session_id="session-1",
            run_id="session-1:turn:1",
        )

    assert runtime.memory.get_memory(memory_id).use_count == 0
    with sqlite3.connect(runtime.memory.db_path) as con:
        assert con.execute("select count(*) from memory_usage_events").fetchone()[0] == 0


def test_resume_uses_new_run_id_and_records_new_memory_evidence_once(tmp_path: Path, make_config) -> None:
    runtime = _runtime(tmp_path, make_config)
    memory_id = runtime.memory.add_memory(
        kind="Lesson",
        title="resume context marker",
        content="A resumed turn may reinforce the same memory once for its new run ID.",
        project_id=runtime.project.id,
    )

    assert runtime.run("resume context marker") == "complete"
    session_id = runtime.last_session_id
    assert runtime.memory.get_memory(memory_id).use_count == 1
    assert runtime.resume("resume context marker", session_id) == "complete"
    assert runtime.memory.get_memory(memory_id).use_count == 2
    record = runtime.sessions.load(session_id)
    assert record.state.turn == 2
    assert memory_id in record.state.loaded_memories
    with sqlite3.connect(runtime.memory.db_path) as con:
        usage_runs = con.execute("select run_id from memory_usage_events order by recorded_at, usage_id").fetchall()
    assert {row[0] for row in usage_runs} == {f"{session_id}:turn:1", f"{session_id}:turn:2"}
