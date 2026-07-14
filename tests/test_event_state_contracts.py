from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from dataclasses import fields
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from agent.contracts import EVENT_SCHEMA_VERSION, EVENT_SERIALIZED_FIELDS
from agent.events import Event, EventBus, EventDispatchError, JsonlEventLogger
from agent.project import ProjectManager
from agent.session import SessionManager
from agent.state import AgentState, PlanStep


def _state(tmp_path: Path, make_config) -> AgentState:
    root = tmp_path / "project"
    root.mkdir(parents=True)
    project = ProjectManager(make_config()).resolve_project(root)
    return AgentState.create(
        session_id="state-contract",
        project=project,
        user_request="validate state",
        loaded_memories=[],
        loaded_tools=[],
        git_branch=None,
        context_index_path=str(project.agent_dir / "index.json"),
    )


def test_event_schema_round_trip_and_jsonl_record_are_stable(tmp_path: Path) -> None:
    event = Event(
        "task.finished",
        {"answer": "ok"},
        project_id="project-1",
        session_id="session-1",
        run_id="session-1:turn:1",
        id="event-1",
        timestamp="2026-07-13T16:00:00+00:00",
    )

    assert event.schema_version == EVENT_SCHEMA_VERSION
    assert tuple(event.to_dict()) == EVENT_SERIALIZED_FIELDS
    assert tuple(item.name for item in fields(Event)) == (
        "name",
        "payload",
        "project_id",
        "session_id",
        "run_id",
        "timestamp",
        "id",
        "schema_version",
    )
    assert Event.from_dict(event.to_dict()) == event
    assert event.effective_run_id == "session-1:turn:1"
    assert Event("legacy", {"run_id": "legacy-run"}).effective_run_id == "legacy-run"
    with pytest.raises(ValueError, match="ISO-8601"):
        Event("invalid", {}, timestamp="not-a-timestamp")

    JsonlEventLogger(tmp_path)(event)
    record = json.loads(next(tmp_path.glob("events-*.jsonl")).read_text(encoding="utf-8"))
    assert tuple(record) == EVENT_SERIALIZED_FIELDS
    assert record == event.to_dict() | {"payload": {}}


def test_session_load_rejects_oversized_symlink_and_invalid_messages(tmp_path: Path, make_config) -> None:
    state = _state(tmp_path / "valid", make_config)
    project = ProjectManager(make_config()).resolve_project(Path(state.working_directory))
    sessions = SessionManager(project)
    path = sessions.checkpoint(state, [])

    sessions.MAX_SESSION_FILE_BYTES = 32
    with pytest.raises(ValueError, match="checkpoint exceeds"):
        sessions.checkpoint(state, [])

    sessions.MAX_SESSION_FILE_BYTES = path.stat().st_size - 1
    with pytest.raises(ValueError, match="exceeds"):
        sessions.load(state.session_id)

    sessions.MAX_SESSION_FILE_BYTES = SessionManager.MAX_SESSION_FILE_BYTES
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["messages"] = ["not-a-message"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid session file"):
        sessions.load(state.session_id)

    external = tmp_path / "external-session.json"
    external.write_text(json.dumps(payload | {"messages": []}), encoding="utf-8")
    linked = sessions.session_dir / "linked.json"
    linked.symlink_to(external)
    with pytest.raises(ValueError, match="not a regular file"):
        sessions.load("linked")
    assert all(item.session_id != "linked" for item in sessions.list_sessions(limit=20))


def test_event_bus_supports_unsubscribe_existing_event_and_handler_isolation() -> None:
    bus = EventBus()
    seen: list[tuple[str, str]] = []

    def broken(event: Event) -> None:
        seen.append(("broken", event.id))
        raise RuntimeError("subscriber failed")

    def healthy(event: Event) -> None:
        seen.append(("healthy", event.id))

    cancel_broken = bus.subscribe("task.finished", broken)
    bus.subscribe("task.finished", healthy)
    bus.subscribe("*", lambda event: seen.append(("wildcard", event.id)))

    event = Event("task.finished", {}, id="event-2")
    assert bus.publish(event) is event
    assert seen == [("broken", "event-2"), ("healthy", "event-2"), ("wildcard", "event-2")]
    assert len(bus.last_errors) == 1
    assert "subscriber failed" in bus.last_errors[0]

    cancel_broken()
    assert bus.unsubscribe("task.finished", broken) is False
    bus.publish("task.finished", {}, run_id="run-2")
    assert [name for name, _event_id in seen[-2:]] == ["healthy", "wildcard"]
    with pytest.raises(ValueError, match="cannot be combined"):
        bus.publish(event, {"unexpected": True})


def test_nested_publish_does_not_clobber_outer_handler_errors() -> None:
    bus = EventBus()

    def outer_handler(_event: Event) -> None:
        bus.publish("inner", {})
        raise RuntimeError("outer failed")

    bus.subscribe("outer", outer_handler)
    bus.publish("outer", {})

    assert len(bus.last_errors) == 1
    assert "outer failed" in bus.last_errors[0]


def test_concurrent_tool_events_serialize_shared_side_effect_delivery() -> None:
    bus = EventBus()
    gate = threading.Lock()
    active = 0
    max_active = 0
    seen: list[str] = []

    def shared_side_effect(event: Event) -> None:
        nonlocal active, max_active
        with gate:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.005)
        seen.append(event.id)
        with gate:
            active -= 1

    bus.subscribe("tool.finished", shared_side_effect, name="shared-store")
    events = [Event("tool.finished", {}, id=f"event-{index}") for index in range(12)]

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(bus.publish, events))

    assert max_active == 1
    assert sorted(seen) == sorted(event.id for event in events)


def test_required_dispatch_fails_closed_without_owner_or_on_required_error() -> None:
    bus = EventBus()
    bus.subscribe("*", lambda _event: None, name="audit")

    with pytest.raises(EventDispatchError, match="no required subscribers"):
        bus.dispatch_required("session.checkpoint.requested", {})

    bus.subscribe(
        "session.checkpoint.requested",
        lambda _event: (_ for _ in ()).throw(RuntimeError("disk unavailable")),
        required=True,
        name="session-writer",
    )
    bus.subscribe("session.checkpoint.requested", lambda _event: None, name="metrics")

    with pytest.raises(EventDispatchError, match="disk unavailable") as raised:
        bus.dispatch_required("session.checkpoint.requested", {})

    assert raised.value.event_name == "session.checkpoint.requested"
    assert raised.value.subscriber_succeeded("session-writer") is False
    assert bus.last_dispatch is not None
    assert bus.last_dispatch.handler_count == 3
    assert len(bus.last_dispatch.required_errors) == 1


def test_required_error_exposes_named_owner_delivery_evidence() -> None:
    bus = EventBus()

    def committed(_event: Event) -> None:
        return None

    bus.subscribe("session.checkpoint.requested", committed, required=True, name="session-writer")
    bus.subscribe(
        "session.checkpoint.requested",
        lambda _event: (_ for _ in ()).throw(RuntimeError("required observer failed")),
        required=True,
        name="required-observer",
    )

    with pytest.raises(EventDispatchError) as raised:
        bus.dispatch_required("session.checkpoint.requested", {})

    assert raised.value.subscriber_succeeded("session-writer") is True


def test_required_dispatch_rejects_observer_only_and_duplicate_names() -> None:
    bus = EventBus()
    bus.subscribe("critical", lambda _event: None, name="observer")

    with pytest.raises(EventDispatchError, match="no required subscribers"):
        bus.dispatch_required("critical", {})

    bus.subscribe("critical", lambda _event: None, required=True, name="owner")
    with pytest.raises(ValueError, match="already registered"):
        bus.subscribe("critical", lambda _event: None, required=True, name="owner")


def test_required_dispatch_tolerates_best_effort_failure_after_required_success() -> None:
    bus = EventBus()
    seen: list[str] = []
    bus.subscribe(
        "session.checkpoint.requested",
        lambda _event: seen.append("persisted"),
        required=True,
        name="session-writer",
    )
    bus.subscribe(
        "session.checkpoint.requested",
        lambda _event: (_ for _ in ()).throw(RuntimeError("metrics unavailable")),
        name="metrics",
    )

    dispatch = bus.dispatch_required("session.checkpoint.requested", {})

    assert seen == ["persisted"]
    assert dispatch.required_errors == ()
    assert len(dispatch.errors) == 1


def test_agent_state_rejects_invalid_graph_route_provider_and_context_bounds(tmp_path: Path, make_config) -> None:
    state = _state(tmp_path, make_config)
    state.plan = [
        PlanStep("a", "A", dependencies=["b"]),
        PlanStep("b", "B", dependencies=["a"]),
    ]
    with pytest.raises(ValueError, match="cycle"):
        state.validate()

    state = _state(tmp_path / "provider", make_config)
    state.model_route = {
        "provider": "openai",
        "tier": "standard",
        "model": "not-deepseek",
        "max_tokens": 100,
    }
    with pytest.raises(ValueError, match="provider must be deepseek"):
        state.validate()

    state = _state(tmp_path / "cost", make_config)
    state.model_route = {
        "provider": "deepseek",
        "tier": "standard",
        "model": "deepseek-base",
        "max_tokens": 100,
        "cost_class": "unlimited",
    }
    with pytest.raises(ValueError, match="cost_class"):
        state.validate()

    state = _state(tmp_path / "context", make_config)
    state.context_manifest = {
        "phase": "initial",
        "max_chars": 10,
        "used_chars": 11,
        "rendered_chars": 9,
        "original_user_request_chars": 2,
        "included_memory_ids": [],
    }
    with pytest.raises(ValueError, match="used_chars exceeds"):
        state.validate()


def test_agent_state_frozen_identity_and_future_schema_are_rejected(tmp_path: Path, make_config) -> None:
    state = _state(tmp_path, make_config)
    original = state.to_dict()
    state.user_request = "a mutable field may change"
    assert state.validate_frozen_fields(original) is state

    state.project = {**state.project, "id": "different-project"}
    with pytest.raises(ValueError, match="project"):
        state.validate_frozen_fields(original)
    with pytest.raises(ValueError, match="frozen fields changed"):
        state.to_dict()

    state = AgentState.from_dict(original)
    state.working_directory = str(tmp_path / "moved")
    with pytest.raises(ValueError, match="frozen fields changed"):
        state.complete("must not persist after identity drift")

    future = original | {"schema_version": AgentState.SCHEMA_VERSION + 1}
    with pytest.raises(ValueError, match="unsupported AgentState schema_version"):
        AgentState.from_dict(future)


def test_legacy_state_repairs_only_derived_plan_fields(tmp_path: Path, make_config) -> None:
    legacy = _state(tmp_path, make_config).to_dict()
    legacy["schema_version"] = 1
    legacy["plan"] = [{"id": "legacy", "title": "Legacy", "status": "pending"}]
    legacy["current_step"] = "removed-step"
    legacy["completed_steps"] = ["removed-step"]
    legacy.pop("execution_context")

    restored = AgentState.from_dict(legacy)

    assert restored.schema_version == 1
    assert restored.current_step is None
    assert restored.completed_steps == []
    assert restored.execution_context is not None
    restored.resume("continue")
    assert restored.schema_version == AgentState.SCHEMA_VERSION


def test_agent_state_model_metrics_and_convergence_gates_are_per_turn_while_circuit_survives_resume(
    tmp_path: Path,
    make_config,
) -> None:
    state = _state(tmp_path, make_config)
    state.convergence = {
        "implementation_reads_used": 2,
        "consecutive_read_only_rounds": 8,
        "low_yield_rounds": 5,
        "nudge_count": 2,
        "nudge_sent_for_stall": True,
        "hard_notice_sent": True,
        "notice_turn": 1,
        "seen_targets": ["template.read_file:known"],
        "context_compaction_failure_count": 3,
        "context_compaction_circuit_open": True,
    }
    state.record_model_request("main_loop")
    state.record_model_request("context_compaction")
    state.record_model_request("final_synthesis")
    state.record_model_response(
        SimpleNamespace(
            http_attempt_count=3,
            usage={
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "total_tokens": 150,
            },
        )
    )

    restored = AgentState.from_dict(state.to_dict())

    assert restored.schema_version == 6
    assert restored.model_request_count == 3
    assert restored.main_loop_model_request_count == 1
    assert restored.context_compaction_model_request_count == 1
    assert restored.final_synthesis_model_request_count == 1
    assert restored.model_metrics == {
        "http_attempt_count": 3,
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "total_tokens": 150,
    }

    restored.resume("continue with the remaining verification")

    assert restored.turn == 2
    assert restored.model_request_count == 0
    assert restored.main_loop_model_request_count == 0
    assert restored.context_compaction_model_request_count == 0
    assert restored.final_synthesis_model_request_count == 0
    assert restored.model_metrics == {}
    assert restored.convergence == {
        "seen_targets": ["template.read_file:known"],
        "context_compaction_failure_count": 3,
        "context_compaction_circuit_open": True,
    }


def test_session_summary_names_persisted_tool_turns_without_implying_model_rounds(
    tmp_path: Path,
    make_config,
) -> None:
    root = tmp_path / "session-project"
    root.mkdir()
    project = ProjectManager(make_config()).resolve_project(root)
    state = AgentState.create(
        session_id="tool-turn-summary",
        project=project,
        user_request="inspect one file",
        loaded_memories=[],
        loaded_tools=[],
        git_branch=None,
        context_index_path=str(project.agent_dir / "index.json"),
    )
    state.round = 3
    state.record_tool_call(
        {"tool": "file", "action": "read", "args": {"path": "README.md"}},
        {"success": True, "stdout": "ok", "duration_ms": 4},
    )
    state.complete("done")

    json_path, markdown_path = SessionManager(project).finalize(state, [])

    persisted = json.loads(json_path.read_text(encoding="utf-8"))
    summary = markdown_path.read_text(encoding="utf-8")
    assert persisted["state"]["schema_version"] == 6
    assert "- tool turn 3: file.read success=True duration_ms=4" in summary
    assert "- round 3:" not in summary
