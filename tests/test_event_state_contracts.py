from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import pytest

from agent.contracts import EVENT_SCHEMA_VERSION, EVENT_SERIALIZED_FIELDS
from agent.events import Event, EventBus, JsonlEventLogger
from agent.project import ProjectManager
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
    assert Event.from_dict(record) == event


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
