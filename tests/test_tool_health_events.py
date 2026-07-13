from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from agent.event_pipelines import CapabilityHealthEventPipeline
from agent.events import Event, EventBus
from agent.memory import MemoryStore
from agent.project import ProjectManager
from agent.tools import ToolManager
from agent.tools.base import ToolResult
from agent.tools.registry import ToolCapability


def test_event_pipeline_can_be_imported_before_tool_manager() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from agent.event_pipelines import RuntimeEventPipelines; "
            "from agent.tools import ToolManager; "
            "assert RuntimeEventPipelines and ToolManager",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert completed.returncode == 0, completed.stderr


def _manager(
    tmp_path: Path,
    make_config,
    events: EventBus,
    overrides: dict[str, Any] | None = None,
) -> ToolManager:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config(overrides)
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    memory.sync_project(project)
    return ToolManager(config, project, memory, events=events, yolo=True)


def _register_probe(manager: ToolManager, handler: Callable[..., ToolResult]) -> None:
    manager.registry.register(
        ToolCapability(
            "probe",
            "run",
            "probe_run",
            "Deterministic test capability.",
            {"value": {"type": "string"}},
            permissions=("read",),
        ),
        handler,
    )


def _capture(events: EventBus, event_name: str = "tool.finished") -> list[Event]:
    captured: list[Event] = []
    events.subscribe(event_name, captured.append, name="test.capture")
    return captured


def test_success_updates_health_only_through_tool_finished_event(tmp_path: Path, make_config) -> None:
    events = EventBus()
    manager = _manager(tmp_path, make_config, events)
    CapabilityHealthEventPipeline(manager.health, events)
    captured = _capture(events)
    _register_probe(manager, lambda **_kwargs: ToolResult(True, "ok", data={"private": "not-audit-data"}))

    request, result = manager.execute_model_call("probe_run", {"value": "input-private"})

    assert result.success is True
    assert result.stdout == "ok"
    assert manager.health.records[request.capability]["consecutive_failures"] == 0
    assert captured[0].payload["result"] == {
        "success": True,
        "duration_ms": result.duration_ms,
        "request_id": request.request_id,
        "data_field_count": 1,
        "health_failure": False,
        "error": "",
    }


def test_health_failure_is_classified_and_persisted_with_bounded_redacted_summary(
    tmp_path: Path,
    make_config,
) -> None:
    events = EventBus()
    manager = _manager(tmp_path, make_config, events)
    CapabilityHealthEventPipeline(manager.health, events)
    captured = _capture(events)
    error = "connection refused password=hunter2 " + ("detail " * 300)
    _register_probe(manager, lambda **_kwargs: ToolResult(False, "original stdout", error))

    request, result = manager.execute_model_call("probe_run", {})

    event_result = captured[0].payload["result"]
    health_record = manager.health.records[request.capability]
    assert result.stderr == error
    assert event_result["health_failure"] is True
    assert 0 < len(event_result["error"]) <= 1000
    assert event_result["error"].endswith("...[truncated]")
    assert "hunter2" not in event_result["error"]
    assert health_record["consecutive_failures"] == 1
    assert health_record["last_failure"] == event_result["error"]


def test_business_failure_does_not_degrade_capability_health(tmp_path: Path, make_config) -> None:
    events = EventBus()
    manager = _manager(tmp_path, make_config, events)
    CapabilityHealthEventPipeline(manager.health, events)
    captured = _capture(events)
    _register_probe(
        manager,
        lambda **_kwargs: ToolResult(
            False,
            "business output private-value",
            "validation failed for private-value",
            {"args": {"value": "private-value"}},
        ),
    )

    request, result = manager.execute_model_call("probe_run", {"value": "private-value"})

    assert result.success is False
    assert request.capability not in manager.health.records
    assert captured[0].payload["result"]["health_failure"] is False
    assert captured[0].payload["result"]["error"] == ""


def test_health_subscriber_failure_is_best_effort_and_preserves_tool_result(
    monkeypatch,
    tmp_path: Path,
    make_config,
) -> None:
    events = EventBus()
    manager = _manager(tmp_path, make_config, events)
    CapabilityHealthEventPipeline(manager.health, events)
    captured = _capture(events)

    def fail_health_write(*_args, **_kwargs) -> None:
        raise OSError("health store unavailable")

    monkeypatch.setattr(manager.health, "record", fail_health_write)
    expected_data = {"nested": "kept in ToolResult"}
    _register_probe(manager, lambda **_kwargs: ToolResult(True, "original output", "", expected_data, 37))

    request, result = manager.execute_model_call("probe_run", {})

    assert result == ToolResult(True, "original output", "", expected_data, 37, request.request_id)
    assert len(captured) == 1
    assert any("capability-health" in error and "health store unavailable" in error for error in events.last_errors)


def test_tool_finished_payload_never_contains_arguments_or_tool_output(tmp_path: Path, make_config) -> None:
    events = EventBus()
    manager = _manager(tmp_path, make_config, events)
    captured = _capture(events)
    argument_secret = "argument-private-value"
    stdout_secret = "stdout-private-value"
    stderr_secret = "stderr-private-value"
    data_secret = "data-private-value"
    bearer_secret = "bearer-private-value"
    _register_probe(
        manager,
        lambda **_kwargs: ToolResult(
            False,
            stdout_secret,
            (f"connection refused password={stderr_secret} Authorization=Bearer {bearer_secret}"),
            {"args": argument_secret, "stdout": stdout_secret, "stderr": data_secret},
        ),
    )

    manager.execute_model_call(
        "probe_run",
        {
            "value": argument_secret,
            "path": f"/private/{argument_secret}",
            "token": "request-token-private-value",
        },
    )

    payload = captured[0].payload
    rendered = json.dumps(payload, ensure_ascii=False)
    assert manager.health.records == {}
    assert payload["request"]["argument_count"] == 3
    assert set(payload["request"]) == {"tool", "action", "capability", "request_id", "argument_count"}
    assert set(payload["result"]) == {
        "success",
        "duration_ms",
        "request_id",
        "data_field_count",
        "health_failure",
        "error",
    }
    assert len(payload["result"]["error"]) <= 1000
    for private_value in (
        argument_secret,
        stdout_secret,
        stderr_secret,
        data_secret,
        bearer_secret,
        "request-token-private-value",
    ):
        assert private_value not in rendered
    for forbidden_key in ('"args"', '"stdout"', '"stderr"', '"path"', '"token"'):
        assert forbidden_key not in rendered


def test_permission_denial_still_precedes_handler_and_health_event(tmp_path: Path, make_config) -> None:
    events = EventBus()
    manager = _manager(
        tmp_path,
        make_config,
        events,
        {"permissions": {"deny_capabilities": ["probe.run"]}},
    )
    event_names: list[str] = []
    events.subscribe("*", lambda event: event_names.append(event.name), name="test.capture-all")
    calls = 0

    def handler(**_kwargs) -> ToolResult:
        nonlocal calls
        calls += 1
        return ToolResult(True, "must not run")

    _register_probe(manager, handler)

    request, result = manager.execute_model_call("probe_run", {"value": "blocked"})

    assert result.success is False
    assert "denied by policy" in result.stderr
    assert calls == 0
    assert event_names == ["tool.denied"]
    assert request.capability not in manager.health.records
