from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.event_pipelines import EventMetricsCollector, RuntimeEventPipelines
from agent.events import Event, EventBus, JsonlEventLogger, audit_event_projection, sanitize_for_log
from agent.memory import MemoryStore
from agent.daemon import ProjectDaemon
from agent.parallel import ParallelWorktreeRunner
from agent.project import Project
from agent.session import SessionManager
from agent.tools import ToolManager


class LeakingObject:
    def __str__(self) -> str:
        raise AssertionError("audit must never stringify arbitrary payload objects")


def _read_jsonl(path: Path) -> dict:
    return json.loads(next(path.glob("events-*.jsonl")).read_text(encoding="utf-8"))


def test_audit_internal_events_are_metadata_only_and_never_stringify_objects(tmp_path: Path) -> None:
    logger = JsonlEventLogger(tmp_path / "logs")
    secret = "audit-never-write-this-private-value"
    logger(
        Event(
            "session.finalize.requested",
            {
                "state": LeakingObject(),
                "messages": [{"content": secret}],
                "PROMPT": secret,
                "Reasoning_Content": secret,
                "StdOut": secret,
                "STDERR": secret,
                "tool_args": {"Authorization": secret},
                "unknown": LeakingObject(),
            },
            project_id="project-1",
            session_id="session-1",
            run_id="run-1",
            id="event-1",
            timestamp="2026-07-14T00:00:00+00:00",
        )
    )

    record = _read_jsonl(tmp_path / "logs")
    rendered = json.dumps(record, ensure_ascii=False)
    assert record["payload"] == {}
    assert secret not in rendered
    assert tuple(record) == Event.SERIALIZED_FIELDS


def test_audit_drops_progress_reasoning_content(tmp_path: Path) -> None:
    logger = JsonlEventLogger(tmp_path / "logs")
    secret = "thinking-private-content"
    logger(Event("ui.progress.updated", {"value": {"event": "thinking.delta", "content": secret}}))

    record = _read_jsonl(tmp_path / "logs")
    assert record["payload"] == {}
    assert secret not in json.dumps(record, ensure_ascii=False)


def test_audit_tool_event_projects_only_bounded_non_content_metadata(tmp_path: Path) -> None:
    logger = JsonlEventLogger(tmp_path / "logs")
    secret = "nested-private-tool-body"
    logger(
        Event(
            "tool.finished",
            {
                "request": {
                    "tool": "shell",
                    "action": "run",
                    "capability": "shell.run",
                    "request_id": "request-1",
                    "argument_count": 4,
                    "argument_names": ["cwd", "api_key", "StdOut", "command"],
                    "path": secret,
                    "ArGuMeNtS": {"command": secret},
                    "nested": {"content": secret},
                },
                "result": {
                    "success": False,
                    "duration_ms": 17,
                    "request_id": "request-1",
                    "data_field_count": 3,
                    "data_keys": ["exit_code", "access_token", "content"],
                    "health_failure": True,
                    "stdout": secret,
                    "Error_Summary": secret,
                    "tool_result": LeakingObject(),
                },
                "content": secret,
            },
            project_id="project-1",
            session_id="session-1",
            run_id="run-1",
        )
    )

    record = _read_jsonl(tmp_path / "logs")
    assert record["payload"] == {
        "request": {
            "tool": "shell",
            "action": "run",
            "capability": "shell.run",
            "request_id": "request-1",
            "argument_count": 4,
        },
        "result": {
            "success": False,
            "duration_ms": 17,
            "request_id": "request-1",
            "data_field_count": 3,
            "health_failure": True,
        },
    }
    assert secret not in json.dumps(record, ensure_ascii=False)


def test_audit_projection_drops_unknown_events_and_sanitizer_handles_nested_keys() -> None:
    secret = "must-not-survive"
    projection = audit_event_projection(
        Event(
            "plugin.untrusted.event",
            {
                "safe_looking": {
                    "ReAsOn-InG_Content": secret,
                    "AUTHORIZATION": secret,
                    "toolOutput": secret,
                },
                "object": LeakingObject(),
            },
        )
    )
    sanitized = sanitize_for_log(
        {
            "outer": {
                "ReAsOn-InG_Content": secret,
                "AUTHORIZATION": secret,
                "toolOutput": secret,
                "unknown": LeakingObject(),
            }
        }
    )

    assert projection["payload"] == {}
    assert sanitized == {
        "outer": {
            "ReAsOn-InG_Content": "[redacted]",
            "AUTHORIZATION": "[redacted]",
            "toolOutput": "[redacted]",
            "unknown": "[unsupported]",
        }
    }


def test_audit_refuses_a_symlink_destination(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    logger = JsonlEventLogger(log_dir)
    outside = tmp_path / "outside.jsonl"
    outside.write_text("unchanged", encoding="utf-8")
    (log_dir / "events-2026-07-14.jsonl").symlink_to(outside)

    with pytest.raises(OSError, match="regular file|symbolic link"):
        logger(Event("task.started", {}, timestamp="2026-07-14T00:00:00+00:00"))

    assert outside.read_text(encoding="utf-8") == "unchanged"


@pytest.mark.parametrize(
    "initial",
    [
        [],
        "not-an-object",
        None,
        {"counts": {"task.finished": "4"}, "total_tool_duration_ms": "9", "failed_tools": True},
        {"counts": {"task.finished": -4}, "total_tool_duration_ms": -9, "failed_tools": -2},
        {
            "counts": {"task.finished": 10**40, "unknown.event": 10**40},
            "total_tool_duration_ms": 10**40,
            "failed_tools": 10**40,
        },
    ],
)
def test_metrics_load_rejects_or_clamps_untrusted_json(tmp_path: Path, initial: object) -> None:
    path = tmp_path / "metrics" / "project.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(initial), encoding="utf-8")

    metrics = EventMetricsCollector(path)

    assert set(metrics.counts) <= metrics.ALLOWED_EVENTS
    assert all(0 <= value <= metrics.MAX_COUNTER for value in metrics.counts.values())
    assert 0 <= metrics.total_tool_duration_ms <= metrics.MAX_TOOL_DURATION_MS
    assert 0 <= metrics.failed_tools <= metrics.MAX_COUNTER


def test_metrics_malformed_file_and_event_values_are_bounded_and_private(tmp_path: Path) -> None:
    path = tmp_path / "metrics" / "project.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not-json", encoding="utf-8")
    metrics = EventMetricsCollector(path)
    secret = "metrics-never-write-this"

    metrics(
        Event(
            "tool.finished",
            {
                "result": {
                    "success": False,
                    "duration_ms": 10**40,
                    "stdout": secret,
                    "content": secret,
                },
                "prompt": secret,
            },
        )
    )
    metrics(Event("tool.denied", {"result": {"success": "false", "duration_ms": "15"}}))

    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored == {
        "schema_version": 1,
        "updated_at": stored["updated_at"],
        "counts": {"tool.denied": 1, "tool.finished": 1},
        "total_tool_duration_ms": metrics.MAX_EVENT_DURATION_MS,
        "failed_tools": 1,
    }
    assert secret not in path.read_text(encoding="utf-8")
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert not list(path.parent.glob(".*.tmp"))


def test_metrics_saturate_existing_totals_and_ignore_unlisted_events(tmp_path: Path) -> None:
    path = tmp_path / "metrics" / "project.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "counts": {"tool.finished": EventMetricsCollector.MAX_COUNTER},
                "total_tool_duration_ms": EventMetricsCollector.MAX_TOOL_DURATION_MS,
                "failed_tools": EventMetricsCollector.MAX_COUNTER,
            }
        ),
        encoding="utf-8",
    )
    metrics = EventMetricsCollector(path)

    metrics(Event("tool.finished", {"result": {"success": False, "duration_ms": 1}}))
    before = path.read_text(encoding="utf-8")
    metrics(Event("unlisted", {"prompt": "private"}))

    assert metrics.counts["tool.finished"] == metrics.MAX_COUNTER
    assert metrics.total_tool_duration_ms == metrics.MAX_TOOL_DURATION_MS
    assert metrics.failed_tools == metrics.MAX_COUNTER
    assert path.read_text(encoding="utf-8") == before


def test_metrics_ignore_internal_events_with_live_objects(tmp_path: Path) -> None:
    path = tmp_path / "metrics" / "project.json"
    metrics = EventMetricsCollector(path)

    metrics(
        Event(
            "session.checkpoint.requested",
            {"state": LeakingObject(), "messages": [LeakingObject()]},
        )
    )

    assert not path.exists()
    assert metrics.counts == {}


def test_metrics_refuse_a_symlink_destination(tmp_path: Path) -> None:
    path = tmp_path / "metrics" / "project.json"
    path.parent.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text("unchanged", encoding="utf-8")
    path.symlink_to(outside)
    metrics = EventMetricsCollector(path)

    with pytest.raises(OSError, match="regular file"):
        metrics(Event("task.started", {}))

    assert outside.read_text(encoding="utf-8") == "unchanged"


def test_metrics_do_not_parse_oversized_existing_files(tmp_path: Path) -> None:
    path = tmp_path / "metrics" / "project.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b" " * (EventMetricsCollector.MAX_FILE_BYTES + 1))

    metrics = EventMetricsCollector(path)

    assert metrics.counts == {}
    assert metrics.total_tool_duration_ms == 0
    assert metrics.failed_tools == 0


def test_runtime_pipeline_project_identifier_cannot_escape_data_directories(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    agent_dir = root / ".project-agent"
    agent_dir.mkdir()
    project = Project(
        id="../../escape/项目",
        name="malicious-metadata",
        root=root,
        agent_dir=agent_dir,
        config_path=agent_dir / "project.yaml",
        context_path=agent_dir / "context.md",
        language="Python",
    )
    config = make_config()
    memory = MemoryStore(config)
    tools = ToolManager(config, project, memory, yolo=True)
    pipelines = RuntimeEventPipelines(
        config=config,
        project=project,
        sessions=SessionManager(project),
        memory=memory,
        health=tools.health,
        events=EventBus(),
    )

    assert pipelines.metrics is not None
    assert pipelines.metrics.path.parent == config.data_dir / "metrics"
    assert pipelines.metrics.path.name.endswith(".json")
    assert "/" not in pipelines.metrics.path.name
    assert tools.health.path.parent == config.data_dir / "capability-health"
    assert "/" not in tools.health.path.name
    daemon = ProjectDaemon(config, project, memory)
    parallel = ParallelWorktreeRunner(project, config.data_dir)
    assert daemon.base_dir.parent == config.data_dir / "daemon"
    assert parallel.base_dir.parent == config.data_dir / "worktrees"
    assert not (tmp_path / "escape").exists()
