from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from agent.events import Event, EventBus
from agent.optimizer import PerformanceAnalysisPipeline, PerformanceHistory, TaskPerformanceAnalyzer


class LeakingObject:
    def __str__(self) -> str:
        raise AssertionError("performance projection must not stringify arbitrary payload values")


def _terminal_event(
    run_id: str,
    *,
    event_name: str = "task.finished",
    secret: str = "private-content-must-not-survive",
) -> Event:
    return Event(
        event_name,
        {
            "prompt": secret,
            "final": secret,
            "error": secret,
            "reasoning": secret,
            "state": {
                "run_id": run_id,
                "user_request": secret,
                "final_answer": secret,
                "error": secret,
                "task_route": {"task_type": "bug_fix", "mode": "deep", "reasons": [secret]},
                "task_strategy": {"mode": "deep"},
                "model_request_count": 6,
                "main_loop_model_request_count": 3,
                "context_compaction_model_request_count": 1,
                "final_synthesis_model_request_count": 2,
                "model_metrics": {
                    "prompt_tokens": 120,
                    "completion_tokens": 30,
                    "total_tokens": 150,
                    "raw_response": secret,
                },
                "tool_calls": [
                    {
                        "request": {"args": {"api_key": secret}},
                        "result": {"success": True, "stdout": secret},
                    },
                    {
                        "request": {"args": LeakingObject()},
                        "result": {"success": False, "stderr": secret},
                    },
                    LeakingObject(),
                ],
                "plan": [
                    {"id": "inspect", "status": "completed", "description": secret},
                    {"id": "fix", "status": "in_progress", "completion_criteria": secret},
                ],
                "convergence": {"execution_budget": {"used": {"elapsed_seconds": 12.5}, "stop_reason": secret}},
                "project": {"id": "payload-project", "root": secret},
            },
        },
        project_id="project-1",
        session_id="session-1",
        run_id=run_id,
        timestamp="2026-07-26T12:00:00+00:00",
    )


def test_analyzer_projects_only_bounded_terminal_scalars() -> None:
    secret = "private-content-must-not-survive"
    performance = TaskPerformanceAnalyzer().analyze(_terminal_event("run-1", secret=secret))

    assert performance is not None
    assert performance.to_dict() == {
        "run_id": "run-1",
        "project_id": "project-1",
        "task_type": "bug_fix",
        "task_mode": "deep",
        "outcome": "completed",
        "model_requests_total": 6,
        "model_requests_main_loop": 3,
        "model_requests_context_compaction": 1,
        "model_requests_final_synthesis": 2,
        "model_requests_memory_refinement": 0,
        "prompt_tokens": 120,
        "completion_tokens": 30,
        "total_tokens": 150,
        "tool_calls": 3,
        "tool_failures": 1,
        "plan_steps_total": 2,
        "plan_steps_completed": 1,
        "elapsed_seconds": 12.5,
        "recorded_at": "2026-07-26T12:00:00+00:00",
        "exploration_rounds": 0,
        "schema_version": 3,
    }
    assert secret not in json.dumps(performance.to_dict(), ensure_ascii=False)


def test_analyzer_clamps_untrusted_counts_and_rejects_unlisted_labels() -> None:
    event = _terminal_event("run-2", event_name="task.failed")
    state = event.payload["state"]
    state["task_route"] = {"task_type": "credential-content", "mode": "auto"}
    state["model_request_count"] = 10**40
    state["main_loop_model_request_count"] = -5
    state["context_compaction_model_request_count"] = True
    state["final_synthesis_model_request_count"] = "7"
    state["model_metrics"] = {
        "prompt_tokens": 10**40,
        "completion_tokens": -2,
        "total_tokens": float("inf"),
    }
    state["convergence"] = {"execution_budget": {"used": {"elapsed_seconds": float("nan")}}}

    performance = TaskPerformanceAnalyzer().analyze(event)

    assert performance is not None
    assert performance.outcome == "failed"
    assert performance.task_type == "unknown"
    assert performance.task_mode == "unknown"
    assert performance.model_requests_total == TaskPerformanceAnalyzer.MAX_COUNT
    assert performance.model_requests_main_loop == 0
    assert performance.model_requests_context_compaction == 0
    assert performance.model_requests_final_synthesis == 0
    assert performance.prompt_tokens == 1_000_000_000
    assert performance.completion_tokens == 0
    assert performance.total_tokens == 0
    assert performance.elapsed_seconds == 0.0


def test_pipeline_history_is_idempotent_bounded_private_and_secure(tmp_path: Path) -> None:
    path = tmp_path / "performance" / "project.db"
    history = PerformanceHistory(path, max_records=2)
    events = EventBus()
    PerformanceAnalysisPipeline(history, events)

    events.publish(_terminal_event("run-1"))
    events.publish(_terminal_event("run-1"))
    events.publish(_terminal_event("run-2", event_name="task.failed"))
    events.publish(_terminal_event("run-3"))

    assert history.count() == 2
    assert [item.run_id for item in history.recent(limit=20)] == ["run-3", "run-2"]
    assert [item.outcome for item in history.recent(limit=20)] == ["completed", "failed"]
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert events.required_subscriber_count("task.finished") == 0

    database_bytes = path.read_bytes()
    assert b"private-content-must-not-survive" not in database_bytes
    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("pragma table_info(task_performance)")}
        assert not columns.intersection(
            {"prompt", "final", "error", "args", "stdout", "stderr", "reasoning", "credential", "secret"}
        )


def test_history_write_failure_is_best_effort_and_pipeline_stays_healthy(tmp_path: Path) -> None:
    outside = tmp_path / "outside.db"
    outside.write_bytes(b"unchanged")
    path = tmp_path / "performance.db"
    path.symlink_to(outside)
    history = PerformanceHistory(path)
    events = EventBus()
    PerformanceAnalysisPipeline(history, events)

    events.publish(_terminal_event("run-1"))

    assert history.count() == 0
    assert outside.read_bytes() == b"unchanged"
    assert events.last_errors == []


def test_history_refuses_an_oversized_existing_database(tmp_path: Path) -> None:
    path = tmp_path / "performance.db"
    path.write_bytes(b" " * (PerformanceHistory.MAX_DB_BYTES + 1))
    history = PerformanceHistory(path)
    performance = TaskPerformanceAnalyzer().analyze(_terminal_event("run-1"))

    assert performance is not None
    assert history.record(performance) is False
    assert history.recent() == []
    assert history.count() == 0
    assert path.stat().st_size == PerformanceHistory.MAX_DB_BYTES + 1


def test_analyzer_ignores_nonterminal_or_missing_state_events() -> None:
    analyzer = TaskPerformanceAnalyzer()

    assert analyzer.analyze(Event("task.started", {"state": {}})) is None
    assert analyzer.analyze(Event("task.finished", {"state": LeakingObject()})) is None
