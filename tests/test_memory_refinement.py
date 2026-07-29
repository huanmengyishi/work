from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from agent.deepseek import ChatResponse
from agent.events import EventBus
from agent.memory import MemoryStore
from agent.memory_refinement import sanitize_memory_tags
from agent.model_router import ModelRoute
from agent.optimizer import PerformanceHistory
from agent.project import ProjectManager
from agent.runtime import AgentRuntime
from agent.session import SessionManager
from agent.state import AgentState
from agent.tools import ToolManager


def _valid_refinement(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "kind": "Lesson",
        "title": "Reuse bounded evidence",
        "lesson": "Use the smallest sufficient managed evidence set.",
        "why": "Bounded evidence makes the conclusion reproducible.",
        "when_to_apply": "Apply this after a completed tool-backed task.",
        "evidence_summary": "The current turn completed five managed tool calls.",
        "reflection": "The successful path should be reused without repeating discovery.",
        "tags": ["bounded", "tool-evidence"],
        "confidence": 0.88,
    }
    value.update(overrides)
    return value


@pytest.mark.parametrize(
    "keyword",
    ["password", "cookie", "api_key", "access_token", "refresh_token", "secret"],
)
@pytest.mark.parametrize("separator", [":", "=", "_", "-", "."])
def test_credential_shaped_tags_are_rejected_case_insensitively(keyword: str, separator: str) -> None:
    tag = f"{keyword.upper()}{separator}sEcReT123456"

    assert sanitize_memory_tags([tag]) == ()


@pytest.mark.parametrize("tag", ["PASSWORD=abc", "API_KEY:foo", "ToKeN=short"])
def test_explicit_credential_assignment_tags_reject_short_values(tag: str) -> None:
    assert sanitize_memory_tags([tag]) == ()


def test_normal_engineering_tags_with_credential_terms_remain_searchable() -> None:
    tags = [
        "token-budget",
        "secret-scanning",
        "password-policy",
        "cookie-handling",
        "api-key-rotation",
        "access-token-refresh",
    ]

    assert sanitize_memory_tags(tags) == tuple(tags)


def _chat_response(
    *,
    content: str | None = None,
    finish_reason: str | None = "stop",
    tool_calls: list[dict[str, Any]] | None = None,
) -> ChatResponse:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": json.dumps(_valid_refinement(), ensure_ascii=False) if content is None else content,
    }
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return ChatResponse(
        message=message,
        raw={},
        finish_reason=finish_reason,
        usage={"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        http_attempt_count=1,
    )


class _RecordingClient:
    def __init__(self, outcome: ChatResponse | BaseException | None = None) -> None:
        self.outcome = outcome or _chat_response()
        self.requests: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> ChatResponse:
        self.requests.append(kwargs)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def _model_route() -> ModelRoute:
    return ModelRoute(
        provider="deepseek",
        tier="fast",
        model="deepseek-refinement-test",
        thinking_enabled=False,
        reasoning_effort=None,
        max_tokens=4096,
        cost_class="low",
        reasons=("memory-refinement-test",),
    )


def _append_tool_calls(
    state: AgentState,
    count: int,
    *,
    turn: int | None = None,
    failed_index: int | None = None,
    args: dict[str, Any] | None = None,
    stdout: str = "ok",
    stderr: str = "",
) -> None:
    target_turn = state.turn if turn is None else turn
    for index in range(count):
        failed = index == failed_index
        state.tool_calls.append(
            {
                "turn": target_turn,
                "round": index + 1,
                "request": {
                    "tool": "shell",
                    "action": "run",
                    "args": dict(args or {"command": f"printf check-{index}"}),
                    "request_id": f"call-{target_turn}-{index}",
                },
                "result": {
                    "success": not failed,
                    "stdout": stdout,
                    "stderr": stderr if failed else "",
                    "duration_ms": index + 1,
                    "request_id": f"call-{target_turn}-{index}",
                    "data": {"private_body": stdout},
                },
            }
        )


def _runtime_and_completed_state(
    tmp_path: Path,
    make_config,
    *,
    client: _RecordingClient | None = None,
    tool_call_count: int = 5,
    overrides: dict[str, Any] | None = None,
) -> tuple[AgentRuntime, AgentState, list[dict[str, Any]], _RecordingClient, ModelRoute]:
    config_overrides: dict[str, Any] = {
        "memory": {
            "smart_reflection": True,
            "smart_reflection_min_tool_calls": 5,
        },
        "events": {
            "jsonl_log": False,
            "metrics_enabled": False,
            "performance_history_enabled": False,
        },
    }
    if overrides:
        from agent.config import deep_merge

        config_overrides = deep_merge(config_overrides, overrides)
    config = make_config(config_overrides)
    root = tmp_path / "project"
    root.mkdir()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    selected_client = client or _RecordingClient()
    runtime = AgentRuntime.with_default_services(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=selected_client,
    )
    state = AgentState.create(
        session_id=runtime.sessions.new_session_id(),
        project=project,
        user_request="Summarize the completed managed checks.",
        loaded_memories=[],
        loaded_tools=[],
        git_branch=None,
        context_index_path=str(project.agent_dir / "index.json"),
    )
    route = _model_route()
    state.model_route = route.to_dict()
    state.start()
    _append_tool_calls(state, tool_call_count)
    state.complete("All requested checks completed successfully.")
    runtime.execution_budget.bind(state)
    messages: list[dict[str, Any]] = [{"role": "assistant", "content": state.final_answer}]
    return runtime, state, messages, selected_client, route


@pytest.mark.parametrize(
    ("tool_call_count", "expected_request_count"),
    [(0, 0), (4, 0), (5, 1), (6, 1)],
)
def test_current_turn_tool_call_threshold_is_exact(
    tmp_path: Path,
    make_config,
    tool_call_count: int,
    expected_request_count: int,
) -> None:
    runtime, state, messages, client, route = _runtime_and_completed_state(
        tmp_path,
        make_config,
        tool_call_count=tool_call_count,
    )

    refinement = runtime._maybe_refine_memory(
        state,
        messages,
        final=state.final_answer,
        model_route=route,
    )

    assert len(client.requests) == expected_request_count
    assert state.status == "completed"
    assert state.memory_refinement_model_request_count == expected_request_count
    assert state.model_request_count == expected_request_count
    assert (refinement is not None) is bool(expected_request_count)
    metadata = state.convergence["memory_refinement"]
    assert metadata["tool_call_count"] == tool_call_count
    assert metadata["logical_requests"] == expected_request_count
    if tool_call_count < 5:
        assert metadata["reason"] == "below_tool_call_threshold"


@pytest.mark.parametrize("tool_call_count", [1, 2, 3, 4])
def test_config_cannot_lower_hard_five_tool_call_floor(
    tmp_path: Path,
    make_config,
    tool_call_count: int,
) -> None:
    runtime, state, messages, client, route = _runtime_and_completed_state(
        tmp_path,
        make_config,
        tool_call_count=tool_call_count,
        overrides={"memory": {"smart_reflection_min_tool_calls": 1}},
    )

    refinement = runtime._maybe_refine_memory(
        state,
        messages,
        final=state.final_answer,
        model_route=route,
    )

    assert refinement is None
    assert client.requests == []
    assert state.status == "completed"
    assert runtime.memory_refiner.min_tool_calls == 5
    assert state.convergence["memory_refinement"]["reason"] == "below_tool_call_threshold"


def test_same_completed_turn_never_makes_a_second_refinement_request(tmp_path: Path, make_config) -> None:
    runtime, state, messages, client, route = _runtime_and_completed_state(tmp_path, make_config)

    first = runtime._maybe_refine_memory(state, messages, final=state.final_answer, model_route=route)
    second = runtime._maybe_refine_memory(state, messages, final=state.final_answer, model_route=route)

    assert first is not None
    assert second is None
    assert len(client.requests) == 1
    assert state.status == "completed"
    assert state.memory_refinement_model_request_count == 1
    assert state.convergence["memory_refinement"]["logical_requests"] == 1
    assert state.convergence["memory_refinement"]["reason"] == "already_requested"


def test_old_turn_tool_calls_do_not_accumulate_toward_threshold(tmp_path: Path, make_config) -> None:
    runtime, state, messages, client, route = _runtime_and_completed_state(
        tmp_path,
        make_config,
        tool_call_count=0,
    )
    state.status = "running"
    state.final_answer = ""
    _append_tool_calls(state, 6, turn=1)
    state.resume("Continue with a small second-turn check.")
    _append_tool_calls(state, 4, turn=2)
    state.complete("The second turn completed.")
    runtime.execution_budget.bind(state)

    refinement = runtime._maybe_refine_memory(
        state,
        messages,
        final=state.final_answer,
        model_route=route,
    )

    assert refinement is None
    assert client.requests == []
    assert state.status == "completed"
    assert state.convergence["memory_refinement"]["tool_call_count"] == 4
    assert state.convergence["memory_refinement"]["reason"] == "below_tool_call_threshold"


def test_disabled_smart_reflection_never_requests_model(tmp_path: Path, make_config) -> None:
    runtime, state, messages, client, route = _runtime_and_completed_state(
        tmp_path,
        make_config,
        tool_call_count=6,
        overrides={"memory": {"smart_reflection": False}},
    )

    assert (
        runtime._maybe_refine_memory(
            state,
            messages,
            final=state.final_answer,
            model_route=route,
        )
        is None
    )
    assert client.requests == []
    assert state.status == "completed"
    assert state.convergence["memory_refinement"]["reason"] == "disabled"


def test_refinement_uses_injected_client_and_one_small_tool_free_request(tmp_path: Path, make_config) -> None:
    client = _RecordingClient()
    runtime, state, messages, selected_client, route = _runtime_and_completed_state(
        tmp_path,
        make_config,
        client=client,
        overrides={"memory": {"smart_reflection_max_output_tokens": 192}},
    )

    refinement = runtime._maybe_refine_memory(
        state,
        messages,
        final=state.final_answer,
        model_route=route,
    )

    assert runtime.client is client is selected_client
    assert refinement is not None
    assert len(client.requests) == 1
    request = client.requests[0]
    assert request["tools"] is None
    assert request["tool_choice"] is None
    assert request["thinking"] is False
    assert request["reasoning_effort"] is None
    assert request["max_tokens"] == 192 < 4096
    assert request["model"] == route.model
    assert len(request["messages"]) == 2
    assert state.memory_refinement_model_request_count == 1
    assert state.model_metrics == {
        "http_attempt_count": 1,
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "total_tokens": 30,
    }


def test_refinement_skips_nonfatally_when_request_budget_is_exhausted(tmp_path: Path, make_config) -> None:
    runtime, state, messages, client, route = _runtime_and_completed_state(
        tmp_path,
        make_config,
        overrides={"runtime": {"budget": {"max_model_requests_per_turn": 1}}},
    )
    state.record_model_request("main_loop")
    runtime.execution_budget.bind(state)

    refinement = runtime._maybe_refine_memory(
        state,
        messages,
        final=state.final_answer,
        model_route=route,
    )

    assert refinement is None
    assert client.requests == []
    assert state.status == "completed"
    assert state.model_request_count == 1
    assert state.main_loop_model_request_count == 1
    assert state.memory_refinement_model_request_count == 0
    assert state.convergence["memory_refinement"]["reason"] == "budget_model_requests"


def test_refinement_budget_exception_is_isolated_from_completed_task(
    tmp_path: Path,
    make_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, state, messages, client, route = _runtime_and_completed_state(tmp_path, make_config)

    def fail_budget(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("optional budget unavailable")

    monkeypatch.setattr(runtime.execution_budget, "try_before_model_request", fail_budget)

    refinement = runtime._maybe_refine_memory(state, messages, final=state.final_answer, model_route=route)

    assert refinement is None
    assert client.requests == []
    assert state.status == "completed"
    assert state.final_answer == "All requested checks completed successfully."
    assert state.memory_refinement_model_request_count == 0
    assert state.convergence["memory_refinement"]["status"] == "failed"
    assert state.convergence["memory_refinement"]["reason"].startswith("budget_")


def test_refinement_checkpoint_failure_is_isolated_and_not_retried(
    tmp_path: Path,
    make_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, state, messages, client, route = _runtime_and_completed_state(tmp_path, make_config)

    def fail_checkpoint(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("optional checkpoint unavailable")

    monkeypatch.setattr(runtime, "_checkpoint_session", fail_checkpoint)

    first = runtime._maybe_refine_memory(state, messages, final=state.final_answer, model_route=route)
    first_metadata = dict(state.convergence["memory_refinement"])
    second = runtime._maybe_refine_memory(state, messages, final=state.final_answer, model_route=route)

    assert first is None
    assert second is None
    assert client.requests == []
    assert state.status == "completed"
    assert state.final_answer == "All requested checks completed successfully."
    assert state.error == ""
    assert state.memory_refinement_model_request_count == 1
    assert first_metadata["status"] == "failed"
    assert first_metadata["reason"].startswith("checkpoint_")
    assert state.convergence["memory_refinement"]["reason"] == "already_requested"


@pytest.mark.parametrize("failed_event", ["model.requested", "model.responded"])
def test_refinement_event_failure_is_best_effort(
    tmp_path: Path,
    make_config,
    monkeypatch: pytest.MonkeyPatch,
    failed_event: str,
) -> None:
    runtime, state, messages, client, route = _runtime_and_completed_state(tmp_path, make_config)
    publish = runtime.events.publish

    def fail_selected_event(event_name: str, *args: Any, **kwargs: Any):
        if event_name == failed_event:
            raise RuntimeError("optional event unavailable")
        return publish(event_name, *args, **kwargs)

    monkeypatch.setattr(runtime.events, "publish", fail_selected_event)

    refinement = runtime._maybe_refine_memory(state, messages, final=state.final_answer, model_route=route)

    assert refinement is not None
    assert len(client.requests) == 1
    assert state.status == "completed"
    assert state.final_answer == "All requested checks completed successfully."
    assert state.memory_refinement_model_request_count == 1
    assert failed_event in state.convergence["memory_refinement"]["observability_errors"]


@pytest.mark.parametrize(
    ("outcome", "expected_status"),
    [
        (_chat_response(content=""), "rejected"),
        (_chat_response(content="{not-json"), "rejected"),
        (_chat_response(finish_reason="length"), "rejected"),
        (
            _chat_response(
                tool_calls=[
                    {
                        "id": "unexpected-tool",
                        "type": "function",
                        "function": {"name": "shell_run", "arguments": "{}"},
                    }
                ]
            ),
            "rejected",
        ),
        (_chat_response(content='<｜｜DSML｜｜tool_calls> {"kind":"Lesson"}'), "rejected"),
        (RuntimeError("optional refinement unavailable"), "failed"),
    ],
    ids=["empty", "bad-json", "length", "tool-calls", "dsml", "exception"],
)
def test_invalid_or_failed_refinement_never_changes_completed_outcome(
    tmp_path: Path,
    make_config,
    outcome: ChatResponse | BaseException,
    expected_status: str,
) -> None:
    client = _RecordingClient(outcome)
    runtime, state, messages, _, route = _runtime_and_completed_state(
        tmp_path,
        make_config,
        client=client,
    )

    refinement = runtime._maybe_refine_memory(
        state,
        messages,
        final=state.final_answer,
        model_route=route,
    )

    assert refinement is None
    assert len(client.requests) == 1
    assert state.status == "completed"
    assert state.final_answer == "All requested checks completed successfully."
    assert state.error == ""
    assert state.model_request_count == 1
    assert state.memory_refinement_model_request_count == 1
    assert state.convergence["memory_refinement"]["status"] == expected_status


def test_refinement_redacts_model_input_and_persisted_output(tmp_path: Path, make_config) -> None:
    input_secrets = {
        "api": "INPUT_API_SECRET_123456",
        "password": "INPUT_PASSWORD_123456",
        "bearer": "INPUT_BEARER_123456",
        "deepseek": "INPUT_DEEPSEEK_SECRET_123456",
        "stdout": "INPUT_PRIVATE_STDOUT_123456",
    }
    output_secrets = {
        "password": "OUTPUT_PASSWORD_123456",
        "token": "OUTPUT_TOKEN_123456",
        "key": "sk-OUTPUTKEY123456",
        "tag": "token-OUTPUT_TAG_SECRET_123456",
    }
    model_value = _valid_refinement(
        title=f"password={output_secrets['password']}",
        lesson=f"token={output_secrets['token']}",
        why=f"Never retain {output_secrets['key']}",
        reflection=f"authorization=Bearer {output_secrets['token']}",
        tags=["bounded", output_secrets["tag"], "x" * 65],
    )
    client = _RecordingClient(_chat_response(content=json.dumps(model_value, ensure_ascii=False)))
    runtime, state, messages, _, route = _runtime_and_completed_state(
        tmp_path,
        make_config,
        client=client,
        tool_call_count=0,
    )
    state.user_request = (
        f"api_key={input_secrets['api']} password={input_secrets['password']} "
        f"DEEPSEEK_API_KEY={input_secrets['deepseek']}"
    )
    state.final_answer = f"Authorization: Bearer {input_secrets['bearer']}"
    _append_tool_calls(
        state,
        5,
        failed_index=0,
        args={"api_key": input_secrets["api"], "command": "private"},
        stdout=input_secrets["stdout"],
        stderr=f"cookie={input_secrets['password']} Bearer {input_secrets['bearer']}",
    )

    refinement = runtime._maybe_refine_memory(
        state,
        messages,
        final=state.final_answer,
        model_route=route,
    )

    assert refinement is not None
    request_text = json.dumps(client.requests, ensure_ascii=False, default=str)
    for secret in input_secrets.values():
        assert secret not in request_text
    assert "args" not in str(client.requests[0]["messages"][-1]["content"])
    assert input_secrets["stdout"] not in request_text
    refinement_text = json.dumps(refinement, ensure_ascii=False)
    for secret in output_secrets.values():
        assert secret not in refinement_text
    assert refinement["tags"] == ["bounded"]

    # Re-introduce an untrusted tag at the terminal-event boundary to prove the
    # Memory pipeline's independent second validation pass also rejects it.
    refinement["tags"].append(output_secrets["tag"])
    runtime._finalize_session(state, messages)
    runtime._publish_terminal(
        "task.finished",
        state,
        final=state.final_answer,
        memory_refinement=refinement,
    )
    stored = "\n".join(
        item.title + "\n" + item.content + "\n" + ",".join(item.tags)
        for item in runtime.memory.recent(runtime.project.id, 20)
    )
    for secret in [*input_secrets.values(), *output_secrets.values()]:
        assert secret not in stored
    assert "[redacted]" in stored or "[redacted-key]" in stored


def test_pipeline_persists_model_selected_kind_and_confidence(tmp_path: Path, make_config) -> None:
    response = _chat_response(
        content=json.dumps(
            _valid_refinement(kind="Decision", confidence=0.61, title="Keep one completion refinement"),
            ensure_ascii=False,
        )
    )
    runtime, state, messages, _client, route = _runtime_and_completed_state(
        tmp_path,
        make_config,
        client=_RecordingClient(response),
    )

    refinement = runtime._maybe_refine_memory(state, messages, final=state.final_answer, model_route=route)
    assert refinement is not None
    runtime._publish_terminal(
        "task.finished",
        state,
        final=state.final_answer,
        memory_refinement=refinement,
    )

    decisions = [item for item in runtime.memory.recent(runtime.project.id, 20) if item.kind == "Decision"]
    assert len(decisions) == 1
    assert decisions[0].title == "Decision: Keep one completion refinement"
    assert decisions[0].confidence == pytest.approx(0.61)


def test_schema_7_loads_refinement_count_as_zero_and_resume_promotes_and_resets(
    tmp_path: Path,
    make_config,
) -> None:
    runtime, state, _messages, _client, _route = _runtime_and_completed_state(
        tmp_path,
        make_config,
        tool_call_count=0,
    )
    state.status = "running"
    state.final_answer = ""
    state.record_model_request("main_loop")
    legacy = state.to_dict()
    legacy["schema_version"] = 7
    legacy.pop("memory_refinement_model_request_count")

    restored = AgentState.from_dict(legacy)

    assert restored.schema_version == 7
    assert restored.model_request_count == 1
    assert restored.memory_refinement_model_request_count == 0
    restored.resume("Continue the legacy session.")
    assert restored.schema_version == 8
    assert restored.model_request_count == 0
    assert restored.main_loop_model_request_count == 0
    assert restored.memory_refinement_model_request_count == 0

    restored.record_model_request("memory_refinement")
    assert restored.memory_refinement_model_request_count == 1
    restored.resume("Start another turn.")
    assert restored.model_request_count == 0
    assert restored.memory_refinement_model_request_count == 0
    runtime.close()


def test_performance_history_migrates_v1_database_with_zero_refinement_count(tmp_path: Path) -> None:
    path = tmp_path / "performance" / "legacy.db"
    path.parent.mkdir()
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            create table task_performance (
                sequence integer primary key,
                run_id text not null unique,
                project_id text not null,
                task_type text not null,
                task_mode text not null,
                outcome text not null check(outcome in ('completed', 'failed')),
                model_requests_total integer not null,
                model_requests_main_loop integer not null,
                model_requests_context_compaction integer not null,
                model_requests_final_synthesis integer not null,
                prompt_tokens integer not null,
                completion_tokens integer not null,
                total_tokens integer not null,
                tool_calls integer not null,
                tool_failures integer not null,
                plan_steps_total integer not null,
                plan_steps_completed integer not null,
                elapsed_seconds real not null,
                recorded_at text not null,
                schema_version integer not null
            )
            """
        )
        connection.execute(
            """
            insert into task_performance (
                run_id, project_id, task_type, task_mode, outcome,
                model_requests_total, model_requests_main_loop,
                model_requests_context_compaction, model_requests_final_synthesis,
                prompt_tokens, completion_tokens, total_tokens, tool_calls,
                tool_failures, plan_steps_total, plan_steps_completed,
                elapsed_seconds, recorded_at, schema_version
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-run",
                "legacy-project",
                "review",
                "standard",
                "completed",
                3,
                2,
                0,
                1,
                100,
                20,
                120,
                5,
                0,
                2,
                2,
                4.5,
                "2026-07-27T00:00:00+00:00",
                1,
            ),
        )

    history = PerformanceHistory(path)
    records = history.recent(limit=10)

    assert len(records) == 1
    assert records[0].run_id == "legacy-run"
    assert records[0].model_requests_memory_refinement == 0
    assert records[0].schema_version == 1
    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("pragma table_info(task_performance)")}
    assert "model_requests_memory_refinement" in columns


class _OrderingSessions(SessionManager):
    def __init__(self, project, order: list[str]) -> None:
        super().__init__(project)
        self.order = order

    def checkpoint(self, state, messages):
        self.order.append("session.checkpoint")
        return super().checkpoint(state, messages)

    def finalize(self, state, messages):
        self.order.append("session.finalize")
        return super().finalize(state, messages)


def _tool_call_response(index: int) -> ChatResponse:
    return ChatResponse(
        message={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"managed-check-{index}",
                    "type": "function",
                    "function": {
                        "name": "shell_run",
                        "arguments": json.dumps({"command": f"printf check-{index}"}),
                    },
                }
            ],
        },
        raw={},
        finish_reason="tool_calls",
    )


class _EndToEndClient:
    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.responses = [
            *[_tool_call_response(index) for index in range(5)],
            ChatResponse(
                message={"role": "assistant", "content": "Five managed checks completed."},
                raw={},
                finish_reason="stop",
            ),
            _chat_response(),
        ]
        self.requests: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> ChatResponse:
        self.requests.append(kwargs)
        is_refinement = kwargs.get("tools") is None and any(
            "durable project memory" in str(item.get("content") or "") for item in kwargs.get("messages", [])
        )
        self.order.append("model.refinement" if is_refinement else "model.main")
        if not self.responses:
            raise AssertionError("unexpected extra model request")
        return self.responses.pop(0)


def test_runtime_refines_before_finalize_and_terminal_memory_pipeline_runs_after_finalize(
    tmp_path: Path,
    make_config,
) -> None:
    config = make_config(
        {
            "memory": {"smart_reflection": True, "smart_reflection_min_tool_calls": 5},
            "runtime": {"task_mode": "standard"},
            "events": {
                "jsonl_log": False,
                "metrics_enabled": False,
                "performance_history_enabled": False,
            },
        }
    )
    root = tmp_path / "e2e-project"
    root.mkdir()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    events = EventBus()
    order: list[str] = []
    sessions = _OrderingSessions(project, order)
    client = _EndToEndClient(order)
    runtime = AgentRuntime.with_default_services(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
        events=events,
        sessions=sessions,
    )
    events.subscribe(
        "memory.reflection.persisted",
        lambda _event: order.append("memory.reflection.persisted"),
        name="test.memory-reflection-order",
    )
    events.subscribe(
        "task.finished",
        lambda _event: order.append("task.finished.observer"),
        name="test.terminal-order",
    )

    final = runtime.run("Run five managed read-only checks and summarize the result.")

    assert final == "Five managed checks completed."
    assert client.responses == []
    refinement_index = order.index("model.refinement")
    finalize_index = order.index("session.finalize")
    memory_index = order.index("memory.reflection.persisted")
    terminal_index = order.index("task.finished.observer")
    assert refinement_index < finalize_index < memory_index < terminal_index

    state = sessions.load(runtime.last_session_id).state
    assert state.status == "completed"
    assert len([item for item in state.tool_calls if item.get("turn") == state.turn]) == 5
    assert state.memory_refinement_model_request_count == 1
    assert state.convergence["memory_refinement"]["status"] == "accepted"
    assert [item.kind for item in memory.recent(project.id, 20)].count("Reflection") == 1
    assert len(client.requests) == 7
    assert client.requests[-1]["tools"] is None
    assert client.requests[-1]["thinking"] is False
