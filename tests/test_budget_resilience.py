from __future__ import annotations

from dataclasses import replace

import pytest

from agent.budget import ExecutionBudgetController, ExecutionBudgetExceeded
from agent.deepseek import ChatResponse
from agent.memory import MemoryStore
from agent.project import ProjectManager
from agent.resilience import ErrorCategory, ResiliencePolicy
from agent.runtime import AgentRuntime
from agent.state import AgentState, PlanStep
from agent.task_plan import TaskPlanFactory
from agent.task_router import TaskRouter
from agent.tools import ToolManager


def _state() -> AgentState:
    return AgentState(
        session_id="budget-state",
        project={"id": "project", "name": "project", "root": "/workspace"},
        objective="bounded task",
        user_request="bounded task",
        request_history=["bounded task"],
        working_directory="/workspace",
    )


def test_execution_budget_stops_before_an_unreserved_request_and_resets_on_resume(make_config) -> None:
    now = [10.0]
    config = make_config(
        {
            "runtime": {
                "budget": {
                    "max_model_requests_per_turn": 1,
                    "max_total_tokens_per_turn": 2_000,
                    "max_elapsed_seconds_per_turn": 10,
                }
            }
        }
    )
    budget = ExecutionBudgetController(config, clock=lambda: now[0])
    state = _state()
    budget.bind(state)
    budget.before_model_request(
        state,
        phase="main_loop",
        estimated_input_tokens=200,
        requested_output_tokens=100,
    )
    state.record_model_request("main_loop")

    with pytest.raises(ExecutionBudgetExceeded, match="model-request"):
        budget.before_model_request(
            state,
            phase="main_loop",
            estimated_input_tokens=1,
            requested_output_tokens=1,
        )
    assert state.convergence["execution_budget"]["stop_reason"] == "model-request limit reached"

    state.resume("continue")
    budget.before_model_request(
        state,
        phase="main_loop",
        estimated_input_tokens=200,
        requested_output_tokens=100,
    )
    assert state.turn == 2
    assert state.model_request_count == 0
    assert state.convergence["execution_budget"]["turn"] == 2


def test_execution_budget_checks_tokens_and_elapsed_time(make_config) -> None:
    now = [0.0]
    config = make_config(
        {
            "runtime": {
                "budget": {
                    "max_model_requests_per_turn": 8,
                    "max_total_tokens_per_turn": 1_024,
                    "max_elapsed_seconds_per_turn": 2,
                }
            }
        }
    )
    budget = ExecutionBudgetController(config, clock=lambda: now[0])
    state = _state()
    budget.bind(state)
    state.model_metrics = {"total_tokens": 900}
    with pytest.raises(ExecutionBudgetExceeded, match="token limit"):
        budget.before_model_request(
            state,
            phase="final_synthesis",
            estimated_input_tokens=100,
            requested_output_tokens=100,
        )

    state.model_metrics = {}
    budget.bind(state)
    now[0] = 3.0
    with pytest.raises(ExecutionBudgetExceeded, match="elapsed-time"):
        budget.before_tool_batch(state)


def test_runtime_budget_exhaustion_is_a_saved_resumable_failure(tmp_path, make_config, monkeypatch) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config(
        {
            "model": {"max_tokens": 256},
            "runtime": {
                "task_mode": "standard",
                "max_tool_rounds": 2,
                "max_tool_rounds_hard_limit": 3,
                "budget": {
                    "max_model_requests_per_turn": 1,
                    "max_total_tokens_per_turn": 100_000,
                    "max_elapsed_seconds_per_turn": 60,
                },
            },
        }
    )
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)

    class Client:
        calls = 0

        def chat(self, **_kwargs):
            self.calls += 1
            return ChatResponse(
                message={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "read-once",
                            "type": "function",
                            "function": {"name": "list_dir", "arguments": '{"path":".","depth":1}'},
                        }
                    ],
                },
                raw={},
                finish_reason="tool_calls",
                usage={"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
                http_attempt_count=1,
            )

    client = Client()
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )
    prompt = "Inspect this small project and summarize it."
    route = replace(runtime.task_router.route(prompt), require_plan=False, max_tool_rounds=2)
    monkeypatch.setattr(runtime.task_router, "route", lambda *_args, **_kwargs: route)

    answer = runtime.run(prompt)

    state = runtime.sessions.load(runtime.last_session_id).state
    assert client.calls == 1
    assert state.status == "failed"
    assert state.main_loop_model_request_count == 1
    assert "execution budget exhausted" in state.error
    assert f"agent resume --session {state.session_id}" in answer


def test_resilience_policy_is_bounded_and_classifies_without_runtime_retry(make_config) -> None:
    policy = ResiliencePolicy.from_config(
        make_config(
            {
                "runtime": {
                    "resilience": {
                        "max_corrective_rounds": 999,
                        "max_abnormal_finish_recoveries": -1,
                    }
                }
            }
        )
    )
    assert policy.max_corrective_rounds == 8
    assert policy.max_abnormal_finish_recoveries == 0
    assert policy.classify(TimeoutError("timed out")) is ErrorCategory.TIMEOUT
    assert policy.classify(RuntimeError("429 rate limit")) is ErrorCategory.RATE_LIMIT
    assert policy.classify(PermissionError("denied")) is ErrorCategory.PERMISSION


def test_plan_step_schema_7_round_trips_semantics_and_factory_uses_task_type(make_config) -> None:
    legacy = PlanStep.from_dict({"id": "verify", "title": "Verify"})
    assert legacy.step_type == "verify"
    typed = PlanStep.from_dict(
        {
            "id": "custom-check",
            "title": "Check",
            "step_type": "verify",
            "estimated_tool_rounds": 2,
            "artifact_ids": ["report.json"],
            "validation_rules": ["managed_validation"],
            "progress_weight": 2.5,
        }
    ).validate()
    assert typed.progress_weight == 2.5

    router = TaskRouter(make_config())
    factory = TaskPlanFactory()
    bug_plan = factory.build(replace(router.route("Fix this repository bug and run tests"), require_plan=True))
    review_plan = factory.build(replace(router.route("Review this repository architecture"), require_plan=True))
    assert next(item for item in bug_plan if item["id"] == "implement")["step_type"] == "implement"
    assert next(item for item in review_plan if item["id"] == "synthesize")["step_type"] == "synthesize"
    assert bug_plan[-1]["validation_rules"] == ["managed_validation"]
