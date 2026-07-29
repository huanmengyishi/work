from __future__ import annotations

from dataclasses import replace

import pytest

from agent.budget import ExecutionBudgetController, ExecutionBudgetExceeded
from agent.deepseek import ChatResponse
from agent.memory import MemoryStore
from agent.project import ProjectManager
from agent.resilience import CapabilityRecoveryController, ErrorCategory, ResiliencePolicy
from agent.runtime import AgentRuntime
from agent.state import AgentState, PlanStep
from agent.task_plan import TaskPlanFactory
from agent.task_router import TASK_TYPES, TaskRouter
from agent.tools import ToolManager
from agent.tools.base import ToolResult


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
    runtime = AgentRuntime.with_default_services(
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


def test_capability_recovery_applies_round_backoff_and_health_circuit(make_config) -> None:
    policy = ResiliencePolicy.from_config(
        make_config(
            {
                "runtime": {
                    "resilience": {
                        "capability_backoff_enabled": True,
                        "max_capability_backoff_rounds": 8,
                        "circuit_recovery_rounds": 4,
                    }
                }
            }
        )
    )
    controller = CapabilityRecoveryController(policy)
    convergence: dict[str, object] = {}

    first = controller.observe(
        convergence,
        "shell.run",
        current_round=1,
        success=False,
        health_failure=True,
        health_status="Available",
    )
    assert first.action == "backoff"
    assert first.blocked_through_round == 2
    assert controller.before_call(convergence, "shell.run", current_round=2).allowed is False
    assert controller.before_call(convergence, "shell.run", current_round=3).allowed is True

    opened = controller.observe(
        convergence,
        "shell.run",
        current_round=3,
        success=False,
        health_failure=True,
        health_status="Broken",
    )
    assert opened.action == "skip_broken"
    assert opened.blocked_through_round == 7
    assert controller.before_call(convergence, "shell.run", current_round=7).allowed is False
    assert controller.before_call(convergence, "shell.run", current_round=8).allowed is True

    reset = controller.observe(
        convergence,
        "shell.run",
        current_round=8,
        success=True,
        health_failure=False,
        health_status="Available",
    )
    assert reset.action == "reset"
    assert controller.before_call(convergence, "shell.run", current_round=8).allowed is True


def test_runtime_capability_circuit_prevents_immediate_repeat_execution(tmp_path, make_config, monkeypatch) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config(
        {
            "runtime": {
                "task_mode": "simple",
                "max_tool_rounds": 4,
                "max_tool_rounds_hard_limit": 4,
                "capability_failure_threshold": 1,
                "resilience": {
                    "max_corrective_rounds": 0,
                    "capability_backoff_enabled": True,
                    "circuit_recovery_rounds": 4,
                },
            }
        }
    )
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)

    class Client:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, **_kwargs):
            self.calls += 1
            if self.calls <= 2:
                request_id = f"repeat-{self.calls}"
                return ChatResponse(
                    message={
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": request_id,
                                "type": "function",
                                "function": {
                                    "name": "list_dir",
                                    "arguments": '{"path":".","depth":1}',
                                },
                            }
                        ],
                    },
                    raw={},
                    finish_reason="tool_calls",
                )
            return ChatResponse(
                message={"role": "assistant", "content": "The dependency remained unavailable."},
                raw={},
                finish_reason="stop",
            )

    client = Client()
    tools = ToolManager(config, project, memory, yolo=True)
    executions = 0

    def unavailable_list_dir(**_kwargs):
        nonlocal executions
        executions += 1
        return ToolResult(False, "", "dependency unavailable")

    tools.registry._handlers["template.list_dir"] = unavailable_list_dir
    runtime = AgentRuntime.with_default_services(
        config=config,
        project=project,
        memory=memory,
        tools=tools,
        client=client,
    )
    prompt = "Inspect the project once and report the bounded result."
    route = replace(runtime.task_router.route(prompt), require_plan=False, max_tool_rounds=4)
    monkeypatch.setattr(runtime.task_router, "route", lambda *_args, **_kwargs: route)

    runtime.run(prompt)

    state = runtime.sessions.load(runtime.last_session_id).state
    assert executions == 1
    assert client.calls == 3
    assert len(state.tool_calls) == 2
    assert state.tool_calls[0]["result"]["stderr"] == "dependency unavailable"
    assert state.tool_calls[1]["result"]["data"]["runtime_denied"] is True
    recovery = state.convergence["capability_recovery"]["template.list_dir"]
    assert recovery["circuit_open"] is True
    assert recovery["action"] == "skip_broken"


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


@pytest.mark.parametrize("task_type", sorted(TASK_TYPES))
def test_task_plan_factory_covers_every_registered_task_type(task_type, make_config) -> None:
    route = TaskRouter(make_config()).route("Review this repository and produce bounded evidence")
    reasons: tuple[str, ...] = ()
    artifact_hints: tuple[str, ...] = ()
    if task_type == "document_workflow":
        reasons = ("artifact-required", "word-artifact-required")
        artifact_hints = ("report.docx",)
    route = replace(
        route,
        task_type=task_type,
        require_plan=True,
        reasons=reasons,
        artifact_hints=artifact_hints,
    )

    plan = TaskPlanFactory().build(route)

    assert plan
    assert plan[0]["id"] == "scope"
    assert [step["status"] for step in plan].count("in_progress") == 1
    assert all(step["status"] in {"in_progress", "pending"} for step in plan)
    assert all(step["estimated_tool_rounds"] >= 1 for step in plan)
    assert all(step["progress_weight"] > 0 for step in plan)
    known_ids: set[str] = set()
    for step in plan:
        assert set(step.get("dependencies", [])) <= known_ids
        known_ids.add(step["id"])
    if task_type in {"bug_fix", "feature_development", "refactor"}:
        assert any(step["step_type"] == "implement" for step in plan)
    elif task_type == "document_workflow":
        assert any(step["step_type"] == "render" for step in plan)
        assert plan[-1]["artifact_ids"] == ["report.docx"]
        assert "document_parse" in plan[-1]["validation_rules"]
    else:
        assert any(step["step_type"] == "synthesize" for step in plan)


def test_task_plan_factory_routes_mutation_reason_to_change_strategy(make_config) -> None:
    route = replace(
        TaskRouter(make_config()).route("Explain this module"),
        task_type="code_explanation",
        require_plan=True,
        reasons=("mutation-request",),
    )

    plan = TaskPlanFactory().build(route)

    assert any(step["id"] == "implement" and step["step_type"] == "implement" for step in plan)
    assert plan[-1]["validation_rules"] == ["managed_validation"]
