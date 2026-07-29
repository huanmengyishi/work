from __future__ import annotations

from dataclasses import replace

import pytest

from agent.adaptive import StrategyAdjuster
from agent.model_router import ModelRouter
from agent.optimizer import PerformanceHistory, TaskPerformance
from agent.task_router import TaskRouter


def _performance(index: int, *, outcome: str, exploration_rounds: int) -> TaskPerformance:
    return TaskPerformance(
        run_id=f"run-{index}",
        project_id="project-1",
        task_type="bug_fix",
        task_mode="standard",
        outcome=outcome,
        model_requests_total=4,
        model_requests_main_loop=4,
        model_requests_context_compaction=0,
        model_requests_final_synthesis=0,
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        tool_calls=exploration_rounds,
        tool_failures=int(outcome == "failed"),
        plan_steps_total=4,
        plan_steps_completed=4 if outcome == "completed" else 2,
        elapsed_seconds=2.0,
        recorded_at=f"2026-07-27T00:00:{index:02d}+00:00",
        exploration_rounds=exploration_rounds,
    )


def test_strategy_adjuster_uses_task_type_p95_and_upgrades_low_success_history(tmp_path, make_config) -> None:
    config = make_config(
        {
            "optimizer": {
                "strategy_adjustment_enabled": True,
                "adaptive_convergence_enabled": True,
                "min_samples": 8,
            }
        }
    )
    history = PerformanceHistory(tmp_path / "performance.db")
    for index, rounds in enumerate((3, 4, 5, 6, 7, 8, 9, 12)):
        assert history.record(_performance(index, outcome="failed", exploration_rounds=rounds))
    route = replace(TaskRouter(config).route("Fix this bug"), mode="standard", risk="medium")
    model = ModelRouter(config).route(route, explicit_tier="standard")

    decision = StrategyAdjuster(config, history, project_id="project-1").adjust(
        route,
        model,
        configured_exploration_limit=6,
    )

    assert decision.applied is True
    assert decision.samples == 8
    assert decision.exploration_round_limit == 12
    assert decision.model_tier == "deep"
    assert "p95_exploration" in decision.reason
    assert "low_recent_success" in decision.reason


def test_strategy_adjustment_is_default_off_and_never_downgrades_high_risk(tmp_path, make_config) -> None:
    default_config = make_config()
    history = PerformanceHistory(tmp_path / "performance.db")
    route = replace(TaskRouter(default_config).route("Fix this bug"), mode="standard", risk="medium")
    deep = ModelRouter(default_config).route(route, explicit_tier="deep")
    disabled = StrategyAdjuster(default_config, history, project_id="project-1").adjust(
        route,
        deep,
        configured_exploration_limit=6,
    )
    assert disabled.enabled is False
    assert disabled.applied is False
    assert disabled.model_tier == "deep"

    enabled_config = make_config({"optimizer": {"strategy_adjustment_enabled": True, "min_samples": 3}})
    for index in range(3):
        assert history.record(_performance(index, outcome="completed", exploration_rounds=4))
    high_risk = replace(route, risk="high")
    decision = StrategyAdjuster(enabled_config, history, project_id="project-1").adjust(
        high_risk,
        deep,
        configured_exploration_limit=6,
    )
    assert decision.model_tier == "deep"


@pytest.mark.parametrize(
    ("strategy_enabled", "adaptive_enabled", "expected_limit", "expected_tier", "expected_reasons"),
    [
        (False, False, 6, "standard", {"disabled"}),
        (False, True, 12, "standard", {"p95_exploration"}),
        (True, False, 6, "deep", {"low_recent_success"}),
        (True, True, 12, "deep", {"p95_exploration", "low_recent_success"}),
    ],
)
def test_adaptive_convergence_and_model_strategy_have_independent_switches(
    tmp_path,
    make_config,
    strategy_enabled: bool,
    adaptive_enabled: bool,
    expected_limit: int,
    expected_tier: str,
    expected_reasons: set[str],
) -> None:
    config = make_config(
        {
            "optimizer": {
                "strategy_adjustment_enabled": strategy_enabled,
                "adaptive_convergence_enabled": adaptive_enabled,
                "min_samples": 8,
            }
        }
    )
    history = PerformanceHistory(tmp_path / "performance.db")
    for index, rounds in enumerate((3, 4, 5, 6, 7, 8, 9, 12)):
        assert history.record(_performance(index, outcome="failed", exploration_rounds=rounds))
    route = replace(TaskRouter(config).route("Fix this bug"), mode="standard", risk="medium")
    model = ModelRouter(config).route(route, explicit_tier="standard")

    decision = StrategyAdjuster(config, history, project_id="project-1").adjust(
        route,
        model,
        configured_exploration_limit=6,
    )

    assert decision.enabled is (strategy_enabled or adaptive_enabled)
    assert decision.applied is (strategy_enabled or adaptive_enabled)
    assert decision.exploration_round_limit == expected_limit
    assert decision.model_tier == expected_tier
    assert set(decision.reason.split(",")) == expected_reasons


def test_adaptive_convergence_p95_includes_zero_exploration_samples(tmp_path, make_config) -> None:
    config = make_config(
        {
            "optimizer": {
                "strategy_adjustment_enabled": False,
                "adaptive_convergence_enabled": True,
                "min_samples": 20,
            }
        }
    )
    history = PerformanceHistory(tmp_path / "performance.db")
    for index, rounds in enumerate((*([0] * 19), 24)):
        assert history.record(_performance(index, outcome="completed", exploration_rounds=rounds))
    route = replace(TaskRouter(config).route("Fix this bug"), mode="standard", risk="medium")
    model = ModelRouter(config).route(route, explicit_tier="standard")
    adjuster = StrategyAdjuster(config, history, project_id="project-1")

    profile = adjuster.profile(route.task_type)
    decision = adjuster.adjust(route, model, configured_exploration_limit=6)

    assert profile.samples == 20
    assert profile.exploration_samples == 20
    assert profile.p95_exploration_rounds == 0
    assert decision.applied is True
    assert decision.reason == "p95_exploration"
    assert decision.exploration_round_limit == 2
    assert decision.model_tier == "standard"


def test_adaptive_convergence_ignores_legacy_default_zero_until_new_samples_exist(tmp_path, make_config) -> None:
    config = make_config(
        {
            "optimizer": {
                "strategy_adjustment_enabled": False,
                "adaptive_convergence_enabled": True,
                "min_samples": 3,
            }
        }
    )
    history = PerformanceHistory(tmp_path / "performance.db")
    for index in range(8):
        legacy = replace(
            _performance(index, outcome="completed", exploration_rounds=0),
            schema_version=1,
        )
        assert history.record(legacy)
    route = replace(TaskRouter(config).route("Fix this bug"), mode="standard", risk="medium")
    model = ModelRouter(config).route(route, explicit_tier="standard")
    adjuster = StrategyAdjuster(config, history, project_id="project-1")

    legacy_profile = adjuster.profile(route.task_type)
    legacy_decision = adjuster.adjust(route, model, configured_exploration_limit=6)

    assert legacy_profile.samples == 8
    assert legacy_profile.exploration_samples == 0
    assert legacy_decision.samples == 0
    assert legacy_decision.applied is False
    assert legacy_decision.reason == "insufficient_samples"
    assert legacy_decision.exploration_round_limit == 6

    for index in range(8, 11):
        assert history.record(_performance(index, outcome="completed", exploration_rounds=0))

    migrated_profile = adjuster.profile(route.task_type)
    migrated_decision = adjuster.adjust(route, model, configured_exploration_limit=6)

    assert migrated_profile.samples == 11
    assert migrated_profile.exploration_samples == 3
    assert migrated_profile.p95_exploration_rounds == 0
    assert migrated_decision.samples == 3
    assert migrated_decision.applied is True
    assert migrated_decision.exploration_round_limit == 2


def test_model_strategy_can_use_legacy_success_history_without_adapting_from_unknown_rounds(
    tmp_path,
    make_config,
) -> None:
    config = make_config(
        {
            "optimizer": {
                "strategy_adjustment_enabled": True,
                "adaptive_convergence_enabled": True,
                "min_samples": 3,
            }
        }
    )
    history = PerformanceHistory(tmp_path / "performance.db")
    for index in range(3):
        legacy = replace(
            _performance(index, outcome="failed", exploration_rounds=0),
            schema_version=1,
        )
        assert history.record(legacy)
    route = replace(TaskRouter(config).route("Fix this bug"), mode="standard", risk="medium")
    model = ModelRouter(config).route(route, explicit_tier="standard")
    adjuster = StrategyAdjuster(config, history, project_id="project-1")

    profile = adjuster.profile(route.task_type)
    decision = adjuster.adjust(route, model, configured_exploration_limit=6)

    assert profile.samples == 3
    assert profile.exploration_samples == 0
    assert decision.applied is True
    assert decision.reason == "low_recent_success"
    assert decision.exploration_round_limit == 6
    assert decision.model_tier == "deep"


def test_enabled_adjustment_waits_for_minimum_samples(tmp_path, make_config) -> None:
    config = make_config(
        {
            "optimizer": {
                "strategy_adjustment_enabled": False,
                "adaptive_convergence_enabled": True,
                "min_samples": 8,
            }
        }
    )
    history = PerformanceHistory(tmp_path / "performance.db")
    for index in range(7):
        assert history.record(_performance(index, outcome="failed", exploration_rounds=12))
    route = replace(TaskRouter(config).route("Fix this bug"), mode="standard", risk="medium")
    model = ModelRouter(config).route(route, explicit_tier="standard")

    decision = StrategyAdjuster(config, history, project_id="project-1").adjust(
        route,
        model,
        configured_exploration_limit=6,
    )

    assert decision.enabled is True
    assert decision.applied is False
    assert decision.reason == "insufficient_samples"
    assert decision.exploration_round_limit == 6
    assert decision.model_tier == "standard"
