from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

from agent import config as config_module
from agent.config import DEFAULT_CONFIG, load_config, merge_yaml_defaults, read_yaml
from agent.model_router import ModelRoute, ModelRouter, more_capable_model_route
from agent.task_router import TaskRoute, TaskRouter, more_capable_task_route
from agent.task_plan import TaskPlanFactory
from agent.task_strategy import TaskStrategySelector


def test_task_router_returns_structured_deterministic_routes(make_config) -> None:
    router = TaskRouter(make_config())

    simple = router.route("什么是 Python？")
    bug = router.route("修复这个函数并运行测试")
    large = router.route("分析整个代码库的所有文件并总结")
    deep = router.route("全面审计整个仓库的所有安全问题并完成大规模重构")

    assert (simple.task_type, simple.scale, simple.risk, simple.mode) == (
        "question",
        "small",
        "low",
        "simple",
    )
    assert (bug.task_type, bug.scale, bug.risk, bug.mode) == (
        "bug_fix",
        "medium",
        "medium",
        "standard",
    )
    assert (large.task_type, large.scale, large.risk, large.mode) == (
        "review",
        "large",
        "low",
        "large",
    )
    assert deep.task_type == "refactor"
    assert deep.scale == "large"
    assert deep.risk == "high"
    assert deep.mode == "deep"
    assert deep.max_tool_rounds == 24
    assert router.route("什么是 Python？").to_dict() == simple.to_dict()


def test_task_router_uses_bounded_thresholds_and_hard_round_limit(make_config) -> None:
    router = TaskRouter(make_config({"runtime": {"max_tool_rounds_hard_limit": 12}}))

    below = router.route("x" * 599, source_file_count=499, file_count=1999)
    detailed = router.route("x" * 600)
    source_large = router.route("continue", source_file_count=500)
    files_large = router.route("continue", file_count=2000)

    assert below.score == 0
    assert detailed.score == 1
    assert "detailed-request" in detailed.reasons
    assert source_large.scale == "large"
    assert "large-codebase" in source_large.reasons
    assert files_large.scale == "large"
    assert "many-files" in files_large.reasons
    assert source_large.max_tool_rounds == 12


def test_task_router_tracks_failures_for_model_escalation(make_config) -> None:
    router = TaskRouter(make_config())

    once = router.route("继续", failure_count=1)
    repeated = router.route("继续", failure_count=2)

    assert once.failure_count == 1
    assert "prior-failure" in once.reasons
    assert repeated.failure_count == 2
    assert "repeated-failure" in repeated.reasons
    assert repeated.score > once.score
    assert ModelRouter(make_config()).route(repeated).tier == "deep"


def test_task_route_round_trip_and_legacy_strategy_promotion(make_config) -> None:
    route = TaskRouter(make_config()).route("修复错误并测试")
    assert TaskRoute.from_dict(route.to_dict()) == route

    legacy = TaskRoute.from_dict(
        {
            "mode": "deep",
            "score": 6,
            "reasons": ["legacy"],
            "max_tool_rounds": 24,
            "require_plan": True,
            "chunked_context": True,
        }
    )
    assert legacy.mode == "deep"
    assert legacy.scale == "large"
    assert legacy.task_type == "question"


def test_model_router_uses_safe_base_fallback_for_all_tiers(make_config) -> None:
    config = make_config({"model": {"model": "deepseek-user-base"}})
    task_router = TaskRouter(config)
    model_router = ModelRouter(config)

    fast = model_router.route(task_router.route("什么是 Python？"))
    standard = model_router.route(task_router.route("修复这个函数并测试"))
    deep = model_router.route(task_router.route("全面审计整个仓库并重构所有安全问题"))

    assert (fast.tier, fast.model, fast.thinking_enabled, fast.reasoning_effort) == (
        "fast",
        "deepseek-user-base",
        False,
        None,
    )
    assert (standard.tier, standard.model, standard.thinking_enabled, standard.reasoning_effort) == (
        "standard",
        "deepseek-user-base",
        True,
        "high",
    )
    assert (deep.tier, deep.model, deep.thinking_enabled, deep.reasoning_effort) == (
        "deep",
        "deepseek-user-base",
        True,
        "max",
    )
    assert all("base-model-fallback" in item.reasons for item in (fast, standard, deep))
    assert (fast.cost_class, standard.cost_class, deep.cost_class) == ("low", "balanced", "high")
    assert "simple-low-risk" in fast.reasons
    assert "cost-balanced" in standard.reasons
    assert "deep-capability-required" in deep.reasons


def test_model_router_cost_aware_policy_is_local_and_explainable(make_config) -> None:
    config = make_config()
    task_router = TaskRouter(config)
    model_router = ModelRouter(config)

    simple = model_router.route(task_router.route("什么是 Python？"))
    large_read_only = model_router.route(task_router.route("分析整个代码库的所有文件并总结"))
    repeated_failure = model_router.route(task_router.route("继续", failure_count=2))

    assert (simple.tier, simple.cost_class) == ("fast", "low")
    assert (large_read_only.tier, large_read_only.cost_class) == ("standard", "balanced")
    assert "large-low-risk" in large_read_only.reasons
    assert (repeated_failure.tier, repeated_failure.cost_class) == ("deep", "high")
    assert "repeated-failure" in repeated_failure.reasons
    assert model_router.route(task_router.route("什么是 Python？")).to_dict() == simple.to_dict()


def test_model_router_accepts_user_deepseek_tier_models(make_config) -> None:
    config = make_config(
        {
            "model": {
                "model": "deepseek-base",
                "routing": {
                    "fast_model": "deepseek-fast-configured",
                    "standard_model": "deepseek-standard-configured",
                    "deep_model": "deepseek-deep-configured",
                },
            }
        }
    )
    task = TaskRouter(config).route("普通问题")
    router = ModelRouter(config)

    assert router.route(task, explicit_tier="fast").model == "deepseek-fast-configured"
    assert router.route(task, explicit_tier="standard").model == "deepseek-standard-configured"
    assert router.route(task, explicit_tier="deep").model == "deepseek-deep-configured"
    assert "tier-model" in router.route(task, explicit_tier="fast").reasons


def test_disabling_model_routing_uses_base_model(make_config) -> None:
    config = make_config(
        {
            "model": {
                "model": "deepseek-base",
                "routing": {
                    "enabled": False,
                    "fast_model": "deepseek-fast-configured",
                },
            }
        }
    )
    task = TaskRouter(config).route("什么是 Python？")
    route = ModelRouter(config).route(task)

    assert route.tier == "fast"
    assert route.model == "deepseek-base"
    assert "routing-disabled" in route.reasons


def test_model_router_rejects_other_providers_and_invalid_tiers(make_config) -> None:
    with pytest.raises(ValueError, match="only the DeepSeek"):
        ModelRouter(make_config({"model": {"provider": "openai"}}))

    config = make_config()
    task = TaskRouter(config).route("普通问题")
    with pytest.raises(ValueError, match="model.routing.tier"):
        ModelRouter(config).route(task, explicit_tier="unknown")
    with pytest.raises(TypeError, match="requires a TaskRoute"):
        ModelRouter(config).route({"mode": "simple"})


def test_explicit_task_mode_remains_a_compatibility_override(make_config) -> None:
    config = make_config()
    task = TaskRouter(config).route(
        "全面审计整个仓库并重构所有安全问题",
        explicit_mode="simple",
    )
    model = ModelRouter(config).route(task)

    assert task.mode == "simple"
    assert "configured-mode" in task.reasons
    assert model.tier == "fast"
    assert model.thinking_enabled is False
    assert "configured-task-mode" in model.reasons


def test_explicit_model_tier_takes_priority_over_cost_aware_policy(make_config) -> None:
    config = make_config()
    task = TaskRouter(config).route("全面审计整个仓库并重构所有安全问题")
    route = ModelRouter(config).route(task, explicit_tier="fast")

    assert (route.tier, route.cost_class) == ("fast", "low")
    assert route.reasons[0] == "configured-tier"
    assert "deep-capability-required" not in route.reasons


def test_model_router_respects_non_adaptive_thinking(make_config) -> None:
    config = make_config(
        {
            "runtime": {"adaptive_thinking": False},
            "model": {"thinking": {"type": "disabled"}, "reasoning_effort": "custom"},
        }
    )
    task = TaskRouter(config).route("全面审计整个仓库并重构所有安全问题")
    route = ModelRouter(config).route(task)

    assert route.tier == "deep"
    assert route.thinking_enabled is False
    assert route.reasoning_effort == "custom"


def test_resume_helpers_do_not_downgrade_or_change_equal_tier_model(make_config) -> None:
    config = make_config({"model": {"model": "deepseek-base"}})
    task_router = TaskRouter(config)
    model_router = ModelRouter(config)
    simple_task = task_router.route("什么是 Python？")
    standard_task = task_router.route("修复这个函数并测试")
    generic_continue = task_router.route("继续")
    deep_task = task_router.route("全面审计整个仓库并重构所有安全问题")

    assert more_capable_task_route(deep_task, simple_task) is deep_task
    assert more_capable_task_route(simple_task, deep_task) is deep_task
    assert more_capable_task_route(standard_task, generic_continue) is standard_task
    assert more_capable_task_route(generic_continue, standard_task) is standard_task

    failed_continue = task_router.route("继续", failure_count=2)
    tied_previous = replace(standard_task, score=failed_continue.score)
    recovered_route = more_capable_task_route(tied_previous, failed_continue)
    assert recovered_route.task_type == standard_task.task_type
    assert recovered_route.failure_count == 2
    assert model_router.route(recovered_route).tier == "deep"

    architecture = task_router.route("请解释并设计系统架构")
    higher_scored_bug = replace(standard_task, score=architecture.score + 2)
    architecture_resume = more_capable_task_route(higher_scored_bug, architecture)
    assert architecture_resume.task_type == "architecture"
    assert model_router.route(architecture_resume).tier == "deep"

    standard = model_router.route(task_router.route("普通问题"), explicit_tier="standard")
    same_tier_new_model = replace(standard, model="deepseek-config-changed")
    deep = model_router.route(deep_task)
    assert more_capable_model_route(standard, same_tier_new_model) is standard
    assert more_capable_model_route(deep, standard) is deep
    assert more_capable_model_route(standard, deep) is deep


def test_model_route_round_trip_and_strategy_selector_compatibility(make_config) -> None:
    config = make_config()
    task = TaskRouter(config).route("全面审计整个代码库并重构所有安全问题")
    model = ModelRouter(config).route(task)
    assert ModelRoute.from_dict(model.to_dict()) == model

    with pytest.warns(DeprecationWarning, match="TaskStrategySelector is deprecated"):
        selector = TaskStrategySelector(config)
    strategy = selector.select("全面审计整个代码库并重构所有安全问题")
    assert strategy.mode == "deep"
    assert strategy.reasoning_effort == "max"
    assert strategy.max_tool_rounds == 24
    with pytest.warns(DeprecationWarning, match="re-routes the prompt"):
        assert [item["id"] for item in selector.initial_plan("修复整个仓库", strategy)] == [
            "scope",
            "inspect-chunks",
            "implement",
            "verify",
        ]


def test_model_route_promotes_legacy_cost_class_and_rejects_invalid_values(make_config) -> None:
    config = make_config()
    route = ModelRouter(config).route(TaskRouter(config).route("什么是 Python？"))
    legacy = route.to_dict()
    legacy.pop("cost_class")

    assert ModelRoute.from_dict(legacy).cost_class == "low"
    legacy["cost_class"] = "unlimited"
    with pytest.raises(ValueError, match="invalid cost class"):
        ModelRoute.from_dict(legacy)


def test_task_strategy_module_contains_no_classifier_rules() -> None:
    source = Path(__file__).parents[1].joinpath("agent", "task_strategy.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names
    }
    assigned_names = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name)
    }

    assert "re" not in imports
    assert not any(name.endswith("_MARKERS") for name in assigned_names)
    assert not any(
        isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == "score"
        for node in ast.walk(tree)
    )


def test_task_plan_factory_consumes_task_route_without_classifying(make_config) -> None:
    config = make_config()
    router = TaskRouter(config)
    factory = TaskPlanFactory()

    mutation = router.route("全面修复整个仓库的所有安全问题")
    architecture_implementation = router.route("为整个项目设计架构并实现迁移")
    read_only = router.route("全面审计整个仓库的所有安全问题")

    assert [item["id"] for item in factory.build(mutation)] == [
        "scope",
        "inspect-chunks",
        "implement",
        "verify",
    ]
    assert [item["id"] for item in factory.build(read_only)] == [
        "scope",
        "inspect-chunks",
        "synthesize",
        "verify",
    ]
    assert architecture_implementation.task_type == "architecture"
    assert "mutation-request" in architecture_implementation.reasons
    assert factory.build(architecture_implementation)[2]["id"] == "implement"
    assert factory.build(router.route("什么是 Python？")) == []
    with pytest.raises(TypeError, match="requires a TaskRoute"):
        factory.build({"mode": "deep"})


def test_model_routing_config_migration_is_add_only(tmp_path: Path) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(
        "model:\n  model: deepseek-user-model\n  routing:\n    fast_model: deepseek-user-fast\n",
        encoding="utf-8",
    )

    merge_yaml_defaults(path, {"model": DEFAULT_CONFIG["model"]})
    model = read_yaml(path)["model"]

    assert model["model"] == "deepseek-user-model"
    assert model["routing"]["fast_model"] == "deepseek-user-fast"
    assert "standard_model" in model["routing"]
    assert "deep_model" in model["routing"]


def test_load_config_generated_model_defaults_do_not_shadow_primary_overrides(tmp_path: Path, monkeypatch) -> None:
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    config_dir.mkdir()
    data_dir.mkdir()
    monkeypatch.setattr(config_module.paths, "config_dir", lambda: config_dir)
    monkeypatch.setattr(config_module.paths, "data_dir", lambda: data_dir)
    monkeypatch.setattr(config_module.paths, "ensure_base_dirs", lambda: None)
    (config_dir / "config.yaml").write_text(
        "model:\n  timeout_seconds: 777\n  routing:\n    fast_model: deepseek-config-fast\n",
        encoding="utf-8",
    )
    (config_dir / "model.yaml").write_text(
        "model:\n"
        "  timeout_seconds: 300\n"
        "  routing:\n"
        "    fast_model: null\n"
        "    standard_model: deepseek-model-standard\n",
        encoding="utf-8",
    )

    loaded = load_config()

    assert loaded.get("model.timeout_seconds") == 777
    assert loaded.get("model.routing.fast_model") == "deepseek-config-fast"
    assert loaded.get("model.routing.standard_model") == "deepseek-model-standard"
