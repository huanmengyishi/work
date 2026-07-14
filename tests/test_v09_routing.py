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
    assert "conditional-mutation" not in bug.reasons
    assert (large.task_type, large.scale, large.risk, large.mode) == (
        "review",
        "large",
        "low",
        "large",
    )

    document = router.route("总结当前目录的所有文档，新建word给我汇总内容")
    assert document.task_type == "document_workflow"
    assert document.scale == "large"
    assert document.mode == "large"
    assert document.require_plan is True
    assert "artifact-required" in document.reasons
    assert "word-artifact-required" in document.reasons

    no_artifact = router.route("阅读材料.txt，总结安全要求，不要新建文件，完成后直接回复")
    assert "artifact-required" not in no_artifact.reasons
    assert no_artifact.require_plan is False
    assert no_artifact.mode == "standard"

    fix_without_report = router.route("修复这个 bug，但不要创建报告文件")
    assert fix_without_report.task_type == "bug_fix"
    assert "mutation-request" in fix_without_report.reasons
    assert "artifact-required" not in fix_without_report.reasons
    conditional_fix = router.route(
        "忽略 node_modules、dist、构建产物和生成文件；分析整个项目。只修复证据确凿的 Bug，"
        "若没有充分证据就不要修改代码。"
    )
    assert conditional_fix.task_type == "bug_fix"
    assert "mutation-request" in conditional_fix.reasons
    assert "conditional-mutation" in conditional_fix.reasons
    assert "artifact-required" not in conditional_fix.reasons
    assert deep.task_type == "refactor"
    assert deep.scale == "large"
    assert deep.risk == "high"
    assert deep.mode == "deep"
    assert deep.max_tool_rounds == 24
    assert router.route("什么是 Python？").to_dict() == simple.to_dict()


def test_task_router_does_not_confuse_prohibited_credential_output_with_artifact_request(make_config) -> None:
    router = TaskRouter(make_config())
    acceptance_prompt = (
        "忽略 node_modules、dist、构建产物和生成文件；审查整个项目。"
        "若研究快照缺少内部源码或类型而产生大量基线错误，应如实记录为验证限制，"
        "不得为了‘全绿’而批量补文件或逐条打补丁；不得读取或输出任何真实凭据。"
        "只运行一次项目已有的静态检查或与目标代码相称的验证，不要重复等价命令。"
        "若没有充分证据，‘未找到可证实缺陷’是合格结论，应跳过 implement、完成 verify，不要修改代码；"
        "最终回复必须列出项目优点、Bug 证据、修改文件、验证结果和剩余风险。"
    )

    routed = router.route(acceptance_prompt)

    assert routed.task_type == "bug_fix"
    assert "mutation-request" in routed.reasons
    assert "conditional-mutation" in routed.reasons
    assert "single-validation" in routed.reasons
    assert "artifact-required" not in routed.reasons
    assert "word-artifact-required" not in routed.reasons
    assert "directory-artifact-required" not in routed.reasons


def test_task_router_keeps_explicit_artifact_when_a_separate_output_is_prohibited(make_config) -> None:
    router = TaskRouter(make_config())

    positive_with_secret_guard = router.route("请创建报告文件，但不得读取或输出任何真实密钥")
    positive_with_extra_file_guard = router.route("请创建汇总报告文件，但不要创建额外日志文件")
    negative_then_positive = router.route("不要创建临时日志文件；请输出汇总报告文件")
    same_clause_negative_then_positive = router.route("不要创建临时日志文件但请输出汇总报告文件")
    negative_only = router.route("检查所有文件，不得读取或输出任何真实凭据，直接回复检查结果")
    separate_clauses = router.route("检查项目文件；输出结论时不得包含任何凭据；直接回复即可")
    protected_report_path = router.route("不得输出真实凭据到报告文件，直接回复检查结果")
    english_protected_report = router.route("Do not output credentials to a report file; reply with the result.")
    pdf_after_rejected_word = router.route("不要生成 Word 文档；请生成 PDF 报告")

    assert "artifact-required" in positive_with_secret_guard.reasons
    assert "artifact-required" in positive_with_extra_file_guard.reasons
    assert "artifact-required" in negative_then_positive.reasons
    assert "artifact-required" in same_clause_negative_then_positive.reasons
    assert "artifact-required" not in negative_only.reasons
    assert "artifact-required" not in separate_clauses.reasons
    assert "artifact-required" not in protected_report_path.reasons
    assert "artifact-required" not in english_protected_report.reasons
    assert "artifact-required" in pdf_after_rejected_word.reasons
    assert "word-artifact-required" not in pdf_after_rejected_word.reasons


def test_task_router_uses_bounded_artifact_negation_and_filename_hints(make_config) -> None:
    router = TaskRouter(make_config())

    negative_requests = (
        "请检查代码，不生成报告文件，直接回复。",
        "Ensure the command does not create a report file; reply directly.",
        "确认运行后没有生成报告文件，直接回复。",
        "检查路由器是否会把‘生成报告文件’误判为真实请求，只需解释。",
        'Does the quoted phrase "create a report file" trigger artifact routing? Explain only.',
    )
    for prompt in negative_requests:
        routed = router.route(prompt)
        assert "artifact-required" not in routed.reasons, prompt
        assert "directory-artifact-required" not in routed.reasons, prompt
        assert routed.artifact_hints == (), prompt

    quoted_positive = router.route("请生成“汇总报告文件”，并保存到当前目录。")
    filename_word = router.route("Please create an output file named summary.docx.")
    passive_word = router.route("A Word report should be generated.")

    assert "artifact-required" in quoted_positive.reasons
    assert "word-artifact-required" in filename_word.reasons
    assert filename_word.artifact_hints == ("summary.docx",)
    assert "word-artifact-required" in passive_word.reasons
    assert passive_word.artifact_hints == (".docx",)


def test_task_router_distinguishes_directory_artifacts_from_directory_context(make_config) -> None:
    router = TaskRouter(make_config())

    positive_requests = (
        ("Create a new output directory.", "output"),
        ("Generate a results folder for the artifacts.", "results"),
        ("请新建输出目录。", "输出"),
        ("生成一个结果文件夹。", "结果"),
    )
    for prompt, directory_hint in positive_requests:
        routed = router.route(prompt)
        assert "artifact-required" in routed.reasons, prompt
        assert "directory-artifact-required" in routed.reasons, prompt
        assert routed.artifact_hints == (directory_hint,), prompt
        assert routed.directory_hints == (directory_hint,), prompt
        assert routed.schema_version == 2
        assert TaskRoute.from_dict(routed.to_dict()) == routed

    named_requests = (
        "Generate a directory named reports.",
        "Create the directory called reports.",
        "Create directory reports.",
        "创建一个名为 reports 的目录。",
        "创建 reports 目录。",
        "创建目录 reports。",
    )
    for prompt in named_requests:
        routed = router.route(prompt)
        assert routed.artifact_hints == ("reports",), prompt
        assert routed.directory_hints == ("reports",), prompt

    unnamed_requests = (
        "Create a directory.",
        "Create a new directory.",
        "Generate a directory for the artifacts.",
        "Create a directory under output.",
        "Create a directory beneath output.",
        "Create a directory below output.",
        "Create a directory inside output.",
        "Create a directory within output.",
        "Create a directory near output.",
        "Create a directory at the project root.",
        "Create a directory where reports can be stored.",
        "Create a temporary directory.",
        "Create a separate directory.",
        "Create an empty directory.",
        "创建一个目录。",
        "创建一个新的目录。",
        "创建一个空目录。",
        "创建临时目录。",
        "创建目录在 output 下。",
        "创建目录于 output 下。",
        "创建目录到 output 下。",
    )
    for prompt in unnamed_requests:
        routed = router.route(prompt)
        assert "directory-artifact-required" in routed.reasons, prompt
        assert routed.artifact_hints == (), prompt
        assert routed.directory_hints == (), prompt

    file_in_directory = router.route("Create the report file summary.md in the output directory.")
    chinese_file_in_directory = router.route("创建汇总报告文件 summary.md 并放到输出目录。")
    assert "artifact-required" in file_in_directory.reasons
    assert "directory-artifact-required" not in file_in_directory.reasons
    assert file_in_directory.directory_hints == ()
    assert "artifact-required" in chinese_file_in_directory.reasons
    assert "directory-artifact-required" not in chinese_file_in_directory.reasons
    assert chinese_file_in_directory.directory_hints == ()

    negative_or_meta = (
        "Do not create a directory; reply directly.",
        "不要创建任何文件夹，直接回复。",
        'Does the quoted phrase "create a directory" trigger artifact routing? Explain only.',
        "Does the wording create a directory trigger artifact routing? Explain only.",
        "检查路由器是否会把‘创建目录’误判为真实请求，只需解释。",
        "创建目录会不会触发 artifact 路由？只需解释。",
    )
    for prompt in negative_or_meta:
        routed = router.route(prompt)
        assert "artifact-required" not in routed.reasons, prompt
        assert "directory-artifact-required" not in routed.reasons, prompt


def test_task_router_distinguishes_conditional_fix_from_unconditional_and_report_only_work(make_config) -> None:
    router = TaskRouter(make_config())

    conditional_audit = router.route("审计代码；若没有找到可证实缺陷，不要修改代码")
    unconditional_fix = router.route("修复已经确认的解析缺陷并运行测试")
    report_only = router.route("审计代码并生成报告文件，禁止修改源码")
    unrelated_condition = router.route("如果没有网络则记录验证限制，不要修改配置")

    assert "conditional-mutation" in conditional_audit.reasons
    assert "conditional-mutation" not in unconditional_fix.reasons
    assert "conditional-mutation" not in report_only.reasons
    assert "artifact-required" in report_only.reasons
    assert "conditional-mutation" not in unrelated_condition.reasons


def test_task_router_scopes_conditional_mutation_to_its_bounded_clause(make_config) -> None:
    router = TaskRouter(make_config())

    mixed_chinese = router.route("若没有审计问题，不要修改审计说明。修复已经确认的解析缺陷。")
    mixed_english = router.route(
        "If no evidence of a bug is found, do not edit the audit notes. Implement the already confirmed parser fix."
    )
    positive_chinese = router.route("如果发现真实缺陷就修复，否则保持原样。")
    positive_english = router.route("If a real defect is found, fix it; otherwise leave the code unchanged.")

    assert "mutation-request" in mixed_chinese.reasons
    assert "conditional-mutation" not in mixed_chinese.reasons
    assert "mutation-request" in mixed_english.reasons
    assert "conditional-mutation" not in mixed_english.reasons
    assert "conditional-mutation" in positive_chinese.reasons
    assert "conditional-mutation" in positive_english.reasons


def test_task_router_recognizes_one_global_validation_without_per_scope_leakage(make_config) -> None:
    router = TaskRouter(make_config())

    global_once = (
        "Run the tests once and report the result.",
        "验证只运行一次并报告结果。",
        "仅运行一次静态检查。",
    )
    per_scope = (
        "Only run one check per package.",
        "Run tests once for each module.",
        "每个模块只运行一次测试。",
    )

    for prompt in global_once:
        assert "single-validation" in router.route(prompt).reasons, prompt
    for prompt in per_scope:
        assert "single-validation" not in router.route(prompt).reasons, prompt


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


def test_resume_task_upgrade_preserves_sticky_validation_and_artifact_constraints(make_config) -> None:
    router = TaskRouter(make_config())
    original = router.route("只运行一次静态检查并创建 output.docx 文件，再创建 output 目录。")
    upgraded = router.route("全面审计整个仓库的所有安全问题")

    resumed = more_capable_task_route(original, upgraded)

    assert resumed.mode == "deep"
    assert "single-validation" in resumed.reasons
    assert "artifact-required" in resumed.reasons
    assert "directory-artifact-required" in resumed.reasons
    assert "word-artifact-required" in resumed.reasons
    assert resumed.artifact_hints == ("output.docx", "output")
    assert resumed.directory_hints == ("output",)
    assert resumed.require_plan is True

    original = router.route("Create directory reports.")
    same_tier_addition = router.route("Also create directory archive.")
    resumed = more_capable_task_route(original, same_tier_addition)

    assert resumed.artifact_hints == ("reports", "archive")
    assert resumed.directory_hints == ("reports", "archive")


def test_task_route_artifact_hints_round_trip_and_legacy_default(make_config) -> None:
    routed = TaskRouter(make_config()).route("Create an output file named summary.docx.")

    assert TaskRoute.from_dict(routed.to_dict()) == routed
    legacy = routed.to_dict()
    legacy.pop("artifact_hints")
    legacy.pop("directory_hints")
    legacy["schema_version"] = 1
    assert TaskRoute.from_dict(legacy).artifact_hints == ()
    assert TaskRoute.from_dict(legacy).directory_hints == ()

    unsafe = routed.to_dict()
    unsafe["artifact_hints"] = [".docx", "../../secret.txt", "prompt text", "summary.docx"]
    assert TaskRoute.from_dict(unsafe).artifact_hints == ("summary.docx", ".docx")
    unsafe["directory_hints"] = ["reports", "../outside", ".", "reports"]
    assert TaskRoute.from_dict(unsafe).directory_hints == ("reports",)


def test_task_route_bounds_artifact_hints_before_serialization_and_resume(make_config) -> None:
    router = TaskRouter(make_config())
    prompt = " ".join(f"Create an output file named f{index}.md." for index in range(40))

    routed = router.route(prompt)

    assert len(routed.artifact_hints) == 32
    assert routed.artifact_hints == tuple(f"f{index}.md" for index in range(32))
    assert TaskRoute.from_dict(routed.to_dict()) == routed

    previous = replace(routed, artifact_hints=tuple(f"old{index}.md" for index in range(32)))
    selected = replace(
        routed,
        mode="deep",
        score=routed.score + 10,
        artifact_hints=tuple(f"new{index}.md" for index in range(32)),
    )
    resumed = more_capable_task_route(previous, selected)

    assert resumed.artifact_hints == previous.artifact_hints
    assert len(resumed.artifact_hints) == 32
    assert TaskRoute.from_dict(resumed.to_dict()) == resumed

    previous = replace(previous, directory_hints=tuple(f"old{index}" for index in range(32)))
    selected = replace(selected, directory_hints=tuple(f"new{index}" for index in range(32)))
    resumed = more_capable_task_route(previous, selected)

    assert resumed.directory_hints == previous.directory_hints
    assert len(resumed.directory_hints) == 32
    assert TaskRoute.from_dict(resumed.to_dict()) == resumed

    directory_prompt = " ".join(f"Create directory d{index}." for index in range(40))
    routed_directories = router.route(directory_prompt)
    expected_directories = tuple(f"d{index}" for index in range(32))

    assert routed_directories.artifact_hints == expected_directories
    assert routed_directories.directory_hints == expected_directories
    assert TaskRoute.from_dict(routed_directories.to_dict()) == routed_directories


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
    conditional = factory.build(router.route("全面审计整个大型代码库；若找到真实缺陷则修复，若没有充分证据不要修改"))
    assert "implementation is skipped" in conditional[2]["completion_criteria"]
    assert "exact outcomes are reported" in conditional[-1]["completion_criteria"]
    assert "pre-existing failures" in conditional[-1]["completion_criteria"]
    assert factory.build(router.route("什么是 Python？")) == []
    assert [item["id"] for item in factory.build(router.route("总结所有文档并直接回复"))] == [
        "scope",
        "parse-documents",
        "synthesize",
        "verify",
    ]
    assert [item["id"] for item in factory.build(router.route("总结所有文档并生成 Word 文件"))] == [
        "scope",
        "parse-documents",
        "synthesize",
        "render-artifact",
        "verify",
    ]
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
