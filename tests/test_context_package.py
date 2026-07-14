from __future__ import annotations

from pathlib import Path

from agent.context import ContextBuildRequest, ContextBuilder
from agent.memory import MemoryItem
from agent.project import ProjectManager
from agent.prompt import PromptBuilder, SYSTEM_PROMPT
from agent.state import AgentState, PlanStep


def _state(project, snapshot, request: str = "continue the bounded audit") -> AgentState:
    state = AgentState.create(
        session_id="context-package-session",
        project=project,
        user_request=request,
        loaded_memories=[],
        loaded_tools=[],
        git_branch=snapshot.git_branch,
        context_index_path=str(snapshot.index_path),
    )
    state.task_strategy = {
        "mode": "large",
        "thinking_enabled": True,
        "reasoning_effort": "high",
        "chunked_context": True,
        "require_plan": True,
    }
    state.plan = [
        PlanStep(
            id="inspect",
            title="Inspect complete context boundaries",
            status="in_progress",
            completion_criteria="Every selected section fits the declared package budget",
        )
    ]
    return state


def test_package_budget_keeps_task_and_project_instructions_private_to_memory(
    tmp_path: Path,
    make_config,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "AGENTS.md").write_text(
        "# Rules\n\nMUST_KEEP_PROJECT_INSTRUCTION\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "# Large README\n\n" + "repository documentation paragraph\n\n" * 2_000,
        encoding="utf-8",
    )
    (root / "main.py").write_text("def main():\n    return 1\n", encoding="utf-8")
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    builder = ContextBuilder(config)
    snapshot = builder.build(project, refresh=True)
    state = _state(project, snapshot)
    generated_before = snapshot.generated_path.read_text(encoding="utf-8")

    package = builder.build_package(
        ContextBuildRequest(
            snapshot=snapshot,
            state=state,
            memory_context="PRIVATE_MEMORY_MUST_NOT_BE_CACHED\n" + "memory detail\n" * 1_000,
            capability_summary="- `read_file`: read\n" + "- `large_tool`: capability\n" * 1_000,
            max_chars=1_800,
        )
    )

    assert package.used_chars == len(package.rendered) + len(package.user_request)
    assert package.used_chars <= package.max_chars == 1_800
    assert "Inspect complete context boundaries" in package.rendered
    assert "MUST_KEEP_PROJECT_INSTRUCTION" in package.rendered
    assert package.omitted_sections or package.truncated_sections
    assert snapshot.generated_path.read_text(encoding="utf-8") == generated_before
    assert "PRIVATE_MEMORY_MUST_NOT_BE_CACHED" not in generated_before


def test_package_truncates_memory_only_at_complete_record_boundaries(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    builder = ContextBuilder(config)
    snapshot = builder.build(project, refresh=True)
    state = _state(project, snapshot)
    state.execution_context = None
    first = "- [project/Lesson] FIRST_COMPLETE_RECORD\n  tags: first\n  " + "A" * 180
    second = "- [project/Lesson] SECOND_RECORD_MUST_NOT_BE_SPLIT\n  tags: second\n  " + "B" * 2_000

    package = builder.build_package(
        ContextBuildRequest(
            snapshot=snapshot,
            state=state,
            memory_context=first + "\n" + second,
            max_chars=2_200,
        )
    )

    assert package.used_chars == len(package.rendered) + len(package.user_request)
    assert package.used_chars <= 2_200
    assert "FIRST_COMPLETE_RECORD" in package.rendered
    assert "A" * 180 in package.rendered
    assert "SECOND_RECORD_MUST_NOT_BE_SPLIT" not in package.rendered
    assert "B" * 50 not in package.rendered
    assert "...[truncated]" in package.rendered


def test_resume_package_selects_previous_outcome_and_prompt_only_renders_package(
    tmp_path: Path,
    make_config,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    builder = ContextBuilder(config)
    snapshot = builder.build(project, refresh=True)
    state = _state(project, snapshot, request="继续")
    state.fail("PERSISTED_FAILURE_BEFORE_RESUME")
    state.resume("继续")
    previous = "PREVIOUS_OUTCOME_HEAD" + "x" * 8_000 + "PREVIOUS_OUTCOME_TAIL"
    history = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "RAW_TOOL_TRANSCRIPT" * 10_000},
        {"role": "assistant", "content": previous},
    ]

    package = builder.build_package(
        ContextBuildRequest(
            snapshot=snapshot,
            state=state,
            prior_messages=history,
            phase="resume",
            max_chars=10_000,
        )
    )
    messages = PromptBuilder().build_resume(package)

    assert package.used_chars == len(package.rendered) + len(package.user_request)
    assert package.used_chars <= 10_000
    assert "PREVIOUS_OUTCOME_HEAD" in package.rendered
    assert "PREVIOUS_OUTCOME_TAIL" in package.rendered
    assert "PERSISTED_FAILURE_BEFORE_RESUME" in package.rendered
    assert "RAW_TOOL_TRANSCRIPT" not in package.rendered
    assert messages == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": package.rendered},
        {"role": "user", "content": "继续"},
    ]


def test_recovery_package_is_a_bounded_delta_and_tracks_included_memory(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "README.md").write_text("PUBLIC_PROJECT_CONTEXT\n" * 100, encoding="utf-8")
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    builder = ContextBuilder(config)
    snapshot = builder.build(project, refresh=True)
    state = _state(project, snapshot)
    memory = MemoryItem(
        id=42,
        project_id=project.id,
        kind="Correction",
        title="Use the verified recovery path",
        content="Do not repeat the failed command.",
        tags=["correction:test"],
        updated_at="2026-07-13T00:00:00+00:00",
    )
    initial = builder.build_package(
        ContextBuildRequest(snapshot=snapshot, state=state, memory_items=[memory], max_chars=4_000)
    )
    recovery = builder.build_package(
        ContextBuildRequest(
            snapshot=snapshot,
            state=state,
            recovery_context="Use the prior correction before retrying.",
            recovery_memory_ids=[42],
            phase="recovery",
            max_chars=1_200,
        )
    )

    assert initial.included_memory_ids == (42,)
    assert recovery.included_memory_ids == (42,)
    assert {section.key for section in recovery.sections} <= {"task", "recovery"}
    assert "PUBLIC_PROJECT_CONTEXT" not in recovery.rendered
    assert recovery.used_chars <= recovery.max_chars == 1_200


def test_package_counts_and_bounds_long_user_request(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    builder = ContextBuilder(config)
    snapshot = builder.build(project, refresh=True)
    request = "REQUEST_HEAD\n" + "中间文本" * 20_000 + "\nREQUEST_TAIL"
    state = _state(project, snapshot, request=request)

    package = builder.build_package(ContextBuildRequest(snapshot=snapshot, state=state, max_chars=4_000))
    messages = PromptBuilder().build_initial(package)

    assert package.used_chars == len(package.rendered) + len(package.user_request)
    assert package.used_chars <= 4_000
    assert package.original_user_request_chars == len(request)
    assert package.user_request_truncated is True
    assert "REQUEST_HEAD" in package.user_request
    assert "REQUEST_TAIL" in package.user_request
    assert "middle truncated" in package.user_request
    assert messages[-1]["content"] == package.user_request


def test_single_paragraph_project_instructions_keep_head_and_tail(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "AGENTS.md").write_text(
        "HEAD_MANDATORY_RULE " + "x" * 9_000 + " TAIL_MANDATORY_RULE",
        encoding="utf-8",
    )
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    builder = ContextBuilder(config)
    snapshot = builder.build(project, refresh=True)
    state = _state(project, snapshot)

    package = builder.build_package(ContextBuildRequest(snapshot=snapshot, state=state, max_chars=12_000))

    assert "HEAD_MANDATORY_RULE" in package.rendered
    assert "TAIL_MANDATORY_RULE" in package.rendered
    assert "middle truncated" in package.rendered
