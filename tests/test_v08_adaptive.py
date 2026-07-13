from __future__ import annotations

import json
import hashlib
import subprocess
import time
from pathlib import Path

import pytest

from agent.context import ContextBuilder, read_git_branch
from agent.daemon import ProjectDaemon
from agent.events import JsonlEventLogger, Event, sanitize_for_log
from agent.memory import MemoryItem, MemoryStore
from agent.planner import PlanManager
from agent.project import ProjectManager
from agent.state import AgentState
from agent.task_strategy import TaskStrategySelector
from agent.tools.pathsafe import resolve_project_path


def make_state(project) -> AgentState:
    return AgentState.create(
        session_id="session-v08",
        project=project,
        user_request="adaptive task",
        loaded_memories=[],
        loaded_tools=[],
        git_branch="main",
        context_index_path=str(project.agent_dir / "index.json"),
    )


def test_task_strategy_selects_light_standard_large_and_deep(make_config) -> None:
    selector = TaskStrategySelector(make_config())

    assert selector.select("什么是 Python？").mode == "simple"
    assert selector.select("请比较两个方案并给出建议").mode == "standard"
    assert selector.select("修复这个函数并运行测试").mode == "standard"
    assert selector.select("分析整个代码库的所有文件并总结").mode == "large"
    deep = selector.select("全面审计整个仓库，找出所有安全根因并大规模重构")

    assert deep.mode == "deep"
    assert deep.reasoning_effort == "max"
    assert deep.require_plan is True
    assert deep.max_tool_rounds == 24
    plan = selector.initial_plan("全面修复整个仓库", deep)
    assert [item["id"] for item in plan] == ["scope", "inspect-chunks", "implement", "verify"]


def test_plan_ids_are_deduplicated_and_truncation_is_visible(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    project = ProjectManager(make_config()).resolve_project(root)
    state = make_state(project)
    planner = PlanManager()

    plan = planner.replace(
        state,
        [
            {"id": "step-1", "title": "First"},
            {"id": "step@1", "title": "Second"},
        ],
    )
    assert [item.id for item in plan] == ["step-1", "step-1-2"]

    with pytest.warns(RuntimeWarning, match="truncated from 60 to 50"):
        plan = planner.replace(state, [f"step {index}" for index in range(60)])
    assert len(plan) == 50


def test_worktree_git_branch_and_external_index_are_rendered_safely(tmp_path: Path, make_config) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    git_dir = tmp_path / "metadata"
    git_dir.mkdir()
    (root / ".git").write_text(f"gitdir: {git_dir}\n", encoding="utf-8")
    (git_dir / "HEAD").write_text("ref: refs/heads/feature/adaptive\n", encoding="utf-8")
    assert read_git_branch(root) == "feature/adaptive"

    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    builder = ContextBuilder(config)
    rendered, _ = builder._render_context(
        project,
        {
            "file_count": 1,
            "entries": [],
            "symbols": [
                {"path": None, "line": 1, "kind": "function", "name": "bad"},
                {"path": "main.py", "line": 2, "kind": "function", "name": "good"},
            ],
        },
    )
    assert "None:1" not in rendered
    assert "main.py:2" in rendered


def test_tsx_is_in_default_semantic_language_set(make_config) -> None:
    assert "tsx" in make_config().get("context.semantic_languages")


def test_memory_directory_is_private_and_ignored(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    project = ProjectManager(make_config()).resolve_project(root)

    assert "memory/" in (project.agent_dir / ".gitignore").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="private Agent data"):
        resolve_project_path(root, ".project-agent/memory/lesson/private.md")


def test_project_agent_symlink_and_source_symlink_are_rejected(tmp_path: Path, make_config) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "project"
    root.mkdir()
    (root / ".project-agent").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        ProjectManager(make_config()).resolve_project(root)

    (root / ".project-agent").unlink()
    external_source = outside / "outside.py"
    external_source.write_text("def leaked_symbol():\n    pass\n", encoding="utf-8")
    (root / "link.py").symlink_to(external_source)
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    snapshot = ContextBuilder(config).build(project, refresh=True)
    assert "leaked_symbol" not in json.dumps(snapshot.index)


def test_log_sanitization_redacts_secret_values_and_private_permissions(tmp_path: Path) -> None:
    sanitized = sanitize_for_log(
        {
            "prompt": "use sk-abcdefghijklmnopqrstuvwxyz or Bearer abcdefghijklmnop",
            "nested": "DEEPSEEK_API_KEY=super-secret-value",
        }
    )
    rendered = json.dumps(sanitized)
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in rendered
    assert "abcdefghijklmnop" not in rendered
    assert "super-secret-value" not in rendered

    logger = JsonlEventLogger(tmp_path / "logs")
    logger(Event("test", {"prompt": "sk-abcdefghijklmnopqrstuvwxyz"}))
    path = next((tmp_path / "logs").glob("events-*.jsonl"))
    assert path.stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "logs").stat().st_mode & 0o777 == 0o700


def test_daemon_queue_timeout_is_capped(monkeypatch, tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config({"daemon": {"queue_timeout_seconds": 90}, "tools": {"shell": {"timeout_seconds": 300}}})
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    daemon = ProjectDaemon(config, project, memory)

    class Pending:
        id = "queue-1"
        status = "pending"
        tasks = [object()] * 100

    monkeypatch.setattr("agent.daemon.TaskQueueManager.list", lambda self, limit=100: [Pending()])
    seen: dict[str, int] = {}

    def fake_run(*args, **kwargs):
        seen["timeout"] = kwargs["timeout"]
        return subprocess.CompletedProcess(args[0], 0, "", "")

    monkeypatch.setattr("agent.daemon.subprocess.run", fake_run)

    assert daemon._run_one_pending_queue() == {"id": "queue-1", "returncode": 0}
    assert seen["timeout"] == 90


def test_memory_dedupe_prefilter_avoids_global_all_pairs() -> None:
    items = []
    for index in range(1200):
        digest = hashlib.sha256(str(index).encode()).hexdigest()
        items.append(
            MemoryItem(
                id=index,
                project_id="p",
                kind="Lesson",
                title=f"{digest} unique topic",
                content=f"{digest[::-1]} independent content",
                tags=[],
                updated_at="2026-07-13T00:00:00+00:00",
            )
        )
    started = time.monotonic()
    assert MemoryStore._duplicate_groups(items, 0.94) == []
    assert time.monotonic() - started < 2.5

    first = MemoryItem(
        id=2001,
        project_id="p",
        kind="Lesson",
        title="Docker proxy timeout",
        content="configure daemon proxy for image pulls",
        tags=[],
        updated_at="2026-07-13T00:00:00+00:00",
    )
    second = MemoryItem(
        id=2002,
        project_id="p",
        kind="Lesson",
        title="Fix Docker proxy timeout",
        content="configure daemon proxy for image pulls",
        tags=[],
        updated_at="2026-07-13T00:00:00+00:00",
    )
    assert MemoryStore._duplicate_groups([first, second], 0.8) == [[first, second]]

    typo_first = MemoryItem(
        id=2003,
        project_id="p",
        kind="Lesson",
        title="fooBarIdentifierValue",
        content="",
        tags=[],
        updated_at="2026-07-13T00:00:00+00:00",
    )
    typo_second = MemoryItem(
        id=2004,
        project_id="p",
        kind="Lesson",
        title="fooBazIdentifierValue",
        content="",
        tags=[],
        updated_at="2026-07-13T00:00:00+00:00",
    )
    assert MemoryStore._duplicate_groups([typo_first, typo_second], 0.94) == [[typo_first, typo_second]]
