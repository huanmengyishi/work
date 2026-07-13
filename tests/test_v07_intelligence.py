from __future__ import annotations

import json
import fcntl
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

from agent.context import ContextBuildRequest, ContextBuilder
from agent.daemon import ProjectDaemon
from agent.memory import MemoryStore
from agent.project import ProjectManager
from agent.prompt import PromptBuilder, SYSTEM_PROMPT
from agent.state import AgentState
from agent.task_queue import TaskQueueManager
from agent.tools import ToolManager
from agent.tools.base import ToolResult
from agent.tools.lsp import LSPManager


def test_memory_usage_dedupe_and_expiry_lifecycle(tmp_path: Path, make_config) -> None:
    config = make_config({"memory": {"dedupe_similarity": 0.9, "protect_kinds": ["Correction", "Decision"]}})
    memory = MemoryStore(config)
    first = memory.add_memory(
        kind="Lesson",
        title="Docker proxy timeout",
        content="Configure the Docker daemon proxy when image pulls time out.",
        project_id="project-x",
        confidence=0.8,
    )
    duplicate = memory.add_memory(
        kind="Lesson",
        title="Docker proxy timeout",
        content="Configure the Docker daemon proxy when image pulls time out.",
        project_id="project-x",
        confidence=0.7,
    )
    expired = memory.add_memory(
        kind="Reflection",
        title="Old low-confidence note",
        content="This note can expire.",
        project_id="project-x",
        confidence=0.2,
        expires_at=(datetime.now(UTC) - timedelta(days=1)).isoformat(),
    )
    protected = memory.add_memory(
        kind="Correction",
        title="Protected correction",
        content="Never expire a correction automatically.",
        project_id="project-x",
        confidence=0.2,
        expires_at=(datetime.now(UTC) - timedelta(days=1)).isoformat(),
    )

    found = memory.search("Docker proxy timeout", project_id="project-x")
    assert {item.id for item in found} == {first, duplicate}
    assert all((memory.get_memory(item.id) or item).use_count == 1 for item in found)

    preview = memory.maintain(project_id="project-x")
    assert preview["merge_count"] == 1
    assert preview["expired"] == [expired]
    assert memory.get_memory(duplicate).merged_into is None

    applied = memory.maintain(project_id="project-x", apply=True)
    assert applied["merge_count"] == 1
    assert memory.get_memory(expired) is None
    assert memory.get_memory(protected) is not None
    active = memory.list_memories(project_id="project-x")
    assert sum(item.kind == "Lesson" for item in active) == 1
    merged = memory.get_memory(duplicate)
    assert merged is not None and merged.merged_into == first
    assert duplicate not in {item.id for item in memory.search("Docker proxy timeout", project_id="project-x")}
    assert memory.stats(project_id="project-x").total == 2


def test_new_non_protected_memory_gets_default_expiry(make_config) -> None:
    memory = MemoryStore(make_config({"memory": {"expiry_days": 30}}))
    lesson_id = memory.add_memory(kind="Lesson", title="Temporary", content="Lifecycle managed")
    correction_id = memory.add_memory(kind="Correction", title="Durable", content="Protected")

    assert memory.get_memory(lesson_id).expires_at is not None
    assert memory.get_memory(correction_id).expires_at is None


def test_lsp_supports_partial_install_and_parses_python(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "broken.py"
    source.write_text("value: int = 'wrong'\n", encoding="utf-8")
    manager = LSPManager(tmp_path)
    monkeypatch.setattr("agent.tools.lsp.shutil.which", lambda name: "/bin/pyright" if name == "pyright" else None)
    payload = {
        "generalDiagnostics": [
            {
                "file": str(source),
                "severity": "error",
                "message": "Type string is not assignable to int",
                "rule": "reportAssignmentType",
                "range": {"start": {"line": 0, "character": 0}},
            }
        ]
    }
    monkeypatch.setattr(
        manager,
        "_run",
        lambda _args: subprocess.CompletedProcess(_args, 1, json.dumps(payload), ""),
    )

    available, reason = manager.available()
    result = manager.diagnostics("broken.py")

    assert available is True
    assert "Pyright" in reason
    assert result.success is False
    assert result.data["diagnostics"][0]["file"] == "broken.py"
    assert result.data["diagnostics"][0]["line"] == 1


def test_lsp_parses_typescript_diagnostics(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "broken.ts"
    source.write_text("const value: number = 'wrong';\n", encoding="utf-8")
    manager = LSPManager(tmp_path)
    monkeypatch.setattr("agent.tools.lsp.shutil.which", lambda name: "/bin/tsc" if name == "tsc" else None)
    output = f"{source}(1,7): error TS2322: Type 'string' is not assignable to type 'number'.\n"
    monkeypatch.setattr(
        manager,
        "_run",
        lambda _args: subprocess.CompletedProcess(_args, 2, output, ""),
    )

    result = manager.diagnostics("broken.ts")

    assert result.success is False
    assert result.data["diagnostics"][0]["code"] == "TS2322"
    assert result.data["diagnostics"][0]["file"] == "broken.ts"


def test_file_apply_remains_successful_when_diagnostics_find_errors(tmp_path: Path, make_config, monkeypatch) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "main.py").write_text("value = 1\n", encoding="utf-8")
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    tools = ToolManager(config, project, memory, yolo=True)
    state = AgentState.create(
        session_id="session-v07",
        project=project,
        user_request="edit",
        loaded_memories=[],
        loaded_tools=[],
        git_branch=None,
        context_index_path=str(project.agent_dir / "index.json"),
    )
    tools.bind_state(state)
    monkeypatch.setattr(
        tools.lsp,
        "diagnostics",
        lambda _path: ToolResult(
            False,
            "",
            "Error main.py:1:1 reportAssignmentType bad type",
            data={"error_count": 1, "diagnostics": [{"file": "main.py", "line": 1}]},
        ),
    )

    _, preview = tools.execute_model_call("file_diff", {"path": "main.py", "content": "value: int = 'bad'\n"})
    _, applied = tools.execute_model_call("file_apply", {"preview_id": preview.data["preview_id"]})

    assert applied.success is True
    assert applied.data["lsp"]["error_count"] == 1
    assert "bad type" in applied.stdout
    assert (root / "main.py").read_text(encoding="utf-8") == "value: int = 'bad'\n"


def test_semantic_index_resolves_internal_relationships_only(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg" / "helper.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (root / "pkg" / "service.py").write_text(
        "from .helper import helper\nimport requests\n\ndef run():\n    return helper()\n",
        encoding="utf-8",
    )
    config = make_config({"context": {"semantic_index_enabled": True, "semantic_languages": ["python"]}})
    project = ProjectManager(config).resolve_project(root)

    snapshot = ContextBuilder(config).build(project, refresh=True)
    semantic = json.loads((project.agent_dir / "index.semantic.json").read_text(encoding="utf-8"))

    assert {item["target"] for item in semantic["relationships"]} == {"pkg/helper.py"}
    assert "relation `pkg/service.py` -> `pkg/helper.py`" in snapshot.rendered
    assert len(snapshot.rendered) <= int(config.get("context.max_prompt_chars"))


def test_daemon_run_once_refreshes_context_without_mutating_active_queue(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "main.py").write_text("print('ok')\n", encoding="utf-8")
    config = make_config({"daemon": {"memory_maintenance_seconds": 60}})
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    queues = TaskQueueManager(project)
    queue = queues.create(["task"])
    queue.status = "running"
    queue.tasks[0].status = "running"
    queues.save(queue)
    daemon = ProjectDaemon(config, project, memory)

    assert daemon.run(once=True) == 0

    status = daemon.status()
    recovered = queues.load(queue.id)
    assert status.running is False
    assert status.state["status"] == "stopped"
    assert (project.agent_dir / "index.json").is_file()
    assert recovered.status == "running"
    assert recovered.tasks[0].status == "running"


def test_daemon_removes_stale_pid(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    daemon = ProjectDaemon(config, project, MemoryStore(config))
    daemon.pid_path.write_text("99999999\n", encoding="ascii")

    status = daemon.status()

    assert status.running is False
    assert not daemon.pid_path.exists()


def test_resume_prompt_compacts_raw_history(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    context = ContextBuilder(config).build(project)
    state = AgentState.create(
        session_id="compact-session",
        project=project,
        user_request="continue",
        loaded_memories=[],
        loaded_tools=[],
        git_branch=None,
        context_index_path=str(context.index_path),
    )
    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1"}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "x" * 50_000},
        {"role": "assistant", "content": "Previous task completed."},
    ]

    package = ContextBuilder(config).build_package(
        ContextBuildRequest(
            snapshot=context,
            state=state,
            memory_context="none",
            capability_summary="none",
            prior_messages=history,
            phase="resume",
            max_chars=12_000,
        )
    )
    messages = PromptBuilder().build_resume(package)

    assert len(messages) == 3
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert "Previous task completed." in messages[1]["content"]
    assert "x" * 100 not in json.dumps(messages)


def test_queue_lock_prevents_duplicate_runner(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    project = ProjectManager(make_config()).resolve_project(root)
    queues = TaskQueueManager(project)
    record = queues.create(["task"])
    lock_path = queues.queue_dir / f"{record.id}.lock"
    lock = lock_path.open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        try:
            queues.run(record, lambda _task, _record: ("ok", "session", "completed"))
        except RuntimeError as exc:
            assert "already running" in str(exc)
        else:
            raise AssertionError("duplicate queue runner was not rejected")
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
