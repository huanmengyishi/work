from __future__ import annotations

import json
from pathlib import Path

from agent.memory import MemoryStore
from agent.project import ProjectManager
from agent.state import AgentState
from agent.tools import ToolManager


def build_tools(root: Path, make_config, *, approve=None, auto_approve=False):
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    memory.sync_project(project)
    tools = ToolManager(
        config,
        project,
        memory,
        approval_handler=approve,
        auto_approve=auto_approve,
    )
    state = AgentState.create(
        session_id="edit-session",
        project=project,
        user_request="edit files",
        loaded_memories=[],
        loaded_tools=[],
        git_branch=None,
        context_index_path=str(project.agent_dir / "index.json"),
    )
    tools.bind_state(state)
    return project, tools


def test_preview_apply_and_undo_existing_file(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    target = root / "example.txt"
    target.write_text("before\n", encoding="utf-8")
    project, tools = build_tools(root, make_config, auto_approve=True)

    _, preview = tools.execute_model_call(
        "file_diff",
        {"path": "example.txt", "old_text": "before", "new_text": "after"},
    )
    assert preview.success is True
    assert "-before" in preview.stdout
    assert "+after" in preview.stdout
    assert target.read_text(encoding="utf-8") == "before\n"

    _, applied = tools.execute_model_call("file_apply", {"preview_id": preview.data["preview_id"]})
    assert applied.success is True
    assert applied.data["before_exists"] is True
    assert applied.data["after_exists"] is True
    assert target.read_text(encoding="utf-8") == "after\n"
    snapshot = project.agent_dir / "snapshots" / "edit-session" / applied.data["snapshot_id"]
    assert (snapshot / "before.bin").read_text(encoding="utf-8") == "before\n"

    _, undone = tools.execute_model_call("file_undo", {})
    assert undone.success is True
    assert undone.data["restored_exists"] is True
    assert target.read_text(encoding="utf-8") == "before\n"


def test_apply_and_undo_report_create_and_delete_existence(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _, tools = build_tools(root, make_config, auto_approve=True)
    target = root / "transition.txt"

    _, create_preview = tools.execute_model_call(
        "file_diff",
        {"path": "transition.txt", "content": "created\n"},
    )
    _, created = tools.execute_model_call("file_apply", {"preview_id": create_preview.data["preview_id"]})

    assert created.success is True
    assert created.data["before_exists"] is False
    assert created.data["after_exists"] is True
    assert target.is_file()

    _, creation_undone = tools.execute_model_call(
        "file_undo",
        {"snapshot_id": created.data["snapshot_id"]},
    )
    assert creation_undone.success is True
    assert creation_undone.data["restored_exists"] is False
    assert not target.exists()

    target.write_text("delete me\n", encoding="utf-8")
    _, delete_preview = tools.execute_model_call(
        "file_diff",
        {"path": "transition.txt", "delete": True},
    )
    _, deleted = tools.execute_model_call("file_apply", {"preview_id": delete_preview.data["preview_id"]})

    assert deleted.success is True
    assert deleted.data["before_exists"] is True
    assert deleted.data["after_exists"] is False
    assert not target.exists()

    _, deletion_undone = tools.execute_model_call(
        "file_undo",
        {"snapshot_id": deleted.data["snapshot_id"]},
    )
    assert deletion_undone.success is True
    assert deletion_undone.data["restored_exists"] is True
    assert target.read_text(encoding="utf-8") == "delete me\n"


def test_preview_rejects_identical_replacement_before_searching_file(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    target = root / "example.txt"
    target.write_text("actual source\n", encoding="utf-8")
    _, tools = build_tools(root, make_config, auto_approve=True)

    _, preview = tools.execute_model_call(
        "file_diff",
        {"path": "example.txt", "old_text": "not in file", "new_text": "not in file"},
    )

    assert preview.success is False
    assert "old_text and new_text are identical" in preview.stderr
    assert "old_text was not found" not in preview.stderr
    assert target.read_text(encoding="utf-8") == "actual source\n"


def test_create_delete_and_conflict_protection(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    _, tools = build_tools(root, make_config, auto_approve=True)

    _, create_preview = tools.execute_model_call("file_diff", {"path": "new.txt", "content": "created\n"})
    _, created = tools.execute_model_call("file_apply", {"preview_id": create_preview.data["preview_id"]})
    assert created.success is True
    assert (root / "new.txt").read_text(encoding="utf-8") == "created\n"

    _, delete_preview = tools.execute_model_call("file_diff", {"path": "new.txt", "delete": True})
    (root / "new.txt").write_text("newer work\n", encoding="utf-8")
    _, conflict = tools.execute_model_call("file_apply", {"preview_id": delete_preview.data["preview_id"]})
    assert conflict.success is False
    assert "changed after preview" in conflict.stderr
    assert (root / "new.txt").read_text(encoding="utf-8") == "newer work\n"


def test_apply_rejects_tampered_preview_content(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    target = root / "example.txt"
    target.write_text("before\n", encoding="utf-8")
    project, tools = build_tools(root, make_config, auto_approve=True)
    _, preview = tools.execute_model_call("file_diff", {"path": "example.txt", "content": "reviewed\n"})
    preview_path = project.agent_dir / "cache" / "file-previews" / f"{preview.data['preview_id']}.json"
    record = json.loads(preview_path.read_text(encoding="utf-8"))
    record["content"] = "tampered\n"
    preview_path.write_text(json.dumps(record), encoding="utf-8")

    _, applied = tools.execute_model_call("file_apply", {"preview_id": preview.data["preview_id"]})

    assert applied.success is False
    assert "preview content changed" in applied.stderr
    assert target.read_text(encoding="utf-8") == "before\n"


def test_confirmation_gate_and_path_escape(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    target = root / "example.txt"
    target.write_text("before\n", encoding="utf-8")
    seen: list[str] = []
    _, tools = build_tools(
        root, make_config, approve=lambda request, capability, summary: seen.append(summary) or False
    )

    _, preview = tools.execute_model_call("file_diff", {"path": "example.txt", "content": "after\n"})
    _, denied = tools.execute_model_call("file_apply", {"preview_id": preview.data["preview_id"]})
    _, escaped = tools.execute_model_call("file_diff", {"path": "../outside.txt", "content": "no"})

    assert denied.success is False
    assert "denied by user" in denied.stderr
    assert denied.data["not_executed"] is True
    assert seen and "example.txt" in seen[0]
    assert target.read_text(encoding="utf-8") == "before\n"
    assert escaped.success is False
    assert "outside" in escaped.stderr


def test_safe_templates_are_parameterised(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "main.py").write_text("def answer():\n    return 42\n", encoding="utf-8")
    _, tools = build_tools(root, make_config)

    _, listed = tools.execute_model_call("list_dir", {"path": ".", "depth": 1})
    _, searched = tools.execute_model_call("search_code", {"query": "answer", "glob": "*.py"})
    _, read = tools.execute_model_call("read_file", {"path": "main.py", "start_line": 1, "end_line": 2})
    _, escaped = tools.execute_model_call("read_file", {"path": "../outside.txt"})
    private_path = root / ".project-agent" / "browser-sessions" / "secret" / "Cookies"
    private_path.parent.mkdir(parents=True)
    private_path.write_text("private", encoding="utf-8")
    _, private = tools.execute_model_call(
        "read_file",
        {"path": ".project-agent/browser-sessions/secret/Cookies"},
    )

    assert listed.success is True and "main.py" in listed.stdout
    assert searched.success is True and "main.py" in searched.stdout
    assert read.success is True and "return 42" in read.stdout
    assert escaped.success is False and "outside" in escaped.stderr
    assert private.success is False and "private Agent data" in private.stderr
