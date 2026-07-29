from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent import cli
from agent.context import ContextPackage
from agent.deepseek import ChatResponse
from agent.global_knowledge import GlobalKnowledgeBase
from agent.memory import MemoryStore
from agent.memory_transfer import MemoryTransferError, export_memory, import_memory
from agent.project import ProjectManager
from agent.prompt import PromptBuilder
from agent.runtime import AgentRuntime
from agent.tools import ToolManager


class _FinalClient:
    def chat(self, **_kwargs) -> ChatResponse:
        return ChatResponse(message={"role": "assistant", "content": "done"}, raw={})


class _RecordingPromptBuilder(PromptBuilder):
    def __init__(self) -> None:
        self.packages: list[ContextPackage] = []

    def build_initial(self, package: ContextPackage) -> list[dict[str, object]]:
        self.packages.append(package)
        return super().build_initial(package)


def _project(tmp_path: Path, make_config):
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    memory.sync_project(project)
    return config, project, memory


def test_runtime_and_store_expose_only_current_project_plus_supported_global_knowledge(
    tmp_path: Path,
    make_config,
) -> None:
    config, project, memory = _project(tmp_path, make_config)
    marker = "cross-project-boundary-marker"
    project_id = memory.add_memory(
        kind="Reflection",
        title="current project reflection",
        content=marker,
        project_id=project.id,
    )
    shared_id = GlobalKnowledgeBase(memory).add(
        kind="Knowledge",
        title="shared convention",
        content=marker,
    )
    hidden_global_id = memory.add_memory(
        kind="Reflection",
        title="legacy global operational record",
        content=marker,
        project_id=None,
    )
    foreign_id = memory.add_memory(
        kind="Knowledge",
        title="foreign project record",
        content=marker,
        project_id="foreign-project",
    )

    selected = memory.search(marker, project.id, record_usage=False)
    assert {item.id for item in selected} == {project_id, shared_id}
    assert {item.id for item in memory.list_memories(project_id=project.id)} == {project_id, shared_id}
    assert hidden_global_id not in {item.id for item in selected}
    assert foreign_id not in {item.id for item in selected}

    prompt_builder = _RecordingPromptBuilder()
    runtime = AgentRuntime.with_default_services(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=_FinalClient(),
        prompt_builder=prompt_builder,
    )

    assert runtime.run(marker) == "done"
    assert set(prompt_builder.packages[0].included_memory_ids) == {project_id, shared_id}


def test_cli_global_boundary_is_append_only_and_project_mutation_is_owned(
    tmp_path: Path,
    make_config,
    monkeypatch,
    capsys,
) -> None:
    config, project, memory = _project(tmp_path, make_config)
    monkeypatch.setattr(cli, "prepare_project", lambda _config: (project, memory))
    parser = cli.build_command_parser()

    rejected = parser.parse_args(["memory", "add", "Reflection", "not shared", "operational", "--global-memory"])
    assert cli.cmd_memory(config, rejected) == 2
    assert "global knowledge kind" in capsys.readouterr().err

    allowed = parser.parse_args(["memory", "add", "Knowledge", "shared rule", "portable", "--global-memory"])
    assert cli.cmd_memory(config, allowed) == 0
    capsys.readouterr()
    shared = GlobalKnowledgeBase(memory).search("portable", record_usage=False)[0]

    for command in (
        ["memory", "edit", str(shared.id), "--content", "forbidden update"],
        ["memory", "delete", str(shared.id)],
    ):
        assert cli.cmd_memory(config, parser.parse_args(command)) == 2
        assert "add-only by default" in capsys.readouterr().err
    assert memory.get_memory(shared.id).content == "portable"

    foreign_id = memory.add_memory(
        kind="Knowledge",
        title="foreign",
        content="private",
        project_id="foreign-project",
    )
    assert (
        cli.cmd_memory(
            config,
            parser.parse_args(["memory", "delete", str(foreign_id)]),
        )
        == 2
    )
    assert memory.get_memory(foreign_id) is not None

    local_id = memory.add_memory(kind="Knowledge", title="local", content="owned", project_id=project.id)
    assert (
        cli.cmd_memory(
            config,
            parser.parse_args(["memory", "edit", str(local_id), "--content", "updated"]),
        )
        == 0
    )
    assert memory.get_memory(local_id).content == "updated"
    assert cli.cmd_memory(config, parser.parse_args(["memory", "delete", str(local_id)])) == 0


def test_tool_global_add_and_search_use_the_same_boundary(tmp_path: Path, make_config) -> None:
    config, project, memory = _project(tmp_path, make_config)
    tools = ToolManager(config, project, memory, yolo=True)
    marker = "tool-global-boundary-marker"
    memory.add_memory(
        kind="Reflection",
        title="hidden operational global",
        content=marker,
        project_id=None,
    )
    memory.add_memory(
        kind="Knowledge",
        title="foreign project",
        content=marker,
        project_id="foreign-project",
    )
    local_id = memory.add_memory(
        kind="Bug",
        title="local operational",
        content=marker,
        project_id=project.id,
    )

    _, denied = tools.execute_model_call(
        "memory_add",
        {
            "kind": "Reflection",
            "title": "not reusable",
            "content": "must stay project scoped",
            "global_memory": True,
        },
    )
    assert denied.success is False
    assert "global knowledge kind" in denied.stderr

    _, added = tools.execute_model_call(
        "memory_add",
        {
            "kind": "Lesson",
            "title": "reusable lesson",
            "content": marker,
            "global_memory": True,
        },
    )
    assert added.success is True
    _, searched = tools.execute_model_call("memory_search", {"query": marker})
    assert searched.success is True
    assert {item["id"] for item in searched.data["items"]} == {local_id, added.data["id"]}


def test_project_maintenance_and_transfer_cannot_mutate_or_recreate_hidden_global_kinds(
    tmp_path: Path,
    make_config,
) -> None:
    _config, project, memory = _project(tmp_path, make_config)
    global_id = GlobalKnowledgeBase(memory).add(
        kind="Lesson",
        title="same lesson",
        content="same reusable content",
    )
    memory.add_memory(
        kind="Lesson",
        title="same lesson",
        content="same reusable content",
        project_id=project.id,
    )

    report = memory.maintain(project_id=project.id, apply=True)
    assert report["scanned"] == 1
    assert report["merge_count"] == 0
    assert memory.get_memory(global_id).merged_into is None

    legacy_global_id = memory.add_memory(
        kind="Reflection",
        title="legacy operational global",
        content="must not cross project boundaries",
        project_id=None,
    )
    export_path = tmp_path / "global.json"
    exported = export_memory(memory, export_path, project_id=project.id, scope="global")
    exported_document = json.loads(export_path.read_text(encoding="utf-8"))
    assert exported["record_count"] == 1
    assert [record["kind"] for record in exported_document["records"]] == ["Lesson"]
    assert legacy_global_id not in {item.id for item in GlobalKnowledgeBase(memory).list()}

    exported_document["records"] = [
        {
            "scope": "global",
            "kind": "Reflection",
            "title": "forbidden import",
            "content": "must remain hidden",
            "tags": [],
            "confidence": 0.7,
            "expires_at": None,
        }
    ]
    export_path.write_text(json.dumps(exported_document), encoding="utf-8")
    with pytest.raises(MemoryTransferError, match="global knowledge kind"):
        import_memory(memory, export_path, project_id=project.id)
