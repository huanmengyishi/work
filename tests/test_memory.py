from __future__ import annotations

from pathlib import Path

from agent.memory import MemoryStore
from agent.project import ProjectManager


def test_global_only_search_excludes_project_memory(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    memory.add_memory(kind="Knowledge", title="shared marker", content="global value", project_id=None)
    memory.add_memory(kind="Knowledge", title="shared marker", content="project value", project_id=project.id)

    items = memory.search("shared marker", project_id=None, global_only=True)

    assert len(items) == 1
    assert items[0].project_id is None
    assert items[0].content == "global value"


def test_memory_crud_stats_and_recovery(tmp_path: Path, make_config) -> None:
    config = make_config()
    memory = MemoryStore(config)
    correction_id = memory.add_memory(
        kind="Correction",
        title="Correct API port",
        content="Use port 8080 after connection refused on 8000.",
        tags=["correction:port", "project-x"],
        project_id="project-x",
    )
    lesson_id = memory.add_memory(
        kind="Lesson",
        title="Connection refused",
        content="Check the configured service port.",
        tags=["network"],
        project_id="project-x",
    )

    listed = memory.list_memories(project_id="project-x", tag="correction:port")
    assert [item.id for item in listed] == [correction_id]
    recovered = memory.search_recovery("connection refused on port 8000", "project-x")
    assert [item.id for item in recovered] == [correction_id, lesson_id]
    updated = memory.update_memory(
        correction_id,
        content="Use port 8080 for this service.",
        tags=["correction:port", "project-x", "verified"],
    )
    assert updated.content == "Use port 8080 for this service."
    stats = memory.stats(project_id="project-x")
    assert stats.total == 2
    assert stats.by_kind == {"Correction": 1, "Lesson": 1}
    assert stats.by_tag["verified"] == 1
    assert memory.delete_memory(lesson_id) is True
    assert memory.get_memory(lesson_id) is None


def test_memory_usage_can_be_recorded_after_context_budget_selection(tmp_path: Path, make_config) -> None:
    memory = MemoryStore(make_config())
    memory_id = memory.add_memory(
        kind="Lesson",
        title="bounded context selection",
        content="Only count this entry after it enters the package.",
        project_id="project-x",
    )

    selected = memory.search("bounded context selection", "project-x", record_usage=False)

    assert [item.id for item in selected] == [memory_id]
    assert memory.get_memory(memory_id).use_count == 0
    memory.record_usage([memory_id])
    assert memory.get_memory(memory_id).use_count == 1
