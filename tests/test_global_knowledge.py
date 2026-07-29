from __future__ import annotations

import pytest

from agent.global_knowledge import GlobalKnowledgeBase
from agent.memory import MemoryKind, MemoryStore


def test_global_knowledge_facade_exposes_only_shared_supported_kinds(make_config) -> None:
    memory = MemoryStore(make_config())
    knowledge = GlobalKnowledgeBase(memory)
    knowledge_id = knowledge.add(
        kind="knowledge",
        title="Shared API convention",
        content="Use bounded retries across projects.",
        tags=["architecture"],
    )
    lesson_id = knowledge.add(kind=MemoryKind.LESSON, title="Shared lesson", content="Verify before retrying.")
    decision_id = knowledge.add(kind="Decision", title="Shared decision", content="DeepSeek remains the provider.")
    hidden_global_id = memory.add_memory(
        kind="Reflection",
        title="operational reflection",
        content="This is not general knowledge.",
        project_id=None,
    )
    hidden_project_id = memory.add_memory(
        kind="Knowledge",
        title="project-only convention",
        content="This belongs to one project.",
        project_id="project-a",
    )

    assert {item.id for item in knowledge.list()} == {knowledge_id, lesson_id, decision_id}
    assert [item.id for item in knowledge.search("bounded", record_usage=False)] == [knowledge_id]
    assert knowledge.get(knowledge_id).project_id is None
    assert knowledge.get(hidden_global_id) is None
    assert knowledge.get(hidden_project_id) is None

    with pytest.raises(ValueError, match="global knowledge kind"):
        knowledge.add(kind="Correction", title="not shared", content="Corrections stay in normal Memory.")
    with pytest.raises(PermissionError, match="add-only by default"):
        knowledge.update(hidden_project_id, content="must not cross scope")
    with pytest.raises(PermissionError, match="add-only by default"):
        knowledge.delete(hidden_global_id)


def test_global_knowledge_filter_update_and_delete_preserve_boundary(make_config) -> None:
    memory = MemoryStore(make_config())
    first_project_id = "project-a"
    second_project_id = "project-b"
    knowledge = GlobalKnowledgeBase(memory, allow_mutation=True)
    shared_id = knowledge.add(
        kind="Knowledge",
        title="portable knowledge",
        content="globally-discoverable-marker from every project",
    )
    knowledge.add(kind="Lesson", title="portable lesson", content="also visible from every project")

    assert [
        item.id for item in memory.search("globally-discoverable-marker", first_project_id, record_usage=False)
    ] == [shared_id]
    assert [
        item.id for item in memory.search("globally-discoverable-marker", second_project_id, record_usage=False)
    ] == [shared_id]
    assert [item.kind for item in knowledge.list(kinds=["Knowledge"])] == [MemoryKind.KNOWLEDGE]
    assert knowledge.list(kinds=[]) == []
    with pytest.raises(ValueError, match="global knowledge kind"):
        knowledge.list(kinds=["Bug"])

    updated = knowledge.update(shared_id, content="updated once for every project", tags=["shared"])
    assert updated.content == "updated once for every project"
    assert memory.search("updated once", first_project_id, record_usage=False)[0].id == shared_id
    assert knowledge.delete(shared_id) is True
    assert memory.search("updated once", second_project_id, record_usage=False) == []


def test_global_knowledge_is_add_only_by_default_and_scoped_mutation_fails_closed(make_config) -> None:
    memory = MemoryStore(make_config())
    add_only = GlobalKnowledgeBase(memory)
    shared_id = add_only.add(kind="Knowledge", title="shared", content="append-only default")

    with pytest.raises(PermissionError, match="add-only by default"):
        add_only.update(shared_id, content="not without explicit maintenance")
    with pytest.raises(PermissionError, match="add-only by default"):
        add_only.delete(shared_id)
    assert add_only.get(shared_id).content == "append-only default"

    hidden_child_id = memory.add_memory(
        kind="Knowledge",
        title="project child",
        content="must not be deleted through the global facade",
        project_id="project-a",
    )
    with memory._connect() as con:
        con.execute("update memories set merged_into = ? where id = ?", (shared_id, hidden_child_id))

    maintenance = GlobalKnowledgeBase(memory, allow_mutation=True)
    assert maintenance.delete(shared_id) is False
    assert memory.get_memory(shared_id) is not None
    assert memory.get_memory(hidden_child_id) is not None

    merged_global_id = memory.add_memory(
        kind="Knowledge",
        title="merged global",
        content="merged records are not active knowledge",
        project_id=None,
    )
    with memory._connect() as con:
        con.execute("update memories set merged_into = ? where id = ?", (shared_id, merged_global_id))
    assert maintenance.get(merged_global_id) is None

    with pytest.raises(ValueError, match="must be a boolean"):
        GlobalKnowledgeBase(memory, allow_mutation="false")  # type: ignore[arg-type]
