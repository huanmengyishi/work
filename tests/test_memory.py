from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from agent.memory import MemoryKind, MemoryStore
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


def test_memory_kind_is_canonical_for_new_writes_but_legacy_unknown_kind_is_readable(make_config) -> None:
    memory = MemoryStore(make_config())

    memory_id = memory.add_memory(kind="lesson", title="canonical", content="Known kinds are canonicalized.")

    item = memory.get_memory(memory_id)
    assert item is not None
    assert item.kind is MemoryKind.LESSON
    with pytest.raises(ValueError, match="unknown memory kind"):
        memory.add_memory(kind="Leson", title="typo", content="This must not be persisted.")

    # Simulate a kind written by an older unrestricted release.  Reads remain
    # lossless even though new writes are now fail closed.
    with sqlite3.connect(memory.db_path) as con:
        con.execute("update memories set kind = 'LegacyCustom' where id = ?", (memory_id,))
    legacy = memory.get_memory(memory_id)
    assert legacy is not None
    assert legacy.kind == "LegacyCustom"
    assert memory.stats().by_kind == {"LegacyCustom": 1}


def test_capacity_maintenance_is_dry_run_by_default_and_protects_durable_kinds(make_config) -> None:
    config = make_config(
        {
            "memory": {
                "max_items": 3,
                "max_storage_mb": 100,
                "capacity_scan_limit": 10,
                "protect_kinds": [],
            }
        }
    )
    memory = MemoryStore(config)
    weak_id = memory.add_memory(kind="Lesson", title="弱", content="低置信度", confidence=0.1, project_id="p")
    strong_id = memory.add_memory(kind="Knowledge", title="strong", content="retain", confidence=0.9, project_id="p")
    correction_id = memory.add_memory(
        kind="Correction", title="protected correction", content="retain", confidence=0.0, project_id="p"
    )
    decision_id = memory.add_memory(
        kind="Decision", title="protected decision", content="retain", confidence=0.0, project_id="p"
    )

    preview = memory.maintain_capacity(project_id="p")

    expected_payload_bytes = 0
    for memory_id in (weak_id, strong_id, correction_id, decision_id):
        item = memory.get_memory(memory_id)
        assert item is not None
        expected_payload_bytes += sum(
            len(value.encode("utf-8"))
            for value in (
                str(item.kind),
                item.title,
                item.content,
                json.dumps(item.tags, ensure_ascii=False),
            )
        )
    assert preview["apply"] is False
    assert preview["current_count"] == 4
    assert preview["current_payload_bytes"] == expected_payload_bytes
    assert preview["eviction_ids"] == [weak_id]
    assert preview["projected_count"] == 3
    assert memory.get_memory(weak_id) is not None

    legacy_apply = memory.maintain(project_id="p", apply=True)
    assert legacy_apply["capacity"]["apply"] is False
    assert memory.get_memory(weak_id) is not None

    applied = memory.maintain_capacity(project_id="p", apply=True)

    assert applied["deleted_ids"] == [weak_id]
    assert applied["complete"] is True
    assert memory.get_memory(weak_id) is None
    assert memory.get_memory(strong_id) is not None
    assert memory.get_memory(correction_id) is not None
    assert memory.get_memory(decision_id) is not None


def test_capacity_candidate_scan_is_bounded_and_reports_unresolved_excess(make_config) -> None:
    memory = MemoryStore(
        make_config(
            {
                "memory": {
                    "max_items": 1,
                    "max_storage_mb": 100,
                    "capacity_scan_limit": 1,
                }
            }
        )
    )
    for index in range(3):
        memory.add_memory(kind="Lesson", title=f"lesson-{index}", content="bounded scan", confidence=0.2)

    preview = memory.maintain_capacity()

    assert preview["scanned"] == 1
    assert preview["eviction_count"] == 1
    assert preview["unresolved_count"] == 1
    assert preview["complete"] is False
    assert memory.stats().total == 3


def test_explicit_feedback_is_idempotent_bounded_and_usage_does_not_change_confidence(make_config) -> None:
    memory = MemoryStore(make_config())
    memory_id = memory.add_memory(kind="Lesson", title="feedback", content="Explicit evidence only.", confidence=0.7)

    memory.record_usage([memory_id])
    used = memory.get_memory(memory_id)
    assert used is not None
    assert used.use_count == 1
    assert used.confidence == pytest.approx(0.7)

    assert memory.record_feedback("feedback-1", memory_id, helpful=True) is True
    assert memory.get_memory(memory_id).confidence == pytest.approx(0.72)
    assert memory.record_feedback("feedback-1", memory_id, helpful=True) is False
    assert memory.get_memory(memory_id).confidence == pytest.approx(0.72)
    with pytest.raises(ValueError, match="different evidence"):
        memory.record_feedback("feedback-1", memory_id, helpful=False)

    assert memory.record_feedback("feedback-2", memory_id, helpful=False) is True
    assert memory.get_memory(memory_id).confidence == pytest.approx(0.57)
    with sqlite3.connect(memory.db_path) as con:
        assert con.execute("select count(*) from memory_feedback_events").fetchone()[0] == 2

    upper_id = memory.add_memory(kind="Knowledge", title="upper", content="bound", confidence=0.94)
    lower_id = memory.add_memory(kind="Knowledge", title="lower", content="bound", confidence=0.12)
    assert memory.record_feedback("upper-bound", upper_id, helpful=True) is True
    assert memory.record_feedback("lower-bound", lower_id, helpful=False) is True
    assert memory.get_memory(upper_id).confidence == pytest.approx(0.95)
    assert memory.get_memory(lower_id).confidence == pytest.approx(0.1)


def test_explicit_feedback_uses_configured_bounds_without_reversing_existing_confidence(make_config) -> None:
    memory = MemoryStore(
        make_config(
            {
                "memory": {
                    "confidence": {
                        "use_bonus": 0.2,
                        "contradiction_penalty": 0.3,
                        "lower_bound": 0.25,
                        "upper_bound": 0.8,
                    }
                }
            }
        )
    )
    upper_id = memory.add_memory(kind="Knowledge", title="upper", content="bound", confidence=0.75)
    lower_id = memory.add_memory(kind="Knowledge", title="lower", content="bound", confidence=0.4)
    existing_high_id = memory.add_memory(kind="Decision", title="existing", content="high", confidence=0.9)

    memory.record_feedback("configured-upper", upper_id, helpful=True)
    memory.record_feedback("configured-lower", lower_id, helpful=False)
    memory.record_feedback("existing-high", existing_high_id, helpful=True)

    assert memory.get_memory(upper_id).confidence == pytest.approx(0.8)
    assert memory.get_memory(lower_id).confidence == pytest.approx(0.25)
    assert memory.get_memory(existing_high_id).confidence == pytest.approx(0.9)
