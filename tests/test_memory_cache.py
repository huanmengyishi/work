from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Event

import agent.memory as memory_module
import pytest
from agent.memory import MemoryStore


def test_query_cache_is_ttl_lru_bounded_and_copies_results(make_config, monkeypatch) -> None:
    clock = [100.0]
    monkeypatch.setattr(memory_module, "monotonic", lambda: clock[0])
    memory = MemoryStore(
        make_config(
            {
                "memory": {
                    "query_cache_max_entries": 2,
                    "query_cache_ttl_seconds": 10,
                }
            }
        )
    )
    for value in ("alpha", "bravo", "charlie"):
        memory.add_memory(kind="Knowledge", title=value, content=f"{value} cache value")

    original = memory._search_uncached
    calls: list[str] = []

    def counted(query: str, **kwargs):
        calls.append(query)
        return original(query, **kwargs)

    monkeypatch.setattr(memory, "_search_uncached", counted)
    first = memory.search("alpha", record_usage=False)
    first[0].tags.append("caller-only")
    assert memory.search("alpha", record_usage=False)[0].tags == []
    memory.search("bravo", record_usage=False)
    memory.search("alpha", record_usage=False)  # alpha is now most recently used
    memory.search("charlie", record_usage=False)
    memory.search("bravo", record_usage=False)  # bravo was evicted
    assert calls == ["alpha", "bravo", "charlie", "bravo"]
    assert memory.query_cache_info()["entries"] == 2

    clock[0] += 11
    memory.search("alpha", record_usage=False)
    assert calls[-1] == "alpha"
    assert memory.query_cache_info()["max_entries"] == 2


def test_query_cache_is_thread_safe_and_hard_capped_at_128(make_config) -> None:
    memory = MemoryStore(
        make_config(
            {
                "memory": {
                    "query_cache_max_entries": 10_000,
                    "query_cache_ttl_seconds": 60,
                }
            }
        )
    )
    for index in range(16):
        memory.add_memory(kind="Knowledge", title=f"item-{index}", content=f"thread-cache-{index}")

    with ThreadPoolExecutor(max_workers=12) as executor:
        results = list(
            executor.map(
                lambda index: memory.search(f"thread-cache-{index % 16}", record_usage=False),
                range(512),
            )
        )

    assert all(len(items) == 1 for items in results)
    info = memory.query_cache_info()
    assert info["max_entries"] == 128
    assert 0 < info["entries"] <= 128
    assert info["hits"] > 0


def test_all_memory_mutations_invalidate_or_refresh_hot_queries(make_config) -> None:
    config = make_config(
        {
            "memory": {
                "dedupe_similarity": 0.9,
                "max_items": 1,
                "max_storage_mb": 100,
                "capacity_scan_limit": 100,
            }
        }
    )
    memory = MemoryStore(config)
    first_id = memory.add_memory(kind="Knowledge", title="needle", content="first needle", project_id="p")
    assert [item.id for item in memory.search("needle", "p", record_usage=False)] == [first_id]

    second_id = memory.add_memory(kind="Knowledge", title="needle two", content="second needle", project_id="p")
    assert {item.id for item in memory.search("needle", "p", record_usage=False)} == {first_id, second_id}

    memory.update_memory(first_id, title="renamed", content="renamed content")
    assert [item.id for item in memory.search("needle", "p", record_usage=False)] == [second_id]
    memory.record_feedback("cache-feedback", second_id, helpful=True)
    assert memory.search("needle", "p", record_usage=False)[0].confidence == 0.72
    memory.delete_memory(second_id)
    assert memory.search("needle", "p", record_usage=False) == []

    duplicate_a = memory.add_memory(kind="Lesson", title="same lesson", content="same content", project_id="p")
    duplicate_b = memory.add_memory(kind="Lesson", title="same lesson", content="same content", project_id="p")
    assert {item.id for item in memory.search("same lesson", "p", record_usage=False)} == {
        duplicate_a,
        duplicate_b,
    }
    report = memory.maintain(project_id="p", apply=True)
    assert report["merge_count"] == 1
    assert len(memory.search("same lesson", "p", record_usage=False)) == 1

    expired_at = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    expired_id = memory.add_memory(
        kind="Reflection",
        title="expired cache item",
        content="expired cache content",
        confidence=0.1,
        expires_at=expired_at,
        project_id="p",
    )
    assert memory.search("expired cache", "p", record_usage=False)[0].id == expired_id
    memory.maintain(project_id="p", apply=True)
    assert memory.search("expired cache", "p", record_usage=False) == []

    capacity_id = memory.add_memory(
        kind="Reflection",
        title="capacity cache item",
        content="capacity cache content",
        confidence=0.0,
        project_id="p",
    )
    assert memory.search("capacity cache", "p", record_usage=False)[0].id == capacity_id
    capacity = memory.maintain_capacity(project_id="p", apply=True)
    assert capacity_id in capacity["deleted_ids"]
    assert memory.search("capacity cache", "p", record_usage=False) == []


def test_cached_usage_metadata_tracks_the_database(make_config) -> None:
    memory = MemoryStore(make_config())
    memory_id = memory.add_memory(kind="Lesson", title="usage cache", content="usage cache content")

    first = memory.search("usage cache")
    second = memory.search("usage cache")

    assert first[0].use_count == 0
    assert second[0].use_count == 1
    assert memory.get_memory(memory_id).use_count == 2


def test_vector_supplement_is_rechecked_against_project_scope(make_config, monkeypatch) -> None:
    memory = MemoryStore(make_config())
    own_id = memory.add_memory(
        kind="Knowledge",
        title="own vector result",
        content="own-vector-only",
        project_id="project-a",
    )
    foreign_id = memory.add_memory(
        kind="Knowledge",
        title="foreign vector result",
        content="foreign-vector-only",
        project_id="project-b",
    )
    monkeypatch.setattr(memory.vector, "is_enabled", lambda: True)
    monkeypatch.setattr(
        memory.vector,
        "query_memory_ids",
        lambda **_kwargs: [foreign_id, own_id],
    )

    result = memory.search("vector-miss", "project-a", limit=8, record_usage=False)

    assert [item.id for item in result] == [own_id]


def test_concurrent_write_prevents_a_stale_miss_from_entering_the_cache(make_config, monkeypatch) -> None:
    memory = MemoryStore(make_config())
    first_id = memory.add_memory(
        kind="Knowledge",
        title="generation first",
        content="generation race marker",
        project_id="p",
    )
    uncached_read_finished = Event()
    allow_cache_store = Event()
    original = memory._search_uncached

    def delayed_search(query: str, **kwargs):
        items = original(query, **kwargs)
        uncached_read_finished.set()
        assert allow_cache_store.wait(timeout=5)
        return items

    monkeypatch.setattr(memory, "_search_uncached", delayed_search)
    with ThreadPoolExecutor(max_workers=2) as executor:
        stale_reader = executor.submit(
            memory.search,
            "generation race marker",
            "p",
            record_usage=False,
        )
        assert uncached_read_finished.wait(timeout=5)
        second_id = memory.add_memory(
            kind="Knowledge",
            title="generation second",
            content="generation race marker",
            project_id="p",
        )
        allow_cache_store.set()
        assert [item.id for item in stale_reader.result(timeout=5)] == [first_id]

    monkeypatch.setattr(memory, "_search_uncached", original)
    assert {item.id for item in memory.search("generation race marker", "p", record_usage=False)} == {
        first_id,
        second_id,
    }


def test_search_query_is_bounded_and_fts_metacharacters_are_plain_text(make_config) -> None:
    memory = MemoryStore(
        make_config(
            {
                "memory": {
                    "search_max_query_chars": 16,
                    "search_max_query_bytes": 32,
                }
            }
        )
    )
    memory_id = memory.add_memory(kind="Knowledge", title="foo bar", content="plain searchable text")

    assert [item.id for item in memory.search('foo"bar', record_usage=False)] == [memory_id]
    assert [item.id for item in memory.search("foo\x00bar", record_usage=False)] == [memory_id]
    with pytest.raises(ValueError, match="configured 16 character or 32 UTF-8 byte limit"):
        memory.search("x" * 17, record_usage=False)
    assert memory.search("foo " * 20, record_usage=False, truncate_query=True)[0].id == memory_id
    assert memory._bounded_search_query("界" * 20, truncate=True) == "界" * 10
    with pytest.raises(ValueError, match="valid Unicode"):
        memory.search("\ud800", record_usage=False)
