from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent.memory_transfer as memory_transfer_module
from agent import cli
from agent.memory import MemoryStore
from agent.memory_transfer import MemoryTransferError, export_memory, import_memory


def test_versioned_export_import_round_trip_scopes_and_deduplicates(tmp_path: Path, make_config) -> None:
    source = MemoryStore(make_config())
    project_id = "source-project"
    project_memory_id = source.add_memory(
        kind="Knowledge",
        title="project portable",
        content="\nproject payload\n",
        tags=["portable"],
        project_id=project_id,
    )
    global_memory_id = source.add_memory(
        kind="Decision",
        title="global portable",
        content="global payload",
        project_id=None,
    )
    source.add_memory(
        kind="Knowledge",
        title="other private",
        content="must not be exported",
        project_id="another-project",
    )
    destination = tmp_path / "memory-export.json"

    exported = export_memory(source, destination, project_id=project_id, scope="both")
    document = json.loads(destination.read_text(encoding="utf-8"))

    assert exported["record_count"] == 2
    assert document["format"] == "deep-agent-memory"
    assert document["version"] == 1
    assert document["selection"] == {"scope": "both", "project_id": project_id}
    assert {record["scope"] for record in document["records"]} == {"project", "global"}
    assert {record["title"] for record in document["records"]} == {"project portable", "global portable"}
    assert project_memory_id != global_memory_id
    assert destination.stat().st_mode & 0o777 == 0o600

    target = MemoryStore(
        make_config(
            {
                "memory": {
                    "sqlite_path": str(tmp_path / "target" / "memory.db"),
                    "vector_path": str(tmp_path / "target" / "vector"),
                }
            }
        )
    )
    assert target.search("portable", "target-project", record_usage=False) == []  # populate the hot cache
    imported = import_memory(target, destination, project_id="target-project")

    assert imported["inserted_count"] == 2
    project_items = target.list_memories(project_id="target-project", limit=10)
    assert {item.title for item in project_items} == {"project portable", "global portable"}
    assert next(item for item in project_items if item.title == "project portable").content == "\nproject payload\n"
    assert {item.title for item in target.list_memories(global_only=True)} == {"global portable"}
    assert target.search("portable", "target-project", record_usage=False)

    repeated = import_memory(target, destination, project_id="target-project")
    assert repeated["inserted_count"] == 0
    assert repeated["deduplicated_count"] == 2
    assert target.stats(project_id="target-project").total == 2


def test_import_conflict_strategy_is_explicit_skip_or_replace(tmp_path: Path, make_config) -> None:
    source = MemoryStore(make_config())
    source.add_memory(
        kind="Knowledge",
        title="same title",
        content="new portable content",
        tags=["new"],
        project_id="source",
    )
    path = tmp_path / "conflict.json"
    export_memory(source, path, project_id="source")

    target = MemoryStore(
        make_config(
            {
                "memory": {
                    "sqlite_path": str(tmp_path / "conflict-target" / "memory.db"),
                    "vector_path": str(tmp_path / "conflict-target" / "vector"),
                }
            }
        )
    )
    existing_id = target.add_memory(
        kind="Knowledge",
        title="same title",
        content="old local content",
        tags=["old"],
        project_id="target",
    )
    assert target.search("old local", "target", record_usage=False)[0].id == existing_id

    skipped = import_memory(target, path, project_id="target", conflict="skip")
    assert skipped["skipped_conflict_count"] == 1
    assert target.get_memory(existing_id).content == "old local content"

    replaced = import_memory(target, path, project_id="target", conflict="replace")
    assert replaced["replaced_count"] == 1
    assert target.get_memory(existing_id).content == "new portable content"
    assert target.search("old local", "target", record_usage=False) == []
    assert target.search("new portable", "target", record_usage=False)[0].id == existing_id


def test_import_replace_never_crosses_project_or_global_scope(tmp_path: Path, make_config) -> None:
    source = MemoryStore(make_config())
    source.add_memory(
        kind="Knowledge",
        title="same title",
        content="incoming project content",
        project_id="source",
    )
    path = tmp_path / "scoped-conflict.json"
    export_memory(source, path, project_id="source")

    target = MemoryStore(
        make_config(
            {
                "memory": {
                    "sqlite_path": str(tmp_path / "scoped-target" / "memory.db"),
                    "vector_path": str(tmp_path / "scoped-target" / "vector"),
                }
            }
        )
    )
    other_id = target.add_memory(
        kind="Knowledge",
        title="same title",
        content="other project content",
        project_id="other-project",
    )
    global_id = target.add_memory(
        kind="Knowledge",
        title="same title",
        content="global content",
        project_id=None,
    )

    result = import_memory(target, path, project_id="target-project", conflict="replace")

    assert result["inserted_count"] == 1
    assert result["replaced_count"] == 0
    assert target.get_memory(other_id).content == "other project content"
    assert target.get_memory(global_id).content == "global content"
    assert target.list_memories(project_id="target-project", kind="Knowledge")[0].content == (
        "incoming project content"
    )


def test_import_validates_complete_document_before_writing(tmp_path: Path, make_config) -> None:
    memory = MemoryStore(make_config())
    path = tmp_path / "invalid.json"
    document = {
        "format": "deep-agent-memory",
        "version": 1,
        "exported_at": datetime.now(UTC).isoformat(),
        "selection": {"scope": "project", "project_id": "source"},
        "records": [
            {
                "scope": "project",
                "kind": "Knowledge",
                "title": "valid first",
                "content": "must not be partially imported",
                "tags": [],
                "confidence": 0.7,
                "expires_at": None,
            },
            {
                "scope": "project",
                "kind": "TypoKind",
                "title": "invalid second",
                "content": "strict kind validation",
                "tags": [],
                "confidence": 0.7,
                "expires_at": None,
            },
        ],
    }
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(MemoryTransferError, match="invalid kind"):
        import_memory(memory, path, project_id="target")

    assert memory.stats().total == 0


def test_import_enforces_record_field_and_path_limits(tmp_path: Path, make_config) -> None:
    memory = MemoryStore(
        make_config(
            {
                "memory": {
                    "transfer": {
                        "max_records": 1,
                        "max_path_chars": 64,
                        "max_title_chars": 5,
                        "max_title_bytes": 20,
                    }
                }
            }
        )
    )
    record = {
        "scope": "project",
        "kind": "Knowledge",
        "title": "short",
        "content": "bounded",
        "tags": [],
        "confidence": 0.7,
        "expires_at": None,
    }
    document = {
        "format": "deep-agent-memory",
        "version": 1,
        "exported_at": datetime.now(UTC).isoformat(),
        "selection": {"scope": "project", "project_id": "source"},
        "records": [record, record],
    }
    path = tmp_path / "limits.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    # The temporary directory itself can exceed a deliberately tiny path
    # policy, so use the real path limit for document validation first.
    memory.config.values["memory"]["transfer"]["max_path_chars"] = 4_096

    with pytest.raises(MemoryTransferError, match="configured limit is 1"):
        import_memory(memory, path, project_id="target")

    document["records"] = [{**record, "title": "sixsix"}]
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(MemoryTransferError, match="title exceeds"):
        import_memory(memory, path, project_id="target")

    memory.config.values["memory"]["transfer"]["max_path_chars"] = 64
    with pytest.raises(MemoryTransferError, match="path is empty or exceeds"):
        import_memory(memory, "x" * 65, project_id="target")
    assert memory.stats().total == 0


def test_export_preflights_record_count_and_payload_before_materializing(tmp_path: Path, make_config) -> None:
    count_limited = MemoryStore(
        make_config(
            {
                "memory": {
                    "transfer": {
                        "max_records": 1,
                    }
                }
            }
        )
    )
    count_limited.add_memory(kind="Knowledge", title="one", content="first", project_id="p")
    count_limited.add_memory(kind="Knowledge", title="two", content="second", project_id="p")
    with pytest.raises(MemoryTransferError, match="configured 1 record limit"):
        export_memory(count_limited, tmp_path / "too-many.json", project_id="p")

    payload_limited = MemoryStore(
        make_config(
            {
                "memory": {
                    "sqlite_path": str(tmp_path / "payload" / "memory.db"),
                    "vector_path": str(tmp_path / "payload" / "vector"),
                    "transfer": {
                        "max_file_bytes": 1_024,
                    },
                }
            }
        )
    )
    payload_limited.add_memory(
        kind="Knowledge",
        title="large payload",
        content="x" * 2_000,
        project_id="p",
    )
    with pytest.raises(MemoryTransferError, match="configured file limit is 1024 bytes"):
        export_memory(payload_limited, tmp_path / "too-large.json", project_id="p")


def test_transfer_rejects_unsafe_paths_oversize_and_overwrite(tmp_path: Path, make_config) -> None:
    memory = MemoryStore(make_config({"memory": {"transfer": {"max_file_bytes": 1024}}}))
    memory.add_memory(kind="Knowledge", title="safe", content="bounded", project_id="p")
    path = tmp_path / "memory.json"
    export_memory(memory, path, project_id="p")

    with pytest.raises(MemoryTransferError, match="pass --force"):
        export_memory(memory, path, project_id="p")
    export_memory(memory, path, project_id="p", overwrite=True)

    symlink = tmp_path / "memory-link.json"
    symlink.symlink_to(path)
    with pytest.raises(MemoryTransferError, match="symbolic-link"):
        import_memory(memory, symlink, project_id="p")
    with pytest.raises(MemoryTransferError, match="symbolic-link"):
        export_memory(memory, symlink, project_id="p", overwrite=True)

    real_directory = tmp_path / "real-directory"
    real_directory.mkdir()
    linked_directory = tmp_path / "linked-directory"
    linked_directory.symlink_to(real_directory, target_is_directory=True)
    with pytest.raises(MemoryTransferError, match="must not contain symbolic links"):
        export_memory(memory, linked_directory / "export.json", project_id="p")

    oversize = tmp_path / "oversize.json"
    oversize.write_bytes(b" " * 1025)
    with pytest.raises(MemoryTransferError, match="configured limit"):
        import_memory(memory, oversize, project_id="p")


def test_atomic_export_no_overwrite_uses_exclusive_install(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "exclusive.json"
    real_link = os.link

    def race_link(source, destination, **kwargs):
        Path(destination).write_text("concurrent writer", encoding="utf-8")
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(memory_transfer_module.os, "link", race_link)

    with pytest.raises(MemoryTransferError, match="appeared during export"):
        memory_transfer_module._atomic_write_private(target, b"private export\n", overwrite=False)

    assert target.read_text(encoding="utf-8") == "concurrent writer"


def test_import_normalizes_path_races_invalid_unicode_and_huge_json_numbers(
    tmp_path: Path,
    make_config,
) -> None:
    memory = MemoryStore(make_config())
    invalid_unicode = tmp_path / "invalid-unicode.json"
    document = {
        "format": "deep-agent-memory",
        "version": 1,
        "exported_at": datetime.now(UTC).isoformat(),
        "selection": {"scope": "project", "project_id": "source"},
        "records": [
            {
                "scope": "project",
                "kind": "Knowledge",
                "title": "\ud800",
                "content": "bounded",
                "tags": [],
                "confidence": 0.7,
                "expires_at": None,
            }
        ],
    }
    invalid_unicode.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(MemoryTransferError, match="valid Unicode"):
        import_memory(memory, invalid_unicode, project_id="target")

    huge_number = tmp_path / "huge-number.json"
    huge_number.write_text('{"version":' + "9" * 5_000 + "}", encoding="utf-8")
    with pytest.raises(MemoryTransferError, match="not valid bounded JSON"):
        import_memory(memory, huge_number, project_id="target")


def test_concurrent_import_is_serialized_and_deduplicated(tmp_path: Path, make_config) -> None:
    source = MemoryStore(make_config())
    source.add_memory(
        kind="Knowledge",
        title="concurrent portable",
        content="one logical payload",
        project_id="source",
    )
    path = tmp_path / "concurrent.json"
    export_memory(source, path, project_id="source")
    target = MemoryStore(
        make_config(
            {
                "memory": {
                    "sqlite_path": str(tmp_path / "concurrent-target" / "memory.db"),
                    "vector_path": str(tmp_path / "concurrent-target" / "vector"),
                }
            }
        )
    )

    with ThreadPoolExecutor(max_workers=8) as executor:
        reports = list(
            executor.map(
                lambda _index: import_memory(target, path, project_id="target"),
                range(16),
            )
        )

    assert sum(report["inserted_count"] for report in reports) == 1
    assert sum(report["deduplicated_count"] for report in reports) == 15
    assert target.stats(project_id="target").total == 1


def test_memory_export_import_and_cleanup_cli(tmp_path: Path, make_config, monkeypatch, capsys) -> None:
    memory = MemoryStore(make_config())
    project = SimpleNamespace(id="cli-project", name="cli-project")
    global_id = memory.add_memory(
        kind="Knowledge",
        title="CLI portable",
        content="CLI portable content",
        project_id=None,
    )
    monkeypatch.setattr(cli, "prepare_project", lambda _config: (project, memory))
    parser = cli.build_command_parser()
    path = tmp_path / "cli-memory.json"

    export_args = parser.parse_args(["memory", "export", os.fspath(path), "--scope", "global"])
    assert cli.cmd_memory(memory.config, export_args) == 0
    assert json.loads(capsys.readouterr().out)["record_count"] == 1
    memory.delete_memory(global_id)

    import_args = parser.parse_args(["memory", "import", os.fspath(path), "--target-scope", "global"])
    assert cli.cmd_memory(memory.config, import_args) == 0
    assert json.loads(capsys.readouterr().out)["inserted_count"] == 1
    assert memory.list_memories(global_only=True)[0].title == "CLI portable"

    search_args = parser.parse_args(["memory", "search", "CLI portable", "--global-only"])
    assert cli.cmd_memory(memory.config, search_args) == 0
    assert "CLI portable" in capsys.readouterr().out
    stats_args = parser.parse_args(["memory", "stats"])
    assert cli.cmd_memory(memory.config, stats_args) == 0
    assert "total: 1" in capsys.readouterr().out

    expired_id = memory.add_memory(
        kind="Reflection",
        title="CLI cleanup",
        content="expired",
        confidence=0.1,
        expires_at=(datetime.now(UTC) - timedelta(days=1)).isoformat(),
        project_id="cli-project",
    )
    preview_args = parser.parse_args(["memory", "cleanup"])
    assert cli.cmd_memory(memory.config, preview_args) == 0
    assert memory.get_memory(expired_id) is not None
    capsys.readouterr()
    apply_args = parser.parse_args(["memory", "cleanup", "--apply"])
    assert cli.cmd_memory(memory.config, apply_args) == 0
    assert memory.get_memory(expired_id) is None


def test_memory_search_cli_reports_bounded_input_errors(make_config, monkeypatch, capsys) -> None:
    memory = MemoryStore(
        make_config(
            {
                "memory": {
                    "search_max_query_chars": 8,
                    "search_max_query_bytes": 32,
                }
            }
        )
    )
    project = SimpleNamespace(id="cli-project", name="cli-project")
    monkeypatch.setattr(cli, "prepare_project", lambda _config: (project, memory))
    args = cli.build_command_parser().parse_args(["memory", "search", "too-long-query"])

    assert cli.cmd_memory(memory.config, args) == 2
    assert "exceeds the configured" in capsys.readouterr().err
