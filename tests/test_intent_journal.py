from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from agent.constants import MAX_INTENT_JOURNAL_ENTRIES
from agent.exceptions import IntentJournalError
from agent.intent_journal import IntentJournal
from agent.project import ProjectManager
from agent.session import SessionManager
from agent.state import AgentState, PlanStep


def _state(tmp_path: Path, make_config) -> tuple[AgentState, SessionManager]:
    config = make_config()
    root = tmp_path / "project"
    root.mkdir()
    project = ProjectManager(config).resolve_project(root)
    state = AgentState.create(
        session_id="intent-session",
        project=project,
        user_request=(
            "Build the report with DEEPSEEK_API_KEY=TOP_SECRET_VALUE and Authorization: Bearer PRIVATE_BEARER_123456."
        ),
        loaded_memories=[],
        loaded_tools=[],
        git_branch=None,
        context_index_path=str(project.agent_dir / "index.json"),
    )
    state.plan = [PlanStep("build", "Build", status="in_progress", step_type="implement")]
    state.current_step = "build"
    if state.execution_context:
        state.execution_context.current_plan_id = "build"
    return state, SessionManager(project)


def test_session_checkpoint_appends_hash_chained_redacted_intent_metadata(tmp_path: Path, make_config) -> None:
    state, sessions = _state(tmp_path, make_config)
    state.start()
    messages = [{"role": "user", "content": state.user_request}]

    sessions.checkpoint(state, messages)
    state.round = 1
    state.record_checkpoint(phase="running", message_count=1)
    sessions.checkpoint(state, messages)

    entries = sessions.intent_journal.read(state.session_id)
    rendered = json.dumps(entries, ensure_ascii=False)
    assert len(entries) == 2
    assert entries[1]["previous_hash"] == entries[0]["entry_hash"]
    assert entries[1]["plan"] == [{"id": "build", "status": "in_progress", "step_type": "implement"}]
    assert "TOP_SECRET_VALUE" not in rendered
    assert "PRIVATE_BEARER_123456" not in rendered
    assert state.intent_journal_head["sequence"] == 2
    assert state.intent_journal_head["path"].endswith("intent-session.intent.jsonl")


def test_intent_journal_refuses_symbolic_link_targets(tmp_path: Path, make_config) -> None:
    state, sessions = _state(tmp_path, make_config)
    target = tmp_path / "outside.jsonl"
    target.write_text("", encoding="utf-8")
    journal_path = sessions.intent_journal.directory / f"{state.session_id}.intent.jsonl"
    journal_path.symlink_to(target)

    with pytest.raises(IntentJournalError, match="regular file"):
        sessions.intent_journal.append(state)


def test_intent_journal_read_rejects_invalid_lines(tmp_path: Path) -> None:
    journal = IntentJournal(tmp_path / "journal")
    path = journal.directory / "broken.intent.jsonl"
    path.write_text("not-json\n", encoding="utf-8")

    with pytest.raises(IntentJournalError, match="invalid JSON"):
        journal.read("broken")


def test_intent_journal_verifies_every_hash_chain_link(tmp_path: Path, make_config) -> None:
    state, sessions = _state(tmp_path, make_config)
    sessions.intent_journal.append(state)
    sessions.intent_journal.append(state)
    path = sessions.intent_journal.directory / f"{state.session_id}.intent.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["objective"] = "tampered objective"
    lines[0] = json.dumps(first, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(IntentJournalError, match="hash chain"):
        sessions.intent_journal.read(state.session_id, limit=1)
    with pytest.raises(IntentJournalError, match="hash chain"):
        sessions.intent_journal.append(state)


def test_intent_journal_serializes_concurrent_appends(tmp_path: Path, make_config) -> None:
    state, sessions = _state(tmp_path, make_config)

    with ThreadPoolExecutor(max_workers=4) as pool:
        heads = list(pool.map(lambda _index: sessions.intent_journal.append(state), range(12)))

    entries = sessions.intent_journal.read(state.session_id)
    assert len(entries) == 12
    assert [item["sequence"] for item in entries] == list(range(1, 13))
    assert {item["sequence"] for item in heads} == set(range(1, 13))


def test_intent_journal_projects_complex_lineage_below_line_limit(tmp_path: Path, make_config) -> None:
    state, sessions = _state(tmp_path, make_config)
    state.plan = [
        PlanStep(
            f"step-{index:03d}-" + "界" * 20,
            f"Step {index}",
            status="in_progress" if index == 0 else "pending",
            step_type="implement",
        )
        for index in range(128)
    ]
    state.current_step = state.plan[0].id
    state.artifact_registry = {
        "schema_version": 1,
        "artifacts": {
            f"reports/{index:03d}-" + "产物" * 40 + ".docx": {
                "kind": "file",
                "state": "verified",
                "snapshot_id": f"snapshot-{index}-" + "快照" * 30,
                "step_ids": [step.id for step in state.plan[:32]],
                "parent_artifacts": [f"parents/{parent:02d}-" + "父级" * 80 for parent in range(32)],
            }
            for index in range(128)
        },
    }

    head = sessions.intent_journal.append(state)
    path = sessions.intent_journal.directory / f"{state.session_id}.intent.jsonl"
    raw_line = path.read_bytes().splitlines()[0]
    entries = sessions.intent_journal.read(state.session_id)

    assert head["sequence"] == 1
    assert len(raw_line) <= sessions.intent_journal.MAX_LINE_BYTES
    assert len(entries) == 1
    assert entries[0]["plan_projection"]["total"] == 128
    assert entries[0]["plan_projection"]["omitted"] > 0
    assert entries[0]["external_impacts_projection"]["total"] == 128
    assert entries[0]["external_impacts_projection"]["included"] > 0
    assert entries[0]["external_impacts_projection"]["omitted"] > 0


def test_intent_journal_rotates_at_entry_limit_and_keeps_checkpointing(tmp_path: Path, make_config) -> None:
    state, sessions = _state(tmp_path, make_config)

    for sequence in range(1, MAX_INTENT_JOURNAL_ENTRIES):
        state.round = sequence
        sessions.intent_journal.append(state)

    with ThreadPoolExecutor(max_workers=4) as pool:
        heads = list(pool.map(lambda _index: sessions.intent_journal.append(state), range(4)))

    entries = sessions.intent_journal.read(state.session_id, limit=MAX_INTENT_JOURNAL_ENTRIES)
    path = sessions.intent_journal.directory / f"{state.session_id}.intent.jsonl"

    assert sorted(item["sequence"] for item in heads) == list(
        range(MAX_INTENT_JOURNAL_ENTRIES, MAX_INTENT_JOURNAL_ENTRIES + 4)
    )
    assert len(path.read_bytes().splitlines()) == 4
    assert [item["event"] for item in entries] == ["projection", "checkpoint", "checkpoint", "checkpoint"]
    assert entries[0]["sequence"] == MAX_INTENT_JOURNAL_ENTRIES
    assert entries[0]["projection"]["compacted_through_sequence"] == MAX_INTENT_JOURNAL_ENTRIES
    assert entries[0]["projection"]["prior_head_hash"] == entries[0]["previous_hash"]
    assert entries[1]["sequence"] == MAX_INTENT_JOURNAL_ENTRIES + 1
    assert entries[1]["previous_hash"] == entries[0]["entry_hash"]

    state.round += 1
    next_head = sessions.intent_journal.append(state)
    assert next_head["sequence"] == MAX_INTENT_JOURNAL_ENTRIES + 4
    assert sessions.intent_journal.read(state.session_id)[-1]["sequence"] == MAX_INTENT_JOURNAL_ENTRIES + 4

    lines = path.read_text(encoding="utf-8").splitlines()
    projection = json.loads(lines[0])
    projection["projection"]["prior_head_hash"] = "0" * 64
    lines[0] = json.dumps(projection, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(IntentJournalError, match="hash chain"):
        sessions.intent_journal.append(state)
