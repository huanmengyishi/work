from __future__ import annotations

import json

import pytest

from agent.constants import (
    MAX_AGENT_STATE_BYTES,
    MAX_AGENT_STATE_KEY_BYTES,
    MAX_AGENT_STATE_MAPPING_KEYS,
    MAX_AGENT_STATE_RECORD_BYTES,
    MAX_TOOL_CALLS_IN_STATE,
)
from agent.exceptions import SessionInconsistencyError
from agent.project import ProjectManager
from agent.session import SessionManager
from agent.state import AgentState


def _state(tmp_path, make_config) -> AgentState:
    config = make_config()
    root = tmp_path / "project"
    root.mkdir()
    project = ProjectManager(config).resolve_project(root)
    return AgentState.create(
        session_id="window-test",
        project=project,
        user_request="Keep long tasks bounded.",
        loaded_memories=[],
        loaded_tools=[],
        git_branch=None,
        context_index_path=str(project.agent_dir / "index.json"),
    )


def test_tool_calls_keep_a_bounded_hot_window_and_aggregate_old_evidence(tmp_path, make_config) -> None:
    state = _state(tmp_path, make_config)
    state.start()

    for index in range(250):
        state.round = index + 1
        state.record_tool_call(
            {
                "tool": "template",
                "action": "read_file" if index % 2 else "search_code",
                "capability": "template.read_file" if index % 2 else "template.search_code",
                "request_id": f"request-{index}",
                "args": {"path": f"src/{index}.py"},
            },
            {
                "success": index % 10 != 0,
                "stderr": "failed" if index % 10 == 0 else "",
                "duration_ms": 1,
                "request_id": f"request-{index}",
            },
        )

    assert len(state.tool_calls) == MAX_TOOL_CALLS_IN_STATE
    assert state.tool_calls[0]["type"] == "pruned_history"
    assert state.tool_history_summary["count"] == 51
    assert state.tool_history_summary["success_count"] + state.tool_history_summary["failure_count"] == 51
    assert state.tool_calls[-1]["request"]["request_id"] == "request-249"
    assert state.resume_checkpoint["last_request_id"] == "request-249"
    assert state.resume_checkpoint["tool_call_count"] == MAX_TOOL_CALLS_IN_STATE - 1
    assert state.resume_checkpoint["pruned_tool_call_count"] == 51
    assert state.resume_checkpoint["tool_call_count"] + state.resume_checkpoint["pruned_tool_call_count"] == 250
    state.record_checkpoint(phase="running", message_count=1)
    assert state.resume_checkpoint["tool_call_count"] == MAX_TOOL_CALLS_IN_STATE - 1
    assert AgentState.from_dict(state.to_dict()).tool_history_summary == state.tool_history_summary


def test_load_for_resume_keeps_first_objective_and_last_sixteen_complete_rounds(tmp_path, make_config) -> None:
    state = _state(tmp_path, make_config)
    sessions = SessionManager(ProjectManager(make_config()).resolve_project(tmp_path / "project"))
    messages = [
        {"role": "system", "content": "system policy"},
        {"role": "user", "content": "original objective"},
    ]
    for index in range(20):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call-{index}",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": f"call-{index}", "content": f"result-{index}"},
            ]
        )
    sessions.checkpoint(state, messages)

    record = sessions.load_for_resume(state.session_id, max_rounds=16)

    assistant = [item for item in record.messages if item.get("role") == "assistant"]
    tools = [item for item in record.messages if item.get("role") == "tool"]
    assert len(assistant) == 16
    assert len(tools) == 16
    assert record.messages[0]["content"] == "system policy"
    assert record.messages[1]["content"] == "original objective"
    assert "4 model round(s)" in record.messages[2]["content"]
    assert tools[0]["tool_call_id"] == "call-4"
    assert tools[-1]["tool_call_id"] == "call-19"


def test_resume_streams_versioned_message_store_without_loading_full_transcript(tmp_path, make_config) -> None:
    state = _state(tmp_path, make_config)
    sessions = SessionManager(ProjectManager(make_config()).resolve_project(tmp_path / "project"))
    old_body = "cold-history-marker-" + "x" * 50_000
    messages = [{"role": "system", "content": "policy"}, {"role": "user", "content": "objective"}]
    for index in range(20):
        messages.append({"role": "assistant", "content": old_body if index < 4 else f"round-{index}"})
    session_path = sessions.checkpoint(state, messages)
    manifest = json.loads(session_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == SessionManager.PAYLOAD_SCHEMA_VERSION
    assert manifest["messages"]["format"] == "jsonl"
    assert old_body not in session_path.read_text(encoding="utf-8")

    def forbidden_full_load(*_args, **_kwargs):
        raise AssertionError("Resume must not materialize the complete transcript")

    sessions._load_messages = forbidden_full_load  # type: ignore[method-assign]
    record = sessions.load_for_resume(state.session_id, max_rounds=16)

    assert old_body not in "\n".join(str(item.get("content") or "") for item in record.messages)
    assert [item.get("content") for item in record.messages if item.get("role") == "assistant"][0] == "round-4"


def test_session_message_store_is_hash_verified_and_previous_generation_is_recoverable(
    tmp_path,
    make_config,
) -> None:
    state = _state(tmp_path, make_config)
    sessions = SessionManager(ProjectManager(make_config()).resolve_project(tmp_path / "project"))
    session_path = sessions.checkpoint(state, [{"role": "user", "content": "first"}])
    first_manifest = json.loads(session_path.read_text(encoding="utf-8"))["messages"]
    first_store = sessions.session_dir / first_manifest["path"]
    assert first_store.exists()

    sessions.checkpoint(state, [{"role": "user", "content": "second"}])
    payload = json.loads(session_path.read_text(encoding="utf-8"))
    second_manifest = payload["messages"]
    second_store = sessions.session_dir / second_manifest["path"]
    assert second_store.exists()
    assert first_store.exists()
    assert [item["path"] for item in payload["cold_messages"]] == [first_manifest["path"]]
    assert sessions.load_cold_messages(state.session_id) == [[{"content": "first", "role": "user"}]]

    first_store.write_bytes(first_store.read_bytes().replace(b"first", b"tampr"))
    with pytest.raises(SessionInconsistencyError, match="manifest"):
        sessions.load_cold_messages(state.session_id)

    second_store.write_bytes(second_store.read_bytes().replace(b"second", b"tamper"))
    with pytest.raises(SessionInconsistencyError, match="manifest"):
        sessions.load(state.session_id)


def test_cold_session_generations_rotate_by_count_without_unbounded_files(tmp_path, make_config) -> None:
    state = _state(tmp_path, make_config)
    sessions = SessionManager(ProjectManager(make_config()).resolve_project(tmp_path / "project"))
    stores = []

    for index in range(SessionManager.MAX_COLD_MESSAGE_GENERATIONS + 4):
        session_path = sessions.checkpoint(state, [{"role": "user", "content": f"generation-{index}"}])
        payload = json.loads(session_path.read_text(encoding="utf-8"))
        stores.append(sessions.session_dir / payload["messages"]["path"])

    cold = payload["cold_messages"]
    referenced = {payload["messages"]["path"], *(item["path"] for item in cold)}
    actual = {item.name for item in sessions.session_dir.glob(f"{state.session_id}.*.messages.jsonl")}
    assert len(cold) == SessionManager.MAX_COLD_MESSAGE_GENERATIONS
    assert sum(item["bytes"] for item in cold) <= SessionManager.MAX_COLD_MESSAGE_BYTES
    assert actual == referenced
    assert stores[-2].name in referenced
    assert not stores[0].exists()


def test_resume_checkpoint_preserves_complete_cold_generation_and_remains_resumable(tmp_path, make_config) -> None:
    state = _state(tmp_path, make_config)
    sessions = SessionManager(ProjectManager(make_config()).resolve_project(tmp_path / "project"))
    messages = [{"role": "system", "content": "policy"}, {"role": "user", "content": "objective"}]
    messages.extend({"role": "assistant", "content": f"round-{index}"} for index in range(20))
    session_path = sessions.checkpoint(state, messages)
    complete_manifest = json.loads(session_path.read_text(encoding="utf-8"))["messages"]
    complete_store = sessions.session_dir / complete_manifest["path"]

    resumed = sessions.load_for_resume(state.session_id, max_rounds=16)
    resumed.state.resume("continue")
    projected_messages = [*resumed.messages, {"role": "user", "content": "continue"}]
    sessions.checkpoint(resumed.state, projected_messages)
    payload = json.loads(session_path.read_text(encoding="utf-8"))

    assert complete_store.exists()
    assert any(
        item["path"] == complete_manifest["path"] and item["complete"] is True for item in payload["cold_messages"]
    )
    assert payload["messages"]["complete"] is False
    assert len([item for item in sessions.load(state.session_id).messages if item.get("role") == "assistant"]) == 16
    assert (
        len([item for item in sessions.load_cold_messages(state.session_id)[0] if item.get("role") == "assistant"])
        == 20
    )
    assert sessions.load_for_resume(state.session_id, max_rounds=16).messages == projected_messages


def test_legacy_inline_session_migrates_without_losing_previous_messages(tmp_path, make_config) -> None:
    state = _state(tmp_path, make_config)
    sessions = SessionManager(ProjectManager(make_config()).resolve_project(tmp_path / "project"))
    legacy_state = state.to_dict()
    legacy_state["schema_version"] = 1
    legacy_messages = [{"role": "user", "content": "legacy objective"}]
    session_path = sessions.session_dir / f"{state.session_id}.json"
    session_path.write_text(
        json.dumps({"schema_version": 1, "state": legacy_state, "messages": legacy_messages}),
        encoding="utf-8",
    )

    record = sessions.load(state.session_id)
    sessions.checkpoint(record.state, [{"role": "user", "content": "current projection"}])
    payload = json.loads(session_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == SessionManager.PAYLOAD_SCHEMA_VERSION
    assert len(payload["cold_messages"]) == 1
    assert sessions.load_cold_messages(state.session_id) == [legacy_messages]


def test_resume_checkpoint_locator_updates_without_message_bodies(tmp_path, make_config) -> None:
    state = _state(tmp_path, make_config)

    state.record_checkpoint(phase="running", message_count=42)

    assert state.resume_checkpoint["turn"] == 1
    assert state.resume_checkpoint["round"] == 0
    assert state.resume_checkpoint["message_count"] == 42
    assert "content" not in state.resume_checkpoint


def test_agent_state_rejects_oversized_record_dictionary_keys_and_total_utf8_bytes(tmp_path, make_config) -> None:
    state = _state(tmp_path, make_config)
    payload = state.to_dict()
    payload["tool_calls"] = [
        {
            "turn": 1,
            "round": 0,
            "request": {},
            "result": {"content": "界" * (MAX_AGENT_STATE_RECORD_BYTES // 3 + 1)},
        }
    ]
    with pytest.raises(ValueError, match=r"tool_calls\[0\].*UTF-8 byte limit"):
        AgentState.from_dict(payload)

    payload = state.to_dict()
    payload["convergence"] = {"界" * (MAX_AGENT_STATE_KEY_BYTES // 3 + 1): True}
    with pytest.raises(ValueError, match="dictionary key exceeds"):
        AgentState.from_dict(payload)

    payload = state.to_dict()
    payload["convergence"] = {f"key-{index}": index for index in range(MAX_AGENT_STATE_MAPPING_KEYS + 1)}
    with pytest.raises(ValueError, match="dictionary exceeds.*key limit"):
        AgentState.from_dict(payload)

    payload = state.to_dict()
    payload["final_answer"] = "界" * (MAX_AGENT_STATE_BYTES // 3)
    with pytest.raises(ValueError, match="serialized UTF-8 byte limit"):
        AgentState.from_dict(payload)
    state.final_answer = payload["final_answer"]
    with pytest.raises(ValueError, match="serialized UTF-8 byte limit"):
        state.to_dict()


def test_legacy_session_load_fails_closed_before_migrating_oversized_state(tmp_path, make_config) -> None:
    state = _state(tmp_path, make_config)
    sessions = SessionManager(ProjectManager(make_config()).resolve_project(tmp_path / "project"))
    payload = state.to_dict()
    payload["schema_version"] = 1
    payload["final_answer"] = "界" * (MAX_AGENT_STATE_BYTES // 3)
    session_path = sessions.session_dir / f"{state.session_id}.json"
    session_path.write_text(
        json.dumps({"schema_version": 1, "state": payload, "messages": []}, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="AgentState input exceeds.*serialized UTF-8 byte limit"):
        sessions.load(state.session_id)
