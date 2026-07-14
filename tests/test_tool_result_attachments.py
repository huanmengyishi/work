from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.deepseek import ChatResponse
from agent.convergence import TaskConvergenceController
from agent.memory import MemoryStore
from agent.project import Project, ProjectManager
from agent.runtime import AgentRuntime
from agent.state import AgentState
from agent.tools import ToolManager, ToolResult
from agent.tools.registry import ToolCapability
from agent.tools.result_store import ToolResultStore, ToolResultStoreError


class RecordingClient:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.requests: list[list[dict]] = []

    def chat(self, *, messages, **_kwargs) -> ChatResponse:
        self.requests.append(list(messages))
        if not self.responses:
            raise AssertionError("response queue exhausted")
        return ChatResponse(message=self.responses.pop(0), raw={})


def _state(project: Project, session_id: str) -> AgentState:
    return AgentState.create(
        session_id=session_id,
        project=project,
        user_request="inspect one large tool result",
        loaded_memories=[],
        loaded_tools=[],
        git_branch=None,
        context_index_path=str(project.agent_dir / "index.json"),
    )


def _manager(root: Path, make_config, overrides: dict | None = None) -> tuple[Project, ToolManager]:
    config = make_config(overrides)
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    tools = ToolManager(config, project, memory, yolo=True)
    return project, tools


def _register_large_tool(tools: ToolManager, content: str) -> None:
    tools.registry.register(
        ToolCapability(
            "test",
            "large",
            "test_large",
            "Return deterministic large test evidence.",
            {},
            permissions=("read",),
        ),
        lambda: ToolResult(True, content, data={"kind": "test-evidence"}),
    )


def test_project_initializes_private_ignored_tool_result_directory(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()

    project = ProjectManager(make_config()).resolve_project(root)
    result_dir = project.agent_dir / "tool-results"

    assert result_dir.is_dir()
    assert stat.S_IMODE(result_dir.stat().st_mode) == 0o700
    assert "tool-results/" in (project.agent_dir / ".gitignore").read_text(encoding="utf-8").splitlines()


def test_large_result_is_private_hashed_and_returned_as_bounded_preview(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    marker = "PRIVATE-MIDDLE-EVIDENCE"
    full_output = "HEAD\n" + ("a" * 8_000) + marker + ("z" * 8_000) + "\nTAIL"
    project, tools = _manager(
        root,
        make_config,
        {
            "tools": {
                "tool_result": {
                    "persist_threshold_bytes": 1_024,
                    "max_attachment_bytes": 64_000,
                    "preview_chars": 1_024,
                }
            }
        },
    )
    state = _state(project, "attachment-session")
    tools.bind_state(state)
    _register_large_tool(tools, full_output)

    request, result = tools.execute_model_call("test_large", {}, request_id="large-request")

    assert result.success is True
    assert len(result.stdout) <= 768
    assert marker not in result.stdout
    attachment = result.data["attachment"]
    path = tools.result_store.path_for_test(session_id=state.session_id, request_id=request.request_id)
    payload = path.read_bytes()
    stored = json.loads(payload)
    assert stored["stdout"] == full_output
    assert attachment["bytes"] == len(payload)
    assert attachment["sha256"] == hashlib.sha256(payload).hexdigest()
    assert attachment["source_truncated"] is False
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert path.name == hashlib.sha256(request.request_id.encode()).hexdigest() + ".json"


def test_small_result_does_not_create_attachment(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    project, tools = _manager(root, make_config)
    tools.bind_state(_state(project, "small-session"))
    _register_large_tool(tools, "small result")

    _, result = tools.execute_model_call("test_large", {}, request_id="small-request")

    assert result.stdout == "small result"
    assert "attachment" not in (result.data or {})
    assert not tools.result_store.path_for_test(
        session_id="small-session",
        request_id="small-request",
    ).exists()


def test_runtime_session_and_model_receive_preview_while_attachment_keeps_full_result(
    tmp_path: Path,
    make_config,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    marker = "RUNTIME-PRIVATE-MIDDLE"
    (root / "evidence.txt").write_text(
        "HEAD" + ("a" * 8_000) + marker + ("z" * 8_000) + "TAIL\n",
        encoding="utf-8",
    )
    config = make_config(
        {
            "runtime": {"task_mode": "simple", "max_tool_rounds": 2},
            "tools": {
                "tool_result": {
                    "persist_threshold_bytes": 1_024,
                    "max_attachment_bytes": 64_000,
                    "preview_chars": 1_024,
                }
            },
        }
    )
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    tools = ToolManager(config, project, memory, yolo=True)
    client = RecordingClient(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "runtime-large",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": "evidence.txt", "start_line": 1, "end_line": 2}),
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "The bounded evidence was inspected."},
        ]
    )
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=tools,
        client=client,
    )

    assert runtime.run("What is in evidence.txt?") == "The bounded evidence was inspected."

    state = runtime.sessions.load(str(runtime.last_session_id)).state
    recorded = state.tool_calls[0]["result"]
    assert marker not in recorded["stdout"]
    assert len(recorded["stdout"]) <= 768
    assert recorded["data"]["attachment"]["request_id"] == "runtime-large"
    tool_message = next(item for item in client.requests[1] if item.get("role") == "tool")
    assert marker not in str(tool_message["content"])
    assert len(str(tool_message["content"])) <= 12_000
    session_json = (project.agent_dir / "sessions" / f"{state.session_id}.json").read_text(encoding="utf-8")
    assert marker not in session_json
    attachment_path = tools.result_store.path_for_test(
        session_id=state.session_id,
        request_id="runtime-large",
    )
    assert marker in attachment_path.read_text(encoding="utf-8")


def test_resume_bound_manager_reads_only_current_session_in_bounded_chunks(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    full_output = "start-" + ("中" * 5_000) + "-end"
    project, tools = _manager(
        root,
        make_config,
        {
            "tools": {
                "tool_result": {
                    "persist_threshold_bytes": 512,
                    "max_attachment_bytes": 64_000,
                    "max_read_chars": 256,
                }
            }
        },
    )
    original_state = _state(project, "resume-session")
    tools.bind_state(original_state)
    _register_large_tool(tools, full_output)
    _, large = tools.execute_model_call("test_large", {}, request_id="resume-target")
    attachment_path = tools.result_store.path_for_test(
        session_id=original_state.session_id,
        request_id="resume-target",
    )
    serialized = attachment_path.read_text(encoding="utf-8")
    offset = serialized.index("中" * 20)
    original_hash = large.data["attachment"]["sha256"]

    resumed = ToolManager(tools.config, project, tools.memory, yolo=True)
    resumed.bind_state(AgentState.from_dict(original_state.to_dict()))
    _, chunk = resumed.execute_model_call(
        "tool_result_read",
        {"request_id": "resume-target", "offset": offset, "max_chars": 64},
        request_id="reader-call",
    )

    assert chunk.success is True
    assert chunk.stdout.startswith("中" * 20)
    assert len(chunk.stdout) == 64
    assert chunk.data["next_offset"] == offset + 64
    assert chunk.data["sha256"] == original_hash
    assert hashlib.sha256(attachment_path.read_bytes()).hexdigest() == original_hash
    assert not resumed.result_store.path_for_test(
        session_id=original_state.session_id,
        request_id="reader-call",
    ).exists()

    resumed.bind_state(_state(project, "different-session"))
    _, cross_session = resumed.execute_model_call(
        "tool_result_read",
        {"request_id": "resume-target", "offset": 0, "max_chars": 64},
        request_id="cross-reader",
    )
    assert cross_session.success is False
    assert "does not exist" in cross_session.stderr


def test_attachment_reader_rejects_unbounded_arguments_and_exposes_no_path_parameter(
    tmp_path: Path,
    make_config,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    project, tools = _manager(root, make_config)
    tools.bind_state(_state(project, "reader-bounds"))
    capability, _ = tools.registry.resolve("tool_result_read")
    assert capability is not None
    assert set(capability.properties) == {"request_id", "offset", "max_chars"}

    _, negative = tools.execute_model_call(
        "tool_result_read",
        {"request_id": "missing", "offset": -1, "max_chars": 10},
    )
    _, oversized = tools.execute_model_call(
        "tool_result_read",
        {"request_id": "missing", "offset": 0, "max_chars": tools.result_store.max_read_chars + 1},
    )
    _, long_id = tools.execute_model_call(
        "tool_result_read",
        {"request_id": "x" * 513, "offset": 0, "max_chars": 10},
    )

    assert negative.success is False and "non-negative" in negative.stderr
    assert oversized.success is False and "between 1" in oversized.stderr
    assert long_id.success is False and "1-512" in long_id.stderr


@pytest.mark.parametrize("attack", ["root-symlink", "session-symlink", "result-symlink", "result-directory"])
def test_attachment_store_rejects_unsafe_paths_without_rewriting_handler_result(
    tmp_path: Path,
    make_config,
    attack: str,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_target = outside / "target.json"
    outside_target.write_text("unchanged", encoding="utf-8")
    project, tools = _manager(
        root,
        make_config,
        {"tools": {"tool_result": {"persist_threshold_bytes": 512, "max_attachment_bytes": 16_000}}},
    )
    state = _state(project, "secure-session")
    tools.bind_state(state)
    _register_large_tool(tools, "x" * 4_000)
    result_root = project.agent_dir / "tool-results"
    session_dir = result_root / state.session_id
    result_path = tools.result_store.path_for_test(session_id=state.session_id, request_id="secure-request")

    if attack == "root-symlink":
        shutil.rmtree(result_root)
        result_root.symlink_to(outside, target_is_directory=True)
    elif attack == "session-symlink":
        session_dir.symlink_to(outside, target_is_directory=True)
    else:
        session_dir.mkdir(mode=0o700)
        if attack == "result-symlink":
            result_path.symlink_to(outside_target)
        else:
            result_path.mkdir()

    _, result = tools.execute_model_call("test_large", {}, request_id="secure-request")

    assert result.success is True
    assert 0 < len(result.stdout) <= tools.result_store.preview_chars * 3 // 4
    assert result.data["kind"] == "test-evidence"
    persistence_error = result.data["attachment_persistence_error"]
    assert persistence_error["type"] == "ToolResultStoreError"
    assert persistence_error["result_preserved"] is True
    assert persistence_error["full_body_available"] is False
    assert persistence_error["stdout_chars"] == 4_000
    assert outside_target.read_text(encoding="utf-8") == "unchanged"


@pytest.mark.parametrize("failure_mode", ["quota", "duplicate-request-id"])
def test_attachment_failure_preserves_side_effect_truth_and_never_triggers_retry(
    tmp_path: Path,
    make_config,
    failure_mode: str,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    project, tools = _manager(
        root,
        make_config,
        {
            "tools": {
                "tool_result": {
                    "persist_threshold_bytes": 512,
                    "max_attachment_bytes": 16_000,
                    "preview_chars": 1_024,
                    "max_attachments_per_session": 1,
                }
            }
        },
    )
    state = _state(project, f"preserve-{failure_mode}")
    tools.bind_state(state)
    request_id = "side-effect-request"
    seed_id = "seed" if failure_mode == "quota" else request_id
    tools.result_store.persist(
        ToolResult(True, "seed-head\n" + ("s" * 2_000) + "\nseed-tail", request_id=seed_id),
        session_id=state.session_id,
        request_id=seed_id,
    )

    effect_count = 0
    marker = root / "effect-count.txt"

    def side_effect() -> ToolResult:
        nonlocal effect_count
        effect_count += 1
        marker.write_text(str(effect_count), encoding="utf-8")
        return ToolResult(
            True,
            "EFFECT-HEAD\n" + ("a" * 3_500) + "PRIVATE-MIDDLE" + ("z" * 3_500) + "\nEFFECT-TAIL",
            "ORIGINAL-WARNING",
            data={"kind": "side-effect"},
        )

    tools.registry.register(
        ToolCapability(
            "test",
            "side_effect",
            "test_side_effect",
            "Perform one deterministic approved side effect.",
            {},
            permissions=("write",),
        ),
        side_effect,
    )

    _, result = tools.execute_model_call("test_side_effect", {}, request_id=request_id)
    # This models a caller that retries only a reported failure. The truthful
    # success result must prevent the already-applied side effect from running
    # a second time when attachment storage alone failed.
    if not result.success:
        _, result = tools.execute_model_call("test_side_effect", {}, request_id=request_id)

    assert effect_count == 1
    assert marker.read_text(encoding="utf-8") == "1"
    assert result.success is True
    assert result.stdout.startswith("EFFECT-HEAD")
    assert result.stdout.endswith("EFFECT-TAIL")
    assert "PRIVATE-MIDDLE" not in result.stdout
    assert result.stderr == "ORIGINAL-WARNING"
    assert result.data["kind"] == "side-effect"
    persistence_error = result.data["attachment_persistence_error"]
    assert persistence_error["type"] == "ToolResultStoreError"
    assert persistence_error["result_preserved"] is True
    assert persistence_error["full_body_available"] is False
    assert persistence_error["stdout_chars"] > len(result.stdout)
    assert len(json.dumps(result.to_dict(), ensure_ascii=False)) < 3_000


def test_store_enforces_per_session_count_and_total_bytes(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    project = ProjectManager(make_config()).resolve_project(root)
    store = ToolResultStore(
        project.agent_dir,
        max_attachment_bytes=4_096,
        persist_threshold_bytes=512,
        preview_chars=512,
        max_read_chars=256,
        max_attachments_per_session=1,
        max_session_bytes=4_096,
    )
    first = ToolResult(True, "a" * 1_000, request_id="first")
    second = ToolResult(True, "b" * 1_000, request_id="second")

    stored = store.persist(first, session_id="quota-session", request_id="first")
    with pytest.raises(ToolResultStoreError, match="count exceeds"):
        store.persist(second, session_id="quota-session", request_id="second")

    first_path = store.path_for_test(session_id="quota-session", request_id="first")
    before = first_path.read_bytes()
    replay = store.persist(first, session_id="quota-session", request_id="first")
    assert first_path.read_bytes() == before
    assert replay.data["attachment"]["sha256"] == stored.data["attachment"]["sha256"]

    byte_limited = ToolResultStore(
        project.agent_dir,
        max_attachment_bytes=2_048,
        persist_threshold_bytes=512,
        preview_chars=512,
        max_read_chars=256,
        max_attachments_per_session=10,
        max_session_bytes=2_048,
    )
    byte_limited.persist(first, session_id="byte-quota", request_id="first")
    with pytest.raises(ToolResultStoreError, match="attachments exceed"):
        byte_limited.persist(second, session_id="byte-quota", request_id="second")


def test_store_reports_known_source_truncation_and_rejects_request_id_conflict(
    tmp_path: Path,
    make_config,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    project = ProjectManager(make_config()).resolve_project(root)
    store = ToolResultStore(
        project.agent_dir,
        max_attachment_bytes=1_024,
        persist_threshold_bytes=512,
        preview_chars=512,
        max_read_chars=256,
        max_attachments_per_session=10,
        max_session_bytes=16_000,
    )
    result = ToolResult(
        True,
        "x" * 4_000,
        data={
            "source_truncated": True,
            "source_original_bytes": 20_000,
            "source_original_bytes_known": True,
        },
        request_id="bounded-source",
    )

    stored = store.persist(result, session_id="bounded-session", request_id="bounded-source")
    metadata = stored.data["attachment"]
    assert metadata["bytes"] <= 1_024
    assert metadata["original_serialized_bytes"] > 1_024
    assert metadata["source_truncated"] is True
    assert metadata["source_original_bytes"] == 20_000
    with pytest.raises(ToolResultStoreError, match="different stored tool result"):
        store.persist(
            ToolResult(True, "different" * 1_000, request_id="bounded-source"),
            session_id="bounded-session",
            request_id="bounded-source",
        )


def test_secure_open_rejects_tampered_attachment_symlink(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    project, tools = _manager(
        root,
        make_config,
        {"tools": {"tool_result": {"persist_threshold_bytes": 512, "max_attachment_bytes": 16_000}}},
    )
    state = _state(project, "read-symlink")
    tools.bind_state(state)
    _register_large_tool(tools, "x" * 4_000)
    tools.execute_model_call("test_large", {}, request_id="read-target")
    path = tools.result_store.path_for_test(session_id=state.session_id, request_id="read-target")
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"request_id": "read-target"}), encoding="utf-8")
    path.unlink()
    os.symlink(outside, path)

    _, result = tools.execute_model_call(
        "tool_result_read",
        {"request_id": "read-target", "offset": 0, "max_chars": 32},
    )

    assert result.success is False
    assert "regular non-symlink" in result.stderr


def test_attachment_reads_count_as_targeted_read_only_exploration_and_close_in_hard_phase() -> None:
    state = SimpleNamespace(plan=[], convergence={}, turn=1, tool_calls=[])
    controller = TaskConvergenceController(
        mode="large",
        max_rounds=8,
        exploration_round_limit=2,
        reserved_rounds=2,
    )
    controller.bind(state)
    request = {
        "tool": "tool_result",
        "action": "read",
        "args": {"request_id": "large-result", "offset": 0, "max_chars": 256},
    }

    assert controller.observe_round(state, [request], [{"success": True}]) is False
    assert controller.observe_round(state, [request], [{"success": True}]) is False
    assert controller.consecutive_read_only_rounds == 2
    assert controller.low_yield_rounds == 1

    action = controller.before_round(7, state)
    assert "tool_result_read" in action.excluded_functions
    schemas = [
        {"function": {"name": "tool_result_read"}},
        {"function": {"name": "run_tests"}},
    ]
    assert controller.filter_schemas(schemas, action.excluded_functions) == [schemas[1]]


def test_hard_phase_allows_only_bounded_reads_of_validation_attachments() -> None:
    validation_call = {
        "request": {
            "tool": "template",
            "action": "run_tests",
            "request_id": "validation-result",
            "args": {"framework": "npm:typecheck", "path": "."},
        },
        "result": {
            "success": False,
            "data": {"attachment": {"request_id": "validation-result", "bytes": 620_000}},
        },
    }
    state = SimpleNamespace(
        plan=[SimpleNamespace(id="implement", status="in_progress")],
        convergence={},
        turn=1,
        tool_calls=[validation_call],
    )
    controller = TaskConvergenceController(
        mode="deep",
        max_rounds=24,
        exploration_round_limit=10,
        reserved_rounds=8,
        implementation_read_limit=0,
        validation_attachment_read_limit=2,
    )
    controller.bind(state)

    first_action = controller.before_round(17, state)

    assert first_action.guard_validation_attachment_read is True
    assert "tool_result_read" not in first_action.excluded_functions
    assert "read_file" in first_action.excluded_functions
    assert any("validation attachment" in message for message in first_action.messages)
    valid = {"request_id": "validation-result", "offset": 0, "max_chars": 12_000}
    assert controller.validation_attachment_read_denial(state, "tool_result_read", valid) == ""
    assert state.convergence["validation_attachment_reads_used"] == 1
    assert (
        controller.validation_attachment_read_denial(
            state,
            "tool_result_read",
            {"request_id": "validation-result", "offset": 12_000, "max_chars": 12_000},
        )
        == ""
    )
    assert "exhausted" in controller.validation_attachment_read_denial(
        state,
        "tool_result_read",
        valid,
    )
    exhausted_action = controller.before_round(18, state)
    assert exhausted_action.guard_validation_attachment_read is False
    assert "tool_result_read" in exhausted_action.excluded_functions


@pytest.mark.parametrize(
    ("tool", "action", "arguments", "expected"),
    [
        (
            "template",
            "read_file",
            {"request_id": "attachment", "offset": 0, "max_chars": 1_000},
            "not an attachment produced",
        ),
        (
            "template",
            "run_tests",
            {"request_id": "unknown", "offset": 0, "max_chars": 1_000},
            "not an attachment produced",
        ),
        (
            "template",
            "run_tests",
            {"request_id": "attachment", "offset": 0, "max_chars": 12_001},
            "max_chars must be between",
        ),
    ],
)
def test_validation_attachment_exception_rejects_nonvalidation_unknown_or_oversized_reads(
    tool: str,
    action: str,
    arguments: dict,
    expected: str,
) -> None:
    state = SimpleNamespace(
        plan=[SimpleNamespace(id="verify", status="in_progress")],
        convergence={},
        turn=1,
        tool_calls=[
            {
                "request": {
                    "tool": tool,
                    "action": action,
                    "request_id": "attachment",
                    "args": {"path": "src/app.ts"},
                },
                "result": {"success": False, "data": {"attachment": {"request_id": "attachment"}}},
            }
        ],
    )
    controller = TaskConvergenceController(
        mode="deep",
        max_rounds=24,
        exploration_round_limit=10,
        reserved_rounds=8,
        implementation_read_limit=0,
        validation_attachment_read_limit=2,
    )
    controller.bind(state)

    denial = controller.validation_attachment_read_denial(state, "tool_result_read", arguments)

    assert expected in denial
    assert controller.validation_attachment_reads_used == 0
