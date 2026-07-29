from __future__ import annotations

import copy
import json
from pathlib import Path

from agent import convergence
from agent.config import DEFAULT_CONFIG
from agent.context_window import repair_tool_message_pairs
from agent.context_window import ContextWindowController as SplitContextWindowController
from agent.history_compaction import ToolHistoryCompactor as SplitToolHistoryCompactor
from agent.history_snip import HistorySnipper
from agent.memory import MemoryStore
from agent.model_router import ModelRoute
from agent.project import ProjectManager
from agent.runtime import AgentRuntime
from agent.state import AgentState
from agent.tools import ToolManager


def tool_round(
    index: int,
    *,
    name: str = "read_file",
    path: str | None = None,
    result: str = "ok",
    call_count: int = 1,
) -> list[dict]:
    calls = []
    results = []
    for call_index in range(call_count):
        call_id = f"round-{index}-call-{call_index}"
        calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(
                        {"path": path or f"src/file-{index}-{call_index}.py"},
                        separators=(",", ":"),
                    ),
                },
            }
        )
        results.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": f"{result}:{call_index}",
            }
        )
    return [{"role": "assistant", "content": None, "tool_calls": calls}, *results]


def retained_call_ids(messages: list[dict]) -> list[str]:
    return [
        str(call["id"])
        for message in messages
        if message.get("role") == "assistant"
        for call in message.get("tool_calls") or []
    ]


def marker_contents(messages: list[dict]) -> list[str]:
    return [
        str(message.get("content") or "")
        for message in messages
        if message.get("role") == "system"
        and str(message.get("content") or "").startswith("[Deep Agent snipped complete API rounds]")
    ]


class NoModelCallClient:
    def __init__(self) -> None:
        self.call_count = 0

    def chat(self, **_kwargs):
        self.call_count += 1
        raise AssertionError("history snip must not call the model")


class OrderingContextWindow(SplitContextWindowController):
    history_snip_seen_before_reasoning_compaction = False

    def compact_old_reasoning(self, messages: list[dict]) -> int:
        self.history_snip_seen_before_reasoning_compaction = bool(marker_contents(messages))
        return super().compact_old_reasoning(messages)


def runtime_fixture(tmp_path: Path, make_config, overrides: dict | None = None):
    root = tmp_path / "project"
    root.mkdir()
    config = make_config(overrides)
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    client = NoModelCallClient()
    progress: list[dict] = []
    runtime = AgentRuntime.with_default_services(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
        progress_handler=progress.append,
    )
    state = AgentState.create(
        session_id="history-snip-runtime",
        project=project,
        user_request="bounded runtime integration objective",
        loaded_memories=[],
        loaded_tools=[],
        git_branch=None,
        context_index_path=str(project.agent_dir / "index.json"),
    )
    return runtime, state, client, progress


def test_convergence_module_keeps_compatible_split_reexports() -> None:
    assert convergence.ContextWindowController is SplitContextWindowController
    assert convergence.ToolHistoryCompactor is SplitToolHistoryCompactor
    assert convergence.repair_tool_message_pairs is repair_tool_message_pairs
    assert convergence.ContextWindowController.__module__ == "agent.context_window"
    assert convergence.ToolHistoryCompactor.__module__ == "agent.history_compaction"
    assert convergence.TaskConvergenceController.__module__ == "agent.convergence"


def test_history_snip_default_config_is_enabled_with_bounded_thresholds() -> None:
    configured = DEFAULT_CONFIG["runtime"]["convergence"]

    assert configured["history_snip_enabled"] is True
    assert configured["history_snip_min_chars"] == 24_000
    assert configured["history_snip_min_complete_rounds"] == 8
    assert configured["history_snip_keep_recent_rounds"] == 4
    assert configured["history_snip_marker_chars"] == 768


def test_runtime_snips_after_pair_repair_before_compaction_without_model_call(
    tmp_path: Path,
    make_config,
    monkeypatch,
) -> None:
    runtime, state, client, progress = runtime_fixture(tmp_path, make_config)
    messages: list[dict] = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "work"},
        {"role": "tool", "tool_call_id": "orphan", "content": "discard me"},
    ]
    for index in range(8):
        messages.extend(tool_round(index, result="x" * 3_500))

    context_window = OrderingContextWindow(
        context_window_tokens=8_192,
        safety_buffer_tokens=1_024,
        keep_recent_rounds=4,
        failure_limit=3,
    )
    context_window.bind(state)
    model_route = ModelRoute(
        provider="deepseek",
        tier="standard",
        model="deepseek-v4-pro",
        thinking_enabled=False,
        reasoning_effort=None,
        max_tokens=1_024,
        reasons=("test",),
    )
    compactor = SplitToolHistoryCompactor(
        aggregate_chars=1_000_000,
        output_reserve_chars=0,
        compacted_result_chars=256,
        keep_recent_results=4,
        failure_limit=3,
    )
    original_compact = compactor.compact
    compactor_observations: list[bool] = []

    def observe_compactor(compactor_messages):
        compactor_observations.append(bool(marker_contents(compactor_messages)))
        return original_compact(compactor_messages)

    def forbid_auto_compaction(*_args, **_kwargs):
        raise AssertionError("snip should reduce the request before model-backed AutoCompact")

    monkeypatch.setattr(compactor, "compact", observe_compactor)
    monkeypatch.setattr(runtime, "_auto_compact_context", forbid_auto_compaction)

    runtime._prepare_model_request(
        state,
        messages,
        tools=None,
        model_route=model_route,
        context_window=context_window,
        history_compactor=compactor,
        auto_compaction_enabled=True,
        auto_compaction_max_tokens=512,
        phase="tool_loop",
        checkpoint=False,
    )

    assert retained_call_ids(messages) == [f"round-{index}-call-0" for index in range(4, 8)]
    assert len(marker_contents(messages)) == 1
    assert len(marker_contents(messages)[0]) <= 768
    assert repair_tool_message_pairs(messages).changed is False
    assert context_window.history_snip_seen_before_reasoning_compaction is True
    assert compactor_observations == [True]
    assert client.call_count == 0
    event_names = [item["event"] for item in progress]
    assert event_names.index("history.pairs_repaired") < event_names.index("history.snipped")
    history_event = next(item for item in progress if item["event"] == "history.snipped")
    assert history_event["removed_rounds"] == 4
    assert history_event["phase"] == "tool_loop"
    assert state.convergence["history_snip_count"] == 1
    assert state.convergence["history_snip_removed_rounds"] == 4
    assert state.convergence["history_snip_phase"] == "tool_loop"
    assert all(
        isinstance(value, (bool, int, float, str)) or value is None
        for key, value in state.convergence.items()
        if key.startswith("history_snip_")
    )


def test_runtime_preserves_explicit_history_snip_false(tmp_path: Path, make_config) -> None:
    runtime, state, client, progress = runtime_fixture(
        tmp_path,
        make_config,
        {"runtime": {"convergence": {"history_snip_enabled": False}}},
    )
    messages: list[dict] = [{"role": "user", "content": "work"}]
    for index in range(8):
        messages.extend(tool_round(index, result="x" * 3_500))
    original = copy.deepcopy(messages)
    context_window = SplitContextWindowController(
        context_window_tokens=65_536,
        safety_buffer_tokens=8_192,
        keep_recent_rounds=4,
        failure_limit=3,
    )
    context_window.bind(state)
    model_route = ModelRoute(
        provider="deepseek",
        tier="standard",
        model="deepseek-v4-pro",
        thinking_enabled=False,
        reasoning_effort=None,
        max_tokens=1_024,
        reasons=("test",),
    )

    runtime._prepare_model_request(
        state,
        messages,
        tools=None,
        model_route=model_route,
        context_window=context_window,
        history_compactor=None,
        auto_compaction_enabled=False,
        auto_compaction_max_tokens=512,
        phase="tool_loop",
        checkpoint=False,
    )

    assert messages == original
    assert client.call_count == 0
    assert not any(key.startswith("history_snip_") for key in state.convergence)
    assert not any(item["event"] == "history.snipped" for item in progress)


def test_snipper_keeps_recent_latest_evidence_and_relevant_rounds() -> None:
    messages: list[dict] = [{"role": "system", "content": "policy"}, {"role": "user", "content": "work"}]
    messages.extend(tool_round(0))
    messages.extend(tool_round(1, path="src/auth.py", result="credential leakage guard evidence"))
    messages.extend(tool_round(2, path="src/billing.py"))
    messages.extend(tool_round(3, name="file_apply"))
    messages.extend(tool_round(4, name="run_tests"))
    messages.extend(tool_round(5, name="file_apply"))
    messages.extend(tool_round(6, name="run_tests"))
    messages.extend(tool_round(7))

    result = HistorySnipper(keep_recent_rounds=1).snip(
        messages,
        objective="Fix billing behavior",
        safety_goals=("prevent credential leakage",),
    )

    assert result.total_rounds == 8
    assert result.removed_rounds == 3
    assert retained_call_ids(result.messages) == [
        "round-1-call-0",
        "round-2-call-0",
        "round-5-call-0",
        "round-6-call-0",
        "round-7-call-0",
    ]
    reasons = dict(result.kept_reasons)
    assert "safety" in reasons[1]
    assert "objective" in reasons[2]
    assert "latest_mutation" in reasons[5]
    assert "latest_verification" in reasons[6]
    assert "recent" in reasons[7]
    assert len(marker_contents(result.messages)) == 1
    assert repair_tool_message_pairs(result.messages).changed is False


def test_snipper_removes_entire_multi_tool_round_without_splitting_pairs() -> None:
    messages = [
        {"role": "user", "content": "task"},
        *tool_round(0, call_count=3),
        *tool_round(1),
    ]

    result = HistorySnipper(keep_recent_rounds=1).snip(messages)

    assert result.removed_rounds == 1
    assert result.removed_messages == 4
    assert retained_call_ids(result.messages) == ["round-1-call-0"]
    assert not any("round-0-call" in json.dumps(item) for item in result.messages)
    assert repair_tool_message_pairs(result.messages).changed is False


def test_snipper_repairs_interrupted_or_orphaned_pairs_before_selection() -> None:
    messages = [
        {"role": "tool", "tool_call_id": "orphan", "content": "must disappear"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"type": "function", "function": {"name": "read_file", "arguments": "{}"}},
                {"id": "kept", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "kept", "content": "available"},
        {"role": "tool", "tool_call_id": "kept", "content": "duplicate"},
    ]

    result = HistorySnipper(keep_recent_rounds=1).snip(messages)

    assert result.repaired_pairs >= 3
    assert result.removed_rounds == 0
    assert repair_tool_message_pairs(result.messages).changed is False
    calls = result.messages[0]["tool_calls"]
    tool_results = result.messages[1:]
    assert [item["tool_call_id"] for item in tool_results] == [item["id"] for item in calls]
    assert "synthetic_repair" in str(tool_results[0]["content"])


def test_snipper_is_deterministic_non_mutating_and_idempotent() -> None:
    messages: list[dict] = [{"role": "user", "content": "task"}]
    for index in range(5):
        messages.extend(tool_round(index, result="x" * 1_000))
    original = copy.deepcopy(messages)
    snipper = HistorySnipper(keep_recent_rounds=1, marker_chars=256)

    first = snipper.snip(messages)
    second = snipper.snip(messages)
    repeated = snipper.snip(first.messages)

    assert messages == original
    assert first == second
    assert repeated.messages == first.messages
    assert repeated.changed is False
    assert len(marker_contents(first.messages)) == 1


def test_snipper_marker_is_bounded_and_does_not_copy_removed_content() -> None:
    secret = "removed-sensitive-payload-" + ("z" * 20_000)
    messages = [
        {"role": "user", "content": "task"},
        *tool_round(0, result=secret),
        *tool_round(1),
    ]

    result = HistorySnipper(keep_recent_rounds=1, marker_chars=128).snip(messages)
    markers = marker_contents(result.messages)

    assert len(markers) == 1
    assert len(markers[0]) <= 128
    assert "removed-sensitive-payload" not in json.dumps(result.messages)
    assert "sha256" in markers[0]


def test_snipper_coalesces_prior_markers_during_later_snip() -> None:
    original: list[dict] = [{"role": "user", "content": "task"}]
    for index in range(4):
        original.extend(tool_round(index))
    snipper = HistorySnipper(keep_recent_rounds=1, marker_chars=256)
    first = snipper.snip(original)
    extended = [*first.messages, *tool_round(4), *tool_round(5)]

    second = snipper.snip(extended)

    assert second.coalesced_markers == 1
    assert len(marker_contents(second.messages)) == 1
    assert retained_call_ids(second.messages) == ["round-5-call-0"]
    assert repair_tool_message_pairs(second.messages).changed is False


def test_snipper_matches_bounded_chinese_objective_and_safety_terms() -> None:
    messages = [
        {"role": "user", "content": "task"},
        *tool_round(0, result="完成支付授权校验"),
        *tool_round(1, result="检查令牌泄露风险"),
        *tool_round(2, result="unrelated"),
    ]

    result = HistorySnipper(keep_recent_rounds=0).snip(
        messages,
        objective="修复支付授权",
        safety_goals=("不得泄露令牌",),
    )

    assert retained_call_ids(result.messages) == ["round-0-call-0", "round-1-call-0"]
    assert result.removed_rounds == 1
    reasons = dict(result.kept_reasons)
    assert "objective" in reasons[0]
    assert "safety" in reasons[1]


def test_snipper_handles_empty_short_and_text_only_boundaries() -> None:
    snipper = HistorySnipper(keep_recent_rounds=10)
    assert snipper.snip([]).messages == []
    short = [{"role": "user", "content": "task"}, {"role": "assistant", "content": "done"}]
    unchanged = snipper.snip(short)
    assert unchanged.messages == short
    assert unchanged.changed is False

    removed = HistorySnipper(keep_recent_rounds=0).snip(short)
    assert removed.removed_rounds == 1
    assert [item["role"] for item in removed.messages] == ["user", "system"]
    assert repair_tool_message_pairs(removed.messages).changed is False
