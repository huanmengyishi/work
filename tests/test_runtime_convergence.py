from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.convergence import (
    ContextWindowController,
    TaskConvergenceController,
    ToolHistoryCompactor,
    estimate_request_tokens,
    repair_tool_message_pairs,
)
from agent.context import ContextBuilder
from agent.deepseek import ChatResponse, DeepSeekContextOverflow, DeepSeekStreamInterrupted
from agent.memory import MemoryStore
from agent.model_router import ModelRoute
from agent.project import ProjectManager
from agent.runtime import AgentRuntime, _normalize_assistant_tool_calls
from agent.state import AgentState
from agent.tools import ToolManager, ToolResult


def tool_pair(index: int, *, name: str = "read_file", chars: int = 12_000) -> list[dict]:
    call_id = f"call-{index}"
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps({"path": f"src/file-{index}.ts", "start_line": 1, "end_line": 240}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": f"result-{index}:" + (str(index % 10) * chars),
        },
    ]


def multi_tool_round(round_index: int, *, count: int = 3, chars: int = 3_000) -> list[dict]:
    calls = []
    results = []
    for result_index in range(count):
        call_id = f"round-{round_index}-call-{result_index}"
        calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": json.dumps(
                        {
                            "path": f"src/round-{round_index}-{result_index}.ts",
                            "start_line": 1,
                            "end_line": 240,
                        }
                    ),
                },
            }
        )
        results.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": f"round-{round_index}-result-{result_index}:" + (str(round_index) * chars),
            }
        )
    return [{"role": "assistant", "content": None, "tool_calls": calls}, *results]


def json_tool_pair(
    index: int,
    *,
    success: bool = True,
    chars: int = 6_000,
) -> tuple[list[dict], str]:
    call_id = f"json-call-{index}"
    content = json.dumps(
        {
            "success": success,
            "stdout": str(index) * chars if success else "",
            "stderr": ("injected failure evidence " * chars)[:chars] if not success else "",
            "duration_ms": 10 + index,
            "request_id": call_id,
            "data": {"path": f"src/file-{index}.ts"},
        },
        ensure_ascii=False,
        indent=2,
    )
    return (
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": f"src/file-{index}.ts", "start_line": 1, "end_line": 100}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": call_id, "content": content},
        ],
        content,
    )


def compacted_metadata(content: str) -> dict:
    lines = content.splitlines()
    assert len(lines) >= 2
    return json.loads(lines[1])


def test_tool_history_compaction_bounds_aggregate_and_preserves_pairs() -> None:
    messages: list[dict] = [{"role": "system", "content": "objective and plan"}]
    for index in range(10):
        name = "run_tests" if index == 4 else "read_file"
        messages.extend(tool_pair(index, name=name))
    original_ids = [item["tool_call_id"] for item in messages if item.get("role") == "tool"]
    latest_contents = [item["content"] for item in messages if item.get("role") == "tool"][-2:]
    protected_validation = next(
        item["content"] for item in messages if item.get("role") == "tool" and item.get("tool_call_id") == "call-4"
    )
    compactor = ToolHistoryCompactor(
        aggregate_chars=50_000,
        output_reserve_chars=10_000,
        compacted_result_chars=400,
        keep_recent_results=2,
        failure_limit=3,
    )

    result = compactor.compact(messages)

    assert result.changed is True
    assert result.final_chars <= compactor.target_chars
    assert [item["tool_call_id"] for item in result.messages if item.get("role") == "tool"] == original_ids
    assert [item["content"] for item in result.messages if item.get("role") == "tool"][-2:] == latest_contents
    assert (
        next(
            item["content"]
            for item in result.messages
            if item.get("role") == "tool" and item.get("tool_call_id") == "call-4"
        )
        == protected_validation
    )
    compacted = [
        item["content"]
        for item in result.messages
        if item.get("role") == "tool" and str(item.get("content") or "").startswith("[Deep Agent compacted")
    ]
    assert compacted
    metadata = compacted_metadata(compacted[0])
    assert metadata["tool"] == "read_file"
    assert str(metadata["target"]["path"]).startswith("src/file-")
    assert len(metadata["sha256"]) == 64


def test_tool_history_compaction_enforces_hard_limit_even_for_protected_results() -> None:
    messages: list[dict] = []
    messages.extend(tool_pair(1, name="file_apply", chars=5_000))
    messages.extend(tool_pair(2, name="run_tests", chars=5_000))
    messages.extend(tool_pair(3, name="read_file", chars=5_000))
    original_ids = [item["tool_call_id"] for item in messages if item.get("role") == "tool"]
    compactor = ToolHistoryCompactor(
        aggregate_chars=4_096,
        output_reserve_chars=3_072,
        compacted_result_chars=300,
        keep_recent_results=2,
        failure_limit=3,
    )

    result = compactor.compact(messages)

    assert compactor.target_chars == 1_024
    assert result.final_chars <= compactor.target_chars
    assert [item["tool_call_id"] for item in result.messages if item.get("role") == "tool"] == original_ids
    assert result.compacted_count == 3


def test_tool_history_recent_window_counts_model_rounds_not_individual_results() -> None:
    messages: list[dict] = []
    for round_index in range(1, 4):
        messages.extend(multi_tool_round(round_index))
    protected = {
        item["tool_call_id"]: item["content"]
        for item in messages
        if item.get("role") == "tool" and str(item.get("tool_call_id") or "").startswith(("round-2-", "round-3-"))
    }
    compactor = ToolHistoryCompactor(
        aggregate_chars=22_000,
        output_reserve_chars=3_000,
        compacted_result_chars=400,
        keep_recent_results=2,
        failure_limit=3,
    )

    result = compactor.compact(messages)

    assert result.final_chars <= compactor.target_chars
    actual = {
        item["tool_call_id"]: item["content"]
        for item in result.messages
        if item.get("role") == "tool" and item.get("tool_call_id") in protected
    }
    assert actual == protected


def test_single_current_api_round_is_never_removed_by_aggregate_compaction() -> None:
    messages = multi_tool_round(1, count=33, chars=500)
    original_ids = [item["tool_call_id"] for item in messages if item.get("role") == "tool"]
    compactor = ToolHistoryCompactor(
        aggregate_chars=4_096,
        output_reserve_chars=0,
        compacted_result_chars=256,
        keep_recent_results=1,
        failure_limit=3,
    )
    result = compactor.compact(messages)
    tool_messages = [item for item in result.messages if item.get("role") == "tool"]
    assert result.final_chars <= compactor.target_chars
    assert [item.get("tool_call_id") for item in tool_messages] == original_ids
    assert len(tool_messages) == 33
    assert not any(
        str(item.get("content") or "").startswith("[Deep Agent collapsed oldest complete tool rounds]")
        for item in result.messages
    )
    assert repair_tool_message_pairs(result.messages).changed is False


def test_tool_history_compaction_preserves_original_metadata_across_recompaction() -> None:
    messages: list[dict] = []
    originals: dict[str, str] = {}
    for index in range(3):
        pair, content = json_tool_pair(index, success=index != 0)
        messages.extend(pair)
        originals[f"json-call-{index}"] = content
    first = ToolHistoryCompactor(
        aggregate_chars=16_000,
        output_reserve_chars=3_000,
        compacted_result_chars=700,
        keep_recent_results=1,
        failure_limit=3,
    ).compact(messages)
    first_summary = next(
        str(item.get("content") or "")
        for item in first.messages
        if item.get("role") == "tool" and item.get("tool_call_id") == "json-call-0"
    )
    assert first_summary.startswith("[Deep Agent compacted tool result]")

    second_compactor = ToolHistoryCompactor(
        aggregate_chars=4_096,
        output_reserve_chars=3_072,
        compacted_result_chars=256,
        keep_recent_results=1,
        failure_limit=3,
    )
    second = second_compactor.compact(first.messages)
    second_summary = next(
        str(item.get("content") or "")
        for item in second.messages
        if item.get("role") == "tool" and item.get("tool_call_id") == "json-call-0"
    )
    original = originals["json-call-0"]
    metadata = compacted_metadata(second_summary)

    assert second.final_chars <= second_compactor.target_chars
    assert metadata == {
        "tool": "read_file",
        "success": False,
        "target": {"path": "src/file-0.ts", "start_line": 1, "end_line": 100},
        "original_chars": len(original),
        "sha256": hashlib.sha256(original.encode()).hexdigest(),
    }


def test_tool_history_inherits_original_metadata_from_single_result_limit() -> None:
    messages: list[dict] = []
    originals: dict[str, str] = {}
    for index in range(3):
        result = ToolResult(
            False,
            "stdout " * 2_000,
            "failure evidence " * 2_000,
            data={"path": f"src/failure-{index}.py"},
            request_id=f"limited-{index}",
        )
        original = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
        originals[f"limited-{index}"] = original
        messages.extend(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": f"limited-{index}",
                            "function": {
                                "name": "read_file",
                                "arguments": json.dumps({"path": f"src/failure-{index}.py"}),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": f"limited-{index}",
                    "content": result.as_text(limit=512),
                },
            ]
        )
    compacted = ToolHistoryCompactor(
        aggregate_chars=4_096,
        output_reserve_chars=3_072,
        compacted_result_chars=256,
        keep_recent_results=1,
        failure_limit=3,
    ).compact(messages)
    content = next(
        str(item.get("content") or "") for item in compacted.messages if item.get("tool_call_id") == "limited-0"
    )
    metadata = compacted_metadata(content)
    original = originals["limited-0"]
    assert metadata["success"] is False
    assert metadata["original_chars"] == len(original)
    assert metadata["sha256"] == hashlib.sha256(original.encode()).hexdigest()


def test_long_target_and_emergency_summaries_remain_structurally_valid() -> None:
    messages: list[dict] = []
    for index in range(8):
        path = ("very-long-directory/" * 20) + f"file-{index}.ts"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": f"long-{index}",
                            "function": {
                                "name": "read_file",
                                "arguments": json.dumps({"path": path, "start_line": 1, "end_line": 200}),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": f"long-{index}",
                    "content": json.dumps({"success": True, "stdout": "x" * 5_000}),
                },
            ]
        )
    compactor = ToolHistoryCompactor(
        aggregate_chars=4_096,
        output_reserve_chars=3_072,
        compacted_result_chars=256,
        keep_recent_results=4,
        failure_limit=3,
    )
    result = compactor.compact(messages)
    assert result.final_chars <= compactor.target_chars
    for item in (entry for entry in result.messages if entry.get("role") == "tool"):
        content = str(item.get("content") or "")
        payload = json.loads(content.splitlines()[1] if content.startswith("[Deep Agent compacted") else content)
        assert isinstance(payload, dict)
        assert "success" in payload or "s" in payload

    extended = list(result.messages)
    for index in range(8, 24):
        extended.extend(json_tool_pair(index)[0])
    recompressed = compactor.compact(extended)
    assert recompressed.final_chars <= compactor.target_chars
    assert repair_tool_message_pairs(recompressed.messages).changed is False
    assert any(
        str(item.get("content") or "").startswith("[Deep Agent collapsed oldest complete tool rounds]")
        for item in recompressed.messages
    )
    remaining_tools = [item for item in recompressed.messages if item.get("role") == "tool"]
    assert 0 < len(remaining_tools) <= 8
    for item in remaining_tools:
        content = str(item.get("content") or "")
        payload = json.loads(content.splitlines()[1] if content.startswith("[Deep Agent compacted") else content)
        assert payload.get("success", payload.get("s")) is True
        assert payload.get("sha256", payload.get("h"))


def test_complete_round_collapse_keeps_bounded_removed_evidence() -> None:
    messages: list[dict] = []
    for index in range(9):
        call_id = f"critical-{index}"
        path = f"critical-{index}.txt"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "function": {"name": "read_file", "arguments": json.dumps({"path": path})},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps({"success": True, "stdout": f"CRITICAL-{index}=42 " + ("x" * 400)}),
                },
            ]
        )
    compactor = ToolHistoryCompactor(
        aggregate_chars=4_096,
        output_reserve_chars=3_072,
        compacted_result_chars=256,
        keep_recent_results=4,
        failure_limit=3,
    )
    result = compactor.compact(messages)
    notice = next(
        str(item.get("content") or "")
        for item in result.messages
        if str(item.get("content") or "").startswith("[Deep Agent collapsed oldest complete tool rounds]")
    )
    assert result.final_chars <= compactor.target_chars
    assert "critical-0.txt" in notice
    assert "CRITICAL-0=42" in notice
    assert repair_tool_message_pairs(result.messages).changed is False


def test_tool_history_compaction_failure_opens_circuit(monkeypatch) -> None:
    messages = tool_pair(1, chars=10_000) + tool_pair(2, chars=10_000)
    compactor = ToolHistoryCompactor(
        aggregate_chars=8_000,
        output_reserve_chars=1_000,
        compacted_result_chars=400,
        keep_recent_results=1,
        failure_limit=3,
    )
    attempts = 0

    def fail(_messages):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("injected compaction failure")

    monkeypatch.setattr(compactor, "_compact_once", fail)
    results = [compactor.compact(messages) for _ in range(4)]

    assert attempts == 3
    assert [item.failure_count for item in results] == [1, 2, 3, 3]
    assert results[-1].circuit_open is True
    assert all(item.final_chars <= compactor.target_chars for item in results)
    assert all(item.changed is True for item in results)
    assert all(item.messages is not messages for item in results)


def test_compaction_circuit_fallback_tolerates_malformed_prior_metadata(monkeypatch) -> None:
    messages = tool_pair(1, chars=10_000) + tool_pair(2, chars=10_000)
    messages[1]["content"] = (
        '[Deep Agent compacted metadata]\n{"tool":"read_file","success":true,'
        '"target":{},"original_chars":"not-an-int","sha256":"invalid"}'
    )
    compactor = ToolHistoryCompactor(
        aggregate_chars=8_000,
        output_reserve_chars=1_000,
        compacted_result_chars=400,
        keep_recent_results=1,
        failure_limit=1,
    )

    def fail(_messages):
        raise RuntimeError("force fallback")

    monkeypatch.setattr(compactor, "_compact_once", fail)
    result = compactor.compact(messages)
    assert result.circuit_open is True
    assert result.final_chars <= compactor.target_chars
    assert result.changed is True


def test_target_key_distinguishes_ranges_queries_and_globs() -> None:
    target_key = TaskConvergenceController._target_key

    assert target_key(
        {"tool": "template", "action": "read_file", "args": {"path": "src/app.ts", "start_line": 1, "end_line": 100}}
    ) != target_key(
        {
            "tool": "template",
            "action": "read_file",
            "args": {"path": "src/app.ts", "start_line": 101, "end_line": 200},
        }
    )
    assert target_key(
        {
            "tool": "template",
            "action": "search_code",
            "args": {"path": "src", "query": "first", "glob": "*.ts"},
        }
    ) != target_key(
        {
            "tool": "template",
            "action": "search_code",
            "args": {"path": "src", "query": "second", "glob": "*.ts"},
        }
    )
    assert target_key(
        {
            "tool": "template",
            "action": "search_code",
            "args": {"path": "src", "query": "needle", "glob": "*.ts"},
        }
    ) != target_key(
        {
            "tool": "template",
            "action": "search_code",
            "args": {"path": "src", "query": "needle", "glob": "*.tsx"},
        }
    )
    assert target_key(
        {"tool": "template", "action": "find_files", "args": {"path": "src", "pattern": "*.py"}}
    ) != target_key({"tool": "template", "action": "find_files", "args": {"path": "src", "pattern": "*.ts"}})


def test_repair_tool_message_pairs_repairs_missing_orphan_and_duplicate_ids() -> None:
    messages = [
        {"role": "tool", "tool_call_id": "orphan-before", "content": "orphan"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call-a", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
                {"id": "call-a", "type": "function", "function": {"name": "read_file", "arguments": "{}"}},
                {"type": "function", "function": {"name": "read_file", "arguments": "{}"}},
                {"id": "call-b", "type": "function", "function": {"name": "list_dir", "arguments": "{}"}},
            ],
        },
        {"role": "system", "content": "recovery guidance"},
        {"role": "tool", "tool_call_id": "call-b", "content": "result-b"},
        {"role": "tool", "tool_call_id": "call-a", "content": "result-a"},
        {"role": "tool", "tool_call_id": "call-a", "content": "duplicate-result-a"},
        {"role": "tool", "tool_call_id": "orphan-after", "content": "orphan"},
        {"role": "assistant", "content": "continue"},
    ]

    repaired = repair_tool_message_pairs(messages)

    assert repaired.changed is True
    assistant_index = next(
        index
        for index, item in enumerate(repaired.messages)
        if item.get("role") == "assistant" and item.get("tool_calls")
    )
    assistant = repaired.messages[assistant_index]
    call_ids = [str(call.get("id") or "") for call in assistant["tool_calls"]]
    assert len(call_ids) == len(set(call_ids)) == 3
    assert call_ids[0] == "call-a"
    assert call_ids[1].startswith("deep-agent-call-")
    assert call_ids[2] == "call-b"
    results = repaired.messages[assistant_index + 1 : assistant_index + 4]
    assert [item.get("role") for item in results] == ["tool", "tool", "tool"]
    assert [item.get("tool_call_id") for item in results] == call_ids
    assert results[0]["content"] == "result-a"
    synthetic = json.loads(results[1]["content"])
    assert synthetic["success"] is False
    assert synthetic["data"]["synthetic_repair"] is True
    assert results[2]["content"] == "result-b"
    assert repaired.messages[assistant_index + 4] == {"role": "system", "content": "recovery guidance"}
    assert not any(
        item.get("role") == "tool" and item.get("tool_call_id") in {"orphan-before", "orphan-after"}
        for item in repaired.messages
    )
    assert repair_tool_message_pairs(repaired.messages).changed is False


def test_pair_repair_marks_pure_interleaving_and_reordering_as_changed() -> None:
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call-a", "function": {"name": "read_file", "arguments": "{}"}},
                {"id": "call-b", "function": {"name": "read_file", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call-b", "content": "result-b"},
        {"role": "system", "content": "recover after all results"},
        {"role": "tool", "tool_call_id": "call-a", "content": "result-a"},
    ]
    repaired = repair_tool_message_pairs(messages)
    assert repaired.changed is True
    assert [item.get("tool_call_id") for item in repaired.messages[1:3]] == ["call-a", "call-b"]
    assert repaired.messages[3] == {"role": "system", "content": "recover after all results"}


def test_context_compaction_preserves_one_or_four_complete_api_rounds() -> None:
    controller = ContextWindowController(
        context_window_tokens=16_384,
        safety_buffer_tokens=2_048,
        keep_recent_rounds=4,
        failure_limit=3,
    )
    one_round = [{"role": "user", "content": "task"}, *tool_pair(1)]
    four_rounds = [{"role": "user", "content": "task"}]
    for index in range(1, 5):
        four_rounds.extend(tool_pair(index))

    assert controller.compaction_span(one_round) is None
    assert controller.compaction_span(four_rounds) is None


def test_context_compaction_summarizes_only_the_oldest_of_five_complete_api_rounds() -> None:
    controller = ContextWindowController(
        context_window_tokens=16_384,
        safety_buffer_tokens=2_048,
        keep_recent_rounds=4,
        failure_limit=3,
    )
    messages = [{"role": "system", "content": "policy"}, {"role": "user", "content": "task"}]
    for index in range(1, 6):
        messages.extend(tool_pair(index))

    span = controller.compaction_span(messages)

    assert span is not None
    start, end = span
    assert [item.get("tool_call_id") for item in messages[start:end] if item.get("role") == "tool"] == ["call-1"]
    assert messages[end]["tool_calls"][0]["id"] == "call-2"


def test_context_compaction_counts_text_only_assistant_as_a_protected_api_round() -> None:
    controller = ContextWindowController(
        context_window_tokens=16_384,
        safety_buffer_tokens=2_048,
        keep_recent_rounds=4,
        failure_limit=3,
    )
    messages = [
        {"role": "user", "content": "task"},
        *tool_pair(1),
        {"role": "assistant", "content": "intermediate synthesis without tools"},
        *tool_pair(2),
        *tool_pair(3),
        *tool_pair(4),
    ]

    span = controller.compaction_span(messages)

    assert span is not None
    start, end = span
    assert [item.get("tool_call_id") for item in messages[start:end] if item.get("role") == "tool"] == ["call-1"]
    assert messages[end] == {"role": "assistant", "content": "intermediate synthesis without tools"}


def test_request_budget_counts_tool_schemas_and_reserves_output_and_safety() -> None:
    controller = ContextWindowController(
        context_window_tokens=16_384,
        safety_buffer_tokens=2_048,
        keep_recent_rounds=2,
        failure_limit=3,
    )
    messages = [{"role": "user", "content": "short request"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "large_schema",
                "description": "x" * 50_000,
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    without_schemas = controller.budget(messages, None, max_output_tokens=4_096)
    with_schemas = controller.budget(messages, tools, max_output_tokens=4_096)

    assert with_schemas.estimated_tokens == estimate_request_tokens(messages, tools)
    assert with_schemas.estimated_tokens > without_schemas.estimated_tokens + 10_000
    assert without_schemas.over_limit is False
    assert with_schemas.over_limit is True
    assert with_schemas.output_reserve_tokens == 4_096
    assert with_schemas.safety_buffer_tokens == 2_048
    assert with_schemas.input_limit_tokens == 10_240
    assert (
        with_schemas.input_limit_tokens + with_schemas.output_reserve_tokens + with_schemas.safety_buffer_tokens
        == controller.context_window_tokens
    )
    small = ContextWindowController(
        context_window_tokens=8_192,
        safety_buffer_tokens=1_024,
        keep_recent_rounds=1,
        failure_limit=3,
    )
    clamped = small.budget(messages, None, max_output_tokens=10_000)
    assert small.effective_output_tokens(10_000) == clamped.output_reserve_tokens == 6_144
    assert clamped.input_limit_tokens + clamped.output_reserve_tokens + clamped.safety_buffer_tokens == 8_192

    maximum_safety = ContextWindowController(
        context_window_tokens=8_192,
        safety_buffer_tokens=8_192,
        keep_recent_rounds=1,
        failure_limit=3,
    )
    extreme = maximum_safety.budget(messages, None, max_output_tokens=10_000)
    assert extreme.input_limit_tokens >= 1_024
    assert extreme.input_limit_tokens + extreme.output_reserve_tokens + extreme.safety_buffer_tokens == 8_192


def test_context_compaction_circuit_survives_resume_and_success_resets_it(
    tmp_path: Path,
    make_config,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    project = ProjectManager(make_config()).resolve_project(root)
    state = AgentState.create(
        session_id="resume-context-circuit",
        project=project,
        user_request="compress a long context",
        loaded_memories=[],
        loaded_tools=[],
        git_branch=None,
        context_index_path=str(project.agent_dir / "index.json"),
    )
    first = ContextWindowController(
        context_window_tokens=8_192,
        safety_buffer_tokens=1_024,
        keep_recent_rounds=1,
        failure_limit=3,
    )
    first.bind(state)
    for _ in range(4):
        first.record_failure()

    assert state.convergence["context_compaction_failure_count"] == 3
    assert state.convergence["context_compaction_circuit_open"] is True

    state.resume("继续完成")
    resumed = ContextWindowController(
        context_window_tokens=8_192,
        safety_buffer_tokens=1_024,
        keep_recent_rounds=1,
        failure_limit=3,
    )
    resumed.bind(state)

    assert resumed.failure_count == 3
    assert resumed.circuit_open is True
    resumed.record_success()
    assert state.convergence["context_compaction_failure_count"] == 0
    assert state.convergence["context_compaction_circuit_open"] is False


def test_overflow_semantic_compaction_respects_open_circuit(tmp_path: Path, make_config, monkeypatch) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=RecordingClient([]),
    )
    state = AgentState.create(
        session_id="overflow-open-circuit",
        project=project,
        user_request="continue without another semantic compaction request",
        loaded_memories=[],
        loaded_tools=[],
        git_branch=None,
        context_index_path=str(project.agent_dir / "index.json"),
    )
    messages = [
        {"role": "system", "content": "old bounded context"},
        {"role": "user", "content": "continue"},
    ]
    context_window = ContextWindowController(
        context_window_tokens=8_192,
        safety_buffer_tokens=1_024,
        keep_recent_rounds=1,
        failure_limit=3,
    )
    context_window.bind(state)
    for _ in range(3):
        context_window.record_failure()
    model_route = ModelRoute(
        provider="deepseek",
        tier="standard",
        model="deepseek-v4-pro",
        thinking_enabled=True,
        reasoning_effort="high",
        max_tokens=1_024,
        reasons=("test",),
    )
    semantic_calls = 0
    fallback_calls = 0

    def forbidden_semantic(*_args, **_kwargs):
        nonlocal semantic_calls
        semantic_calls += 1
        raise AssertionError("an open compaction circuit must prevent another semantic model request")

    def deterministic_fallback(*args, **_kwargs):
        nonlocal fallback_calls
        fallback_calls += 1
        del args[1][0]
        return True

    monkeypatch.setattr(runtime, "_auto_compact_context", forbidden_semantic)
    monkeypatch.setattr(runtime, "_emergency_context_collapse", deterministic_fallback)

    assert (
        runtime._overflow_semantic_compact(
            state,
            messages,
            tools=None,
            model_route=model_route,
            context_window=context_window,
            history_compactor=None,
            auto_compaction_max_tokens=512,
        )
        is True
    )
    assert semantic_calls == 0
    assert fallback_calls == 1
    assert messages == [{"role": "user", "content": "continue"}]


@pytest.mark.parametrize(
    "function_name,arguments",
    [
        ("shell_run", {"command": "pwd && cat src/app.py"}),
        ("shell_run", {"command": "true; rg needle src"}),
        ("shell_run", {"command": "/bin/cat src/app.py"}),
        ("shell_run", {"command": "bash -lc 'cat src/app.py'"}),
        ("shell_run", {"command": "cd src && pwd && cat app.py"}),
        ("shell_run", {"command": "x=$(cat secret.txt)"}),
        ("shell_run", {"command": "python -c \"print(open('secret.txt').read())\""}),
        ("shell_run", {"command": "git show HEAD:secret.txt"}),
        ("shell_run", {"command": "dd if=secret.txt"}),
        ("shell_run", {"command": "source secret.env"}),
        ("python_run", {"code": "import subprocess; subprocess.run(['/bin/cat', 'src/app.py'])"}),
        ("python_run", {"code": "from subprocess import run; run(['cat', 'src/app.py'])"}),
        ("python_run", {"code": "import os; os.execvp('cat', ['cat', 'src/app.py'])"}),
    ],
)
def test_exploration_bypass_detection_covers_chains_and_nested_processes(function_name, arguments) -> None:
    assert TaskConvergenceController.is_exploration_bypass(function_name, arguments) is True


@pytest.mark.parametrize(
    "function_name,arguments",
    [
        ("shell_run", {"command": "npm run typecheck"}),
        ("shell_run", {"command": "npm test"}),
        ("shell_run", {"command": "npm test -- src/example.test.ts"}),
        ("shell_run", {"command": "ruff check ."}),
    ],
)
def test_exploration_bypass_allows_validation_without_file_reading(function_name, arguments) -> None:
    assert TaskConvergenceController.is_exploration_bypass(function_name, arguments) is False


@pytest.mark.parametrize(
    "command",
    ["pwd", "sed -n 1,20p src/app.ts", "head -20 src/app.ts", "git diff", "npm run typecheck; cat src/app.ts"],
)
def test_hard_phase_shell_policy_is_validation_allowlist(command: str) -> None:
    assert TaskConvergenceController.is_exploration_bypass("shell_run", {"command": command}) is True


@pytest.mark.parametrize(
    "command",
    [
        "ruff check --fix .",
        "cargo clippy --fix",
        "npm run lint -- --fix",
        "ruff check --write=true .",
        "cargo clippy --apply=yes",
        "pytest --update=snapshots",
        "pytest --bless=all",
        "pytest --accept=yes",
        "ruff check -wfoo .",
        "mypy --install-types --non-interactive .",
        "npm run build",
    ],
)
def test_hard_phase_rejects_validation_commands_with_mutation_flags(command: str) -> None:
    assert TaskConvergenceController.is_exploration_bypass("shell_run", {"command": command}) is True


def test_noop_plan_update_does_not_reset_exploration_stall() -> None:
    state = SimpleNamespace(plan=[SimpleNamespace(id="scope", status="in_progress")])
    controller = TaskConvergenceController(
        mode="deep",
        max_rounds=24,
        exploration_round_limit=2,
        reserved_rounds=4,
    )
    controller.bind(state)
    for index in range(2):
        controller.observe_round(
            state,
            [{"tool": "template", "action": "read_file", "args": {"path": f"src/{index}.py"}}],
            [{"success": True}],
        )
    assert controller.before_round(3).excluded_functions
    progressed = controller.observe_round(
        state,
        [{"tool": "agent", "action": "update_step", "args": {"step_id": "scope", "status": "in_progress"}}],
        [{"success": True}],
    )
    assert progressed is False
    assert controller.consecutive_read_only_rounds == 3


def test_large_task_convergence_closes_exploration_and_resets_after_progress() -> None:
    steps = [SimpleNamespace(id="scope", status="in_progress")]
    state = SimpleNamespace(plan=steps)
    controller = TaskConvergenceController(
        mode="deep",
        max_rounds=24,
        exploration_round_limit=6,
        reserved_rounds=4,
    )
    controller.bind(state)
    for index in range(6):
        controller.observe_round(
            state,
            [
                {
                    "tool": "template",
                    "action": "read_file",
                    "args": {"path": f"src/file-{index}.ts", "start_line": 1, "end_line": 240},
                }
            ],
        )

    soft = controller.before_round(7)
    assert len(soft.messages) == 1
    assert "Stop broad scanning" in soft.messages[0]
    assert "list_dir" in soft.excluded_functions
    assert "read_file" not in soft.excluded_functions

    controller.observe_round(
        state,
        [{"tool": "template", "action": "read_file", "args": {"path": "src/file-0.ts"}}],
    )
    controller.observe_round(
        state,
        [{"tool": "template", "action": "read_file", "args": {"path": "src/file-1.ts"}}],
    )
    forced = controller.before_round(9)
    assert "list_dir" in forced.excluded_functions
    assert "read_file" in forced.excluded_functions
    assert forced.reason == "continuous exploration threshold reached"

    steps[0].status = "completed"
    controller.observe_round(
        state,
        [{"tool": "agent", "action": "update_step", "args": {"step_id": "scope"}}],
    )
    resumed = controller.before_round(10)
    assert resumed.excluded_functions == frozenset()
    assert controller.consecutive_read_only_rounds == 0


def test_reserved_rounds_close_reads_even_without_duplicate_targets() -> None:
    state = SimpleNamespace(plan=[SimpleNamespace(id="verify", status="in_progress")])
    controller = TaskConvergenceController(
        mode="large",
        max_rounds=16,
        exploration_round_limit=10,
        reserved_rounds=4,
    )
    controller.bind(state)

    action = controller.before_round(13)

    assert "read_file" in action.excluded_functions
    assert "shell_run" not in action.excluded_functions
    assert "python_run" not in action.excluded_functions
    assert "reserved implementation and verification window" == action.reason


def test_reserved_round_allows_only_two_bounded_reads_of_a_known_implementation_path() -> None:
    known_read = {
        "request": {
            "tool": "template",
            "action": "read_file",
            "args": {"path": "src/state.ts", "start_line": 1, "end_line": 80},
        },
        "result": {"success": True},
    }
    state = SimpleNamespace(
        plan=[SimpleNamespace(id="implement", status="in_progress")],
        tool_calls=[known_read],
    )
    controller = TaskConvergenceController(
        mode="deep",
        max_rounds=24,
        exploration_round_limit=10,
        reserved_rounds=8,
        implementation_read_limit=2,
    )
    controller.bind(state)

    action = controller.before_round(17, state)

    assert action.guard_implementation_read is True
    assert "read_file" not in action.excluded_functions
    assert {"list_dir", "find_files", "search_code"} <= action.excluded_functions
    assert "shell_run" not in action.excluded_functions
    assert "python_run" not in action.excluded_functions
    exact_read = {"path": "src/state.ts", "start_line": 1400, "end_line": 1470}
    assert controller.implementation_read_denial(state, "read_file", exact_read) == ""
    assert controller.implementation_read_denial(state, "read_file", exact_read) == ""
    assert "exhausted" in controller.implementation_read_denial(state, "read_file", exact_read)
    assert "read_file" in controller.before_round(18, state).excluded_functions


def test_implementation_read_allowance_persists_across_controller_rebind() -> None:
    state = SimpleNamespace(
        plan=[SimpleNamespace(id="implement", status="in_progress")],
        tool_calls=[
            {
                "request": {"tool": "template", "action": "read_file", "args": {"path": "src/state.ts"}},
                "result": {"success": True},
            }
        ],
        convergence={},
    )
    first = TaskConvergenceController(
        mode="deep",
        max_rounds=24,
        exploration_round_limit=10,
        reserved_rounds=8,
        implementation_read_limit=2,
    )
    first.bind(state)
    exact = {"path": "src/state.ts", "start_line": 1, "end_line": 200}
    assert first.implementation_read_denial(state, "read_file", exact) == ""
    assert state.convergence["implementation_reads_used"] == 1

    resumed = TaskConvergenceController(
        mode="deep",
        max_rounds=24,
        exploration_round_limit=10,
        reserved_rounds=8,
        implementation_read_limit=2,
    )
    resumed.bind(state)
    assert resumed.implementation_reads_used == 1
    assert resumed.implementation_read_denial(state, "read_file", exact) == ""
    assert "exhausted" in resumed.implementation_read_denial(state, "read_file", exact)

    temporarily_disabled = TaskConvergenceController(
        mode="deep",
        max_rounds=24,
        exploration_round_limit=10,
        reserved_rounds=8,
        implementation_read_limit=0,
    )
    temporarily_disabled.bind(state)
    assert temporarily_disabled.implementation_reads_used == 2
    assert state.convergence["implementation_reads_used"] == 2


def test_agent_state_resume_reopens_bounded_convergence_window_without_resetting_durable_safety() -> None:
    state = AgentState.create(
        session_id="resume-convergence-window",
        project=SimpleNamespace(id="project-id", name="project", root=Path("/tmp/project"), language="Python"),
        user_request="inspect and conditionally fix",
        loaded_memories=[],
        loaded_tools=[],
        git_branch=None,
        context_index_path="/tmp/project/.project-agent/index.json",
    )
    state.convergence = {
        "implementation_reads_used": 2,
        "consecutive_read_only_rounds": 8,
        "low_yield_rounds": 5,
        "nudge_count": 2,
        "nudge_sent_for_stall": True,
        "hard_notice_sent": True,
        "notice_turn": 1,
        "seen_targets": ["template.read_file:known"],
        "context_compaction_failure_count": 3,
        "context_compaction_circuit_open": True,
    }

    state.resume("continue with the exact failed edit")

    assert state.turn == 2
    assert state.convergence == {
        "seen_targets": ["template.read_file:known"],
        "context_compaction_failure_count": 3,
        "context_compaction_circuit_open": True,
    }
    controller = TaskConvergenceController(
        mode="deep",
        max_rounds=24,
        exploration_round_limit=6,
        reserved_rounds=8,
        implementation_read_limit=2,
    )
    controller.bind(state)
    assert controller.implementation_reads_used == 0
    assert controller.consecutive_read_only_rounds == 0
    assert controller.low_yield_rounds == 0
    assert controller.before_round(1, state).excluded_functions == frozenset()


def test_exploration_safety_state_is_bounded_and_restored_across_rebind() -> None:
    step = SimpleNamespace(id="scope", status="in_progress")
    state = SimpleNamespace(plan=[step], convergence={}, turn=1)
    first = TaskConvergenceController(
        mode="deep",
        max_rounds=24,
        exploration_round_limit=2,
        reserved_rounds=4,
    )
    first.bind(state)
    request = {"tool": "template", "action": "read_file", "args": {"path": "src/repeated.py"}}
    first.observe_round(state, [request], [{"success": True}])
    first.observe_round(state, [request], [{"success": True}])
    assert first.before_round(3, state).messages

    resumed = TaskConvergenceController(
        mode="deep",
        max_rounds=24,
        exploration_round_limit=2,
        reserved_rounds=4,
    )
    resumed.bind(state)

    assert resumed.consecutive_read_only_rounds == 2
    assert resumed.low_yield_rounds == 1
    assert resumed.seen_targets == {TaskConvergenceController._target_key(request)}
    restored_action = resumed.before_round(3, state)
    assert restored_action.excluded_functions
    assert restored_action.messages == ()

    state.turn = 2
    next_turn = TaskConvergenceController(
        mode="deep",
        max_rounds=24,
        exploration_round_limit=2,
        reserved_rounds=4,
    )
    next_turn.bind(state)
    assert next_turn.consecutive_read_only_rounds == 2
    assert next_turn.before_round(3, state).messages

    for index in range(200):
        next_turn.observe_round(
            state,
            [
                {
                    "tool": "template",
                    "action": "read_file",
                    "args": {"path": ("very-long/" * 100) + f"file-{index}.py"},
                }
            ],
            [{"success": True}],
        )
    assert state.convergence["consecutive_read_only_rounds"] == 4
    assert len(state.convergence["seen_targets"]) == 128
    assert all(len(item) <= 512 for item in state.convergence["seen_targets"])

    step.status = "completed"
    assert next_turn.observe_round(
        state,
        [{"tool": "agent", "action": "update_step", "args": {"step_id": "scope"}}],
        [{"success": True}],
    )
    assert state.convergence["consecutive_read_only_rounds"] == 0
    assert state.convergence["low_yield_rounds"] == 0


def test_implementation_read_notices_report_two_one_zero() -> None:
    state = SimpleNamespace(
        plan=[SimpleNamespace(id="implement", status="in_progress")],
        tool_calls=[
            {
                "request": {"tool": "template", "action": "read_file", "args": {"path": "src/state.ts"}},
                "result": {"success": True},
            }
        ],
        convergence={},
    )
    controller = TaskConvergenceController(
        mode="deep",
        max_rounds=24,
        exploration_round_limit=10,
        reserved_rounds=8,
        implementation_read_limit=2,
    )
    controller.bind(state)
    exact = {"path": "src/state.ts", "start_line": 1, "end_line": 200}

    assert any("At most 2" in message for message in controller.before_round(17, state).messages)
    assert controller.implementation_read_denial(state, "read_file", exact) == ""
    assert any("1 read(s) remaining" in message for message in controller.before_round(18, state).messages)
    assert controller.implementation_read_denial(state, "read_file", exact) == ""
    assert any("0 read(s) remaining" in message for message in controller.before_round(19, state).messages)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"path": "src/new.ts", "start_line": 1, "end_line": 20}, "not read successfully"),
        ({"path": "src/state.ts"}, "explicit positive"),
        ({"path": "src/state.ts", "start_line": 1, "end_line": 201}, "exceeds 200"),
    ],
)
def test_implementation_read_exception_rejects_new_unbounded_or_wide_targets(arguments, message) -> None:
    state = SimpleNamespace(
        plan=[SimpleNamespace(id="implement", status="in_progress")],
        tool_calls=[
            {
                "request": {"tool": "template", "action": "read_file", "args": {"path": "src/state.ts"}},
                "result": {"success": True},
            }
        ],
    )
    controller = TaskConvergenceController(
        mode="deep",
        max_rounds=24,
        exploration_round_limit=10,
        reserved_rounds=8,
        implementation_read_limit=2,
    )

    assert message in controller.implementation_read_denial(state, "read_file", arguments)


def test_implementation_read_exception_is_closed_outside_the_implement_step() -> None:
    state = SimpleNamespace(
        plan=[SimpleNamespace(id="verify", status="in_progress")],
        tool_calls=[],
    )
    controller = TaskConvergenceController(
        mode="deep",
        max_rounds=24,
        exploration_round_limit=10,
        reserved_rounds=8,
        implementation_read_limit=2,
    )

    action = controller.before_round(17, state)

    assert action.guard_implementation_read is False
    assert "read_file" in action.excluded_functions


def test_hard_convergence_requires_plan_transition_before_other_tools() -> None:
    state = SimpleNamespace(
        plan=[
            SimpleNamespace(id="scope", status="completed"),
            SimpleNamespace(id="inspect-chunks", status="in_progress"),
            SimpleNamespace(id="implement", status="pending"),
        ],
        tool_calls=[],
    )
    controller = TaskConvergenceController(
        mode="deep",
        max_rounds=24,
        exploration_round_limit=10,
        reserved_rounds=8,
    )

    action = controller.before_round(17, state)

    assert action.force_plan_transition is True
    assert any("agent_update_step" in message for message in action.messages)


def test_hard_convergence_keeps_transition_gate_when_next_ready_step_is_pending() -> None:
    state = SimpleNamespace(
        plan=[
            SimpleNamespace(id="scope", status="completed"),
            SimpleNamespace(id="inspect-chunks", status="completed"),
            SimpleNamespace(id="implement", status="pending"),
            SimpleNamespace(id="verify", status="pending"),
        ],
        tool_calls=[],
    )
    controller = TaskConvergenceController(
        mode="deep",
        max_rounds=24,
        exploration_round_limit=10,
        reserved_rounds=8,
    )

    action = controller.before_round(17, state)

    assert action.force_plan_transition is True
    assert action.guard_implementation_read is False


def test_implementation_read_notice_is_sent_after_a_late_plan_transition() -> None:
    state = SimpleNamespace(
        plan=[
            SimpleNamespace(id="scope", status="completed"),
            SimpleNamespace(id="inspect-chunks", status="in_progress"),
            SimpleNamespace(id="implement", status="pending"),
        ],
        tool_calls=[
            {
                "request": {
                    "tool": "template",
                    "action": "read_file",
                    "args": {"path": "src/state.ts", "start_line": 1, "end_line": 80},
                },
                "result": {"success": True},
            }
        ],
    )
    controller = TaskConvergenceController(
        mode="deep",
        max_rounds=24,
        exploration_round_limit=10,
        reserved_rounds=8,
        implementation_read_limit=2,
    )

    transition = controller.before_round(17, state)
    assert transition.force_plan_transition is True
    assert any("must transition now" in message for message in transition.messages)

    state.plan[1].status = "completed"
    state.plan[2].status = "in_progress"
    implement = controller.before_round(18, state)

    assert implement.force_plan_transition is False
    assert implement.guard_implementation_read is True
    assert "read_file" not in implement.excluded_functions
    assert any("implement step is now active" in message for message in implement.messages)
    assert any("at most 200 lines" in message for message in implement.messages)
    assert controller.before_round(19, state).messages == ()


def test_hard_phase_explains_conditional_implementation_skip_without_skipping_verify() -> None:
    state = SimpleNamespace(
        plan=[
            SimpleNamespace(id="scope", status="completed"),
            SimpleNamespace(id="inspect-chunks", status="completed"),
            SimpleNamespace(id="implement", status="in_progress"),
            SimpleNamespace(id="verify", status="pending"),
        ],
        task_route={"reasons": ["mutation-request", "conditional-mutation"]},
        tool_calls=[
            {
                "request": {"tool": "template", "action": "read_file", "args": {"path": "src/state.ts"}},
                "result": {"success": True},
            }
        ],
    )
    controller = TaskConvergenceController(
        mode="deep",
        max_rounds=24,
        exploration_round_limit=10,
        reserved_rounds=8,
        implementation_read_limit=2,
    )

    action = controller.before_round(17, state)

    joined = "\n".join(action.messages)
    assert "conditional-mutation plan" in joined
    assert "step_id `implement` and status `skipped`" in joined
    assert "then start `verify`" in joined


def test_hard_phase_does_not_offer_skip_for_unconditional_mutation() -> None:
    state = SimpleNamespace(
        plan=[SimpleNamespace(id="implement", status="in_progress")],
        task_route={"reasons": ["mutation-request"]},
        tool_calls=[],
    )
    controller = TaskConvergenceController(
        mode="deep",
        max_rounds=24,
        exploration_round_limit=10,
        reserved_rounds=8,
    )

    joined = "\n".join(controller.before_round(17, state).messages)

    assert "conditional-mutation plan" not in joined


@pytest.mark.parametrize(
    ("alias", "model_name"),
    [
        ("template.read_file", "read_file"),
        ("template.list_dir", "list_dir"),
        ("shell.run", "shell_run"),
        ("python.run", "python_run"),
    ],
)
def test_tool_manager_normalizes_canonical_aliases_for_runtime_policy(
    tmp_path: Path,
    make_config,
    alias: str,
    model_name: str,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    tools = ToolManager(config, project, memory, yolo=True)

    assert tools.model_function_name(alias) == model_name
    assert tools.model_function_name(model_name) == model_name


class RecordingClient:
    def __init__(self, responses: list[dict | ChatResponse | BaseException]) -> None:
        self.responses = list(responses)
        self.requests: list[list[dict]] = []
        self.schemas: list[list[dict] | None] = []
        self.options: list[dict] = []

    def chat(self, *, messages, tools=None, **_kwargs) -> ChatResponse:
        self.requests.append(list(messages))
        self.schemas.append(tools)
        self.options.append(dict(_kwargs))
        if not self.responses:
            raise AssertionError("fake response queue exhausted")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, ChatResponse):
            return response
        return ChatResponse(message=response, raw={})


class AttemptError(RuntimeError):
    def __init__(self, message: str, *, http_attempt_count: int) -> None:
        super().__init__(message)
        self.http_attempt_count = http_attempt_count


def model_tool_call(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def model_tool_calls(*calls: tuple[str, str, dict]) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
            for call_id, name, arguments in calls
        ],
    }


def test_conditional_large_task_can_complete_with_honest_failed_validation_and_no_mutation(
    tmp_path: Path,
    make_config,
    monkeypatch,
) -> None:
    root = tmp_path / "typescript-snapshot"
    (root / "src").mkdir(parents=True)
    (root / "src" / "engine.ts").write_text("export function answer() {\n  return 42\n}\n", encoding="utf-8")
    (root / "package.json").write_text(
        json.dumps({"scripts": {"typecheck": "tsc --noEmit"}}),
        encoding="utf-8",
    )
    config = make_config(
        {
            "runtime": {
                "task_mode": "deep",
                "max_tool_rounds_hard_limit": 4,
                "convergence": {"reserved_tool_rounds": 1},
            }
        }
    )
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    tools = ToolManager(config, project, memory, yolo=True)
    validation_calls: list[list[str]] = []

    def fail_existing_typecheck(args, **_kwargs):
        validation_calls.append(list(args))
        return ToolResult(False, "existing snapshot type errors", data={"returncode": 2})

    monkeypatch.setattr("agent.tools.templates.run_command", fail_existing_typecheck)
    client = RecordingClient(
        [
            model_tool_calls(
                ("scope-done", "agent_update_step", {"step_id": "scope", "status": "completed"}),
                ("inspect-start", "agent_update_step", {"step_id": "inspect-chunks", "status": "in_progress"}),
            ),
            model_tool_calls(
                (
                    "source-evidence",
                    "read_file",
                    {"path": "src/engine.ts", "start_line": 1, "end_line": 3},
                ),
                (
                    "inspect-done",
                    "agent_update_step",
                    {"step_id": "inspect-chunks", "status": "completed"},
                ),
                ("implement-start", "agent_update_step", {"step_id": "implement", "status": "in_progress"}),
            ),
            model_tool_calls(
                ("validate", "run_tests", {"framework": "npm:typecheck", "path": "."}),
                ("no-change", "agent_update_step", {"step_id": "implement", "status": "skipped"}),
                ("verify-start", "agent_update_step", {"step_id": "verify", "status": "in_progress"}),
            ),
            model_tool_call("verify-done", "agent_update_step", {"step_id": "verify", "status": "completed"}),
            {
                "role": "assistant",
                "content": "静态检查已执行但因研究快照的既有错误失败；未证实独立缺陷，因此未修改文件。",
            },
        ]
    )
    runtime = AgentRuntime(config=config, project=project, memory=memory, tools=tools, client=client)

    answer = runtime.run(
        "全面审计整个大型 TypeScript 代码库；运行静态检查；若找到证据确凿的真实缺陷则修复，"
        "若没有充分证据，‘未找到可证实缺陷’是合格结论，应跳过 implement、完成 verify，不要修改代码；"
        "若快照存在大量基线错误，不得为了‘全绿’而批量补文件或逐条打补丁；"
        "不得读取或输出任何真实凭据；最终回复必须列出项目优点、Bug 证据、修改文件、验证结果和剩余风险"
    )

    assert answer == "静态检查已执行但因研究快照的既有错误失败；未证实独立缺陷，因此未修改文件。"
    assert validation_calls == [["npm", "run", "typecheck"]]
    state = runtime.sessions.load(runtime.last_session_id).state
    assert state.status == "completed"
    assert state.main_loop_model_request_count == 4
    assert state.final_synthesis_model_request_count == 1
    assert "conditional-mutation" in state.task_route["reasons"]
    assert "artifact-required" not in state.task_route["reasons"]
    assert {step.id: step.status for step in state.plan} == {
        "scope": "completed",
        "inspect-chunks": "completed",
        "implement": "skipped",
        "verify": "completed",
    }
    assert not any((item.get("request") or {}).get("tool") == "file" for item in state.tool_calls)
    assert "managed-write" not in answer
    assert "agent resume --session" not in answer
    read_result = next(
        item["result"] for item in state.tool_calls if (item.get("request") or {}).get("action") == "read_file"
    )
    assert read_result["stdout"].splitlines() == [
        "     1→export function answer() {",
        "     2→  return 42",
        "     3→}",
    ]


def test_single_validation_route_blocks_equivalent_shell_and_lsp_retries(
    tmp_path: Path,
    make_config,
    monkeypatch,
) -> None:
    root = tmp_path / "typescript-snapshot"
    root.mkdir()
    (root / "package.json").write_text(
        json.dumps({"scripts": {"typecheck": "tsc --noEmit"}}),
        encoding="utf-8",
    )
    config = make_config({"runtime": {"task_mode": "standard"}})
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    tools = ToolManager(config, project, memory, yolo=True)
    validation_calls: list[list[str]] = []

    def fail_existing_typecheck(args, **_kwargs):
        validation_calls.append(list(args))
        return ToolResult(False, "missing snapshot dependencies", data={"returncode": 2})

    def reject_shell_execution(*_args, **_kwargs):
        raise AssertionError("equivalent shell validation must be denied before execution")

    monkeypatch.setattr("agent.tools.templates.run_command", fail_existing_typecheck)
    monkeypatch.setattr("agent.tools.shell.run_command", reject_shell_execution)
    client = RecordingClient(
        [
            model_tool_call("primary-check", "run_tests", {"framework": "npm:typecheck", "path": "."}),
            model_tool_calls(
                ("equivalent-shell", "shell_run", {"command": "npx tsc --noEmit"}),
                ("equivalent-lsp", "lsp_diagnostics", {"path": "src/example.ts"}),
            ),
            {
                "role": "assistant",
                "content": "只执行了一次项目静态检查；它因快照缺少依赖而失败，已如实记录验证限制。",
            },
        ]
    )
    runtime = AgentRuntime(config=config, project=project, memory=memory, tools=tools, client=client)

    answer = runtime.run("只运行一次项目已有的静态检查，不要重复等价命令；失败时如实报告验证限制")

    assert "只执行了一次" in answer
    assert validation_calls == [["npm", "run", "typecheck"]]
    state = runtime.sessions.load(runtime.last_session_id).state
    assert state.status == "completed"
    assert "single-validation" in state.task_route["reasons"]
    denied = [
        item
        for item in state.tool_calls
        if ((item.get("result") or {}).get("data") or {}).get("runtime_denied") is True
    ]
    assert [(item["request"]["tool"], item["request"]["action"]) for item in denied] == [
        ("shell", "run"),
        ("lsp", "diagnostics"),
    ]
    assert all("single validation attempt" in item["result"]["stderr"] for item in denied)


def test_single_validation_rejects_compound_shell_then_allows_one_real_attempt(
    tmp_path: Path,
    make_config,
    monkeypatch,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "package.json").write_text(json.dumps({"scripts": {"test": "vitest run"}}), encoding="utf-8")
    config = make_config({"runtime": {"task_mode": "standard"}})
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    tools = ToolManager(config, project, memory, yolo=True)
    validation_calls: list[list[str]] = []

    def forbidden_compound(*_args, **_kwargs):
        raise AssertionError("compound validation must be denied before shell execution")

    def run_one(args, **_kwargs):
        validation_calls.append(list(args))
        return ToolResult(True, "one validation passed", data={"returncode": 0})

    monkeypatch.setattr("agent.tools.shell.run_command", forbidden_compound)
    monkeypatch.setattr("agent.tools.templates.run_command", run_one)
    client = RecordingClient(
        [
            model_tool_call("compound", "shell_run", {"command": "pytest && npm test"}),
            model_tool_call("single", "run_tests", {"framework": "npm:test", "path": "."}),
            {"role": "assistant", "content": "只执行了一次测试并通过。"},
        ]
    )
    runtime = AgentRuntime(config=config, project=project, memory=memory, tools=tools, client=client)

    assert runtime.run("只运行一次验证并报告结果") == "只执行了一次测试并通过。"
    assert validation_calls == [["npm", "run", "test"]]
    state = runtime.sessions.load(runtime.last_session_id).state
    assert state.status == "completed"
    assert state.main_loop_model_request_count == 3
    denied = state.tool_calls[0]["result"]
    assert denied["data"] == {"runtime_denied": True, "not_executed": True}
    assert "contains 2 validation commands" in denied["stderr"]


def test_single_validation_same_batch_skips_invalid_call_then_executes_one_valid_call(
    tmp_path: Path,
    make_config,
    monkeypatch,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "package.json").write_text(json.dumps({"scripts": {"test": "vitest run"}}), encoding="utf-8")
    config = make_config({"runtime": {"task_mode": "standard"}})
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    tools = ToolManager(config, project, memory, yolo=True)
    validation_calls: list[list[str]] = []

    def run_one(args, **_kwargs):
        validation_calls.append(list(args))
        return ToolResult(True, "one validation passed", data={"returncode": 0})

    monkeypatch.setattr("agent.tools.templates.run_command", run_one)
    client = RecordingClient(
        [
            model_tool_calls(
                ("invalid", "run_tests", {"unknown": True}),
                ("valid", "run_tests", {"framework": "npm:test", "path": "."}),
            ),
            {"role": "assistant", "content": "无效调用未执行，随后只执行了一次有效测试。"},
        ]
    )
    runtime = AgentRuntime(config=config, project=project, memory=memory, tools=tools, client=client)

    assert runtime.run("只运行一次验证并报告结果") == "无效调用未执行，随后只执行了一次有效测试。"
    assert validation_calls == [["npm", "run", "test"]]
    state = runtime.sessions.load(runtime.last_session_id).state
    assert state.status == "completed"
    assert state.tool_calls[0]["result"]["data"]["not_executed"] is True
    assert state.tool_calls[1]["result"]["success"] is True


@pytest.mark.parametrize(
    "command",
    [
        "bash -lc 'npm test'",
        "env -i CHECK=1 command make test",
        "exec tox",
        "cd src && python -m pytest -q",
        "npm --prefix src test",
        "uv run pytest",
        "timeout 60 nox",
        "python -I -m pytest",
    ],
)
def test_single_validation_recognizes_wrapped_validation_commands(command: str) -> None:
    assert AgentRuntime._looks_like_validation_shell({"command": command}) is True


@pytest.mark.parametrize("command", ["pytest && npm test", "bash -lc 'pytest && pytest'"])
def test_single_validation_counts_each_command_inside_compound_shell(command: str) -> None:
    assert AgentRuntime._shell_command_validation_count(command) == 2


@pytest.mark.parametrize("command", ["echo pytest", "printf 'npm test'", "make clean", "git diff --staged"])
def test_single_validation_does_not_count_mentions_or_read_only_evidence(command: str) -> None:
    assert AgentRuntime._looks_like_validation_shell({"command": command}) is False


def test_main_loop_rejects_dsml_answer_text_then_accepts_protocol_valid_answer(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config({"runtime": {"task_mode": "simple", "max_tool_rounds_hard_limit": 4}})
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    client = RecordingClient(
        [
            {
                "role": "assistant",
                "content": (
                    "这里是调用：\n```\n<｜｜DSML｜｜tool_calls>\n"
                    '<｜｜DSML｜｜invoke name="list_dir">.</｜｜DSML｜｜invoke>\n```'
                ),
            },
            {"role": "assistant", "content": "协议有效的最终答复。"},
        ]
    )
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )

    answer = runtime.run("回答一个简单问题")

    assert answer == "协议有效的最终答复。"
    assert len(client.requests) == 2
    assert any(
        "never print DSML" in str(item.get("content") or "")
        for item in client.requests[1]
        if item.get("role") == "system"
    )
    state = runtime.sessions.load(runtime.last_session_id).state
    assert state.status == "completed"
    assert state.tool_calls == []
    stored = "\n".join(
        str(item.get("content") or "") for item in runtime.sessions.load(runtime.last_session_id).messages
    )
    assert "<｜｜DSML｜｜tool_calls>" not in stored


def test_main_loop_discards_structured_calls_when_content_contains_dsml(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config({"runtime": {"task_mode": "simple", "max_tool_rounds_hard_limit": 4}})
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    mixed = model_tool_call("must-not-run", "list_dir", {"path": ".", "depth": 1})
    mixed["content"] = "preface\n<｜｜DSML｜｜tool_calls>\nstructured call follows"
    client = RecordingClient([mixed, {"role": "assistant", "content": "安全的最终答复。"}])
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )

    assert runtime.run("回答一个简单问题") == "安全的最终答复。"
    state = runtime.sessions.load(runtime.last_session_id).state
    assert state.status == "completed"
    assert state.tool_calls == []
    stored = "\n".join(
        str(item.get("content") or "") for item in runtime.sessions.load(runtime.last_session_id).messages
    )
    assert "<｜｜DSML｜｜tool_calls>" not in stored
    assert "all calls were discarded" in stored


def test_tool_call_normalization_is_stable_for_missing_and_duplicate_ids() -> None:
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"function": {"name": "read_file", "arguments": '{"path":"a.txt"}'}},
            {"id": "duplicate", "function": {"name": "read_file", "arguments": '{"path":"b.txt"}'}},
            {"id": "duplicate", "function": {"name": "read_file", "arguments": '{"path":"c.txt"}'}},
        ],
    }

    first, first_changed, first_dropped = _normalize_assistant_tool_calls(message, turn=1, round_number=2)
    second, second_changed, second_dropped = _normalize_assistant_tool_calls(message, turn=1, round_number=2)
    ids = [call["id"] for call in first["tool_calls"]]

    assert first_changed == second_changed == 3
    assert first_dropped == second_dropped == 0
    assert first == second
    assert len(ids) == len(set(ids)) == 3
    assert all(ids)
    assert ids[1] == "duplicate"
    assert ids[0].startswith("deep-agent-call-t1-r2-i1")
    assert ids[2].startswith("deep-agent-call-t1-r2-i3")


def test_runtime_executes_all_normalized_missing_and_duplicate_id_calls(
    tmp_path: Path,
    make_config,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    for name in ("a.txt", "b.txt", "c.txt"):
        (root / name).write_text(f"evidence from {name}\n", encoding="utf-8")
    config = make_config({"runtime": {"task_mode": "standard", "max_tool_rounds": 2}})
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    malformed = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"function": {"name": "read_file", "arguments": json.dumps({"path": "a.txt"})}},
            {
                "id": "duplicate",
                "function": {"name": "read_file", "arguments": json.dumps({"path": "b.txt"})},
            },
            {
                "id": "duplicate",
                "function": {"name": "read_file", "arguments": json.dumps({"path": "c.txt"})},
            },
        ],
    }
    client = RecordingClient([malformed, {"role": "assistant", "content": "三项证据均已总结。"}])
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )

    assert runtime.run("检查三个文件并总结") == "三项证据均已总结。"

    second_request = client.requests[1]
    assistant_index = next(index for index, item in enumerate(second_request) if len(item.get("tool_calls") or []) == 3)
    normalized_ids = [call["id"] for call in second_request[assistant_index]["tool_calls"]]
    result_ids = [item.get("tool_call_id") for item in second_request[assistant_index + 1 : assistant_index + 4]]
    assert len(normalized_ids) == len(set(normalized_ids)) == 3
    assert result_ids == normalized_ids
    assert repair_tool_message_pairs(second_request).changed is False
    state = runtime.sessions.load(runtime.last_session_id).state
    assert [item["request"]["request_id"] for item in state.tool_calls] == normalized_ids
    assert all(item["result"]["success"] is True for item in state.tool_calls)


def test_runtime_denies_excess_tool_calls_without_executing_handlers_and_keeps_pairs(
    tmp_path: Path,
    make_config,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    for index in range(4):
        (root / f"file-{index}.txt").write_text(f"evidence {index}\n", encoding="utf-8")
    config = make_config(
        {
            "runtime": {
                "task_mode": "standard",
                "max_tool_rounds": 2,
                "convergence": {"max_tool_calls_per_round": 2},
            }
        }
    )
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    tools = ToolManager(config, project, memory, yolo=True)
    capability, original_handler = tools.registry.resolve("read_file")
    assert capability is not None and original_handler is not None
    executed: list[str] = []

    def counting_handler(**kwargs):
        executed.append(str(kwargs.get("path") or ""))
        return original_handler(**kwargs)

    tools.registry._handlers[capability.name] = counting_handler
    expected_ids = [f"bounded-{index}" for index in range(4)]
    client = RecordingClient(
        [
            model_tool_calls(
                *[(call_id, "read_file", {"path": f"file-{index}.txt"}) for index, call_id in enumerate(expected_ids)]
            ),
            {"role": "assistant", "content": "已保留所有调用的配对结果。"},
        ]
    )
    runtime = AgentRuntime(config=config, project=project, memory=memory, tools=tools, client=client)

    assert runtime.run("检查四个文件并总结") == "已保留所有调用的配对结果。"

    assert sorted(executed) == ["file-0.txt", "file-1.txt"]
    state = runtime.sessions.load(runtime.last_session_id).state
    assert len(state.tool_calls) == 4
    assert [item["result"]["success"] for item in state.tool_calls] == [True, True, False, False]
    assert all(item["result"]["data"]["runtime_denied"] is True for item in state.tool_calls[2:])
    assert all("per-round limit of 2" in item["result"]["stderr"] for item in state.tool_calls[2:])
    second_request = client.requests[1]
    assistant_index = next(index for index, item in enumerate(second_request) if len(item.get("tool_calls") or []) == 4)
    result_batch = second_request[assistant_index + 1 : assistant_index + 5]
    assert [item.get("tool_call_id") for item in result_batch] == expected_ids
    assert repair_tool_message_pairs(second_request).changed is False


def test_runtime_hard_caps_protocol_projection_and_reports_calls_beyond_sixty_four(
    tmp_path: Path,
    make_config,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "evidence.txt").write_text("bounded evidence\n", encoding="utf-8")
    config = make_config(
        {
            "runtime": {
                "task_mode": "standard",
                "max_tool_rounds": 2,
                "convergence": {"max_tool_calls_per_round": 16},
            }
        }
    )
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    tools = ToolManager(config, project, memory, yolo=True)
    capability, original_handler = tools.registry.resolve("read_file")
    assert capability is not None and original_handler is not None
    executed: list[str] = []

    def counting_handler(**kwargs):
        executed.append(str(kwargs.get("path") or ""))
        return original_handler(**kwargs)

    tools.registry._handlers[capability.name] = counting_handler
    all_ids = [f"protocol-{index}" for index in range(70)]
    client = RecordingClient(
        [
            model_tool_calls(*[(call_id, "read_file", {"path": "evidence.txt"}) for call_id in all_ids]),
            {"role": "assistant", "content": "协议硬限和配对结果均已验证。"},
        ]
    )
    runtime = AgentRuntime(config=config, project=project, memory=memory, tools=tools, client=client)

    assert runtime.run("检查大量并行读取请求并总结") == "协议硬限和配对结果均已验证。"

    assert len(executed) == 16
    state = runtime.sessions.load(runtime.last_session_id).state
    assert len(state.tool_calls) == 64
    assert [item["request"]["request_id"] for item in state.tool_calls] == all_ids[:64]
    assert all(item["result"]["success"] is True for item in state.tool_calls[:16])
    assert all(item["result"]["data"]["runtime_denied"] is True for item in state.tool_calls[16:])

    second_request = client.requests[1]
    assistant_index = next(
        index for index, item in enumerate(second_request) if len(item.get("tool_calls") or []) == 64
    )
    retained_ids = [call["id"] for call in second_request[assistant_index]["tool_calls"]]
    result_batch = second_request[assistant_index + 1 : assistant_index + 65]
    assert retained_ids == all_ids[:64]
    assert [item.get("tool_call_id") for item in result_batch] == retained_ids
    notice = second_request[assistant_index + 65]
    assert notice["role"] == "system"
    assert "dropped 6 tool calls" in notice["content"]
    assert "hard protocol limit of 64" in notice["content"]
    assert not any(call_id in str(second_request) for call_id in all_ids[64:])
    assert repair_tool_message_pairs(second_request).changed is False


@pytest.mark.parametrize(
    "convergence_overrides",
    [
        {"enabled": True, "auto_compaction_enabled": False},
        {"enabled": False, "auto_compaction_enabled": True},
    ],
    ids=("auto-compaction-disabled", "convergence-disabled"),
)
@pytest.mark.parametrize(
    "request_size",
    ["within-budget", "over-limit"],
)
def test_runtime_request_budget_is_enforced_when_optional_compaction_is_disabled(
    tmp_path: Path,
    make_config,
    monkeypatch,
    convergence_overrides: dict[str, bool],
    request_size: str,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config(
        {
            "model": {"context_window_tokens": 8_192, "max_tokens": 1_024},
            "runtime": {
                "task_mode": "standard",
                "max_tool_rounds": 1,
                "convergence": {
                    "context_safety_buffer_tokens": 1_024,
                    **convergence_overrides,
                },
            },
        }
    )
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    tools = ToolManager(config, project, memory, yolo=True)
    monkeypatch.setattr(tools, "schemas", lambda: [])
    client = RecordingClient(
        [] if request_size == "over-limit" else [{"role": "assistant", "content": "预算内请求正常完成。"}]
    )
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=tools,
        client=client,
    )

    if request_size == "over-limit":
        with pytest.raises(RuntimeError, match="model request exceeds the configured context window"):
            runtime.run("超长请求" * 10_000)
        assert client.requests == []
        state = runtime.sessions.load(runtime.last_session_id).state
        assert state.model_request_count == 0
        return

    assert runtime.run("回答当前问题") == "预算内请求正常完成。"
    assert len(client.requests) == 1
    budget = ContextWindowController(
        context_window_tokens=8_192,
        safety_buffer_tokens=1_024,
        keep_recent_rounds=4,
        failure_limit=3,
    ).budget(client.requests[0], [], max_output_tokens=1_024)
    assert budget.output_reserve_tokens == 1_024
    assert budget.over_trigger is False
    assert budget.over_limit is False
    assert not any(
        str(item.get("content") or "").startswith("[Deep Agent emergency context collapse]")
        for item in client.requests[0]
    )


def test_auto_compaction_off_preserves_over_trigger_request_below_hard_limit(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    client = RecordingClient([])
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )
    state = AgentState.create(
        session_id="over-trigger-auto-off",
        project=project,
        user_request="continue",
        loaded_memories=[],
        loaded_tools=[],
        git_branch=None,
        context_index_path=str(project.agent_dir / "index.json"),
    )
    messages = [
        {"role": "system", "content": "objective"},
        {"role": "user", "content": "continue"},
        *tool_pair(1, chars=11_000),
        *tool_pair(2, chars=11_000),
    ]
    original = json.loads(json.dumps(messages))
    context_window = ContextWindowController(
        context_window_tokens=8_192,
        safety_buffer_tokens=1_024,
        keep_recent_rounds=1,
        failure_limit=3,
    )
    context_window.bind(state)
    model_route = ModelRoute(
        provider="deepseek",
        tier="standard",
        model="deepseek-v4-pro",
        thinking_enabled=True,
        reasoning_effort="high",
        max_tokens=1_024,
        reasons=("test",),
    )
    initial_budget = context_window.budget(messages, None, max_output_tokens=model_route.max_tokens)
    assert initial_budget.over_trigger is True
    assert initial_budget.over_limit is False

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
    assert client.requests == []


def test_runtime_enforces_closed_exploration_phase_against_canonical_alias(
    tmp_path: Path,
    make_config,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "evidence.txt").write_text("bounded evidence\n", encoding="utf-8")
    config = make_config(
        {
            "runtime": {
                "task_mode": "deep",
                "max_tool_rounds_hard_limit": 2,
                "convergence": {"reserved_tool_rounds": 1},
            }
        }
    )
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    client = RecordingClient(
        [
            model_tool_call("initial-read", "read_file", {"path": "evidence.txt"}),
            model_tool_call("alias-read", "template.read_file", {"path": "evidence.txt"}),
            {"role": "assistant", "content": "The closed exploration phase was enforced."},
        ]
    )
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )

    runtime.run("全面审计整个项目并总结证据")

    state = runtime.sessions.load(runtime.last_session_id).state
    alias_result = state.tool_calls[1]["result"]
    assert alias_result["success"] is False
    assert alias_result["data"]["runtime_denied"] is True
    assert "agent_update_step" in alias_result["stderr"]


def test_runtime_hard_phase_schema_forces_plan_transition_before_status_or_tests(
    tmp_path: Path,
    make_config,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config(
        {
            "runtime": {
                "task_mode": "deep",
                "max_tool_rounds_hard_limit": 2,
                "convergence": {"reserved_tool_rounds": 1},
            }
        }
    )
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    client = RecordingClient(
        [
            model_tool_call(
                "start-inspect", "agent_update_step", {"step_id": "inspect-chunks", "status": "in_progress"}
            ),
            model_tool_call("forbidden-status", "git_status", {}),
            {"role": "assistant", "content": "The plan-transition gate was enforced."},
        ]
    )
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )

    runtime.run("全面审计整个项目并总结证据")

    second_schema_names = {str((item.get("function") or {}).get("name") or "") for item in client.schemas[1] or []}
    assert second_schema_names == {"agent_update_step"}
    state = runtime.sessions.load(runtime.last_session_id).state
    assert state.tool_calls[1]["result"]["success"] is False
    assert state.tool_calls[1]["result"]["data"]["runtime_denied"] is True
    assert "agent_update_step" in state.tool_calls[1]["result"]["stderr"]


def test_runtime_allows_canonical_bounded_read_of_known_path_during_implement(
    tmp_path: Path,
    make_config,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "evidence.txt").write_text("bounded evidence\n", encoding="utf-8")
    config = make_config(
        {
            "runtime": {
                "task_mode": "deep",
                "max_tool_rounds_hard_limit": 2,
                "convergence": {
                    "reserved_tool_rounds": 1,
                    "max_implementation_evidence_reads": 2,
                },
            }
        }
    )
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    client = RecordingClient(
        [
            model_tool_call(
                "initial-read",
                "read_file",
                {"path": "evidence.txt", "start_line": 1, "end_line": 1},
            ),
            model_tool_call(
                "bounded-alias-read",
                "template.read_file",
                {"path": "evidence.txt", "start_line": 1, "end_line": 1},
            ),
            {"role": "assistant", "content": "The bounded implementation evidence read was verified."},
        ]
    )
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )

    runtime.run(
        "全面审计整个项目并修复一个问题",
        initial_plan=[
            {
                "id": "implement",
                "title": "Implement bounded change",
                "status": "in_progress",
                "completion_criteria": "The managed change is applied.",
            },
            {
                "id": "verify",
                "title": "Verify the change",
                "dependencies": ["implement"],
                "completion_criteria": "The change is verified.",
            },
        ],
    )

    state = runtime.sessions.load(runtime.last_session_id).state
    assert state.tool_calls[1]["result"]["success"] is True
    assert state.tool_calls[1]["request"]["model_name"] == "template.read_file"
    second_schema_names = {str((item.get("function") or {}).get("name") or "") for item in client.schemas[1] or []}
    assert "read_file" in second_schema_names
    second_request = client.requests[1]
    assert any("exact path already read successfully" in str(item.get("content") or "") for item in second_request)


def test_runtime_bounds_five_same_round_tool_results_without_breaking_pairs(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    for index in range(5):
        (root / f"large-{index}.txt").write_text((f"evidence-{index} " * 1_000) + "\n", encoding="utf-8")
    config = make_config(
        {
            "runtime": {
                "task_mode": "standard",
                "max_tool_rounds": 2,
                "convergence": {
                    "single_tool_result_chars": 3_000,
                    "same_round_tool_result_chars": 4_096,
                    "aggregate_tool_result_chars": 200_000,
                    "output_reserve_chars": 0,
                    "compacted_tool_result_chars": 400,
                    "auto_compaction_enabled": False,
                },
            }
        }
    )
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    expected_ids = [f"same-round-{index}" for index in range(5)]
    client = RecordingClient(
        [
            model_tool_calls(
                *[(call_id, "read_file", {"path": f"large-{index}.txt"}) for index, call_id in enumerate(expected_ids)]
            ),
            {"role": "assistant", "content": "五项工具证据已完成有界综合。"},
        ]
    )
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )

    assert runtime.run("检查五个文件并总结") == "五项工具证据已完成有界综合。"

    second_request = client.requests[1]
    assistant_index = next(index for index, item in enumerate(second_request) if len(item.get("tool_calls") or []) == 5)
    batch = second_request[assistant_index + 1 : assistant_index + 6]
    assert [item.get("role") for item in batch] == ["tool"] * 5
    assert [item.get("tool_call_id") for item in batch] == expected_ids
    assert sum(len(str(item.get("content") or "")) for item in batch) <= 4_096
    assert repair_tool_message_pairs(second_request).changed is False
    state = runtime.sessions.load(runtime.last_session_id).state
    assert len(state.tool_calls) == 5
    assert all(item["result"]["success"] is True for item in state.tool_calls)


def test_runtime_low_context_uses_non_thinking_tool_free_compaction_and_valid_main_request(
    tmp_path: Path,
    make_config,
    monkeypatch,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    for index in range(2):
        (root / f"history-{index}.txt").write_text((str(index) * 30_000) + "\n", encoding="utf-8")
    config = make_config(
        {
            "model": {"context_window_tokens": 12_288, "max_tokens": 1_024},
            "tools": {"tool_result": {"preview_chars": 30_000}},
            "runtime": {
                "task_mode": "standard",
                "max_tool_rounds": 3,
                "convergence": {
                    "single_tool_result_chars": 30_000,
                    "same_round_tool_result_chars": 50_000,
                    "aggregate_tool_result_chars": 200_000,
                    "output_reserve_chars": 0,
                    "keep_recent_tool_results": 1,
                    "auto_compaction_enabled": True,
                    "auto_compaction_max_tokens": 512,
                    "context_safety_buffer_tokens": 1_024,
                },
            },
        }
    )
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    tools = ToolManager(config, project, memory, yolo=True)
    active_schema = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read one bounded project file.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    monkeypatch.setattr(tools, "schemas", lambda: active_schema)
    client = RecordingClient(
        [
            ChatResponse(
                message=model_tool_call("history-0", "read_file", {"path": "history-0.txt"}),
                raw={},
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
                http_attempt_count=1,
            ),
            ChatResponse(
                message=model_tool_call("history-1", "read_file", {"path": "history-1.txt"}),
                raw={},
                finish_reason="tool_calls",
                usage={"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22},
                http_attempt_count=2,
            ),
            ChatResponse(
                message={
                    "role": "assistant",
                    "content": "history-0.txt 已成功读取；保留路径、成功状态和后续总结任务。",
                },
                raw={},
                finish_reason="stop",
                usage={"prompt_tokens": 30, "completion_tokens": 3, "total_tokens": 33},
                http_attempt_count=1,
            ),
            ChatResponse(
                message={"role": "assistant", "content": "压缩后继续使用最近完整工具轮次并完成总结。"},
                raw={},
                finish_reason="stop",
                usage={"prompt_tokens": 40, "completion_tokens": 4, "total_tokens": 44},
                http_attempt_count=1,
            ),
        ]
    )
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=tools,
        client=client,
    )

    assert runtime.run("检查两个历史文件并总结") == "压缩后继续使用最近完整工具轮次并完成总结。"

    compaction_indexes = [
        index
        for index, request in enumerate(client.requests)
        if request and "Compress prior Deep Agent API rounds" in str(request[0].get("content") or "")
    ]
    assert len(compaction_indexes) == 1
    compaction_index = compaction_indexes[0]
    assert client.schemas[compaction_index] is None
    assert client.options[compaction_index]["tool_choice"] is None
    assert client.options[compaction_index]["thinking"] is False
    assert client.options[compaction_index]["reasoning_effort"] is None

    main_index = compaction_index + 1
    main_request = client.requests[main_index]
    assert client.schemas[main_index] == active_schema
    assert any(
        str(item.get("content") or "").startswith("[Deep Agent automatic context summary]")
        for item in main_request
        if item.get("role") == "system"
    )
    assert repair_tool_message_pairs(main_request).changed is False
    budget = ContextWindowController(
        context_window_tokens=12_288,
        safety_buffer_tokens=1_024,
        keep_recent_rounds=1,
        failure_limit=3,
    ).budget(main_request, active_schema, max_output_tokens=1_024)
    assert budget.over_trigger is False
    assert budget.over_limit is False
    state = runtime.sessions.load(runtime.last_session_id).state
    assert state.main_loop_model_request_count == 3
    assert state.context_compaction_model_request_count == 1
    assert state.final_synthesis_model_request_count == 0
    assert state.model_request_count == 4
    assert state.model_metrics == {
        "http_attempt_count": 5,
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "total_tokens": 110,
    }
    assert state.convergence["context_compaction_count"] == 1
    assert state.convergence["latest_transition"] == "context_compacted"
    assert state.convergence["phase"] == "tool_loop"
    assert state.round == 3


def test_runtime_uses_emergency_collapse_after_three_compaction_failures_without_fourth_call(
    tmp_path: Path,
    make_config,
    monkeypatch,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    client = RecordingClient([])
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )
    state = AgentState.create(
        session_id="context-circuit",
        project=project,
        user_request="continue after compaction failures",
        loaded_memories=[],
        loaded_tools=[],
        git_branch=None,
        context_index_path=str(project.agent_dir / "index.json"),
    )
    state.round = 2
    state.record_tool_call(
        {"tool": "template", "action": "read_file", "args": {"path": "verified-evidence.py"}},
        {
            "success": True,
            "stdout": "verified evidence CRITICAL=42",
            "stderr": "",
            "data": {"path": "verified-evidence.py"},
        },
    )
    messages = [
        {"role": "system", "content": "bounded objective"},
        {"role": "user", "content": "continue"},
        *tool_pair(1, chars=16_000),
        {"role": "system", "content": "old oversized context " + ("历史证据" * 10_000)},
        *tool_pair(2, chars=16_000),
    ]
    context_window = ContextWindowController(
        context_window_tokens=8_192,
        safety_buffer_tokens=1_024,
        keep_recent_rounds=1,
        failure_limit=3,
    )
    context_window.bind(state)
    model_route = ModelRoute(
        provider="deepseek",
        tier="standard",
        model="deepseek-v4-pro",
        thinking_enabled=True,
        reasoning_effort="high",
        max_tokens=1_024,
        reasons=("test",),
    )
    for _ in range(3):
        assert (
            runtime._auto_compact_context(
                state,
                [{"role": "user", "content": "no complete API round is available"}],
                tools=None,
                model_route=model_route,
                context_window=context_window,
                auto_compaction_max_tokens=512,
                phase="tool_loop",
            )
            is False
        )
    assert context_window.failure_count == 3
    assert context_window.circuit_open is True
    checkpointed = runtime.sessions.load(state.session_id).state
    assert checkpointed.convergence["context_compaction_failure_count"] == 3
    assert checkpointed.convergence["context_compaction_circuit_open"] is True
    assert checkpointed.convergence["latest_transition"] == "context_compaction_failed"
    automatic_calls = 0

    def forbidden_fourth_call(*_args, **_kwargs):
        nonlocal automatic_calls
        automatic_calls += 1
        return False

    monkeypatch.setattr(runtime, "_auto_compact_context", forbidden_fourth_call)

    runtime._prepare_model_request(
        state,
        messages,
        tools=None,
        model_route=model_route,
        context_window=context_window,
        history_compactor=None,
        auto_compaction_enabled=True,
        auto_compaction_max_tokens=512,
        phase="tool_loop",
        checkpoint=False,
    )

    assert automatic_calls == 0
    assert client.requests == []
    assert any(
        str(item.get("content") or "").startswith("[Deep Agent emergency context collapse]") for item in messages
    )
    emergency = next(
        str(item.get("content") or "")
        for item in messages
        if str(item.get("content") or "").startswith("[Deep Agent emergency context collapse]")
    )
    assert "template.read_file" in emergency
    assert "verified-evidence.py" in emergency
    assert "CRITICAL=42" in emergency
    assert repair_tool_message_pairs(messages).changed is False
    budget = context_window.budget(messages, None, max_output_tokens=1_024)
    assert budget.over_limit is False
    assert state.convergence["latest_transition"] == "context_emergency_collapsed"
    assert state.convergence["phase"] == "tool_loop"


def test_context_compaction_network_failure_records_attempts_and_checkpoint(
    tmp_path: Path,
    make_config,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    client = RecordingClient([AttemptError("compaction transport failed", http_attempt_count=2)])
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )
    state = AgentState.create(
        session_id="compaction-network-failure",
        project=project,
        user_request="continue",
        loaded_memories=[],
        loaded_tools=[],
        git_branch=None,
        context_index_path=str(project.agent_dir / "index.json"),
    )
    messages = [{"role": "user", "content": "continue"}, *tool_pair(1, chars=100), *tool_pair(2, chars=100)]
    context_window = ContextWindowController(
        context_window_tokens=8_192,
        safety_buffer_tokens=1_024,
        keep_recent_rounds=1,
        failure_limit=3,
    )
    context_window.bind(state)
    model_route = ModelRoute(
        provider="deepseek",
        tier="standard",
        model="deepseek-v4-pro",
        thinking_enabled=True,
        reasoning_effort="high",
        max_tokens=1_024,
        reasons=("test",),
    )

    assert (
        runtime._auto_compact_context(
            state,
            messages,
            tools=None,
            model_route=model_route,
            context_window=context_window,
            auto_compaction_max_tokens=512,
            phase="tool_loop",
        )
        is False
    )

    checkpointed = runtime.sessions.load(state.session_id).state
    assert checkpointed.model_metrics == {"http_attempt_count": 2}
    assert checkpointed.context_compaction_model_request_count == 1
    assert checkpointed.convergence["context_compaction_failure_count"] == 1
    assert checkpointed.convergence["latest_transition"] == "context_compaction_failed"
    assert checkpointed.convergence["phase"] == "tool_loop"


def test_context_compaction_rejects_unusable_finish_reason(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    client = RecordingClient(
        [
            ChatResponse(
                message={"role": "assistant", "content": "filtered summary must not be used"},
                raw={},
                finish_reason="content_filter",
                usage={"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
                http_attempt_count=1,
            )
        ]
    )
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )
    state = AgentState.create(
        session_id="compaction-content-filter",
        project=project,
        user_request="continue",
        loaded_memories=[],
        loaded_tools=[],
        git_branch=None,
        context_index_path=str(project.agent_dir / "index.json"),
    )
    messages = [{"role": "user", "content": "continue"}, *tool_pair(1, chars=100), *tool_pair(2, chars=100)]
    context_window = ContextWindowController(
        context_window_tokens=8_192,
        safety_buffer_tokens=1_024,
        keep_recent_rounds=1,
        failure_limit=3,
    )
    context_window.bind(state)
    model_route = ModelRoute(
        provider="deepseek",
        tier="standard",
        model="deepseek-v4-pro",
        thinking_enabled=True,
        reasoning_effort="high",
        max_tokens=1_024,
        reasons=("test",),
    )

    assert (
        runtime._auto_compact_context(
            state,
            messages,
            tools=None,
            model_route=model_route,
            context_window=context_window,
            auto_compaction_max_tokens=512,
            phase="tool_loop",
        )
        is False
    )

    checkpointed = runtime.sessions.load(state.session_id).state
    assert checkpointed.model_metrics == {
        "http_attempt_count": 1,
        "prompt_tokens": 10,
        "completion_tokens": 1,
        "total_tokens": 11,
    }
    assert checkpointed.context_compaction_model_request_count == 1
    assert checkpointed.convergence["context_compaction_failure_count"] == 1
    assert checkpointed.convergence["latest_transition"] == "context_compaction_failed"
    assert checkpointed.convergence["phase"] == "tool_loop"


def test_drop_oldest_api_round_handles_text_only_assistant_rounds() -> None:
    history = [
        {"role": "assistant", "content": "old text-only analysis"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1", "function": {}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "newer evidence"},
    ]

    reduced = AgentRuntime._drop_oldest_api_round(history)

    assert reduced == history[1:]


def test_runtime_does_not_execute_tool_calls_from_length_truncated_response(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "evidence.txt").write_text("evidence\n", encoding="utf-8")
    config = make_config({"runtime": {"task_mode": "standard", "max_tool_rounds": 1}})
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    tools = ToolManager(config, project, memory, yolo=True)
    capability, original_handler = tools.registry.resolve("read_file")
    assert capability is not None and original_handler is not None
    executed = 0

    def counting_handler(**kwargs):
        nonlocal executed
        executed += 1
        return original_handler(**kwargs)

    tools.registry._handlers[capability.name] = counting_handler
    client = RecordingClient(
        [
            ChatResponse(
                message=model_tool_call("truncated", "read_file", {"path": "evidence.txt"}),
                raw={},
                finish_reason="length",
            ),
            ChatResponse(
                message={"role": "assistant", "content": "截断调用未执行，已安全完成。"},
                raw={},
                finish_reason="stop",
            ),
        ]
    )
    runtime = AgentRuntime(config=config, project=project, memory=memory, tools=tools, client=client)

    assert runtime.run("检查文件并总结") == "截断调用未执行，已安全完成。"
    assert executed == 0
    state = runtime.sessions.load(runtime.last_session_id).state
    assert state.tool_calls == []
    assert state.main_loop_model_request_count == 2


def test_runtime_does_not_execute_tool_calls_from_content_filtered_response(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "evidence.txt").write_text("evidence\n", encoding="utf-8")
    config = make_config({"runtime": {"task_mode": "standard", "max_tool_rounds": 1}})
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    tools = ToolManager(config, project, memory, yolo=True)
    capability, original_handler = tools.registry.resolve("read_file")
    assert capability is not None and original_handler is not None
    executed = 0

    def counting_handler(**kwargs):
        nonlocal executed
        executed += 1
        return original_handler(**kwargs)

    tools.registry._handlers[capability.name] = counting_handler
    client = RecordingClient(
        [
            ChatResponse(
                message=model_tool_call("filtered-call", "read_file", {"path": "evidence.txt"}),
                raw={},
                finish_reason="content_filter",
                usage={"prompt_tokens": 11, "completion_tokens": 1, "total_tokens": 12},
                http_attempt_count=1,
            ),
            ChatResponse(
                message={"role": "assistant", "content": "异常响应已丢弃，未执行工具。"},
                raw={},
                finish_reason="stop",
                usage={"prompt_tokens": 13, "completion_tokens": 4, "total_tokens": 17},
                http_attempt_count=1,
            ),
        ]
    )
    runtime = AgentRuntime(config=config, project=project, memory=memory, tools=tools, client=client)

    assert runtime.run("检查文件并总结") == "异常响应已丢弃，未执行工具。"
    assert executed == 0
    assert not any(item.get("tool_calls") for item in client.requests[1])
    assert any("None of its tool calls were executed" in str(item.get("content")) for item in client.requests[1])
    state = runtime.sessions.load(runtime.last_session_id).state
    assert state.tool_calls == []
    assert state.main_loop_model_request_count == 2
    assert state.convergence["abnormal_finish_recovery_count"] == 1
    assert state.model_metrics == {
        "http_attempt_count": 2,
        "prompt_tokens": 24,
        "completion_tokens": 5,
        "total_tokens": 29,
    }


def test_repeated_unusable_finish_reason_fails_with_resume_and_zero_tool_execution(
    tmp_path: Path,
    make_config,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config({"runtime": {"task_mode": "standard", "max_tool_rounds": 1}})
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    client = RecordingClient(
        [
            ChatResponse(
                message=model_tool_call("filtered", "list_dir", {"path": ".", "depth": 1}),
                raw={},
                finish_reason="content_filter",
            ),
            ChatResponse(
                message=model_tool_call("unknown", "list_dir", {"path": ".", "depth": 1}),
                raw={},
                finish_reason="provider_specific_unknown",
            ),
        ]
    )
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )

    answer = runtime.run("检查项目")

    assert "任务尚未完成" in answer
    assert "finish_reason=provider_specific_unknown" in answer
    assert "agent resume --session" in answer
    state = runtime.sessions.load(runtime.last_session_id).state
    assert state.status == "failed"
    assert state.tool_calls == []
    assert state.main_loop_model_request_count == 2
    assert state.convergence["abnormal_finish_recovery_count"] == 1
    assert state.convergence["latest_transition"] == "abnormal_finish_failed"


def test_runtime_continues_length_truncated_text_and_merges_answer(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config({"runtime": {"task_mode": "standard", "max_tool_rounds": 1}})
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    client = RecordingClient(
        [
            ChatResponse(
                message={"role": "assistant", "content": "第一部分，"},
                raw={},
                finish_reason="length",
                usage={"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
                http_attempt_count=2,
            ),
            ChatResponse(
                message={"role": "assistant", "content": "第二部分。"},
                raw={},
                finish_reason="stop",
                usage={"prompt_tokens": 13, "completion_tokens": 5, "total_tokens": 18},
                http_attempt_count=1,
            ),
        ]
    )
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )

    assert runtime.run("给出完整回答") == "第一部分，第二部分。"
    state = runtime.sessions.load(runtime.last_session_id).state
    assert state.main_loop_model_request_count == 2
    assert state.final_synthesis_model_request_count == 0
    assert state.model_request_count == 2
    assert state.model_metrics == {
        "http_attempt_count": 3,
        "prompt_tokens": 24,
        "completion_tokens": 8,
        "total_tokens": 32,
    }
    assert state.convergence["length_continuation_count"] == 1
    assert state.convergence["latest_transition"] == "length_continuation"
    assert state.convergence["phase"] == "main_loop"
    assert state.round == 1
    assert state.tool_calls == []


def test_length_continuation_failure_preserves_completed_usage_and_failed_attempts(
    tmp_path: Path,
    make_config,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config({"runtime": {"task_mode": "standard", "max_tool_rounds": 1}})
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    client = RecordingClient(
        [
            ChatResponse(
                message={"role": "assistant", "content": "partial"},
                raw={},
                finish_reason="length",
                usage={"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
                http_attempt_count=2,
            ),
            ChatResponse(
                message={"role": "assistant", "content": " then more"},
                raw={},
                finish_reason="length",
                usage={"prompt_tokens": 13, "completion_tokens": 5, "total_tokens": 18},
                http_attempt_count=1,
            ),
            AttemptError("continuation transport failed", http_attempt_count=3),
        ]
    )
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )

    with pytest.raises(AttemptError, match="continuation transport failed"):
        runtime.run("give a complete answer")

    record = runtime.sessions.load(runtime.last_session_id)
    state = record.state
    assert state.model_metrics == {
        "http_attempt_count": 6,
        "prompt_tokens": 24,
        "completion_tokens": 8,
        "total_tokens": 32,
    }
    assert state.main_loop_model_request_count == 3
    assert state.convergence["length_continuation_count"] == 2
    assert state.convergence["latest_transition"] == "length_continuation"
    assert state.convergence["phase"] == "main_loop"
    assert state.round == 1
    assert state.tool_calls == []
    assert any(item.get("role") == "assistant" and item.get("content") == "partial" for item in record.messages)
    assert any(item.get("role") == "assistant" and item.get("content") == " then more" for item in record.messages)


def test_final_synthesis_length_continuation_keeps_phase_and_tool_turn_metrics(
    tmp_path: Path,
    make_config,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config({"runtime": {"task_mode": "standard", "max_tool_rounds": 1}})
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    client = RecordingClient(
        [
            ChatResponse(
                message=model_tool_call("inspect", "list_dir", {"path": ".", "depth": 1}),
                raw={},
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                http_attempt_count=1,
            ),
            ChatResponse(
                message={"role": "assistant", "content": "综合结果第一部分，"},
                raw={},
                finish_reason="length",
                usage={"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
                http_attempt_count=2,
            ),
            ChatResponse(
                message={"role": "assistant", "content": "第二部分。"},
                raw={},
                finish_reason="stop",
                usage={"prompt_tokens": 21, "completion_tokens": 4, "total_tokens": 25},
                http_attempt_count=1,
            ),
        ]
    )
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )

    assert runtime.run("检查项目并给出综合结果") == "综合结果第一部分，第二部分。"

    state = runtime.sessions.load(runtime.last_session_id).state
    assert state.main_loop_model_request_count == 1
    assert state.context_compaction_model_request_count == 0
    assert state.final_synthesis_model_request_count == 2
    assert state.model_request_count == 3
    assert state.model_metrics == {
        "http_attempt_count": 4,
        "prompt_tokens": 51,
        "completion_tokens": 9,
        "total_tokens": 60,
    }
    assert state.convergence["length_continuation_count"] == 1
    assert state.convergence["latest_transition"] == "length_continuation"
    assert state.convergence["phase"] == "final_synthesis"
    assert state.round == 1
    assert len(state.tool_calls) == 1


def test_final_synthesis_rejects_unusable_finish_reason_and_preserves_resume(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config(
        {
            "runtime": {
                "task_mode": "standard",
                "max_tool_rounds": 1,
                "max_tool_rounds_hard_limit": 3,
            }
        }
    )
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    client = RecordingClient(
        [
            model_tool_call("inspect", "list_dir", {"path": ".", "depth": 1}),
            ChatResponse(
                message={"role": "assistant", "content": "This filtered final must not be accepted."},
                raw={},
                finish_reason="content_filter",
            ),
        ]
    )
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )

    answer = runtime.run("检查项目并总结")

    assert "This filtered final must not be accepted" not in answer
    assert "final synthesis ended with an unusable finish_reason=content_filter" in answer
    assert "agent resume --session" in answer
    state = runtime.sessions.load(runtime.last_session_id).state
    assert state.status == "failed"
    assert state.convergence["latest_transition"] == "final_synthesis_rejected"
    assert state.convergence["phase"] == "final_synthesis"
    assert state.final_synthesis_model_request_count == 1


@pytest.mark.parametrize(
    ("final_message", "expected_protocol"),
    [
        (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "late-tool",
                        "type": "function",
                        "function": {"name": "list_dir", "arguments": '{"path":"."}'},
                    }
                ],
            },
            "structured tool calls",
        ),
        (
            {
                "role": "assistant",
                "content": (
                    '<｜｜DSML｜｜tool_calls>\n<｜｜DSML｜｜invoke name="file_diff">attempted edit</｜｜DSML｜｜invoke>'
                ),
            },
            "DeepSeek tool-call protocol text",
        ),
        (
            {
                "role": "assistant",
                "content": (
                    "先给出结论。\n```text\n<｜｜DSML｜｜tool_calls>\n"
                    '<｜｜DSML｜｜invoke name="file_diff">attempted edit</｜｜DSML｜｜invoke>\n```'
                ),
            },
            "DeepSeek tool-call protocol text",
        ),
    ],
)
def test_final_synthesis_rejects_tool_protocol_without_execution(
    tmp_path: Path,
    make_config,
    final_message: dict,
    expected_protocol: str,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config({"runtime": {"task_mode": "standard", "max_tool_rounds": 1, "max_tool_rounds_hard_limit": 3}})
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    client = RecordingClient(
        [
            model_tool_call("inspect", "list_dir", {"path": ".", "depth": 1}),
            ChatResponse(message=final_message, raw={}, finish_reason="tool_calls"),
        ]
    )
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )

    answer = runtime.run("检查项目并总结")

    assert expected_protocol not in answer.split("任务尚未完成：", maxsplit=1)[0]
    assert "tool-free final synthesis attempted tool use" in answer
    assert "no tool was executed" in answer
    assert "agent resume --session" in answer
    state = runtime.sessions.load(runtime.last_session_id).state
    assert state.status == "failed"
    assert state.convergence["final_synthesis_rejected_protocol"] == expected_protocol
    assert len(state.tool_calls) == 1


def test_soft_target_completion_failure_is_not_reported_as_hard_limit(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config(
        {
            "runtime": {
                "task_mode": "standard",
                "max_tool_rounds": 1,
                "max_tool_rounds_hard_limit": 3,
            }
        }
    )
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    client = RecordingClient(
        [
            model_tool_call("inspect", "list_dir", {"path": ".", "depth": 1}),
            {"role": "assistant", "content": "I will verify next."},
        ]
    )
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )

    answer = runtime.run("检查项目")

    assert "soft tool-turn target was reached" in answer
    assert "hard tool-turn limit" not in answer
    state = runtime.sessions.load(runtime.last_session_id).state
    assert state.status == "failed"
    assert state.error.startswith("soft_target reached:")


def test_non_stream_model_failure_persists_http_attempt_count(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config({"runtime": {"task_mode": "standard", "max_tool_rounds": 1}})
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    client = RecordingClient([AttemptError("all network attempts failed", http_attempt_count=3)])
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )

    with pytest.raises(AttemptError, match="all network attempts failed"):
        runtime.run("inspect the project")

    state = runtime.sessions.load(runtime.last_session_id).state
    assert state.status == "failed"
    assert state.main_loop_model_request_count == 1
    assert state.model_metrics == {"http_attempt_count": 3}
    assert state.round == 1
    assert state.tool_calls == []


def test_stream_interruption_records_attempts_without_advancing_tool_turn(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config({"runtime": {"task_mode": "standard", "max_tool_rounds": 1}})
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    client = RecordingClient([DeepSeekStreamInterrupted("partial output cannot be replayed", http_attempt_count=3)])
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )

    with pytest.raises(RuntimeError, match="Session:"):
        runtime.run("检查项目")

    state = runtime.sessions.load(runtime.last_session_id).state
    assert state.status == "failed"
    assert state.main_loop_model_request_count == 1
    assert state.model_request_count == 1
    assert state.model_metrics == {"http_attempt_count": 3}
    assert state.round == 1
    assert state.tool_calls == []


def test_runtime_recovers_overflow_with_cheap_then_semantic_compaction(
    tmp_path: Path, make_config, monkeypatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config({"runtime": {"task_mode": "standard", "max_tool_rounds": 1}})
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    client = RecordingClient(
        [
            DeepSeekContextOverflow("first overflow", http_attempt_count=1),
            DeepSeekContextOverflow("second overflow", http_attempt_count=1),
            ChatResponse(
                message={"role": "assistant", "content": "两级恢复后完成。"},
                raw={},
                finish_reason="stop",
                usage={"prompt_tokens": 17, "completion_tokens": 4, "total_tokens": 21},
                http_attempt_count=2,
            ),
        ]
    )
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )
    stages: list[str] = []

    def cheap(*args, **_kwargs):
        stages.append("cheap")
        del args[1][0]
        return True

    def semantic(*args, **_kwargs):
        stages.append("semantic")
        del args[1][0]
        return True

    monkeypatch.setattr(runtime, "_overflow_cheap_collapse", cheap)
    monkeypatch.setattr(runtime, "_overflow_semantic_compact", semantic)

    assert runtime.run("处理超长上下文") == "两级恢复后完成。"
    assert stages == ["cheap", "semantic"]
    state = runtime.sessions.load(runtime.last_session_id).state
    assert state.main_loop_model_request_count == 3
    assert state.model_request_count == 3
    assert state.model_metrics == {
        "http_attempt_count": 4,
        "prompt_tokens": 17,
        "completion_tokens": 4,
        "total_tokens": 21,
    }
    assert state.convergence["overflow_recovery_count"] == 2
    assert state.convergence["latest_transition"] == "overflow_semantic_compact"
    assert state.convergence["phase"] == "main_loop"
    assert state.round == 1
    assert state.tool_calls == []


def test_final_synthesis_recovers_typed_overflow_without_advancing_tool_turn(
    tmp_path: Path,
    make_config,
    monkeypatch,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config({"runtime": {"task_mode": "standard", "max_tool_rounds": 1}})
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    client = RecordingClient(
        [
            ChatResponse(
                message=model_tool_call("inspect", "list_dir", {"path": ".", "depth": 1}),
                raw={},
                finish_reason="tool_calls",
                usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                http_attempt_count=1,
            ),
            DeepSeekContextOverflow("final overflow one", http_attempt_count=2),
            DeepSeekContextOverflow("final overflow two", http_attempt_count=1),
            ChatResponse(
                message={"role": "assistant", "content": "两级收缩后完成最终总结。"},
                raw={},
                finish_reason="stop",
                usage={"prompt_tokens": 17, "completion_tokens": 4, "total_tokens": 21},
                http_attempt_count=2,
            ),
        ]
    )
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )
    stages: list[str] = []
    recovery_checkpoints: list[tuple[str, int, int]] = []

    def cheap(*args, **_kwargs):
        stages.append("cheap")
        del args[1][0]
        return True

    def semantic(*args, **_kwargs):
        stages.append("semantic")
        del args[1][0]
        return True

    original_checkpoint = runtime._checkpoint_session

    def checkpoint_spy(state, messages):
        transition = str(state.convergence.get("latest_transition") or "")
        if transition.startswith("overflow_"):
            recovery_checkpoints.append((transition, state.final_synthesis_model_request_count, state.round))
        original_checkpoint(state, messages)

    monkeypatch.setattr(runtime, "_overflow_cheap_collapse", cheap)
    monkeypatch.setattr(runtime, "_overflow_semantic_compact", semantic)
    monkeypatch.setattr(runtime, "_checkpoint_session", checkpoint_spy)

    assert runtime.run("检查项目并给出最终总结") == "两级收缩后完成最终总结。"

    state = runtime.sessions.load(runtime.last_session_id).state
    assert stages == ["cheap", "semantic"]
    assert recovery_checkpoints == [
        ("overflow_cheap_collapse", 1, 1),
        ("overflow_semantic_compact", 2, 1),
    ]
    assert state.main_loop_model_request_count == 1
    assert state.final_synthesis_model_request_count == 3
    assert state.model_request_count == 4
    assert state.model_metrics == {
        "http_attempt_count": 6,
        "prompt_tokens": 27,
        "completion_tokens": 6,
        "total_tokens": 33,
    }
    assert state.convergence["overflow_recovery_count"] == 2
    assert state.convergence["latest_transition"] == "overflow_semantic_compact"
    assert state.convergence["phase"] == "final_synthesis"
    assert state.round == 1
    assert len(state.tool_calls) == 1


def test_incomplete_final_preserves_substantive_synthesis_and_appends_resume(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config({"runtime": {"task_mode": "deep", "max_tool_rounds_hard_limit": 1}})
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    substantive = "已定位 src/runtime.ts 的状态机缺口，并确认 typecheck 尚未执行。"
    client = RecordingClient(
        [
            model_tool_call("list", "list_dir", {"path": ".", "depth": 1}),
            {"role": "assistant", "content": substantive},
        ]
    )
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )

    answer = runtime.run("全面审计项目并完成验证")

    assert substantive in answer
    assert "任务尚未完成" in answer
    assert "agent resume --session" in answer
    assert runtime.sessions.load(runtime.last_session_id).state.status == "failed"


def test_emergency_projection_scales_down_to_the_actual_context_budget(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=RecordingClient([]),
    )
    state = AgentState.create(
        session_id="bounded-emergency",
        project=project,
        user_request="长期目标" * 10_000,
        loaded_memories=[],
        loaded_tools=[],
        git_branch=None,
        context_index_path=str(project.agent_dir / "index.json"),
    )
    state.tool_calls = [
        {
            "round": index + 1,
            "request": {
                "tool": "template",
                "action": "read_file",
                "args": {"path": ("very-long-path/" * 50) + f"file-{index}.txt"},
            },
            "result": {
                "success": True,
                "stdout": f"CRITICAL-{index}=42 " + ("中文证据" * 1_000),
                "stderr": "",
                "data": {},
            },
        }
        for index in range(12)
    ]
    state.execution_context.modified_files = [
        ("very-long-modified-path/" * 30) + f"file-{index}.txt" for index in range(200)
    ]
    messages = [
        {"role": "system", "content": "bounded objective"},
        {"role": "user", "content": "continue"},
        *tool_pair(1, chars=16_000),
        {"role": "system", "content": "old oversized context " + ("历史证据" * 10_000)},
        *tool_pair(2, chars=16_000),
    ]
    context_window = ContextWindowController(
        context_window_tokens=8_192,
        safety_buffer_tokens=1_024,
        keep_recent_rounds=1,
        failure_limit=3,
    )
    for _ in range(3):
        context_window.record_failure()
    model_route = ModelRoute(
        provider="deepseek",
        tier="standard",
        model="deepseek-v4-pro",
        thinking_enabled=True,
        reasoning_effort="high",
        max_tokens=1_024,
        reasons=("test",),
    )
    runtime._prepare_model_request(
        state,
        messages,
        tools=None,
        model_route=model_route,
        context_window=context_window,
        history_compactor=ToolHistoryCompactor(
            aggregate_chars=4_096,
            output_reserve_chars=3_072,
            compacted_result_chars=256,
            keep_recent_results=1,
            failure_limit=3,
        ),
        auto_compaction_enabled=True,
        auto_compaction_max_tokens=512,
        phase="tool_loop",
        checkpoint=False,
    )
    emergency = next(
        str(item.get("content") or "")
        for item in messages
        if str(item.get("content") or "").startswith("[Deep Agent emergency context collapse]")
    )
    budget = context_window.budget(messages, None, max_output_tokens=1_024)
    assert budget.over_limit is False
    assert len(emergency) < 20_000
    assert repair_tool_message_pairs(messages).changed is False


def test_runtime_compacts_last_tool_result_before_final_synthesis(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "large.txt").write_text((("bounded evidence " * 8) + "\n") * 100, encoding="utf-8")
    config = make_config(
        {
            "runtime": {
                "task_mode": "standard",
                "max_tool_rounds": 1,
                "convergence": {
                    "single_tool_result_chars": 8_000,
                    "aggregate_tool_result_chars": 4_096,
                    "output_reserve_chars": 3_072,
                    "compacted_tool_result_chars": 300,
                    "keep_recent_tool_results": 1,
                },
            }
        }
    )
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    client = RecordingClient(
        [
            model_tool_call("read-last", "read_file", {"path": "large.txt", "start_line": 1, "end_line": 100}),
            {"role": "assistant", "content": "已根据末轮有界证据完成总结。"},
        ]
    )
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )

    assert runtime.run("检查 large.txt 并总结") == "已根据末轮有界证据完成总结。"

    synthesis_request = client.requests[-1]
    tool_contents = [str(item.get("content") or "") for item in synthesis_request if item.get("role") == "tool"]
    assert len(tool_contents) == 1
    assert sum(map(len, tool_contents)) <= 1_024
    assert tool_contents[0].startswith("[Deep Agent compacted")
    state = runtime.sessions.load(runtime.last_session_id).state
    assert state.main_loop_model_request_count == 1
    assert state.context_compaction_model_request_count == 0
    assert state.final_synthesis_model_request_count == 1
    assert state.model_request_count == 2


def test_runtime_keeps_multi_tool_results_contiguous_before_recovery_context(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "present.txt").write_text("available evidence\n", encoding="utf-8")
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    memory.sync_project(project)
    memory.add_memory(
        kind="Correction",
        title="File does not exist recovery",
        content="When a file does not exist, keep processing the other requested files before retrying.",
        tags=["correction:file"],
        project_id=project.id,
    )
    client = RecordingClient(
        [
            model_tool_calls(
                ("missing-read", "read_file", {"path": "missing.txt"}),
                ("present-read", "read_file", {"path": "present.txt"}),
            ),
            {"role": "assistant", "content": "已保留失败证据并完成其余读取。"},
        ]
    )
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )

    assert runtime.run("检查两个文件并总结结果") == "已保留失败证据并完成其余读取。"

    second_request = client.requests[1]
    assistant_index = next(index for index, item in enumerate(second_request) if len(item.get("tool_calls") or []) == 2)
    paired = second_request[assistant_index + 1 : assistant_index + 3]
    assert [item.get("role") for item in paired] == ["tool", "tool"]
    assert [item.get("tool_call_id") for item in paired] == ["missing-read", "present-read"]
    recovery_indexes = [
        index
        for index, item in enumerate(second_request)
        if item.get("role") == "system" and "Failure Recovery Memory" in str(item.get("content") or "")
    ]
    assert recovery_indexes
    assert min(recovery_indexes) > assistant_index + 2


@pytest.mark.parametrize(
    ("step_id", "expected_broad_schema"),
    [("scope", True), ("missing-step", False)],
)
def test_runtime_resets_convergence_only_after_successful_real_progress(
    tmp_path: Path,
    make_config,
    step_id: str,
    expected_broad_schema: bool,
) -> None:
    root = tmp_path / f"project-{step_id}"
    root.mkdir()
    for index in range(6):
        (root / f"file-{index}.txt").write_text(f"evidence {index}\n", encoding="utf-8")
    config = make_config(
        {
            "runtime": {
                "task_mode": "deep",
                "max_tool_rounds_hard_limit": 12,
                "convergence": {
                    "max_consecutive_exploration_rounds": 6,
                    "reserved_tool_rounds": 1,
                },
            }
        }
    )
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    responses = [model_tool_call(f"read-{index}", "read_file", {"path": f"file-{index}.txt"}) for index in range(6)]
    responses.append(model_tool_call("step-progress", "agent_update_step", {"step_id": step_id, "status": "completed"}))
    client = RecordingClient(responses)
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )

    with pytest.raises(AssertionError, match="fake response queue exhausted"):
        runtime.run("全面审计整个项目并总结证据")

    eighth_schema = {str((item.get("function") or {}).get("name") or "") for item in (client.schemas[7] or [])}
    assert ("list_dir" in eighth_schema) is expected_broad_schema
    state = runtime.sessions.load(runtime.last_session_id).state
    assert state.tool_calls[-1]["result"]["success"] is expected_broad_schema


def test_runtime_checkpoints_exploration_state_after_observing_tool_round(
    tmp_path: Path,
    make_config,
    monkeypatch,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / "evidence.txt").write_text("bounded evidence\n", encoding="utf-8")
    config = make_config({"runtime": {"task_mode": "deep", "checkpoint_each_tool": False}})
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    client = RecordingClient(
        [
            model_tool_call("read", "read_file", {"path": "evidence.txt"}),
            AttemptError("stop after checkpoint", http_attempt_count=1),
        ]
    )
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )
    observed_checkpoints: list[dict] = []
    original_checkpoint = runtime._checkpoint_session

    def checkpoint_spy(state, messages):
        if state.tool_calls:
            observed_checkpoints.append(json.loads(json.dumps(state.convergence)))
        original_checkpoint(state, messages)

    monkeypatch.setattr(runtime, "_checkpoint_session", checkpoint_spy)

    with pytest.raises(AttemptError, match="stop after checkpoint"):
        runtime.run("全面检查项目并完成根因分析")

    assert observed_checkpoints
    first = observed_checkpoints[0]
    assert first["consecutive_read_only_rounds"] == 1
    assert first["low_yield_rounds"] == 0
    assert first["seen_targets"] == [
        TaskConvergenceController._target_key(
            {"tool": "template", "action": "read_file", "args": {"path": "evidence.txt"}}
        )
    ]


def test_runtime_resets_corrective_budget_after_successful_tool_progress(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config({"runtime": {"task_mode": "deep", "max_tool_rounds_hard_limit": 12}})
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    responses: list[dict] = []
    for step_id in ("scope", "inspect-chunks", "synthesize", "verify"):
        responses.append({"role": "assistant", "content": f"{step_id} 阶段证据已整理。"})
        responses.append(
            model_tool_call(
                f"complete-{step_id}",
                "agent_update_step",
                {"step_id": step_id, "status": "completed"},
            )
        )
        if step_id == "scope":
            responses.append(model_tool_call("inspect-project", "list_dir", {"path": ".", "depth": 1}))
    responses.append({"role": "assistant", "content": "全部步骤已完成，结论已经验证。"})
    client = RecordingClient(responses)
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )

    assert runtime.run("全面审计整个项目并总结证据") == "全部步骤已完成，结论已经验证。"
    assert len(client.requests) == 10
    assert all(step.status == "completed" for step in runtime.sessions.load(runtime.last_session_id).state.plan)


def test_runtime_does_not_run_history_compactor_when_convergence_disabled(
    tmp_path: Path,
    make_config,
    monkeypatch,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config({"runtime": {"convergence": {"enabled": False}}})
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    client = RecordingClient([{"role": "assistant", "content": "无需压缩即可完成。"}])
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
    )

    def forbidden_compaction(_self, _messages):
        raise AssertionError("history compactor must be bypassed when convergence is disabled")

    monkeypatch.setattr(ToolHistoryCompactor, "compact", forbidden_compaction)

    assert runtime.run("回答当前问题") == "无需压缩即可完成。"


def test_runtime_converges_after_read_only_rounds_and_compacts_history(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    for index in range(6):
        (root / f"file-{index}.txt").write_text((f"evidence {index} " * 500) + "\n", encoding="utf-8")
    config = make_config(
        {
            "runtime": {
                "task_mode": "deep",
                "max_tool_rounds_hard_limit": 10,
                "convergence": {
                    "max_consecutive_exploration_rounds": 6,
                    "reserved_tool_rounds": 2,
                    "single_tool_result_chars": 2_500,
                    "aggregate_tool_result_chars": 6_000,
                    "output_reserve_chars": 1_000,
                    "compacted_tool_result_chars": 500,
                    "keep_recent_tool_results": 1,
                },
            }
        }
    )
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    responses = [model_tool_call(f"read-{index}", "read_file", {"path": f"file-{index}.txt"}) for index in range(6)]
    responses.extend(
        [
            model_tool_call("blocked-scan", "list_dir", {"path": ".", "depth": 1}),
            model_tool_calls(
                ("complete-scope", "agent_update_step", {"step_id": "scope", "status": "completed"}),
                (
                    "complete-inspect",
                    "agent_update_step",
                    {"step_id": "inspect-chunks", "status": "completed"},
                ),
                (
                    "complete-synthesize",
                    "agent_update_step",
                    {"step_id": "synthesize", "status": "completed"},
                ),
                ("complete-verify", "agent_update_step", {"step_id": "verify", "status": "completed"}),
            ),
            {"role": "assistant", "content": "基于六个文件的证据完成综合与验证。"},
        ]
    )
    client = RecordingClient(responses)
    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=ToolManager(config, project, memory, yolo=True),
        client=client,
        context_builder=ContextBuilder(config),
    )

    answer = runtime.run("全面审计整个项目并总结证据")

    assert answer == "基于六个文件的证据完成综合与验证。"
    seventh_request = client.requests[6]
    assert any("Exploration budget checkpoint" in str(item.get("content")) for item in seventh_request)
    schema_names = {str((item.get("function") or {}).get("name") or "") for item in (client.schemas[6] or [])}
    assert "list_dir" not in schema_names
    assert "search_code" not in schema_names
    assert "read_file" in schema_names
    assert any(
        str(item.get("content") or "").startswith("[Deep Agent compacted")
        for item in seventh_request
        if item.get("role") == "tool"
    )
    state = runtime.sessions.load(runtime.last_session_id).state
    assert len(state.tool_calls) == 11
    assert state.tool_calls[6]["result"]["success"] is False
    assert state.tool_calls[6]["result"]["data"]["runtime_denied"] is True
    assert all(item["result"]["success"] is True for item in state.tool_calls[7:])
    assert len(str((state.tool_calls[0]["result"] or {}).get("stdout") or "")) > 500
    assert all(step.status == "completed" for step in state.plan)
