from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import convergence
from agent.context_window import (
    ContextWindowController,
    PairRepairResult,
    RequestTokenBudget,
    estimate_request_tokens,
    repair_tool_message_pairs,
)
from agent.convergence import TaskConvergenceController, ToolHistoryCompactor
from agent.convergence_guards import is_bounded_validation_command
from agent.history_compaction import (
    ToolHistoryCompactor as SplitToolHistoryCompactor,
    ToolHistoryResult as SplitToolHistoryResult,
)


def test_convergence_facade_keeps_compatibility_exports_and_small_owners() -> None:
    expected = {
        "ConvergenceAction",
        "ContextWindowController",
        "PairRepairResult",
        "RequestTokenBudget",
        "TaskConvergenceController",
        "ToolHistoryCompactor",
        "ToolHistoryResult",
        "estimate_request_tokens",
        "repair_tool_message_pairs",
    }

    assert expected <= set(convergence.__all__)
    assert convergence.ContextWindowController is ContextWindowController
    assert convergence.PairRepairResult is PairRepairResult
    assert convergence.RequestTokenBudget is RequestTokenBudget
    assert convergence.ToolHistoryCompactor is SplitToolHistoryCompactor
    assert convergence.ToolHistoryResult is SplitToolHistoryResult
    assert convergence.estimate_request_tokens is estimate_request_tokens
    assert convergence.repair_tool_message_pairs is repair_tool_message_pairs
    assert TaskConvergenceController.__module__ == "agent.convergence"
    assert SplitToolHistoryCompactor.__module__ == "agent.history_compaction"
    assert Path(convergence.__file__).stat().st_size <= 25_000
    assert Path(__file__).parents[1].joinpath("agent/history_compaction.py").stat().st_size <= 20_000


@pytest.mark.parametrize(
    ("command", "allowed"),
    [
        ("pytest -q tests/test_runtime_convergence.py", True),
        ("ruff format --check agent tests", True),
        ("ruff check --fix agent", False),
        ("pytest -q; sed -n 1,20p secret.txt", False),
    ],
)
def test_split_validation_guard_matches_controller_contract(command: str, allowed: bool) -> None:
    assert is_bounded_validation_command(command) is allowed
    assert TaskConvergenceController._is_bounded_validation_command(command) is allowed
    assert TaskConvergenceController.is_exploration_bypass("shell_run", {"command": command}) is not allowed


def test_split_implementation_guard_preserves_consumption_semantics() -> None:
    state = SimpleNamespace(
        turn=1,
        convergence={},
        plan=[{"id": "implement", "step_type": "implement", "status": "in_progress"}],
        tool_calls=[
            {
                "request": {
                    "tool": "template",
                    "action": "read_file",
                    "args": {"path": "agent/example.py"},
                },
                "result": {"success": True},
            }
        ],
    )
    controller = TaskConvergenceController(
        mode="deep",
        max_rounds=8,
        exploration_round_limit=2,
        reserved_rounds=2,
        implementation_read_limit=1,
    )

    assert (
        controller.implementation_read_denial(
            state,
            "read_file",
            {"path": "agent/example.py", "start_line": 1, "end_line": 200},
        )
        == ""
    )
    assert controller.implementation_reads_used == 1
    assert "exhausted" in controller.implementation_read_denial(
        state,
        "read_file",
        {"path": "agent/example.py", "start_line": 1, "end_line": 2},
    )


def test_split_history_evidence_codec_round_trips_through_compactor_proxy() -> None:
    compactor = ToolHistoryCompactor(
        aggregate_chars=4_096,
        output_reserve_chars=1_024,
        compacted_result_chars=512,
        keep_recent_results=1,
        failure_limit=2,
    )
    original = '{"success":true,"payload":"bounded"}'

    evidence = compactor._evidence(original, name="read_file", args={"path": "agent/example.py"})
    encoded = compactor._metadata(evidence)
    restored = compactor._parse_evidence(encoded)

    assert restored == evidence
    assert restored is not None
    assert restored.tool == "read_file"
    assert restored.target == {"path": "agent/example.py"}
    assert restored.original_chars == len(original)
