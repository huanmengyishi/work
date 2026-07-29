from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .state import AgentState


@dataclass(frozen=True)
class PairRepairResult:
    messages: list[dict[str, Any]]
    repaired_count: int = 0

    @property
    def changed(self) -> bool:
        return self.repaired_count > 0


@dataclass(frozen=True)
class RequestTokenBudget:
    estimated_tokens: int
    trigger_tokens: int
    input_limit_tokens: int
    output_reserve_tokens: int
    safety_buffer_tokens: int

    @property
    def over_trigger(self) -> bool:
        return self.estimated_tokens > self.trigger_tokens

    @property
    def over_limit(self) -> bool:
        return self.estimated_tokens > self.input_limit_tokens


class ContextWindowController:
    """Prepare protocol-valid requests and reserve room for model output."""

    def __init__(
        self,
        *,
        context_window_tokens: int,
        safety_buffer_tokens: int,
        keep_recent_rounds: int,
        failure_limit: int,
    ) -> None:
        self.context_window_tokens = max(8_192, int(context_window_tokens))
        self.safety_buffer_tokens = max(
            1_024,
            min(int(safety_buffer_tokens), self.context_window_tokens // 2),
        )
        self.keep_recent_rounds = max(1, min(int(keep_recent_rounds), 100))
        self.failure_limit = max(1, min(int(failure_limit), 20))
        self.failure_count = 0
        self.circuit_open = False
        self._bound_state: AgentState | None = None

    def bind(self, state: AgentState) -> None:
        """Restore the semantic-compaction breaker from durable Session state."""

        self._bound_state = state
        metadata = state.convergence if isinstance(state.convergence, dict) else {}
        raw_count = metadata.get("context_compaction_failure_count", 0)
        failure_count = (
            max(0, min(raw_count, self.failure_limit))
            if isinstance(raw_count, int) and not isinstance(raw_count, bool)
            else 0
        )
        raw_open = metadata.get("context_compaction_circuit_open", False)
        if raw_open is True:
            failure_count = self.failure_limit
        for counter in (
            "overflow_recovery_count",
            "length_continuation_count",
            "context_compaction_count",
        ):
            if counter not in metadata:
                continue
            raw_value = metadata[counter]
            metadata[counter] = (
                max(0, min(raw_value, 10_000)) if isinstance(raw_value, int) and not isinstance(raw_value, bool) else 0
            )
        if "latest_transition" in metadata:
            metadata["latest_transition"] = (
                str(metadata["latest_transition"])[:64] if isinstance(metadata["latest_transition"], str) else ""
            )
        if "phase" in metadata:
            metadata["phase"] = str(metadata["phase"])[:32] if isinstance(metadata["phase"], str) else ""
        self.failure_count = failure_count
        self.circuit_open = failure_count >= self.failure_limit
        self._sync()

    def _sync(self) -> None:
        if self._bound_state is None:
            return
        metadata = self._bound_state.convergence
        if not isinstance(metadata, dict):
            metadata = {}
            self._bound_state.convergence = metadata
        metadata["context_compaction_failure_count"] = self.failure_count
        metadata["context_compaction_circuit_open"] = self.circuit_open

    def budget(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        max_output_tokens: int,
    ) -> RequestTokenBudget:
        output_reserve = self.effective_output_tokens(max_output_tokens)
        input_limit = max(1, self.context_window_tokens - output_reserve - self.safety_buffer_tokens)
        trigger = max(1, int(input_limit * 0.9))
        return RequestTokenBudget(
            estimated_tokens=estimate_request_tokens(messages, tools),
            trigger_tokens=trigger,
            input_limit_tokens=input_limit,
            output_reserve_tokens=output_reserve,
            safety_buffer_tokens=self.safety_buffer_tokens,
        )

    def effective_output_tokens(self, requested_tokens: int) -> int:
        capacity = max(1, self.context_window_tokens - self.safety_buffer_tokens - 1_024)
        return max(1, min(int(requested_tokens), 20_000, capacity))

    def record_success(self) -> None:
        self.failure_count = 0
        self.circuit_open = False
        self._sync()

    def record_failure(self) -> None:
        self.failure_count = min(self.failure_limit, self.failure_count + 1)
        self.circuit_open = self.failure_count >= self.failure_limit
        self._sync()

    def compact_old_reasoning(self, messages: list[dict[str, Any]]) -> int:
        assistant_indexes = [index for index, item in enumerate(messages) if item.get("role") == "assistant"]
        protected = set(assistant_indexes[-self.keep_recent_rounds :])
        changed = 0
        for index in assistant_indexes:
            if index in protected or not messages[index].get("reasoning_content"):
                continue
            updated = dict(messages[index])
            updated.pop("reasoning_content", None)
            messages[index] = updated
            changed += 1
        return changed

    def compaction_span(self, messages: list[dict[str, Any]]) -> tuple[int, int] | None:
        assistant_indexes = [index for index, item in enumerate(messages) if item.get("role") == "assistant"]
        # Every assistant response is one API round, whether it called tools or
        # returned text only.  Compaction may summarize only rounds older than
        # the protected tail; otherwise exactly N tool rounds could collapse to
        # one merely because a text-only assistant response followed them.
        if len(assistant_indexes) <= self.keep_recent_rounds:
            return None
        end = assistant_indexes[-self.keep_recent_rounds]
        user_indexes = [index for index, item in enumerate(messages[:end]) if item.get("role") == "user"]
        start = user_indexes[-1] + 1 if user_indexes else 0
        if start >= end:
            return None
        return start, end


def estimate_request_tokens(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> int:
    serialized = json.dumps(
        {"messages": messages, "tools": tools or []},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    ascii_chars = sum(ord(char) < 128 for char in serialized)
    non_ascii_chars = len(serialized) - ascii_chars
    return max(1, (ascii_chars + 3) // 4 + non_ascii_chars + len(messages) * 4 + 32)


def repair_tool_message_pairs(messages: list[dict[str, Any]]) -> PairRepairResult:
    """Repair only the model-visible projection, preserving complete API rounds."""

    repaired: list[dict[str, Any]] = []
    repair_count = 0
    index = 0
    while index < len(messages):
        item = messages[index]
        if item.get("role") == "tool":
            repair_count += 1
            index += 1
            continue
        calls = item.get("tool_calls") if item.get("role") == "assistant" else None
        if not calls:
            repaired.append(item)
            index += 1
            continue

        normalized_calls: list[dict[str, Any]] = []
        call_ids: list[str] = []
        seen_call_ids: set[str] = set()
        for call_index, call in enumerate(calls):
            if not isinstance(call, dict):
                repair_count += 1
                continue
            call_id = str(call.get("id") or f"deep-agent-call-{index}-{call_index}")
            if call_id in seen_call_ids:
                repair_count += 1
                continue
            seen_call_ids.add(call_id)
            normalized = dict(call)
            if normalized.get("id") != call_id:
                normalized["id"] = call_id
                repair_count += 1
            normalized_calls.append(normalized)
            call_ids.append(call_id)
        assistant = dict(item)
        assistant["tool_calls"] = normalized_calls
        repaired.append(assistant)

        segment_end = index + 1
        while segment_end < len(messages) and messages[segment_end].get("role") not in {"assistant", "user"}:
            segment_end += 1
        tool_results: dict[str, dict[str, Any]] = {}
        trailing: list[dict[str, Any]] = []
        for candidate in messages[index + 1 : segment_end]:
            if candidate.get("role") != "tool":
                trailing.append(candidate)
                continue
            call_id = str(candidate.get("tool_call_id") or "")
            if call_id not in seen_call_ids or call_id in tool_results:
                repair_count += 1
                continue
            tool_results[call_id] = candidate
        for call_id in call_ids:
            result = tool_results.get(call_id)
            if result is None:
                result = {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(
                        {
                            "success": False,
                            "stdout": "",
                            "stderr": "tool result was unavailable after interruption; do not assume it succeeded",
                            "data": {"synthetic_repair": True},
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
                repair_count += 1
            repaired.append(result)
        repaired.extend(trailing)
        index = segment_end
    if not repair_count and repaired != messages:
        repair_count = 1
    return PairRepairResult(repaired, repair_count)


__all__ = [
    "ContextWindowController",
    "PairRepairResult",
    "RequestTokenBudget",
    "estimate_request_tokens",
    "repair_tool_message_pairs",
]
