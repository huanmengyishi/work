from __future__ import annotations

import hashlib
import json
import re
import shlex
from dataclasses import dataclass
from typing import Any, Iterable

from .state import AgentState


_BROAD_EXPLORATION_FUNCTIONS = frozenset({"list_dir", "find_files", "search_code"})
_TARGETED_EXPLORATION_FUNCTIONS = frozenset({"read_file", "tool_result_read"})
_READ_ONLY_CAPABILITIES = frozenset(
    {
        "template.list_dir",
        "template.find_files",
        "template.search_code",
        "template.read_file",
        "tool_result.read",
        "project.read_context",
        "git.status",
        "git.diff",
        "git.log",
        "document.parse",
    }
)
_MUTATION_RESULT_FUNCTIONS = frozenset({"file_apply", "file_undo"})
_VERIFICATION_RESULT_FUNCTIONS = frozenset({"document_parse", "run_tests", "lsp_diagnostics", "git_diff_staged"})
_SAFE_ARGUMENT_KEYS = (
    "path",
    "start_line",
    "end_line",
    "query",
    "glob",
    "pattern",
    "framework",
    "depth",
    "request_id",
    "offset",
    "max_chars",
)
_COMPACTED_MARKER = "[Deep Agent compacted tool result]"
_METADATA_MARKER = "[Deep Agent compacted metadata]"
_MIN_EMERGENCY_RESULT_CHARS = 128
_MAX_IMPLEMENTATION_READ_LINES = 200
_MAX_VALIDATION_ATTACHMENT_READ_CHARS = 12_000
_MAX_PERSISTED_SEEN_TARGETS = 128
_MAX_TARGET_KEY_CHARS = 512
_VALIDATION_ATTACHMENT_CAPABILITIES = frozenset(
    {"template.run_tests", "lsp.diagnostics", "document.parse", "template.git_diff_staged"}
)


@dataclass(frozen=True)
class _ToolEvidence:
    tool: str
    success: bool | None
    target: dict[str, Any]
    original_chars: int
    sha256: str


@dataclass(frozen=True)
class ToolHistoryResult:
    messages: list[dict[str, Any]]
    original_chars: int
    final_chars: int
    compacted_count: int = 0
    failure_count: int = 0
    circuit_open: bool = False
    error: str = ""

    @property
    def changed(self) -> bool:
        return self.compacted_count > 0


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


class ToolHistoryCompactor:
    """Bound the aggregate model-visible tool history without breaking pairs.

    AgentState keeps bounded ToolResult previews and metadata; an oversized
    complete body exists only in its Session-private attachment. This class
    replaces older ``role=tool`` message bodies before the next model request;
    assistant tool calls and their tool_call_id values remain untouched.
    """

    def __init__(
        self,
        *,
        aggregate_chars: int,
        output_reserve_chars: int,
        compacted_result_chars: int,
        keep_recent_results: int,
        failure_limit: int,
    ) -> None:
        self.aggregate_chars = max(4_096, int(aggregate_chars))
        self.output_reserve_chars = max(0, min(int(output_reserve_chars), self.aggregate_chars - 1_024))
        self.compacted_result_chars = max(256, min(int(compacted_result_chars), 8_000))
        self.keep_recent_results = max(1, min(int(keep_recent_results), 100))
        self.failure_limit = max(1, min(int(failure_limit), 20))
        self.failure_count = 0
        self.circuit_open = False

    @property
    def target_chars(self) -> int:
        return max(1_024, self.aggregate_chars - self.output_reserve_chars)

    def compact(self, messages: list[dict[str, Any]]) -> ToolHistoryResult:
        original_chars = self._tool_chars(messages)
        if original_chars <= self.target_chars:
            return ToolHistoryResult(
                messages=messages,
                original_chars=original_chars,
                final_chars=original_chars,
                failure_count=self.failure_count,
                circuit_open=self.circuit_open,
            )
        if self.circuit_open:
            compacted, count = self._emergency_hard_limit(messages)
            return ToolHistoryResult(
                messages=compacted,
                original_chars=original_chars,
                final_chars=self._tool_chars(compacted),
                compacted_count=count,
                failure_count=self.failure_count,
                circuit_open=True,
                error="compaction circuit is open; deterministic hard-limit fallback applied",
            )
        try:
            compacted, count = self._compact_once(messages)
        except Exception as exc:  # defensive: compaction must never abort the Agent loop
            self.failure_count += 1
            self.circuit_open = self.failure_count >= self.failure_limit
            compacted, count = self._emergency_hard_limit(messages)
            return ToolHistoryResult(
                messages=compacted,
                original_chars=original_chars,
                final_chars=self._tool_chars(compacted),
                compacted_count=count,
                failure_count=self.failure_count,
                circuit_open=self.circuit_open,
                error=f"{type(exc).__name__}: {exc}"[:500],
            )
        self.failure_count = 0
        return ToolHistoryResult(
            messages=compacted,
            original_chars=original_chars,
            final_chars=self._tool_chars(compacted),
            compacted_count=count,
        )

    def _emergency_hard_limit(self, messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        result, removed = self._collapse_excess_tool_rounds(messages)
        indexes = [index for index, item in enumerate(result) if item.get("role") == "tool"]
        if not indexes:
            return result, removed
        base, remainder = divmod(self.target_chars, len(indexes))
        call_details = self._tool_call_details(result)
        changed = removed
        for position, index in enumerate(indexes):
            original = str(result[index].get("content") or "")
            call_id = str(result[index].get("tool_call_id") or "")
            name, args = call_details.get(call_id, ("unknown", {}))
            evidence = self._fallback_evidence(original, name=name, args=args)
            replacement = self._emergency_summary(
                original,
                evidence=evidence,
                limit=base + int(position < remainder),
            )
            if replacement != original:
                updated = dict(result[index])
                updated["content"] = replacement
                result[index] = updated
                changed += 1
        if self._tool_chars(result) > self.target_chars:
            raise RuntimeError("deterministic tool-history fallback exceeded its hard limit")
        return result, changed

    def _fallback_evidence(self, content: str, *, name: str, args: dict[str, Any]) -> _ToolEvidence:
        try:
            return self._evidence(content, name=name, args=args)
        except Exception:
            return _ToolEvidence(
                tool=name,
                success=None,
                target={},
                original_chars=len(content),
                sha256=hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest(),
            )

    def _compact_once(self, messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        messages, removed = self._collapse_excess_tool_rounds(messages)
        tool_indexes = [index for index, item in enumerate(messages) if item.get("role") == "tool"]
        if not tool_indexes:
            return messages, removed
        call_details = self._tool_call_details(messages)
        recent = self._recent_round_indexes(messages, tool_indexes, self.keep_recent_results)
        essential: set[int] = {tool_indexes[-1]}
        latest_mutation: int | None = None
        latest_verification: int | None = None
        for index in tool_indexes:
            call_id = str(messages[index].get("tool_call_id") or "")
            name, _args = call_details.get(call_id, ("unknown", {}))
            if name in _MUTATION_RESULT_FUNCTIONS:
                latest_mutation = index
            if name in _VERIFICATION_RESULT_FUNCTIONS:
                latest_verification = index
        essential.update(item for item in (latest_mutation, latest_verification) if item is not None)

        result = list(messages)
        total = self._tool_chars(result)
        compacted_indexes: set[int] = set()

        # Stage 1: micro-compact the oldest and largest raw results to bounded
        # head/tail previews. Recent API rounds and the latest mutation/check
        # remain preferred, but that preference is not allowed to violate the
        # aggregate hard limit in the later stages.
        oldest = [index for index in tool_indexes if index not in recent and index not in essential]
        preferred = [index for index in tool_indexes if index in recent and index not in essential]
        protected = [index for index in tool_indexes if index in essential]
        for group in (oldest, preferred, protected):
            group.sort(key=lambda index: (-len(str(result[index].get("content") or "")), index))
        preview_limit = self.compacted_result_chars
        for group in (oldest, preferred, protected):
            for index in group:
                if total <= self.target_chars:
                    break
                original = str(result[index].get("content") or "")
                if self._is_compacted(original) or len(original) <= preview_limit:
                    continue
                call_id = str(result[index].get("tool_call_id") or "")
                name, args = call_details.get(call_id, ("unknown", {}))
                evidence = self._evidence(original, name=name, args=args)
                replacement = self._preview(original, evidence=evidence, limit=preview_limit)
                if len(replacement) >= len(original):
                    continue
                updated = dict(result[index])
                updated["content"] = replacement
                result[index] = updated
                total -= len(original) - len(replacement)
                compacted_indexes.add(index)

            # Stage 2: automatic metadata compaction. Fully reduce the older
            # priority group before touching a more recent or essential API
            # round, while retaining stable evidence from the original body.
            if total > self.target_chars:
                for index in group:
                    if total <= self.target_chars:
                        break
                    original = str(result[index].get("content") or "")
                    call_id = str(result[index].get("tool_call_id") or "")
                    name, args = call_details.get(call_id, ("unknown", {}))
                    evidence = self._evidence(original, name=name, args=args)
                    replacement = self._metadata(evidence)
                    if len(replacement) >= len(original):
                        continue
                    updated = dict(result[index])
                    updated["content"] = replacement
                    result[index] = updated
                    total -= len(original) - len(replacement)
                    compacted_indexes.add(index)

        # Stage 3: a deterministic emergency squeeze guarantees the advertised
        # hard limit even when protected/recent results alone exceed it. Tool
        # messages and tool_call_id pairings are never removed.
        if total > self.target_chars:
            minimum, remaining = divmod(self.target_chars, len(tool_indexes))
            priority = protected + preferred + oldest
            allocations = {index: minimum for index in tool_indexes}
            for index in priority[:remaining]:
                allocations[index] += 1
            for index in tool_indexes:
                original = str(result[index].get("content") or "")
                keep = allocations[index]
                call_id = str(result[index].get("tool_call_id") or "")
                name, args = call_details.get(call_id, ("unknown", {}))
                evidence = self._evidence(original, name=name, args=args)
                replacement = self._emergency_summary(original, evidence=evidence, limit=keep)
                updated = dict(result[index])
                updated["content"] = replacement
                result[index] = updated
                if replacement != original:
                    compacted_indexes.add(index)
        if self._tool_chars(result) > self.target_chars:
            raise RuntimeError("tool history compaction failed to satisfy the aggregate hard limit")
        return result, removed + len(compacted_indexes)

    def _collapse_excess_tool_rounds(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int]:
        result = list(repair_tool_message_pairs(messages).messages)
        removed_results = 0
        removed_rounds = 0
        removed_evidence: list[dict[str, Any]] = []
        insertion_index: int | None = None
        while sum(item.get("role") == "tool" for item in result) * _MIN_EMERGENCY_RESULT_CHARS > self.target_chars:
            tool_rounds = [
                index for index, item in enumerate(result) if item.get("role") == "assistant" and item.get("tool_calls")
            ]
            if len(tool_rounds) <= 1:
                break
            assistant_index = tool_rounds[0]
            end = assistant_index + 1
            while end < len(result) and result[end].get("role") == "tool":
                end += 1
            call_details = self._tool_call_details(result[assistant_index:end])
            for tool_message in result[assistant_index + 1 : end]:
                content = str(tool_message.get("content") or "")
                call_id = str(tool_message.get("tool_call_id") or "")
                name, args = call_details.get(call_id, ("unknown", {}))
                evidence = self._fallback_evidence(content, name=name, args=args)
                target_text = json.dumps(
                    evidence.target,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if len(target_text) > 256:
                    target_text = json.dumps(
                        {
                            "sha256": hashlib.sha256(target_text.encode("utf-8", errors="replace")).hexdigest(),
                            "truncated": True,
                        },
                        separators=(",", ":"),
                    )
                excerpt = content if len(content) <= 320 else content[:150] + "...[omitted]..." + content[-150:]
                removed_evidence.append(
                    {
                        "tool": evidence.tool,
                        "success": evidence.success,
                        "target": target_text,
                        "original_chars": evidence.original_chars,
                        "sha256": evidence.sha256,
                        "excerpt": excerpt,
                    }
                )
            removed_results += sum(item.get("role") == "tool" for item in result[assistant_index:end])
            removed_rounds += 1
            insertion_index = assistant_index if insertion_index is None else min(insertion_index, assistant_index)
            del result[assistant_index:end]
        if removed_rounds and insertion_index is not None:
            result.insert(
                insertion_index,
                {
                    "role": "system",
                    "content": (
                        "[Deep Agent collapsed oldest complete tool rounds] "
                        f"rounds={removed_rounds} results={removed_results}. The model-visible projection removed "
                        "whole call/result groups because a minimum structured evidence record per call exceeded "
                        "the hard aggregate budget; no partial protocol pair was retained. Bounded removed evidence: "
                        + json.dumps(
                            {
                                "entries": removed_evidence[-16:],
                                "omitted_entries": max(0, len(removed_evidence) - 16),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    ),
                },
            )
        return result, removed_results

    def _preview(self, content: str, *, evidence: _ToolEvidence, limit: int) -> str:
        prefix = _COMPACTED_MARKER + "\n" + self._evidence_json(evidence) + "\npreview:\n"
        if len(prefix) >= limit:
            return self._emergency_summary(content, evidence=evidence, limit=limit)
        available = max(0, limit - len(prefix) - len("\n...[middle omitted]...\n"))
        head = available // 2
        tail = available - head
        if len(content) <= available:
            body = content
        else:
            body = content[:head] + "\n...[middle omitted]...\n" + (content[-tail:] if tail else "")
        return (prefix + body)[:limit]

    def _metadata(self, evidence: _ToolEvidence) -> str:
        return _METADATA_MARKER + "\n" + self._evidence_json(evidence)

    def _emergency_summary(self, content: str, *, evidence: _ToolEvidence, limit: int) -> str:
        if limit <= 0:
            return ""
        if len(content) <= limit:
            return content
        compact_target = json.dumps(evidence.target, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        candidates = [
            self._metadata(evidence),
            json.dumps(
                {
                    "s": evidence.success,
                    "n": evidence.original_chars,
                    "h": evidence.sha256,
                    "t": compact_target,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            json.dumps(
                {"s": evidence.success, "n": evidence.original_chars, "h": evidence.sha256},
                separators=(",", ":"),
            ),
            json.dumps(
                {
                    "compacted": True,
                    "success": evidence.success,
                    "original_chars": evidence.original_chars,
                    "sha256": evidence.sha256,
                },
                separators=(",", ":"),
            ),
            json.dumps(
                {
                    "compacted": True,
                    "success": evidence.success,
                    "original_chars": evidence.original_chars,
                },
                separators=(",", ":"),
            ),
            json.dumps({"success": evidence.success}, separators=(",", ":")),
            "{}",
            "0",
        ]
        return next((item for item in candidates if len(item) <= limit), "")

    @staticmethod
    def _evidence_json(evidence: _ToolEvidence) -> str:
        return json.dumps(
            {
                "tool": evidence.tool,
                "success": evidence.success,
                "target": evidence.target,
                "original_chars": evidence.original_chars,
                "sha256": evidence.sha256,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _evidence(self, content: str, *, name: str, args: dict[str, Any]) -> _ToolEvidence:
        parsed = self._parse_evidence(content)
        if parsed is not None:
            return parsed
        try:
            payload = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            payload = None
        success_value = payload.get("success", payload.get("s")) if isinstance(payload, dict) else None
        success = success_value if isinstance(success_value, bool) else None
        original_chars = len(content)
        sha256 = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
        target = self._safe_target(args)
        if isinstance(payload, dict):
            raw_target = payload.get("target", payload.get("t"))
            if isinstance(raw_target, str):
                try:
                    raw_target = json.loads(raw_target)
                except json.JSONDecodeError:
                    raw_target = None
            if isinstance(raw_target, dict):
                target = raw_target
            try:
                recorded_chars = int(payload.get("original_chars", payload.get("n")) or 0)
            except (TypeError, ValueError):
                recorded_chars = 0
            recorded_hash = str(payload.get("sha256", payload.get("h")) or "")
            if recorded_chars >= len(content):
                original_chars = recorded_chars
            if re.fullmatch(r"[0-9a-f]{64}", recorded_hash):
                sha256 = recorded_hash
        return _ToolEvidence(
            tool=str(payload.get("tool") or name) if isinstance(payload, dict) else name,
            success=success,
            target=target,
            original_chars=original_chars,
            sha256=sha256,
        )

    @staticmethod
    def _parse_evidence(content: str) -> _ToolEvidence | None:
        if not ToolHistoryCompactor._is_compacted(content):
            return None
        lines = content.splitlines()
        if len(lines) >= 2:
            try:
                payload = json.loads(lines[1])
            except (TypeError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, dict):
                success = payload.get("success")
                try:
                    original_chars = max(0, int(payload.get("original_chars") or 0))
                except (TypeError, ValueError):
                    original_chars = len(content)
                recorded_hash = str(payload.get("sha256") or "")
                if not re.fullmatch(r"[0-9a-f]{64}", recorded_hash):
                    recorded_hash = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
                return _ToolEvidence(
                    tool=str(payload.get("tool") or "unknown"),
                    success=success if isinstance(success, bool) else None,
                    target=payload.get("target") if isinstance(payload.get("target"), dict) else {},
                    original_chars=original_chars,
                    sha256=recorded_hash,
                )

        # Compatibility with previews produced by early v0.11.0 candidates.
        values: dict[str, str] = {}
        legacy_lines = list(lines[1:])
        if lines and lines[0].startswith(_METADATA_MARKER):
            legacy_lines.insert(0, lines[0][len(_METADATA_MARKER) :].strip())
        for line in legacy_lines:
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
        if not values:
            return None
        try:
            target = json.loads(values.get("target") or "{}")
        except json.JSONDecodeError:
            target = {}
        success_value = values.get("success", "").casefold()
        success = True if success_value == "true" else False if success_value == "false" else None
        raw_chars = values.get("original_chars") or values.get("chars") or "0"
        try:
            original_chars = max(0, int(raw_chars))
        except ValueError:
            original_chars = 0
        return _ToolEvidence(
            tool=values.get("tool") or "unknown",
            success=success,
            target=target if isinstance(target, dict) else {},
            original_chars=original_chars,
            sha256=values.get("sha256") or "",
        )

    @staticmethod
    def _safe_target(args: dict[str, Any]) -> dict[str, Any]:
        return {key: args[key] for key in _SAFE_ARGUMENT_KEYS if key in args}

    @staticmethod
    def _is_compacted(content: str) -> bool:
        return content.startswith(_COMPACTED_MARKER) or content.startswith(_METADATA_MARKER)

    @staticmethod
    def _recent_round_indexes(
        messages: list[dict[str, Any]], tool_indexes: list[int], keep_recent_rounds: int
    ) -> set[int]:
        call_round: dict[str, int] = {}
        api_round = 0
        for item in messages:
            if item.get("role") != "assistant" or not item.get("tool_calls"):
                continue
            api_round += 1
            for call in item.get("tool_calls") or []:
                if isinstance(call, dict):
                    call_round[str(call.get("id") or "")] = api_round
        round_for_index: dict[int, tuple[str, int]] = {}
        for index in tool_indexes:
            call_id = str(messages[index].get("tool_call_id") or "")
            round_for_index[index] = ("round", call_round[call_id]) if call_id in call_round else ("orphan", index)
        ordered_rounds: list[tuple[str, int]] = []
        for index in tool_indexes:
            value = round_for_index[index]
            if value not in ordered_rounds:
                ordered_rounds.append(value)
        recent_rounds = set(ordered_rounds[-keep_recent_rounds:])
        return {index for index, value in round_for_index.items() if value in recent_rounds}

    @staticmethod
    def _tool_chars(messages: Iterable[dict[str, Any]]) -> int:
        return sum(len(str(item.get("content") or "")) for item in messages if item.get("role") == "tool")

    @staticmethod
    def _tool_call_details(messages: Iterable[dict[str, Any]]) -> dict[str, tuple[str, dict[str, Any]]]:
        details: dict[str, tuple[str, dict[str, Any]]] = {}
        for item in messages:
            if item.get("role") != "assistant":
                continue
            for call in item.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                function = call.get("function") if isinstance(call.get("function"), dict) else {}
                raw_args = function.get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except (TypeError, json.JSONDecodeError):
                    args = {}
                details[str(call.get("id") or "")] = (
                    str(function.get("name") or "unknown"),
                    args if isinstance(args, dict) else {},
                )
        return details


@dataclass(frozen=True)
class ConvergenceAction:
    messages: tuple[str, ...] = ()
    excluded_functions: frozenset[str] = frozenset()
    reason: str = ""
    block_exploration_bypass: bool = False
    guard_implementation_read: bool = False
    guard_validation_attachment_read: bool = False
    force_plan_transition: bool = False


class TaskConvergenceController:
    """Detect low-yield exploration and preserve implementation/verify rounds."""

    def __init__(
        self,
        *,
        mode: str,
        max_rounds: int,
        exploration_round_limit: int,
        reserved_rounds: int,
        implementation_read_limit: int = 2,
        validation_attachment_read_limit: int = 2,
    ) -> None:
        self.enabled = mode in {"large", "deep"}
        self.max_rounds = max(1, int(max_rounds))
        self.exploration_round_limit = max(2, min(int(exploration_round_limit), self.max_rounds))
        self.reserved_rounds = max(1, min(int(reserved_rounds), max(1, self.max_rounds - 1)))
        self.implementation_read_limit = max(0, min(int(implementation_read_limit), 4))
        self.implementation_reads_used = 0
        self.validation_attachment_read_limit = max(0, min(int(validation_attachment_read_limit), 4))
        self.validation_attachment_reads_used = 0
        self.consecutive_read_only_rounds = 0
        self.low_yield_rounds = 0
        self.seen_targets: set[str] = set()
        self.last_plan_fingerprint: tuple[tuple[str, str], ...] = ()
        self.nudge_count = 0
        self._nudge_sent_for_stall = False
        self._hard_notice_sent = False
        self._implementation_notice_sent = False
        self._last_implementation_notice_remaining: int | None = None
        self._last_validation_attachment_notice_remaining: int | None = None
        self._bound_state: AgentState | None = None
        self._seen_target_order: list[str] = []

    def bind(self, state: AgentState) -> None:
        self._bound_state = state
        self.last_plan_fingerprint = self._plan_fingerprint(state)
        metadata = getattr(state, "convergence", {})
        current_turn = self._bounded_turn(state)
        if isinstance(metadata, dict):
            used = metadata.get("implementation_reads_used", 0)
            if isinstance(used, int) and not isinstance(used, bool):
                self.implementation_reads_used = max(0, min(used, 4))
            validation_reads = metadata.get("validation_attachment_reads_used", 0)
            if isinstance(validation_reads, int) and not isinstance(validation_reads, bool):
                self.validation_attachment_reads_used = max(0, min(validation_reads, 4))
            consecutive = metadata.get("consecutive_read_only_rounds", 0)
            if isinstance(consecutive, int) and not isinstance(consecutive, bool):
                self.consecutive_read_only_rounds = max(
                    0,
                    min(consecutive, self.exploration_round_limit + 2),
                )
            low_yield = metadata.get("low_yield_rounds", 0)
            if isinstance(low_yield, int) and not isinstance(low_yield, bool):
                self.low_yield_rounds = max(0, min(low_yield, 5))
            raw_targets = metadata.get("seen_targets", [])
            if isinstance(raw_targets, list):
                targets = [
                    item
                    for item in raw_targets[-_MAX_PERSISTED_SEEN_TARGETS:]
                    if isinstance(item, str) and 0 < len(item) <= _MAX_TARGET_KEY_CHARS
                ]
                self._seen_target_order = list(dict.fromkeys(targets))
                self.seen_targets = set(self._seen_target_order)
            nudge_count = metadata.get("nudge_count", 0)
            same_turn = metadata.get("notice_turn") == current_turn
            if same_turn:
                if isinstance(nudge_count, int) and not isinstance(nudge_count, bool):
                    self.nudge_count = max(0, min(nudge_count, 2))
                self._nudge_sent_for_stall = metadata.get("nudge_sent_for_stall") is True
                self._hard_notice_sent = metadata.get("hard_notice_sent") is True
        self._sync()

    def _sync(self) -> None:
        if self._bound_state is None:
            return
        metadata = getattr(self._bound_state, "convergence", None)
        if not isinstance(metadata, dict):
            metadata = {}
            setattr(self._bound_state, "convergence", metadata)
        metadata.update(
            {
                "implementation_reads_used": self.implementation_reads_used,
                "validation_attachment_reads_used": self.validation_attachment_reads_used,
                "consecutive_read_only_rounds": self.consecutive_read_only_rounds,
                "low_yield_rounds": self.low_yield_rounds,
                "seen_targets": list(self._seen_target_order[-_MAX_PERSISTED_SEEN_TARGETS:]),
                "nudge_count": self.nudge_count,
                "nudge_sent_for_stall": self._nudge_sent_for_stall,
                "hard_notice_sent": self._hard_notice_sent,
                "notice_turn": self._bounded_turn(self._bound_state),
            }
        )

    @staticmethod
    def _bounded_turn(state: Any) -> int:
        value = getattr(state, "turn", 0)
        return max(0, min(value, 1_000_000)) if isinstance(value, int) and not isinstance(value, bool) else 0

    def before_round(self, round_number: int, state: Any | None = None) -> ConvergenceAction:
        if not self.enabled:
            return ConvergenceAction()
        reserve_due = round_number > self.max_rounds - self.reserved_rounds
        stalled = self.consecutive_read_only_rounds >= self.exploration_round_limit or self.low_yield_rounds >= 3
        forced = self.consecutive_read_only_rounds >= self.exploration_round_limit + 2 or self.low_yield_rounds >= 5
        hard = reserve_due or forced
        force_plan_transition = hard and (self._exploration_step_active(state) or self._plan_requires_transition(state))
        implementation_read_open = hard and self._implementation_step_active(state) and self._read_allowance_remaining()
        validation_attachment_read_open = (
            hard
            and self._implementation_or_verification_step_active(state)
            and self._validation_attachment_allowance_remaining()
            and self._has_validation_attachment(state)
        )
        if not stalled and not reserve_due:
            return ConvergenceAction()

        notices: list[str] = []
        implementation_remaining = max(0, self.implementation_read_limit - self.implementation_reads_used)
        remaining = self.max_rounds - round_number + 1
        if not self._nudge_sent_for_stall and self.nudge_count < 2:
            notices.append(
                "Exploration budget checkpoint: the task has spent "
                f"{self.consecutive_read_only_rounds} consecutive rounds on read-only or non-progress inspection "
                "without advancing "
                "the Task Graph. Stop broad scanning, update the current plan step, and synthesize the evidence "
                f"already collected. Preserve the remaining {remaining} tool rounds for a concrete change when "
                "justified, static checks, verification, and the final answer. Read only an exact missing target "
                "that is necessary for implementation; do not use shell or Python to bypass this checkpoint."
            )
            self._nudge_sent_for_stall = True
            self.nudge_count += 1
        if hard and not self._hard_notice_sent:
            notice = (
                "The exploration window is now closed because its continuous-read threshold or reserved-round "
                "boundary was reached. Use the existing evidence, plan updates, managed edits, diagnostics/tests, "
                "or return a substantive evidence-based answer. Shell or Python file-reading commands are also "
                "blocked in this phase; they must not be used to bypass the managed exploration tools."
            )
            if implementation_read_open:
                notice += (
                    " Because the implement step is active, read_file remains available only for an exact path "
                    f"already read successfully, with explicit start/end lines covering at most "
                    f"{_MAX_IMPLEMENTATION_READ_LINES} lines. "
                    f"At most {self.implementation_read_limit - self.implementation_reads_used} such evidence "
                    "read(s) remain; broad or new-target inspection is still closed."
                )
            elif force_plan_transition:
                notice += (
                    " The current scope/inspection step must transition now. In this response, use "
                    "agent_update_step to complete the current exploration step and start the next ready step; "
                    "do not spend another round on status, tests, diagnostics, or file inspection."
                )
            if self._conditional_mutation_step_active(state):
                notice += (
                    " This is a conditional-mutation plan. If the evidence already collected does not prove a real "
                    "issue that justifies a change, do not invent one: call agent_update_step with step_id "
                    "`implement` and status `skipped`, then start `verify` and report the exact validation outcome."
                )
            notices.append(notice)
            self._hard_notice_sent = True
            if implementation_read_open:
                self._implementation_notice_sent = True
                self._last_implementation_notice_remaining = implementation_remaining
        elif implementation_read_open and (
            not self._implementation_notice_sent
            or self._last_implementation_notice_remaining != implementation_remaining
        ):
            notices.append(
                "The implement step is now active inside the closed exploration window. read_file is available "
                "only as a bounded implementation-evidence exception: use an exact path that was read "
                "successfully before the window closed, provide explicit start_line/end_line values covering at "
                f"most {_MAX_IMPLEMENTATION_READ_LINES} lines. Bounded implementation evidence allowance: "
                f"{self.implementation_read_limit - self.implementation_reads_used} read(s) remaining. "
                "New targets, broad reads, shell/Python file exploration, and verify-phase reads remain closed."
            )
            self._implementation_notice_sent = True
            self._last_implementation_notice_remaining = implementation_remaining
        elif (
            hard
            and self._implementation_step_active(state)
            and self._last_implementation_notice_remaining != implementation_remaining
        ):
            notices.append(
                f"Bounded implementation evidence allowance: {implementation_remaining} read(s) remaining. "
                + (
                    "read_file is now closed; proceed with the managed edit, verification, or final answer."
                    if implementation_remaining == 0
                    else "Only an exact previously-read path and a range of at most 200 lines is allowed."
                )
            )
            self._last_implementation_notice_remaining = implementation_remaining

        validation_attachment_remaining = max(
            0,
            self.validation_attachment_read_limit - self.validation_attachment_reads_used,
        )
        if (
            validation_attachment_read_open
            and self._last_validation_attachment_notice_remaining != validation_attachment_remaining
        ):
            notices.append(
                "A bounded validation attachment is available in the closed exploration window. tool_result_read "
                "may read only an attachment produced by run_tests, diagnostics, document verification, or staged-"
                f"diff verification in this Session; each chunk is limited to {_MAX_VALIDATION_ATTACHMENT_READ_CHARS} "
                f"characters and {validation_attachment_remaining} "
                "read(s) remain. It cannot read ordinary exploration attachments."
            )
            self._last_validation_attachment_notice_remaining = validation_attachment_remaining

        excluded = set(_BROAD_EXPLORATION_FUNCTIONS)
        reason = "read-only exploration stalled"
        if hard:
            excluded.update(_TARGETED_EXPLORATION_FUNCTIONS)
            if implementation_read_open:
                excluded.discard("read_file")
            if validation_attachment_read_open:
                excluded.discard("tool_result_read")
            reason = (
                "reserved implementation and verification window"
                if reserve_due
                else "continuous exploration threshold reached"
            )
        action = ConvergenceAction(
            tuple(notices),
            frozenset(excluded),
            reason,
            block_exploration_bypass=hard,
            guard_implementation_read=implementation_read_open,
            guard_validation_attachment_read=validation_attachment_read_open,
            force_plan_transition=force_plan_transition,
        )
        self._sync()
        return action

    def implementation_read_denial(
        self,
        state: Any,
        function_name: str,
        arguments: str | dict[str, Any] | None,
    ) -> str:
        """Consume one narrowly scoped implementation evidence read or explain denial."""

        if function_name != "read_file":
            return "only read_file can use the bounded implementation evidence exception"
        if not self._implementation_step_active(state):
            return "the implement step is not in progress"
        if not self._read_allowance_remaining():
            return "the bounded implementation evidence read allowance is exhausted"
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (TypeError, json.JSONDecodeError):
            return "arguments are not valid JSON"
        if not isinstance(args, dict):
            return "arguments must be an object"
        path = self._normalized_path(args.get("path"))
        start_line = args.get("start_line")
        end_line = args.get("end_line")
        if not path:
            return "an exact path is required"
        if (
            isinstance(start_line, bool)
            or isinstance(end_line, bool)
            or not isinstance(start_line, int)
            or not isinstance(end_line, int)
            or start_line < 1
            or end_line < start_line
        ):
            return "explicit positive start_line/end_line values are required"
        if end_line - start_line + 1 > _MAX_IMPLEMENTATION_READ_LINES:
            return f"the requested implementation evidence range exceeds {_MAX_IMPLEMENTATION_READ_LINES} lines"
        if not self._path_was_read_successfully(state, path):
            return "the path was not read successfully before the exploration window closed"
        self.implementation_reads_used += 1
        self._implementation_notice_sent = False
        if self._bound_state is None and hasattr(state, "convergence"):
            self._bound_state = state
        self._sync()
        return ""

    def _read_allowance_remaining(self) -> bool:
        return self.implementation_reads_used < self.implementation_read_limit

    def validation_attachment_read_denial(
        self,
        state: Any,
        function_name: str,
        arguments: str | dict[str, Any] | None,
    ) -> str:
        """Consume one bounded read of a validation-produced private attachment."""

        if function_name != "tool_result_read":
            return "only tool_result_read can use the bounded validation attachment exception"
        if not self._implementation_or_verification_step_active(state):
            return "the implement or verify step is not in progress"
        if not self._validation_attachment_allowance_remaining():
            return "the bounded validation attachment read allowance is exhausted"
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (TypeError, json.JSONDecodeError):
            return "arguments are not valid JSON"
        if not isinstance(args, dict):
            return "arguments must be an object"
        request_id = str(args.get("request_id") or "").strip()
        offset = args.get("offset", 0)
        max_chars = args.get("max_chars", _MAX_VALIDATION_ATTACHMENT_READ_CHARS)
        if not request_id:
            return "a validation attachment request_id is required"
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            return "offset must be a non-negative integer"
        if (
            isinstance(max_chars, bool)
            or not isinstance(max_chars, int)
            or max_chars < 1
            or max_chars > _MAX_VALIDATION_ATTACHMENT_READ_CHARS
        ):
            return f"max_chars must be between 1 and {_MAX_VALIDATION_ATTACHMENT_READ_CHARS}"
        if not self._is_validation_attachment(state, request_id):
            return "the request_id is not an attachment produced by a bounded validation tool in this Session"
        self.validation_attachment_reads_used += 1
        if self._bound_state is None and hasattr(state, "convergence"):
            self._bound_state = state
        self._sync()
        return ""

    def _validation_attachment_allowance_remaining(self) -> bool:
        return self.validation_attachment_reads_used < self.validation_attachment_read_limit

    @classmethod
    def _has_validation_attachment(cls, state: Any | None) -> bool:
        return any(cls._validation_attachment_id(item) for item in getattr(state, "tool_calls", ()) or ())

    @classmethod
    def _is_validation_attachment(cls, state: Any, request_id: str) -> bool:
        return any(cls._validation_attachment_id(item) == request_id for item in getattr(state, "tool_calls", ()) or ())

    @classmethod
    def _validation_attachment_id(cls, item: Any) -> str:
        if not isinstance(item, dict):
            return ""
        request = item.get("request") if isinstance(item.get("request"), dict) else {}
        result = item.get("result") if isinstance(item.get("result"), dict) else {}
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        attachment = data.get("attachment") if isinstance(data.get("attachment"), dict) else {}
        capability = f"{request.get('tool', '')}.{request.get('action', '')}"
        if capability not in _VALIDATION_ATTACHMENT_CAPABILITIES:
            if capability != "shell.run" or cls.is_exploration_bypass("shell_run", request.get("args")):
                return ""
        request_id = str(request.get("request_id") or "").strip()
        attachment_id = str(attachment.get("request_id") or "").strip()
        return request_id if request_id and request_id == attachment_id else ""

    @staticmethod
    def _implementation_step_active(state: Any | None) -> bool:
        if state is None:
            return False
        for step in getattr(state, "plan", ()) or ():
            step_id = step.get("id") if isinstance(step, dict) else getattr(step, "id", "")
            status = step.get("status") if isinstance(step, dict) else getattr(step, "status", "")
            if str(step_id) == "implement" and str(status) == "in_progress":
                return True
        return False

    @staticmethod
    def _implementation_or_verification_step_active(state: Any | None) -> bool:
        if state is None:
            return False
        for step in getattr(state, "plan", ()) or ():
            step_id = step.get("id") if isinstance(step, dict) else getattr(step, "id", "")
            status = step.get("status") if isinstance(step, dict) else getattr(step, "status", "")
            if str(step_id) in {"implement", "verify"} and str(status) == "in_progress":
                return True
        return False

    @classmethod
    def _conditional_mutation_step_active(cls, state: Any | None) -> bool:
        if state is None or not cls._implementation_step_active(state):
            return False
        route = getattr(state, "task_route", {})
        reasons = route.get("reasons") if isinstance(route, dict) else None
        return isinstance(reasons, list) and "conditional-mutation" in reasons

    @staticmethod
    def _exploration_step_active(state: Any | None) -> bool:
        if state is None:
            return False
        for step in getattr(state, "plan", ()) or ():
            step_id = step.get("id") if isinstance(step, dict) else getattr(step, "id", "")
            status = step.get("status") if isinstance(step, dict) else getattr(step, "status", "")
            if str(step_id) in {"scope", "inspect-chunks"} and str(status) == "in_progress":
                return True
        return False

    @staticmethod
    def _plan_requires_transition(state: Any | None) -> bool:
        if state is None:
            return False
        steps = list(getattr(state, "plan", ()) or ())
        if not steps:
            return False
        statuses = [
            str(step.get("status") if isinstance(step, dict) else getattr(step, "status", "")) for step in steps
        ]
        return "in_progress" not in statuses and any(status == "pending" for status in statuses)

    @staticmethod
    def _normalized_path(value: Any) -> str:
        return str(value or "").strip().replace("\\", "/").rstrip("/")

    @classmethod
    def _path_was_read_successfully(cls, state: Any, path: str) -> bool:
        for item in getattr(state, "tool_calls", ()) or ():
            if not isinstance(item, dict):
                continue
            request = item.get("request") if isinstance(item.get("request"), dict) else {}
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            args = request.get("args") if isinstance(request.get("args"), dict) else {}
            if (
                str(request.get("tool") or "") == "template"
                and str(request.get("action") or "") == "read_file"
                and bool(result.get("success"))
                and cls._normalized_path(args.get("path")) == path
            ):
                return True
        return False

    def observe_round(
        self,
        state: AgentState,
        requests: list[dict[str, Any]],
        results: list[dict[str, Any]] | None = None,
    ) -> bool:
        plan_fingerprint = self._plan_fingerprint(state)
        plan_progressed = plan_fingerprint != self.last_plan_fingerprint
        self.last_plan_fingerprint = plan_fingerprint
        capabilities = [f"{item.get('tool', '')}.{item.get('action', '')}" for item in requests]
        read_only = bool(capabilities) and all(item in _READ_ONLY_CAPABILITIES for item in capabilities)
        targets = [self._target_key(item) for item in requests if self._target_key(item)]
        repeated_targets = sum(target in self.seen_targets for target in targets)
        new_targets = len(targets) - repeated_targets
        for target in targets:
            if target in self.seen_targets:
                continue
            self.seen_targets.add(target)
            self._seen_target_order.append(target)
        while len(self._seen_target_order) > _MAX_PERSISTED_SEEN_TARGETS:
            removed = self._seen_target_order.pop(0)
            self.seen_targets.discard(removed)

        result_items = results or []
        productive_capabilities = {
            "file.apply",
            "file.undo",
            "document.render_docx",
            "template.run_tests",
            "lsp.diagnostics",
        }
        productive = any(
            capability in productive_capabilities
            and index < len(result_items)
            and bool(result_items[index].get("success"))
            for index, capability in enumerate(capabilities)
        )
        progressed = plan_progressed or productive
        if not self.enabled:
            self._sync()
            return progressed
        if progressed:
            self.consecutive_read_only_rounds = 0
            self.low_yield_rounds = 0
            self._nudge_sent_for_stall = False
            self._hard_notice_sent = False
            self._sync()
            return True
        if requests:
            self.consecutive_read_only_rounds = min(
                self.exploration_round_limit + 2,
                self.consecutive_read_only_rounds + 1,
            )
        if read_only and targets and repeated_targets >= max(1, new_targets):
            self.low_yield_rounds = min(5, self.low_yield_rounds + 1)
        else:
            self.low_yield_rounds = max(0, self.low_yield_rounds - 1)
        self._sync()
        return False

    @staticmethod
    def filter_schemas(schemas: list[dict[str, Any]], excluded_functions: frozenset[str]) -> list[dict[str, Any]]:
        if not excluded_functions:
            return schemas
        return [
            item for item in schemas if str((item.get("function") or {}).get("name") or "") not in excluded_functions
        ]

    @staticmethod
    def _plan_fingerprint(state: AgentState) -> tuple[tuple[str, str], ...]:
        return tuple((step.id, step.status) for step in state.plan)

    @staticmethod
    def _target_key(request: dict[str, Any]) -> str:
        args = request.get("args") if isinstance(request.get("args"), dict) else {}
        capability = f"{request.get('tool', '')}.{request.get('action', '')}"
        target: dict[str, Any] = {}
        for key in _SAFE_ARGUMENT_KEYS:
            if key not in args:
                continue
            value = args[key]
            target[key] = value.strip().casefold() if isinstance(value, str) else value
        if not target:
            return ""
        key = capability + ":" + json.dumps(target, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if len(key) <= _MAX_TARGET_KEY_CHARS:
            return key
        return capability + ":sha256:" + hashlib.sha256(key.encode("utf-8", errors="replace")).hexdigest()

    @staticmethod
    def is_exploration_bypass(function_name: str, arguments: str | dict[str, Any] | None) -> bool:
        if function_name == "shell_run":
            try:
                parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            except (TypeError, json.JSONDecodeError):
                return True
            command = str((parsed or {}).get("command") or "") if isinstance(parsed, dict) else ""
            return not TaskConvergenceController._is_bounded_validation_command(command)
        if function_name == "python_run":
            try:
                parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            except (TypeError, json.JSONDecodeError):
                return True
            return True
        return False

    @staticmethod
    def _is_bounded_validation_command(command: str) -> bool:
        value = str(command or "").strip()
        if not value or any(marker in value for marker in ("\n", ";", "|", "&", "<", ">", "$", "`")):
            return False
        try:
            args = shlex.split(value, posix=True)
        except ValueError:
            return False
        if not args or len(args) > 32:
            return False
        program = args[0].replace("\\", "/").rsplit("/", 1)[-1].casefold()
        rest = [item.casefold() for item in args[1:]]
        mutation_flags = {
            "--fix",
            "--write",
            "--apply",
            "--update",
            "--bless",
            "--accept",
            "--install-types",
        }
        if any(
            item in mutation_flags
            or any(item.startswith(flag + "=") for flag in mutation_flags)
            or (item.startswith("-w") and not item.startswith("--"))
            for item in rest
        ):
            return False
        package_scripts = {"test", "typecheck", "check", "lint"}
        if program in {"npm", "pnpm", "yarn", "bun"}:
            if rest and rest[0] == "test":
                return True
            return len(rest) >= 2 and rest[0] == "run" and rest[1] in package_scripts
        if program in {"pytest", "py.test"}:
            return True
        if re.fullmatch(r"python(?:3(?:\.\d+)?)?", program):
            return len(rest) >= 2 and rest[:2] == ["-m", "pytest"]
        if program == "ruff":
            return bool(rest) and (rest[0] == "check" or rest[:2] == ["format", "--check"])
        if program in {"pyright", "mypy"}:
            return True
        if program == "tsc":
            return "--noemit" in rest
        if program == "npx":
            return len(rest) >= 2 and rest[0] == "tsc" and "--noemit" in rest[1:]
        if program == "cargo":
            return bool(rest) and rest[0] in {"test", "check", "clippy"}
        if program == "go":
            return bool(rest) and rest[0] == "test"
        if program in {"mvn", "mvnw", "gradle", "gradlew"}:
            return any(item in {"test", "check", "verify"} for item in rest)
        return program == "git" and rest == ["diff", "--check"]


__all__ = [
    "ConvergenceAction",
    "ContextWindowController",
    "PairRepairResult",
    "RequestTokenBudget",
    "TaskConvergenceController",
    "ToolHistoryCompactor",
    "ToolHistoryResult",
    "estimate_request_tokens",
    "repair_tool_message_pairs",
]
