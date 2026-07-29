from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

from .context_window import repair_tool_message_pairs
from .history_evidence import (
    ToolEvidence as _ToolEvidence,
    emergency_summary,
    evidence_json,
    extract_evidence,
    is_compacted,
    metadata,
    parse_evidence,
    preview,
    safe_target,
)


_MUTATION_RESULT_FUNCTIONS = frozenset({"file_apply", "file_undo"})
_VERIFICATION_RESULT_FUNCTIONS = frozenset({"document_parse", "run_tests", "lsp_diagnostics", "git_diff_staged"})
_MIN_EMERGENCY_RESULT_CHARS = 128


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
        return preview(content, evidence=evidence, limit=limit)

    @staticmethod
    def _metadata(evidence: _ToolEvidence) -> str:
        return metadata(evidence)

    @staticmethod
    def _emergency_summary(content: str, *, evidence: _ToolEvidence, limit: int) -> str:
        return emergency_summary(content, evidence=evidence, limit=limit)

    @staticmethod
    def _evidence_json(evidence: _ToolEvidence) -> str:
        return evidence_json(evidence)

    def _evidence(self, content: str, *, name: str, args: dict[str, Any]) -> _ToolEvidence:
        return extract_evidence(content, name=name, args=args)

    @staticmethod
    def _parse_evidence(content: str) -> _ToolEvidence | None:
        return parse_evidence(content)

    @staticmethod
    def _safe_target(args: dict[str, Any]) -> dict[str, Any]:
        return safe_target(args)

    @staticmethod
    def _is_compacted(content: str) -> bool:
        return is_compacted(content)

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


__all__ = ["ToolHistoryCompactor", "ToolHistoryResult"]
