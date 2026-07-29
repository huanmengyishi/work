from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .context_window import repair_tool_message_pairs


_MARKER_PREFIX = "[Deep Agent snipped complete API rounds]"
_ASCII_TERM = re.compile(r"[a-z0-9][a-z0-9_./:-]{1,63}")
_CJK_RUN = re.compile(r"[\u3400-\u9fff]+")
_STOP_TERMS = frozenset(
    {
        "agent",
        "all",
        "and",
        "current",
        "ensure",
        "for",
        "from",
        "keep",
        "latest",
        "only",
        "round",
        "task",
        "that",
        "the",
        "this",
        "with",
        "任务",
        "当前",
        "所有",
        "进行",
        "需要",
        "这个",
    }
)
_DEFAULT_MUTATION_TOOLS = frozenset(
    {
        "file_apply",
        "file_undo",
        "render_docx",
        "write_file",
    }
)
_DEFAULT_VERIFICATION_TOOLS = frozenset(
    {
        "document_parse",
        "git_diff_staged",
        "lsp_diagnostics",
        "run_tests",
    }
)


@dataclass(frozen=True)
class HistorySnipResult:
    """A deterministic, model-free projection of complete API rounds."""

    messages: list[dict[str, Any]]
    total_rounds: int
    kept_rounds: int
    removed_rounds: int
    original_chars: int = 0
    final_chars: int = 0
    removed_messages: int = 0
    repaired_pairs: int = 0
    coalesced_markers: int = 0
    skip_reason: str = ""
    kept_reasons: tuple[tuple[int, tuple[str, ...]], ...] = ()

    @property
    def changed(self) -> bool:
        return self.removed_rounds > 0 or self.repaired_pairs > 0 or self.coalesced_markers > 0


@dataclass(frozen=True)
class _ApiRound:
    number: int
    start: int
    end: int
    tool_names: tuple[str, ...]


class HistorySnipper:
    """Remove expendable complete assistant/tool rounds without an LLM call.

    An API round is one assistant response and all immediately paired tool
    results. System and user messages are never classified as rounds. Before
    selection, the model-visible projection is repaired deterministically so
    the result can never retain only one side of a tool call/result pair.
    """

    def __init__(
        self,
        *,
        keep_recent_rounds: int,
        min_history_chars: int = 0,
        min_complete_rounds: int = 0,
        marker_chars: int = 768,
        relevance_scan_chars: int = 12_000,
        max_objective_chars: int = 4_000,
        max_relevance_terms: int = 64,
        min_relevance_matches: int = 1,
        mutation_tools: Iterable[str] | None = None,
        verification_tools: Iterable[str] | None = None,
    ) -> None:
        self.keep_recent_rounds = self._bounded_int(keep_recent_rounds, minimum=0, maximum=100)
        self.min_history_chars = self._bounded_int(
            min_history_chars,
            minimum=0,
            maximum=100_000_000,
        )
        self.min_complete_rounds = self._bounded_int(
            min_complete_rounds,
            minimum=0,
            maximum=10_000,
        )
        self.marker_chars = self._bounded_int(marker_chars, minimum=128, maximum=4_096)
        self.relevance_scan_chars = self._bounded_int(
            relevance_scan_chars,
            minimum=256,
            maximum=100_000,
        )
        self.max_objective_chars = self._bounded_int(
            max_objective_chars,
            minimum=128,
            maximum=32_000,
        )
        self.max_relevance_terms = self._bounded_int(
            max_relevance_terms,
            minimum=1,
            maximum=256,
        )
        self.min_relevance_matches = self._bounded_int(
            min_relevance_matches,
            minimum=1,
            maximum=16,
        )
        self.mutation_tools = self._normalize_tool_set(mutation_tools, _DEFAULT_MUTATION_TOOLS)
        self.verification_tools = self._normalize_tool_set(
            verification_tools,
            _DEFAULT_VERIFICATION_TOOLS,
        )

    def snip(
        self,
        messages: list[dict[str, Any]],
        *,
        objective: str = "",
        safety_goals: Sequence[str] = (),
    ) -> HistorySnipResult:
        """Return a protocol-safe projection while leaving ``messages`` untouched."""

        repaired = repair_tool_message_pairs(messages)
        projection = repaired.messages
        rounds = self._complete_rounds(projection)
        original_chars = self._history_chars(projection)
        if not rounds:
            return HistorySnipResult(
                messages=projection,
                total_rounds=0,
                kept_rounds=0,
                removed_rounds=0,
                original_chars=original_chars,
                final_chars=original_chars,
                repaired_pairs=repaired.repaired_count,
                skip_reason="no_complete_rounds",
            )
        if len(rounds) < self.min_complete_rounds:
            return HistorySnipResult(
                messages=projection,
                total_rounds=len(rounds),
                kept_rounds=len(rounds),
                removed_rounds=0,
                original_chars=original_chars,
                final_chars=original_chars,
                repaired_pairs=repaired.repaired_count,
                skip_reason="below_complete_round_threshold",
            )
        if original_chars < self.min_history_chars:
            return HistorySnipResult(
                messages=projection,
                total_rounds=len(rounds),
                kept_rounds=len(rounds),
                removed_rounds=0,
                original_chars=original_chars,
                final_chars=original_chars,
                repaired_pairs=repaired.repaired_count,
                skip_reason="below_character_threshold",
            )

        protected: dict[int, set[str]] = {}

        def protect(round_number: int, reason: str) -> None:
            protected.setdefault(round_number, set()).add(reason)

        if self.keep_recent_rounds:
            for api_round in rounds[-self.keep_recent_rounds :]:
                protect(api_round.number, "recent")

        latest_mutation = self._latest_tool_round(rounds, self.mutation_tools)
        latest_verification = self._latest_tool_round(rounds, self.verification_tools)
        if latest_mutation is not None:
            protect(latest_mutation, "latest_mutation")
        if latest_verification is not None:
            protect(latest_verification, "latest_verification")

        objective_terms = self._relevance_terms(str(objective or "")[: self.max_objective_chars])
        bounded_safety = " ".join(str(item or "")[: self.max_objective_chars] for item in list(safety_goals)[:16])[
            : self.max_objective_chars
        ]
        safety_terms = self._relevance_terms(bounded_safety)
        for api_round in rounds:
            scan_text = self._round_scan_text(projection, api_round)
            if self._is_relevant(scan_text, objective_terms):
                protect(api_round.number, "objective")
            if self._is_relevant(scan_text, safety_terms):
                protect(api_round.number, "safety")

        removed = [api_round for api_round in rounds if api_round.number not in protected]
        if not removed:
            return HistorySnipResult(
                messages=projection,
                total_rounds=len(rounds),
                kept_rounds=len(rounds),
                removed_rounds=0,
                original_chars=original_chars,
                final_chars=original_chars,
                repaired_pairs=repaired.repaired_count,
                skip_reason="all_rounds_protected",
                kept_reasons=self._kept_reasons(protected),
            )

        remove_indexes = {index for api_round in removed for index in range(api_round.start, api_round.end)}
        marker_indexes = {index for index, message in enumerate(projection) if self._is_marker(message)}
        insertion_index = min(remove_indexes | marker_indexes)
        removed_messages = [projection[index] for index in sorted(remove_indexes)]
        prior_markers = [projection[index] for index in sorted(marker_indexes)]
        marker = {
            "role": "system",
            "content": self._marker_content(
                removed,
                removed_messages,
                prior_markers=prior_markers,
            ),
        }

        result: list[dict[str, Any]] = []
        for index, message in enumerate(projection):
            if index == insertion_index:
                result.append(marker)
            if index in remove_indexes or index in marker_indexes:
                continue
            result.append(message)

        final_repair = repair_tool_message_pairs(result)
        return HistorySnipResult(
            messages=final_repair.messages,
            total_rounds=len(rounds),
            kept_rounds=len(rounds) - len(removed),
            removed_rounds=len(removed),
            original_chars=original_chars,
            final_chars=self._history_chars(final_repair.messages),
            removed_messages=len(removed_messages),
            repaired_pairs=repaired.repaired_count + final_repair.repaired_count,
            coalesced_markers=len(marker_indexes),
            kept_reasons=self._kept_reasons(protected),
        )

    @staticmethod
    def _bounded_int(value: Any, *, minimum: int, maximum: int) -> int:
        if isinstance(value, bool):
            parsed = minimum
        else:
            try:
                parsed = int(value)
            except (TypeError, ValueError, OverflowError):
                parsed = minimum
        return max(minimum, min(parsed, maximum))

    @classmethod
    def _normalize_tool_set(
        cls,
        configured: Iterable[str] | None,
        default: frozenset[str],
    ) -> frozenset[str]:
        values = default if configured is None else configured
        return frozenset(normalized for value in values if (normalized := cls._canonical_tool_name(value)))

    @classmethod
    def _complete_rounds(cls, messages: list[dict[str, Any]]) -> list[_ApiRound]:
        rounds: list[_ApiRound] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            if message.get("role") != "assistant":
                index += 1
                continue
            calls = message.get("tool_calls")
            names: list[str] = []
            result_count = 0
            if isinstance(calls, list):
                result_count = len(calls)
                for call in calls:
                    function = call.get("function") if isinstance(call, dict) else None
                    name = function.get("name") if isinstance(function, dict) else ""
                    normalized = cls._canonical_tool_name(name)
                    if normalized:
                        names.append(normalized)
            end = index + 1 + result_count
            if end > len(messages) or any(
                messages[result_index].get("role") != "tool" for result_index in range(index + 1, end)
            ):
                # The repair layer should make this unreachable. Keeping the
                # tail outside selection is safer than partially deleting it.
                index += 1
                continue
            rounds.append(
                _ApiRound(
                    number=len(rounds),
                    start=index,
                    end=end,
                    tool_names=tuple(names),
                )
            )
            index = end
        return rounds

    @classmethod
    def _history_chars(cls, messages: list[dict[str, Any]]) -> int:
        """Measure visible scalar characters without serializing a second history copy."""

        total = 0
        stack: list[Any] = list(messages)
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                stack.extend(value.keys())
                stack.extend(value.values())
            elif isinstance(value, (list, tuple)):
                stack.extend(value)
            elif value is not None:
                total += len(str(value))
        return total

    @classmethod
    def _canonical_tool_name(cls, value: Any) -> str:
        normalized = str(value or "").strip().casefold().replace("-", "_")
        return normalized.rsplit(".", 1)[-1]

    @staticmethod
    def _latest_tool_round(rounds: list[_ApiRound], tools: frozenset[str]) -> int | None:
        return next(
            (api_round.number for api_round in reversed(rounds) if any(name in tools for name in api_round.tool_names)),
            None,
        )

    def _relevance_terms(self, text: str) -> tuple[str, ...]:
        normalized = str(text or "").casefold()
        terms: list[str] = []
        seen: set[str] = set()

        def add(term: str) -> None:
            value = term.strip("._/:-")
            if len(value) < 2 or value in _STOP_TERMS or value in seen:
                return
            seen.add(value)
            terms.append(value)

        for match in _ASCII_TERM.finditer(normalized):
            add(match.group(0))
            if len(terms) >= self.max_relevance_terms:
                return tuple(terms)
        for match in _CJK_RUN.finditer(normalized):
            run = match.group(0)
            if len(run) <= 64:
                add(run)
            for index in range(max(0, len(run) - 1)):
                add(run[index : index + 2])
                if len(terms) >= self.max_relevance_terms:
                    return tuple(terms)
        return tuple(terms)

    def _round_scan_text(self, messages: list[dict[str, Any]], api_round: _ApiRound) -> str:
        chunks: list[str] = []
        remaining = self.relevance_scan_chars
        for message in messages[api_round.start : api_round.end]:
            values: list[Any] = [message.get("content")]
            calls = message.get("tool_calls")
            if isinstance(calls, list):
                for call in calls:
                    function = call.get("function") if isinstance(call, dict) else None
                    if isinstance(function, dict):
                        values.extend((function.get("name"), function.get("arguments")))
            for value in values:
                if remaining <= 0:
                    break
                chunk = str(value or "")[:remaining].casefold()
                chunks.append(chunk)
                remaining -= len(chunk)
            if remaining <= 0:
                break
        return "\n".join(chunks)

    def _is_relevant(self, scan_text: str, terms: tuple[str, ...]) -> bool:
        if not terms:
            return False
        matches = sum(term in scan_text for term in terms)
        return matches >= min(self.min_relevance_matches, len(terms))

    @staticmethod
    def _kept_reasons(protected: dict[int, set[str]]) -> tuple[tuple[int, tuple[str, ...]], ...]:
        return tuple((round_number, tuple(sorted(reasons))) for round_number, reasons in sorted(protected.items()))

    @staticmethod
    def _is_marker(message: dict[str, Any]) -> bool:
        return message.get("role") == "system" and str(message.get("content") or "").startswith(_MARKER_PREFIX)

    def _marker_content(
        self,
        removed: list[_ApiRound],
        removed_messages: list[dict[str, Any]],
        *,
        prior_markers: list[dict[str, Any]],
    ) -> str:
        canonical = json.dumps(
            {"prior_markers": prior_markers, "removed_messages": removed_messages},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=lambda value: f"<{type(value).__name__}>",
        )
        digest = hashlib.sha256(canonical.encode("utf-8", errors="replace")).hexdigest()
        tool_calls = sum(len(api_round.tool_names) for api_round in removed)
        metadata = {
            "algorithm": "complete-api-round-v1",
            "prior_markers": len(prior_markers),
            "removed_messages": len(removed_messages),
            "removed_rounds": len(removed),
            "removed_tool_calls": tool_calls,
            "sha256": digest,
        }
        full = (
            _MARKER_PREFIX
            + "\n"
            + json.dumps(
                metadata,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        if len(full) <= self.marker_chars:
            return full
        compact = (
            _MARKER_PREFIX
            + " "
            + json.dumps(
                {"removed_rounds": len(removed), "sha256": digest},
                separators=(",", ":"),
            )
        )
        if len(compact) <= self.marker_chars:
            return compact
        # marker_chars is clamped to at least 128, so this fallback retains a
        # useful digest prefix even if the marker wording grows in the future.
        return compact[: self.marker_chars]


__all__ = ["HistorySnipResult", "HistorySnipper"]
