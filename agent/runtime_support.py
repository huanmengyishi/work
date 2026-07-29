from __future__ import annotations

import hashlib
import re
from typing import Any

from .constants import (
    DEEPSEEK_TOOL_PROTOCOL_MARKERS,
    MAX_TOOL_CALLS_PER_MODEL_RESPONSE,
    USABLE_FINISH_REASONS,
)
from .unicode_text import normalize_unicode_text


_DATE_LITERAL_RE = re.compile(r"(?<!\d)20\d{2}(?:年\s*\d{1,2}月(?:\s*\d{1,2}日)?|[-/.]\d{1,2}(?:[-/.]\d{1,2})?)(?!\d)")


def _finish_reason_label(value: object) -> str:
    return normalize_unicode_text(str(value or "")).strip().lower()[:64]


def _has_usable_finish_reason(value: object) -> bool:
    return _finish_reason_label(value) in USABLE_FINISH_REASONS


def _tool_protocol_violation(message: dict[str, Any]) -> str:
    """Describe model tool protocol that cannot be accepted as answer text."""

    raw_tool_calls = message.get("tool_calls")
    if raw_tool_calls:
        return "structured tool calls"
    return _tool_protocol_text_violation(message)


def _tool_protocol_text_violation(message: dict[str, Any]) -> str:
    content = str(message.get("content") or "")
    if any(marker in content for marker in DEEPSEEK_TOOL_PROTOCOL_MARKERS):
        return "DeepSeek tool-call protocol text"
    return ""


def _date_key(value: str) -> tuple[int, ...] | None:
    numbers = [int(item) for item in re.findall(r"\d+", value)]
    if len(numbers) < 2:
        return None
    year, month = numbers[:2]
    if not (2000 <= year <= 2099 and 1 <= month <= 12):
        return None
    if len(numbers) >= 3:
        day = numbers[2]
        if not 1 <= day <= 31:
            return None
        return year, month, day
    return year, month


def _date_keys_from_text(value: str) -> set[tuple[int, ...]]:
    return {key for item in _DATE_LITERAL_RE.findall(value) if (key := _date_key(item)) is not None}


def _normalize_assistant_tool_calls(
    message: dict[str, Any],
    *,
    turn: int,
    round_number: int,
) -> tuple[dict[str, Any], int, int]:
    """Return one protocol-safe assistant message before any tool executes."""

    raw_calls = message.get("tool_calls")
    if not raw_calls:
        return message, 0, 0
    if isinstance(raw_calls, (list, tuple)):
        dropped = max(0, len(raw_calls) - MAX_TOOL_CALLS_PER_MODEL_RESPONSE)
        items = list(raw_calls[:MAX_TOOL_CALLS_PER_MODEL_RESPONSE])
    else:
        dropped = 0
        items = [raw_calls]
    normalized_calls: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    changed = 0
    for index, raw_call in enumerate(items):
        source = raw_call if isinstance(raw_call, dict) else {}
        call = dict(source)
        raw_function = call.get("function")
        function = dict(raw_function) if isinstance(raw_function, dict) else {}
        function["name"] = str(function.get("name") or "")
        if function.get("arguments") is None:
            function["arguments"] = "{}"
        call["type"] = "function"
        call["function"] = function

        call_id = str(call.get("id") or "").strip()
        if len(call_id) > 200:
            digest = hashlib.sha256(call_id.encode("utf-8", errors="replace")).hexdigest()[:32]
            call_id = "deep-agent-call-" + digest
        if not call_id or call_id in seen_ids:
            base = f"deep-agent-call-t{turn}-r{round_number}-i{index + 1}"
            call_id = base
            suffix = 2
            while call_id in seen_ids:
                call_id = f"{base}-{suffix}"
                suffix += 1
        seen_ids.add(call_id)
        call["id"] = call_id
        normalized_calls.append(call)
        if not isinstance(raw_call, dict) or call != raw_call:
            changed += 1

    updated = dict(message)
    updated["tool_calls"] = normalized_calls
    return updated, changed, dropped
