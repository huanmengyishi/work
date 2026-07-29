"""Stable evidence codec for bounded model-visible tool history."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any


COMPACTED_MARKER = "[Deep Agent compacted tool result]"
METADATA_MARKER = "[Deep Agent compacted metadata]"
SAFE_ARGUMENT_KEYS = (
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


@dataclass(frozen=True)
class ToolEvidence:
    tool: str
    success: bool | None
    target: dict[str, Any]
    original_chars: int
    sha256: str


def preview(content: str, *, evidence: ToolEvidence, limit: int) -> str:
    prefix = COMPACTED_MARKER + "\n" + evidence_json(evidence) + "\npreview:\n"
    if len(prefix) >= limit:
        return emergency_summary(content, evidence=evidence, limit=limit)
    available = max(0, limit - len(prefix) - len("\n...[middle omitted]...\n"))
    head = available // 2
    tail = available - head
    if len(content) <= available:
        body = content
    else:
        body = content[:head] + "\n...[middle omitted]...\n" + (content[-tail:] if tail else "")
    return (prefix + body)[:limit]


def metadata(evidence: ToolEvidence) -> str:
    return METADATA_MARKER + "\n" + evidence_json(evidence)


def emergency_summary(content: str, *, evidence: ToolEvidence, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(content) <= limit:
        return content
    compact_target = json.dumps(evidence.target, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    candidates = [
        metadata(evidence),
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


def evidence_json(evidence: ToolEvidence) -> str:
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


def extract_evidence(content: str, *, name: str, args: dict[str, Any]) -> ToolEvidence:
    parsed = parse_evidence(content)
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
    target = safe_target(args)
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
    return ToolEvidence(
        tool=str(payload.get("tool") or name) if isinstance(payload, dict) else name,
        success=success,
        target=target,
        original_chars=original_chars,
        sha256=sha256,
    )


def parse_evidence(content: str) -> ToolEvidence | None:
    if not is_compacted(content):
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
            return ToolEvidence(
                tool=str(payload.get("tool") or "unknown"),
                success=success if isinstance(success, bool) else None,
                target=payload.get("target") if isinstance(payload.get("target"), dict) else {},
                original_chars=original_chars,
                sha256=recorded_hash,
            )

    # Compatibility with previews produced by early v0.11.0 candidates.
    values: dict[str, str] = {}
    legacy_lines = list(lines[1:])
    if lines and lines[0].startswith(METADATA_MARKER):
        legacy_lines.insert(0, lines[0][len(METADATA_MARKER) :].strip())
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
    return ToolEvidence(
        tool=values.get("tool") or "unknown",
        success=success,
        target=target if isinstance(target, dict) else {},
        original_chars=original_chars,
        sha256=values.get("sha256") or "",
    )


def safe_target(args: dict[str, Any]) -> dict[str, Any]:
    return {key: args[key] for key in SAFE_ARGUMENT_KEYS if key in args}


def is_compacted(content: str) -> bool:
    return content.startswith(COMPACTED_MARKER) or content.startswith(METADATA_MARKER)


__all__ = [
    "COMPACTED_MARKER",
    "METADATA_MARKER",
    "ToolEvidence",
    "emergency_summary",
    "evidence_json",
    "extract_evidence",
    "is_compacted",
    "metadata",
    "parse_evidence",
    "preview",
    "safe_target",
]
