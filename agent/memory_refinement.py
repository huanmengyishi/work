from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .config import AppConfig


_ALLOWED_KINDS = frozenset({"Lesson", "Bug", "Decision"})
_SAFE_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b((?:[a-z0-9]+[_-])*(?:api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"authorization|cookie|password|passwd|secret|token))"
    r"(\s*[=:]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_BEARER_TOKEN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}")
_KEY_SHAPED_VALUE = re.compile(r"(?i)\b(?:sk|key|token)-[A-Za-z0-9_-]{8,}")
_TAG_CREDENTIAL_FRAGMENT = re.compile(
    r"(?i)(?:^|[:=_.-])(?:api(?:[_.-]?key)|access(?:[_.-]?token)|refresh(?:[_.-]?token)|"
    r"authorization|bearer|cookie|password|passwd|secret|token|sk|key)"
    r"(?P<separator>[:=_.-]+)(?P<value>[A-Za-z0-9_.:=-]+)$"
)
_TAG_VALUE_SEPARATOR = re.compile(r"[:=_.-]+")
_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL,
)
_DSML_MARKERS = ("<｜｜DSML｜｜tool_calls>", "<｜｜DSML｜｜invoke")


@dataclass(frozen=True)
class MemoryRefinement:
    """One validated, bounded model refinement reused by both Memory records."""

    kind: str
    title: str
    experience: str
    reflection: str
    tags: tuple[str, ...]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["tags"] = list(self.tags)
        return value


class MemoryRefiner:
    """Build and validate one tool-free DeepSeek memory-refinement request.

    This component performs no network I/O. AgentRuntime owns the injected
    DeepSeek client, execution-budget admission, checkpoint, request metrics,
    and terminal ordering.
    """

    MAX_EVIDENCE_ITEMS = 32

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.min_tool_calls = self._bounded_int(
            config.get("memory.smart_reflection_min_tool_calls", 5),
            default=5,
            # Five managed calls is a cost-control floor, not merely a default.
            # Configuration may make refinement more selective, but must never
            # spend an extra model request on a smaller task.
            minimum=5,
            maximum=1_000,
        )
        self.max_input_chars = self._bounded_int(
            config.get("memory.smart_reflection_max_input_chars", 12_000),
            default=12_000,
            minimum=1_000,
            maximum=100_000,
        )
        self.max_output_tokens = self._bounded_int(
            config.get("memory.smart_reflection_max_output_tokens", 768),
            default=768,
            minimum=64,
            maximum=4_096,
        )
        self.max_output_chars = self._bounded_int(
            config.get("memory.smart_reflection_max_output_chars", 5_000),
            default=5_000,
            minimum=256,
            maximum=20_000,
        )

    def eligible(self, *, success: bool, current_tool_calls: int) -> tuple[bool, str]:
        if not success:
            return False, "task_not_completed"
        if not bool(self.config.get("memory.smart_reflection", True)):
            return False, "disabled"
        if current_tool_calls < self.min_tool_calls:
            return False, "below_tool_call_threshold"
        return True, "eligible"

    def build_messages(
        self,
        *,
        prompt: str,
        final: str,
        tool_calls: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        evidence: list[dict[str, Any]] = []
        for item in tool_calls[-self.MAX_EVIDENCE_ITEMS :]:
            request = item.get("request") if isinstance(item.get("request"), Mapping) else {}
            result = item.get("result") if isinstance(item.get("result"), Mapping) else {}
            capability = f"{request.get('tool', '?')}.{request.get('action', '?')}"
            record: dict[str, Any] = {
                "capability": redact_sensitive_text(capability, maximum=160),
                "success": bool(result.get("success")),
                "duration_ms": self._bounded_int(
                    result.get("duration_ms"),
                    default=0,
                    minimum=0,
                    maximum=86_400_000,
                ),
            }
            if not record["success"]:
                # Arguments, stdout, result bodies, and arbitrary data are
                # deliberately excluded. A short redacted diagnostic is enough
                # for the model to identify a durable engineering lesson.
                diagnostic = str(result.get("stderr") or "managed tool failed")
                record["failure_summary"] = redact_sensitive_text(diagnostic, maximum=320)
            evidence.append(record)

        payload = {
            "request": redact_sensitive_text(prompt, maximum=max(500, self.max_input_chars // 3)),
            "outcome": redact_sensitive_text(final, maximum=max(500, self.max_input_chars // 3)),
            "tool_call_count": len(tool_calls),
            "tool_evidence": evidence,
        }
        rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if len(rendered) > self.max_input_chars:
            # Rebuild rather than slicing JSON so the provider always receives a
            # complete structure. The most recent bounded evidence is retained.
            payload["request"] = str(payload["request"])[: max(256, self.max_input_chars // 5)]
            payload["outcome"] = str(payload["outcome"])[: max(256, self.max_input_chars // 5)]
            while (
                payload["tool_evidence"]
                and len(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                > self.max_input_chars
            ):
                payload["tool_evidence"].pop(0)
            rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

        return [
            {
                "role": "system",
                "content": (
                    "You refine completed engineering work into durable project memory. Return exactly one JSON "
                    "object and no markdown. Use only supplied evidence; do not infer credentials, source text, or "
                    "unseen actions. Required fields: kind (Lesson, Bug, or Decision), title, lesson, why, "
                    "when_to_apply, evidence_summary, reflection, tags (array), confidence (0 to 1). Keep every "
                    "field concise and actionable."
                ),
            },
            {"role": "user", "content": rendered},
        ]

    def parse_response(self, message: Mapping[str, Any], *, finish_reason: object) -> MemoryRefinement | None:
        normalized_finish = str(finish_reason or "").strip().lower()
        if normalized_finish not in {"", "stop"} or message.get("tool_calls"):
            return None
        raw = str(message.get("content") or "").strip()
        if not raw or len(raw) > self.max_output_chars or any(marker in raw for marker in _DSML_MARKERS):
            return None
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, RecursionError, ValueError):
            return None
        if not isinstance(value, dict) or len(value) > 16:
            return None

        kind = str(value.get("kind") or "").strip().title()
        if kind not in _ALLOWED_KINDS:
            return None
        fields: dict[str, str] = {}
        limits = {
            "title": 160,
            "lesson": 1_500,
            "why": 1_200,
            "when_to_apply": 1_200,
            "evidence_summary": 1_500,
            "reflection": 2_000,
        }
        for name, maximum in limits.items():
            text = redact_sensitive_text(str(value.get(name) or "").strip(), maximum=maximum)
            if not text:
                return None
            fields[name] = text

        tags = sanitize_memory_tags(value.get("tags"))
        try:
            confidence = float(value.get("confidence", 0.7))
        except (TypeError, ValueError, OverflowError):
            return None
        if not 0.0 <= confidence <= 1.0:
            return None

        experience = "\n".join(
            [
                "经验",
                fields["lesson"],
                "",
                "原因",
                fields["why"],
                "",
                "适用时机",
                fields["when_to_apply"],
                "",
                "证据摘要",
                fields["evidence_summary"],
            ]
        )
        return MemoryRefinement(
            kind=kind,
            title=fields["title"],
            experience=redact_sensitive_text(experience, maximum=self.max_output_chars),
            reflection=fields["reflection"],
            tags=tags,
            confidence=confidence,
        )

    @staticmethod
    def current_turn_tool_calls(tool_calls: list[dict[str, Any]], *, turn: int) -> list[dict[str, Any]]:
        return [
            item
            for item in tool_calls
            if isinstance(item, dict)
            and int(item.get("turn") or 1) == turn
            and isinstance(item.get("request"), dict)
            and isinstance(item.get("result"), dict)
        ]

    @staticmethod
    def _bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
        if isinstance(value, bool):
            return default
        try:
            parsed = int(value)
        except (TypeError, ValueError, OverflowError):
            return default
        return max(minimum, min(maximum, parsed))


def redact_sensitive_text(value: str, *, maximum: int) -> str:
    """Remove credential-shaped content before API input or Memory output."""

    text = str(value).replace("\x00", "")
    text = _PRIVATE_KEY.sub("[redacted-private-key]", text)
    # Redact a complete Bearer credential before assignment redaction.  Doing
    # this in the opposite order would turn ``Authorization: Bearer <token>``
    # into ``Authorization: [redacted] <token>`` and leave the secret behind.
    text = _BEARER_TOKEN.sub("Bearer [redacted]", text)
    text = _SECRET_ASSIGNMENT.sub(lambda match: f"{match.group(1)}{match.group(2)}[redacted]", text)
    text = _KEY_SHAPED_VALUE.sub("[redacted-key]", text)
    if len(text) <= maximum:
        return text
    suffix = "...[truncated]"
    if maximum <= len(suffix):
        return suffix[: max(0, maximum)]
    return text[: maximum - len(suffix)] + suffix


def sanitize_memory_tags(value: object, *, maximum_items: int = 12) -> tuple[str, ...]:
    """Return bounded, syntax-safe tags after rejecting credential-shaped values.

    Tags are independently validated because they bypass the prose fields and
    are persisted as searchable metadata.  A value changed by the credential
    redactor is discarded rather than persisting a bracketed redaction marker
    as a misleading tag.
    """

    if not isinstance(value, (list, tuple)):
        return ()
    limit = max(0, min(12, int(maximum_items)))
    accepted: list[str] = []
    for item in value[:limit]:
        if not isinstance(item, str):
            continue
        candidate = item.strip()
        if (
            not candidate
            or len(candidate) > 64
            or _looks_like_tagged_credential(candidate)
            or not _SAFE_TAG.fullmatch(candidate)
        ):
            continue
        normalized = candidate.lower()
        if normalized not in accepted:
            accepted.append(normalized)
    return tuple(accepted)


def _looks_like_tagged_credential(value: str) -> bool:
    """Detect opaque values attached to credential-keyword tag prefixes.

    A keyword alone is not sensitive: tags such as ``token-budget`` and
    ``secret-scanning`` describe engineering concepts and remain useful.  A
    sufficiently long mixed alphanumeric payload (or a very long opaque
    numeric/alphabetic payload) is credential-shaped and is rejected.
    """

    match = _TAG_CREDENTIAL_FRAGMENT.search(value)
    if match is None:
        return False
    separator = match.group("separator")
    if ":" in separator or "=" in separator:
        return True
    compact = _TAG_VALUE_SEPARATOR.sub("", match.group("value"))
    if len(compact) < 10:
        return False
    has_letter = any(character.isalpha() for character in compact)
    has_digit = any(character.isdigit() for character in compact)
    if has_letter and has_digit:
        return True
    if compact.isdigit() and len(compact) >= 10:
        return True
    return len(compact) >= 24 and len(set(compact.casefold())) >= 10


__all__ = ["MemoryRefinement", "MemoryRefiner", "redact_sensitive_text", "sanitize_memory_tags"]
