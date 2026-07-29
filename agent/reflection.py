from __future__ import annotations

from collections import Counter
from typing import Any

from .config import AppConfig
from .memory_refinement import redact_sensitive_text


class ReflectionEngine:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def reflect(
        self,
        *,
        prompt: str,
        final: str,
        error: str,
        tool_calls: list[dict[str, Any]],
        success: bool,
        smart_text: str | None = None,
    ) -> str | None:
        rule = self._rule_reflection(prompt, final, error, tool_calls, success)
        smart = redact_sensitive_text(str(smart_text or "").strip(), maximum=5_000)
        if not smart:
            return rule
        if not rule:
            rule = "\n".join(
                [
                    "Reflection",
                    f"Request: {redact_sensitive_text(prompt, maximum=1200)}",
                    f"Outcome: {'success' if success else 'failed'}",
                    f"Tool calls: {len(tool_calls)}",
                ]
            )
        return f"{rule}\n\nAI Reflection\n{smart}"

    @staticmethod
    def _rule_reflection(
        prompt: str,
        final: str,
        error: str,
        tool_calls: list[dict[str, Any]],
        success: bool,
    ) -> str | None:
        failures: list[tuple[str, str]] = []
        durations: list[int] = []
        for item in tool_calls:
            request = item.get("request") or {}
            result = item.get("result") or {}
            name = f"{request.get('tool', '?')}.{request.get('action', '?')}"
            durations.append(int(result.get("duration_ms") or 0))
            if not result.get("success"):
                failures.append((name, redact_sensitive_text(str(result.get("stderr") or "failed"), maximum=500)))
        repeated = Counter(name for name, _ in failures)
        timeouts = [detail for _, detail in failures if "timeout" in detail.lower() or "timed out" in detail.lower()]
        high_cost = len(tool_calls) >= 8 or sum(durations) >= 120_000
        if success and not failures and not high_cost:
            return None
        observations: list[str] = []
        if failures:
            observations.append(f"Failed tool calls: {len(failures)} of {len(tool_calls)}.")
        observations.extend(
            f"{name} failed {count} times; change the approach before another retry."
            for name, count in repeated.items()
            if count >= 2
        )
        if len(timeouts) >= 3:
            observations.append("Three or more timeouts occurred; check proxy, dependency health, and timeout limits.")
        if high_cost:
            observations.append(
                "Execution used at least 8 tools or 120 seconds of tool time; improve plan granularity."
            )
        if not success:
            observations.append(
                "The task did not complete; resume from the saved execution context instead of restarting."
            )
        if not observations:
            observations.append("The task completed after recoverable failures; preserve the successful recovery path.")
        evidence = "\n".join(f"- {name}: {detail}" for name, detail in failures[:8]) or "- No failed tool evidence."
        return "\n".join(
            [
                "Reflection",
                f"Request: {redact_sensitive_text(prompt, maximum=1200)}",
                f"Outcome: {'success' if success else 'failed'}",
                "Observations:",
                *[f"- {item}" for item in observations],
                "Evidence:",
                evidence,
                "Prevention:",
                "Use Task Graph dependencies, capability health, and execution context before retrying.",
                f"Result excerpt: {redact_sensitive_text(final or error, maximum=1200)}",
            ]
        )
