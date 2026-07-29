"""Permission-gated lifecycle for one managed tool request.

``ToolExecutor`` is intentionally independent from ``ToolManager``.  It owns
the complete request lifecycle after a :class:`ToolRequest` has been created:
capability resolution, permission evaluation, argument binding, approval,
handler entry, bounded result persistence, and lifecycle events.  A caller
cannot reach a registered handler through this class without crossing the
``PermissionManager`` boundary first.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from ..constants import (
    EVENT_COUNT_MAX,
    EVENT_DURATION_MAX_MS,
    EVENT_LABEL_MAX_CHARS,
    HEALTH_ERROR_MAX_CHARS,
)
from ..events import EventBus, sanitize_for_log
from .base import ToolRequest, ToolResult, _bounded_result_data, _head_tail, elapsed_ms
from .permission import PermissionManager
from .registry import ToolCapability, ToolCapabilityRegistry
from .result_store import ToolResultStoreError


ApprovalHandler = Callable[[ToolRequest, ToolCapability, str], bool]
ApprovalSummary = Callable[[ToolRequest, ToolCapability], str]
AutoApproveCapabilities = Callable[[], set[str]]

_SENSITIVE_ERROR_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|authorization|cookie|password|passwd|secret|token)"
    r"(\s*[=:]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_URL_CREDENTIALS = re.compile(r"(?i)\b(https?://)[^\s/@:]+:[^\s/@]+@")


@dataclass(frozen=True)
class ToolExecutionOwnership:
    """Immutable state/event attribution captured before worker dispatch."""

    state: Any | None
    session_id: str | None
    run_id: str | None
    events: EventBus | None


class ResultStoreProtocol(Protocol):
    preview_chars: int

    def persist(self, result: ToolResult, *, session_id: str, request_id: str) -> ToolResult: ...


class ToolExecutor:
    """Execute registered capabilities through permission and result boundaries."""

    def __init__(
        self,
        registry: ToolCapabilityRegistry,
        permission: PermissionManager,
        *,
        project_id: str,
        result_store: ResultStoreProtocol | None = None,
        approval_handler: ApprovalHandler | None = None,
        approval_summary: ApprovalSummary | None = None,
        auto_approve_capabilities: AutoApproveCapabilities | None = None,
        auto_approve: bool = False,
        yolo: bool = False,
        super_yolo: bool = False,
    ) -> None:
        self.registry = registry
        self.permission = permission
        self.project_id = project_id
        self.result_store = result_store
        self.approval_handler = approval_handler
        self.approval_summary = approval_summary or self._default_approval_summary
        self.auto_approve_capabilities = auto_approve_capabilities or (lambda: set())
        self.auto_approve = bool(auto_approve)
        self.yolo = bool(yolo)
        self.super_yolo = bool(super_yolo)

    def schemas(self) -> list[dict[str, Any]]:
        return self.registry.schemas()

    def execute_model_call(
        self,
        name: str,
        arguments: str | dict[str, Any] | None,
        *,
        request_id: str | None = None,
        runtime_denied_reason: str | None = None,
    ) -> tuple[ToolRequest, ToolResult]:
        """Convenience entry for standalone use without Manager state ownership."""

        argument_error: str | None = None
        try:
            if arguments is None:
                args: dict[str, Any] = {}
            elif isinstance(arguments, dict):
                args = arguments
            elif isinstance(arguments, str):
                if not arguments.strip():
                    args = {}
                else:
                    decoded = json.loads(arguments)
                    if not isinstance(decoded, dict):
                        raise ValueError("tool arguments must decode to an object")
                    args = decoded
            else:
                raise TypeError("tool arguments must be a JSON object or string")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            args = {}
            argument_error = f"invalid arguments for {name}: {exc}"
        request = self.registry.request(name, args, request_id=request_id)
        ownership = ToolExecutionOwnership(None, None, None, None)
        return request, self.execute(
            request,
            ownership=ownership,
            runtime_denied_reason=runtime_denied_reason,
            argument_error=argument_error,
        )

    def configure_permissions(
        self,
        *,
        yolo: bool | None = None,
        super_yolo: bool | None = None,
    ) -> None:
        if yolo is not None:
            self.yolo = bool(yolo)
        if super_yolo is not None:
            self.super_yolo = bool(super_yolo)

    def execute(
        self,
        request: ToolRequest,
        *,
        ownership: ToolExecutionOwnership,
        runtime_denied_reason: str | None = None,
        argument_error: str | None = None,
    ) -> ToolResult:
        """Run one complete ``ToolRequest -> PermissionManager -> ToolResult`` lifecycle."""

        capability, handler = self.registry.resolve(request.capability)
        if capability is None or handler is None:
            if runtime_denied_reason:
                result = self.runtime_denied_result(request, runtime_denied_reason)
                self.publish("tool.denied", request, result, ownership=ownership)
                return result
            return self.not_executed_result(
                request,
                f"unknown tool capability: {request.capability}",
            )

        # This call deliberately precedes every possible handler entry.  Even
        # SUPER YOLO is a decision made by PermissionManager, not a direct
        # dispatch path around it.
        decision = self.permission.evaluate(request, capability, super_yolo=self.super_yolo)
        if runtime_denied_reason:
            result = self.runtime_denied_result(request, runtime_denied_reason)
            self.publish("tool.denied", request, result, ownership=ownership)
            return result
        if argument_error:
            result = self.not_executed_result(request, argument_error)
            self.publish("tool.denied", request, result, ownership=ownership)
            return result
        if not decision.allowed:
            result = self.not_executed_result(request, decision.reason)
            self.publish("tool.denied", request, result, ownership=ownership)
            return result

        binding_error = self.handler_argument_error(handler, request.args, capability.name)
        if binding_error:
            result = self.not_executed_result(request, binding_error)
            self.publish("tool.denied", request, result, ownership=ownership)
            return result

        auto_approved = (
            self.super_yolo or self.yolo or (self.auto_approve and capability.name in self.auto_approve_capabilities())
        )
        if capability.requires_confirmation and not auto_approved:
            result = self._request_approval(request, capability, ownership)
            if result is not None:
                return result

        self.publish("tool.started", request, None, ownership=ownership)
        started = time.monotonic()
        try:
            result = handler(**request.args)
            if not isinstance(result, ToolResult):
                result = ToolResult(True, str(result))
        except Exception as exc:
            result = ToolResult(False, "", str(exc))

        handler_result = result.with_execution(
            request_id=request.request_id,
            duration_ms=result.duration_ms,
        )
        result = self._persist_result(request, handler_result, ownership)
        result = result.with_execution(
            request_id=request.request_id,
            duration_ms=elapsed_ms(started),
        )
        self.publish("tool.finished", request, result, ownership=ownership)
        return result

    def _request_approval(
        self,
        request: ToolRequest,
        capability: ToolCapability,
        ownership: ToolExecutionOwnership,
    ) -> ToolResult | None:
        try:
            summary = self.approval_summary(request, capability)
        except Exception as exc:
            result = self.not_executed_result(request, f"could not prepare approval: {exc}")
            self.publish("tool.denied", request, result, ownership=ownership)
            return result
        if self.approval_handler is None:
            result = self.not_executed_result(
                request,
                "operation requires user confirmation; use interactive mode or --yolo",
            )
            self.publish("tool.denied", request, result, ownership=ownership)
            return result
        try:
            approved = self.approval_handler(request, capability, summary)
        except Exception as exc:
            result = self.not_executed_result(request, f"approval failed: {exc}")
            self.publish("tool.denied", request, result, ownership=ownership)
            return result
        if approved:
            return None
        result = self.not_executed_result(request, "operation denied by user")
        self.publish("tool.denied", request, result, ownership=ownership)
        return result

    def _persist_result(
        self,
        request: ToolRequest,
        result: ToolResult,
        ownership: ToolExecutionOwnership,
    ) -> ToolResult:
        if self.result_store is None or request.capability == "tool_result.read" or ownership.session_id is None:
            return result
        try:
            return self.result_store.persist(
                result,
                session_id=ownership.session_id,
                request_id=request.request_id,
            )
        except ToolResultStoreError as exc:
            # The approved handler may already have performed a side effect.
            # A private attachment failure must never relabel that execution
            # as failed and encourage an unsafe repeat.
            return self.attachment_persistence_fallback(result, exc)

    def attachment_persistence_fallback(
        self,
        result: ToolResult,
        error: ToolResultStoreError,
    ) -> ToolResult:
        if self.result_store is None:
            return result
        if result.success:
            stdout_chars = self.result_store.preview_chars * 3 // 4
            stderr_chars = self.result_store.preview_chars - stdout_chars
        else:
            stderr_chars = self.result_store.preview_chars * 3 // 4
            stdout_chars = self.result_store.preview_chars - stderr_chars
        bounded_data = _bounded_result_data(result.data or {})
        data = dict(bounded_data) if isinstance(bounded_data, dict) else {"value": bounded_data}
        data["attachment_persistence_error"] = {
            "type": type(error).__name__,
            "message": self.health_error_summary(ToolResult(False, "", str(error))),
            "result_preserved": True,
            "full_body_available": False,
            "stdout_chars": len(result.stdout),
            "stderr_chars": len(result.stderr),
            "stdout_sha256": hashlib.sha256(result.stdout.encode("utf-8", errors="replace")).hexdigest(),
            "stderr_sha256": hashlib.sha256(result.stderr.encode("utf-8", errors="replace")).hexdigest(),
        }
        return ToolResult(
            result.success,
            _head_tail(result.stdout, stdout_chars),
            _head_tail(result.stderr, stderr_chars),
            data=data,
            duration_ms=result.duration_ms,
            request_id=result.request_id,
        )

    @staticmethod
    def runtime_denied_result(request: ToolRequest, reason: str) -> ToolResult:
        return ToolExecutor.not_executed_result(
            request,
            str(reason)[:2_000],
            data={"runtime_denied": True},
        )

    @staticmethod
    def not_executed_result(
        request: ToolRequest,
        error: str,
        *,
        data: dict[str, Any] | None = None,
    ) -> ToolResult:
        result_data = dict(data or {})
        result_data["not_executed"] = True
        return ToolResult(
            False,
            "",
            str(error),
            data=result_data,
            request_id=request.request_id,
        )

    @staticmethod
    def handler_argument_error(
        handler: Callable[..., ToolResult],
        arguments: dict[str, Any],
        capability_name: str,
    ) -> str | None:
        try:
            signature = inspect.signature(handler)
        except (TypeError, ValueError):
            return None
        try:
            signature.bind(**arguments)
        except TypeError as exc:
            return f"invalid arguments for {capability_name}: {exc}"
        return None

    @staticmethod
    def result_is_health_failure(result: ToolResult) -> bool:
        error = str(result.stderr or "").lower()
        markers = (
            "timeout",
            "timed out",
            "command not found",
            "dependency",
            "unavailable",
            "could not start",
            "connection refused",
            "connection reset",
            "not installed",
            "closed its input",
            "broken pipe",
        )
        return any(marker in error for marker in markers)

    @staticmethod
    def health_error_summary(result: ToolResult) -> str:
        value = str(sanitize_for_log(str(result.stderr or "")))
        value = _URL_CREDENTIALS.sub(r"\1[redacted]@", value)
        value = _SENSITIVE_ERROR_ASSIGNMENT.sub(
            lambda match: f"{match.group(1)}{match.group(2)}[redacted]",
            value,
        )
        value = " ".join(value.replace("\x00", " ").split())
        if len(value) <= HEALTH_ERROR_MAX_CHARS:
            return value
        suffix = "...[truncated]"
        return value[: HEALTH_ERROR_MAX_CHARS - len(suffix)] + suffix

    def publish(
        self,
        event_name: str,
        request: ToolRequest,
        result: ToolResult | None,
        *,
        ownership: ToolExecutionOwnership,
    ) -> None:
        if not ownership.events:
            return
        payload: dict[str, Any] = {
            "request": {
                "tool": self._event_label(request.tool),
                "action": self._event_label(request.action),
                "capability": self._event_label(request.capability),
                "request_id": self._event_label(request.request_id),
                "argument_count": self._event_count(request.args),
            }
        }
        if result is not None:
            health_failure = (
                event_name == "tool.finished" and not result.success and self.result_is_health_failure(result)
            )
            payload["result"] = {
                "success": result.success,
                "duration_ms": self._event_duration_ms(result.duration_ms),
                "request_id": self._event_label(result.request_id),
                "data_field_count": self._event_count(result.data) if isinstance(result.data, dict) else 0,
                "health_failure": health_failure,
                "error": self.health_error_summary(result) if health_failure else "",
            }
        ownership.events.publish(
            event_name,
            payload,
            project_id=self.project_id,
            session_id=ownership.session_id,
            run_id=ownership.run_id,
        )

    @staticmethod
    def _event_label(value: Any) -> str | None:
        if value is None:
            return None
        label = str(sanitize_for_log(str(value))).strip()
        if len(label) <= EVENT_LABEL_MAX_CHARS:
            return label
        suffix = "...[truncated]"
        return label[: EVENT_LABEL_MAX_CHARS - len(suffix)] + suffix

    @staticmethod
    def _event_count(value: Any) -> int:
        try:
            count = len(value)
        except (TypeError, ValueError, OverflowError):
            return 0
        return max(0, min(count, EVENT_COUNT_MAX))

    @staticmethod
    def _event_duration_ms(value: Any) -> int:
        try:
            duration = int(value or 0)
        except (TypeError, ValueError, OverflowError):
            return 0
        return max(0, min(duration, EVENT_DURATION_MAX_MS))

    @staticmethod
    def _default_approval_summary(request: ToolRequest, capability: ToolCapability) -> str:
        return f"Allow {capability.name}?"


__all__ = ["ToolExecutionOwnership", "ToolExecutor"]
