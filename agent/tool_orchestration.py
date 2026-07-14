from __future__ import annotations

from concurrent.futures import CancelledError, FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

from .tools.base import ToolRequest, ToolResult
from .tools.registry import ToolCapability, ToolHandler


DEFAULT_MAX_PARALLEL_READS = 4
MAX_PARALLEL_READS_HARD_LIMIT = 16
_READ_ONLY_PERMISSIONS = frozenset({"read"})
# file.diff writes a durable preview even though it does not modify the target
# file. Treat that internal state transition as a serial barrier.
_STATEFUL_READ_CAPABILITIES = frozenset({"file.diff"})


@dataclass(frozen=True)
class PreparedToolCall:
    """One normalized model tool call after Runtime policy decisions."""

    model_name: str
    arguments: str | dict[str, Any] | None
    request_id: str | None = None
    runtime_denied_reason: str | None = None


ToolExecution = tuple[ToolRequest, ToolResult]


class ToolBatchInterrupted(BaseException):
    """Carry a complete, ordered result batch while preserving an interrupt."""

    def __init__(self, executions: Sequence[ToolExecution], cause: BaseException) -> None:
        super().__init__(str(cause))
        self.executions = list(executions)
        self.cause = cause


class _CapabilityRegistry(Protocol):
    def resolve(self, name: str) -> tuple[ToolCapability | None, ToolHandler | None]: ...


class ModelToolExecutor(Protocol):
    registry: _CapabilityRegistry

    def execute_model_call(
        self,
        name: str,
        arguments: str | dict[str, Any] | None,
        *,
        request_id: str | None = None,
        runtime_denied_reason: str | None = None,
    ) -> ToolExecution: ...


def execute_model_tool_calls(
    executor: ModelToolExecutor,
    calls: Sequence[PreparedToolCall],
    *,
    max_concurrency: int = DEFAULT_MAX_PARALLEL_READS,
    sequential_policy: Callable[[PreparedToolCall, Sequence[ToolExecution]], PreparedToolCall] | None = None,
) -> list[ToolExecution]:
    """Execute consecutive concurrency-safe reads and retain model call order.

    Capabilities that explicitly opt in as concurrency-safe, are pure reads,
    and require no confirmation may overlap up to ``max_concurrency``. Every
    other call is a serial barrier: the preceding read batch finishes before
    it starts, and the following batch cannot start before it finishes.
    Tool execution still goes through ``execute_model_call``, preserving the
    ToolRequest -> PermissionManager -> ToolResult boundary.
    """

    workers = _bounded_workers(max_concurrency)
    execution_context = _capture_execution_context(executor)
    executions: list[ToolExecution] = []
    index = 0
    while index < len(calls):
        call = calls[index]
        if sequential_policy is not None:
            call = sequential_policy(call, tuple(executions))
        if sequential_policy is not None or not _is_concurrency_safe_read(executor, call):
            try:
                executions.append(_execute_one(executor, call, execution_context))
            except CancelledError as exc:
                reason = _interruption_reason(exc)
                executions.append(
                    _synthetic_failed_execution(
                        executor,
                        call,
                        reason,
                        interrupted=True,
                        execution_context=execution_context,
                    )
                )
                executions.extend(
                    _synthetic_failed_execution(
                        executor,
                        item,
                        reason,
                        interrupted=True,
                        execution_context=execution_context,
                    )
                    for item in calls[index + 1 :]
                )
                raise ToolBatchInterrupted(executions, exc) from exc
            except Exception as exc:
                executions.append(
                    _synthetic_failed_execution(
                        executor,
                        call,
                        _exception_reason(exc),
                        execution_context=execution_context,
                    )
                )
            except BaseException as exc:
                reason = _interruption_reason(exc)
                executions.append(
                    _synthetic_failed_execution(
                        executor,
                        call,
                        reason,
                        interrupted=True,
                        execution_context=execution_context,
                    )
                )
                executions.extend(
                    _synthetic_failed_execution(
                        executor,
                        item,
                        reason,
                        interrupted=True,
                        execution_context=execution_context,
                    )
                    for item in calls[index + 1 :]
                )
                raise ToolBatchInterrupted(executions, exc) from exc
            index += 1
            continue

        batch_end = index + 1
        while batch_end < len(calls) and _is_concurrency_safe_read(executor, calls[batch_end]):
            batch_end += 1
        batch = calls[index:batch_end]
        if workers == 1 or len(batch) == 1:
            for batch_index, item in enumerate(batch):
                try:
                    executions.append(_execute_one(executor, item, execution_context))
                except CancelledError as exc:
                    reason = _interruption_reason(exc)
                    executions.append(
                        _synthetic_failed_execution(
                            executor,
                            item,
                            reason,
                            interrupted=True,
                            execution_context=execution_context,
                        )
                    )
                    executions.extend(
                        _synthetic_failed_execution(
                            executor,
                            remaining,
                            reason,
                            interrupted=True,
                            execution_context=execution_context,
                        )
                        for remaining in calls[index + batch_index + 1 :]
                    )
                    raise ToolBatchInterrupted(executions, exc) from exc
                except Exception as exc:
                    executions.append(
                        _synthetic_failed_execution(
                            executor,
                            item,
                            _exception_reason(exc),
                            execution_context=execution_context,
                        )
                    )
                except BaseException as exc:
                    reason = _interruption_reason(exc)
                    executions.append(
                        _synthetic_failed_execution(
                            executor,
                            item,
                            reason,
                            interrupted=True,
                            execution_context=execution_context,
                        )
                    )
                    executions.extend(
                        _synthetic_failed_execution(
                            executor,
                            remaining,
                            reason,
                            interrupted=True,
                            execution_context=execution_context,
                        )
                        for remaining in calls[index + batch_index + 1 :]
                    )
                    raise ToolBatchInterrupted(executions, exc) from exc
        else:
            try:
                executions.extend(_execute_parallel_batch(executor, batch, workers, execution_context))
            except ToolBatchInterrupted as exc:
                reason = _interruption_reason(exc.cause)
                complete_executions = [*executions, *exc.executions]
                complete_executions.extend(
                    _synthetic_failed_execution(
                        executor,
                        item,
                        reason,
                        interrupted=True,
                        execution_context=execution_context,
                    )
                    for item in calls[batch_end:]
                )
                raise ToolBatchInterrupted(complete_executions, exc.cause) from exc.cause
        index = batch_end
    return executions


def _execute_parallel_batch(
    executor: ModelToolExecutor,
    batch: Sequence[PreparedToolCall],
    workers: int,
    execution_context: object | None,
) -> list[ToolExecution]:
    """Run one safe-read batch and retain completed work across interruption."""

    pool = ThreadPoolExecutor(
        max_workers=min(workers, len(batch)),
        thread_name_prefix="deep-agent-read",
    )
    futures: list[Future[ToolExecution]] = [
        pool.submit(_execute_one, executor, item, execution_context) for item in batch
    ]
    future_indexes = {future: index for index, future in enumerate(futures)}
    executions: list[ToolExecution | None] = [None] * len(batch)
    pending = set(futures)
    interruption: BaseException | None = None
    try:
        while pending and interruption is None:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                index = future_indexes[future]
                try:
                    executions[index] = future.result()
                except CancelledError as exc:
                    interruption = exc
                    break
                except Exception as exc:
                    executions[index] = _synthetic_failed_execution(
                        executor,
                        batch[index],
                        _exception_reason(exc),
                        execution_context=execution_context,
                    )
                except BaseException as exc:
                    interruption = exc
                    break
    except BaseException as exc:
        interruption = exc

    if interruption is None:
        pool.shutdown(wait=True)
        return [item for item in executions if item is not None]

    reason = _interruption_reason(interruption)
    for future in pending:
        future.cancel()
    for index, future in enumerate(futures):
        if executions[index] is not None:
            continue
        if future.done() and not future.cancelled():
            try:
                executions[index] = future.result()
                continue
            except CancelledError:
                pass
            except Exception as exc:
                executions[index] = _synthetic_failed_execution(
                    executor,
                    batch[index],
                    _exception_reason(exc),
                    execution_context=execution_context,
                )
                continue
            except BaseException:
                pass
        executions[index] = _synthetic_failed_execution(
            executor,
            batch[index],
            reason,
            interrupted=True,
            execution_context=execution_context,
        )
    # Read-only, explicitly concurrency-safe handlers may still be unwinding.
    # Do not make Ctrl+C wait for them; queued futures are cancelled and every
    # unresolved model call already has a synthetic failed protocol result.
    pool.shutdown(wait=False, cancel_futures=True)
    raise ToolBatchInterrupted([item for item in executions if item is not None], interruption) from interruption


def _execute_one(
    executor: ModelToolExecutor,
    call: PreparedToolCall,
    execution_context: object | None = None,
) -> ToolExecution:
    contextual_execute = getattr(executor, "execute_model_call_in_context", None)
    if execution_context is not None and callable(contextual_execute):
        return contextual_execute(
            execution_context,
            call.model_name,
            call.arguments,
            request_id=call.request_id,
            runtime_denied_reason=call.runtime_denied_reason,
        )
    return executor.execute_model_call(
        call.model_name,
        call.arguments,
        request_id=call.request_id,
        runtime_denied_reason=call.runtime_denied_reason,
    )


def _synthetic_failed_execution(
    executor: ModelToolExecutor,
    call: PreparedToolCall,
    reason: str,
    *,
    interrupted: bool = False,
    execution_context: object | None = None,
) -> ToolExecution:
    """Create one failed pair without invoking the requested tool handler."""

    denied_call = PreparedToolCall(
        model_name=call.model_name,
        arguments=call.arguments,
        request_id=call.request_id,
        runtime_denied_reason=reason,
    )
    try:
        request, result = _execute_one(executor, denied_call, execution_context)
    except BaseException:
        request = _fallback_request(executor, call)
        result = ToolResult(False, "", reason, request_id=request.request_id)
    data = dict(result.data or {})
    data.update({"synthetic": True, "interrupted": interrupted})
    return request, ToolResult(
        False,
        result.stdout,
        result.stderr or reason,
        data=data,
        duration_ms=result.duration_ms,
        request_id=request.request_id,
    )


def _fallback_request(executor: ModelToolExecutor, call: PreparedToolCall) -> ToolRequest:
    try:
        capability, _handler = executor.registry.resolve(call.model_name)
    except BaseException:
        capability = None
    args = dict(call.arguments) if isinstance(call.arguments, dict) else {}
    if capability is None:
        return ToolRequest(
            "unknown",
            call.model_name or "unknown",
            args,
            request_id=call.request_id or call.model_name or "unknown",
            model_name=call.model_name or None,
        )
    return ToolRequest(
        capability.tool,
        capability.action,
        args,
        request_id=call.request_id or call.model_name or capability.model_name,
        model_name=call.model_name,
    )


def _capture_execution_context(executor: ModelToolExecutor) -> object | None:
    capture = getattr(executor, "capture_model_call_context", None)
    return capture() if callable(capture) else None


def _exception_reason(exc: Exception) -> str:
    detail = str(exc).strip()
    suffix = f": {detail}" if detail else ""
    return f"Synthetic failed tool result: execution raised {type(exc).__name__}{suffix}"[:2_000]


def _interruption_reason(exc: BaseException) -> str:
    return (
        "Synthetic failed tool result: tool batch interrupted by "
        f"{type(exc).__name__}; this call did not complete and may not have started."
    )[:2_000]


def _is_concurrency_safe_read(executor: ModelToolExecutor, call: PreparedToolCall) -> bool:
    if call.runtime_denied_reason:
        return False
    try:
        capability, _handler = executor.registry.resolve(call.model_name)
    except Exception:
        return False
    if (
        capability is None
        or not capability.active
        or not capability.concurrency_safe
        or capability.requires_confirmation
    ):
        return False
    if capability.name in _STATEFUL_READ_CAPABILITIES:
        return False
    return frozenset(str(item) for item in capability.permissions) == _READ_ONLY_PERMISSIONS


def _bounded_workers(value: int) -> int:
    try:
        workers = int(value)
    except (TypeError, ValueError):
        workers = DEFAULT_MAX_PARALLEL_READS
    return max(1, min(workers, MAX_PARALLEL_READS_HARD_LIMIT))
