from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .constants import MAX_TOOL_CALLS_PER_MODEL_RESPONSE
from .context import ContextBuildRequest, ContextSnapshot
from .convergence import ConvergenceAction, TaskConvergenceController, ToolHistoryCompactor
from .state import AgentState
from .tool_orchestration import PreparedToolCall, ToolBatchInterrupted, execute_model_tool_calls


@dataclass(frozen=True)
class ToolBatchOutcome:
    made_progress: bool
    recovery_chars_used: int


class RuntimeToolBatchMixin:
    def _execute_tool_batch(
        self,
        *,
        state: AgentState,
        messages: list[dict[str, Any]],
        snapshot: ContextSnapshot,
        message: dict[str, Any],
        tool_calls: list[dict[str, Any]],
        dropped_tool_call_count: int,
        max_tool_calls_per_round: int,
        convergence_action: ConvergenceAction,
        convergence: TaskConvergenceController,
        single_validation: bool,
        validation_consumed: bool,
        single_tool_result_chars: int,
        recovery_injected: set[int],
        recovery_char_limit: int,
        recovery_chars_used: int,
        round_compactor: ToolHistoryCompactor | None,
    ) -> ToolBatchOutcome:
        round_requests: list[dict[str, Any]] = []
        round_results: list[dict[str, Any]] = []
        tool_messages: list[dict[str, Any]] = []
        protocol_messages: list[dict[str, Any]] = []
        recovery_messages: list[dict[str, Any]] = []
        if dropped_tool_call_count:
            protocol_messages.append(
                {
                    "role": "system",
                    "content": (
                        f"Deep Agent dropped {dropped_tool_call_count} tool calls beyond the hard protocol "
                        f"limit of {MAX_TOOL_CALLS_PER_MODEL_RESPONSE}. They were not executed or written "
                        "to Agent "
                        "State. The retained calls and results remain one-to-one protocol pairs."
                    ),
                }
            )
        prepared_calls: list[PreparedToolCall] = []
        validation_request_ids: set[str] = set()
        for call_index, call in enumerate(tool_calls):
            function = call.get("function") or {}
            model_tool_name = str(function.get("name") or "")
            policy_tool_name = self.tools.model_function_name(model_tool_name)
            canonical_capability = self.tools.canonical_capability_name(model_tool_name)
            model_arguments = function.get("arguments") or "{}"
            validation_count = self._validation_model_call_count(policy_tool_name, model_arguments)
            validation_call = validation_count > 0
            runtime_denied_reason = None
            if call_index >= max_tool_calls_per_round:
                runtime_denied_reason = (
                    f"{model_tool_name or 'tool call'} was not executed because this assistant response "
                    f"contained {len(tool_calls)} tool calls, exceeding the configured per-round limit of "
                    f"{max_tool_calls_per_round}. The denied result remains paired with its tool_call_id."
                )
            elif convergence_action.force_plan_transition and policy_tool_name != "agent_update_step":
                runtime_denied_reason = (
                    f"{model_tool_name or 'tool call'} is unavailable until the current scope/inspection "
                    "step is completed and the next ready Task Graph step is started with agent_update_step."
                )
            elif convergence_action.guard_implementation_read and policy_tool_name == "read_file":
                implementation_read_denial = convergence.implementation_read_denial(
                    state,
                    policy_tool_name,
                    model_arguments,
                )
                if implementation_read_denial:
                    runtime_denied_reason = (
                        f"{model_tool_name} cannot use the bounded implementation evidence exception: "
                        f"{implementation_read_denial}. Use the evidence already collected, a managed edit, "
                        "verification, or the final answer."
                    )
            elif convergence_action.guard_validation_attachment_read and policy_tool_name == "tool_result_read":
                attachment_read_denial = convergence.validation_attachment_read_denial(
                    state,
                    policy_tool_name,
                    model_arguments,
                )
                if attachment_read_denial:
                    runtime_denied_reason = (
                        f"{model_tool_name} cannot use the bounded validation attachment exception: "
                        f"{attachment_read_denial}. Use the validated evidence already available, finish "
                        "implementation/verification, or provide the final answer."
                    )
            elif single_validation and validation_count > 1:
                runtime_denied_reason = (
                    f"{model_tool_name or 'validation call'} was not executed because it contains "
                    f"{validation_count} validation commands while the user allowed only one validation "
                    "attempt. Submit exactly one bounded validation command."
                )
            elif single_validation and validation_consumed and validation_call:
                runtime_denied_reason = (
                    f"{model_tool_name or 'validation call'} was not executed because the user requested "
                    "a single validation attempt and that attempt is already recorded. Report its exact "
                    "outcome as the validation limit; do not substitute an equivalent shell, LSP, or test "
                    "command."
                )
            elif policy_tool_name in convergence_action.excluded_functions:
                runtime_denied_reason = (
                    f"{model_tool_name} is unavailable in this task phase: {convergence_action.reason}. "
                    "Use the evidence already collected, advance the Task Graph, then implement, validate, "
                    "or provide the substantive final answer."
                )
            elif convergence_action.block_exploration_bypass and convergence.is_exploration_bypass(
                policy_tool_name, model_arguments
            ):
                runtime_denied_reason = (
                    f"{model_tool_name} cannot be used for file exploration in this task phase: "
                    f"{convergence_action.reason}. Use existing evidence and the managed implementation or "
                    "verification tools instead of bypassing the exploration threshold."
                )
            if runtime_denied_reason is None:
                recovery_decision = self.capability_recovery.before_call(
                    state.convergence,
                    canonical_capability,
                    current_round=state.round,
                )
                if not recovery_decision.allowed:
                    runtime_denied_reason = recovery_decision.reason
            prepared_calls.append(
                PreparedToolCall(
                    model_name=model_tool_name,
                    arguments=model_arguments,
                    request_id=str(call.get("id") or "") or None,
                    runtime_denied_reason=runtime_denied_reason,
                )
            )
            if validation_call:
                validation_request_ids.add(str(call.get("id") or ""))

        def single_validation_policy(
            prepared: PreparedToolCall,
            prior_executions: tuple[tuple[Any, Any], ...],
        ) -> PreparedToolCall:
            if prepared.request_id not in validation_request_ids or prepared.runtime_denied_reason is not None:
                return prepared
            consumed_in_batch = any(
                request.request_id in validation_request_ids and not bool((result.data or {}).get("not_executed"))
                for request, result in prior_executions
            )
            if not validation_consumed and not consumed_in_batch:
                return prepared
            return PreparedToolCall(
                model_name=prepared.model_name,
                arguments=prepared.arguments,
                request_id=prepared.request_id,
                runtime_denied_reason=(
                    f"{prepared.model_name or 'validation call'} was not executed because the user "
                    "requested a single validation attempt and that attempt is already recorded. Report "
                    "its exact outcome as the validation limit; do not substitute an equivalent shell, "
                    "LSP, or test command."
                ),
            )

        tool_interruption: BaseException | None = None
        try:
            self.execution_budget.before_tool_batch(state)
            executions = execute_model_tool_calls(
                self.tools,
                prepared_calls,
                max_concurrency=self._bounded_config_int(
                    "runtime.convergence.max_parallel_read_tools",
                    4,
                    minimum=1,
                    maximum=16,
                ),
                sequential_policy=single_validation_policy if single_validation else None,
            )
        except ToolBatchInterrupted as exc:
            executions = exc.executions
            tool_interruption = exc.cause
        for call, (request, result) in zip(tool_calls, executions, strict=True):
            self._progress(
                "tool.finished",
                state,
                tool=request.capability,
                success=result.success,
                duration_ms=result.duration_ms,
            )
            state.record_tool_call(request.to_dict(), result.to_dict())
            recovery_decision = self.capability_recovery.observe(
                state.convergence,
                request.capability,
                current_round=state.round,
                success=result.success,
                health_failure=self.tools.result_is_health_failure(result),
                health_status=self.tools.capability_health_status(request.capability),
            )
            if not result.success and recovery_decision.action in {"backoff", "skip_broken"}:
                recovery_messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Capability recovery decision: "
                            + recovery_decision.reason
                            + ". Do not immediately repeat the same failing call."
                        ),
                    }
                )
            round_requests.append(request.to_dict())
            round_results.append(result.to_dict())
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": result.as_text(limit=single_tool_result_chars),
                }
            )
            runtime_denied = bool((result.data or {}).get("runtime_denied"))
            if tool_interruption is None and not result.success and not runtime_denied:
                recovery = self.memory.search_recovery(
                    "\n".join(part for part in (result.stderr, result.stdout) if part),
                    self.project.id,
                )
                unseen = [item for item in recovery if item.id not in recovery_injected]
                remaining_recovery_chars = recovery_char_limit - recovery_chars_used
                if unseen and remaining_recovery_chars >= 256:
                    recovery_package = self.context_builder.build_package(
                        ContextBuildRequest(
                            snapshot=snapshot,
                            state=state,
                            memory_items=unseen,
                            recovery_context=(
                                "The last tool call failed. Diagnose it before retrying and do not repeat an "
                                "already documented failed approach."
                            ),
                            phase="recovery",
                            max_chars=remaining_recovery_chars,
                        )
                    )
                    if any(section.key == "recovery" for section in recovery_package.sections):
                        recovery_chars_used += recovery_package.used_chars
                        recovery_injected.update(recovery_package.included_memory_ids)
                        self._record_included_memories(state, recovery_package.included_memory_ids)
                        recovery_messages.append({"role": "system", "content": recovery_package.rendered})
        # DeepSeek requires all tool results for one assistant response
        # to remain contiguous. Recovery guidance is appended only
        # after the complete result batch.
        if tool_interruption is None:
            tool_messages = self._compact_tool_batch(
                state,
                message,
                tool_messages,
                round_compactor,
            )
        messages.extend(tool_messages)
        messages.extend(protocol_messages)
        messages.extend(recovery_messages)
        if tool_interruption is not None:
            interruption_name = type(tool_interruption).__name__
            state.fail(f"resumable tool interruption: {interruption_name}")
            if state.execution_context:
                state.execution_context.prompt_phase = "interrupted"
            self._checkpoint_session(state, messages)
            raise tool_interruption
        made_progress = convergence.observe_round(state, round_requests, round_results)
        if convergence.enabled or self.config.get("runtime.checkpoint_each_tool", True):
            self._checkpoint_session(state, messages)

        return ToolBatchOutcome(
            made_progress=made_progress,
            recovery_chars_used=recovery_chars_used,
        )


__all__ = ["RuntimeToolBatchMixin", "ToolBatchOutcome"]
