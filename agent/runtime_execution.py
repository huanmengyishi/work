from __future__ import annotations

from typing import Any, cast

from .budget import ExecutionBudgetExceeded
from .context import ContextSnapshot
from .runtime_execution_setup import (
    ExecutionLoopState,
    ExecutionSetup,
    RuntimeExecutionSetupMixin,
)
from .runtime_response import RuntimeResponseMixin
from .runtime_termination import RuntimeTerminationMixin
from .state import AgentState


class RuntimeExecutionMixin(
    RuntimeExecutionSetupMixin,
    RuntimeResponseMixin,
    RuntimeTerminationMixin,
):
    """Coordinate one turn while specialized mixins own each execution phase."""

    def _execute(
        self,
        state: AgentState,
        messages: list[dict[str, Any]],
        *,
        snapshot: ContextSnapshot,
    ) -> str:
        setup, loop = self._initialize_execution(state, messages, snapshot=snapshot)
        try:
            while loop.tool_turn < setup.hard_tool_turn_limit:
                model_round = self._request_model_round(state, messages, setup, loop)
                outcome = self._process_model_response(state, messages, setup, loop, model_round)
                if outcome.action == "retry":
                    continue
                if outcome.action == "terminal":
                    return outcome.final

                message = cast(dict[str, Any], outcome.message)
                batch_outcome = self._execute_tool_batch(
                    state=state,
                    messages=messages,
                    snapshot=setup.snapshot,
                    message=message,
                    tool_calls=list(outcome.tool_calls),
                    dropped_tool_call_count=outcome.dropped_tool_call_count,
                    max_tool_calls_per_round=setup.max_tool_calls_per_round,
                    convergence_action=model_round.convergence_action,
                    convergence=setup.convergence,
                    single_validation=model_round.single_validation,
                    validation_consumed=model_round.validation_consumed,
                    single_tool_result_chars=setup.single_tool_result_chars,
                    recovery_injected=loop.recovery_injected,
                    recovery_char_limit=setup.recovery_char_limit,
                    recovery_chars_used=loop.recovery_chars_used,
                    round_compactor=setup.round_compactor,
                )
                loop.recovery_chars_used = batch_outcome.recovery_chars_used
                if batch_outcome.made_progress:
                    loop.corrective_rounds = 0
                if self._advance_tool_loop(state, messages, setup, loop):
                    break
            return self._finish_after_tool_loop(state, messages, setup, loop)
        except ExecutionBudgetExceeded as exc:
            return self._finish_budget_exhaustion(state, messages, exc)
        except Exception as exc:
            self._raise_execution_exception(state, messages, exc)

    def _advance_tool_loop(
        self,
        state: AgentState,
        messages: list[dict[str, Any]],
        setup: ExecutionSetup,
        loop: ExecutionLoopState,
    ) -> bool:
        loop.tool_turn += 1
        execution_evidence_issue = self._execution_evidence_issue(state)
        if loop.tool_turn < setup.soft_tool_turn_target:
            return False
        if not execution_evidence_issue:
            state.convergence.pop("soft_target_evidence_issue", None)
            loop.loop_exit_reason = "soft_target"
            return True
        previous_issue = str(state.convergence.get("soft_target_evidence_issue") or "")
        if previous_issue != execution_evidence_issue:
            state.convergence["soft_target_evidence_issue"] = execution_evidence_issue
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "The soft tool-turn target was reached, but tool execution remains open because "
                        "required evidence is still missing: "
                        + execution_evidence_issue
                        + ". Use the remaining hard-limit budget only for these missing requirements."
                    ),
                }
            )
            self._checkpoint_session(state, messages)
        return False


__all__ = ["RuntimeExecutionMixin"]
