from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .constants import MAX_TOOL_CALLS_PER_MODEL_RESPONSE, SINGLE_VALIDATION_MODEL_FUNCTIONS
from .convergence import ConvergenceAction
from .deepseek import ChatResponse
from .runtime_execution_setup import ExecutionLoopState, ExecutionSetup
from .runtime_support import (
    _finish_reason_label,
    _has_usable_finish_reason,
    _normalize_assistant_tool_calls,
    _tool_protocol_text_violation,
    _tool_protocol_violation,
)
from .state import AgentState


@dataclass(frozen=True)
class ModelRound:
    round_number: int
    convergence_action: ConvergenceAction
    single_validation: bool
    validation_consumed: bool
    response: ChatResponse


@dataclass(frozen=True)
class ModelResponseOutcome:
    action: Literal["retry", "tool_calls", "terminal"]
    final: str = ""
    message: dict[str, Any] | None = None
    tool_calls: tuple[dict[str, Any], ...] = ()
    dropped_tool_call_count: int = 0


class RuntimeResponseMixin:
    def _request_model_round(
        self,
        state: AgentState,
        messages: list[dict[str, Any]],
        setup: ExecutionSetup,
        loop: ExecutionLoopState,
    ) -> ModelRound:
        loop.model_round += 1
        round_number = loop.model_round
        state.round = loop.tool_turn + 1
        state.touch()
        convergence_action = setup.convergence.before_round(
            min(loop.tool_turn + 1, setup.soft_tool_turn_target),
            state,
        )
        for notice in convergence_action.messages:
            messages.append({"role": "system", "content": notice})
        if convergence_action.messages:
            self._checkpoint_session(state, messages)

        active_tools = setup.convergence.filter_schemas(
            self.tools.schemas(),
            convergence_action.excluded_functions,
        )
        single_validation = self._single_validation_requested(state)
        validation_consumed = single_validation and self._single_validation_used(state)
        if validation_consumed:
            active_tools = [
                item
                for item in active_tools
                if str((item.get("function") or {}).get("name") or "") not in SINGLE_VALIDATION_MODEL_FUNCTIONS
            ]
        if convergence_action.force_plan_transition:
            active_tools = [
                item
                for item in active_tools
                if str((item.get("function") or {}).get("name") or "") == "agent_update_step"
            ]

        self._prepare_model_request(
            state,
            messages,
            tools=active_tools,
            model_route=setup.model_route,
            context_window=setup.context_window,
            history_compactor=setup.history_compactor,
            auto_compaction_enabled=setup.auto_compaction_enabled,
            auto_compaction_max_tokens=setup.auto_compaction_max_tokens,
            phase="tool_loop",
            checkpoint=True,
        )
        self._reserve_model_request(
            state,
            messages,
            phase="main_loop",
            tools=active_tools,
            max_tokens=setup.context_window.effective_output_tokens(setup.model_route.max_tokens),
        )
        self.events.publish(
            "model.requested",
            {
                "run_id": state.run_id,
                "round": round_number,
                "message_count": len(messages),
                "model_tier": setup.model_route.tier,
                "model": setup.model_route.model,
            },
            project_id=self.project.id,
            session_id=state.session_id,
            run_id=state.run_id,
        )
        self._progress(
            "model.requested",
            state,
            round=loop.tool_turn + 1,
            max_rounds=setup.soft_tool_turn_target,
            hard_limit=setup.hard_tool_turn_limit,
            current_step=state.current_step,
        )
        chat_kwargs = {
            "messages": messages,
            "tools": active_tools,
            "tool_choice": "auto",
            "thinking": setup.strategy.thinking_enabled,
            "reasoning_effort": setup.strategy.reasoning_effort,
            "max_tokens": setup.context_window.effective_output_tokens(setup.model_route.max_tokens),
            "model": setup.model_route.model,
        }
        response = self._chat_with_recovery(
            state,
            messages,
            active_tools,
            chat_kwargs,
            strategy=setup.strategy,
            model_route=setup.model_route,
            context_window=setup.context_window,
            history_compactor=setup.history_compactor,
            auto_compaction_max_tokens=setup.auto_compaction_max_tokens,
            round_number=round_number,
            request_phase="main_loop",
        )
        response = self._complete_length_response(
            state,
            messages,
            response,
            chat_kwargs,
            strategy=setup.strategy,
            round_number=round_number,
            request_phase="main_loop",
        )
        if response.finish_reason != "length":
            state.record_model_response(response)
        return ModelRound(
            round_number=round_number,
            convergence_action=convergence_action,
            single_validation=single_validation,
            validation_consumed=validation_consumed,
            response=response,
        )

    def _process_model_response(
        self,
        state: AgentState,
        messages: list[dict[str, Any]],
        setup: ExecutionSetup,
        loop: ExecutionLoopState,
        model_round: ModelRound,
    ) -> ModelResponseOutcome:
        if not _has_usable_finish_reason(model_round.response.finish_reason):
            return self._reject_unusable_finish(state, messages, setup, loop, model_round)
        return self._accept_model_response(state, messages, setup, loop, model_round)

    def _reject_unusable_finish(
        self,
        state: AgentState,
        messages: list[dict[str, Any]],
        setup: ExecutionSetup,
        loop: ExecutionLoopState,
        model_round: ModelRound,
    ) -> ModelResponseOutcome:
        finish_reason = _finish_reason_label(model_round.response.finish_reason) or "missing"
        raw_tool_calls = model_round.response.message.get("tool_calls")
        discarded_tool_calls = len(raw_tool_calls) if isinstance(raw_tool_calls, list) else int(bool(raw_tool_calls))
        messages.append(
            {
                "role": "assistant",
                "content": (
                    "[Deep Agent rejected an unusable model response with "
                    f"finish_reason={finish_reason}; {discarded_tool_calls} tool call(s) were not executed]"
                ),
            }
        )
        self._publish_model_response(
            state,
            round_number=model_round.round_number,
            tool_call_count=0,
            discarded_tool_call_count=discarded_tool_calls,
            finish_reason=finish_reason,
        )
        if loop.abnormal_finish_recoveries < setup.max_abnormal_finish_recoveries:
            loop.abnormal_finish_recoveries += 1
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "The previous model response ended with an unusable finish reason. None of its "
                        "tool calls were executed. Return one complete protocol-valid response now. "
                        "Use finish_reason=tool_calls only with complete tool calls, or finish_reason=stop "
                        "with a substantive final answer."
                    ),
                }
            )
            self._checkpoint_convergence_transition(
                state,
                messages,
                transition="abnormal_finish_retry",
                phase="main_loop",
                counter="abnormal_finish_recovery_count",
            )
            return ModelResponseOutcome("retry")

        self._checkpoint_convergence_transition(
            state,
            messages,
            transition="abnormal_finish_failed",
            phase="main_loop",
        )
        final = self._incomplete_answer(
            state,
            f"DeepSeek repeatedly returned an unusable finish_reason={finish_reason}",
        )
        self._finish_failed_execution(
            state,
            messages,
            final=final,
            error=f"unusable finish_reason: {finish_reason}",
        )
        return ModelResponseOutcome("terminal", final=final)

    def _accept_model_response(
        self,
        state: AgentState,
        messages: list[dict[str, Any]],
        setup: ExecutionSetup,
        loop: ExecutionLoopState,
        model_round: ModelRound,
    ) -> ModelResponseOutcome:
        message, normalized_count, dropped_count = _normalize_assistant_tool_calls(
            model_round.response.message,
            turn=state.turn,
            round_number=model_round.round_number,
        )
        protocol_text_violation = _tool_protocol_text_violation(message)
        protocol_discarded_tool_calls = 0
        if protocol_text_violation and message.get("tool_calls"):
            protocol_discarded_tool_calls = len(message.get("tool_calls") or [])
            message = dict(message)
            message.pop("tool_calls", None)
            message["content"] = (
                f"[Deep Agent rejected {protocol_text_violation} accompanying structured tool calls; "
                "all calls were discarded and no tool was executed]"
            )
        messages.append(message)
        tool_calls = message.get("tool_calls") or []
        self._report_normalized_response(
            state,
            setup,
            model_round.round_number,
            message,
            tool_calls,
            normalized_count,
            dropped_count,
            protocol_discarded_tool_calls,
        )
        if tool_calls:
            return ModelResponseOutcome(
                "tool_calls",
                message=message,
                tool_calls=tuple(tool_calls),
                dropped_tool_call_count=dropped_count,
            )
        return self._process_answer_response(
            state,
            messages,
            setup,
            loop,
            message,
            protocol_text_violation=protocol_text_violation,
            protocol_discarded_tool_calls=protocol_discarded_tool_calls,
        )

    def _report_normalized_response(
        self,
        state: AgentState,
        setup: ExecutionSetup,
        round_number: int,
        message: dict[str, Any],
        tool_calls: list[dict[str, Any]],
        normalized_count: int,
        dropped_count: int,
        protocol_discarded_tool_calls: int,
    ) -> None:
        if normalized_count:
            self._progress(
                "protocol.tool_calls_normalized",
                state,
                round=round_number,
                normalized_count=normalized_count,
            )
        if dropped_count:
            self._progress(
                "protocol.tool_calls_dropped",
                state,
                round=round_number,
                retained_count=len(tool_calls),
                dropped_count=dropped_count,
                hard_limit=MAX_TOOL_CALLS_PER_MODEL_RESPONSE,
            )
        reasoning = str(message.get("reasoning_content") or "").strip()
        if reasoning and not (setup.strategy.thinking_enabled and hasattr(self.client, "chat_stream")):
            self._progress(
                "thinking.content",
                state,
                round=round_number,
                content=reasoning[: int(self.config.get("runtime.max_reasoning_display_chars", 4000))],
            )
        self._publish_model_response(
            state,
            round_number=round_number,
            tool_call_count=len(tool_calls),
            dropped_tool_call_count=dropped_count,
            discarded_protocol_tool_call_count=protocol_discarded_tool_calls,
        )

    def _publish_model_response(
        self,
        state: AgentState,
        *,
        round_number: int,
        tool_call_count: int,
        **details: Any,
    ) -> None:
        payload = {
            "run_id": state.run_id,
            "round": round_number,
            "tool_call_count": tool_call_count,
            **details,
        }
        self.events.publish(
            "model.responded",
            payload,
            project_id=self.project.id,
            session_id=state.session_id,
            run_id=state.run_id,
        )
        self._progress("model.responded", state, round=round_number, tool_call_count=tool_call_count, **details)

    def _process_answer_response(
        self,
        state: AgentState,
        messages: list[dict[str, Any]],
        setup: ExecutionSetup,
        loop: ExecutionLoopState,
        message: dict[str, Any],
        *,
        protocol_text_violation: str,
        protocol_discarded_tool_calls: int,
    ) -> ModelResponseOutcome:
        final = str(message.get("content") or "").strip()
        protocol_violation = protocol_text_violation or _tool_protocol_violation(message)
        if protocol_violation:
            if protocol_discarded_tool_calls:
                message["content"] = (
                    f"[Deep Agent rejected {protocol_violation} accompanying structured tool calls; "
                    "all calls were discarded and no tool was executed]"
                )
            else:
                message["content"] = (
                    f"[Deep Agent rejected {protocol_violation} returned as answer text; no tool was executed]"
                )
            final = ""
            completion_issue = f"the model returned unusable {protocol_violation}; no tool call was accepted"
        else:
            completion_issue = self._completion_issue(state, final)

        if completion_issue and loop.corrective_rounds < setup.max_corrective_rounds:
            loop.corrective_rounds += 1
            protocol_guidance = (
                " Return any tool request only through the registered structured tool interface; never "
                "print DSML or other tool-call markup as answer text."
                if protocol_violation
                else ""
            )
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "The task is not complete yet: "
                        + completion_issue
                        + " Continue executing the missing work with the registered tools. "
                        "Do not return a progress note as the final answer." + protocol_guidance
                    ),
                }
            )
            self._checkpoint_session(state, messages)
            return ModelResponseOutcome("retry")
        if completion_issue:
            final = self._incomplete_answer(state, completion_issue, substantive=final)
            self._finish_failed_execution(
                state,
                messages,
                final=final,
                error=f"completion gate: {completion_issue}",
            )
            return ModelResponseOutcome("terminal", final=final)

        self._finish_successful_execution(
            state,
            messages,
            final=final,
            model_route=setup.model_route,
        )
        return ModelResponseOutcome("terminal", final=final)


__all__ = [
    "ModelResponseOutcome",
    "ModelRound",
    "RuntimeResponseMixin",
]
