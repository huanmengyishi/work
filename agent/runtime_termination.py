from __future__ import annotations

from typing import Any, NoReturn

from .budget import ExecutionBudgetExceeded
from .convergence import repair_tool_message_pairs
from .deepseek import ChatResponse, DeepSeekStreamInterrupted
from .event_pipelines import SESSION_CHECKPOINT_REQUESTED, SESSION_FINALIZE_REQUESTED
from .events import EventDispatchError
from .model_router import ModelRoute
from .runtime_execution_setup import ExecutionLoopState, ExecutionSetup
from .state import AgentState


class RuntimeTerminationMixin:
    def _finish_successful_execution(
        self,
        state: AgentState,
        messages: list[dict[str, Any]],
        *,
        final: str,
        model_route: ModelRoute,
        append_message: bool = False,
    ) -> None:
        state.complete(final)
        if append_message:
            messages.append({"role": "assistant", "content": final})
        memory_refinement = self._maybe_refine_memory(
            state,
            messages,
            final=final,
            model_route=model_route,
        )
        self._finalize_session(state, messages)
        self._publish_terminal(
            "task.finished",
            state,
            final=final,
            memory_refinement=memory_refinement,
        )

    def _finish_failed_execution(
        self,
        state: AgentState,
        messages: list[dict[str, Any]],
        *,
        final: str,
        error: str,
    ) -> None:
        state.fail(error, final)
        messages.append({"role": "assistant", "content": final})
        self._finalize_session(state, messages)
        self._publish_terminal("task.failed", state, final=final, error=state.error)

    def _finish_after_tool_loop(
        self,
        state: AgentState,
        messages: list[dict[str, Any]],
        setup: ExecutionSetup,
        loop: ExecutionLoopState,
    ) -> str:
        synthesis = self._final_synthesis(
            state,
            messages,
            model_route=setup.model_route,
            strategy=setup.strategy,
            history_compactor=setup.history_compactor,
            context_window=setup.context_window,
            auto_compaction_enabled=setup.auto_compaction_enabled,
            auto_compaction_max_tokens=setup.auto_compaction_max_tokens,
        )
        completion_issue = self._completion_issue(state, synthesis)
        if synthesis and not completion_issue:
            self._finish_successful_execution(
                state,
                messages,
                final=synthesis,
                model_route=setup.model_route,
                append_message=True,
            )
            return synthesis

        rejected_finish_reason = str(state.convergence.get("final_synthesis_rejected_finish_reason") or "")
        rejected_protocol = str(state.convergence.get("final_synthesis_rejected_protocol") or "")
        if rejected_finish_reason:
            incomplete_reason = (
                "the final synthesis ended with an unusable "
                f"finish_reason={rejected_finish_reason}; no such response was accepted"
            )
        elif rejected_protocol:
            incomplete_reason = (
                "the tool-free final synthesis attempted tool use as "
                f"{rejected_protocol}; no tool was executed and no such response was accepted"
            )
        elif loop.loop_exit_reason == "soft_target":
            incomplete_reason = (
                "the soft tool-turn target was reached, but the completion gate still reports: " + completion_issue
            )
        else:
            incomplete_reason = "the hard tool-turn limit was reached before verified completion"
            if completion_issue:
                incomplete_reason += ": " + completion_issue
        final = self._incomplete_answer(state, incomplete_reason, substantive=synthesis)
        self._finish_failed_execution(
            state,
            messages,
            final=final,
            error=f"{loop.loop_exit_reason} reached: {completion_issue or incomplete_reason}",
        )
        return final

    def _finish_budget_exhaustion(
        self,
        state: AgentState,
        messages: list[dict[str, Any]],
        exc: ExecutionBudgetExceeded,
    ) -> str:
        repair = repair_tool_message_pairs(messages)
        if repair.changed:
            messages[:] = repair.messages
        final = self._incomplete_answer(state, f"execution budget exhausted: {exc.reason}")
        self._finish_failed_execution(
            state,
            messages,
            final=final,
            error=f"execution budget exhausted: {exc.reason}",
        )
        return final

    def _raise_execution_exception(
        self,
        state: AgentState,
        messages: list[dict[str, Any]],
        exc: Exception,
    ) -> NoReturn:
        state.convergence["last_error_category"] = self.resilience.classify(exc).value
        if isinstance(exc, EventDispatchError) and exc.event_name == SESSION_FINALIZE_REQUESTED:
            raise exc
        if isinstance(exc, EventDispatchError) and exc.event_name == SESSION_CHECKPOINT_REQUESTED:
            state.fail(str(exc))
            if exc.subscriber_succeeded("session.checkpoint-writer"):
                self._persist_failed_terminal(state, messages)
            raise exc
        if isinstance(exc, DeepSeekStreamInterrupted):
            state.record_model_response(ChatResponse(message={}, raw={}, http_attempt_count=exc.http_attempt_count))
            state.fail(f"resumable interruption: {exc}")
            if state.execution_context:
                state.execution_context.prompt_phase = "interrupted"
            self._persist_failed_terminal(state, messages)
            raise RuntimeError(f"{exc} Session: {state.session_id}") from exc
        http_attempt_count = getattr(exc, "http_attempt_count", 0)
        if isinstance(http_attempt_count, int) and not isinstance(http_attempt_count, bool) and http_attempt_count > 0:
            state.record_model_response(ChatResponse(message={}, raw={}, http_attempt_count=http_attempt_count))
        state.fail(str(exc))
        self._persist_failed_terminal(state, messages)
        raise exc


__all__ = ["RuntimeTerminationMixin"]
