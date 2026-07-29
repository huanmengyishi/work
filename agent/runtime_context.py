from __future__ import annotations

from typing import Any

from .convergence import (
    ContextWindowController,
    ToolHistoryCompactor,
    repair_tool_message_pairs,
)
from .deepseek import ChatResponse, DeepSeekContextOverflow
from .history_snip import HistorySnipper
from .model_router import ModelRoute
from .state import AgentState
from .task_strategy import TaskStrategy


class RuntimeContextMixin:
    def _snip_history(
        self,
        state: AgentState,
        messages: list[dict[str, Any]],
        *,
        phase: str,
    ) -> bool:
        """Drop expendable complete rounds before any model-backed compaction."""

        if not bool(self.config.get("runtime.convergence.enabled", True)):
            return False
        if self.config.get("runtime.convergence.history_snip_enabled", True) is False:
            return False
        snipper = HistorySnipper(
            keep_recent_rounds=self._bounded_config_int(
                "runtime.convergence.history_snip_keep_recent_rounds",
                4,
                minimum=0,
                maximum=100,
            ),
            min_history_chars=self._bounded_config_int(
                "runtime.convergence.history_snip_min_chars",
                24_000,
                minimum=0,
                maximum=100_000_000,
            ),
            min_complete_rounds=self._bounded_config_int(
                "runtime.convergence.history_snip_min_complete_rounds",
                8,
                minimum=0,
                maximum=10_000,
            ),
            marker_chars=self._bounded_config_int(
                "runtime.convergence.history_snip_marker_chars",
                768,
                minimum=128,
                maximum=4_096,
            ),
        )
        result = snipper.snip(
            messages,
            objective=f"{state.user_request}\n{state.objective}",
        )
        if not result.changed:
            return False

        messages[:] = result.messages
        metadata = state.convergence if isinstance(state.convergence, dict) else {}
        state.convergence = metadata
        prior_count = metadata.get("history_snip_count", 0)
        if not isinstance(prior_count, int) or isinstance(prior_count, bool):
            prior_count = 0

        def bounded_metric(value: int) -> int:
            return max(0, min(int(value), 1_000_000_000))

        metadata.update(
            {
                "history_snip_count": min(10_000, prior_count + 1),
                "history_snip_removed_rounds": bounded_metric(result.removed_rounds),
                "history_snip_removed_messages": bounded_metric(result.removed_messages),
                "history_snip_original_chars": bounded_metric(result.original_chars),
                "history_snip_final_chars": bounded_metric(result.final_chars),
                "history_snip_phase": str(phase)[:32],
            }
        )
        state.touch()
        self._progress(
            "history.snipped",
            state,
            total_rounds=bounded_metric(result.total_rounds),
            kept_rounds=bounded_metric(result.kept_rounds),
            removed_rounds=bounded_metric(result.removed_rounds),
            removed_messages=bounded_metric(result.removed_messages),
            original_chars=bounded_metric(result.original_chars),
            final_chars=bounded_metric(result.final_chars),
            coalesced_markers=bounded_metric(result.coalesced_markers),
            phase=str(phase)[:32],
        )
        return True

    def _compact_tool_history(
        self,
        state: AgentState,
        messages: list[dict[str, Any]],
        compactor: ToolHistoryCompactor | None,
        *,
        phase: str,
        checkpoint: bool,
    ) -> bool:
        if compactor is None:
            return False
        history_result = compactor.compact(messages)
        if history_result.messages is not messages:
            messages[:] = history_result.messages
        if history_result.changed:
            self._progress(
                "history.compacted",
                state,
                original_chars=history_result.original_chars,
                final_chars=history_result.final_chars,
                compacted_count=history_result.compacted_count,
                phase=phase,
            )
            if checkpoint:
                self._checkpoint_session(state, messages)
        if history_result.error:
            self._progress(
                "history.compaction_failed",
                state,
                failure_count=history_result.failure_count,
                circuit_open=history_result.circuit_open,
                phase=phase,
            )
        return history_result.changed

    def _compact_tool_batch(
        self,
        state: AgentState,
        assistant_message: dict[str, Any],
        tool_messages: list[dict[str, Any]],
        compactor: ToolHistoryCompactor | None,
    ) -> list[dict[str, Any]]:
        if compactor is None or not tool_messages:
            return tool_messages
        result = compactor.compact([assistant_message, *tool_messages])
        compacted_tools = [item for item in result.messages if item.get("role") == "tool"]
        if result.changed:
            self._progress(
                "history.compacted",
                state,
                original_chars=result.original_chars,
                final_chars=result.final_chars,
                compacted_count=result.compacted_count,
                phase="same_api_round",
            )
        if result.error:
            self._progress(
                "history.compaction_failed",
                state,
                failure_count=result.failure_count,
                circuit_open=result.circuit_open,
                phase="same_api_round",
            )
        return compacted_tools

    def _prepare_model_request(
        self,
        state: AgentState,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        model_route: ModelRoute,
        context_window: ContextWindowController,
        history_compactor: ToolHistoryCompactor | None,
        auto_compaction_enabled: bool,
        auto_compaction_max_tokens: int,
        phase: str,
        checkpoint: bool,
    ) -> None:
        changed = False
        repair = repair_tool_message_pairs(messages)
        if repair.changed:
            messages[:] = repair.messages
            changed = True
            self._progress(
                "history.pairs_repaired",
                state,
                repaired_count=repair.repaired_count,
                phase=phase,
            )
        changed = self._snip_history(state, messages, phase=phase) or changed
        if auto_compaction_enabled:
            reasoning_count = context_window.compact_old_reasoning(messages)
            if reasoning_count:
                changed = True
                self._progress(
                    "history.reasoning_compacted",
                    state,
                    compacted_count=reasoning_count,
                    phase=phase,
                )
        changed = (
            self._compact_tool_history(
                state,
                messages,
                history_compactor,
                phase=phase,
                checkpoint=False,
            )
            or changed
        )

        budget = context_window.budget(messages, tools, max_output_tokens=model_route.max_tokens)
        if auto_compaction_enabled and budget.over_trigger:
            compacted = False
            if not context_window.circuit_open:
                compacted = self._auto_compact_context(
                    state,
                    messages,
                    tools=tools,
                    model_route=model_route,
                    context_window=context_window,
                    auto_compaction_max_tokens=auto_compaction_max_tokens,
                    phase=phase,
                )
                changed = compacted or changed
            budget = context_window.budget(messages, tools, max_output_tokens=model_route.max_tokens)

        if budget.over_limit or (context_window.circuit_open and budget.over_trigger):
            collapsed = self._emergency_context_collapse(
                state,
                messages,
                tools=tools,
                model_route=model_route,
                context_window=context_window,
                phase=phase,
            )
            changed = collapsed or changed
            budget = context_window.budget(messages, tools, max_output_tokens=model_route.max_tokens)

        if budget.over_limit:
            raise RuntimeError(
                "model request exceeds the configured context window after bounded compaction; "
                f"estimated={budget.estimated_tokens} input_limit={budget.input_limit_tokens}"
            )
        if changed and checkpoint:
            self._checkpoint_session(state, messages)

    def _chat_with_recovery(
        self,
        state: AgentState,
        messages: list[dict[str, Any]],
        active_tools: list[dict[str, Any]] | None,
        chat_kwargs: dict[str, Any],
        *,
        strategy: TaskStrategy,
        model_route: ModelRoute,
        context_window: ContextWindowController,
        history_compactor: ToolHistoryCompactor | None,
        auto_compaction_max_tokens: int,
        round_number: int,
        request_phase: str = "main_loop",
    ) -> ChatResponse:
        """Run one logical model request with two bounded overflow recovery stages."""

        overflow_stage = 0
        previous_tokens = context_window.budget(
            messages,
            active_tools,
            max_output_tokens=model_route.max_tokens,
        ).estimated_tokens
        while True:
            try:
                if strategy.thinking_enabled and hasattr(self.client, "chat_stream"):
                    return self.client.chat_stream(
                        **chat_kwargs,
                        on_reasoning=lambda chunk: self._progress(
                            "thinking.delta",
                            state,
                            round=round_number,
                            content=chunk,
                        ),
                        on_content=None,
                    )
                return self.client.chat(**chat_kwargs)
            except DeepSeekContextOverflow as exc:
                state.record_model_response(ChatResponse(message={}, raw={}, http_attempt_count=exc.http_attempt_count))
                overflow_stage += 1
                if overflow_stage == 1:
                    recovered = self._overflow_cheap_collapse(
                        state,
                        messages,
                        tools=active_tools,
                        model_route=model_route,
                        context_window=context_window,
                    )
                    transition = "cheap_collapse"
                elif overflow_stage == 2:
                    recovered = self._overflow_semantic_compact(
                        state,
                        messages,
                        tools=active_tools,
                        model_route=model_route,
                        context_window=context_window,
                        history_compactor=history_compactor,
                        auto_compaction_max_tokens=auto_compaction_max_tokens,
                    )
                    transition = "semantic_compact"
                else:
                    raise RuntimeError("DeepSeek context overflow remained after two bounded recovery stages") from None
                if not recovered:
                    raise RuntimeError(f"DeepSeek context overflow recovery failed during {transition}") from None
                current_tokens = context_window.budget(
                    messages,
                    active_tools,
                    max_output_tokens=model_route.max_tokens,
                ).estimated_tokens
                if current_tokens >= previous_tokens:
                    raise RuntimeError(
                        f"DeepSeek context overflow recovery {transition} did not reduce the request"
                    ) from None
                previous_tokens = current_tokens
                self._progress(
                    "context.overflow_recovered",
                    state,
                    stage=transition,
                    estimated_tokens=current_tokens,
                    phase=request_phase,
                )
                chat_kwargs["messages"] = messages
                self._reserve_model_request(
                    state,
                    messages,
                    phase=request_phase,
                    tools=active_tools,
                    max_tokens=chat_kwargs.get("max_tokens"),
                    checkpoint=False,
                )
                self._checkpoint_convergence_transition(
                    state,
                    messages,
                    transition=f"overflow_{transition}",
                    phase=request_phase,
                    counter="overflow_recovery_count",
                )

    def _overflow_cheap_collapse(
        self,
        state: AgentState,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        model_route: ModelRoute,
        context_window: ContextWindowController,
    ) -> bool:
        if self._compact_tool_history(
            state,
            messages,
            ToolHistoryCompactor(
                aggregate_chars=4_096,
                output_reserve_chars=3_072,
                compacted_result_chars=256,
                keep_recent_results=1,
                failure_limit=1,
            ),
            phase="overflow_cheap_collapse",
            checkpoint=False,
        ):
            return True
        return self._emergency_context_collapse(
            state,
            messages,
            tools=tools,
            model_route=model_route,
            context_window=context_window,
            phase="overflow_cheap_collapse",
        )

    def _overflow_semantic_compact(
        self,
        state: AgentState,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        model_route: ModelRoute,
        context_window: ContextWindowController,
        history_compactor: ToolHistoryCompactor | None,
        auto_compaction_max_tokens: int,
    ) -> bool:
        if not context_window.circuit_open:
            if self._auto_compact_context(
                state,
                messages,
                tools=tools,
                model_route=model_route,
                context_window=context_window,
                auto_compaction_max_tokens=auto_compaction_max_tokens,
                phase="overflow_semantic_compact",
            ):
                return True
        return self._emergency_context_collapse(
            state,
            messages,
            tools=tools,
            model_route=model_route,
            context_window=context_window,
            phase="overflow_semantic_fallback",
        )


__all__ = ["RuntimeContextMixin"]
