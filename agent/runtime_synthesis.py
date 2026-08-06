from __future__ import annotations

from typing import Any

from .convergence import (
    ContextWindowController,
    ToolHistoryCompactor,
)
from .deepseek import ChatResponse
from .model_router import ModelRoute
from .state import AgentState
from .task_strategy import TaskStrategy
from .runtime_support import (
    _finish_reason_label,
    _has_usable_finish_reason,
    _tool_protocol_violation,
)


class RuntimeSynthesisMixin:
    def _complete_length_response(
        self,
        state: AgentState,
        messages: list[dict[str, Any]],
        response: ChatResponse,
        chat_kwargs: dict[str, Any],
        *,
        strategy: TaskStrategy,
        round_number: int,
        request_phase: str,
    ) -> ChatResponse:
        if response.finish_reason != "length" or not isinstance(response.message, dict):
            return response
        partial = str(response.message.get("content") or "")
        # Tool-call JSON may be incomplete. Never write or execute it.
        continuation_messages = [
            *messages,
            {
                "role": "assistant",
                "content": partial or "[Deep Agent discarded incomplete tool calls from a length-truncated response]",
            },
            {
                "role": "system",
                "content": (
                    "The previous response hit the output limit. Continue the answer directly from the exact cutoff. "
                    "Do not repeat prior text. Do not call tools in this recovery response."
                ),
            },
        ]
        combined = partial
        total_http_attempts = max(0, int(response.http_attempt_count or 0))
        total_usage = self._merge_usage({}, response.usage)
        max_continuations = self._bounded_config_int(
            "runtime.convergence.max_length_continuations",
            2,
            minimum=1,
            maximum=4,
        )
        last = response
        for attempt in range(1, max_continuations + 1):
            self._reserve_model_request(
                state,
                continuation_messages,
                phase=request_phase,
                tools=None,
                max_tokens=chat_kwargs.get("max_tokens"),
                checkpoint=False,
            )
            self._checkpoint_convergence_transition(
                state,
                continuation_messages,
                transition="length_continuation",
                phase=request_phase,
                counter="length_continuation_count",
            )
            self._progress(
                "model.length_continuation_requested",
                state,
                round=round_number,
                attempt=attempt,
                max_attempts=max_continuations,
                phase=request_phase,
            )
            try:
                last = self.client.chat(
                    messages=continuation_messages,
                    tools=None,
                    tool_choice=None,
                    thinking=strategy.thinking_enabled,
                    reasoning_effort=strategy.reasoning_effort,
                    max_tokens=chat_kwargs.get("max_tokens"),
                    model=chat_kwargs.get("model"),
                )
            except Exception:
                state.record_model_response(
                    ChatResponse(
                        message={},
                        raw={},
                        usage=total_usage,
                        http_attempt_count=total_http_attempts,
                    )
                )
                messages[:] = continuation_messages
                raise
            total_http_attempts += max(0, int(last.http_attempt_count or 0))
            total_usage = self._merge_usage(total_usage, last.usage)
            if not isinstance(last.message, dict):
                return ChatResponse(
                    message=last.message,
                    raw=last.raw,
                    finish_reason=last.finish_reason,
                    usage=total_usage,
                    http_attempt_count=total_http_attempts,
                )
            piece = str(last.message.get("content") or "")
            combined += piece
            continuation_messages.append({"role": "assistant", "content": piece})
            self._progress(
                "model.length_continued",
                state,
                round=round_number,
                attempt=attempt,
            )
            if last.finish_reason != "length":
                merged = dict(last.message)
                merged.pop("tool_calls", None)
                merged["content"] = combined
                return ChatResponse(
                    message=merged,
                    raw=last.raw,
                    finish_reason=last.finish_reason,
                    usage=total_usage,
                    http_attempt_count=total_http_attempts,
                )
        messages[:] = continuation_messages
        state.record_model_response(
            ChatResponse(
                message={},
                raw={},
                usage=total_usage,
                http_attempt_count=total_http_attempts,
            )
        )
        raise RuntimeError("DeepSeek output remained length-truncated after bounded continuation")

    @staticmethod
    def _merge_usage(current: dict[str, Any], usage: dict[str, Any] | None) -> dict[str, int]:
        result = {key: int(value) for key, value in current.items() if isinstance(value, int) and value >= 0}
        if not isinstance(usage, dict):
            return result
        for key, value in usage.items():
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                result[str(key)] = result.get(str(key), 0) + value
        return result

    def _final_synthesis(
        self,
        state: AgentState,
        messages: list[dict[str, Any]],
        *,
        model_route: ModelRoute,
        strategy: TaskStrategy,
        history_compactor: ToolHistoryCompactor | None = None,
        context_window: ContextWindowController | None = None,
        auto_compaction_enabled: bool = False,
        auto_compaction_max_tokens: int = 2_048,
    ) -> str:
        state.convergence.pop("final_synthesis_rejected_finish_reason", None)
        state.convergence.pop("final_synthesis_rejected_protocol", None)
        messages.append(
            {
                "role": "system",
                "content": (
                    "Tool execution budget is closed. Produce the final user-facing answer from the evidence already "
                    "available. Do not call tools. If required work or verification is missing, state that the task is "
                    "incomplete and give the exact resume command."
                ),
            }
        )
        if context_window is None:
            self._compact_tool_history(
                state,
                messages,
                history_compactor,
                phase="final_synthesis",
                checkpoint=False,
            )
        else:
            self._prepare_model_request(
                state,
                messages,
                tools=None,
                model_route=model_route,
                context_window=context_window,
                history_compactor=history_compactor,
                auto_compaction_enabled=auto_compaction_enabled,
                auto_compaction_max_tokens=auto_compaction_max_tokens,
                phase="final_synthesis",
                checkpoint=False,
            )
        synthesis_round = state.round + 1
        final_output_tokens = (
            context_window.effective_output_tokens(model_route.max_tokens)
            if context_window is not None
            else model_route.max_tokens
        )
        self._reserve_model_request(
            state,
            messages,
            phase="final_synthesis",
            tools=None,
            max_tokens=final_output_tokens,
        )
        self.events.publish(
            "model.requested",
            {
                "run_id": state.run_id,
                "round": synthesis_round,
                "message_count": len(messages),
                "model_tier": model_route.tier,
                "model": model_route.model,
                "phase": "final_synthesis",
            },
            project_id=self.project.id,
            session_id=state.session_id,
            run_id=state.run_id,
        )
        self._progress(
            "model.requested",
            state,
            round=synthesis_round,
            max_rounds=synthesis_round,
            current_step="最终总结",
            phase="final_synthesis",
        )
        final_chat_kwargs = {
            "messages": messages,
            "tools": None,
            "tool_choice": None,
            "thinking": strategy.thinking_enabled,
            "reasoning_effort": strategy.reasoning_effort,
            "max_tokens": final_output_tokens,
            "model": model_route.model,
        }
        if context_window is None:
            response = self.client.chat(**final_chat_kwargs)
        else:
            response = self._chat_with_recovery(
                state,
                messages,
                None,
                final_chat_kwargs,
                strategy=strategy,
                model_route=model_route,
                context_window=context_window,
                history_compactor=history_compactor,
                auto_compaction_max_tokens=auto_compaction_max_tokens,
                round_number=synthesis_round,
                request_phase="final_synthesis",
            )
        if response.finish_reason == "length":
            response = self._complete_length_response(
                state,
                messages,
                response,
                final_chat_kwargs,
                strategy=strategy,
                round_number=synthesis_round,
                request_phase="final_synthesis",
            )
        state.record_model_response(response)
        if not _has_usable_finish_reason(response.finish_reason):
            finish_reason = _finish_reason_label(response.finish_reason) or "missing"
            state.convergence["final_synthesis_rejected_finish_reason"] = finish_reason
            self._checkpoint_convergence_transition(
                state,
                messages,
                transition="final_synthesis_rejected",
                phase="final_synthesis",
            )
            return ""
        protocol_violation = _tool_protocol_violation(response.message)
        if protocol_violation:
            state.convergence["final_synthesis_rejected_protocol"] = protocol_violation
            self._checkpoint_convergence_transition(
                state,
                messages,
                transition="final_synthesis_rejected",
                phase="final_synthesis",
            )
            return ""
        reasoning = str(response.message.get("reasoning_content") or "").strip()
        if reasoning:
            self._progress(
                "thinking.content",
                state,
                round=synthesis_round,
                content=reasoning[
                    : self._bounded_config_int(
                        "runtime.max_reasoning_display_chars",
                        4_000,
                        minimum=0,
                        maximum=1_000_000,
                    )
                ],
            )
        self.events.publish(
            "model.responded",
            {
                "run_id": state.run_id,
                "round": synthesis_round,
                "tool_call_count": 0,
                "phase": "final_synthesis",
            },
            project_id=self.project.id,
            session_id=state.session_id,
            run_id=state.run_id,
        )
        self._progress(
            "model.responded",
            state,
            round=synthesis_round,
            tool_call_count=0,
            phase="final_synthesis",
        )
        return str(response.message.get("content") or "").strip()

    @staticmethod
    def _incomplete_answer(state: AgentState, reason: str, *, substantive: str = "") -> str:
        resume = (
            "任务尚未完成："
            + reason
            + f"。会话已保存为 {state.session_id}。可执行 `agent resume --session {state.session_id} 继续完成原任务`，"
            "或在交互界面输入 `/resume " + state.session_id + "` 后继续。"
        )
        evidence = substantive.strip()
        return f"{evidence}\n\n{resume}" if evidence else resume

    def _validate_prompt_size(self, prompt: str) -> None:
        limit = self._bounded_config_int(
            "runtime.max_user_request_chars",
            250_000,
            minimum=1,
            maximum=10_000_000,
        )
        if len(prompt) > limit:
            raise ValueError(
                f"request exceeds runtime.max_user_request_chars ({limit}); save large text/code in the project "
                "and ask the Agent to inspect it in bounded chunks"
            )

    def _bounded_config_int(self, dotted: str, default: int, *, minimum: int, maximum: int) -> int:
        return self.config.get_int(dotted, default, minimum=minimum, maximum=maximum)


__all__ = ["RuntimeSynthesisMixin"]
