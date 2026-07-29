from __future__ import annotations

import hashlib
import json
from typing import Any

from .budget import ExecutionBudgetExceeded
from .convergence import (
    ContextWindowController,
    repair_tool_message_pairs,
)
from .deepseek import ChatResponse
from .events import EventDispatchError
from .model_router import ModelRoute
from .state import AgentState
from .runtime_support import (
    _finish_reason_label,
    _has_usable_finish_reason,
)


class RuntimeCompactionMixin:
    def _auto_compact_context(
        self,
        state: AgentState,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        model_route: ModelRoute,
        context_window: ContextWindowController,
        auto_compaction_max_tokens: int,
        phase: str,
    ) -> bool:
        span = context_window.compaction_span(messages)
        if span is None:
            context_window.record_failure()
            self._progress(
                "context.compaction_failed",
                state,
                failure_count=context_window.failure_count,
                circuit_open=context_window.circuit_open,
                reason="no complete old API round is available",
                phase=phase,
            )
            self._checkpoint_convergence_transition(
                state,
                messages,
                transition="context_compaction_failed",
                phase=phase,
            )
            return False
        start, end = span
        history = list(messages[start:end])
        auto_max_tokens = context_window.effective_output_tokens(
            min(auto_compaction_max_tokens, model_route.max_tokens)
        )
        state_evidence = {
            "objective": self._bounded_context_text(state.objective),
            "current_request": self._bounded_context_text(state.user_request),
            "plan": [{"id": step.id, "title": step.title, "status": step.status} for step in state.plan],
            "modified_files": list(state.execution_context.modified_files) if state.execution_context else [],
            "recent_error": state.execution_context.recent_error if state.execution_context else "",
        }

        compact_messages = self._context_compaction_prompt(state_evidence, history)
        ptl_drops = 0
        compact_budget = context_window.budget(
            compact_messages,
            None,
            max_output_tokens=auto_max_tokens,
        )
        while compact_budget.over_limit:
            reduced = self._drop_oldest_api_round(history)
            if reduced is None:
                break
            history = reduced
            ptl_drops += 1
            compact_messages = self._context_compaction_prompt(state_evidence, history)
            compact_budget = context_window.budget(
                compact_messages,
                None,
                max_output_tokens=auto_max_tokens,
            )
        if compact_budget.over_limit:
            context_window.record_failure()
            self._progress(
                "context.compaction_failed",
                state,
                failure_count=context_window.failure_count,
                circuit_open=context_window.circuit_open,
                reason="compaction input remained over limit after all droppable complete API rounds were removed",
                phase=phase,
            )
            self._checkpoint_convergence_transition(
                state,
                messages,
                transition="context_compaction_failed",
                phase=phase,
            )
            return False

        synthesis_round = state.round
        self._reserve_model_request(
            state,
            compact_messages,
            phase="context_compaction",
            tools=None,
            max_tokens=auto_max_tokens,
            checkpoint=False,
        )
        self._checkpoint_convergence_transition(
            state,
            messages,
            transition="context_compaction_started",
            phase=phase,
        )
        self.events.publish(
            "model.requested",
            {
                "run_id": state.run_id,
                "round": synthesis_round,
                "message_count": len(compact_messages),
                "model_tier": model_route.tier,
                "model": model_route.model,
                "phase": "context_compaction",
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
            current_step="上下文压缩",
            phase="context_compaction",
        )
        try:
            response = self.client.chat(
                messages=compact_messages,
                tools=None,
                tool_choice=None,
                thinking=False,
                reasoning_effort=None,
                max_tokens=auto_max_tokens,
                model=model_route.model,
            )
            state.record_model_response(response)
            if response.finish_reason == "length":
                context_window.record_failure()
                self._progress(
                    "context.compaction_failed",
                    state,
                    failure_count=context_window.failure_count,
                    circuit_open=context_window.circuit_open,
                    reason="length-truncated summary",
                    phase=phase,
                )
                self._checkpoint_convergence_transition(
                    state,
                    messages,
                    transition="context_compaction_failed",
                    phase=phase,
                )
                return False
            if not _has_usable_finish_reason(response.finish_reason):
                finish_reason = _finish_reason_label(response.finish_reason) or "missing"
                context_window.record_failure()
                self._progress(
                    "context.compaction_failed",
                    state,
                    failure_count=context_window.failure_count,
                    circuit_open=context_window.circuit_open,
                    reason=f"unusable finish_reason={finish_reason}",
                    phase=phase,
                )
                self._checkpoint_convergence_transition(
                    state,
                    messages,
                    transition="context_compaction_failed",
                    phase=phase,
                )
                return False
            summary = str(response.message.get("content") or "").strip()
            if not summary:
                raise RuntimeError("context compaction returned an empty summary")
        except (EventDispatchError, ExecutionBudgetExceeded):
            raise
        except Exception as exc:
            http_attempt_count = getattr(exc, "http_attempt_count", 0)
            if (
                isinstance(http_attempt_count, int)
                and not isinstance(http_attempt_count, bool)
                and http_attempt_count > 0
            ):
                state.record_model_response(ChatResponse(message={}, raw={}, http_attempt_count=http_attempt_count))
            context_window.record_failure()
            self._progress(
                "context.compaction_failed",
                state,
                failure_count=context_window.failure_count,
                circuit_open=context_window.circuit_open,
                reason=type(exc).__name__,
                phase=phase,
            )
            self._checkpoint_convergence_transition(
                state,
                messages,
                transition="context_compaction_failed",
                phase=phase,
            )
            return False
        self.events.publish(
            "model.responded",
            {
                "run_id": state.run_id,
                "round": synthesis_round,
                "tool_call_count": 0,
                "phase": "context_compaction",
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
            phase="context_compaction",
        )

        original_budget = context_window.budget(messages, tools, max_output_tokens=model_route.max_tokens)
        candidate = [
            *messages[:start],
            {
                "role": "system",
                "content": (
                    "[Deep Agent automatic context summary]\n"
                    + summary
                    + (f"\nOldest complete API rounds omitted before summarization: {ptl_drops}." if ptl_drops else "")
                ),
            },
            *messages[end:],
        ]
        candidate = repair_tool_message_pairs(candidate).messages
        compacted_budget = context_window.budget(candidate, tools, max_output_tokens=model_route.max_tokens)
        if compacted_budget.estimated_tokens >= original_budget.estimated_tokens or compacted_budget.over_trigger:
            context_window.record_failure()
            self._progress(
                "context.compaction_failed",
                state,
                failure_count=context_window.failure_count,
                circuit_open=context_window.circuit_open,
                reason="summary did not reduce the request below the trigger",
                phase=phase,
            )
            self._checkpoint_convergence_transition(
                state,
                messages,
                transition="context_compaction_failed",
                phase=phase,
            )
            return False
        messages[:] = candidate
        context_window.record_success()
        self._progress(
            "context.compacted",
            state,
            original_tokens=original_budget.estimated_tokens,
            final_tokens=compacted_budget.estimated_tokens,
            summarized_messages=end - start,
            ptl_drops=ptl_drops,
            phase=phase,
        )
        self._checkpoint_convergence_transition(
            state,
            messages,
            transition="context_compacted",
            phase=phase,
            counter="context_compaction_count",
        )
        return True

    @staticmethod
    def _bounded_context_text(value: str, limit: int = 8_000) -> str:
        if len(value) <= limit:
            return value
        marker = "\n...[middle omitted]...\n"
        available = limit - len(marker)
        head = available // 2
        return value[:head] + marker + value[-(available - head) :]

    @staticmethod
    def _context_compaction_prompt(
        state_evidence: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": (
                    "Compress prior Deep Agent API rounds into a factual continuation summary. Preserve the original "
                    "objective, source evidence and paths, Task Graph progress, managed modifications/snapshots, test "
                    "and diagnostic results, failures, unresolved questions, and exact remaining work. Do not invent "
                    "facts. Do not call tools and do not answer the user's task."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"state": state_evidence, "old_api_rounds": history},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]

    @staticmethod
    def _drop_oldest_api_round(history: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        start = next(
            (index for index, item in enumerate(history) if item.get("role") == "assistant"),
            None,
        )
        if start is None:
            return None
        end = start + 1
        while end < len(history) and history[end].get("role") != "assistant":
            end += 1
        return [*history[:start], *history[end:]]

    def _emergency_context_collapse(
        self,
        state: AgentState,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        model_route: ModelRoute,
        context_window: ContextWindowController,
        phase: str,
    ) -> bool:
        span = context_window.compaction_span(messages)
        if span is None:
            return False
        start, end = span
        prefix = list(messages[:start])
        removed = list(messages[start:end])
        suffix = list(messages[end:])
        levels = (
            (8_000, 12, 600, 12, 1_000, 200, 512),
            (4_000, 8, 300, 8, 500, 100, 256),
            (2_000, 4, 160, 4, 300, 50, 160),
            (1_000, 2, 80, 2, 160, 20, 80),
            (500, 1, 0, 1, 80, 10, 40),
        )
        selected: list[dict[str, Any]] | None = None
        selected_budget = None
        for text_limit, tool_count, excerpt_limit, preview_count, preview_limit, file_count, target_limit in levels:
            removed_previews = [
                {
                    "tool_call_id": str(item.get("tool_call_id") or ""),
                    "content": self._bounded_context_text(
                        str(item.get("content") or ""),
                        limit=preview_limit,
                    ),
                }
                for item in removed
                if item.get("role") == "tool"
            ][-preview_count:]
            recent_tools: list[dict[str, Any]] = []
            for item in state.tool_calls[-tool_count:]:
                request = item.get("request") if isinstance(item.get("request"), dict) else {}
                result = item.get("result") if isinstance(item.get("result"), dict) else {}
                args = request.get("args") if isinstance(request.get("args"), dict) else {}
                data = result.get("data") if isinstance(result.get("data"), dict) else {}
                target = {
                    key: (
                        self._bounded_context_text(str(args[key]), limit=target_limit)
                        if isinstance(args[key], str)
                        else args[key]
                    )
                    for key in ("path", "start_line", "end_line", "query", "glob", "pattern")
                    if key in args
                }
                tool_evidence = {
                    "round": int(item.get("round") or 0),
                    "capability": f"{request.get('tool', '?')}.{request.get('action', '?')}",
                    "success": bool(result.get("success")),
                    "target": target,
                    "result_path": self._bounded_context_text(str(data.get("path") or ""), limit=target_limit),
                    "stdout_sha256": hashlib.sha256(
                        str(result.get("stdout") or "").encode("utf-8", errors="replace")
                    ).hexdigest(),
                    "stderr_sha256": hashlib.sha256(
                        str(result.get("stderr") or "").encode("utf-8", errors="replace")
                    ).hexdigest(),
                }
                if excerpt_limit:
                    tool_evidence.update(
                        {
                            "stdout_excerpt": self._bounded_context_text(
                                str(result.get("stdout") or ""), limit=excerpt_limit
                            ),
                            "stderr_excerpt": self._bounded_context_text(
                                str(result.get("stderr") or ""), limit=excerpt_limit
                            ),
                        }
                    )
                recent_tools.append(tool_evidence)
            evidence = {
                "objective": self._bounded_context_text(state.objective, limit=text_limit),
                "current_request": self._bounded_context_text(state.user_request, limit=text_limit),
                "plan": [
                    {
                        "id": step.id,
                        "title": self._bounded_context_text(step.title, limit=256),
                        "status": step.status,
                    }
                    for step in state.plan
                ],
                "modified_files": [
                    self._bounded_context_text(path, limit=target_limit)
                    for path in (
                        state.execution_context.modified_files[-file_count:] if state.execution_context else []
                    )
                ],
                "recent_tools": recent_tools,
                "removed_tool_previews": removed_previews,
            }
            candidate = [
                *prefix,
                {
                    "role": "system",
                    "content": (
                        "[Deep Agent emergency context collapse]\n"
                        f"{end - start} old model-visible messages were removed only after automatic compaction was "
                        "unavailable or repeatedly failed. Continue from this bounded authoritative state projection; "
                        "do not infer omitted tool bodies.\n"
                        + json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    ),
                },
                *suffix,
            ]
            candidate = repair_tool_message_pairs(candidate).messages
            candidate_budget = context_window.budget(candidate, tools, max_output_tokens=model_route.max_tokens)
            if not candidate_budget.over_limit:
                selected = candidate
                selected_budget = candidate_budget
                break
        if selected is None or selected_budget is None:
            return False
        messages[:] = selected
        self._progress(
            "context.emergency_collapsed",
            state,
            final_tokens=selected_budget.estimated_tokens,
            removed_messages=end - start,
            phase=phase,
        )
        self._checkpoint_convergence_transition(
            state,
            messages,
            transition="context_emergency_collapsed",
            phase=phase,
        )
        return True


__all__ = ["RuntimeCompactionMixin"]
