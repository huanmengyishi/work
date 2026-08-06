from __future__ import annotations

import hashlib
import logging
from typing import Any

from .convergence import (
    estimate_request_tokens,
)
from .context import ContextBuildRequest, ContextPackage, ContextSnapshot
from .deepseek import ChatResponse
from .event_pipelines import (
    MEMORY_USAGE_RECORDED,
    PROGRESS_UPDATED,
    SESSION_CHECKPOINT_REQUESTED,
    SESSION_FINALIZE_REQUESTED,
)
from .memory import MemoryItem
from .model_router import ModelRoute
from .progress import ProgressTracker
from .state import AgentState
from .task_router import TaskRoute
from .task_strategy import TaskStrategy


logger = logging.getLogger(__name__)


class RuntimeLifecycleMixin:
    def close(self) -> None:
        self.tools.close()

    def _publish_terminal(
        self,
        event_name: str,
        state: AgentState,
        *,
        final: str = "",
        error: str = "",
        memory_refinement: dict[str, Any] | None = None,
    ) -> None:
        self.execution_budget.snapshot(state)
        state_payload = state.to_dict()
        state_payload["run_id"] = state.run_id
        self.events.publish(
            event_name,
            {
                "run_id": state.run_id,
                "prompt": state.user_request,
                "final": final,
                "error": error,
                "state": state_payload,
                "memory_refinement": dict(memory_refinement) if memory_refinement else None,
            },
            project_id=self.project.id,
            session_id=state.session_id,
            run_id=state.run_id,
        )

    def _maybe_refine_memory(
        self,
        state: AgentState,
        messages: list[dict[str, Any]],
        *,
        final: str,
        model_route: ModelRoute,
    ) -> dict[str, Any] | None:
        """Run at most one optional completion-time Memory model request."""

        completed_outcome = (
            (
                state.final_answer,
                state.error,
                state.failure_count,
                state.execution_context.prompt_phase if state.execution_context is not None else None,
            )
            if state.status == "completed"
            else None
        )
        metadata: dict[str, Any] = {}
        stage = "eligibility"
        try:
            tool_calls = self.memory_refiner.current_turn_tool_calls(state.tool_calls, turn=state.turn)
            eligible, reason = self.memory_refiner.eligible(
                success=state.status == "completed",
                current_tool_calls=len(tool_calls),
            )
            already_requested = state.memory_refinement_model_request_count > 0
            metadata = {
                "run_id": state.run_id,
                "eligible": eligible,
                "tool_call_count": len(tool_calls),
                "min_tool_calls": self.memory_refiner.min_tool_calls,
                "logical_requests": 1 if already_requested else 0,
                "status": "skipped" if not eligible or already_requested else "pending",
                "reason": "already_requested" if already_requested else reason,
            }
            state.convergence["memory_refinement"] = metadata
            if already_requested or not eligible:
                state.touch()
                return None

            stage = "budget"
            refinement_messages = self.memory_refiner.build_messages(
                prompt=state.user_request,
                final=final,
                tool_calls=tool_calls,
            )
            admission = self.execution_budget.try_before_model_request(
                state,
                phase="memory_refinement",
                estimated_input_tokens=estimate_request_tokens(refinement_messages, None),
                requested_output_tokens=self.memory_refiner.max_output_tokens,
            )
            if not admission.allowed:
                metadata.update({"status": "skipped", "reason": f"budget_{admission.reason}"})
                state.touch()
                return None

            # Reserve and durably checkpoint the one logical request before any
            # network I/O. If this safety checkpoint fails, refinement is
            # abandoned and the counter prevents a same-turn retry.
            stage = "request_reservation"
            state.record_model_request("memory_refinement")
            metadata.update({"logical_requests": 1, "status": "requested", "reason": ""})
            self.execution_budget.snapshot(state)
            stage = "checkpoint"
            self._checkpoint_session(state, messages)

            # Telemetry is optional. A broken observer must neither spend a
            # second request nor discard an otherwise valid refinement.
            try:
                self.events.publish(
                    "model.requested",
                    {
                        "run_id": state.run_id,
                        "round": state.round + 1,
                        "message_count": len(refinement_messages),
                        "model_tier": model_route.tier,
                        "model": model_route.model,
                        "phase": "memory_refinement",
                    },
                    project_id=self.project.id,
                    session_id=state.session_id,
                    run_id=state.run_id,
                )
            except Exception as exc:
                metadata.setdefault("observability_errors", []).append("model.requested")
                self._log_refinement_exception("model.requested", exc)

            stage = "model_request"
            response = self.client.chat(
                messages=refinement_messages,
                tools=None,
                tool_choice=None,
                thinking=False,
                reasoning_effort=None,
                max_tokens=self.memory_refiner.max_output_tokens,
                model=model_route.model,
            )
            try:
                state.record_model_response(response)
            except Exception as exc:
                metadata.setdefault("observability_errors", []).append("model.metrics")
                self._log_refinement_exception("model.metrics", exc)
            stage = "response_validation"
            refinement = self.memory_refiner.parse_response(
                response.message,
                finish_reason=response.finish_reason,
            )
            metadata.update(
                {
                    "status": "accepted" if refinement is not None else "rejected",
                    "reason": "" if refinement is not None else "invalid_response",
                }
            )
            try:
                self.events.publish(
                    "model.responded",
                    {
                        "run_id": state.run_id,
                        "round": state.round + 1,
                        "tool_call_count": 0,
                        "phase": "memory_refinement",
                        "accepted": refinement is not None,
                    },
                    project_id=self.project.id,
                    session_id=state.session_id,
                    run_id=state.run_id,
                )
            except Exception as exc:
                metadata.setdefault("observability_errors", []).append("model.responded")
                self._log_refinement_exception("model.responded", exc)
            try:
                state.touch()
            except Exception as exc:
                self._log_refinement_exception("state.touch", exc)
            return refinement.to_dict() if refinement is not None else None
        except Exception as exc:
            http_attempt_count = getattr(exc, "http_attempt_count", 0)
            if (
                isinstance(http_attempt_count, int)
                and not isinstance(http_attempt_count, bool)
                and http_attempt_count > 0
            ):
                try:
                    state.record_model_response(ChatResponse(message={}, raw={}, http_attempt_count=http_attempt_count))
                except Exception as metric_exc:
                    self._log_refinement_exception("failed_response.metrics", metric_exc)
            try:
                category = self.resilience.classify(exc).value
            except Exception as classify_exc:
                self._log_refinement_exception("failure.classification", classify_exc)
                category = "internal_error"
            metadata.update({"status": "failed", "reason": f"{stage}_{category}"[:160]})
            self._log_refinement_exception(stage, exc, category=category)
            try:
                state.convergence["memory_refinement"] = metadata
                state.touch()
            except Exception as persist_exc:
                self._log_refinement_exception("failure.state_update", persist_exc)
            return None
        finally:
            # This optional stage is downstream of verified completion. Restore
            # the authoritative terminal fields even if an injected dependency
            # failed while checkpointing or publishing telemetry.
            if completed_outcome is not None:
                final_answer, error, failure_count, prompt_phase = completed_outcome
                state.status = "completed"
                state.final_answer = final_answer
                state.error = error
                state.failure_count = failure_count
                if state.execution_context is not None and prompt_phase is not None:
                    state.execution_context.prompt_phase = prompt_phase

    @staticmethod
    def _log_refinement_exception(stage: str, exc: BaseException, *, category: str = "recoverable") -> None:
        level = logging.ERROR if isinstance(exc, OSError) else logging.WARNING
        logger.log(
            level,
            "memory_refinement_failure stage=%s category=%s exception=%s errno=%s",
            stage,
            category,
            type(exc).__name__,
            getattr(exc, "errno", None),
        )

    def _progress(self, event: str, state: AgentState, **payload: Any) -> None:
        if self.event_pipelines.progress is None:
            return
        progress = ProgressTracker.snapshot(
            state,
            model_request_budget=self.execution_budget.max_model_requests,
            token_budget=self.execution_budget.max_total_tokens,
        )
        self.events.publish(
            PROGRESS_UPDATED,
            {
                "value": {
                    "event": event,
                    "session_id": state.session_id,
                    "mode": (state.task_strategy or {}).get("mode", "standard"),
                    "task_type": (state.task_route or {}).get("task_type", "question"),
                    "model_tier": (state.model_route or {}).get("tier", "standard"),
                    "progress": progress.to_dict(),
                    **payload,
                }
            },
            project_id=self.project.id,
            session_id=state.session_id,
            run_id=state.run_id,
        )

    @classmethod
    def _strategy_from_state(cls, state: AgentState) -> TaskStrategy:
        value = state.task_strategy or {}
        return cls._build_task_strategy(
            mode=str(value.get("mode") or "standard"),
            score=int(value.get("score") or 0),
            reasons=tuple(str(item) for item in value.get("reasons", [])),
            thinking_enabled=bool(value.get("thinking_enabled", False)),
            reasoning_effort=str(value.get("reasoning_effort")) if value.get("reasoning_effort") else None,
            max_tool_rounds=max(1, int(value.get("max_tool_rounds") or 8)),
            require_plan=bool(value.get("require_plan", False)),
            chunked_context=bool(value.get("chunked_context", False)),
        )

    @staticmethod
    def _build_task_strategy(
        *,
        mode: str,
        score: int,
        reasons: tuple[str, ...],
        thinking_enabled: bool,
        reasoning_effort: str | None,
        max_tool_rounds: int,
        require_plan: bool,
        chunked_context: bool,
    ) -> TaskStrategy:
        return TaskStrategy(
            mode=mode,
            score=score,
            reasons=reasons,
            thinking_enabled=thinking_enabled,
            reasoning_effort=reasoning_effort,
            max_tool_rounds=max_tool_rounds,
            require_plan=require_plan,
            chunked_context=chunked_context,
        )

    def _adjust_strategy(
        self,
        state: AgentState,
        task_route: TaskRoute,
        model_route: ModelRoute,
    ) -> ModelRoute:
        configured_limit = self._bounded_config_int(
            "runtime.convergence.max_consecutive_exploration_rounds",
            6,
            minimum=2,
            maximum=32,
        )
        decision = self.strategy_adjuster.adjust(
            task_route,
            model_route,
            configured_exploration_limit=configured_limit,
        )
        experiment = state.convergence.get("experiment")
        parameters = experiment.get("parameters") if isinstance(experiment, dict) else None
        experiment_tier = str(parameters.get("model_tier") or "") if isinstance(parameters, dict) else ""
        target_tier = experiment_tier or decision.model_tier
        adjusted_model = (
            self.model_router.route(task_route, explicit_tier=target_tier)
            if target_tier != model_route.tier
            else model_route
        )
        state.convergence["strategy_adjustment"] = {
            "enabled": decision.enabled,
            "applied": decision.applied,
            "samples": decision.samples,
            "reason": (f"experiment:{experiment_tier}" if experiment_tier else decision.reason)[:160],
            "exploration_round_limit": decision.exploration_round_limit,
            "model_tier": adjusted_model.tier,
        }
        state.touch()
        return adjusted_model

    @staticmethod
    def _more_capable_strategy(previous: TaskStrategy, selected: TaskStrategy) -> TaskStrategy:
        ranks = {"simple": 0, "standard": 1, "large": 2, "deep": 3}
        return previous if ranks.get(previous.mode, 1) > ranks.get(selected.mode, 1) else selected

    @classmethod
    def _strategy_from_routes(cls, task: TaskRoute, model: ModelRoute) -> TaskStrategy:
        return cls._build_task_strategy(
            mode=task.mode,
            score=task.score,
            reasons=task.reasons,
            thinking_enabled=model.thinking_enabled,
            reasoning_effort=model.reasoning_effort,
            max_tool_rounds=task.max_tool_rounds,
            require_plan=task.require_plan,
            chunked_context=task.chunked_context,
        )

    def _build_context_package(
        self,
        *,
        state: AgentState,
        snapshot: ContextSnapshot,
        memory_items: list[MemoryItem],
        phase: str,
        prior_messages: list[dict[str, Any]] | None = None,
    ) -> ContextPackage:
        mode = str((state.task_route or state.task_strategy).get("mode") or "standard")
        default_limit = {"simple": 12_000, "standard": 32_000, "large": 48_000, "deep": 64_000}.get(mode, 32_000)
        configured_limit = self._bounded_config_int(
            f"context.package_limits.{mode}",
            default_limit,
            minimum=1,
            maximum=1_000_000,
        )
        hard_limit = self._bounded_config_int(
            "context.max_package_chars_hard_limit",
            96_000,
            minimum=1,
            maximum=1_000_000,
        )
        package = self.context_builder.build_package(
            ContextBuildRequest(
                snapshot=snapshot,
                state=state,
                memory_items=memory_items,
                capability_summary=self.tools.capability_summary(),
                prior_messages=prior_messages or (),
                phase=phase,
                max_chars=min(configured_limit, hard_limit),
            )
        )
        self._record_included_memories(state, package.included_memory_ids)
        state.context_manifest = {
            "schema_version": package.schema_version,
            "phase": package.phase,
            "fingerprint": package.fingerprint,
            "max_chars": package.max_chars,
            "used_chars": package.used_chars,
            "rendered_chars": package.rendered_chars,
            "original_user_request_chars": package.original_user_request_chars,
            "user_request_truncated": package.user_request_truncated,
            "sections": [section.key for section in package.sections],
            "included_memory_ids": list(package.included_memory_ids),
            "omitted_sections": list(package.omitted_sections),
            "truncated_sections": list(package.truncated_sections),
        }
        return package

    def _record_included_memories(self, state: AgentState, memory_ids: tuple[int, ...]) -> None:
        new_ids = list(dict.fromkeys(memory_id for memory_id in memory_ids if memory_id not in state.loaded_memories))
        if not new_ids:
            return
        self.events.dispatch_required(
            MEMORY_USAGE_RECORDED,
            {
                "memory_ids": new_ids,
                "usage_id": self._memory_usage_id(state, new_ids),
            },
            project_id=self.project.id,
            session_id=state.session_id,
            run_id=state.run_id,
        )
        state.loaded_memories.extend(new_ids)

    def _reserve_model_request(
        self,
        state: AgentState,
        messages: list[dict[str, Any]],
        *,
        phase: str,
        tools: list[dict[str, Any]] | None,
        max_tokens: int | None,
        checkpoint: bool = True,
    ) -> None:
        """Reserve and persist one model request before network I/O begins."""

        requested_output = max(1, int(max_tokens or 1))
        self.execution_budget.before_model_request(
            state,
            phase=phase,
            estimated_input_tokens=estimate_request_tokens(messages, tools),
            requested_output_tokens=requested_output,
        )
        state.record_model_request(phase)
        self.execution_budget.snapshot(state)
        if checkpoint:
            self._checkpoint_session(state, messages)

    def _checkpoint_session(self, state: AgentState, messages: list[dict[str, Any]]) -> None:
        phase = state.execution_context.prompt_phase if state.execution_context is not None else state.status
        state.record_checkpoint(phase=phase, message_count=len(messages))
        self.events.dispatch_required(
            SESSION_CHECKPOINT_REQUESTED,
            {"state": state, "messages": messages},
            project_id=self.project.id,
            session_id=state.session_id,
            run_id=state.run_id,
        )
        self._refresh_session_notes(state)

    def _checkpoint_convergence_transition(
        self,
        state: AgentState,
        messages: list[dict[str, Any]],
        *,
        transition: str,
        phase: str,
        counter: str | None = None,
    ) -> None:
        """Persist one bounded recovery transition before execution continues."""

        metadata = state.convergence if isinstance(state.convergence, dict) else {}
        state.convergence = metadata
        metadata["latest_transition"] = str(transition)[:64]
        metadata["phase"] = str(phase)[:32]
        if counter is not None:
            raw_count = metadata.get(counter, 0)
            current = (
                max(0, min(raw_count, 10_000)) if isinstance(raw_count, int) and not isinstance(raw_count, bool) else 0
            )
            metadata[counter] = min(10_000, current + 1)
        self._checkpoint_session(state, messages)

    def _finalize_session(self, state: AgentState, messages: list[dict[str, Any]]) -> None:
        state.record_checkpoint(phase="finalize", message_count=len(messages))
        self.events.dispatch_required(
            SESSION_FINALIZE_REQUESTED,
            {"state": state, "messages": messages},
            project_id=self.project.id,
            session_id=state.session_id,
            run_id=state.run_id,
        )
        self._refresh_session_notes(state)

    def _refresh_session_notes(self, state: AgentState) -> None:
        builder = getattr(self, "session_memory_builder", None)
        if builder is None:
            return
        try:
            builder.refresh(state.session_id)
        except Exception as exc:
            logger.warning(
                "session_notes_refresh_failed exception=%s errno=%s",
                type(exc).__name__,
                getattr(exc, "errno", None),
            )

    def _persist_failed_terminal(self, state: AgentState, messages: list[dict[str, Any]]) -> None:
        """Persist the failed State before publishing its terminal derivatives.

        If the required Session writer itself fails, do not recursively retry it
        and do not publish a terminal event from an unpersisted State.
        """

        self._finalize_session(state, messages)
        self._publish_terminal("task.failed", state, error=state.error)

    @staticmethod
    def _memory_usage_id(state: AgentState, memory_ids: list[int]) -> str:
        joined_ids = ",".join(str(item) for item in sorted(memory_ids))
        evidence = f"{state.run_id}\0{joined_ids}".encode("utf-8")
        return f"memory-usage:{hashlib.sha256(evidence).hexdigest()}"


__all__ = ["RuntimeLifecycleMixin"]
