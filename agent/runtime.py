from __future__ import annotations

from typing import Any, Callable

from .config import AppConfig
from .context import ContextBuildRequest, ContextBuilder, ContextPackage, ContextSnapshot
from .deepseek import DeepSeekClient, DeepSeekStreamInterrupted
from .events import EventBus, JsonlEventLogger
from .memory import MemoryItem, MemoryStore
from .memory_pipeline import MemoryPipeline
from .model_router import ModelRoute, ModelRouter, more_capable_model_route
from .project import Project
from .prompt import PromptBuilder
from .session import SessionManager
from .state import AgentState
from .task_router import TaskRoute, TaskRouter, more_capable_task_route
from .task_strategy import TaskStrategy, TaskStrategySelector
from .tools import ToolManager
from .unicode_text import normalize_unicode_text


class AgentRuntime:
    def __init__(
        self,
        *,
        config: AppConfig,
        project: Project,
        memory: MemoryStore,
        tools: ToolManager,
        client: DeepSeekClient | None = None,
        events: EventBus | None = None,
        context_builder: ContextBuilder | None = None,
        prompt_builder: PromptBuilder | None = None,
        sessions: SessionManager | None = None,
        progress_handler: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config
        self.project = project
        self.memory = memory
        self.tools = tools
        self.client = client or DeepSeekClient(config)
        self.events = events or EventBus()
        self.context_builder = context_builder or ContextBuilder(config)
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.sessions = sessions or SessionManager(project)
        self.progress_handler = progress_handler
        self.strategy_selector = TaskStrategySelector(config)
        self.task_router = TaskRouter(config)
        self.model_router = ModelRouter(config)
        self.last_session_id: str | None = None
        self.tools.set_event_bus(self.events)
        if config.get("events.jsonl_log", True):
            self.events.subscribe("*", JsonlEventLogger(config.data_dir / "logs"))
        self.memory_pipeline = MemoryPipeline(
            config=config,
            project=project,
            memory=memory,
            events=self.events,
        )

    def run(
        self,
        prompt: str,
        *,
        initial_plan: list[str | dict[str, Any]] | None = None,
        queue_id: str | None = None,
    ) -> str:
        prompt = normalize_unicode_text(prompt).strip()
        if not prompt:
            raise ValueError("prompt must not be empty")
        self._validate_prompt_size(prompt)
        context = self.context_builder.build(self.project)
        memory_items = self.memory.search(prompt, self.project.id, record_usage=False)
        state = AgentState.create(
            session_id=self.sessions.new_session_id(),
            project=self.project,
            user_request=prompt,
            loaded_memories=[],
            loaded_tools=[
                item.name
                for item in self.tools.capabilities(enabled_only=True)
                if self.tools.health.evaluate(item).status == "Available"
            ],
            git_branch=context.git_branch,
            context_index_path=str(context.index_path),
        )
        task_route = self.task_router.route(
            prompt,
            source_file_count=int(context.index.get("source_file_count") or 0),
            file_count=int(context.index.get("file_count") or 0),
        )
        model_route = self.model_router.route(task_route)
        strategy = self._strategy_from_routes(task_route, model_route)
        state.task_route = task_route.to_dict()
        state.model_route = model_route.to_dict()
        state.task_strategy = strategy.to_dict()
        plan = initial_plan or self.strategy_selector.initial_plan(prompt, strategy)
        if plan:
            self.tools.plan_manager.replace(state, plan)
        if state.execution_context:
            state.execution_context.current_queue_id = queue_id
        package = self._build_context_package(
            state=state,
            snapshot=context,
            memory_items=memory_items,
            phase="initial",
        )
        messages = self.prompt_builder.build_initial(package)
        self.last_session_id = state.session_id
        self._progress(
            "strategy.selected",
            state,
            strategy=strategy.to_dict(),
            task_route=task_route.to_dict(),
            model_route=model_route.to_dict(),
        )
        return self._execute(state, messages, snapshot=context)

    def resume(self, prompt: str, session_id: str | None = None) -> str:
        prompt = normalize_unicode_text(prompt).strip()
        if not prompt:
            raise ValueError("resume prompt must not be empty")
        self._validate_prompt_size(prompt)
        resolved_session_id = self.sessions.resolve_session_id(session_id)
        with self.sessions.acquire(resolved_session_id):
            return self._resume_locked(prompt, resolved_session_id)

    def _resume_locked(self, prompt: str, session_id: str) -> str:
        record = self.sessions.load(session_id)
        state = record.state
        if str(state.project.get("id") or "") != self.project.id:
            raise ValueError("saved session belongs to a different project")
        failure_count = self._failure_count(state)
        context = self.context_builder.build(self.project)
        memory_items = self.memory.search(prompt, self.project.id, record_usage=False)
        state.resume(prompt)
        state.loaded_memories = []
        state.loaded_tools = [
            item.name
            for item in self.tools.capabilities(enabled_only=True)
            if self.tools.health.evaluate(item).status == "Available"
        ]
        state.git_branch = context.git_branch
        state.context_index_path = str(context.index_path)
        if state.execution_context:
            state.execution_context.current_directory = state.working_directory
            state.execution_context.git_branch = context.git_branch
            state.execution_context.prompt_phase = "resumed"
        selected_task_route = self.task_router.route(
            prompt,
            source_file_count=int(context.index.get("source_file_count") or 0),
            file_count=int(context.index.get("file_count") or 0),
            failure_count=failure_count,
        )
        previous_task_route = TaskRoute.from_dict(state.task_route or state.task_strategy)
        task_route = more_capable_task_route(previous_task_route, selected_task_route)
        selected_model_route = self.model_router.route(selected_task_route)
        retained_task_model_route = self.model_router.route(task_route)
        candidate_model_route = more_capable_model_route(retained_task_model_route, selected_model_route)
        previous_model_route = (
            ModelRoute.from_dict(state.model_route)
            if state.model_route
            else self.model_router.route(previous_task_route)
        )
        model_route = more_capable_model_route(previous_model_route, candidate_model_route)
        strategy = self._strategy_from_routes(task_route, model_route)
        state.task_route = task_route.to_dict()
        state.model_route = model_route.to_dict()
        state.task_strategy = strategy.to_dict()
        if strategy.require_plan and not state.plan:
            self.tools.plan_manager.replace(state, self.strategy_selector.initial_plan(prompt, strategy))
        package = self._build_context_package(
            state=state,
            snapshot=context,
            memory_items=memory_items,
            phase="resume",
            prior_messages=record.messages,
        )
        messages = self.prompt_builder.build_resume(package)
        self.last_session_id = state.session_id
        self._progress(
            "strategy.selected",
            state,
            strategy=strategy.to_dict(),
            task_route=task_route.to_dict(),
            model_route=model_route.to_dict(),
        )
        return self._execute(state, messages, snapshot=context)

    def _execute(
        self,
        state: AgentState,
        messages: list[dict[str, Any]],
        *,
        snapshot: ContextSnapshot,
    ) -> str:
        self.tools.bind_state(state)
        state.start()
        self.sessions.checkpoint(state, messages)
        self.events.publish(
            "task.started",
            {"run_id": state.run_id, "prompt": state.user_request},
            project_id=self.project.id,
            session_id=state.session_id,
        )
        strategy = self._strategy_from_state(state)
        model_route = ModelRoute.from_dict(state.model_route)
        max_rounds = strategy.max_tool_rounds
        recovery_injected: set[int] = set()
        recovery_chars_used = 0
        recovery_char_limit = self._bounded_config_int(
            "context.max_recovery_context_chars",
            6_000,
            minimum=0,
            maximum=1_000_000,
        )
        try:
            for round_number in range(1, max_rounds + 1):
                state.round = round_number
                state.touch()
                self.events.publish(
                    "model.requested",
                    {
                        "run_id": state.run_id,
                        "round": round_number,
                        "message_count": len(messages),
                        "model_tier": model_route.tier,
                        "model": model_route.model,
                    },
                    project_id=self.project.id,
                    session_id=state.session_id,
                )
                self._progress(
                    "model.requested",
                    state,
                    round=round_number,
                    max_rounds=max_rounds,
                    current_step=state.current_step,
                )
                chat_kwargs = {
                    "messages": messages,
                    "tools": self.tools.schemas(),
                    "tool_choice": "auto",
                    "thinking": strategy.thinking_enabled,
                    "reasoning_effort": strategy.reasoning_effort,
                    "max_tokens": model_route.max_tokens,
                    "model": model_route.model,
                }
                if strategy.thinking_enabled and hasattr(self.client, "chat_stream"):
                    response = self.client.chat_stream(
                        **chat_kwargs,
                        on_reasoning=lambda chunk: self._progress(
                            "thinking.delta",
                            state,
                            round=round_number,
                            content=chunk,
                        ),
                        on_content=None,
                    )
                else:
                    response = self.client.chat(**chat_kwargs)
                message = response.message
                messages.append(message)
                tool_calls = message.get("tool_calls") or []
                reasoning = str(message.get("reasoning_content") or "").strip()
                if reasoning and not (strategy.thinking_enabled and hasattr(self.client, "chat_stream")):
                    self._progress(
                        "thinking.content",
                        state,
                        round=round_number,
                        content=reasoning[: int(self.config.get("runtime.max_reasoning_display_chars", 4000))],
                    )
                self.events.publish(
                    "model.responded",
                    {"run_id": state.run_id, "round": round_number, "tool_call_count": len(tool_calls)},
                    project_id=self.project.id,
                    session_id=state.session_id,
                )
                if not tool_calls:
                    final = str(message.get("content") or "").strip()
                    state.complete(final)
                    self.sessions.finalize(state, messages)
                    self._publish_terminal("task.finished", state, final=final)
                    return final

                for call in tool_calls:
                    function = call.get("function") or {}
                    request, result = self.tools.execute_model_call(
                        str(function.get("name") or ""),
                        function.get("arguments") or "{}",
                        request_id=str(call.get("id") or "") or None,
                    )
                    self._progress(
                        "tool.finished",
                        state,
                        tool=request.capability,
                        success=result.success,
                        duration_ms=result.duration_ms,
                    )
                    state.record_tool_call(request.to_dict(), result.to_dict())
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id"),
                            "content": result.as_text(),
                        }
                    )
                    if not result.success:
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
                                messages.append({"role": "system", "content": recovery_package.rendered})
                    if self.config.get("runtime.checkpoint_each_tool", True):
                        self.sessions.checkpoint(state, messages)

            final = "Tool round limit reached. The task did not finish cleanly. Resume the session to continue."
            state.fail("max_tool_rounds reached", final)
            self.sessions.finalize(state, messages)
            self._publish_terminal("task.failed", state, final=final, error=state.error)
            return final
        except Exception as exc:
            if isinstance(exc, DeepSeekStreamInterrupted):
                state.fail(f"resumable interruption: {exc}")
                if state.execution_context:
                    state.execution_context.prompt_phase = "interrupted"
                self.sessions.finalize(state, messages)
                self._publish_terminal("task.failed", state, error=state.error)
                raise RuntimeError(f"{exc} Session: {state.session_id}") from exc
            state.fail(str(exc))
            self.sessions.finalize(state, messages)
            self._publish_terminal("task.failed", state, error=str(exc))
            raise

    def close(self) -> None:
        self.tools.close()

    def _publish_terminal(
        self,
        event_name: str,
        state: AgentState,
        *,
        final: str = "",
        error: str = "",
    ) -> None:
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
            },
            project_id=self.project.id,
            session_id=state.session_id,
        )

    def _progress(self, event: str, state: AgentState, **payload: Any) -> None:
        if self.progress_handler is None:
            return
        try:
            self.progress_handler(
                {
                    "event": event,
                    "session_id": state.session_id,
                    "mode": (state.task_strategy or {}).get("mode", "standard"),
                    "task_type": (state.task_route or {}).get("task_type", "question"),
                    "model_tier": (state.model_route or {}).get("tier", "standard"),
                    **payload,
                }
            )
        except Exception:
            return

    @staticmethod
    def _strategy_from_state(state: AgentState) -> TaskStrategy:
        value = state.task_strategy or {}
        return TaskStrategy(
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
    def _more_capable_strategy(previous: TaskStrategy, selected: TaskStrategy) -> TaskStrategy:
        ranks = {"simple": 0, "standard": 1, "large": 2, "deep": 3}
        return previous if ranks.get(previous.mode, 1) > ranks.get(selected.mode, 1) else selected

    @staticmethod
    def _strategy_from_routes(task: TaskRoute, model: ModelRoute) -> TaskStrategy:
        return TaskStrategy(
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
        new_ids = [memory_id for memory_id in memory_ids if memory_id not in state.loaded_memories]
        if not new_ids:
            return
        state.loaded_memories.extend(new_ids)
        self.memory.record_usage(new_ids)

    @staticmethod
    def _failure_count(state: AgentState) -> int:
        failed_tools = sum(
            1
            for call in state.tool_calls[-20:]
            if isinstance(call, dict)
            and isinstance(call.get("result"), dict)
            and not bool(call["result"].get("success", False))
        )
        return min(10, failed_tools + int(bool(state.error)))

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
        try:
            parsed = int(self.config.get(dotted, default))
        except (TypeError, ValueError, OverflowError):
            parsed = default
        return max(minimum, min(parsed, maximum))
