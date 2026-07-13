from __future__ import annotations

from typing import Any, Callable

from .config import AppConfig
from .context import ContextBuilder
from .deepseek import DeepSeekClient, DeepSeekStreamInterrupted
from .events import EventBus, JsonlEventLogger
from .memory import MemoryStore
from .memory_pipeline import MemoryPipeline
from .project import Project
from .prompt import PromptBuilder
from .session import SessionManager
from .state import AgentState
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
        context = self.context_builder.build(self.project)
        memory_items = self.memory.search(prompt, self.project.id)
        state = AgentState.create(
            session_id=self.sessions.new_session_id(),
            project=self.project,
            user_request=prompt,
            loaded_memories=[item.id for item in memory_items],
            loaded_tools=[
                item.name
                for item in self.tools.capabilities(enabled_only=True)
                if self.tools.health.evaluate(item).status == "Available"
            ],
            git_branch=context.git_branch,
            context_index_path=str(context.index_path),
        )
        selected_strategy = self.strategy_selector.select(
            prompt,
            source_file_count=int(context.index.get("source_file_count") or 0),
            file_count=int(context.index.get("file_count") or 0),
        )
        previous_strategy = self._strategy_from_state(state) if state.task_strategy else selected_strategy
        strategy = self._more_capable_strategy(previous_strategy, selected_strategy)
        state.task_strategy = strategy.to_dict()
        plan = initial_plan or self.strategy_selector.initial_plan(prompt, strategy)
        if plan:
            self.tools.plan_manager.replace(state, plan)
        if state.execution_context:
            state.execution_context.current_queue_id = queue_id
        messages = self.prompt_builder.build_initial(
            state=state,
            context=context,
            memory_context=self.memory.context_block_from_items(memory_items),
            capability_summary=self.tools.capability_summary(),
        )
        self.last_session_id = state.session_id
        self._progress("strategy.selected", state, strategy=strategy.to_dict())
        return self._execute(state, messages)

    def resume(self, prompt: str, session_id: str | None = None) -> str:
        prompt = normalize_unicode_text(prompt).strip()
        if not prompt:
            raise ValueError("resume prompt must not be empty")
        record = self.sessions.load(session_id)
        state = record.state
        if str(state.project.get("id") or "") != self.project.id:
            raise ValueError("saved session belongs to a different project")
        context = self.context_builder.build(self.project)
        memory_items = self.memory.search(prompt, self.project.id)
        state.resume(prompt)
        state.loaded_memories = [item.id for item in memory_items]
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
        selected_strategy = self.strategy_selector.select(
            prompt,
            source_file_count=int(context.index.get("source_file_count") or 0),
            file_count=int(context.index.get("file_count") or 0),
        )
        previous_strategy = self._strategy_from_state(state) if state.task_strategy else selected_strategy
        strategy = self._more_capable_strategy(previous_strategy, selected_strategy)
        state.task_strategy = strategy.to_dict()
        if strategy.require_plan and not state.plan:
            self.tools.plan_manager.replace(state, self.strategy_selector.initial_plan(prompt, strategy))
        messages = self.prompt_builder.append_resume(
            record.messages,
            state=state,
            context=context,
            memory_context=self.memory.context_block_from_items(memory_items),
            capability_summary=self.tools.capability_summary(),
        )
        self.last_session_id = state.session_id
        self._progress("strategy.selected", state, strategy=strategy.to_dict())
        return self._execute(state, messages)

    def _execute(self, state: AgentState, messages: list[dict[str, Any]]) -> str:
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
        max_rounds = strategy.max_tool_rounds
        recovery_injected: set[int] = set()
        try:
            for round_number in range(1, max_rounds + 1):
                state.round = round_number
                state.touch()
                self.events.publish(
                    "model.requested",
                    {"run_id": state.run_id, "round": round_number, "message_count": len(messages)},
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
                        if unseen:
                            recovery_injected.update(item.id for item in unseen)
                            state.loaded_memories.extend(
                                item.id for item in unseen if item.id not in state.loaded_memories
                            )
                            messages.append(
                                {
                                    "role": "system",
                                    "content": "## Failure Recovery Memory\n\n"
                                    "The last tool call failed. Use these prior corrections or lessons to diagnose "
                                    "the error before retrying. Do not repeat an already documented failed approach.\n\n"
                                    + self.memory.context_block_from_items(unseen),
                                }
                            )
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
