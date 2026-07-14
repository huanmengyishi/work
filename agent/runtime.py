from __future__ import annotations

import hashlib
import json
import re
import shlex
from pathlib import Path
from typing import Any, Callable

from .config import AppConfig
from .convergence import (
    ContextWindowController,
    TaskConvergenceController,
    ToolHistoryCompactor,
    repair_tool_message_pairs,
)
from .context import ContextBuildRequest, ContextBuilder, ContextPackage, ContextSnapshot
from .deepseek import ChatResponse, DeepSeekClient, DeepSeekContextOverflow, DeepSeekStreamInterrupted
from .event_pipelines import (
    MEMORY_USAGE_RECORDED,
    PROGRESS_UPDATED,
    SESSION_CHECKPOINT_REQUESTED,
    SESSION_FINALIZE_REQUESTED,
    RuntimeEventPipelines,
)
from .events import EventBus, EventDispatchError
from .memory import MemoryItem, MemoryStore
from .model_router import ModelRoute, ModelRouter, more_capable_model_route
from .project import Project
from .prompt import PromptBuilder
from .session import SessionManager
from .state import AgentState
from .task_plan import TaskPlanFactory
from .task_router import TaskRoute, TaskRouter, more_capable_task_route
from .task_strategy import TaskStrategy
from .tool_orchestration import PreparedToolCall, ToolBatchInterrupted, execute_model_tool_calls
from .tools import ToolManager
from .unicode_text import normalize_unicode_text


_DATE_LITERAL_RE = re.compile(r"(?<!\d)20\d{2}(?:年\s*\d{1,2}月(?:\s*\d{1,2}日)?|[-/.]\d{1,2}(?:[-/.]\d{1,2})?)(?!\d)")
_MAX_TOOL_CALLS_PROTOCOL = 64
_USABLE_FINISH_REASONS = frozenset({"", "stop", "tool_calls"})
_DEEPSEEK_TOOL_PROTOCOL_MARKERS = (
    "<｜｜DSML｜｜tool_calls>",
    "<｜｜DSML｜｜invoke",
)
_SINGLE_VALIDATION_MODEL_FUNCTIONS = frozenset({"run_tests", "lsp_diagnostics"})
_SINGLE_VALIDATION_CAPABILITIES = frozenset({"template.run_tests", "lsp.diagnostics"})
_VALIDATION_SHELL_PROGRAMS = frozenset(
    {
        "bun",
        "cargo",
        "go",
        "gradle",
        "gradlew",
        "mvn",
        "mypy",
        "nox",
        "npm",
        "npx",
        "pnpm",
        "py.test",
        "pyright",
        "pytest",
        "ruff",
        "tsc",
        "tox",
        "yarn",
    }
)


def _finish_reason_label(value: object) -> str:
    return normalize_unicode_text(str(value or "")).strip().lower()[:64]


def _has_usable_finish_reason(value: object) -> bool:
    return _finish_reason_label(value) in _USABLE_FINISH_REASONS


def _tool_protocol_violation(message: dict[str, Any]) -> str:
    """Describe model tool protocol that cannot be accepted as answer text."""

    raw_tool_calls = message.get("tool_calls")
    if raw_tool_calls:
        return "structured tool calls"
    return _tool_protocol_text_violation(message)


def _tool_protocol_text_violation(message: dict[str, Any]) -> str:
    content = str(message.get("content") or "")
    if any(marker in content for marker in _DEEPSEEK_TOOL_PROTOCOL_MARKERS):
        return "DeepSeek tool-call protocol text"
    return ""


def _date_key(value: str) -> tuple[int, ...] | None:
    numbers = [int(item) for item in re.findall(r"\d+", value)]
    if len(numbers) < 2:
        return None
    year, month = numbers[:2]
    if not (2000 <= year <= 2099 and 1 <= month <= 12):
        return None
    if len(numbers) >= 3:
        day = numbers[2]
        if not 1 <= day <= 31:
            return None
        return year, month, day
    return year, month


def _date_keys_from_text(value: str) -> set[tuple[int, ...]]:
    return {key for item in _DATE_LITERAL_RE.findall(value) if (key := _date_key(item)) is not None}


def _normalize_assistant_tool_calls(
    message: dict[str, Any],
    *,
    turn: int,
    round_number: int,
) -> tuple[dict[str, Any], int, int]:
    """Return one protocol-safe assistant message before any tool executes."""

    raw_calls = message.get("tool_calls")
    if not raw_calls:
        return message, 0, 0
    if isinstance(raw_calls, (list, tuple)):
        dropped = max(0, len(raw_calls) - _MAX_TOOL_CALLS_PROTOCOL)
        items = list(raw_calls[:_MAX_TOOL_CALLS_PROTOCOL])
    else:
        dropped = 0
        items = [raw_calls]
    normalized_calls: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    changed = 0
    for index, raw_call in enumerate(items):
        source = raw_call if isinstance(raw_call, dict) else {}
        call = dict(source)
        raw_function = call.get("function")
        function = dict(raw_function) if isinstance(raw_function, dict) else {}
        function["name"] = str(function.get("name") or "")
        if function.get("arguments") is None:
            function["arguments"] = "{}"
        call["type"] = "function"
        call["function"] = function

        call_id = str(call.get("id") or "").strip()
        if len(call_id) > 200:
            digest = hashlib.sha256(call_id.encode("utf-8", errors="replace")).hexdigest()[:32]
            call_id = "deep-agent-call-" + digest
        if not call_id or call_id in seen_ids:
            base = f"deep-agent-call-t{turn}-r{round_number}-i{index + 1}"
            call_id = base
            suffix = 2
            while call_id in seen_ids:
                call_id = f"{base}-{suffix}"
                suffix += 1
        seen_ids.add(call_id)
        call["id"] = call_id
        normalized_calls.append(call)
        if not isinstance(raw_call, dict) or call != raw_call:
            changed += 1

    updated = dict(message)
    updated["tool_calls"] = normalized_calls
    return updated, changed, dropped


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
        self.task_router = TaskRouter(config)
        self.model_router = ModelRouter(config)
        self.task_plan_factory = TaskPlanFactory()
        self.last_session_id: str | None = None
        self.tools.set_event_bus(self.events)
        self.event_pipelines = RuntimeEventPipelines(
            config=config,
            project=project,
            sessions=self.sessions,
            memory=memory,
            health=self.tools.health,
            events=self.events,
            progress_handler=progress_handler,
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
        plan = initial_plan or self.task_plan_factory.build(task_route)
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
            self.tools.plan_manager.replace(state, self.task_plan_factory.build(task_route))
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
        try:
            self._checkpoint_session(state, messages)
        except EventDispatchError as exc:
            state.fail(str(exc))
            raise
        self.events.publish(
            "task.started",
            {"run_id": state.run_id, "prompt": state.user_request},
            project_id=self.project.id,
            session_id=state.session_id,
            run_id=state.run_id,
        )
        strategy = self._strategy_from_state(state)
        model_route = ModelRoute.from_dict(state.model_route)
        soft_tool_turn_target = strategy.max_tool_rounds
        hard_tool_turn_limit = self._bounded_config_int(
            "runtime.max_tool_rounds_hard_limit",
            32,
            minimum=soft_tool_turn_target,
            maximum=10_000,
        )
        tool_turn = 0
        model_round = 0
        corrective_rounds = 0
        max_corrective_rounds = 2
        abnormal_finish_recoveries = 0
        max_abnormal_finish_recoveries = 1
        loop_exit_reason = "hard_limit"
        recovery_injected: set[int] = set()
        recovery_chars_used = 0
        recovery_char_limit = self._bounded_config_int(
            "context.max_recovery_context_chars",
            6_000,
            minimum=0,
            maximum=1_000_000,
        )
        convergence_enabled = bool(self.config.get("runtime.convergence.enabled", True))
        configured_reserved_rounds = self._bounded_config_int(
            "runtime.convergence.reserved_tool_rounds",
            4,
            minimum=1,
            maximum=16,
        )
        mode_reserved_rounds = soft_tool_turn_target // 3 if strategy.mode in {"large", "deep"} else 1
        convergence = TaskConvergenceController(
            mode=strategy.mode if convergence_enabled else "standard",
            max_rounds=soft_tool_turn_target,
            exploration_round_limit=self._bounded_config_int(
                "runtime.convergence.max_consecutive_exploration_rounds",
                6,
                minimum=2,
                maximum=32,
            ),
            reserved_rounds=max(configured_reserved_rounds, mode_reserved_rounds),
            implementation_read_limit=self._bounded_config_int(
                "runtime.convergence.max_implementation_evidence_reads",
                2,
                minimum=0,
                maximum=4,
            ),
            validation_attachment_read_limit=self._bounded_config_int(
                "runtime.convergence.max_validation_attachment_reads",
                2,
                minimum=0,
                maximum=4,
            ),
        )
        convergence.bind(state)
        keep_recent_rounds = self._bounded_config_int(
            "runtime.convergence.keep_recent_tool_results",
            4,
            minimum=1,
            maximum=100,
        )
        compaction_failure_limit = self._bounded_config_int(
            "runtime.convergence.compaction_failure_limit",
            3,
            minimum=1,
            maximum=20,
        )
        compacted_result_chars = self._bounded_config_int(
            "runtime.convergence.compacted_tool_result_chars",
            1_200,
            minimum=256,
            maximum=8_000,
        )
        history_compactor: ToolHistoryCompactor | None = None
        round_compactor: ToolHistoryCompactor | None = None
        if convergence_enabled:
            history_compactor = ToolHistoryCompactor(
                aggregate_chars=self._bounded_config_int(
                    "runtime.convergence.aggregate_tool_result_chars",
                    96_000,
                    minimum=4_096,
                    maximum=2_000_000,
                ),
                output_reserve_chars=self._bounded_config_int(
                    "runtime.convergence.output_reserve_chars",
                    24_000,
                    minimum=0,
                    maximum=1_000_000,
                ),
                compacted_result_chars=compacted_result_chars,
                keep_recent_results=keep_recent_rounds,
                failure_limit=compaction_failure_limit,
            )
            round_compactor = ToolHistoryCompactor(
                aggregate_chars=self._bounded_config_int(
                    "runtime.convergence.same_round_tool_result_chars",
                    48_000,
                    minimum=4_096,
                    maximum=1_000_000,
                ),
                output_reserve_chars=0,
                compacted_result_chars=compacted_result_chars,
                keep_recent_results=1,
                failure_limit=compaction_failure_limit,
            )
        context_window = ContextWindowController(
            context_window_tokens=self._bounded_config_int(
                "model.context_window_tokens",
                65_536,
                minimum=8_192,
                maximum=4_000_000,
            ),
            safety_buffer_tokens=self._bounded_config_int(
                "runtime.convergence.context_safety_buffer_tokens",
                8_192,
                minimum=1_024,
                maximum=1_000_000,
            ),
            keep_recent_rounds=keep_recent_rounds,
            failure_limit=compaction_failure_limit,
        )
        context_window.bind(state)
        auto_compaction_enabled = convergence_enabled and bool(
            self.config.get("runtime.convergence.auto_compaction_enabled", True)
        )
        auto_compaction_max_tokens = self._bounded_config_int(
            "runtime.convergence.auto_compaction_max_tokens",
            2_048,
            minimum=256,
            maximum=20_000,
        )
        single_tool_result_chars = self._bounded_config_int(
            "runtime.convergence.single_tool_result_chars",
            12_000,
            minimum=512,
            maximum=100_000,
        )
        max_tool_calls_per_round = self._bounded_config_int(
            "runtime.convergence.max_tool_calls_per_round",
            16,
            minimum=1,
            maximum=64,
        )
        try:
            while tool_turn < hard_tool_turn_limit:
                model_round += 1
                round_number = model_round
                state.round = tool_turn + 1
                state.touch()
                convergence_action = convergence.before_round(min(tool_turn + 1, soft_tool_turn_target), state)
                for notice in convergence_action.messages:
                    messages.append({"role": "system", "content": notice})
                if convergence_action.messages:
                    self._checkpoint_session(state, messages)
                active_tools = convergence.filter_schemas(
                    self.tools.schemas(),
                    convergence_action.excluded_functions,
                )
                single_validation = self._single_validation_requested(state)
                validation_consumed = single_validation and self._single_validation_used(state)
                if validation_consumed:
                    active_tools = [
                        item
                        for item in active_tools
                        if str((item.get("function") or {}).get("name") or "") not in _SINGLE_VALIDATION_MODEL_FUNCTIONS
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
                    model_route=model_route,
                    context_window=context_window,
                    history_compactor=history_compactor,
                    auto_compaction_enabled=auto_compaction_enabled,
                    auto_compaction_max_tokens=auto_compaction_max_tokens,
                    phase="tool_loop",
                    checkpoint=True,
                )
                state.record_model_request("main_loop")
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
                    run_id=state.run_id,
                )
                self._progress(
                    "model.requested",
                    state,
                    round=tool_turn + 1,
                    max_rounds=soft_tool_turn_target,
                    hard_limit=hard_tool_turn_limit,
                    current_step=state.current_step,
                )
                chat_kwargs = {
                    "messages": messages,
                    "tools": active_tools,
                    "tool_choice": "auto",
                    "thinking": strategy.thinking_enabled,
                    "reasoning_effort": strategy.reasoning_effort,
                    "max_tokens": context_window.effective_output_tokens(model_route.max_tokens),
                    "model": model_route.model,
                }
                response = self._chat_with_recovery(
                    state,
                    messages,
                    active_tools,
                    chat_kwargs,
                    strategy=strategy,
                    model_route=model_route,
                    context_window=context_window,
                    history_compactor=history_compactor,
                    auto_compaction_max_tokens=auto_compaction_max_tokens,
                    round_number=round_number,
                    request_phase="main_loop",
                )
                response = self._complete_length_response(
                    state,
                    messages,
                    response,
                    chat_kwargs,
                    strategy=strategy,
                    round_number=round_number,
                    request_phase="main_loop",
                )
                if response.finish_reason != "length":
                    state.record_model_response(response)
                if not _has_usable_finish_reason(response.finish_reason):
                    finish_reason = _finish_reason_label(response.finish_reason) or "missing"
                    raw_tool_calls = response.message.get("tool_calls")
                    discarded_tool_calls = (
                        len(raw_tool_calls) if isinstance(raw_tool_calls, list) else int(bool(raw_tool_calls))
                    )
                    messages.append(
                        {
                            "role": "assistant",
                            "content": (
                                "[Deep Agent rejected an unusable model response with "
                                f"finish_reason={finish_reason}; {discarded_tool_calls} tool call(s) were not executed]"
                            ),
                        }
                    )
                    self.events.publish(
                        "model.responded",
                        {
                            "run_id": state.run_id,
                            "round": round_number,
                            "tool_call_count": 0,
                            "discarded_tool_call_count": discarded_tool_calls,
                            "finish_reason": finish_reason,
                        },
                        project_id=self.project.id,
                        session_id=state.session_id,
                        run_id=state.run_id,
                    )
                    self._progress(
                        "model.responded",
                        state,
                        round=round_number,
                        tool_call_count=0,
                        discarded_tool_call_count=discarded_tool_calls,
                        finish_reason=finish_reason,
                    )
                    if abnormal_finish_recoveries < max_abnormal_finish_recoveries:
                        abnormal_finish_recoveries += 1
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
                        continue
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
                    state.fail(f"unusable finish_reason: {finish_reason}", final)
                    messages.append({"role": "assistant", "content": final})
                    self._finalize_session(state, messages)
                    self._publish_terminal("task.failed", state, final=final, error=state.error)
                    return final
                message, normalized_tool_call_count, dropped_tool_call_count = _normalize_assistant_tool_calls(
                    response.message,
                    turn=state.turn,
                    round_number=round_number,
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
                if normalized_tool_call_count:
                    self._progress(
                        "protocol.tool_calls_normalized",
                        state,
                        round=round_number,
                        normalized_count=normalized_tool_call_count,
                    )
                if dropped_tool_call_count:
                    self._progress(
                        "protocol.tool_calls_dropped",
                        state,
                        round=round_number,
                        retained_count=len(tool_calls),
                        dropped_count=dropped_tool_call_count,
                        hard_limit=_MAX_TOOL_CALLS_PROTOCOL,
                    )
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
                    {
                        "run_id": state.run_id,
                        "round": round_number,
                        "tool_call_count": len(tool_calls),
                        "dropped_tool_call_count": dropped_tool_call_count,
                        "discarded_protocol_tool_call_count": protocol_discarded_tool_calls,
                    },
                    project_id=self.project.id,
                    session_id=state.session_id,
                    run_id=state.run_id,
                )
                self._progress(
                    "model.responded",
                    state,
                    round=round_number,
                    tool_call_count=len(tool_calls),
                    dropped_tool_call_count=dropped_tool_call_count,
                    discarded_protocol_tool_call_count=protocol_discarded_tool_calls,
                )
                if not tool_calls:
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
                                f"[Deep Agent rejected {protocol_violation} returned as answer text; "
                                "no tool was executed]"
                            )
                        final = ""
                        completion_issue = (
                            f"the model returned unusable {protocol_violation}; no tool call was accepted"
                        )
                    else:
                        completion_issue = self._completion_issue(state, final)
                    if completion_issue and corrective_rounds < max_corrective_rounds:
                        corrective_rounds += 1
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
                        continue
                    if completion_issue:
                        final = self._incomplete_answer(state, completion_issue, substantive=final)
                        state.fail(f"completion gate: {completion_issue}", final)
                        messages.append({"role": "assistant", "content": final})
                        self._finalize_session(state, messages)
                        self._publish_terminal("task.failed", state, final=final, error=state.error)
                        return final
                    state.complete(final)
                    self._finalize_session(state, messages)
                    self._publish_terminal("task.finished", state, final=final)
                    return final

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
                                f"limit of {_MAX_TOOL_CALLS_PROTOCOL}. They were not executed or written to Agent "
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
                        request.request_id in validation_request_ids
                        and not bool((result.data or {}).get("not_executed"))
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
                if made_progress:
                    corrective_rounds = 0
                tool_turn += 1
                execution_evidence_issue = self._execution_evidence_issue(state)
                if tool_turn >= soft_tool_turn_target:
                    if not execution_evidence_issue:
                        state.convergence.pop("soft_target_evidence_issue", None)
                        loop_exit_reason = "soft_target"
                        break
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

            synthesis = self._final_synthesis(
                state,
                messages,
                model_route=model_route,
                strategy=strategy,
                history_compactor=history_compactor,
                context_window=context_window,
                auto_compaction_enabled=auto_compaction_enabled,
                auto_compaction_max_tokens=auto_compaction_max_tokens,
            )
            completion_issue = self._completion_issue(state, synthesis)
            if synthesis:
                if not completion_issue:
                    state.complete(synthesis)
                    messages.append({"role": "assistant", "content": synthesis})
                    self._finalize_session(state, messages)
                    self._publish_terminal("task.finished", state, final=synthesis)
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
            elif loop_exit_reason == "soft_target":
                incomplete_reason = (
                    "the soft tool-turn target was reached, but the completion gate still reports: " + completion_issue
                )
            else:
                incomplete_reason = "the hard tool-turn limit was reached before verified completion"
                if completion_issue:
                    incomplete_reason += ": " + completion_issue
            final = self._incomplete_answer(state, incomplete_reason, substantive=synthesis)
            state.fail(f"{loop_exit_reason} reached: {completion_issue or incomplete_reason}", final)
            messages.append({"role": "assistant", "content": final})
            self._finalize_session(state, messages)
            self._publish_terminal("task.failed", state, final=final, error=state.error)
            return final
        except Exception as exc:
            if isinstance(exc, EventDispatchError) and exc.event_name == SESSION_FINALIZE_REQUESTED:
                raise
            if isinstance(exc, EventDispatchError) and exc.event_name == SESSION_CHECKPOINT_REQUESTED:
                state.fail(str(exc))
                if exc.subscriber_succeeded("session.checkpoint-writer"):
                    self._persist_failed_terminal(state, messages)
                raise
            if isinstance(exc, DeepSeekStreamInterrupted):
                state.record_model_response(ChatResponse(message={}, raw={}, http_attempt_count=exc.http_attempt_count))
                state.fail(f"resumable interruption: {exc}")
                if state.execution_context:
                    state.execution_context.prompt_phase = "interrupted"
                self._persist_failed_terminal(state, messages)
                raise RuntimeError(f"{exc} Session: {state.session_id}") from exc
            http_attempt_count = getattr(exc, "http_attempt_count", 0)
            if (
                isinstance(http_attempt_count, int)
                and not isinstance(http_attempt_count, bool)
                and http_attempt_count > 0
            ):
                state.record_model_response(ChatResponse(message={}, raw={}, http_attempt_count=http_attempt_count))
            state.fail(str(exc))
            self._persist_failed_terminal(state, messages)
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
            run_id=state.run_id,
        )

    def _progress(self, event: str, state: AgentState, **payload: Any) -> None:
        if self.event_pipelines.progress is None:
            return
        self.events.publish(
            PROGRESS_UPDATED,
            {
                "value": {
                    "event": event,
                    "session_id": state.session_id,
                    "mode": (state.task_strategy or {}).get("mode", "standard"),
                    "task_type": (state.task_route or {}).get("task_type", "question"),
                    "model_tier": (state.model_route or {}).get("tier", "standard"),
                    **payload,
                }
            },
            project_id=self.project.id,
            session_id=state.session_id,
            run_id=state.run_id,
        )

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

    def _checkpoint_session(self, state: AgentState, messages: list[dict[str, Any]]) -> None:
        self.events.dispatch_required(
            SESSION_CHECKPOINT_REQUESTED,
            {"state": state, "messages": messages},
            project_id=self.project.id,
            session_id=state.session_id,
            run_id=state.run_id,
        )

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
        self.events.dispatch_required(
            SESSION_FINALIZE_REQUESTED,
            {"state": state, "messages": messages},
            project_id=self.project.id,
            session_id=state.session_id,
            run_id=state.run_id,
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

    @staticmethod
    def _failure_count(state: AgentState) -> int:
        failed_tools = sum(
            1
            for call in state.tool_calls[-20:]
            if isinstance(call, dict)
            and isinstance(call.get("result"), dict)
            and not bool(call["result"].get("success", False))
        )
        return min(10, max(state.failure_count, failed_tools + int(bool(state.error))))

    @staticmethod
    def _single_validation_requested(state: AgentState) -> bool:
        reasons = (state.task_route or {}).get("reasons")
        return isinstance(reasons, list) and "single-validation" in reasons

    @classmethod
    def _single_validation_used(cls, state: AgentState) -> bool:
        for item in state.tool_calls:
            if not isinstance(item, dict):
                continue
            request = item.get("request") if isinstance(item.get("request"), dict) else {}
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            data = result.get("data") if isinstance(result.get("data"), dict) else {}
            if data.get("runtime_denied") is True or data.get("not_executed") is True:
                continue
            capability = f"{request.get('tool', '')}.{request.get('action', '')}"
            if capability in _SINGLE_VALIDATION_CAPABILITIES:
                return True
            if capability == "shell.run" and cls._looks_like_validation_shell(request.get("args")):
                return True
        return False

    @classmethod
    def _is_validation_model_call(cls, function_name: str, arguments: str | dict[str, Any] | None) -> bool:
        return cls._validation_model_call_count(function_name, arguments) > 0

    @classmethod
    def _validation_model_call_count(cls, function_name: str, arguments: str | dict[str, Any] | None) -> int:
        if function_name in _SINGLE_VALIDATION_MODEL_FUNCTIONS:
            return 1
        if function_name != "shell_run":
            return 0
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (TypeError, json.JSONDecodeError):
            return 0
        if not isinstance(parsed, dict):
            return 0
        return cls._shell_command_validation_count(str(parsed.get("command") or ""))

    @classmethod
    def _looks_like_validation_shell(cls, arguments: str | dict[str, Any] | None) -> bool:
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (TypeError, json.JSONDecodeError):
            return False
        if not isinstance(parsed, dict):
            return False
        command = str(parsed.get("command") or "").strip()
        if not command:
            return False
        return cls._shell_command_contains_validation(command)

    @classmethod
    def _shell_command_contains_validation(cls, command: str) -> bool:
        return cls._shell_command_validation_count(command) > 0

    @classmethod
    def _shell_command_validation_count(cls, command: str) -> int:
        try:
            lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|")
            lexer.whitespace_split = True
            lexer.commenters = ""
            tokens = list(lexer)
        except ValueError:
            return False
        segments: list[list[str]] = [[]]
        for token in tokens:
            if token and set(token) <= {";", "&", "|"}:
                if segments[-1]:
                    segments.append([])
                continue
            segments[-1].append(token)
        return sum(cls._validation_argv_count(segment) for segment in segments if segment)

    @classmethod
    def _argv_is_validation(cls, args: list[str]) -> bool:
        return cls._validation_argv_count(args) > 0

    @classmethod
    def _validation_argv_count(cls, args: list[str]) -> int:
        if not args:
            return 0
        program = args[0].replace("\\", "/").rsplit("/", 1)[-1].casefold()
        rest = [item.casefold() for item in args[1:]]
        if program in {"bash", "dash", "sh", "zsh"}:
            command_index = next(
                (
                    index
                    for index, flag in enumerate(rest)
                    if flag == "-c" or (flag.startswith("-") and "c" in flag[1:])
                ),
                None,
            )
            if command_index is None or command_index + 2 > len(args) - 1:
                return 0
            return cls._shell_command_validation_count(args[command_index + 2])
        if program == "env":
            index = 1
            while index < len(args):
                item = args[index]
                option = item.casefold()
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", item):
                    index += 1
                    continue
                if option == "--":
                    index += 1
                    break
                if option in {"-i", "--ignore-environment", "-0", "--null"}:
                    index += 1
                    continue
                if option in {"-u", "--unset", "-c", "--chdir"}:
                    index += 2
                    continue
                if option.startswith(("--unset=", "--chdir=")):
                    index += 1
                    continue
                if option.startswith("-"):
                    return 0
                break
            return cls._validation_argv_count(args[index:])
        if program in {"command", "exec"}:
            index = 1
            while index < len(args) and args[index].startswith("-"):
                option = args[index].casefold()
                if option == "--":
                    index += 1
                    break
                if program == "command" and option == "-v":
                    return 0
                if program == "exec" and option == "-a":
                    index += 2
                else:
                    index += 1
            return cls._validation_argv_count(args[index:])
        if program == "timeout":
            index = 1
            while index < len(args) and args[index].startswith("-"):
                option = args[index].casefold()
                index += 2 if option in {"-k", "--kill-after", "-s", "--signal"} else 1
            if index >= len(args):
                return 0
            return cls._validation_argv_count(args[index + 1 :])
        if program == "uv":
            return cls._validation_argv_count(args[2:]) if rest and rest[0] == "run" else 0
        if program in {"tox", "nox"}:
            return 1
        if program == "make":
            return int(any(item in {"test", "tests", "check", "lint", "typecheck", "verify"} for item in rest))
        if program in {"npm", "pnpm", "yarn", "bun"}:
            command = cls._package_manager_command(rest)
            if not command:
                return 0
            if command[0] in {"test", "check", "lint", "typecheck"}:
                return 1
            return int(
                len(command) >= 2
                and command[0] in {"run", "run-script"}
                and command[1] in {"test", "check", "lint", "typecheck", "build"}
            )
        if re.fullmatch(r"python(?:3(?:\.\d+)?)?", program):
            index = 0
            while index < len(rest):
                option = rest[index]
                if option == "-m":
                    return int(index + 1 < len(rest) and rest[index + 1] in {"pytest", "mypy", "pyright"})
                if option in {"-c", "--check-hash-based-pycs"} or not option.startswith("-"):
                    return 0
                index += 2 if option in {"-w", "-x"} and index + 1 < len(rest) else 1
            return 0
        if program == "npx":
            return int(bool(rest) and rest[0] in {"tsc", "eslint", "jest", "vitest", "mocha"})
        if program == "cargo":
            return int(bool(rest) and rest[0] in {"test", "check", "clippy", "build"})
        if program == "go":
            return int(bool(rest) and rest[0] in {"test", "vet"})
        if program in {"mvn", "mvnw", "gradle", "gradlew"}:
            return int(any(item in {"test", "check", "verify"} for item in rest))
        if program == "git":
            return int(rest[:2] == ["diff", "--check"])
        return int(program in _VALIDATION_SHELL_PROGRAMS)

    @staticmethod
    def _package_manager_command(rest: list[str]) -> list[str]:
        value_options = {"--prefix", "--workspace", "--cwd", "--dir", "-c"}
        index = 0
        while index < len(rest) and rest[index].startswith("-"):
            option = rest[index]
            if option in value_options and index + 1 < len(rest):
                index += 2
            else:
                index += 1
        return rest[index:]

    @classmethod
    def _completion_issue(cls, state: AgentState, final: str) -> str:
        issues = list(cls._execution_evidence_issues(state))
        if not final.strip():
            issues.append("the model returned an empty final answer")
        else:
            lowered = final.strip().lower()
            progress_only = (
                "need to use" in lowered
                or "let me " in lowered
                or "i will " in lowered
                or "需要用" in final
                or "接下来" in final
                or "让我" in final
            )
            if progress_only and len(final) < 500:
                issues.append("the response is only a progress note and explicitly describes remaining work")
        return "; ".join(dict.fromkeys(issues))

    @classmethod
    def _execution_evidence_issue(cls, state: AgentState) -> str:
        """Return missing execution evidence independently of answer prose.

        The same gate controls ordinary completion and whether the soft tool
        target may close tool execution.  This prevents a model-authored plan
        status from sending an artifact or validation task into tool-free final
        synthesis before the corresponding tool evidence exists.
        """

        return "; ".join(cls._execution_evidence_issues(state))

    @classmethod
    def _execution_evidence_issues(cls, state: AgentState) -> tuple[str, ...]:
        """Return all independently actionable execution-evidence gaps."""

        issues: list[str] = []
        requires_plan = bool((state.task_route or {}).get("require_plan")) or bool(
            (state.task_strategy or {}).get("require_plan")
        )
        if requires_plan:
            if not state.plan:
                issues.append("the selected execution mode requires a Task Graph, but no plan exists")
            else:
                incomplete = [step.id for step in state.plan if not state.plan_step_satisfied(step)]
                if incomplete:
                    shown = ", ".join(incomplete[:5])
                    suffix = "..." if len(incomplete) > 5 else ""
                    issues.append(f"required Task Graph steps are still incomplete: {shown}{suffix}")
                else:
                    plan_evidence_issue = cls._plan_evidence_issue(state)
                    if plan_evidence_issue:
                        issues.append(plan_evidence_issue)
        reasons = set((state.task_route or {}).get("reasons") or [])
        if "single-validation" in reasons:
            executed = cls._executed_non_plan_calls(state)
            if not any(cls._recorded_call_is_validation(item) for item in executed):
                issues.append("the single-validation task has no executed validation attempt")
        artifact_issue = cls._artifact_evidence_issue(state, reasons)
        if artifact_issue:
            issues.append(artifact_issue)
        return tuple(dict.fromkeys(issues))

    @classmethod
    def _artifact_evidence_issue(cls, state: AgentState, reasons: set[str]) -> str:
        if "artifact-required" not in reasons:
            return ""

        route = TaskRoute.from_dict(state.task_route or {})
        successful = cls._successful_tool_records(state)
        directory_writes = [
            item
            for item in successful
            if str((item.get("request") or {}).get("tool") or "") == "template"
            and str((item.get("request") or {}).get("action") or "") == "make_dir"
        ]
        if "directory-artifact-required" in reasons and not directory_writes:
            return "the requested output directory has no successful managed make_dir evidence"
        unmatched_directories = [
            hint
            for hint in route.directory_hints
            if not any(
                cls._same_recorded_path(
                    state,
                    hint,
                    str(((item.get("result") or {}).get("data") or {}).get("path") or ""),
                )
                for item in directory_writes
            )
        ]
        if unmatched_directories:
            return "the requested output directory has no successful managed make_dir evidence matching: " + ", ".join(
                unmatched_directories[:8]
            )
        directory_hints = tuple(hint for hint in route.artifact_hints if "." not in hint)
        if directory_hints:
            unmatched_directories = [
                hint
                for hint in directory_hints
                if not any(
                    cls._artifact_hint_matches_path(
                        hint,
                        str(((item.get("result") or {}).get("data") or {}).get("path") or "")
                        or str(((item.get("request") or {}).get("args") or {}).get("path") or ""),
                    )
                    for item in directory_writes
                )
            ]
            if unmatched_directories:
                return (
                    "the requested output directory has no successful managed make_dir evidence matching: "
                    + ", ".join(unmatched_directories[:8])
                )

        active_applies = [
            item
            for item in cls._active_file_applies(state)
            if item["after_exists"] is True or (item["after_exists"] is None and route.schema_version < 2)
        ]
        file_hints = tuple(hint for hint in route.artifact_hints if hint not in directory_hints)
        needs_file_artifact = (
            "directory-artifact-required" not in reasons or bool(file_hints) or "word-artifact-required" in reasons
        )
        if not needs_file_artifact:
            return ""
        if not active_applies:
            return "the requested output artifact has no active successful managed-write evidence"

        unmatched_hints = [
            hint
            for hint in file_hints
            if not any(cls._artifact_hint_matches_path(hint, str(item["path"])) for item in active_applies)
        ]
        if unmatched_hints:
            return (
                "the requested output artifact has no active successful managed-write evidence matching: "
                + ", ".join(unmatched_hints[:8])
            )
        if "word-artifact-required" not in reasons:
            return ""

        word_hints = tuple(hint for hint in route.artifact_hints if hint.lower().endswith(".docx"))
        word_applies: list[dict[str, Any]] = []
        if word_hints:
            for hint in word_hints:
                matching = [
                    item
                    for item in active_applies
                    if cls._artifact_hint_matches_path(hint, str(item["path"]))
                    and str(item["path"]).lower().endswith(".docx")
                ]
                if matching:
                    word_applies.append(max(matching, key=lambda item: (item["round"], item["index"])))
        else:
            matching = [item for item in active_applies if str(item["path"]).lower().endswith(".docx")]
            if matching:
                word_applies.append(max(matching, key=lambda item: (item["round"], item["index"])))
        if not word_applies:
            return "the requested Word artifact has no active applied .docx preview"

        seen_applies: set[tuple[str, str, str]] = set()
        for applied in word_applies:
            identity = (str(applied["path"]), str(applied["preview_id"]), str(applied["snapshot_id"]))
            if identity in seen_applies:
                continue
            seen_applies.add(identity)
            issue = cls._word_artifact_evidence_issue(
                state,
                applied=applied,
                successful=successful,
                route_schema=route.schema_version,
            )
            if issue:
                return issue
        return ""

    @classmethod
    def _word_artifact_evidence_issue(
        cls,
        state: AgentState,
        *,
        applied: dict[str, Any],
        successful: list[dict[str, Any]],
        route_schema: int,
    ) -> str:
        artifact_path = str(applied["path"])
        artifact_index = int(applied["index"])
        artifact_preview_id = str(applied["preview_id"] or "")
        previews: list[dict[str, Any]] = []
        for index, item in enumerate(state.tool_calls):
            if item not in successful:
                continue
            request = item.get("request") if isinstance(item.get("request"), dict) else {}
            if (str(request.get("tool") or ""), str(request.get("action") or "")) != ("document", "render_docx"):
                continue
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            data = result.get("data") if isinstance(result.get("data"), dict) else {}
            previews.append(
                {
                    "record": item,
                    "round": int(item.get("round") or 0),
                    "index": index,
                    "path": str(data.get("path") or ""),
                    "preview_id": str(data.get("preview_id") or ""),
                }
            )
        path_previews = [item for item in previews if cls._same_recorded_path(state, str(item["path"]), artifact_path)]
        if not path_previews:
            return "the requested Word artifact has no matching document_render_docx preview"
        if route_schema >= 2 and not artifact_preview_id:
            return "the requested Word artifact apply record is missing preview lineage"
        if artifact_preview_id:
            matching_previews = [item for item in path_previews if item["preview_id"] == artifact_preview_id]
        else:
            matching_previews = path_previews
        if not matching_previews:
            return "the requested Word artifact apply does not match its document_render_docx preview_id"
        latest_path_preview = max(path_previews, key=lambda item: (item["round"], item["index"]))
        if (
            latest_path_preview["preview_id"]
            and artifact_preview_id
            and latest_path_preview["preview_id"] != artifact_preview_id
        ) or (latest_path_preview["round"], latest_path_preview["index"]) > (
            int(applied["round"]),
            artifact_index,
        ):
            return "the latest generated document preview has not been applied"

        verified = any(
            index > artifact_index
            and str((item.get("request") or {}).get("tool") or "") == "document"
            and str((item.get("request") or {}).get("action") or "") == "parse"
            and bool((item.get("result") or {}).get("success"))
            and cls._same_recorded_path(
                state,
                str(((item.get("request") or {}).get("args") or {}).get("path") or ""),
                artifact_path,
            )
            for index, item in enumerate(state.tool_calls)
        )
        if not verified:
            return "the generated document has not been re-opened with document_parse"

        latest_render_record = max(matching_previews, key=lambda item: (item["round"], item["index"]))["record"]
        render_result = (
            latest_render_record.get("result") if isinstance(latest_render_record.get("result"), dict) else {}
        )
        render_data = render_result.get("data") if isinstance(render_result.get("data"), dict) else {}
        generated_date_values = {
            str(item) for item in render_data.get("generated_metadata_dates", []) if str(item).strip()
        }
        if not generated_date_values:
            render_request = (
                latest_render_record.get("request") if isinstance(latest_render_record.get("request"), dict) else {}
            )
            render_args = render_request.get("args") if isinstance(render_request.get("args"), dict) else {}
            markdown = str(render_args.get("markdown") or "")
            generated_date_values = {
                date
                for line in markdown.splitlines()
                if ("生成" in line or "汇总" in line or "报告" in line) and ("时间" in line or "日期" in line)
                for date in _DATE_LITERAL_RE.findall(line)
            }
        generated_dates = {key for value in generated_date_values if (key := _date_key(value)) is not None}
        invalid_generated_dates = {value for value in generated_date_values if _date_key(value) is None}
        allowed_dates = _date_keys_from_text(state.objective + "\n" + state.user_request)
        allowed_sources = {
            ("document", "parse"),
            ("ocr", "parse"),
            ("template", "read_file"),
        }
        for index, item in enumerate(state.tool_calls):
            if index >= artifact_index:
                break
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            if not bool(result.get("success")):
                continue
            request = item.get("request") if isinstance(item.get("request"), dict) else {}
            source = (str(request.get("tool") or ""), str(request.get("action") or ""))
            if source not in allowed_sources:
                continue
            data = result.get("data") if isinstance(result.get("data"), dict) else {}
            allowed_dates.update(_date_keys_from_text(str(data.get("date_literals") or "")))
            allowed_dates.update(_date_keys_from_text(str(result.get("stdout") or "")))
        unsupported_date_keys = generated_dates - allowed_dates
        unsupported_dates = sorted(
            invalid_generated_dates
            | {
                value
                for value in generated_date_values
                if (key := _date_key(value)) is not None and key in unsupported_date_keys
            }
        )
        if unsupported_dates:
            return (
                "the generated document contains unsupported generation-date metadata not present in the "
                "request or source documents: " + ", ".join(unsupported_dates[:5])
            )
        return ""

    @staticmethod
    def _artifact_hint_matches_path(hint: str, path: str) -> bool:
        normalized_hint = str(hint).strip()
        normalized_path = str(path).strip().replace("\\", "/").rstrip("/")
        if not normalized_hint or not normalized_path:
            return False
        basename = normalized_path.rsplit("/", maxsplit=1)[-1]
        if normalized_hint.startswith("."):
            return basename.lower().endswith(normalized_hint.lower())
        return basename == normalized_hint

    @staticmethod
    def _successful_tool_records(state: AgentState) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for item in state.tool_calls:
            if not isinstance(item, dict):
                continue
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            data = result.get("data") if isinstance(result.get("data"), dict) else {}
            if bool(result.get("success")) and not bool(data.get("not_executed")):
                records.append(item)
        return records

    @classmethod
    def _active_file_applies(cls, state: AgentState) -> list[dict[str, Any]]:
        successful = cls._successful_tool_records(state)
        undone_snapshots = {
            str(((item.get("result") or {}).get("data") or {}).get("snapshot_id") or "")
            for item in successful
            if str((item.get("request") or {}).get("tool") or "") == "file"
            and str((item.get("request") or {}).get("action") or "") == "undo"
        }
        undone_snapshots.discard("")
        applies: list[dict[str, Any]] = []
        for index, item in enumerate(state.tool_calls):
            if item not in successful:
                continue
            request = item.get("request") if isinstance(item.get("request"), dict) else {}
            if (str(request.get("tool") or ""), str(request.get("action") or "")) != ("file", "apply"):
                continue
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            data = result.get("data") if isinstance(result.get("data"), dict) else {}
            request_args = request.get("args") if isinstance(request.get("args"), dict) else {}
            snapshot_id = str(data.get("snapshot_id") or "")
            if snapshot_id and snapshot_id in undone_snapshots:
                continue
            applies.append(
                {
                    "record": item,
                    "round": int(item.get("round") or 0),
                    "index": index,
                    "path": str(data.get("path") or ""),
                    "preview_id": str(data.get("preview_id") or request_args.get("preview_id") or ""),
                    "snapshot_id": snapshot_id,
                    "before_exists": data.get("before_exists") if isinstance(data.get("before_exists"), bool) else None,
                    "after_exists": data.get("after_exists") if isinstance(data.get("after_exists"), bool) else None,
                }
            )
        latest_by_path: dict[Path, dict[str, Any]] = {}
        for item in applies:
            path_key = cls._normalized_recorded_path(state, str(item["path"]))
            if path_key is not None:
                latest_by_path[path_key] = item
        return sorted(latest_by_path.values(), key=lambda item: int(item["index"]))

    @classmethod
    def _plan_evidence_issue(cls, state: AgentState) -> str:
        executed = cls._executed_non_plan_calls(state)
        if not executed:
            return "the completed Task Graph has no executed non-plan tool evidence"
        return ""

    @staticmethod
    def _executed_non_plan_calls(state: AgentState) -> list[dict[str, Any]]:
        executed: list[dict[str, Any]] = []
        for item in state.tool_calls:
            if not isinstance(item, dict):
                continue
            request = item.get("request") if isinstance(item.get("request"), dict) else {}
            if not request or str(request.get("tool") or "") == "agent":
                continue
            result = item.get("result") if isinstance(item.get("result"), dict) else {}
            data = result.get("data") if isinstance(result.get("data"), dict) else {}
            if bool(data.get("runtime_denied")) or bool(data.get("not_executed")):
                continue
            executed.append(item)
        return executed

    @classmethod
    def _recorded_call_is_validation(cls, item: dict[str, Any]) -> bool:
        request = item.get("request") if isinstance(item.get("request"), dict) else {}
        capability = f"{request.get('tool', '')}.{request.get('action', '')}"
        if capability in _SINGLE_VALIDATION_CAPABILITIES:
            return True
        return capability == "shell.run" and cls._looks_like_validation_shell(request.get("args"))

    @staticmethod
    def _normalized_recorded_path(state: AgentState, value: str) -> Path | None:
        raw = str(value).strip().replace("\\", "/")
        if not raw:
            return None
        path = Path(raw)
        if not path.is_absolute():
            root = Path(str((state.project or {}).get("root") or state.working_directory))
            path = root / path
        return path.resolve(strict=False)

    @classmethod
    def _same_recorded_path(cls, state: AgentState, left: str, right: str) -> bool:
        left_value = cls._normalized_recorded_path(state, left)
        right_value = cls._normalized_recorded_path(state, right)
        if left_value is None or right_value is None:
            return False
        return left_value == right_value

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
                self._checkpoint_convergence_transition(
                    state,
                    messages,
                    transition=f"overflow_{transition}",
                    phase=request_phase,
                    counter="overflow_recovery_count",
                )
                chat_kwargs["messages"] = messages
                state.record_model_request(request_phase)

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
        if response.finish_reason != "length":
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
            state.record_model_request(request_phase)
            self._checkpoint_convergence_transition(
                state,
                continuation_messages,
                transition="length_continuation",
                phase=request_phase,
                counter="length_continuation_count",
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
        state.record_model_request("context_compaction")
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
        except EventDispatchError:
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
        state.record_model_request("final_synthesis")
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
            "max_tokens": (
                context_window.effective_output_tokens(model_route.max_tokens)
                if context_window is not None
                else model_route.max_tokens
            ),
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
                content=reasoning[: int(self.config.get("runtime.max_reasoning_display_chars", 4000))],
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
        try:
            parsed = int(self.config.get(dotted, default))
        except (TypeError, ValueError, OverflowError):
            parsed = default
        return max(minimum, min(parsed, maximum))
