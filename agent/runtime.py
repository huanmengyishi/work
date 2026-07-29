from __future__ import annotations

from typing import Any, Callable

from .artifact_registry import ArtifactRegistry
from .adaptive import StrategyAdjuster
from .budget import ExecutionBudgetController
from .config import AppConfig
from .contracts import (
    ContextBuilderProtocol,
    ModelClientProtocol,
    PromptBuilderProtocol,
    RuntimeToolsProtocol,
    SessionStoreProtocol,
)
from .context import ContextBuilder
from .event_pipelines import (
    RuntimeEventPipelines,
)
from .events import EventBus
from .experiments import ExperimentRunner
from .memory import MemoryStore
from .memory_refinement import MemoryRefiner
from .model_router import ModelRoute, ModelRouter, more_capable_model_route
from .optimizer import PerformanceHistory
from .paths import storage_key
from .project import Project
from .resilience import CapabilityRecoveryController, ResiliencePolicy
from .prompt import PromptBuilder
from .session import SessionManager
from .state import AgentState
from .task_plan import TaskPlanFactory
from .task_router import TaskRoute, TaskRouter, more_capable_task_route
from .unicode_text import normalize_unicode_text
from .runtime_compaction import RuntimeCompactionMixin
from .runtime_context import RuntimeContextMixin
from .runtime_execution import RuntimeExecutionMixin
from .runtime_lifecycle import RuntimeLifecycleMixin
from .runtime_support import _normalize_assistant_tool_calls
from .runtime_tool_batch import RuntimeToolBatchMixin
from .runtime_synthesis import RuntimeSynthesisMixin
from .runtime_validation import RuntimeValidationMixin

__all__ = ["AgentRuntime", "_normalize_assistant_tool_calls"]


class AgentRuntime(
    RuntimeExecutionMixin,
    RuntimeToolBatchMixin,
    RuntimeLifecycleMixin,
    RuntimeValidationMixin,
    RuntimeContextMixin,
    RuntimeCompactionMixin,
    RuntimeSynthesisMixin,
):
    def __init__(
        self,
        *,
        config: AppConfig,
        project: Project,
        memory: MemoryStore,
        tools: RuntimeToolsProtocol,
        client: ModelClientProtocol,
        events: EventBus,
        context_builder: ContextBuilderProtocol,
        prompt_builder: PromptBuilderProtocol,
        sessions: SessionStoreProtocol,
        progress_handler: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.config = config
        self.project = project
        self.memory = memory
        self.tools = tools
        self.client = client
        self.events = events
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder
        self.sessions = sessions
        self.task_router = TaskRouter(config)
        self.model_router = ModelRouter(config)
        self.task_plan_factory = TaskPlanFactory()
        self.execution_budget = ExecutionBudgetController(config)
        self.resilience = ResiliencePolicy.from_config(config)
        self.capability_recovery = CapabilityRecoveryController(self.resilience)
        self.memory_refiner = MemoryRefiner(config)
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
        performance_history = (
            self.event_pipelines.performance.history
            if self.event_pipelines.performance is not None
            else PerformanceHistory(
                config.data_dir / "performance" / f"{storage_key(project.id)}.db",
                max_records=config.get("events.performance_history_max_records", 200),
            )
        )
        self.strategy_adjuster = StrategyAdjuster(config, performance_history, project_id=project.id)
        self.experiment_runner = ExperimentRunner(config, project_id=project.id)
        self.experiment_runner.attach(self.events)

    @classmethod
    def with_default_services(
        cls,
        *,
        config: AppConfig,
        project: Project,
        memory: MemoryStore,
        tools: RuntimeToolsProtocol,
        client: ModelClientProtocol,
        events: EventBus | None = None,
        context_builder: ContextBuilderProtocol | None = None,
        prompt_builder: PromptBuilderProtocol | None = None,
        sessions: SessionStoreProtocol | None = None,
        progress_handler: Callable[[dict[str, Any]], None] | None = None,
    ) -> AgentRuntime:
        """Explicit application/test composition root for standard services.

        The constructor itself never creates lifecycle dependencies. Callers
        that want the built-in implementations opt into them through this
        named factory; integration tests may inject structural protocol fakes
        directly into ``AgentRuntime(...)``.
        """

        return cls(
            config=config,
            project=project,
            memory=memory,
            tools=tools,
            client=client,
            events=events if events is not None else EventBus(),
            context_builder=(context_builder if context_builder is not None else ContextBuilder(config)),
            prompt_builder=prompt_builder if prompt_builder is not None else PromptBuilder(),
            sessions=sessions if sessions is not None else SessionManager(project),
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
        # Memory retrieval is auxiliary context. Project the lookup query to
        # Memory's own input budget while preserving the complete user request
        # for the authoritative Context Window check below.
        memory_items = self.memory.search(
            prompt,
            self.project.id,
            record_usage=False,
            truncate_query=True,
        )
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
        assignment = self.experiment_runner.assign(run_id=state.run_id, task_type=task_route.task_type)
        if assignment is not None:
            state.convergence["experiment"] = assignment.to_dict()
            tier = str(assignment.parameters.get("model_tier") or "")
            if tier:
                model_route = self.model_router.route(task_route, explicit_tier=tier)
        model_route = self._adjust_strategy(state, task_route, model_route)
        strategy = self._strategy_from_routes(task_route, model_route)
        state.task_route = task_route.to_dict()
        state.model_route = model_route.to_dict()
        state.task_strategy = strategy.to_dict()
        plan = initial_plan or self.task_plan_factory.build(task_route)
        if plan:
            self.tools.plan_manager.replace(state, plan)
        ArtifactRegistry.sync_planned(state)
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
        record = self.sessions.load_for_resume(
            session_id,
            max_rounds=self._bounded_config_int(
                "runtime.resume_window_rounds",
                16,
                minimum=1,
                maximum=128,
            ),
        )
        state = record.state
        if str(state.project.get("id") or "") != self.project.id:
            raise ValueError("saved session belongs to a different project")
        failure_count = self._failure_count(state)
        context = self.context_builder.build(self.project)
        memory_items = self.memory.search(
            prompt,
            self.project.id,
            record_usage=False,
            truncate_query=True,
        )
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
        assignment = self.experiment_runner.assign(run_id=state.run_id, task_type=task_route.task_type)
        if assignment is not None:
            state.convergence["experiment"] = assignment.to_dict()
            tier = str(assignment.parameters.get("model_tier") or "")
            if tier:
                candidate_model_route = self.model_router.route(task_route, explicit_tier=tier)
        candidate_model_route = self._adjust_strategy(state, task_route, candidate_model_route)
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
        ArtifactRegistry.sync_planned(state)
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
