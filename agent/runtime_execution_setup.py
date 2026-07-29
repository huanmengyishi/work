from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .context import ContextSnapshot
from .convergence import ContextWindowController, TaskConvergenceController, ToolHistoryCompactor
from .events import EventDispatchError
from .model_router import ModelRoute
from .state import AgentState
from .task_strategy import TaskStrategy


@dataclass(frozen=True)
class ExecutionSetup:
    """Immutable controls shared by every model/tool round in one turn."""

    snapshot: ContextSnapshot
    strategy: TaskStrategy
    model_route: ModelRoute
    soft_tool_turn_target: int
    hard_tool_turn_limit: int
    max_corrective_rounds: int
    max_abnormal_finish_recoveries: int
    recovery_char_limit: int
    convergence: TaskConvergenceController
    history_compactor: ToolHistoryCompactor | None
    round_compactor: ToolHistoryCompactor | None
    context_window: ContextWindowController
    auto_compaction_enabled: bool
    auto_compaction_max_tokens: int
    single_tool_result_chars: int
    max_tool_calls_per_round: int


@dataclass
class ExecutionLoopState:
    """Small mutable cursor for the orchestrator; durable truth stays in AgentState."""

    tool_turn: int = 0
    model_round: int = 0
    corrective_rounds: int = 0
    abnormal_finish_recoveries: int = 0
    loop_exit_reason: str = "hard_limit"
    recovery_injected: set[int] = field(default_factory=set)
    recovery_chars_used: int = 0


class RuntimeExecutionSetupMixin:
    def _initialize_execution(
        self,
        state: AgentState,
        messages: list[dict[str, Any]],
        *,
        snapshot: ContextSnapshot,
    ) -> tuple[ExecutionSetup, ExecutionLoopState]:
        self.tools.bind_state(state)
        state.start()
        self.execution_budget.bind(state)
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
        adaptive_exploration_limit = self._adaptive_exploration_limit(state)
        convergence = TaskConvergenceController(
            mode=strategy.mode if convergence_enabled else "standard",
            max_rounds=soft_tool_turn_target,
            exploration_round_limit=adaptive_exploration_limit,
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
        history_compactor, round_compactor = self._build_history_compactors(
            enabled=convergence_enabled,
            keep_recent_rounds=keep_recent_rounds,
            failure_limit=compaction_failure_limit,
            compacted_result_chars=compacted_result_chars,
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

        setup = ExecutionSetup(
            snapshot=snapshot,
            strategy=strategy,
            model_route=model_route,
            soft_tool_turn_target=soft_tool_turn_target,
            hard_tool_turn_limit=hard_tool_turn_limit,
            max_corrective_rounds=self.resilience.max_corrective_rounds,
            max_abnormal_finish_recoveries=self.resilience.max_abnormal_finish_recoveries,
            recovery_char_limit=recovery_char_limit,
            convergence=convergence,
            history_compactor=history_compactor,
            round_compactor=round_compactor,
            context_window=context_window,
            auto_compaction_enabled=convergence_enabled
            and bool(self.config.get("runtime.convergence.auto_compaction_enabled", True)),
            auto_compaction_max_tokens=self._bounded_config_int(
                "runtime.convergence.auto_compaction_max_tokens",
                2_048,
                minimum=256,
                maximum=20_000,
            ),
            single_tool_result_chars=self._bounded_config_int(
                "runtime.convergence.single_tool_result_chars",
                12_000,
                minimum=512,
                maximum=100_000,
            ),
            max_tool_calls_per_round=self._bounded_config_int(
                "runtime.convergence.max_tool_calls_per_round",
                16,
                minimum=1,
                maximum=64,
            ),
        )
        return setup, ExecutionLoopState()

    def _adaptive_exploration_limit(self, state: AgentState) -> int:
        adjustment = state.convergence.get("strategy_adjustment")
        limit = (
            int(adjustment.get("exploration_round_limit"))
            if isinstance(adjustment, dict) and isinstance(adjustment.get("exploration_round_limit"), int)
            else self._bounded_config_int(
                "runtime.convergence.max_consecutive_exploration_rounds",
                6,
                minimum=2,
                maximum=32,
            )
        )
        experiment = state.convergence.get("experiment")
        parameters = experiment.get("parameters") if isinstance(experiment, dict) else None
        if isinstance(parameters, dict) and isinstance(parameters.get("exploration_round_limit"), int):
            return max(2, min(32, int(parameters["exploration_round_limit"])))
        return limit

    def _build_history_compactors(
        self,
        *,
        enabled: bool,
        keep_recent_rounds: int,
        failure_limit: int,
        compacted_result_chars: int,
    ) -> tuple[ToolHistoryCompactor | None, ToolHistoryCompactor | None]:
        if not enabled:
            return None, None
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
            failure_limit=failure_limit,
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
            failure_limit=failure_limit,
        )
        return history_compactor, round_compactor


__all__ = [
    "ExecutionLoopState",
    "ExecutionSetup",
    "RuntimeExecutionSetupMixin",
]
