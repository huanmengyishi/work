from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from typing import Any, ClassVar

from .contracts import (
    AGENT_STATE_FROZEN_FIELDS,
    AGENT_STATE_SCHEMA_VERSION,
    AGENT_STATE_SERIALIZED_FIELDS,
)
from .model_router import COST_CLASSES, MODEL_TIERS
from .project import Project
from .task_router import TASK_MODES, TASK_RISKS, TASK_SCALES, TASK_TYPES
from .timeutil import utc_now_iso


PLAN_STEP_STATUSES = frozenset({"pending", "in_progress", "completed", "failed", "skipped"})
AGENT_STATUSES = frozenset({"initialized", "running", "completed", "failed"})
PROMPT_PHASES = frozenset({"initial", "running", "resumed", "completed", "failed", "interrupted"})
CONTEXT_PHASES = frozenset({"initial", "resume", "recovery"})


@dataclass
class PlanStep:
    id: str
    title: str
    status: str = "pending"
    description: str = ""
    dependencies: list[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 0
    allow_parallel: bool = False
    completion_criteria: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PlanStep":
        return cls(
            id=str(value.get("id") or "step"),
            title=str(value.get("title") or ""),
            status=str(value.get("status") or "pending"),
            description=str(value.get("description") or ""),
            dependencies=[str(item) for item in value.get("dependencies", [])],
            retry_count=max(0, int(value.get("retry_count") or 0)),
            max_retries=max(0, int(value.get("max_retries") or 0)),
            allow_parallel=bool(value.get("allow_parallel", False)),
            completion_criteria=str(value.get("completion_criteria") or ""),
        )

    def validate(self) -> "PlanStep":
        _non_empty_string(self.id, "plan step id")
        _non_empty_string(self.title, f"plan step {self.id} title")
        if len(self.id) > 80:
            raise ValueError(f"plan step id is too long: {self.id}")
        if self.status not in PLAN_STEP_STATUSES:
            raise ValueError(f"invalid plan step status for {self.id}: {self.status}")
        if not _is_non_negative_int(self.retry_count):
            raise ValueError(f"plan step retry_count must be a non-negative integer: {self.id}")
        if not _is_non_negative_int(self.max_retries):
            raise ValueError(f"plan step max_retries must be a non-negative integer: {self.id}")
        if self.retry_count > self.max_retries and self.status == "pending":
            raise ValueError(f"pending plan step retry_count exceeds max_retries: {self.id}")
        if not isinstance(self.dependencies, list) or not all(
            isinstance(item, str) and item for item in self.dependencies
        ):
            raise ValueError(f"plan step dependencies must be non-empty strings: {self.id}")
        if len(self.dependencies) != len(set(self.dependencies)):
            raise ValueError(f"plan step dependencies must be unique: {self.id}")
        if self.id in self.dependencies:
            raise ValueError(f"plan step cannot depend on itself: {self.id}")
        return self


@dataclass
class ExecutionContext:
    current_directory: str
    git_branch: str | None = None
    modified_files: list[str] = field(default_factory=list)
    current_plan_id: str | None = None
    current_queue_id: str | None = None
    recent_tool: str | None = None
    recent_error: str = ""
    current_snapshot: str | None = None
    prompt_phase: str = "initial"
    last_checkpoint_at: str = field(default_factory=utc_now_iso)

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, working_directory: str, git_branch: str | None) -> "ExecutionContext":
        return cls(
            current_directory=str(value.get("current_directory") or working_directory),
            git_branch=value.get("git_branch", git_branch),
            modified_files=[str(item) for item in value.get("modified_files", [])],
            current_plan_id=value.get("current_plan_id"),
            current_queue_id=value.get("current_queue_id"),
            recent_tool=value.get("recent_tool"),
            recent_error=str(value.get("recent_error") or ""),
            current_snapshot=value.get("current_snapshot"),
            prompt_phase=str(value.get("prompt_phase") or "initial"),
            last_checkpoint_at=str(value.get("last_checkpoint_at") or utc_now_iso()),
        )

    def validate(self) -> "ExecutionContext":
        _non_empty_string(self.current_directory, "execution_context.current_directory")
        if self.prompt_phase not in PROMPT_PHASES:
            raise ValueError(f"invalid execution_context.prompt_phase: {self.prompt_phase}")
        if not isinstance(self.modified_files, list) or not all(
            isinstance(item, str) and item for item in self.modified_files
        ):
            raise ValueError("execution_context.modified_files must contain non-empty strings")
        if len(self.modified_files) != len(set(self.modified_files)):
            raise ValueError("execution_context.modified_files must be unique")
        _validate_iso_timestamp(self.last_checkpoint_at, "execution_context.last_checkpoint_at")
        return self


@dataclass
class AgentState:
    """Serializable runtime state with a versioned, validation-backed schema.

    ``FROZEN_FIELDS`` identify the Session identity that Resume must preserve.
    Mutable execution fields (request, routes, plan, progress, and result) may
    evolve between turns after ``validate_frozen_fields`` succeeds.
    """

    SCHEMA_VERSION: ClassVar[int] = AGENT_STATE_SCHEMA_VERSION
    SERIALIZED_FIELDS: ClassVar[tuple[str, ...]] = AGENT_STATE_SERIALIZED_FIELDS
    FROZEN_FIELDS: ClassVar[tuple[str, ...]] = AGENT_STATE_FROZEN_FIELDS
    _frozen_baseline: dict[str, Any] = field(init=False, repr=False, compare=False)

    session_id: str
    project: dict[str, Any]
    objective: str
    user_request: str
    request_history: list[str]
    working_directory: str
    status: str = "initialized"
    plan: list[PlanStep] = field(default_factory=list)
    current_step: str | None = None
    completed_steps: list[str] = field(default_factory=list)
    loaded_memories: list[int] = field(default_factory=list)
    loaded_tools: list[str] = field(default_factory=list)
    git_branch: str | None = None
    context_index_path: str | None = None
    execution_context: ExecutionContext | None = None
    task_strategy: dict[str, Any] = field(default_factory=dict)
    task_route: dict[str, Any] = field(default_factory=dict)
    model_route: dict[str, Any] = field(default_factory=dict)
    context_manifest: dict[str, Any] = field(default_factory=dict)
    convergence: dict[str, Any] = field(default_factory=dict)
    model_metrics: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    round: int = 0
    model_request_count: int = 0
    main_loop_model_request_count: int = 0
    context_compaction_model_request_count: int = 0
    final_synthesis_model_request_count: int = 0
    turn: int = 1
    final_answer: str = ""
    error: str = ""
    failure_count: int = 0
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    schema_version: int = AGENT_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self._frozen_baseline = self.frozen_values()

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        project: Project,
        user_request: str,
        loaded_memories: list[int],
        loaded_tools: list[str],
        git_branch: str | None,
        context_index_path: str,
    ) -> "AgentState":
        return cls(
            session_id=session_id,
            project={
                "id": project.id,
                "name": project.name,
                "root": str(project.root),
                "language": project.language,
            },
            objective=user_request,
            user_request=user_request,
            request_history=[user_request],
            working_directory=str(project.root),
            loaded_memories=loaded_memories,
            loaded_tools=loaded_tools,
            git_branch=git_branch,
            context_index_path=context_index_path,
            execution_context=ExecutionContext(
                current_directory=str(project.root),
                git_branch=git_branch,
            ),
        ).validate()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AgentState":
        if not isinstance(value, dict):
            raise TypeError("AgentState record must be a dictionary")
        plan = [PlanStep.from_dict(item) for item in value.get("plan", []) if isinstance(item, dict)]
        working_directory = str(value.get("working_directory") or "")
        git_branch = value.get("git_branch")
        execution_value = value.get("execution_context")
        execution_context = ExecutionContext.from_dict(
            execution_value if isinstance(execution_value, dict) else {},
            working_directory=working_directory,
            git_branch=git_branch,
        )
        state = cls(
            session_id=str(value.get("session_id") or ""),
            project=dict(value.get("project") or {}),
            objective=str(value.get("objective") or value.get("user_request") or ""),
            user_request=str(value.get("user_request") or ""),
            request_history=[str(item) for item in value.get("request_history", []) if str(item).strip()]
            or [str(value.get("user_request") or "")],
            working_directory=working_directory,
            status=str(value.get("status") or "initialized"),
            plan=plan,
            current_step=value.get("current_step"),
            completed_steps=[str(item) for item in value.get("completed_steps", [])],
            loaded_memories=[int(item) for item in value.get("loaded_memories", [])],
            loaded_tools=[str(item) for item in value.get("loaded_tools", [])],
            git_branch=git_branch,
            context_index_path=value.get("context_index_path"),
            execution_context=execution_context,
            task_strategy=dict(value.get("task_strategy") or {}),
            task_route=dict(value.get("task_route") or {}),
            model_route=dict(value.get("model_route") or {}),
            context_manifest=dict(value.get("context_manifest") or {}),
            convergence=dict(value.get("convergence") or {}),
            model_metrics=dict(value.get("model_metrics") or {}),
            tool_calls=list(value.get("tool_calls") or []),
            round=int(value.get("round") or 0),
            model_request_count=max(0, int(value.get("model_request_count") or 0)),
            main_loop_model_request_count=max(0, int(value.get("main_loop_model_request_count") or 0)),
            context_compaction_model_request_count=max(
                0,
                int(value.get("context_compaction_model_request_count") or 0),
            ),
            final_synthesis_model_request_count=max(
                0,
                int(value.get("final_synthesis_model_request_count") or 0),
            ),
            turn=int(value.get("turn") or 1),
            final_answer=str(value.get("final_answer") or ""),
            error=str(value.get("error") or ""),
            failure_count=max(0, int(value.get("failure_count") or 0)),
            created_at=str(value.get("created_at") or utc_now_iso()),
            updated_at=str(value.get("updated_at") or utc_now_iso()),
            schema_version=max(1, int(value.get("schema_version") or 1)),
        )
        if state.schema_version < cls.SCHEMA_VERSION:
            state._normalize_legacy_derived_fields()
        state._frozen_baseline = state.frozen_values()
        return state.validate()

    def validate(self) -> "AgentState":
        """Validate serialized identity, routes, plan graph, and numeric bounds."""

        serialized_fields = tuple(item.name for item in fields(self) if item.init)
        if serialized_fields != self.SERIALIZED_FIELDS:
            raise ValueError("AgentState serialized field order does not match its interface contract")
        if not _is_positive_int(self.schema_version):
            raise ValueError("AgentState schema_version must be a positive integer")
        if self.schema_version > self.SCHEMA_VERSION:
            raise ValueError(
                f"unsupported AgentState schema_version {self.schema_version}; maximum is {self.SCHEMA_VERSION}"
            )
        _non_empty_string(self.session_id, "AgentState.session_id")
        _non_empty_string(self.objective, "AgentState.objective")
        _non_empty_string(self.user_request, "AgentState.user_request")
        _non_empty_string(self.working_directory, "AgentState.working_directory")
        if self.status not in AGENT_STATUSES:
            raise ValueError(f"invalid AgentState.status: {self.status}")
        if not isinstance(self.project, dict):
            raise ValueError("AgentState.project must be a dictionary")
        for key in ("id", "name", "root"):
            _non_empty_string(self.project.get(key), f"AgentState.project.{key}")
        if str(self.project["root"]) != self.working_directory:
            raise ValueError("AgentState.project.root must match working_directory")
        if not _is_non_negative_int(self.round):
            raise ValueError("AgentState.round must be a non-negative integer")
        for field_name in (
            "model_request_count",
            "main_loop_model_request_count",
            "context_compaction_model_request_count",
            "final_synthesis_model_request_count",
        ):
            if not _is_non_negative_int(getattr(self, field_name)):
                raise ValueError(f"AgentState.{field_name} must be a non-negative integer")
        if self.model_request_count != (
            self.main_loop_model_request_count
            + self.context_compaction_model_request_count
            + self.final_synthesis_model_request_count
        ):
            raise ValueError("AgentState.model_request_count must equal the sum of request phase counters")
        if not _is_positive_int(self.turn):
            raise ValueError("AgentState.turn must be a positive integer")
        if not _is_non_negative_int(self.failure_count):
            raise ValueError("AgentState.failure_count must be a non-negative integer")
        if not isinstance(self.plan, list) or not all(isinstance(step, PlanStep) for step in self.plan):
            raise ValueError("AgentState.plan must contain PlanStep records")
        self._validate_plan()
        self._validate_collections()
        self._validate_routes()
        self._validate_context_manifest()
        if not isinstance(self.convergence, dict):
            raise ValueError("AgentState.convergence must be a dictionary")
        implementation_reads_used = self.convergence.get("implementation_reads_used", 0)
        if not _is_non_negative_int(implementation_reads_used) or implementation_reads_used > 4:
            raise ValueError("AgentState.convergence.implementation_reads_used must be between 0 and 4")
        validation_attachment_reads_used = self.convergence.get("validation_attachment_reads_used", 0)
        if not _is_non_negative_int(validation_attachment_reads_used) or validation_attachment_reads_used > 4:
            raise ValueError("AgentState.convergence.validation_attachment_reads_used must be between 0 and 4")
        if not isinstance(self.model_metrics, dict):
            raise ValueError("AgentState.model_metrics must be a dictionary")
        for key in ("http_attempt_count", "prompt_tokens", "completion_tokens", "total_tokens"):
            if not _is_non_negative_int(self.model_metrics.get(key, 0)):
                raise ValueError(f"AgentState.model_metrics.{key} must be a non-negative integer")
        if self.execution_context is not None:
            if not isinstance(self.execution_context, ExecutionContext):
                raise ValueError("AgentState.execution_context must be an ExecutionContext")
            self.execution_context.validate()
            if self.execution_context.current_plan_id != self.current_step:
                raise ValueError("execution_context.current_plan_id must match AgentState.current_step")
        _validate_iso_timestamp(self.created_at, "AgentState.created_at")
        _validate_iso_timestamp(self.updated_at, "AgentState.updated_at")
        self.validate_frozen_fields(self._frozen_baseline)
        return self

    def frozen_values(self) -> dict[str, Any]:
        """Return the stable Session identity for comparison before Resume."""

        return {field_name: _nested_value(self, field_name) for field_name in self.FROZEN_FIELDS}

    def validate_frozen_fields(self, previous: "AgentState | dict[str, Any]") -> "AgentState":
        """Reject identity drift between a loaded Session and its resumed state."""

        if isinstance(previous, AgentState):
            expected = previous.frozen_values()
        elif isinstance(previous, dict):
            expected = (
                {field_name: previous.get(field_name) for field_name in self.FROZEN_FIELDS}
                if set(self.FROZEN_FIELDS) <= set(previous)
                else {field_name: _nested_mapping_value(previous, field_name) for field_name in self.FROZEN_FIELDS}
            )
        else:
            raise TypeError("previous AgentState identity must be an AgentState or dictionary")
        current = self.frozen_values()
        changed = [field_name for field_name in self.FROZEN_FIELDS if expected.get(field_name) != current[field_name]]
        if changed:
            raise ValueError(f"AgentState frozen fields changed: {', '.join(changed)}")
        return self

    def can_skip_plan_step(self, step_id: str) -> bool:
        """Return whether one step has the plan-owned conditional skip exception."""

        reasons = (self.task_route or {}).get("reasons")
        return step_id == "implement" and isinstance(reasons, list) and "conditional-mutation" in reasons

    def plan_step_satisfied(self, step: PlanStep) -> bool:
        """Return whether a step may satisfy dependencies and completion gates."""

        return step.status == "completed" or (step.status == "skipped" and self.can_skip_plan_step(step.id))

    def _normalize_legacy_derived_fields(self) -> None:
        """Repair v1 fields that were derived but not validated at write time."""

        known = {step.id for step in self.plan}
        if self.current_step not in known:
            self.current_step = None
        self.completed_steps = [step.id for step in self.plan if step.status == "completed"]
        if self.execution_context:
            self.execution_context.current_plan_id = self.current_step

    def _validate_plan(self) -> None:
        ids = [step.validate().id for step in self.plan]
        if len(ids) != len(set(ids)):
            raise ValueError("AgentState plan step IDs must be unique")
        known = set(ids)
        completed = {step.id for step in self.plan if self.plan_step_satisfied(step)}
        for step in self.plan:
            missing = [dependency for dependency in step.dependencies if dependency not in known]
            if missing:
                raise ValueError(f"unknown plan dependencies for {step.id}: {', '.join(missing)}")
            if step.status == "in_progress" and not set(step.dependencies) <= completed:
                raise ValueError(f"in-progress plan step dependencies are not complete: {step.id}")
        graph = {step.id: step.dependencies for step in self.plan}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(step_id: str) -> None:
            if step_id in visiting:
                raise ValueError("AgentState plan dependencies contain a cycle")
            if step_id in visited:
                return
            visiting.add(step_id)
            for dependency in graph[step_id]:
                visit(dependency)
            visiting.remove(step_id)
            visited.add(step_id)

        for step_id in graph:
            visit(step_id)
        if self.current_step is not None and self.current_step not in known:
            raise ValueError("AgentState.current_step must reference a plan step")
        expected_completed = [step.id for step in self.plan if step.status == "completed"]
        if self.completed_steps != expected_completed:
            raise ValueError("AgentState.completed_steps must match completed plan steps")

    def _validate_collections(self) -> None:
        if not isinstance(self.completed_steps, list) or not all(
            isinstance(item, str) and item for item in self.completed_steps
        ):
            raise ValueError("AgentState.completed_steps must contain non-empty strings")
        if len(self.completed_steps) != len(set(self.completed_steps)):
            raise ValueError("AgentState.completed_steps must be unique")
        if not isinstance(self.loaded_memories, list) or not all(
            _is_positive_int(item) for item in self.loaded_memories
        ):
            raise ValueError("AgentState.loaded_memories must contain positive integer IDs")
        if len(self.loaded_memories) != len(set(self.loaded_memories)):
            raise ValueError("AgentState.loaded_memories must be unique")
        if not isinstance(self.loaded_tools, list) or not all(
            isinstance(item, str) and item for item in self.loaded_tools
        ):
            raise ValueError("AgentState.loaded_tools must contain non-empty strings")
        if len(self.loaded_tools) != len(set(self.loaded_tools)):
            raise ValueError("AgentState.loaded_tools must be unique")
        if not isinstance(self.tool_calls, list) or not all(isinstance(item, dict) for item in self.tool_calls):
            raise ValueError("AgentState.tool_calls must contain dictionaries")
        if not isinstance(self.request_history, list) or not all(
            isinstance(item, str) and item.strip() for item in self.request_history
        ):
            raise ValueError("AgentState.request_history must contain non-empty strings")
        if len(self.request_history) > 50:
            raise ValueError("AgentState.request_history exceeds 50 turns")
        for index, item in enumerate(self.tool_calls):
            if not _is_positive_int(item.get("turn", 1)):
                raise ValueError(f"AgentState.tool_calls[{index}].turn must be a positive integer")
            if not _is_non_negative_int(item.get("round", 0)):
                raise ValueError(f"AgentState.tool_calls[{index}].round must be a non-negative integer")
            if "request" in item and not isinstance(item.get("request"), dict):
                raise ValueError(f"AgentState.tool_calls[{index}].request must be a dictionary")
            if "result" in item and not isinstance(item.get("result"), dict):
                raise ValueError(f"AgentState.tool_calls[{index}].result must be a dictionary")

    def _validate_routes(self) -> None:
        if not isinstance(self.task_strategy, dict):
            raise ValueError("AgentState.task_strategy must be a dictionary")
        if not isinstance(self.task_route, dict):
            raise ValueError("AgentState.task_route must be a dictionary")
        if not isinstance(self.model_route, dict):
            raise ValueError("AgentState.model_route must be a dictionary")
        if self.task_route:
            _mapping_enum(self.task_route, "task_type", TASK_TYPES, "task_route")
            _mapping_enum(self.task_route, "scale", TASK_SCALES, "task_route")
            _mapping_enum(self.task_route, "risk", TASK_RISKS, "task_route")
            _mapping_enum(self.task_route, "mode", TASK_MODES, "task_route")
            if not _is_non_negative_int(self.task_route.get("score")):
                raise ValueError("AgentState.task_route.score must be a non-negative integer")
            if not _is_positive_int(self.task_route.get("max_tool_rounds")):
                raise ValueError("AgentState.task_route.max_tool_rounds must be a positive integer")
            if not _is_non_negative_int(self.task_route.get("failure_count", 0)):
                raise ValueError("AgentState.task_route.failure_count must be a non-negative integer")
        if self.model_route:
            if str(self.model_route.get("provider") or "").lower() != "deepseek":
                raise ValueError("AgentState.model_route.provider must be deepseek")
            _mapping_enum(self.model_route, "tier", MODEL_TIERS, "model_route")
            _non_empty_string(self.model_route.get("model"), "AgentState.model_route.model")
            if not _is_positive_int(self.model_route.get("max_tokens")):
                raise ValueError("AgentState.model_route.max_tokens must be a positive integer")
            if "cost_class" in self.model_route:
                _mapping_enum(self.model_route, "cost_class", COST_CLASSES, "model_route")

    def _validate_context_manifest(self) -> None:
        if not isinstance(self.context_manifest, dict):
            raise ValueError("AgentState.context_manifest must be a dictionary")
        if not self.context_manifest:
            return
        _mapping_enum(self.context_manifest, "phase", CONTEXT_PHASES, "context_manifest")
        for key in ("max_chars", "used_chars", "rendered_chars", "original_user_request_chars"):
            if not _is_non_negative_int(self.context_manifest.get(key)):
                raise ValueError(f"AgentState.context_manifest.{key} must be a non-negative integer")
        if self.context_manifest["used_chars"] > self.context_manifest["max_chars"]:
            raise ValueError("AgentState.context_manifest.used_chars exceeds max_chars")
        if self.context_manifest["rendered_chars"] > self.context_manifest["used_chars"]:
            raise ValueError("AgentState.context_manifest.rendered_chars exceeds used_chars")
        memory_ids = self.context_manifest.get("included_memory_ids", [])
        if not isinstance(memory_ids, list) or not all(_is_positive_int(item) for item in memory_ids):
            raise ValueError("AgentState.context_manifest.included_memory_ids must contain positive integers")

    def start(self) -> None:
        self.status = "running"
        self.error = ""
        self.final_answer = ""
        if self.execution_context:
            self.execution_context.prompt_phase = "running"
            self.execution_context.last_checkpoint_at = utc_now_iso()
        self.touch()
        self.validate_frozen_fields(self._frozen_baseline)

    def resume(self, user_request: str) -> None:
        self.schema_version = max(self.SCHEMA_VERSION, self.schema_version)
        self.turn += 1
        self.user_request = user_request
        self.request_history.append(user_request)
        self.request_history = self.request_history[-50:]
        self.model_request_count = 0
        self.main_loop_model_request_count = 0
        self.context_compaction_model_request_count = 0
        self.final_synthesis_model_request_count = 0
        # Transport attempts, token usage, and convergence gates describe one
        # Session turn. Preserve durable targets and compaction-circuit state,
        # but do not carry an exhausted read/stall window into a user-initiated
        # Resume: that would make the advertised recovery command unable to
        # reacquire the exact evidence needed to finish the task.
        self.model_metrics = {}
        for key in (
            "implementation_reads_used",
            "validation_attachment_reads_used",
            "consecutive_read_only_rounds",
            "low_yield_rounds",
            "nudge_count",
            "nudge_sent_for_stall",
            "hard_notice_sent",
            "notice_turn",
        ):
            self.convergence.pop(key, None)
        self.start()
        self.validate_frozen_fields(self._frozen_baseline)

    def complete(self, final_answer: str) -> None:
        self.status = "completed"
        self.final_answer = final_answer
        self.error = ""
        self.failure_count = 0
        if self.execution_context:
            self.execution_context.prompt_phase = "completed"
            self.execution_context.last_checkpoint_at = utc_now_iso()
        self.touch()
        self.validate_frozen_fields(self._frozen_baseline)

    def fail(self, error: str, final_answer: str = "") -> None:
        self.status = "failed"
        self.error = error
        self.final_answer = final_answer
        self.failure_count = min(10_000, self.failure_count + 1)
        if self.execution_context:
            self.execution_context.prompt_phase = "failed"
            self.execution_context.recent_error = error
            self.execution_context.last_checkpoint_at = utc_now_iso()
        self.touch()
        self.validate_frozen_fields(self._frozen_baseline)

    def record_model_request(self, phase: str) -> None:
        counters = {
            "main_loop": "main_loop_model_request_count",
            "context_compaction": "context_compaction_model_request_count",
            "final_synthesis": "final_synthesis_model_request_count",
        }
        field_name = counters.get(phase)
        if field_name is None:
            raise ValueError(f"unsupported model request phase: {phase}")
        setattr(self, field_name, getattr(self, field_name) + 1)
        self.model_request_count += 1
        self.touch()
        self.validate_frozen_fields(self._frozen_baseline)

    def record_model_response(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        metrics = dict(self.model_metrics)
        metrics["http_attempt_count"] = int(metrics.get("http_attempt_count") or 0) + max(
            0, int(getattr(response, "http_attempt_count", 0) or 0)
        )
        if isinstance(usage, dict):
            aliases = {
                "prompt_tokens": ("prompt_tokens", "input_tokens"),
                "completion_tokens": ("completion_tokens", "output_tokens"),
                "total_tokens": ("total_tokens",),
            }
            for target, source_keys in aliases.items():
                raw = next((usage.get(key) for key in source_keys if key in usage), 0)
                if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
                    metrics[target] = int(metrics.get(target) or 0) + raw
        self.model_metrics = metrics
        self.touch()
        self.validate_frozen_fields(self._frozen_baseline)

    def record_tool_call(self, request: dict[str, Any], result: dict[str, Any]) -> None:
        self.tool_calls.append(
            {
                "turn": self.turn,
                "round": self.round,
                "request": limit_state_value(request),
                "result": limit_state_value(result),
                "recorded_at": utc_now_iso(),
            }
        )
        self._update_execution_context(request, result)
        self.touch()
        self.validate_frozen_fields(self._frozen_baseline)

    def _update_execution_context(self, request: dict[str, Any], result: dict[str, Any]) -> None:
        if self.execution_context is None:
            self.execution_context = ExecutionContext(
                current_directory=self.working_directory,
                git_branch=self.git_branch,
            )
        tool = str(request.get("tool") or "unknown")
        action = str(request.get("action") or "unknown")
        args = request.get("args") if isinstance(request.get("args"), dict) else {}
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        self.execution_context.recent_tool = f"{tool}.{action}"
        self.execution_context.recent_error = "" if result.get("success") else str(result.get("stderr") or "")[:2000]
        self.execution_context.current_directory = str(args.get("cwd") or self.working_directory)
        self.execution_context.git_branch = self.git_branch
        snapshot_id = data.get("snapshot_id")
        if snapshot_id:
            self.execution_context.current_snapshot = str(snapshot_id)
        path = data.get("path") or args.get("path")
        if path and tool == "file" and action in {"apply", "undo"}:
            normalized = str(path)
            if normalized not in self.execution_context.modified_files:
                self.execution_context.modified_files.append(normalized)
                self.execution_context.modified_files = self.execution_context.modified_files[-200:]
        self.execution_context.current_plan_id = self.current_step
        self.execution_context.last_checkpoint_at = utc_now_iso()

    def touch(self) -> None:
        self.updated_at = utc_now_iso()

    @property
    def run_id(self) -> str:
        return f"{self.session_id}:turn:{self.turn}"

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        serialized = asdict(self)
        return {field_name: serialized[field_name] for field_name in self.SERIALIZED_FIELDS}


def limit_state_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 5:
        return "[depth-limited]"
    if isinstance(value, dict):
        return {str(key): limit_state_value(item, depth=depth + 1) for key, item in value.items()}
    if isinstance(value, list):
        return [limit_state_value(item, depth=depth + 1) for item in value[:100]]
    if isinstance(value, str):
        return value if len(value) <= 5000 else value[:5000] + "...[truncated]"
    return value


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_positive_int(value: Any) -> bool:
    return _is_non_negative_int(value) and value >= 1


def _mapping_enum(mapping: dict[str, Any], key: str, allowed: set[str], section: str) -> str:
    value = mapping.get(key)
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"AgentState.{section}.{key} must be one of: {choices}")
    return str(value)


def _validate_iso_timestamp(value: Any, field_name: str) -> None:
    _non_empty_string(value, field_name)
    try:
        datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc


def _nested_value(state: AgentState, field_name: str) -> Any:
    root, *parts = field_name.split(".")
    value: Any = getattr(state, root)
    for part in parts:
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _nested_mapping_value(mapping: dict[str, Any], field_name: str) -> Any:
    value: Any = mapping
    for part in field_name.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value
