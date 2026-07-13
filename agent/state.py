from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .project import Project
from .timeutil import utc_now_iso


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


@dataclass
class AgentState:
    session_id: str
    project: dict[str, Any]
    user_request: str
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
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    round: int = 0
    turn: int = 1
    final_answer: str = ""
    error: str = ""
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    schema_version: int = 1

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
            user_request=user_request,
            working_directory=str(project.root),
            loaded_memories=loaded_memories,
            loaded_tools=loaded_tools,
            git_branch=git_branch,
            context_index_path=context_index_path,
            execution_context=ExecutionContext(
                current_directory=str(project.root),
                git_branch=git_branch,
            ),
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AgentState":
        plan = [PlanStep.from_dict(item) for item in value.get("plan", []) if isinstance(item, dict)]
        working_directory = str(value.get("working_directory") or "")
        git_branch = value.get("git_branch")
        execution_value = value.get("execution_context")
        execution_context = ExecutionContext.from_dict(
            execution_value if isinstance(execution_value, dict) else {},
            working_directory=working_directory,
            git_branch=git_branch,
        )
        return cls(
            session_id=str(value["session_id"]),
            project=dict(value.get("project") or {}),
            user_request=str(value.get("user_request") or ""),
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
            tool_calls=list(value.get("tool_calls") or []),
            round=int(value.get("round") or 0),
            turn=int(value.get("turn") or 1),
            final_answer=str(value.get("final_answer") or ""),
            error=str(value.get("error") or ""),
            created_at=str(value.get("created_at") or utc_now_iso()),
            updated_at=str(value.get("updated_at") or utc_now_iso()),
            schema_version=int(value.get("schema_version") or 1),
        )

    def start(self) -> None:
        self.status = "running"
        self.error = ""
        self.final_answer = ""
        if self.execution_context:
            self.execution_context.prompt_phase = "running"
            self.execution_context.last_checkpoint_at = utc_now_iso()
        self.touch()

    def resume(self, user_request: str) -> None:
        self.turn += 1
        self.user_request = user_request
        self.start()

    def complete(self, final_answer: str) -> None:
        self.status = "completed"
        self.final_answer = final_answer
        self.error = ""
        if self.execution_context:
            self.execution_context.prompt_phase = "completed"
            self.execution_context.last_checkpoint_at = utc_now_iso()
        self.touch()

    def fail(self, error: str, final_answer: str = "") -> None:
        self.status = "failed"
        self.error = error
        self.final_answer = final_answer
        if self.execution_context:
            self.execution_context.prompt_phase = "failed"
            self.execution_context.recent_error = error
            self.execution_context.last_checkpoint_at = utc_now_iso()
        self.touch()

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
        return asdict(self)


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
