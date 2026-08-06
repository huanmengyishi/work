"""Version markers for the public v1.0-preparation interface boundary.

The executable contract is enforced by ``tests/test_interface_contracts.py``.
Changing the chain, a frozen schema, or one of the tested public signatures is
a compatibility change and must increment ``CORE_INTERFACE_CONTRACT_VERSION``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ContextManager, Iterable, Protocol, runtime_checkable


CORE_INTERFACE_CONTRACT_VERSION = 6
CORE_INTERFACE_CHAIN = (
    "CLI",
    "Runtime",
    "AgentState",
    "Prompt",
    "Capability",
    "Permission",
)
CONTEXT_INTERFACE_CHAIN = (
    "ContextBuilder",
    "ContextPackage",
    "PromptBuilder",
)
EVENT_SCHEMA_VERSION = 1
EVENT_SERIALIZED_FIELDS = (
    "schema_version",
    "id",
    "name",
    "timestamp",
    "project_id",
    "session_id",
    "run_id",
    "payload",
)
AGENT_STATE_SCHEMA_VERSION = 8
AGENT_STATE_SERIALIZED_FIELDS = (
    "session_id",
    "project",
    "objective",
    "user_request",
    "request_history",
    "working_directory",
    "status",
    "plan",
    "current_step",
    "completed_steps",
    "loaded_memories",
    "loaded_tools",
    "git_branch",
    "context_index_path",
    "execution_context",
    "task_strategy",
    "task_route",
    "model_route",
    "context_manifest",
    "convergence",
    "model_metrics",
    "tool_calls",
    "tool_history_summary",
    "artifact_registry",
    "resume_checkpoint",
    "intent_journal_head",
    "round",
    "model_request_count",
    "main_loop_model_request_count",
    "context_compaction_model_request_count",
    "final_synthesis_model_request_count",
    "memory_refinement_model_request_count",
    "turn",
    "final_answer",
    "error",
    "failure_count",
    "created_at",
    "updated_at",
    "schema_version",
)
AGENT_STATE_FROZEN_FIELDS = (
    "session_id",
    "project",
    "objective",
    "working_directory",
    "created_at",
)


@runtime_checkable
class ModelClientProtocol(Protocol):
    """The DeepSeek-only request boundary consumed by Runtime."""

    def chat(self, **kwargs: Any) -> Any: ...


@runtime_checkable
class RuntimeToolsProtocol(Protocol):
    """Complete managed-tool surface consumed by Runtime.

    This is deliberately the Runtime-facing manager contract rather than the
    smaller single-call executor contract. Permissions and handler execution
    remain internal to the implementation.
    """

    registry: Any
    health: Any
    plan_manager: Any

    def schemas(self) -> list[dict[str, Any]]: ...

    def capabilities(self, *, enabled_only: bool = False) -> list[Any]: ...

    def bind_state(self, state: Any) -> None: ...

    def set_event_bus(self, events: Any) -> None: ...

    def model_function_name(self, name: str) -> str: ...

    def canonical_capability_name(self, name: str) -> str: ...

    def result_is_health_failure(self, result: Any) -> bool: ...

    def capability_health_status(self, capability_name: str) -> str: ...

    def capability_summary(self) -> str: ...

    def close(self) -> None: ...

    def execute_model_call(
        self,
        name: str,
        arguments: str | dict[str, Any] | None,
        *,
        request_id: str | None = None,
        runtime_denied_reason: str | None = None,
    ) -> tuple[Any, Any]: ...


@runtime_checkable
class ToolExecutorProtocol(Protocol):
    """Permission-gated single-tool execution surface.

    ``RuntimeToolsProtocol`` describes the higher-level manager consumed by
    Runtime.  Keeping this smaller executor contract distinct prevents callers
    from accidentally requiring manager-only state (health, plans, binding)
    from the component that owns one
    ``ToolRequest -> PermissionManager -> ToolResult`` lifecycle.
    """

    registry: Any
    permission: Any

    def schemas(self) -> list[dict[str, Any]]: ...

    def configure_permissions(
        self,
        *,
        yolo: bool | None = None,
        super_yolo: bool | None = None,
    ) -> None: ...

    def execute_model_call(
        self,
        name: str,
        arguments: str | dict[str, Any] | None,
        *,
        request_id: str | None = None,
        runtime_denied_reason: str | None = None,
    ) -> tuple[Any, Any]: ...

    def execute(
        self,
        request: Any,
        *,
        ownership: Any,
        runtime_denied_reason: str | None = None,
        argument_error: str | None = None,
    ) -> Any: ...


@runtime_checkable
class SessionStoreProtocol(Protocol):
    """Bounded Session persistence used by Runtime and event pipelines."""

    def new_session_id(self) -> str: ...

    def resolve_session_id(self, session_id: str | None = None) -> str: ...

    def acquire(self, session_id: str) -> ContextManager[Any]: ...

    def load_for_resume(self, session_id: str, *, max_rounds: int = 16) -> Any: ...

    def checkpoint(self, state: Any, messages: list[dict[str, Any]]) -> Path: ...

    def finalize(self, state: Any, messages: list[dict[str, Any]]) -> tuple[Path, Path]: ...

    def load(self, session_id: str | None = None) -> Any: ...

    def list_sessions(self, limit: int = 20) -> list[Any]: ...


@runtime_checkable
class MemoryStoreProtocol(Protocol):
    """Memory surface shared by Runtime, tool, and event-pipeline boundaries."""

    def search(
        self,
        query: str,
        project_id: str | None = None,
        limit: int | None = None,
        *,
        global_only: bool = False,
        record_usage: bool = True,
        kinds: Iterable[Any] | None = None,
        truncate_query: bool = False,
    ) -> list[Any]: ...

    def search_recovery(self, error_text: str, project_id: str, limit: int = 4) -> list[Any]: ...

    def add_memory(
        self,
        *,
        kind: Any,
        title: str,
        content: str,
        tags: Iterable[str] = (),
        project_id: str | None = None,
        confidence: float | None = None,
        expires_at: str | None = None,
    ) -> int: ...

    def get_memory(self, memory_id: int) -> Any | None: ...

    def list_memories(
        self,
        *,
        project_id: str | None = None,
        limit: int = 50,
        kind: str | None = None,
        tag: str | None = None,
        global_only: bool = False,
        include_global: bool = True,
    ) -> list[Any]: ...

    def update_memory(
        self,
        memory_id: int,
        *,
        title: str | None = None,
        content: str | None = None,
        tags: Iterable[str] | None = None,
        confidence: float | None = None,
        expires_at: str | None = None,
    ) -> Any: ...

    def delete_memory(
        self,
        memory_id: int,
        *,
        project_id: object = ...,
        kinds: Iterable[Any] | None = None,
    ) -> bool: ...

    def record_usage_once(
        self,
        usage_id: str,
        memory_ids: Iterable[int],
        *,
        run_id: str,
        project_id: str | None,
    ) -> bool: ...

    def is_pipeline_run_processed(self, run_id: str) -> bool: ...

    def mark_pipeline_run_processed(
        self,
        run_id: str,
        project_id: str | None,
        summary_memory_id: int | None,
        experience_memory_id: int | None,
    ) -> None: ...

    def update_summary(self, *, scope: str, content: str, project_id: str | None = None) -> None: ...

    def persist_lesson_file(
        self,
        *,
        kind: Any,
        title: str,
        content: str,
        project: Any | None,
        global_memory: bool = True,
    ) -> None: ...


@runtime_checkable
class RecoveryControllerProtocol(Protocol):
    """Deterministic capability backoff consumed by the Runtime tool loop."""

    def before_call(
        self,
        convergence: dict[str, Any],
        capability: str,
        *,
        current_round: int,
    ) -> Any: ...

    def observe(
        self,
        convergence: dict[str, Any],
        capability: str,
        *,
        current_round: int,
        success: bool,
        health_failure: bool,
        health_status: str,
    ) -> Any: ...


@runtime_checkable
class ContextBuilderProtocol(Protocol):
    """Project snapshot and ContextPackage construction consumed by Runtime."""

    def build(self, project: Any, *, refresh: bool = False) -> Any: ...

    def build_package(self, request: Any) -> Any: ...


@runtime_checkable
class PromptBuilderProtocol(Protocol):
    """Pure Prompt rendering from one bounded ContextPackage."""

    def build_initial(self, package: Any) -> list[dict[str, Any]]: ...

    def build_resume(self, package: Any) -> list[dict[str, Any]]: ...


__all__ = [
    "CONTEXT_INTERFACE_CHAIN",
    "CORE_INTERFACE_CHAIN",
    "CORE_INTERFACE_CONTRACT_VERSION",
    "EVENT_SCHEMA_VERSION",
    "EVENT_SERIALIZED_FIELDS",
    "AGENT_STATE_SCHEMA_VERSION",
    "AGENT_STATE_SERIALIZED_FIELDS",
    "AGENT_STATE_FROZEN_FIELDS",
    "ContextBuilderProtocol",
    "MemoryStoreProtocol",
    "ModelClientProtocol",
    "PromptBuilderProtocol",
    "RecoveryControllerProtocol",
    "RuntimeToolsProtocol",
    "SessionStoreProtocol",
    "ToolExecutorProtocol",
]
