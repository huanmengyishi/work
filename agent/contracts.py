"""Version markers for the public v1.0-preparation interface boundary.

The executable contract is enforced by ``tests/test_interface_contracts.py``.
Changing the chain, a frozen schema, or one of the tested public signatures is
a compatibility change and must increment ``CORE_INTERFACE_CONTRACT_VERSION``.
"""

from __future__ import annotations


CORE_INTERFACE_CONTRACT_VERSION = 2
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
AGENT_STATE_SCHEMA_VERSION = 6
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
    "round",
    "model_request_count",
    "main_loop_model_request_count",
    "context_compaction_model_request_count",
    "final_synthesis_model_request_count",
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


__all__ = [
    "CONTEXT_INTERFACE_CHAIN",
    "CORE_INTERFACE_CHAIN",
    "CORE_INTERFACE_CONTRACT_VERSION",
    "EVENT_SCHEMA_VERSION",
    "EVENT_SERIALIZED_FIELDS",
    "AGENT_STATE_SCHEMA_VERSION",
    "AGENT_STATE_SERIALIZED_FIELDS",
    "AGENT_STATE_FROZEN_FIELDS",
]
