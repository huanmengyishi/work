"""Versioned runtime policy constants with one discoverable ownership point.

Only hard safety/protocol limits belong here.  User-tunable defaults remain in
``config.DEFAULT_CONFIG`` so configuration migration stays add-only.
"""

from __future__ import annotations


# DeepSeek/tool protocol boundaries.
MAX_TOOL_CALLS_PER_MODEL_RESPONSE = 64
USABLE_FINISH_REASONS = frozenset({"", "stop", "tool_calls"})
DEEPSEEK_TOOL_PROTOCOL_MARKERS = (
    "<｜｜DSML｜｜tool_calls>",
    "<｜｜DSML｜｜invoke",
)

# One-validation policy shared by execution admission and completion evidence.
SINGLE_VALIDATION_MODEL_FUNCTIONS = frozenset({"run_tests", "lsp_diagnostics"})
SINGLE_VALIDATION_CAPABILITIES = frozenset({"template.run_tests", "lsp.diagnostics"})
VALIDATION_SHELL_PROGRAMS = frozenset(
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

# Hot in-memory state windows.  Older evidence is summarized or kept in a
# dedicated durable registry before it leaves the hot list.
MAX_TOOL_CALLS_IN_STATE = 200
RESUME_WINDOW_ROUNDS = 16
MEMORY_QUERY_CACHE_SIZE = 128

# AgentState is a hot, model-adjacent projection rather than an unbounded
# transcript.  These are UTF-8/JSON safety limits, not character estimates.
MAX_AGENT_STATE_BYTES = 2 * 1024 * 1024
MAX_AGENT_STATE_RECORD_BYTES = 64 * 1024
MAX_AGENT_STATE_KEY_BYTES = 512
MAX_AGENT_STATE_MAPPING_KEYS = 512
MAX_AGENT_STATE_COLLECTION_ITEMS = 20_000
MAX_AGENT_STATE_NESTING_DEPTH = 16

# Bounded metadata retained when hot state is compacted.
MAX_TOOL_HISTORY_CAPABILITIES = 64
MAX_ARTIFACT_REGISTRY_ITEMS = 128
MAX_INTENT_JOURNAL_ENTRIES = 512

# Scalar/error projection limits shared by capability and recovery paths.
HEALTH_ERROR_MAX_CHARS = 1_000
EVENT_LABEL_MAX_CHARS = 200
EVENT_COUNT_MAX = 1_000
EVENT_DURATION_MAX_MS = 86_400_000


__all__ = [
    "EVENT_COUNT_MAX",
    "EVENT_DURATION_MAX_MS",
    "EVENT_LABEL_MAX_CHARS",
    "HEALTH_ERROR_MAX_CHARS",
    "DEEPSEEK_TOOL_PROTOCOL_MARKERS",
    "MAX_AGENT_STATE_BYTES",
    "MAX_AGENT_STATE_COLLECTION_ITEMS",
    "MAX_AGENT_STATE_KEY_BYTES",
    "MAX_AGENT_STATE_MAPPING_KEYS",
    "MAX_AGENT_STATE_NESTING_DEPTH",
    "MAX_AGENT_STATE_RECORD_BYTES",
    "MAX_ARTIFACT_REGISTRY_ITEMS",
    "MAX_INTENT_JOURNAL_ENTRIES",
    "MAX_TOOL_CALLS_IN_STATE",
    "MAX_TOOL_CALLS_PER_MODEL_RESPONSE",
    "MAX_TOOL_HISTORY_CAPABILITIES",
    "MEMORY_QUERY_CACHE_SIZE",
    "RESUME_WINDOW_ROUNDS",
    "SINGLE_VALIDATION_CAPABILITIES",
    "SINGLE_VALIDATION_MODEL_FUNCTIONS",
    "USABLE_FINISH_REASONS",
    "VALIDATION_SHELL_PROGRAMS",
]
