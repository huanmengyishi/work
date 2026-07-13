from __future__ import annotations

from typing import Any

from .context import ContextSnapshot
from .state import AgentState


SYSTEM_PROMPT = """You are Deep Agent, a project-centric CLI coding agent powered only by DeepSeek.

Operating rules:
- Treat the current project root as the workspace. The agent installation directory is never the workspace.
- Use external capabilities only through the provided tool-calling interface.
- Modify project files only through file_diff followed by file_apply. Never use shell_run, python_run,
  Docker, or an MCP tool to bypass the preview, snapshot, and approval flow.
- Prefer list_dir, find_files, search_code, read_file, run_tests, and git_diff_staged over equivalent shell commands.
- Inspect relevant files and verify changes before claiming completion.
- For multi-step tasks, first publish a concise plan with agent_update_plan and keep its statuses current.
- Keep durable project facts in .project-agent/context.md only when they will matter in future sessions.
- Store reusable lessons, bugs, decisions, or knowledge through memory_add when the information is genuinely durable.
- If the user explicitly corrects or rejects an earlier answer, behavior, path, port, API, or fact in this
  conversation, first provide the corrected response and then call memory_add with kind `Correction`. Include a
  `correction:<topic>` tag and the current project name as a tag. Record only the corrected durable fact and enough
  context to prevent the same mistake; never store credentials or transient preferences.
- Do not duplicate generated context into context.md; generated context is maintained separately.
- Give a concise, evidence-based final answer.
"""


class PromptBuilder:
    def build_initial(
        self,
        *,
        state: AgentState,
        context: ContextSnapshot,
        memory_context: str,
        capability_summary: str,
    ) -> list[dict[str, Any]]:
        runtime_context = self._runtime_context(state, context, memory_context, capability_summary)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": runtime_context},
            {"role": "user", "content": state.user_request},
        ]

    def append_resume(
        self,
        messages: list[dict[str, Any]],
        *,
        state: AgentState,
        context: ContextSnapshot,
        memory_context: str,
        capability_summary: str,
    ) -> list[dict[str, Any]]:
        refreshed = list(messages)
        refreshed.append(
            {
                "role": "system",
                "content": "Session resumed. Refresh the runtime context before continuing.\n\n"
                + self._runtime_context(state, context, memory_context, capability_summary),
            }
        )
        refreshed.append({"role": "user", "content": state.user_request})
        return refreshed

    @staticmethod
    def _runtime_context(
        state: AgentState,
        context: ContextSnapshot,
        memory_context: str,
        capability_summary: str,
    ) -> str:
        return "\n\n".join(
            [
                "## Agent State\n"
                f"- Session: `{state.session_id}`\n"
                f"- Turn: `{state.turn}`\n"
                f"- Working directory: `{state.working_directory}`\n"
                f"- Git branch: `{state.git_branch or 'not detected'}`",
                context.rendered,
                "## Relevant Long-Term Memory\n\n" + memory_context,
                "## Registered Tool Capabilities\n\n" + capability_summary,
            ]
        )
