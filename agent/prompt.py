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
- For multi-step tasks, first publish a dependency-aware Task Graph with agent_update_plan. Include completion
  criteria, retries, and allow_parallel only when justified. Start only steps whose dependencies are complete.
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
        execution = state.execution_context
        plan_lines = [
            f"- `{step.id}` {step.status}; deps={','.join(step.dependencies) or '-'}; "
            f"retries={step.retry_count}/{step.max_retries}; parallel={str(step.allow_parallel).lower()}; "
            f"done_when={step.completion_criteria or '-'}"
            for step in state.plan
        ]
        execution_lines = (
            [
                f"- Current directory: `{execution.current_directory}`",
                f"- Git branch: `{execution.git_branch or 'not detected'}`",
                f"- Current plan step: `{execution.current_plan_id or 'none'}`",
                f"- Current queue: `{execution.current_queue_id or 'none'}`",
                f"- Recent tool: `{execution.recent_tool or 'none'}`",
                f"- Recent error: `{execution.recent_error[:500] or 'none'}`",
                f"- Current snapshot: `{execution.current_snapshot or 'none'}`",
                f"- Modified files: `{', '.join(execution.modified_files[-20:]) or 'none'}`",
                f"- Prompt phase: `{execution.prompt_phase}`",
            ]
            if execution
            else ["- No execution context was restored."]
        )
        return "\n\n".join(
            [
                "## Agent State\n"
                f"- Session: `{state.session_id}`\n"
                f"- Turn: `{state.turn}`\n"
                f"- Working directory: `{state.working_directory}`\n"
                f"- Git branch: `{state.git_branch or 'not detected'}`",
                "## Task Graph\n\n" + "\n".join(plan_lines or ["No task graph has been published."]),
                "## Execution Context\n\n" + "\n".join(execution_lines),
                context.rendered,
                "## Relevant Long-Term Memory\n\n" + memory_context,
                "## Registered Tool Capabilities\n\n" + capability_summary,
            ]
        )
