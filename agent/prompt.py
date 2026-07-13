from __future__ import annotations

from typing import Any

from .context import ContextBuilder, ContextPackage, ContextSnapshot
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
- Follow the selected execution mode. Simple mode should answer directly when tools are unnecessary. Large/deep
  modes must first decompose the request into bounded, dependency-aware steps, inspect in chunks, and checkpoint
  after each completed step. Never try to solve a repository-wide task in one unbounded reasoning pass.
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
        package: ContextPackage | None = None,
        *,
        state: AgentState | None = None,
        context: ContextSnapshot | None = None,
        memory_context: str = "",
        capability_summary: str = "",
    ) -> list[dict[str, Any]]:
        if package is not None:
            return self._messages_from_package(package)
        if state is None or context is None:
            raise TypeError("build_initial requires a ContextPackage")
        # Compatibility for v0.8 callers. New Runtime code must pass a package.
        runtime_context = self._runtime_context(state, context, memory_context, capability_summary)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": runtime_context},
            {"role": "user", "content": state.user_request},
        ]

    def build_resume(self, package: ContextPackage) -> list[dict[str, Any]]:
        return self._messages_from_package(package)

    def append_resume(
        self,
        messages: list[dict[str, Any]] | None = None,
        *,
        package: ContextPackage | None = None,
        state: AgentState | None = None,
        context: ContextSnapshot | None = None,
        memory_context: str = "",
        capability_summary: str = "",
    ) -> list[dict[str, Any]]:
        if package is not None:
            return self.build_resume(package)
        if state is None or context is None:
            raise TypeError("append_resume requires a ContextPackage")
        # Compatibility for v0.8 checkpoints. Outcome selection now lives in
        # ContextBuilder and is not part of the package-based Prompt renderer.
        previous = ContextBuilder.previous_outcome(messages or [])
        refreshed = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": "Session resumed from a compact checkpoint. The previous raw tool transcript was "
                "removed to keep Prompt growth bounded. Use AgentState and Execution Context as the source of truth."
                + ("\n\n## Previous Outcome\n\n" + previous if previous else "")
                + "\n\n"
                + self._runtime_context(state, context, memory_context, capability_summary),
            },
        ]
        refreshed.append({"role": "user", "content": state.user_request})
        return refreshed

    @staticmethod
    def _messages_from_package(package: ContextPackage) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": package.rendered},
            {"role": "user", "content": package.user_request},
        ]

    @staticmethod
    def _runtime_context(
        state: AgentState,
        context: ContextSnapshot,
        memory_context: str,
        capability_summary: str,
    ) -> str:
        execution = state.execution_context
        strategy = state.task_strategy or {}
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
                f"- Git branch: `{state.git_branch or 'not detected'}`\n"
                f"- Execution mode: `{strategy.get('mode', 'standard')}`\n"
                f"- Thinking: `{str(bool(strategy.get('thinking_enabled', False))).lower()}`\n"
                f"- Reasoning effort: `{strategy.get('reasoning_effort') or 'default'}`\n"
                f"- Chunked context: `{str(bool(strategy.get('chunked_context', False))).lower()}`\n"
                f"- Plan required: `{str(bool(strategy.get('require_plan', False))).lower()}`",
                "## Task Graph\n\n" + "\n".join(plan_lines or ["No task graph has been published."]),
                "## Execution Context\n\n" + "\n".join(execution_lines),
                context.rendered,
                "## Relevant Long-Term Memory\n\n" + memory_context,
                "## Registered Tool Capabilities\n\n" + capability_summary,
            ]
        )
