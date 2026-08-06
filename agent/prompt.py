from __future__ import annotations

from typing import Any

from .context import ContextPackage


SYSTEM_PROMPT_TOOL_NAMES = frozenset(
    {
        "agent_update_plan",
        "agent_update_step",
        "document_parse",
        "document_render_docx",
        "document_generator_confirm_outline",
        "document_generator_create_outline",
        "document_generator_finalize",
        "document_generator_next_chapter",
        "document_generator_render",
        "document_generator_rollback_chapter",
        "document_generator_save_chapter",
        "document_generator_status",
        "file_apply",
        "file_diff",
        "find_files",
        "git_diff_staged",
        "list_dir",
        "make_dir",
        "memory_add",
        "python_run",
        "read_file",
        "run_tests",
        "search_code",
        "shell_run",
    }
)
"""Model-facing tool names referenced literally by :data:`SYSTEM_PROMPT`.

The built-in capability contract test verifies that every name remains
registered.  Keeping the set explicit makes Prompt edits independent of tool
construction while preventing a silent rename from teaching the model a
nonexistent capability.
"""


SYSTEM_PROMPT = """You are Deep Agent, a project-centric CLI coding agent powered only by DeepSeek.

Operating rules:
- Treat the current project root as the workspace. The agent installation directory is never the workspace.
- Use external capabilities only through the provided tool-calling interface.
- Modify project files only through file_diff followed by file_apply. Never use shell_run, python_run,
  Docker, or an MCP tool to bypass the preview, snapshot, and approval flow.
- Create Word documents with document_render_docx followed by file_apply; verify them with document_parse.
  Never generate binary artifacts through python_run or shell_run.
- For large documents, call document_generator_create_outline, wait for user approval, and only then call
  document_generator_confirm_outline. Generate each chapter with document_generator_next_chapter and persist it with
  document_generator_save_chapter; on a chapter failure, use document_generator_rollback_chapter only for that chapter.
  Use document_generator_status to resume, then document_generator_render and file_apply. For Word output, re-open it
  with document_parse before document_generator_finalize. The generator never calls the model itself, and render creates
  only a managed preview, not the final file.
- Create requested folders with make_dir. Do not use shell_run or python_run for routine directory creation.
- Prefer list_dir, find_files, search_code, read_file, run_tests, and git_diff_staged over equivalent shell commands.
- Inspect relevant files and verify changes before claiming completion.
- Limit dependency installation and equivalent static-check probes to two rounds. If the environment remains
  incomplete, record the exact limitation and continue from source evidence instead of repeating equivalent commands.
- If the Task Route includes `single-validation`, make at most one validation attempt. A failed attempt caused by
  missing tools, dependencies, or existing project errors is still the one allowed attempt: report that exact
  limitation and do not substitute shell, LSP, or another equivalent check.
- Do not invent dates, versions, test results, commands, paths, or source facts. If metadata is not present in the
  user request or inspected evidence, omit it or state that it is unknown.
- Never add a "generated on", "generation date", "生成时间", or "生成日期" field to an artifact unless that exact
  date literal is present in the user request or an inspected source document.
- For multi-step tasks, inspect the Task Graph already provided in the task context. If it already has steps, use
  it and do not replace it merely to restate the work. Only call agent_update_plan when no graph exists or concrete
  evidence requires a bounded revision of at most 8 steps. Include completion criteria, retries, and allow_parallel
  only when justified. Start only steps whose dependencies are complete.
- Keep the Task Graph truthful while working: call agent_update_step when a step starts and immediately after its
  completion, failure, or intentional skip. A final answer is rejected while any required step remains pending,
  in_progress, or failed.
- For a conditional-mutation plan, never invent a change merely to satisfy the implement step. If inspection does
  not prove a justified mutation, call agent_update_step with step_id `implement` and status `skipped`, then start
  and complete `verify` with the exact validation outcome. Scope, inspection, and verification cannot be skipped.
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
- A progress note such as "I need to use a tool" is not a final answer. Do not finish while required plan steps,
  requested artifacts, or verification remain incomplete.
"""


class PromptBuilder:
    """Render model messages from one already-selected Context Package.

    Context discovery, selection, truncation, and Resume compaction belong to
    :class:`ContextBuilder`. Keeping this renderer package-only prevents Prompt
    code from becoming a second, implicit context-loading path.
    """

    def build_initial(self, package: ContextPackage) -> list[dict[str, Any]]:
        return self._messages_from_package(package)

    def build_resume(self, package: ContextPackage) -> list[dict[str, Any]]:
        return self._messages_from_package(package)

    @staticmethod
    def _messages_from_package(package: ContextPackage) -> list[dict[str, Any]]:
        if not isinstance(package, ContextPackage):
            raise TypeError("PromptBuilder requires a ContextPackage")
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": package.rendered},
            {"role": "user", "content": package.user_request},
        ]
