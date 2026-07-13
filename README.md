# Deep Agent V3

Project-centric DeepSeek CLI agent for WSL. The Agent is installed as a tool;
the directory where `agent` is started is the workspace.

Current version: `0.8.0`. The runtime chooses a bounded execution mode for each
request: lightweight direct answer, standard engineering, large-scale chunked
inspection, or deep dependency-aware execution. DeepSeek thinking can stream to
the terminal with elapsed time, current plan step, and tool progress.

Version `0.7.0` adds bounded Python/JavaScript/TypeScript diagnostics, richer
Tree-sitter module and import relationships, Memory lifecycle maintenance,
compact Session resume, and an optional per-project background daemon. The
daemon is disabled by default and uses project-specific PID, lock, state, and
log files under the XDG data directory.

Version `0.7.1` fixes interactive input on GNU Readline/WSL terminals. Colored
prompts now mark ANSI control bytes as non-printing so cursor, line wrapping,
and Enter submission are calculated correctly. Empty input gives explicit
feedback, and a visible processing message appears immediately after a request
is submitted. Project discovery now also requires valid Git metadata instead of
treating an arbitrary empty `.git` directory as a repository root.

Version `0.8.0` adds adaptive task strategy and large-input handling. Simple
questions stay on a four-round lightweight path. Standard tasks keep the normal
eight-round tool loop. Repository-wide or long-document requests use bounded
chunks, an explicit Task Graph, 16 rounds, and DeepSeek `high` thinking. Deep
audits/refactors use `max` effort and up to 24 rounds. Mode and limits are stored
in AgentState, so a short `/resume` cannot downgrade an active deep task.

## Quick Start

Store one valid key, or a comma-separated Key pool, in the private secrets file:

```bash
nano ~/.config/deep-agent/secrets.env
```

```bash
DEEPSEEK_API_KEY=key_1,key_2,key_3
```

Then verify and run:

```bash
chmod 600 ~/.config/deep-agent/secrets.env
agent doctor --online
cd /path/to/project
agent "summarize this project"
```

English commas (`,`) and Chinese commas (`，`) are both supported. Whitespace,
empty values, and duplicate keys are ignored. On HTTP `401`, `403`, or `429`,
the current request retries the next key. Transient network, timeout, `408`, and
`5xx` failures retry the same key with bounded exponential backoff.
`agent doctor --online` validates every key and
reports only counts and status codes, never key values. A value in
`secrets.env` takes priority over a legacy shell value.

## Directory Ownership

```text
~/AI-Agent/                         program and tests
~/.config/deep-agent/              user configuration and secrets
~/.local/share/deep-agent/         databases, vectors, logs, backups
<project>/.project-agent/          project context, index, sessions, cache
```

Project-private runtime data is stored under `snapshots/`, `browser-sessions/`,
`downloads/`, and `memory/`. `.project-agent/.gitignore` prevents those paths from being
committed. On Linux filesystems the private directories use mode `700`; WSL
DrvFS mounts such as `/mnt/d` may display `777` unless metadata mounting is
enabled.

The API key belongs in `~/.config/deep-agent/secrets.env`, not in this source
tree, a project directory, or Git.

## V3 Runtime

```text
CLI
  -> AgentRuntime
     -> AgentState + SessionManager
     -> ContextBuilder + PromptBuilder
     -> DeepSeekClient
     -> ToolManager
        -> ToolRequest -> PermissionManager -> ToolResult
     -> EventBus
        -> JSONL logger
        -> MemoryPipeline -> SQLite + Chroma
```

Implemented V3 modules:

- `AgentState`: serializable project, session, plan, tool, and progress state.
- `SessionManager`: JSON checkpoints, Markdown summaries, and resume support.
- `ContextBuilder`: README/AGENTS/config discovery and cached `index.json`.
- `PromptBuilder`: one place for system, project, memory, user, and tool context.
- `ToolCapabilityRegistry`: dynamic schemas, permissions, timeouts, formats, and availability.
- `PermissionManager`: centralized capability, cwd, timeout, and dangerous-command policy.
- `EventBus`: decoupled runtime, logging, and memory events.
- `MemoryPipeline`: idempotent Summary and Lesson/Bug/Decision persistence.
- `PlanManager`: model-maintained plans without an extra mandatory model request.
- `TaskStrategySelector`: local simple/standard/large/deep routing with bounded budgets.

## Commands

```bash
agent --help
agent --version
agent doctor
agent doctor --online
agent init
agent "implement this feature"
agent
agent sessions
agent resume --session SESSION_ID "continue the task"
agent context show
agent context refresh
agent context index
agent tools
agent tools --all
agent mcp status
agent mcp tools
agent mcp config
agent projects
agent memory search "query"
agent memory add Knowledge "title" "content" --global-memory
agent memory list --kind Correction
agent memory stats
agent memory maintain
agent memory maintain --apply
agent queue "task one" "task two"
agent queue resume --id QUEUE_ID
agent parallel "task 1" "task 2" "task 3" "task 4" "task 5" "task 6" "task 7" "task 8"
agent health
agent daemon start
agent daemon status
agent daemon stop
```

Interactive-only commands: `/new`, `/resume [session-id]`, `/sessions`,
`/status`, `/undo`, `/yolo on|off`, `/super-yolo on|off`, `/help`, `/clear`, and
`/exit`. History is stored with mode `600` at
`~/.local/share/deep-agent/cache/repl_history`.

Press `Enter` once to submit a request. The CLI then shows elapsed `Thinking`
time, selected mode, model round, current plan step, and tool status. DeepSeek
`reasoning_content` streams under `DeepSeek Thinking`, so long requests do not
leave the terminal silent. An empty `Enter` keeps the session open and explains
how to submit a task. `Ctrl+C` returns to the prompt; use `/resume` to continue.

## Adaptive Execution Modes

```text
simple    short factual question; thinking off; up to 4 rounds
standard  normal coding/analysis; high thinking; configured 8 rounds
large     repository/long document; chunked Task Graph; 16 rounds
deep      audit/refactor/root cause; max effort; 24 rounds
```

Classification is local and consumes no extra API request. Optional overrides
in `~/.config/deep-agent/config.yaml`:

```yaml
runtime:
  task_mode: auto       # auto | simple | standard | large | deep
  adaptive_thinking: true
  max_tool_rounds: 8
  max_tool_rounds_hard_limit: 32
  large_project_source_files: 500
  large_project_files: 2000

model:
  timeout_seconds: 300
  network_retries: 2
  retry_base_seconds: 1.0
```

Large text/code follows scope -> bounded chunks -> synthesize/implement ->
verify. The Agent does not load an entire repository into one unbounded Prompt.

Approval modes:

- Safe mode asks before confirmation-gated tools.
- `--auto-approve` only auto-approves configured snapshot-backed capabilities,
  `file.apply` and `file.undo` by default.
- `--yolo` skips confirmations but keeps hard permission policy.
- `--super-yolo` also bypasses Permission Manager policy, allowing `sudo`,
  external working directories, privileged Docker arguments, and destructive
  commands. Operating-system authentication still applies.

Persistent switches are `permissions.yolo` and `permissions.super_yolo` in
`~/.config/deep-agent/config.yaml`.

## Safe Editing

Source changes use `file_diff -> file_apply -> file_undo`. `file_diff` stores a
preview ID without modifying the file. `file_apply` validates the original
SHA-256, creates a session snapshot, writes atomically, and verifies the result.
`file_undo` refuses to overwrite work changed after the snapshot. Git projects
record HEAD/branch/path status without stashing or moving existing changes.

## Learning And Memory

Explicit user corrections are stored as `Correction` memory. A
`correction:<topic>` tag is mandatory and the project name is added
automatically. Failed ToolResults trigger one bounded local search over related
Correction/Lesson records and inject recovery context into the next model round
without another API request. Use `agent memory list/edit/delete/stats` to keep
the store accurate. `agent memory maintain` previews duplicate and expiry
cleanup; add `--apply` to merge high-similarity Correction/Lesson/Reflection
records and delete expired low-confidence, non-protected records. Corrections
and Decisions are protected from automatic expiry by default.

## MCP

MCP is opt-in in `~/.config/deep-agent/mcp.yaml`. Version `0.5.0` supports stdio,
Streamable HTTP, legacy SSE, tool calls, and optional `resources/read`.
Resources require explicit `resources_enabled: true`. Existing allowlists,
permissions, limits, and failure isolation apply. HTTP transports reject URL
credentials and redirects. DeepSeek keys are not inherited by stdio servers
unless explicitly listed in `env_passthrough`.

## HTTP, Semantic Index, Queue, And Parallelism

`http_request` is disabled by default and requires `tools.http.enabled` plus an
allowlist. It permits only GET/POST JSON with 30-second and 1 MiB limits and
rejects sensitive headers and redirects.

The optional Tree-sitter sidecar writes `index.semantic.json` and adds a bounded
class/function/import/module relationship summary to project context without
replacing `index.json`. Resume rebuilds a compact Prompt from AgentState,
Execution Context, the previous outcome, current project context, and current
Memory rather than retaining an unlimited raw tool transcript.

`lsp_diagnostics` uses Pyright for Python and `tsc --noEmit` for JavaScript and
TypeScript. Each engine degrades independently. After `file_apply`, supported
files are diagnosed automatically; diagnostics are attached to the successful
write result so the model can continue fixing errors without misclassifying the
atomic write as failed.

`agent queue` persists serial tasks and resumes without repeating completed
entries. `agent parallel` requires at least eight explicit tasks and a clean Git
worktree, then validates per-worktree patches before applying them.

## Optional Daemon

The daemon is opt-in and does not change CLI behavior. `agent daemon start`
starts one background process for the current project. It polls for file changes,
refreshes `index.json`, `workspace_memory.json`, and optional semantic context,
and applies periodic Memory lifecycle maintenance. Queue execution remains off
unless `daemon.queue_enabled: true` is set explicitly.

```yaml
daemon:
  enabled: false
  poll_interval_seconds: 10
  memory_maintenance_seconds: 3600
  queue_enabled: false
  queue_timeout_seconds: 3600
```

Use `agent daemon status` and `agent daemon stop`. Runtime files are stored in
`~/.local/share/deep-agent/daemon/<ProjectID>/`.

## Browser Sessions

`browser_open_url` accepts `session_name` and persists cookies/local storage in
`.project-agent/browser-sessions/<name>/`. `browser_download` saves downloads
under `.project-agent/downloads/<name>/` and returns path, MIME type, and size.
`browser_close_session` can clear stored identity data after confirmation.

If a natural-language task starts with a management command name, escape command
dispatch with `--`:

```bash
agent -- doctor this code path
```

## Verification

```bash
cd ~/AI-Agent
.venv/bin/python -m pytest
.venv/bin/ruff check agent tests scripts
.venv/bin/ruff format --check agent tests scripts
.venv/bin/python -m compileall -q agent tests scripts
```

Current v0.8.0 baseline: 79 pytest cases, including a real PTY smoke test.

See `docs/implementation.md` for architecture, extension rules, Docker proxy,
OCR, memory, and maintenance details.

## Rollback

```bash
cd ~/AI-Agent
git switch --detach v0.7.1
.venv/bin/pip install -e .
```

Return with `git switch main`. v0.8.0 config migration is add-only; v0.7.1
ignores the new keys.
