# Deep Agent V3

Project-centric DeepSeek CLI agent for WSL. The Agent is installed as a tool;
the directory where `agent` is started is the workspace.

Current version: `0.10.0`. The core interface chain is covered by executable
contract tests. ContextBuilder is the only model-context entry, PromptBuilder
only renders a `ContextPackage`, AgentState validates its frozen schema, and the
versioned Event Bus now owns Runtime automatic side-effect pipelines.
DeepSeek thinking can stream to the terminal with elapsed time, current plan
step, and tool progress.

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

Version `0.9.0` replaces loose Prompt inputs with structured Task and Model
routes plus a unified Context Package. Resume routing is monotonic: a short
continuation cannot downgrade the saved task mode or DeepSeek tier, while a
higher-risk continuation can upgrade them.

Version `0.9.1` stabilizes these interfaces for the v1.0 freeze. TaskRouter is
the only classifier, starter plans consume a `TaskRoute`, and ModelRouter adds
an explainable low/balanced/high cost class while remaining DeepSeek-only.

Version `0.10.0` completes the in-process Event Bus migration for automatic
Session persistence, Memory usage/learning, capability health, audit, metrics,
and UI progress. Required persistence fails closed; best-effort observers never
change an already completed tool result. This remains one Runtime and one
DeepSeek provider, not a multi-Agent or cross-process broker design.

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
     -> TaskRouter -> TaskPlanFactory
     -> ModelRouter (DeepSeek only, cost-aware)
     -> AgentState + SessionManager
     -> ContextBuilder -> ContextPackage -> PromptBuilder
     -> DeepSeekClient (selected DeepSeek model)
     -> ToolManager
        -> ToolRequest -> PermissionManager -> ToolResult
     -> EventBus
        -> required Session + Memory-usage pipelines
        -> best-effort Memory/Reflection, Health, Audit, Metrics, UI progress
```

Implemented V3 modules:

- `AgentState`: validated, versioned project/session/plan/tool state with frozen
  identity fields.
- `SessionManager`: JSON checkpoints, Markdown summaries, and resume support.
- `ContextBuilder`: README/AGENTS/config discovery and cached `index.json`.
- `ContextPackage`: one bounded entry for task, project, session, semantic,
  Memory, recovery, and capability-summary context.
- `PromptBuilder`: renders the system policy, one Context Package, and request.
- `ToolCapabilityRegistry`: dynamic schemas, permissions, timeouts, formats, and availability.
- `PermissionManager`: centralized capability, cwd, timeout, and dangerous-command policy.
- `EventBus`: versioned synchronous events, required/best-effort deliveries,
  correlation IDs, named subscribers, dispatch results, and isolated errors.
- `RuntimeEventPipelines`: one registration point for Session, Memory usage,
  automatic Memory/Reflection, Health, Audit, Metrics, and visible Thinking.
- `MemoryPipeline`: idempotent Summary and Lesson/Bug/Decision persistence.
- `PlanManager`: model-maintained plans without an extra mandatory model request.
- `TaskRouter`: the only local type/scale/risk and mode classifier.
- `TaskPlanFactory`: builds starter plans from `TaskRoute` without reclassifying
  Prompt text.
- `ModelRouter`: local fast/standard/deep routing across configured DeepSeek
  model names with low/balanced/high cost classes; non-DeepSeek providers are
  rejected.

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
  max_user_request_chars: 250000
  max_tool_rounds: 8
  max_tool_rounds_hard_limit: 32
  large_project_source_files: 500
  large_project_files: 2000
  progress_interval_seconds: 10
  show_thinking: true
  show_reasoning_content: true

model:
  provider: deepseek
  model: deepseek-v4-pro
  routing:
    enabled: true
    tier: auto          # auto | fast | standard | deep
    fast_model: null
    standard_model: null
    deep_model: null
  timeout_seconds: 300
  network_retries: 2
  retry_base_seconds: 1.0

context:
  max_user_request_chars: 32000
  package_limits:
    simple: 12000
    standard: 32000
    large: 48000
    deep: 64000
  max_package_chars_hard_limit: 96000
  max_recovery_context_chars: 6000
```

The three DeepSeek tiers are capability policies, not separate providers.
Their model overrides default to `null`, so fast, standard, and deep all fall
back to `model.model` until the user explicitly supplies valid DeepSeek model
names. With adaptive thinking enabled, fast disables thinking, standard uses
`high`, and deep uses `max` reasoning effort.

The local cost-aware policy chooses the least expensive tier that still meets
the classified task: simple low-risk requests use `low`, ordinary and large
read-only work use `balanced`, and deep/high-risk/architecture/refactor or
repeated-failure work uses `high`. This is a routing estimate, not a billing
measurement. Explicit tier configuration still takes priority.

## Frozen Interfaces And Event Pipelines

The interface contract marker is in `agent/contracts.py`, and
`tests/test_interface_contracts.py` verifies the public chain:

```text
CLI -> Runtime -> AgentState -> Prompt -> Capability -> Permission
ContextBuilder -> ContextPackage -> PromptBuilder
```

PromptBuilder accepts only one `ContextPackage`; it does not load files or take
separate State, Memory, or capability text. `Event` serializes
`schema_version/id/name/timestamp/project_id/session_id/run_id/payload`.
EventBus remains synchronous and process-local: it isolates handler failures
and distinguishes required owners from best-effort observers, but it does not
provide replay, a network broker, or cross-process ordering. Required Session
writes and Memory usage fail closed; terminal Memory/Reflection, capability
health, audit, metrics, and UI progress are best-effort. See
`docs/architecture-v0.10.0.md` for ownership, ordering, idempotency, and privacy
rules.

Required dispatch carries named delivery evidence from the Session owner. If
that owner failed, Runtime does not try to finalize an uncertain checkpoint; if
the owner succeeded and only a later required observer failed, Runtime can
safely persist a failed terminal for Resume.

JSONL audit is metadata-only. It drops Prompt, reasoning, model messages,
AgentState, tool arguments, stdout/stderr, output bodies, and credentials.
Project metrics store only allowed event counts, bounded aggregate tool time,
and failed-tool count under `~/.local/share/deep-agent/metrics/`. Both outputs
use private permissions and reject symbolic-link targets. Configure only by
adding defaults; existing user values are preserved:

```yaml
events:
  jsonl_log: true
  metrics_enabled: true
```

The Context Package budget counts the bounded user request plus fully rendered
sections, including headings and separators. The fixed system prompt, active
tool JSON schemas, and ToolResult messages produced during later rounds are
outside this character budget and keep their own limits. Oversized requests
retain bounded head/tail content. `context.generated.md` contains only public
project context; Session, Memory, and recovery text are not written to that
cache. Requests above `runtime.max_user_request_chars` are rejected with advice
to save the text/code in the project for complete chunked inspection.

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

Important: normal/YOLO Docker calls reject host-root, socket, device, and host
namespace access. Unstructured shell/Python commands are still host processes;
their working directory is checked, but the operating system is the actual
security boundary for paths referenced inside command text. Use safe mode for
untrusted requests and reserve `--super-yolo` for deliberate host access.

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

The repository includes `.github/workflows/test.yml` for Python 3.11, 3.12,
and 3.13. It uses `actions/checkout@v5` and `actions/setup-python@v6` (current
Node.js runtime, no Node.js 20 deprecation), then runs Ruff, all 173 pytest
cases, and compileall for pushes to `main`, pull requests, and manual dispatch.
The v0.10.0 hosted run is verified during release publication.

A runnable Chinese walkthrough is available at
`user-docs/实用案例-v0.9.0/README.md`. Its order-summary project intentionally
starts with two failing business-rule tests so users can observe simple,
standard, large, deep, Thinking, and Resume behavior on a real repair task.
An offline interface example at
`user-docs/实用案例-v0.9.1/interface-routing-demo.py` demonstrates cost-aware
routing, TaskRoute-only plan creation, event correlation, and subscriber error
isolation without using an API key.
`user-docs/实用案例-v0.10.0/event-runtime-demo.py` demonstrates required
delivery, best-effort isolation, Memory usage idempotency, safe audit, and
bounded metrics entirely offline.

See `docs/implementation.md` for architecture, extension rules, Docker proxy,
OCR, memory, and maintenance details.

## Rollback

```bash
cd ~/AI-Agent
git switch --detach v0.9.1
.venv/bin/pip install -e .
```

Return with `git switch main`. v0.10.0 only adds defaults and an internal
SQLite idempotency table; it does not overwrite configuration or delete
Session, Memory, or project data. Older code safely ignores the added table and
metrics file.
