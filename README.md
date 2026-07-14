# Deep Agent V3

Project-centric DeepSeek CLI agent for WSL. The Agent is installed as a tool;
the directory where `agent` is started is the workspace.

Current version: `0.11.0`. The core interface chain is covered by executable
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

Version `0.11.0` makes the tool loop usable for long documents and repository
analysis. Every request reserves output space, bounds individual and aggregate
tool results, micro-compacts old evidence, and can ask the same DeepSeek model
for a tool-free context summary. Consecutive summary failures open a circuit
breaker and fall back to a deterministic bounded state projection. Large/deep
tasks stop repeated exploration early enough to synthesize and verify, and a
separate local final-synthesis phase replaces the old generic tool-limit
message. Explicitly concurrency-safe read batches may overlap while mutations
remain serial; interruption still produces one ordered result per tool call and
checkpoints a resumable Session. Large results use private, bounded Session
attachments. Terminal progress is grapheme-width aware, and Word artifacts must
be rendered, applied, re-opened, and free of unsupported generated dates.

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

`pip install -e .` installs all three required runtime dependencies from
`pyproject.toml`: PyYAML plus the v0.11.0 `regex` and `wcwidth` additions used for
grapheme-safe terminal width. Browser, vector, semantic-index, and document
packages remain optional dependency groups.

Running `agent` without a task opens the interactive UI. Its startup banner
shows the key location check, `/resume`, `/undo`, permission modes, and concrete
task examples; `/help` expands the interactive workflow. `agent resume` may be
used without an extra prompt to enter the latest saved Session directly.

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
`downloads/`, `memory/`, and `tool-results/`. `.project-agent/.gitignore`
prevents those paths from being committed. On Linux filesystems the private
directories use mode `700`; WSL DrvFS mounts such as `/mnt/d` may display `777`
unless metadata mounting is enabled.

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
     -> ContextWindowController -> bounded compaction / output reservation
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
- `ContextWindowController`: estimates the complete request, reserves model
  output, protects recent API rounds, and selects automatic or deterministic
  compaction before the request is sent.
- `TaskConvergenceController`: closes repeated large/deep exploration while
  preserving implementation, verification, and final-answer rounds.
- `execute_model_tool_calls`: overlaps only consecutive, explicitly
  concurrency-safe pure reads; serial barriers preserve mutation order.
- `ToolResultStore`: keeps oversized, bounded result payloads in the current
  Session's private project storage and returns a hash-bearing preview.
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
agent resume
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
On a narrow PTY, the single-line progress display is cropped by Unicode
grapheme cluster and terminal cell width, so CJK, combining marks, flags, emoji
modifiers, keycaps, and ZWJ sequences are not split.

## Adaptive Execution Modes

```text
simple    short factual question; thinking off; 4-tool-turn soft target
standard  normal coding/analysis; high thinking; configured 8-turn soft target
large     repository/long document; chunked Task Graph; 16-turn soft target
deep      audit/refactor/root cause; max effort; 24-turn soft target
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
  convergence:
    enabled: true
    max_consecutive_exploration_rounds: 6
    reserved_tool_rounds: 4
    max_tool_calls_per_round: 16
    max_parallel_read_tools: 4
    max_length_continuations: 2
    max_implementation_evidence_reads: 2
    max_validation_attachment_reads: 2
    single_tool_result_chars: 12000
    same_round_tool_result_chars: 48000
    aggregate_tool_result_chars: 96000
    output_reserve_chars: 24000
    compacted_tool_result_chars: 1200
    keep_recent_tool_results: 4
    compaction_failure_limit: 3
    auto_compaction_enabled: true
    auto_compaction_max_tokens: 2048
    context_safety_buffer_tokens: 8192

model:
  provider: deepseek
  model: deepseek-v4-pro
  context_window_tokens: 65536
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

tools:
  tool_result:
    enabled: true
    max_attachment_bytes: 8388608
    persist_threshold_bytes: 12000
    preview_chars: 12000
    max_read_chars: 32000
    max_attachments_per_session: 512
    max_session_bytes: 268435456
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

## Reliable Tool Loop And Context Budget

The Agent loop counts completed tool-result batches, not every model request.
The selected 4/8/16/24 value is a soft tool-turn target; the default hard limit
is 32. A tool turn increases only after all calls from one assistant response
have one ordered `ToolResult` and the batch is checkpointed. Context compaction,
bounded `finish_reason=length` continuation, and final synthesis have separate
model-request counters and do not consume a tool turn. Tool execution closes at
the soft target only after the unified gate has the required plan, real
non-plan tool, explicit single-validation, and requested-artifact evidence.
Missing evidence is reported together and continues toward the hard limit;
the hard limit always stops tool execution.

`stop`, `tool_calls`, and a missing/empty finish reason are accepted. A
`content_filter` or unknown finish reason causes every accompanying tool call to
be discarded without execution; Runtime permits one corrective response and
then fails with the exact Resume command. Length-truncated tool-call JSON is
also never executed. After the tool loop closes, an independent tool-free final
synthesis phase may use bounded length continuations. This phase and the
soft/hard turn split are local v0.11.0 designs, not features attributed to the
reference project. A main-loop response that combines structured calls with
known DSML protocol text discards the complete call batch with zero execution.

Every normal request follows this bounded path:

```text
structured ToolResult hard limit
  -> same assistant-round aggregate limit
  -> complete tool-history micro/metadata compaction
  -> complete request estimate with Tool Schema + output reserve + safety buffer
  -> same-DeepSeek automatic summary of old complete rounds when needed
  -> deterministic emergency projection after repeated failure
  -> model request
```

Tool calls and matching results stay paired and contiguous. Missing, duplicate,
or out-of-order IDs are repaired only in the model-visible projection. AgentState
stores the bounded preview and attachment metadata; it does not store an
unlimited raw result. Results above 12,000 serialized bytes are written under
the current Session's private `tool-results/` directory, subject to an 8 MiB
per-attachment hard limit, count and total-byte quotas, SHA-256 metadata,
write-once request IDs, and symlink/path checks. `tool_result_read` returns only
bounded chunks. A persistence failure preserves an already successful side
effect and returns bounded head/tail, SHA-256, and
`attachment_persistence_error`; over-limit or upstream-truncated results do not
claim that a complete body is available.

Micro-compaction prefers old results, then metadata, then a deterministic hard
projection; if minimum records cannot fit, it collapses oldest complete
call/result groups into bounded system evidence rather than leaving half a
protocol pair. A compaction exception applies the deterministic hard limit on
the first failure. Three consecutive failures open the richer-path circuit; the
durable semantic-compaction breaker survives Resume until a successful compact
resets it. When that circuit is open, provider-overflow recovery also skips the
semantic model request and goes directly to deterministic fallback.

Automatic context summarization uses the configured DeepSeek model with tools
and Thinking disabled. It must reduce the request below the proactive trigger
or it is rejected. Disabling `auto_compaction_enabled` prevents this proactive
summary call but does not disable the hard request limit or output reservation;
a provider-reported context overflow still has its separately bounded semantic
recovery stage. Setting `runtime.convergence.enabled: false` additionally
disables convergence nudges,
same-round/history micro-compaction, and proactive summarization; per-result
limits, private attachment quotas, request budgeting, and emergency hard
projection remain active.

Consecutive calls overlap only when the capability is active, explicitly marked
`concurrency_safe=True`, has exactly read permission, and needs no confirmation.
All other calls are serial barriers, and returned results retain model-call
order. On `Ctrl+C` or another `BaseException`, completed results are retained and
unresolved/not-started calls receive synthetic failed results. Runtime
checkpoints the complete pair set, marks the Session interrupted, and re-raises
the original interruption. Subprocess tools bound head/tail output and terminate
their complete process group.

Large/deep execution tracks consecutive read-only rounds and repeated targets.
It first removes broad discovery tools, then closes targeted reading and the
Shell/Python aliases at the reserved-round boundary. Canonical capability names
and advertised model aliases share the same policy. In the closed window, at
most two 12,000-character chunks may be read only from a validation attachment
created in this Session by tests, diagnostics, document verification, staged
diff verification, or a non-exploratory shell validation, and only while
`implement` or `verify` is active. Ordinary exploration attachments stay closed.
Only a conditional-mutation `implement` step may be skipped; scope, inspection,
and verification remain required. Requested artifacts need managed-write
evidence, and Word artifacts additionally need re-open verification.
TaskRoute schema 2 stores at most 32 sanitized file and directory hints;
explicit directory names require matching `make_dir`. File/Word completion
replays active apply/delete/undo state and matching preview lineage, so an
artifact deleted after creation cannot still satisfy completion.
Artifact intent is evaluated per bounded clause, so credential-output bans and
ignored generated files do not create a false artifact requirement. A separate
explicit request to create a report still remains enforceable. Longer
"no sufficient evidence -> skip implement/do not modify" clauses are classified
as conditional mutation without weakening unconditional repair tasks.

The fixed `free/claude-code` snapshot informed session-scoped local tool-result
offloading with model-visible previews, explicit concurrency safety,
interruption pairing, grapheme display width, and bounded compression/recovery.
Deep Agent's private permission/path/quota attachment checks, soft/hard
tool-turn split, defensive
finish-reason handling, independent final synthesis, and completion-gate rules
were derived and tested locally. See `docs/releases/v0.11.0.md` for the exact
reference boundary and rollback constraints, and
`docs/v0.11.0-end-to-end-gap-analysis.md` for the evidence-by-evidence matrix.

AgentState schema 6 persists phase-specific model counters, convergence state,
model metrics, plan state, and bounded tool evidence. Resume preserves the
objective and frozen Session identity, but it is not a durable transaction log
or an exactly-once replay system; an external side effect interrupted mid-call
may still require user verification. v0.11.0 can migrate supported older state
when it is resumed. v0.10.0 rejects a schema 6 Session as a future schema, so
finish or export needed evidence before rolling code back.

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

`read_file` renders each source line as a right-aligned six-column line number,
the explicit `→` boundary, then the source text, for example
`     1→export const value = 1`. The arrow is not part of the file and prevents
source indentation from being confused with display padding in a later
`file_diff`. `run_tests framework=auto` checks language manifests before using a
generic `tests/` directory: Python markers, `package.json`, Cargo, Go, Gradle,
and Maven are considered in that order. For npm it runs only an existing
allowlisted script, preferring `test`, `typecheck`, `check`, `lint`, then `build`.
When the request explicitly allows only one validation attempt, TaskRouter adds
`single-validation`. After the first executed test, diagnostic, or recognized
validation shell command, equivalent validation schemas are removed and Runtime
denies shell/LSP/test substitutions. A failed environment or baseline check is
reported as the one allowed attempt instead of triggering a probe cascade.
`not_executed` preflight failures do not consume the attempt; common wrappers
are recognized and compound multi-validation shell commands are denied before
execution. Hard phase rejects value-bearing write/fix flag variants.

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
Node.js runtime, no Node.js 20 deprecation), then runs Ruff, the complete pytest
suite, and compileall for pushes to `main`, pull requests, and manual dispatch.
The frozen v0.11.0 candidate collected and passed 454 tests locally, including
18 CLI/Console tests, a real 25-column complex-grapheme progress PTY, a real
40-column Readline PTY, and the exact clause pattern from the failed large
acceptance run. A clean Python 3.14 environment installed
`.[dev,browser,semantic]`, passed `pip check`, and passed the same 454 tests.
The Word and plain-text online cases passed. The last large TypeScript
online case produced a complete evidence-based report but ended `failed`
because the old router falsely required a managed artifact and rejected the
conditional no-change path. The clause-bounded routing fix passed deterministic
4-main-loop + 1-final-synthesis regression; it was not rerun online to avoid
another high-token API charge, so this release does not claim a post-fix online
large-case pass.

A runnable Chinese walkthrough is available at
`user-docs/测试与验收/实用案例/v0.9.0/README.md`. Its order-summary project intentionally
starts with two failing business-rule tests so users can observe simple,
standard, large, deep, Thinking, and Resume behavior on a real repair task.
An offline interface example at
`user-docs/测试与验收/实用案例/v0.9.1/interface-routing-demo.py` demonstrates cost-aware
routing, TaskRoute-only plan creation, event correlation, and subscriber error
isolation without using an API key.
`user-docs/测试与验收/实用案例/v0.10.0/event-runtime-demo.py` demonstrates required
delivery, best-effort isolation, Memory usage idempotency, safe audit, and
bounded metrics entirely offline.

See `docs/implementation.md` for architecture, extension rules, Docker proxy,
OCR, memory, and maintenance details.

## Rollback

```bash
cd ~/AI-Agent
git switch --detach v0.10.0
.venv/bin/pip install -e .
```

Return with `git switch main`. v0.11.0 configuration migration only adds
defaults; do not delete Memory, `.project-agent`, or private tool-result data.
However, v0.10.0 cannot directly Resume an AgentState schema 6 Session and will
reject it as a future schema. Finish the Session under v0.11.0, or preserve it
and start a separate v0.10.0 Session after rollback.
