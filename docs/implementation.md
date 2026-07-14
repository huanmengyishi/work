# Deep Agent V3 Implementation and Extension Guide

## 1. Scope

Deep Agent V3 is a local, project-centric coding Agent powered only by DeepSeek.
It runs under WSL Ubuntu and can be started from any directory. Program files,
user configuration, long-term data, and project-local context remain separate.

Version `0.11.0` keeps the interfaces stabilized in v0.9.1. ContextBuilder
is the only context-selection entry, PromptBuilder only consumes a
`ContextPackage`, AgentState has a validated/frozen schema, and the versioned
Event Bus owns automatic Runtime side effects. The runtime remains DeepSeek-only
and preserves streamed Thinking from v0.8.0. Request preflight now bounds tool
evidence, reserves output space, compacts complete old API rounds, and forces
large/deep work from exploration into synthesis and verification.

## 2. Runtime Architecture

```text
CLI
  -> ProjectManager / ProjectRegistry
  -> AgentRuntime
     -> TaskRouter -> TaskPlanFactory
     -> ModelRouter (DeepSeek only, cost-aware)
     -> AgentState
     -> SessionManager
     -> ContextBuilder -> ContextPackage
     -> PromptBuilder
     -> ContextWindowController / ToolHistoryCompactor
     -> DeepSeekClient
     -> bounded tool-batch orchestrator
     -> ToolManager
        -> ToolCapabilityRegistry
        -> ToolRequest
        -> PermissionManager
        -> approval mode
        -> local / browser / MCP adapter
        -> ToolResult -> ToolResultStore
     -> EventBus
        -> required SessionEventPipeline / MemoryUsageEventPipeline
        -> best-effort MemoryPipeline / CapabilityHealthEventPipeline
        -> best-effort Audit / Metrics / Progress subscribers
```

`agent/cli.py` only parses commands and constructs the runtime. Orchestration
belongs in `agent/runtime.py`. Model-independent state belongs in
`agent/state.py`; context selection and budgeting belong in `agent/context.py`;
Prompt rendering belongs in `agent/prompt.py`.

The executable contract version and frozen field sets live in
`agent/contracts.py`. Full boundary ownership, compatibility rules, and the
Event delivery and side-effect ownership are documented in
`docs/architecture-v0.10.0.md`.

## 3. Program, Config, Data, and Project Files

```text
~/AI-Agent/
  agent/
  agent/tools/
  tests/
  docs/
  launcher/

~/.config/deep-agent/
  config.yaml
  model.yaml
  tools.yaml
  memory.yaml
  mcp.yaml
  secrets.env          mode 0600

~/.local/share/deep-agent/
  projects.db
  sqlite/memory.db
  memory/
  vector/
  cache/
  logs/
  metrics/
  capability-health/
  backup/

<project>/.project-agent/
  project.yaml
  context.md
  todo.md
  architecture.md
  ignore
  index.json
  sessions/*.json
  sessions/*.md
  tool-results/<session-id>/*.json
  cache/context.generated.md
  snapshots/<session-id>/
  browser-sessions/<name>/
  downloads/<name>/
  memory/<kind>/*.md
  .gitignore
```

The required runtime dependencies are `PyYAML`, `regex`, and `wcwidth`.
`regex` and `wcwidth` are new in v0.11.0 and implement grapheme-cluster-aware
terminal cropping; installing the project from `pyproject.toml` installs them.
Browser, vector, semantic, and document stacks remain optional groups. Private
`tool-results/` data is ignored by Git and must not be copied into a release.

V2 source was backed up before the upgrade:

```text
~/.local/share/deep-agent/backup/AI-Agent-v2-before-v3-20260712.tar.gz
SHA-256: 5ec7002ab0922bc1702470c0d669bff65d3a76607a75e98e34c630882899b056
```

## 4. Startup and Task Flow

1. `load_config()` creates missing defaults and loads `secrets.env`.
2. `ProjectManager` finds an existing `.project-agent`, then `.git`, then cwd.
3. The project UUID is loaded or generated and registered globally.
4. `ContextBuilder` scans durable context files and refreshes `index.json` only
   when the file fingerprint changes.
5. SQLite FTS and optional Chroma retrieve project and global memory.
6. `TaskRouter` locally classifies type, scale, risk, and execution mode.
7. `TaskPlanFactory` consumes the route when a local starter plan is required;
   it never scans or classifies the original Prompt.
8. `ModelRouter` maps that route to a fast, standard, or deep DeepSeek tier and
   records a low, balanced, or high cost class.
9. `AgentState` validates and stores both routes and resumable execution state.
10. `ContextBuilder` selects and bounds task, project, Session, semantic,
   Memory, recovery, and capability-summary sections into one Context Package.
11. `PromptBuilder` renders system policy, the Package, and the user request.
12. `ContextWindowController` estimates messages plus Tool Schema, reserves
    output and safety capacity, and applies bounded compaction when required.
13. DeepSeek either answers or emits tool calls.
14. Tool call IDs are normalized before execution; the model-visible protocol
    keeps at most 64 calls from one response and executes at most the configured
    per-round limit. Calls beyond the execution limit receive paired denied
    results rather than disappearing.
15. Every call becomes `ToolRequest`, passes centralized permission checks, and
    returns a structured `ToolResult`. Consecutive explicitly safe reads may
    overlap; all other calls remain serial barriers.
16. An oversized result is persisted in bounded Session-private storage before
    AgentState records its preview and attachment metadata. Model messages then
    receive the separate structured character limit.
17. State and messages are checkpointed after the complete contiguous result
    batch. Interruption also checkpoints real and synthetic failed pairs before
    re-raising the original `BaseException`.
18. A terminal event finalizes session files and triggers the idempotent memory
    pipeline.

Both routers are deterministic and consume no extra API request. Task routing
uses request markers, size, project file counts, prior failures, and configured
overrides. Model routing rejects non-DeepSeek providers and selects only among
configured DeepSeek model names.

Artifact routing evaluates positive action/object matches inside bounded
clauses and filters a negated action using its local prefix. An unrelated
"do not output credentials" or "ignore generated files" clause therefore
cannot create or globally suppress an artifact requirement elsewhere in the
request. Conditional mutation combines a bounded no-evidence signal with a
later skip/no-change signal; unconditional repairs do not receive that skip
permission.

## 4.1 Agent Loop, Termination, and Convergence

Runtime uses a `while` loop with two different counters. `tool_turn` advances
only after one assistant tool-call batch has a complete ordered result set and
is checkpointed. Context compaction, output continuation, corrective responses,
and final synthesis are model requests but are not tool turns. AgentState schema
6 records total, main-loop, context-compaction, and final-synthesis model-request
counts separately.

The mode's 4/8/16/24 tool-round value is a soft target. Tool execution closes at
that target only after the unified execution gate has the required Task Graph,
real non-plan tool, explicit single-validation, and requested-artifact evidence.
Missing evidence is reported together and continues toward the configured hard
limit, 32 by default; the hard limit closes tool execution unconditionally.

Runtime accepts `stop`, `tool_calls`, and an empty/missing finish reason. It
handles `finish_reason=length` with at most the configured number of tool-free
continuations, default two; incomplete tool-call JSON from the truncated
response is discarded. `content_filter` and unknown finish reasons are
unusable: any attached calls receive zero execution, Runtime asks for one
protocol-correct response, then fails and checkpoints the exact Resume command
if the problem repeats. A response is also rejected when it contains structured
or DSML tool-call protocol text. If structured calls and DSML text occur
together in the main loop, the complete call batch receives zero execution.

When tool execution closes, Runtime enters an independent tool-free final
synthesis phase. A length continuation remains in that phase and can make its
phase counter greater than one. The completion gate rejects empty/progress-only
answers, incomplete required plan steps, requested artifacts without active
managed-write evidence, and Word artifacts without matching render/apply
lineage plus re-open verification. TaskRoute schema 2 carries at most 32
sanitized file and directory hints; explicit directories require matching
`make_dir`, and later delete/undo records are replayed before an earlier apply
can count as active. Only the
`implement` step of a conditional-mutation plan may be skipped; scope,
inspection, and verification
cannot be skipped. The soft/hard split, defensive finish-reason handling,
independent final synthesis, and these plan rules are local designs verified in
this repository, not behavior attributed to the external reference snapshot.

`TaskConvergenceController` is active only for large/deep work. It counts
consecutive read-only rounds, repeated targets, plan changes, managed mutations,
and verification. It first removes broad discovery, then closes targeted reads
and Shell/Python exploration aliases at the continuous-read or reserved-round
boundary. Enforcement resolves advertised model aliases to canonical
capabilities, so an alias cannot bypass the phase policy. The hard phase permits
only narrowly bounded, previously read implementation evidence and validated
attachment exceptions described below.

## 4.2 Request Preflight, Compaction, and Overflow Recovery

`agent/convergence.py` owns deterministic request preparation. It never executes
tools or bypasses Capability and Permission ownership. With the default
convergence configuration enabled, before each model call:

1. `ToolResult.as_text()` produces valid bounded JSON for each model-visible
   result, retaining success, bounded stdout/stderr/data, original length, and a
   SHA-256 when space permits.
2. A second compactor bounds all results from one assistant response.
3. Pair repair restores one contiguous result per retained tool call in the
   model-visible projection; it does not mutate the underlying external world.
4. Complete-history compaction uses head/tail previews, stable metadata, then a
   deterministic per-result allocation. If minimum result records cannot fit,
   it removes oldest complete call/result groups and inserts bounded evidence;
   it never leaves a half pair.
5. `ContextWindowController.budget()` estimates serialized messages and Tool
   Schema, then reserves requested output and a safety buffer.
6. Above the proactive trigger, Runtime may summarize only complete old API
   rounds outside the protected recent tail. It uses the same configured
   DeepSeek model with `tools=None`, `tool_choice=None`, and Thinking disabled.
7. If the request still exceeds the hard input budget, or the semantic circuit
   is open above the trigger, a deterministic AgentState/evidence projection is
   applied. A request still over limit is rejected before it is sent.

A tool-history compaction exception invokes the deterministic hard limit on the
first failure; raw oversized evidence never passes through. The three-failure
circuit only stops retrying the richer path. The semantic-compaction counter and
circuit are persisted in AgentState and survive Resume; a successful compact
resets them. Neither disabling convergence nor disabling proactive automatic
summarization turns off the request hard limit or output reservation.
Turning off convergence does disable convergence nudges, same-round/history
micro-compaction, and proactive summarization; per-result limits, attachment
quotas, request budgeting, and emergency hard projection remain active.

Typed `DeepSeekContextOverflow` has two additional bounded recovery stages: a
cheap deterministic collapse followed by semantic compaction with deterministic
fallback. Each retry must reduce the estimated request; a third overflow fails
instead of looping forever. This provider-error recovery is separate from the
proactive `auto_compaction_enabled` switch and may make one bounded semantic
request even when proactive summarization is disabled. If the durable semantic
circuit is already open, this stage makes zero compaction model calls and goes
straight to deterministic fallback.

## 4.3 Tool Batches, Private Attachments, and Interruption

`agent/tool_orchestration.py` overlaps only consecutive calls whose capability
is active, explicitly declares `concurrency_safe=True`, has exactly `read`
permission, needs no confirmation, and is not a stateful read such as
`file.diff`. Every unmarked read, mutation, execution, network call, confirmation
gate, disabled/unavailable tool, and Runtime-denied call is a serial barrier.
Results are returned in model-call order and every call still travels through
`ToolRequest -> PermissionManager -> ToolResult`.

On `KeyboardInterrupt`, cancellation, or another `BaseException`, completed
results are retained. Running, cancelled, and not-yet-started calls receive
synthetic failed `ToolResult` records with the original request IDs. Runtime
appends the complete contiguous pair set, checkpoints the interrupted Session,
and re-raises the original exception. Subprocess adapters start a new process
group, drain stdout/stderr into bounded head/tail captures, and terminate the
whole group on timeout or interruption.

`ToolResultStore` persists results over 12,000 serialized bytes under
`.project-agent/tool-results/<session-id>/`. Defaults are an 8 MiB attachment
hard limit, a 12,000-character preview, 32,000-character ordinary read chunks,
512 files and 256 MiB per Session. Files/directories use private permissions;
paths, request IDs, write-once collisions, symlinks, file types, count, and total
bytes are checked, and reads return SHA-256 metadata. Source or attachment
truncation is explicit in metadata. AgentState stores the preview plus attachment
metadata, not an unlimited body. An attachment persistence error preserves the
truth of an already successful tool side effect and returns bounded head/tail,
SHA-256, and `attachment_persistence_error`; an over-8-MiB or upstream-truncated
result never claims that a complete body is available.

The closed exploration phase does not reopen arbitrary attachments. While an
`implement` or `verify` step is active, at most two reads of at most 12,000
characters each may target an attachment created in the same Session by
`run_tests`, diagnostics, document verification, staged-diff verification, or a
non-exploratory shell validation. Unknown, ordinary exploration, oversized, or
exhausted requests are denied with paired results.

## 5. Agent State, Plans, and Resume

`AgentState` stores project identity, current request, working directory, Git
branch, loaded memory IDs, loaded tools, plan, current step, completed steps,
bounded tool evidence and attachment metadata, status, round, turn, structured
Task/Model routes, convergence state, model metrics, phase-specific model
request counters, and a compact Context Package manifest.

`AgentState.validate()` checks the published serialized field order, frozen
Session identity, supported schema version, timestamps, plan graph, derived
step fields, DeepSeek route, cost class, counters, convergence metadata, and
Context manifest bounds. The current AgentState schema is 6. Supported older
state is normalized and upgraded to schema 6 on Resume; an unknown future
schema fails closed.

The model can call:

```text
agent_update_plan
agent_update_step
```

V3 does not force an extra planning API call for every task. Large/deep work
receives a local starter plan from `TaskPlanFactory`; the same DeepSeek tool
loop refines it. TaskRouter is the only classifier. `task_strategy.py` is a
deprecated compatibility facade and Runtime does not instantiate it. On
Resume, `more_capable_task_route()` retains the previous route unless the new
request is more capable or higher risk. `more_capable_model_route()` retains
the exact previous DeepSeek model at equal/lower tiers and changes it only for
a tier upgrade. A short “continue” therefore cannot downgrade the Session.

Automatic model tiers are capability policies:

```text
fast      thinking off
standard  thinking on, reasoning_effort=high
deep      thinking on, reasoning_effort=max
```

The persisted cost classes are:

```text
low       simple, low-risk request
balanced  standard work or large read-only work
high      deep/high-risk/architecture/refactor/repeated-failure work
```

They explain local resource selection and are not token-price calculations.

`model.routing.fast_model`, `standard_model`, and `deep_model` default to
`null`. Consequently all three tiers use `model.model` by default; distinct
DeepSeek model names are used only when explicitly configured. Setting
`model.provider` to anything other than `deepseek` is rejected.

### Thinking, streaming, and timeout recovery

Thinking uses DeepSeek's OpenAI-compatible `thinking` and `reasoning_effort`
fields selected by the saved Model route. Tool-calling rounds preserve
`reasoning_content` in the assistant message as required by DeepSeek. SSE
deltas are reassembled by tool-call index.

The terminal shows elapsed time before the first byte and keeps streamed
reasoning separate from the carriage-return spinner. On a narrow PTY, progress
is truncated by Unicode grapheme cluster and `wcwidth` display cells rather than
Python code points, so CJK, combining marks, flags, emoji modifiers, keycaps, and
ZWJ sequences are not split.

Timeouts, connection resets, incomplete reads, TLS record-layer failures, and
transient 408/5xx responses receive bounded same-key retry. A streaming failure
is retryable only before any valid delta. Once reasoning, content, or a tool-call
delta is observed, the request is not replayed because that could duplicate
model work or a future side effect; the Session is saved as resumable and its
exact ID is reported.

Large inputs follow scope -> bounded chunks -> synthesize/implement -> verify.
Context size, file count, line reads, tool output, model rounds, and displayed
reasoning are all bounded.

Resume commands:

```bash
agent sessions
agent resume "continue the latest session"
agent resume --session SESSION_ID "continue with tests"
```

Each resumed request increments `turn`. Memory classification only processes
tool evidence from the current turn, preventing duplicate lessons. Resume keeps
the frozen objective/project identity, routes, plan, durable target set,
semantic-compaction breaker, bounded results, and private attachment references,
while reopening per-turn read/stall allowances and resetting turn metrics.

This is checkpoint continuation, not full transaction replay. There is no
Durable Intent Journal or exactly-once guarantee for an external side effect
interrupted between the operating-system action and its ToolResult checkpoint;
the user must verify such state before retrying. v0.10.0 also cannot directly
load or Resume a schema 6 Session because that version correctly rejects future
AgentState schemas. Complete needed work under v0.11.0 before a code rollback,
or start a separate v0.10.0 Session.

## 6. Context and Source Index

`ContextBuilder` reads durable project files such as:

```text
.project-agent/context.md
.project-agent/architecture.md
.project-agent/todo.md
README.md
CLAUDE.md
AGENTS.md
pyproject.toml
package.json
Cargo.toml
go.mod
pom.xml
build.gradle
project.godot
.gitignore
```

The lightweight `.project-agent/index.json` contains language, likely entry
points, file metadata, and Python/general source symbols. It is intentionally
not a full language-server database. Optional Tree-sitter semantic data remains
a bounded sidecar and never replaces the base index.

`ContextBuilder.build_package()` is the only supported path for model-visible base
context. It produces typed sections, records omitted/truncated sources, and
computes `used_chars` from the final rendering including headings and section
separators. Task state and project instructions are selected first; other
sources are admitted at complete record, line, or paragraph boundaries.

Default Package limits are selected by Task mode and clamped by a hard limit:

```yaml
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

The separate `runtime.max_user_request_chars` input ceiling defaults to 250000.
Larger pasted input is rejected with guidance to save it as a project file, so
the Agent can inspect the complete content through bounded file chunks instead
of silently discarding most of one message.

`used_chars` counts the bounded user request plus rendered Context Package
sections. Oversized requests preserve bounded head/tail content. The fixed
system prompt, active tool JSON schemas, and `ToolResult` messages added in
later rounds are not counted. They retain their own limits, so the Package
limit must not be described as a bound on the entire API payload.

`.project-agent/cache/context.generated.md` contains only generated project
context. Long-term Memory, Session outcomes, and failure-recovery text stay in
the in-memory Package and Session checkpoint. Resume rebuilds the Package from
current project data and a bounded previous outcome instead of restoring an
unlimited raw tool transcript.

PromptBuilder has only `build_initial(package)` and `build_resume(package)`.
It imports no State/Snapshot builder and performs no file I/O. This is an
intentional compatibility break for external v0.8 callers that passed separate
context parameters.

### Event contract and migrated side effects

`Event` schema version 1 serializes the following stable fields:

```text
schema_version, id, name, timestamp,
project_id, session_id, run_id, payload
```

`EventBus.subscribe()` returns a cancellation callback and accepts a stable
subscriber name plus `required=True` ownership. `unsubscribe()` is idempotent,
`publish()` accepts either a name plus metadata or an existing Event, and
subscriber exceptions are isolated in delivery results. `dispatch_required()`
fails closed when an exact required owner is missing or a required handler
fails; exact best-effort and wildcard Audit/Metrics handlers never count as an
owner. Subscriber names are unique within one event so delivery evidence cannot
be confused with another handler.

`RuntimeEventPipelines` is the only automatic subscriber-registration point:

- `session.checkpoint.requested` and `session.finalize.requested` are required;
- `memory.usage.recorded` is required and commits one `usage_id` atomically;
- `task.finished/task.failed` drive idempotent automatic Memory/Reflection;
- `tool.finished` updates capability health best-effort;
- Audit, aggregate Metrics, and `ui.progress.updated` are best-effort.

Runtime persists Session state before it publishes a terminal task event. A
failed Session finalize therefore does not generate Memory, Audit, or Metrics
from an unpersisted terminal state. Memory usage updates SQLite before adding
the ID to `AgentState.loaded_memories`; `memory_usage_events` prevents a replay
from incrementing `use_count` twice. Tool permission checks and the handler
still run before `tool.finished`, so Event subscribers cannot execute or bypass
a capability.

Audit JSONL uses an allow-list metadata projection. It never serializes live
AgentState, messages, Prompt, reasoning, tool argument values, stdout/stderr,
result bodies, or arbitrary objects. Metrics persists only allowed event
counts, bounded aggregate tool duration, and failed-tool count. Both use
private paths, atomic/bounded writes, and reject symbolic-link destinations.

The bus is synchronous and process-local. There is no replay, network broker,
delivery retry, cross-process ordering, or exactly-once process-crash guarantee
in this release. See
`docs/architecture-v0.10.0.md` before adding subscribers or
moving side effects.

## 7. Tool Protocol and Capability Registry

All model calls use a uniform internal protocol:

```python
ToolRequest(tool="shell", action="run", args={"command": "pytest"})
ToolResult(success=True, stdout="...", stderr="", duration_ms=120)
```

Capabilities register model name, description, JSON schema, permissions,
timeout, streaming support, input/output formats, enabled state, local
dependency availability, confirmation policy, and an explicit
`concurrency_safe` flag. That flag defaults to false and cannot by itself bypass
the orchestrator's strict pure-read checks. Only active capabilities enter the
DeepSeek tool schema.

To add a tool:

1. Add a narrow adapter under `agent/tools/`.
2. Return `ToolResult`; do not expose raw subprocess behavior to Runtime.
3. Register `ToolCapability` and its handler in `ToolManager`.
4. Add config defaults under `tools.capabilities`.
5. Add permission rules only when generic policy is insufficient.
6. Add isolated tests for schema, policy, execution, and unavailable dependency behavior.

This protocol is also the future adapter boundary for MCP and remote tools.

## 8. Permission Policy

The centralized policy currently enforces:

- Disabled or unavailable capabilities are denied.
- Tool working directories stay inside the current project by default.
- Model-supplied timeouts cannot exceed capability limits.
- `sudo`, `su`, shutdown, disk formatting, dangerous root deletion, Docker
  `--privileged`, and host-root Docker mounts are denied by default.

Approval levels are explicit:

- Safe: confirmation-gated tools call the interactive approval handler.
- Auto approve: only names listed under `permissions.auto_approve_capabilities`
  bypass confirmation; defaults are snapshot-backed `file.apply/file.undo`.
- YOLO: all confirmations are skipped, but hard Permission Manager rules remain.
- SUPER YOLO: confirmations and Permission Manager hard rules are bypassed.
  Capability enablement, dependency availability, JSON argument shape, and OS
  authentication still apply.

Use `agent --yolo`, `agent --super-yolo`, `/yolo on|off`, or
`/super-yolo on|off`. Persistent config keys are `permissions.yolo` and
`permissions.super_yolo`. SUPER YOLO is intentionally visible in prompts and
`agent doctor`.

## 9. Safe File Editing And Snapshots

The `file_diff`, `file_apply`, and `file_undo` capabilities form one protocol.
Apply accepts only a stored preview ID. Before writing it verifies the base
SHA-256, stores the exact original bytes and Git metadata under the active
session, performs an atomic replacement, and verifies the result hash. Undo
checks that the target still matches the applied hash, so newer user work is
not overwritten. This is non-destructive for dirty Git repositories; no default
stash, branch, or worktree changes are made.

## 10. Safe Templates

`list_dir`, `find_files`, `search_code`, `read_file`, `git_diff_staged`, and
`run_tests` use separated process arguments or direct Python APIs, never shell
string interpolation. Project private directories containing snapshots,
sessions, tool-result attachments, and browser identity state cannot be read
through ordinary project tools.

`read_file` renders each line as `{line_number:>6}→{source}`, for example
`     1→export const value = 1`. The arrow is an output boundary, not file
content. It makes source indentation mechanically distinct from display padding
when the model prepares `file_diff.old_text`.

`run_tests framework=auto` chooses a language marker before treating a generic
`tests/` directory as pytest: `pyproject.toml`/`pytest.ini`, `package.json`,
`Cargo.toml`, `go.mod`, `gradlew`, and `pom.xml` are checked first. npm execution
is argument-separated and restricted to an existing `test`, `typecheck`,
`check`, `lint`, or `build` script, in that preference order.

An explicit “run validation only once” request adds the deterministic
`single-validation` route reason. Once one non-Runtime-denied validation attempt
is recorded, Runtime removes registered test/diagnostic schemas and rejects
recognized shell validation substitutions. Failure from missing dependencies or
pre-existing project errors still consumes the requested single attempt and is
reported honestly. Handler-preflight `not_executed` results do not consume it;
same-batch calls are decided sequentially from actual ToolResults. Wrappers such
as `uv run`, `npm --prefix`, `timeout`, and `python -I -m pytest` are recognized,
while compound commands containing more than one validation are denied before
execution. Hard phase also rejects value-bearing write/fix variants including
`--write=true`, `--apply=yes`, `--update=...`, `--bless=...`, `--accept=...`,
`-wfoo`, and `--install-types`.

## 11. MCP Client

`MCPManager` implements stdio JSON-RPC initialize, paginated `tools/list`, and
`tools/call`. Remote schemas become ToolCapabilities and therefore reuse
Permission Manager, confirmation, ToolRequest/ToolResult, events, and state.
Configuration is opt-in and add-only; a disabled SQLite example is provided.
Allowlist filtering, 10-server/80-tool defaults, minimal environment inheritance,
timeouts, graceful server failure, and key isolation are enforced. Use
`agent mcp status/tools/config` to inspect it.

## 12. Browser Persistence And Downloads

Named Playwright contexts persist project-local cookies, LocalStorage, and
permissions. Browser calls close the process each time but retain the profile.
Downloads are deduplicated and return path, MIME type, byte size, and source URL.
Only HTTP(S) URLs without embedded credentials are accepted.

## 13. Memory Pipeline

Terminal task events follow this pipeline:

```text
task.finished / task.failed
  -> Summary
  -> deterministic classification
     -> Lesson / Bug / Decision
  -> SQLite + FTS
  -> Chroma when enabled
  -> project-local Markdown experience file
  -> pipeline_runs idempotency marker
```

Classification is deterministic in V3 to avoid a second model call after every
task. It can later be replaced by a DeepSeek classifier behind the same
`MemoryPipeline` interface when higher semantic quality justifies the cost.

## 14. OCR and Documents

`DocumentTool` first calls the existing local parser:

```text
~/.local/bin/ai-parser
~/.local/share/ai-tools/app/parser.py
```

The launcher now prefers `~/AI-Agent/.venv/bin/python`, where PyMuPDF,
python-docx, Pillow, and Docling are installed. Fallbacks remain `pdftotext`,
Tesseract, and ImageMagick. Output to DeepSeek is Markdown.

Verified path on 2026-07-12:

```text
DocumentTool -> ai-parser -> Tesseract -> Markdown
```

## 15. Docker and WSL Clash Proxy

Docker 29.1.3 is installed and `docker run --rm hello-world` succeeds.

User-space networking uses the existing `HTTP_PROXY`, `HTTPS_PROXY`, and
`ALL_PROXY` environment variables. DeepSeek's HTTP client, Shell tools, Git,
pip/uv/npm, and other subprocesses inherit them. Playwright reads the same
variables and passes the proxy explicitly to Chromium. `agent doctor` displays
the proxy endpoint with any credentials removed.

The daemon proxy is generated from the current WSL default gateway before every
Docker start, so WSL IP changes do not require editing a hard-coded host:

```text
/etc/default/deep-agent-proxy
/usr/local/lib/deep-agent/configure-docker-proxy
/etc/systemd/system/deep-agent-docker-proxy.service
/etc/systemd/system/docker.service.d/proxy.conf
/run/deep-agent/docker-proxy.env
```

The default Clash port is `7897`. To change it:

```bash
sudo nano /etc/default/deep-agent-proxy
sudo systemctl daemon-reload
sudo systemctl restart docker
```

Verify with:

```bash
docker info --format 'http={{.HTTPProxy}} https={{.HTTPSProxy}}'
docker run --rm hello-world
agent doctor
```

## 16. DeepSeek API Key

Recommended location:

```text
~/.config/deep-agent/secrets.env
```

Content (a single key or comma-separated Key pool):

```bash
DEEPSEEK_API_KEY=key_1,key_2,key_3
```

Then:

```bash
chmod 600 ~/.config/deep-agent/secrets.env
agent doctor --online
```

English commas (`,`) and Chinese commas (`，`) are accepted. The loader trims
whitespace, ignores empty entries, and removes duplicate keys without logging
their values. A normal model request tries the next key only after HTTP `401`,
`403`, or `429`. Retryable transport failures and HTTP `408`, `500`, `502`,
`503`, and `504` retry the same key with bounded exponential backoff; other
errors fail without hiding the cause. Streaming retry additionally stops as soon
as one valid delta has been observed. A successful request advances the next
starting key for the current process. `agent doctor --online` checks every key
and reports `ready/total` only. Values in `secrets.env` override the legacy shell
environment. Never put keys in `model.yaml`, project files, source code, logs,
or Git.

## 17. Version 0.5.0 Extensions

- Correction memory requires `correction:<topic>` and receives the project tag.
- Failed tools locally retrieve relevant Correction/Lesson records once per run.
- Memory get/list/edit/delete/stats synchronizes SQLite FTS and Chroma.
- MCP supports stdio, Streamable HTTP, legacy SSE, and opt-in Resources.
- Restricted HTTP uses one activation switch, domain allowlists, 30s/1 MiB
  limits, and rejects sensitive headers and redirects.
- Optional Tree-sitter indexing remains a sidecar and contributes a bounded
  context summary.
- Persistent queues create one Session per task and resume unfinished entries.
- Parallel worktree execution requires at least eight tasks and clean Git,
  extracts patches from one baseline, checks conflicts, and cleans resources.

## 18. Verification

Run the release checks locally without exposing credentials:

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check agent tests scripts
.venv/bin/ruff format --check agent tests scripts
.venv/bin/python -m compileall -q agent tests scripts
```

`.github/workflows/test.yml` uses `actions/checkout@v5` and
`actions/setup-python@v6`, installs browser/semantic integration dependencies,
and runs Ruff, the complete pytest suite, and compileall for Python 3.11, 3.12,
and 3.13. The frozen v0.11.0 candidate collected and passed 454 tests locally,
including real PTY interaction, grapheme width, tool interruption/pairing,
attachments, convergence, finish reasons, Resume, and real local validation
command selection. The CLI/Console subset has 18 tests; real PTYs cover a
25-column complex-grapheme progress line and 40-column Readline input. A clean
Python 3.14 environment installed `.[dev,browser,semantic]`, passed `pip check`,
and passed the same 454 tests. The online Word and plain-text cases passed. The last large
TypeScript run generated a complete report but the pre-fix router falsely
classified a prohibited credential-output clause as an artifact request and
missed the conditional no-change clause, so its Session failed the managed-write
gate. The exact prompt pattern now passes deterministic routing and 4-main-loop
+ 1-final-synthesis regression. It was not rerun online to avoid another
high-token charge; run `agent doctor --online` only with private credentials
present.

## 19. Reference Decisions

The v0.11.0 reliability comparison used the fixed `claude`-branch snapshot of
`https://gitee.com/free/claude-code` at commit
`b17913e26fd4278ad5cd4b32ed3bde86bf1444e9` (tree
`68ef0259999328e531cc23c81fc80e81cbdabecb`), reviewed 2026-07-15. Directly
informed mechanisms are session-scoped local tool-result offloading with
model-visible previews, explicit concurrency safety, interrupted-call result
pairing, grapheme/display-width rendering, and compression with bounded
recovery. The private permission/path/quota attachment checks and deterministic
emergency projection are local Deep Agent designs.

The code in this repository remains an independent implementation behind
Deep Agent's existing Capability and Permission boundaries. The reference's
multi-provider abstractions, full-screen TUI, plugin ecosystem, and permission
model were not added. Deep Agent's soft/hard tool-turn split, independent
tool-free final synthesis, unknown-finish-reason zero-execution rule,
conditional-plan completion gate, manifest-first validation selection, and
schema-6 Resume state are local designs backed by this repository's tests. In
particular, final synthesis must not be described as copied or natively provided
by that reference snapshot.

The complete source-path matrix, including known non-equivalences and both
large-acceptance root-cause chains, is in
`docs/v0.11.0-end-to-end-gap-analysis.md`.

ECC (`affaan-m/ECC`, reviewed 2026-07-12) informed four choices: MCP connectors
remain opt-in, active connector/tool counts are bounded, configuration examples
are add-only, and external/session state has versioned JSON records and health
status. ECC's multi-agent, hook, and full plugin ecosystem were not copied; they
would add disproportionate complexity to this single-model local CLI.

## 20. Rollback Compatibility

The previous code tag is `v0.10.0`:

```bash
git switch --detach v0.10.0
.venv/bin/pip install -e .
```

Configuration migration is add-only, so no configuration or private data needs
to be deleted. Do not delete Memory, `.project-agent`, Session files, or
tool-result attachments while testing rollback. AgentState compatibility is
one-way at this boundary: v0.11.0 can normalize supported older state, but
v0.10.0 cannot directly load or Resume a schema 6 Session. Finish it with
v0.11.0 or start a separate v0.10.0 Session. Return with `git switch main`.

## 21. Deferred Work

The following were intentionally not implemented in this upgrade:

- General multi-agent role scheduling.
- Durable Intent Journal, replay, or a cross-process Event broker.
- Worker Runtime and complete Runtime redesign.
- GitHub/Jira/Notion integrations.
- Non-DeepSeek providers.
- A full semantic call/reference graph or LSP database.
- Web UI and remote collaboration.
- MCP prompts, subscriptions, and resource change notifications.

These additions can use `AgentState`, `EventBus`, `ToolRequest/ToolResult`, and
the capability registry without rewriting CLI or the main runtime.
