# Deep Agent V3 Implementation and Extension Guide

## 1. Scope

Deep Agent V3 is a local, project-centric coding Agent powered only by DeepSeek.
It runs under WSL Ubuntu and can be started from any directory. Program files,
user configuration, long-term data, and project-local context remain separate.

Version `0.9.0` adds deterministic Task and DeepSeek-only Model routers plus a
single bounded `ContextPackage` entry for Prompt rendering. It preserves the
capability and permission boundaries and the streamed Thinking behavior from
v0.8.0.

## 2. Runtime Architecture

```text
CLI
  -> ProjectManager / ProjectRegistry
  -> AgentRuntime
     -> TaskRouter
     -> ModelRouter (DeepSeek only)
     -> AgentState
     -> SessionManager
     -> ContextBuilder -> ContextPackage
     -> PromptBuilder
     -> DeepSeekClient
     -> ToolManager
        -> ToolCapabilityRegistry
        -> ToolRequest
        -> PermissionManager
        -> approval mode
        -> local / browser / MCP adapter
        -> ToolResult
     -> EventBus
        -> JsonlEventLogger
        -> MemoryPipeline
           -> SQLite FTS
           -> Chroma
           -> Markdown memory files
```

`agent/cli.py` only parses commands and constructs the runtime. Orchestration
belongs in `agent/runtime.py`. Model-independent state belongs in
`agent/state.py`; context selection and budgeting belong in `agent/context.py`;
Prompt rendering belongs in `agent/prompt.py`.

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
  cache/context.generated.md
  snapshots/<session-id>/
  browser-sessions/<name>/
  downloads/<name>/
  memory/<kind>/*.md
  .gitignore
```

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
7. `ModelRouter` maps that route to a fast, standard, or deep DeepSeek tier.
8. `AgentState` stores both routes and the resumable execution state.
9. `ContextBuilder` selects and bounds task, project, Session, semantic,
   Memory, recovery, and capability-summary sections into one Context Package.
10. `PromptBuilder` renders system policy, the Package, and the user request.
11. DeepSeek either answers or emits tool calls.
12. Every call becomes `ToolRequest`, passes centralized permission checks, and
   returns structured `ToolResult` with stdout, stderr, data, and duration.
13. State and messages are checkpointed after each tool call.
14. A terminal event finalizes session files and triggers the idempotent memory
    pipeline.

Both routers are deterministic and consume no extra API request. Task routing
uses request markers, size, project file counts, prior failures, and configured
overrides. Model routing rejects non-DeepSeek providers and selects only among
configured DeepSeek model names.

## 5. Agent State, Plans, and Resume

`AgentState` stores project identity, current request, working directory, Git
branch, loaded memory IDs, loaded tools, plan, current step, completed steps,
tool evidence, status, round, turn, structured Task/Model routes, and a compact
Context Package manifest.

The model can call:

```text
agent_update_plan
agent_update_step
```

V3 does not force an extra planning API call for every task. Large/deep work
receives a local starter plan; the same DeepSeek tool loop refines it. On
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

`model.routing.fast_model`, `standard_model`, and `deep_model` default to
`null`. Consequently all three tiers use `model.model` by default; distinct
DeepSeek model names are used only when explicitly configured. Setting
`model.provider` to anything other than `deepseek` is rejected.

### Thinking, streaming, and timeout recovery

Thinking uses DeepSeek's OpenAI-compatible `thinking` and `reasoning_effort`
fields selected by the saved Model route. Tool-calling rounds preserve
`reasoning_content` in the assistant message as required by DeepSeek. SSE
deltas are reassembled by tool-call index.

The terminal shows elapsed time before the first byte. Network timeouts and
transient 408/5xx responses receive bounded same-key retry. If a stream breaks
after partial output, it is not replayed because that could duplicate an
in-flight tool call; the Session is finalized as resumable and its exact ID is
reported.

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
tool evidence from the current turn, preventing duplicate lessons.

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

`ContextBuilder.build_package()` is the only v0.9 path for model-visible base
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

## 7. Tool Protocol and Capability Registry

All model calls use a uniform internal protocol:

```python
ToolRequest(tool="shell", action="run", args={"command": "pytest"})
ToolResult(success=True, stdout="...", stderr="", duration_ms=120)
```

Capabilities register model name, description, JSON schema, permissions,
timeout, streaming support, input/output formats, enabled state, and local
dependency availability. Only active capabilities enter the DeepSeek tool
schema.

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
sessions, and browser identity state cannot be read through project tools.

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
`403`, or `429`; network failures and other HTTP errors stop immediately to
avoid masking service failures. A successful request advances the next starting
key for the current process. `agent doctor --online` checks every key and
reports `ready/total` only. Values in `secrets.env` override the legacy shell
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
and runs the same Ruff, 113-test, and compileall checks for Python 3.11, 3.12,
and 3.13. Hosted run `29257906807` passed for all matrix entries. This does not
claim an online DeepSeek API check; run `agent doctor --online` only with
private credentials present.

## 19. ECC Reference Decisions

ECC (`affaan-m/ECC`, reviewed 2026-07-12) informed four choices: MCP connectors
remain opt-in, active connector/tool counts are bounded, configuration examples
are add-only, and external/session state has versioned JSON records and health
status. ECC's multi-agent, hook, and full plugin ecosystem were not copied; they
would add disproportionate complexity to this single-model local CLI.

## 20. Deferred Work

The following were intentionally not implemented in this upgrade:

- General multi-agent role scheduling.
- A full semantic call/reference graph or LSP database.
- Web UI and remote collaboration.
- MCP prompts, subscriptions, and resource change notifications.

These additions can use `AgentState`, `EventBus`, `ToolRequest/ToolResult`, and
the capability registry without rewriting CLI or the main runtime.
