# Deep Agent V3

Project-centric DeepSeek CLI agent for WSL/Linux. Start `agent` inside the
project you want it to inspect or modify.

Current release: `0.13.0` · AgentState schema: `8` · core interface contract: `5`

DeepSeek remains the only inference provider. System actions continue through
`ToolRequest -> PermissionManager -> ToolResult`; the model and Runtime do not
bypass the Tool Manager.

Latest Chinese documents:

- [Usage guide (Markdown)](user-docs/DeepSeek-Agent-V3-使用说明.md)
- [Work log (Markdown)](user-docs/DeepSeek-Agent-V3-工作日志.md)
- [Usage guide (Word)](DeepSeek-Agent-V3-使用说明-0.13.0.docx)
- [Work log (Word)](DeepSeek-Agent-V3-工作日志-0.13.0.docx)

## What changed in 0.13.0

- A successful task can make at most one extra, tool-free, budgeted DeepSeek
  logical request to refine the lesson learned. Bounded transport retries and
  key rotation still apply. `memory.smart_reflection` defaults to `true` when
  the key is missing, but fewer than five managed tool calls in the current
  turn always skip refinement. Failure never changes the completed task.
- Runtime, convergence, history handling, tool declarations, and tool execution
  are split into bounded modules while preserving one Runtime and one tool path.
- AgentState keeps at most 200 hot tool records and applies byte, collection,
  key, and nesting limits. Resume loads only the newest 16 complete model rounds;
  complete transcripts remain in bounded, hashed cold generations.
- Memory gains a bounded TTL/LRU query cache, strict JSON import/export, global
  knowledge isolation, and model-produced kind/confidence/tags persistence.
- Zero-model History Snip runs before model compaction. Capability backoff,
  typed task-plan strategies, durable intent journals, and artifact lineage are
  wired into the existing execution path.
- Adaptive convergence, strategy adjustment, and deterministic experiments are
  available but remain disabled by default.
- The GitHub tree contains the application, tests/CI, the current release note,
  and only the latest concise user documents. Detailed development evidence and
  historical reports remain local and are not published.

See [the 0.13.0 release note](docs/releases/v0.13.0.md) for compatibility notes.

## Install and run

```bash
cd ~/AI-Agent
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
.venv/bin/agent --version
```

Optional heavy capabilities are installed explicitly:

```bash
.venv/bin/python -m pip install -e '.[browser,semantic,document,vector]'
```

Store the private key outside the source tree:

```bash
nano ~/.config/deep-agent/secrets.env
chmod 600 ~/.config/deep-agent/secrets.env
```

```text
DEEPSEEK_API_KEY=replace_with_your_valid_key
```

English and Chinese commas are accepted for a key pool. Key values are
never shown by `doctor`; `secrets.env` takes priority over a legacy shell value.

Run an offline check before any paid request:

```bash
agent doctor
cd /path/to/project
agent init
agent "inspect this project and verify the requested change"
```

`agent doctor --online` validates every configured key and transient failures
may be retried, so it can send multiple real DeepSeek requests. Use it only when
an online credential check is intentionally required.

## Memory refinement

The default for a new or missing configuration key is:

```yaml
memory:
  smart_reflection: true
  smart_reflection_min_tool_calls: 5
  smart_reflection_max_input_chars: 12000
  smart_reflection_max_output_tokens: 768
  smart_reflection_max_output_chars: 5000
```

The hard trigger floor is five managed tool calls in the current turn; setting
a smaller configuration value cannot lower it. Refinement runs only after a
successful completion, uses the injected DeepSeek client with tools disabled,
shares the turn budget, and makes at most one logical request per turn. Existing
explicit `false` values are preserved during migration. Disable it with
`smart_reflection: false` when the extra completion-time request is not desired.

## Common commands

```bash
agent --help
agent --version
agent doctor
agent init
agent "task"
agent sessions
agent resume
agent resume --session SESSION_ID "continue"
agent context show
agent context refresh
agent tools --all
agent health
agent memory search "query"
agent memory list --kind Correction
agent memory stats
agent memory maintain
agent memory maintain --apply
agent memory export backup.json --scope project
agent memory import backup.json --target-scope preserve --conflict skip
```

Memory export files contain real Memory content. Keep them private and out of
Git; use `--force` or `--conflict replace` only after making a separate backup.

Interactive commands include `/new`, `/resume`, `/sessions`, `/status`, `/undo`,
`/yolo on|off`, `/super-yolo on|off`, `/help`, and `/exit`. Press `Enter` once to
submit. Empty input gives feedback, `Ctrl+C` returns to a recoverable prompt, and
the terminal shows processing/thinking progress while a request is active.

Exit codes are `0` for completed, `2` for saved resumable incomplete, and `1`
for configuration or Runtime failure.

## Optional strategy controls

```yaml
optimizer:
  adaptive_convergence_enabled: false
  strategy_adjustment_enabled: false
  experiments:
    enabled: false

runtime:
  convergence:
    history_snip_enabled: true
```

Adaptive and experimental behavior is opt-in. Vector search is also disabled by
default and remains lazy even when enabled. Configuration migration only adds
missing defaults; it does not replace existing user choices.

## Data ownership and safety

```text
~/AI-Agent/                         application, tests, current public docs
~/.config/deep-agent/              private configuration and API key
~/.local/share/deep-agent/         Memory, vectors, logs, metrics, shared data
<project>/.project-agent/          project context, checkpoints, attachments
```

Do not commit keys, Memory, Session transcripts, logs, browser state, caches, or
`.project-agent` data. File mutations use snapshot-backed `file.diff`,
`file.apply`, and `file.undo`. Network and dangerous actions remain permission
controlled; `--super-yolo` deliberately relaxes hard policies and should be
treated as unsafe.

## Verify a checkout

```bash
.venv/bin/ruff check agent tests scripts
.venv/bin/ruff format --check agent tests scripts
.venv/bin/python -m compileall -q agent tests scripts
.venv/bin/python -m pytest
.venv/bin/python -m pip check
git diff --check
```

The test suite includes real PTY coverage for Enter, processing feedback,
Unicode-width rendering, and recoverable `Ctrl+C`. Release tests also reopen the
generated Word files and enforce the public document tree.

## Roll back

Finish or preserve 0.13.0 sessions before running an older version; older code
does not understand schema 8 checkpoints.

```bash
cd ~/AI-Agent
git switch --detach v0.12.2
.venv/bin/python -m pip install -e .
```

Do not delete Memory, Session, vector, log, or `.project-agent` data during a
rollback. Return with `git switch main` and reinstall the editable package.
