# Deep Agent v0.10.0 Event Runtime Architecture

## Purpose and unchanged boundaries

Version 0.10.0 completes the in-process migration of automatic Runtime side
effects to the Event Bus. It keeps the v0.9.1 public chains unchanged:

```text
CLI -> Runtime -> AgentState -> Prompt -> Capability -> Permission
ContextBuilder -> ContextPackage -> PromptBuilder -> DeepSeek
ToolRequest -> PermissionManager -> ToolResult
```

DeepSeek remains the only model provider. Events do not execute tools, query
state, replace ContextBuilder, or bypass PermissionManager. This release does
not add multiple Agents, Worker Runtime, a network broker, replay, or a durable
tool Intent Journal.

## Runtime event topology

`RuntimeEventPipelines` in `agent/event_pipelines.py` is the single registration
point for automatic subscribers:

```text
AgentRuntime / ToolManager publishers
  -> EventBus (synchronous, process-local)
     -> required
        -> SessionEventPipeline
        -> MemoryUsageEventPipeline
     -> best-effort
        -> MemoryPipeline / Reflection
        -> CapabilityHealthEventPipeline
        -> AuditEventSubscriber
        -> EventMetricsCollector
        -> ProgressEventPipeline -> ConsoleUI
```

Read operations remain direct method calls. Session load/resolve/acquire,
Memory search, Context building, and capability queries are not Event RPCs.
Explicit user Memory CRUD commands also remain normal capability/CLI calls.

## Delivery contract

The Event schema remains version 1:

```text
schema_version, id, name, timestamp,
project_id, session_id, run_id, payload
```

`subscribe(name, handler, required=False, name=...)` records stable ownership
metadata and returns an unsubscribe callback. Duplicate registration of the
same handler is ignored. `publish()` delivers to exact subscribers and then
wildcard observers. Handler exceptions are recorded as `EventDelivery` values
and never prevent later handlers from running.

`dispatch_required()` adds fail-closed semantics:

1. at least one exact subscriber marked `required` must exist; exact
   best-effort observers and wildcard Audit/Metrics handlers do not count as an
   owner;
2. every subscriber marked `required` must succeed;
3. failures from best-effort subscribers are visible in the dispatch result but
   do not roll back a successful required owner;
4. nested publication does not overwrite the outer delivery diagnostics.

This is synchronous application-level delivery, not a distributed guarantee.
If the process or machine stops between filesystem/database operations, v0.10.0
does not promise replay or exactly once across that crash boundary.

## Ordering and terminal-state safety

Runtime uses the following order:

```text
state.start
  -> required Session checkpoint
  -> task.started
  -> model/tool loop
  -> state.complete or state.fail
  -> required Session finalize
  -> task.finished or task.failed
  -> automatic Memory/Reflection + Audit/Metrics
```

An initial checkpoint failure stops before `task.started`. A per-tool
checkpoint failure is safe to finalize only when the named Session writer's
delivery succeeded and another required observer failed; Runtime uses immutable
delivery evidence rather than trusting mutable payload data. When the Session
writer itself failed, Runtime does not attempt a later finalize from an uncertain state. A
failed finalize is not recursively retried and no terminal task event is
published. Thus automatic terminal Memory, audit, and metrics never claim a
terminal state that the Session owner did not persist.

The Session subscriber receives a live `AgentState` and message list because it
is the in-process owner. That internal payload is deliberately non-serializable
as an external contract and must not be forwarded to logs or plugins.

## Memory idempotency

Only Memory entries actually included in a `ContextPackage` publish
`memory.usage.recorded`. The required subscriber validates 1 to 1000 positive,
deduplicated IDs and requires `run_id` plus a deterministic `usage_id`.

`MemoryStore.record_usage_once()` performs one SQLite transaction:

1. check `memory_usage_events` for the usage ID;
2. reject replay with different run/project/ID evidence;
3. increment `use_count` and set `last_used_at`;
4. insert the idempotency record.

Runtime updates `AgentState.loaded_memories` only after this transaction
succeeds. Re-delivery of the same event is a no-op. Resume creates a new
`session:turn:N` run ID, so a Memory may be reinforced once again when it is
actually included in the new turn.

Automatic Summary/Lesson/Bug/Decision/Reflection remains driven by terminal
events. Existing `pipeline_runs.run_id` ownership prevents duplicate terminal
publication from creating duplicate automatic memories.

## Capability health and permission boundary

`ToolManager.execute()` retains this invariant:

```text
ToolRequest -> PermissionManager -> optional approval -> handler -> ToolResult
                                                   -> tool.finished Event
```

Direct `CapabilityHealthManager.record()` calls were removed from ToolManager.
The `tool.finished` event contains only bounded metadata: tool/action/capability
labels, request ID, argument count, success, duration, result-field count, a
health-failure flag, and an optional redacted error summary. It never contains
argument values, paths, stdout/stderr, result bodies, or data keys.

The Health subscriber is best-effort. A health-store write failure cannot turn
an already completed ToolResult into a failure. Business validation failures do
not degrade capability health; only bounded infrastructure markers such as
timeout, missing dependency, unavailable command, or connection failure do.

## Audit, metrics, and Thinking privacy

Audit is an allow-list projection by event name and field. Unknown events have
an empty payload. The logger never calls `str()` on arbitrary payload objects
and excludes:

- Prompt, final response, reasoning, messages, and live AgentState;
- tool arguments, stdin, stdout/stderr, response bodies, and file contents;
- API keys, Authorization, cookies, passwords, tokens, and secrets.

Log directories/files use `0700/0600`, reject symbolic-link paths, and use
no-follow open behavior where available.

Metrics accepts only task/model/tool events and stores:

- bounded event counts;
- bounded aggregate tool duration;
- bounded failed-tool count;
- schema version and update time.

Malformed/non-object JSON, booleans masquerading as integers, negative numbers,
and oversized integers are rejected or clamped. Writes use a private temporary
file, `fsync`, and atomic replacement. Prompt, reasoning, tool metadata, output,
and credentials are never persisted.

Thinking and progress use `ui.progress.updated`. Its payload may contain a
transient reasoning chunk for the ConsoleUI, but Audit projects the payload to
empty and Metrics does not subscribe to that event. The UI subscriber is
best-effort; a broken terminal renderer does not interrupt model execution.

## Configuration

The migration adds defaults without replacing existing configuration:

```yaml
events:
  jsonl_log: true
  metrics_enabled: true
```

Audit output is under `~/.local/share/deep-agent/logs/`. Per-project aggregate
metrics are under `~/.local/share/deep-agent/metrics/`. Capability health and
Memory idempotency use existing private data ownership. None of these paths may
be committed to a project repository. Project identifiers are mapped to a
a stable safe storage component before they form Metrics, Health, Daemon, or
Parallel paths, so damaged/malicious project metadata cannot escape those data
directories. Existing normal UUID/sha256 identifiers keep their historical
path names.

## Subscriber extension checklist

Before adding a subscriber:

1. decide whether the effect is a command/query, required persistence, or a
   best-effort observer;
2. keep commands behind ToolManager and PermissionManager;
3. choose a stable event name and bounded payload projection;
4. define `project_id/session_id/run_id` correlation and an idempotency key;
5. make required ownership exact and test the missing-owner/failure paths;
6. keep Audit/Metrics metadata-only and test secret/content absence;
7. test duplicate delivery, ordering, Resume, malformed external values, and
   subscriber failure;
8. if interactive output changes, run a real PTY test;
9. do not claim durability, replay, or exactly-once behavior without a separate
   journal/broker design.

## Deferred reliability work

The next reliability layer should be a Durable Intent Journal for operations
whose side effects may survive a process crash. That design must record intent,
original hashes, execution result, and commit state before offering replay. It
is intentionally separate from this synchronous notification bus so events do
not become an unsafe tool-execution channel.
