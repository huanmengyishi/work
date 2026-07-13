# Deep Agent v0.9.1 Interface Architecture

## Purpose

Version 0.9.1 is a compatibility-stabilization release before the v1.0 core
interface freeze. It does not replace the Runtime, migrate every side effect to
events, add another model provider, or introduce multiple Agents. It makes the
existing boundaries explicit and executable through contract tests.

The frozen high-level chain is:

```text
CLI -> Runtime -> AgentState -> Prompt -> Capability -> Permission
```

The model-context chain is:

```text
ContextBuilder -> ContextPackage -> PromptBuilder -> DeepSeek
```

The version marker lives in `agent/contracts.py`. A deliberate incompatible
change to a tested public signature or serialized field set must update the
contract version and release notes instead of silently changing consumers.

## Boundary ownership

### CLI

`agent/cli.py` parses commands, creates project-scoped services, maps failures
to user-facing output, and calls `AgentRuntime.run()` or `resume()`. It does not
assemble model context, execute model tool calls directly, or make permission
decisions.

### Runtime

`AgentRuntime` owns orchestration order:

1. build the current project snapshot;
2. classify the request with `TaskRouter`;
3. choose one DeepSeek route with `ModelRouter`;
4. create or resume `AgentState`;
5. ask `ContextBuilder` for a bounded `ContextPackage`;
6. pass that package to `PromptBuilder`;
7. call DeepSeek;
8. send model tool calls to `ToolManager`;
9. checkpoint state and publish observational events.

Runtime may coordinate these components but must not read model context files
on Prompt's behalf and must not bypass ToolManager for a requested capability.

### AgentState

`AgentState` is the serializable source of truth for one Session. Version 0.9.1
publishes:

- `SCHEMA_VERSION`;
- `SERIALIZED_FIELDS`;
- `FROZEN_FIELDS`;
- `validate()`;
- `validate_frozen_fields()`.

Validation covers identity, timestamps, plan IDs/dependencies/cycles, derived
plan fields, route enums, DeepSeek-only provider, cost class, Context manifest
bounds, and collection types. Schema v1 checkpoints are still loaded and only
their derived plan fields are repaired. A future unknown schema fails closed.

Frozen identity is `session_id`, the project mapping, `working_directory`, and
`created_at`. Request text, routes, plan progress, results, and `updated_at` are
intentionally mutable between turns.

### AgentState extension rules

- Adding a serialized field is an interface change: add it to
  `AGENT_STATE_SERIALIZED_FIELDS`, choose a deterministic default for old
  Session files, and add round-trip plus Resume migration tests.
- Do not repurpose or rename an existing field in place. Add the replacement,
  read both during a compatibility window, then remove the legacy field only
  in a versioned breaking release.
- A field belongs in `FROZEN_FIELDS` only when changing it would change Session
  identity or ownership. Runtime progress must remain mutable.
- `validate()` must be deterministic and side-effect free. It may reject or
  normalize explicitly supported legacy-derived values, but it must not read
  files, call tools, repair current project data, or silently accept an unknown
  future schema.
- Keep derived fields consistent at the mutation point. For example,
  `current_step`, `completed_steps`, and
  `execution_context.current_plan_id` are updated together by PlanManager.
- Load/save boundaries call validation. Invalid persisted state should fail
  with a field-specific error before it reaches Prompt or tool execution.

When a new schema is necessary, increment `AGENT_STATE_SCHEMA_VERSION`, write
an explicit vN-to-vN+1 promotion path, keep migration add-only, and verify that
the original frozen identity survives the promotion. Never rewrite unrelated
Memory, project files, or user configuration while loading a Session.

### ContextBuilder and ContextPackage

ContextBuilder is the only model-context selection entry. It may read bounded
project files and combine task, execution, project, Session, Semantic, Memory,
capability, and recovery sections. It owns ordering, truncation, privacy, and
the total Package budget.

`ContextPackage` is an immutable transfer object. PromptBuilder must receive
one Package; it cannot accept `AgentState`, `ContextSnapshot`, raw Memory text,
or capability summaries as separate parameters.

### PromptBuilder

PromptBuilder is a pure renderer with exactly two public paths:

```python
build_initial(package: ContextPackage)
build_resume(package: ContextPackage)
```

It adds the fixed system policy and renders the already-selected Package. It
does not import ContextBuilder or AgentState and performs no file I/O. This
prevents a second context budget, hidden Resume compaction, or duplicated
Memory injection from reappearing in Prompt code.

### Capability and Permission

Every model call follows:

```text
model tool call
  -> ToolCapabilityRegistry creates ToolRequest
  -> ToolManager resolves capability
  -> PermissionManager.evaluate
  -> optional user confirmation
  -> registered handler
  -> ToolResult
```

The interface contract test uses a recording PermissionManager and handler to
prove permission evaluation occurs before execution. Event publication,
Runtime code, or Prompt code must never become an alternate execution path.

## Task and model routing

`TaskRouter` is the only classifier. It owns task type, scale, risk, mode,
score, reasons, failure evidence, and mutation evidence. `TaskPlanFactory`
consumes only `TaskRoute` and selects a bounded plan template; it has no Prompt
regular expressions or scoring rules.

`task_strategy.py` is a deprecated v0.8 compatibility facade. New Runtime code
does not instantiate it. Its legacy DTO may still be loaded during the
compatibility window, but all classification delegates to TaskRouter.

ModelRouter remains local, deterministic, and DeepSeek-only:

```text
simple + low risk                  -> fast     / low cost
normal work                        -> standard / balanced cost
large read-only work               -> standard / balanced cost
deep or high-risk work             -> deep     / high cost
architecture, refactor, failures   -> deep     / high cost
```

An explicit configured tier wins. Distinct model names are used only when the
user configures valid DeepSeek tier models; otherwise all tiers safely fall
back to `model.model`. `ModelRoute.cost_class` and reasons are persisted for
inspection. They are estimates for routing, not billing measurements.

## Minimal Event Bus contract

The 0.9.1 Event Bus is deliberately small and synchronous. `Event` has stable
serialized fields:

```text
schema_version, id, name, timestamp,
project_id, session_id, run_id, payload
```

Public operations are:

```python
cancel = bus.subscribe("task.finished", handler)
bus.publish("task.finished", payload, project_id=..., session_id=..., run_id=...)
bus.publish(existing_event)
bus.unsubscribe("task.finished", handler)
cancel()
```

Subscribers run in registration order. One subscriber exception is recorded
in `last_errors` and does not prevent later handlers. Nested publication keeps
the outer publication's final error list. JSONL logging serializes the same
event shape and continues to sanitize secrets.

This is not a durable broker. It does not promise replay, cross-process
delivery, ordering across processes, retries, at-least-once delivery, or
transactional coupling with tools. Existing MemoryPipeline subscriptions stay
in place, and Session persistence remains directly owned by Runtime. A future
Event migration must be incremental and keep ToolRequest/Permission boundaries
unchanged.

### Event publisher rules

- Use a stable, namespaced event name such as `task.finished`, `tool.started`,
  or `memory.summary.persisted`; do not encode IDs into the name.
- Put small correlation metadata in `project_id`, `session_id`, and `run_id`.
  `Event.effective_run_id` preserves compatibility with older payloads that
  stored `run_id` only inside `payload`.
- Keep payloads bounded and serializable. Do not publish API keys, Cookies,
  passwords, raw model messages, complete file contents, or unbounded tool
  output. JSONL sanitization is a final guard, not permission to publish
  secrets.
- Treat a published payload as shared observation data. Do not mutate it in a
  subscriber while later subscribers may still consume it.
- Publish only after the authoritative state transition is complete. Events
  are observations in v0.9.1, not the transaction or source of truth.

### Subscriber template and lifecycle

```python
class MetricsSubscriber:
    def __call__(self, event: Event) -> None:
        # Validate only the fields this subscriber owns. Bound all work and
        # catch expected I/O errors locally when a fallback is possible.
        duration_ms = int(event.payload.get("duration_ms") or 0)
        self.record(event.name, max(0, duration_ms))


subscriber = MetricsSubscriber()
cancel = bus.subscribe("tool.finished", subscriber)
try:
    run_runtime()
finally:
    cancel()
```

Handlers execute synchronously in registration order. Keep them fast and
bounded: a slow handler delays the publisher. An exception is captured in
`last_errors` and later handlers still run, but there is no retry. A component
that owns a shorter lifecycle than Runtime must retain and call the returned
cancellation callback; long-lived duplicate registrations are otherwise a
memory and duplicate-side-effect risk.

### Recommended future migration sequence

1. Select one observational side effect with a clear idempotency key, such as
   metrics or audit output. Do not start with Session commits or file writes.
2. Define its event name, required payload keys, size limit, and redaction rule
   in tests before changing the publisher.
3. Dual-run the direct path and subscriber in a comparison mode that performs
   only one real write; verify equivalent output and failure behavior.
4. Make the subscriber idempotent by `event.id` or `run_id`, then move the
   authoritative side effect behind it.
5. Add crash-window tests. If loss or replay is unacceptable, stop: the
   in-process bus is insufficient and a durable journal must be designed first.
6. Migrate one module at a time and preserve the existing ToolManager and
   PermissionManager execution boundary throughout.

Do not infer that an Event was durably processed merely because `publish()`
returned. A future durable Event layer needs an explicit journal, subscriber
checkpoint, replay policy, schema migration, and dead-letter strategy; those
are intentionally outside v0.9.1.

## Extension checklist

Before extending a frozen boundary:

1. decide whether the change is additive or incompatible;
2. update the typed object and its serialization path together;
3. add or update `tests/test_interface_contracts.py`;
4. add migration coverage for saved state or event records;
5. keep DeepSeek as the only provider unless project policy explicitly changes;
6. keep external input limits and log redaction;
7. run pytest, Ruff, format, compileall, and `git diff --check`;
8. document rollback and tag the release.

Do not place GitHub/Jira/Notion integrations, Worker Runtime, multiple Agents,
or a complete Event migration inside this interface patch. Such work needs a
separate design and release boundary.
