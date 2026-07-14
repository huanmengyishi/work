from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import CancelledError
from pathlib import Path
from typing import Any

import pytest

from agent.deepseek import ChatResponse
from agent.events import EventBus
from agent.memory import MemoryStore
from agent.project import ProjectManager
from agent.runtime import AgentRuntime
from agent.state import AgentState
from agent.tool_orchestration import PreparedToolCall, ToolBatchInterrupted, execute_model_tool_calls
from agent.tools import ToolManager, ToolRequest, ToolResult
from agent.tools.permission import PermissionDecision
from agent.tools.registry import ToolCapability


class FakeRegistry:
    def __init__(self, capabilities: list[ToolCapability]) -> None:
        self.capabilities = {item.model_name: item for item in capabilities}

    def resolve(self, name: str):
        capability = self.capabilities.get(name)
        return capability, (lambda: None) if capability is not None else None


class FakeExecutor:
    def __init__(
        self,
        capabilities: list[ToolCapability],
        handler: Callable[[str, str | dict[str, Any] | None], ToolResult],
    ) -> None:
        self.registry = FakeRegistry(capabilities)
        self.handler = handler

    def execute_model_call(
        self,
        name: str,
        arguments: str | dict[str, Any] | None,
        *,
        request_id: str | None = None,
        runtime_denied_reason: str | None = None,
    ) -> tuple[ToolRequest, ToolResult]:
        capability, _handler = self.registry.resolve(name)
        request = (
            ToolRequest(
                capability.tool,
                capability.action,
                arguments if isinstance(arguments, dict) else {},
                request_id=request_id or name,
                model_name=name,
            )
            if capability is not None
            else ToolRequest(
                "unknown",
                name,
                arguments if isinstance(arguments, dict) else {},
                request_id=request_id or name,
                model_name=name,
            )
        )
        if runtime_denied_reason:
            return request, ToolResult(False, "", runtime_denied_reason, request_id=request.request_id)
        result = self.handler(name, arguments)
        return request, result.with_execution(request_id=request.request_id)


def capability(
    name: str,
    *,
    permissions: tuple[str, ...] = ("read",),
    requires_confirmation: bool = False,
    concurrency_safe: bool = True,
    enabled: bool = True,
    available: bool = True,
) -> ToolCapability:
    return ToolCapability(
        "probe",
        name,
        name,
        f"Probe {name}",
        {},
        permissions=permissions,
        requires_confirmation=requires_confirmation,
        concurrency_safe=concurrency_safe,
        enabled=enabled,
        available=available,
    )


def test_consecutive_reads_run_concurrently_with_a_bounded_worker_count() -> None:
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def handler(name: str, _arguments) -> ToolResult:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.04)
        with lock:
            active -= 1
        return ToolResult(True, name)

    names = [f"read_{index}" for index in range(6)]
    executor = FakeExecutor([capability(name) for name in names], handler)
    executions = execute_model_tool_calls(
        executor,
        [PreparedToolCall(name, {}, request_id=f"call-{index}") for index, name in enumerate(names)],
        max_concurrency=2,
    )

    assert maximum_active == 2
    assert [result.stdout for _request, result in executions] == names


def test_unmarked_write_execute_network_confirmation_and_denied_calls_are_serial_barriers() -> None:
    lock = threading.Lock()
    active = 0
    maximum_active = 0
    events: list[str] = []

    def handler(name: str, _arguments) -> ToolResult:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
            events.append(f"start:{name}")
        time.sleep(0.015)
        with lock:
            events.append(f"finish:{name}")
            active -= 1
        return ToolResult(True, name)

    capabilities = [
        capability("unmarked_read", concurrency_safe=False),
        capability("write", permissions=("write",), concurrency_safe=True),
        capability("state", permissions=("state",), concurrency_safe=True),
        capability("execute", permissions=("read", "execute"), concurrency_safe=True),
        capability("network", permissions=("read", "network"), concurrency_safe=True),
        capability("confirmation", requires_confirmation=True),
        capability("disabled", enabled=False),
        capability("unavailable", available=False),
        capability("denied_read"),
    ]
    executor = FakeExecutor(capabilities, handler)
    executions = execute_model_tool_calls(
        executor,
        [
            PreparedToolCall("unmarked_read", {}),
            PreparedToolCall("write", {}),
            PreparedToolCall("state", {}),
            PreparedToolCall("execute", {}),
            PreparedToolCall("network", {}),
            PreparedToolCall("confirmation", {}),
            PreparedToolCall("disabled", {}),
            PreparedToolCall("unavailable", {}),
            PreparedToolCall("unknown", {}),
            PreparedToolCall("denied_read", {}, runtime_denied_reason="runtime denied"),
        ],
        max_concurrency=8,
    )

    assert maximum_active == 1
    assert events == [
        "start:unmarked_read",
        "finish:unmarked_read",
        "start:write",
        "finish:write",
        "start:state",
        "finish:state",
        "start:execute",
        "finish:execute",
        "start:network",
        "finish:network",
        "start:confirmation",
        "finish:confirmation",
        "start:disabled",
        "finish:disabled",
        "start:unavailable",
        "finish:unavailable",
        "start:unknown",
        "finish:unknown",
    ]
    assert executions[-1][1].success is False
    assert executions[-1][1].stderr == "runtime denied"


def test_capability_concurrency_opt_in_defaults_false_and_is_serialized() -> None:
    unmarked = ToolCapability("probe", "read", "probe_read", "Probe read", {})

    assert unmarked.concurrency_safe is False
    assert unmarked.to_dict()["concurrency_safe"] is False


def test_only_audited_builtin_reads_opt_in_to_concurrency(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    tools = ToolManager(config, project, memory, yolo=True)

    opted_in = {item.name for item in tools.capabilities() if item.concurrency_safe}

    assert opted_in == {
        "browser.list_sessions",
        "project.read_context",
        "template.find_files",
        "template.list_dir",
        "template.read_file",
        "template.search_code",
        "tool_result.read",
    }
    assert (
        not {
            "file.diff",
            "memory.search",
            "document.parse",
            "ocr.parse",
            "git.status",
            "http.request",
        }
        & opted_in
    )
    tools.close()


def test_results_keep_original_order_across_read_batches_and_serial_barriers() -> None:
    delays = {"slow_read": 0.06, "fast_read": 0.005, "write": 0.01, "last_read": 0.005}
    lock = threading.Lock()
    finished: list[str] = []

    def handler(name: str, _arguments) -> ToolResult:
        time.sleep(delays[name])
        with lock:
            finished.append(name)
        return ToolResult(True, f"result:{name}")

    executor = FakeExecutor(
        [
            capability("slow_read"),
            capability("fast_read"),
            capability("write", permissions=("write",)),
            capability("last_read"),
        ],
        handler,
    )
    executions = execute_model_tool_calls(
        executor,
        [
            PreparedToolCall("slow_read", {}, request_id="call-1"),
            PreparedToolCall("fast_read", {}, request_id="call-2"),
            PreparedToolCall("write", {}, request_id="call-3"),
            PreparedToolCall("last_read", {}, request_id="call-4"),
        ],
        max_concurrency=4,
    )

    assert finished == ["fast_read", "slow_read", "write", "last_read"]
    assert [request.request_id for request, _result in executions] == ["call-1", "call-2", "call-3", "call-4"]
    assert [result.stdout for _request, result in executions] == [
        "result:slow_read",
        "result:fast_read",
        "result:write",
        "result:last_read",
    ]


def test_executor_exception_becomes_one_failed_pair_and_later_calls_continue() -> None:
    invoked: list[str] = []

    def handler(name: str, _arguments) -> ToolResult:
        invoked.append(name)
        if name == "broken":
            raise RuntimeError("handler exploded")
        return ToolResult(True, name)

    names = ["first", "broken", "last"]
    executor = FakeExecutor(
        [capability(name, concurrency_safe=False) for name in names],
        handler,
    )

    executions = execute_model_tool_calls(
        executor,
        [PreparedToolCall(name, {}, request_id=f"call-{name}") for name in names],
    )

    assert invoked == names
    assert [request.request_id for request, _result in executions] == [f"call-{name}" for name in names]
    assert [result.success for _request, result in executions] == [True, False, True]
    assert executions[1][1].data == {"synthetic": True, "interrupted": False}
    assert "RuntimeError: handler exploded" in executions[1][1].stderr


def test_serial_keyboard_interrupt_returns_complete_failed_pairs_without_starting_later_calls() -> None:
    invoked: list[str] = []

    def handler(name: str, _arguments) -> ToolResult:
        invoked.append(name)
        if name == "interrupt":
            raise KeyboardInterrupt("stop now")
        return ToolResult(True, name)

    names = ["first", "interrupt", "never_started"]
    executor = FakeExecutor(
        [capability(name, concurrency_safe=False) for name in names],
        handler,
    )

    with pytest.raises(ToolBatchInterrupted) as captured:
        execute_model_tool_calls(
            executor,
            [PreparedToolCall(name, {}, request_id=f"call-{name}") for name in names],
        )

    assert isinstance(captured.value.cause, KeyboardInterrupt)
    assert invoked == ["first", "interrupt"]
    executions = captured.value.executions
    assert [request.request_id for request, _result in executions] == [f"call-{name}" for name in names]
    assert [result.success for _request, result in executions] == [True, False, False]
    assert executions[1][1].data == {"synthetic": True, "interrupted": True}
    assert executions[2][1].data == {"synthetic": True, "interrupted": True}
    assert all("KeyboardInterrupt" in result.stderr for _request, result in executions[1:])


def test_cancelled_call_fails_remaining_pairs_and_propagates_cancellation() -> None:
    invoked: list[str] = []

    def handler(name: str, _arguments) -> ToolResult:
        invoked.append(name)
        raise CancelledError("cancelled")

    names = ["cancelled", "queued"]
    executor = FakeExecutor(
        [capability(name, concurrency_safe=False) for name in names],
        handler,
    )

    with pytest.raises(ToolBatchInterrupted) as captured:
        execute_model_tool_calls(
            executor,
            [PreparedToolCall(name, {}, request_id=f"call-{name}") for name in names],
        )

    assert isinstance(captured.value.cause, CancelledError)
    assert invoked == ["cancelled"]
    assert [result.success for _request, result in captured.value.executions] == [False, False]
    assert all(result.data == {"synthetic": True, "interrupted": True} for _, result in captured.value.executions)


def test_parallel_keyboard_interrupt_preserves_completed_result_and_fails_in_progress_call() -> None:
    first_done = threading.Event()
    release_in_progress = threading.Event()
    invoked: list[str] = []
    invoked_lock = threading.Lock()

    def handler(name: str, _arguments) -> ToolResult:
        with invoked_lock:
            invoked.append(name)
        if name == "first":
            first_done.set()
            return ToolResult(True, "completed")
        if name == "interrupt":
            assert first_done.wait(timeout=1)
            raise KeyboardInterrupt("parallel stop")
        assert release_in_progress.wait(timeout=2)
        return ToolResult(True, "too late")

    names = ["first", "interrupt", "in_progress"]
    executor = FakeExecutor([capability(name) for name in names], handler)
    try:
        with pytest.raises(ToolBatchInterrupted) as captured:
            execute_model_tool_calls(
                executor,
                [PreparedToolCall(name, {}, request_id=f"call-{name}") for name in names],
                max_concurrency=2,
            )
    finally:
        release_in_progress.set()

    assert isinstance(captured.value.cause, KeyboardInterrupt)
    executions = captured.value.executions
    assert [request.request_id for request, _result in executions] == [f"call-{name}" for name in names]
    assert [result.success for _request, result in executions] == [True, False, False]
    assert executions[0][1].stdout == "completed"
    assert executions[1][1].data == {"synthetic": True, "interrupted": True}
    assert executions[2][1].data == {"synthetic": True, "interrupted": True}


def test_interrupted_parallel_read_keeps_captured_session_after_manager_rebind(
    tmp_path: Path,
    make_config,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config(
        {
            "tools": {
                "tool_result": {
                    "persist_threshold_bytes": 512,
                    "max_attachment_bytes": 16_000,
                    "preview_chars": 1_024,
                }
            }
        }
    )
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    events = EventBus()
    tools = ToolManager(config, project, memory, events=events, yolo=True)

    def state(session_id: str) -> AgentState:
        return AgentState.create(
            session_id=session_id,
            project=project,
            user_request="exercise interrupted read ownership",
            loaded_memories=[],
            loaded_tools=[],
            git_branch=None,
            context_index_path=str(project.agent_dir / "index.json"),
        )

    old_state = state("old-session")
    new_state = state("new-session")
    tools.bind_state(old_state)
    late_dispatched = threading.Event()
    release_late = threading.Event()
    late_finished = threading.Event()
    observed_handler_sessions: list[str] = []
    seen_events = []

    def record_event(event) -> None:
        request = event.payload.get("request") if isinstance(event.payload, dict) else {}
        if isinstance(request, dict) and request.get("request_id") == "call-late":
            seen_events.append(event)
            result = event.payload.get("result") if isinstance(event.payload, dict) else {}
            if event.name == "tool.finished" and isinstance(result, dict) and result.get("success") is True:
                late_finished.set()

    events.subscribe("*", record_event, name="ownership-recorder")

    original_contextual_execute = tools.execute_model_call_in_context

    def delay_before_execute(context, name, arguments, **kwargs):
        if name == "late_read" and not kwargs.get("runtime_denied_reason"):
            # The Future is already running, but ToolManager.execute has not
            # started. This deterministically covers the narrow scheduler
            # window in which a post-interrupt rebind used to steal ownership.
            late_dispatched.set()
            assert release_late.wait(timeout=5)
        return original_contextual_execute(context, name, arguments, **kwargs)

    tools.execute_model_call_in_context = delay_before_execute  # type: ignore[method-assign]

    def interrupting_read() -> ToolResult:
        assert late_dispatched.wait(timeout=2)
        raise KeyboardInterrupt("stop parallel batch")

    def late_read() -> ToolResult:
        observed_handler_sessions.append(tools._require_state().session_id)
        return ToolResult(True, "OLD-HEAD\n" + ("x" * 4_000) + "\nOLD-TAIL")

    tools.registry.register(capability("interrupt_read"), interrupting_read)
    tools.registry.register(capability("late_read"), late_read)

    try:
        with pytest.raises(ToolBatchInterrupted) as captured:
            execute_model_tool_calls(
                tools,
                [
                    PreparedToolCall("interrupt_read", {}, request_id="call-interrupt"),
                    PreparedToolCall("late_read", {}, request_id="call-late"),
                ],
                max_concurrency=2,
            )
        assert isinstance(captured.value.cause, KeyboardInterrupt)

        # Rebind exactly as the interactive CLI can after Ctrl+C, while the
        # non-cancellable read Future is still unwinding in its worker thread.
        tools.bind_state(new_state)
        release_late.set()
        assert late_finished.wait(timeout=5)

        old_attachment = tools.result_store.path_for_test(
            session_id=old_state.session_id,
            request_id="call-late",
        )
        new_attachment = tools.result_store.path_for_test(
            session_id=new_state.session_id,
            request_id="call-late",
        )
        assert old_attachment.exists()
        assert "OLD-HEAD" in old_attachment.read_text(encoding="utf-8")
        assert not new_attachment.exists()
        assert observed_handler_sessions == [old_state.session_id]
        assert seen_events
        assert all(event.session_id == old_state.session_id for event in seen_events)
        assert all(event.run_id == old_state.run_id for event in seen_events)
        assert not any(event.session_id == new_state.session_id for event in seen_events)
    finally:
        release_late.set()
        tools.close()


def test_parallel_reads_still_pass_permission_before_each_handler(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    tools = ToolManager(config, project, memory, yolo=True)
    lock = threading.Lock()
    events: list[tuple[str, str]] = []

    def register(name: str) -> None:
        tools.registry.register(
            capability(name),
            lambda _name=name: record_handler(_name),
        )

    def record_handler(name: str) -> ToolResult:
        with lock:
            events.append(("handler", name))
        time.sleep(0.01)
        return ToolResult(True, name)

    class RecordingPermissionManager:
        def evaluate(self, request, _resolved_capability, *, super_yolo=False):
            with lock:
                events.append(("permission", str(request.model_name)))
            return PermissionDecision(True)

    register("read_one")
    register("read_two")
    tools.permission = RecordingPermissionManager()

    executions = execute_model_tool_calls(
        tools,
        [PreparedToolCall("read_one", {}), PreparedToolCall("read_two", {})],
        max_concurrency=2,
    )

    assert all(result.success for _request, result in executions)
    for name in ("read_one", "read_two"):
        assert events.index(("permission", name)) < events.index(("handler", name))
    tools.close()


def test_runtime_uses_parallel_reads_and_records_results_in_model_order(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config({"runtime": {"convergence": {"max_parallel_read_tools": 2}}})
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    tools = ToolManager(config, project, memory, yolo=True)
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def handler(name: str, delay: float) -> ToolResult:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(delay)
        with lock:
            active -= 1
        return ToolResult(True, f"result:{name}")

    tools.registry.register(capability("slow_read"), lambda: handler("slow_read", 0.05))
    tools.registry.register(capability("fast_read"), lambda: handler("fast_read", 0.005))

    class FakeClient:
        def __init__(self) -> None:
            self.responses = [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-slow",
                            "type": "function",
                            "function": {"name": "slow_read", "arguments": "{}"},
                        },
                        {
                            "id": "call-fast",
                            "type": "function",
                            "function": {"name": "fast_read", "arguments": "{}"},
                        },
                    ],
                },
                {"role": "assistant", "content": "probe answer"},
            ]

        def chat(self, **_kwargs) -> ChatResponse:
            assert self.responses
            return ChatResponse(message=self.responses.pop(0), raw={})

    runtime = AgentRuntime(config=config, project=project, memory=memory, tools=tools, client=FakeClient())

    assert runtime.run("what is the probe answer") == "probe answer"
    state = runtime.sessions.load(str(runtime.last_session_id)).state
    assert maximum_active == 2
    assert [item["request"]["request_id"] for item in state.tool_calls] == ["call-slow", "call-fast"]
    assert [item["result"]["stdout"] for item in state.tool_calls] == ["result:slow_read", "result:fast_read"]
    tools.close()


def test_runtime_checkpoints_complete_tool_pairs_before_reraising_keyboard_interrupt(
    tmp_path: Path,
    make_config,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    config = make_config()
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    tools = ToolManager(config, project, memory, yolo=True)
    invoked: list[str] = []

    def handler(name: str) -> ToolResult:
        invoked.append(name)
        if name == "interrupt":
            raise KeyboardInterrupt("user interrupted tool batch")
        return ToolResult(True, name)

    names = ["completed", "interrupt", "queued"]
    for name in names:
        tools.registry.register(
            capability(name, concurrency_safe=False),
            lambda _name=name: handler(_name),
        )

    class InterruptingClient:
        def chat(self, **_kwargs) -> ChatResponse:
            return ChatResponse(
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call-{name}",
                            "type": "function",
                            "function": {"name": name, "arguments": "{}"},
                        }
                        for name in names
                    ],
                },
                raw={},
                finish_reason="tool_calls",
            )

    runtime = AgentRuntime(
        config=config,
        project=project,
        memory=memory,
        tools=tools,
        client=InterruptingClient(),
    )
    try:
        with pytest.raises(KeyboardInterrupt, match="user interrupted tool batch"):
            runtime.run("exercise an interrupted tool batch")

        assert invoked == ["completed", "interrupt"]
        record = runtime.sessions.load(str(runtime.last_session_id))
        assert record.state.status == "failed"
        assert record.state.execution_context is not None
        assert record.state.execution_context.prompt_phase == "interrupted"
        assert [item["request"]["request_id"] for item in record.state.tool_calls] == [f"call-{name}" for name in names]
        assert [item["result"]["success"] for item in record.state.tool_calls] == [True, False, False]
        assert all(item["result"]["data"].get("synthetic") for item in record.state.tool_calls[1:])

        assistant_index = next(index for index, message in enumerate(record.messages) if message.get("tool_calls"))
        assistant_calls = record.messages[assistant_index]["tool_calls"]
        tool_messages = record.messages[assistant_index + 1 : assistant_index + 1 + len(assistant_calls)]
        assert [message.get("role") for message in tool_messages] == ["tool", "tool", "tool"]
        assert [message.get("tool_call_id") for message in tool_messages] == [call["id"] for call in assistant_calls]
    finally:
        tools.close()
