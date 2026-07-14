from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any
from uuid import uuid4


DEFAULT_MAX_RESULT_SOURCE_BYTES = 8 * 1024 * 1024
MAX_RESULT_SOURCE_BYTES_HARD_LIMIT = 64 * 1024 * 1024
_CAPTURE_MARKER = b"\n...[source middle omitted]...\n"
_PIPE_CHUNK_BYTES = 64 * 1024


@dataclass(frozen=True)
class ToolRequest:
    tool: str
    action: str
    args: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid4()))
    model_name: str | None = None

    @property
    def capability(self) -> str:
        return f"{self.tool}.{self.action}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolResult:
    success: bool
    stdout: str
    stderr: str = ""
    data: dict[str, Any] | None = None
    duration_ms: int = 0
    request_id: str | None = None

    @property
    def ok(self) -> bool:
        return self.success

    @property
    def output(self) -> str:
        return self.stdout

    @property
    def error(self) -> str:
        return self.stderr

    def with_execution(self, *, request_id: str, duration_ms: int | None = None) -> "ToolResult":
        return replace(
            self,
            request_id=request_id,
            duration_ms=self.duration_ms if self.duration_ms else int(duration_ms or 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": self.duration_ms,
            "request_id": self.request_id,
            "data": self.data or {},
        }

    def as_text(self, limit: int = 12000) -> str:
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        bounded_limit = max(0, int(limit))
        if len(text) <= bounded_limit:
            return text
        if bounded_limit < 2:
            return "0"[:bounded_limit]

        data = _bounded_result_data(self.data or {})
        payload: dict[str, Any] = {
            "success": self.success,
            "stdout": "",
            "stderr": "",
            "duration_ms": self.duration_ms,
            "request_id": self.request_id,
            "data": data,
            "truncated": True,
            "original_chars": len(text),
            "sha256": hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest(),
        }

        def render(excerpt_chars: int) -> str:
            if self.success:
                stdout_chars = excerpt_chars * 3 // 4
                stderr_chars = excerpt_chars - stdout_chars
            else:
                stderr_chars = excerpt_chars * 3 // 4
                stdout_chars = excerpt_chars - stderr_chars
            payload["stdout"] = _head_tail(self.stdout, stdout_chars)
            payload["stderr"] = _head_tail(self.stderr, stderr_chars)
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

        base = render(0)
        if len(base) > bounded_limit:
            payload["data"] = {"keys": sorted(str(key) for key in (self.data or {}))[:30]}
            base = render(0)
        if len(base) > bounded_limit:
            payload.pop("request_id", None)
            payload.pop("duration_ms", None)
            base = render(0)
        if len(base) > bounded_limit:
            payload.pop("data", None)
            payload.pop("sha256", None)
            base = render(0)
        if len(base) > bounded_limit:
            minimal = json.dumps(
                {"success": self.success, "truncated": True},
                separators=(",", ":"),
            )
            return minimal if len(minimal) <= bounded_limit else "{}"[:bounded_limit]

        low = 0
        high = max(len(self.stdout) + len(self.stderr), bounded_limit)
        best = base
        while low <= high:
            middle = (low + high) // 2
            candidate = render(middle)
            if len(candidate) <= bounded_limit:
                best = candidate
                low = middle + 1
            else:
                high = middle - 1
        return best


def _head_tail(value: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(value) <= limit:
        return value
    marker = "\n...[middle omitted]...\n"
    if limit <= len(marker):
        return value[:limit]
    available = limit - len(marker)
    head = available // 2
    tail = available - head
    return value[:head] + marker + value[-tail:]


def _bounded_result_data(value: Any, *, depth: int = 0) -> Any:
    if depth >= 3:
        return "[depth-limited]"
    if isinstance(value, dict):
        return {str(key): _bounded_result_data(item, depth=depth + 1) for key, item in list(value.items())[:30]}
    if isinstance(value, list):
        return [_bounded_result_data(item, depth=depth + 1) for item in value[:20]]
    if isinstance(value, str):
        return _head_tail(value, 256)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:256]


class BoundedByteCapture:
    """Drain a byte stream while retaining bounded head/tail evidence.

    ``total_bytes`` and ``sha256`` describe every byte presented to ``feed``;
    ``value_bytes`` never exceeds ``max_bytes``.  This lets subprocess pipes be
    drained to EOF without either deadlocking the child or retaining unbounded
    output in memory.
    """

    def __init__(self, max_bytes: int = DEFAULT_MAX_RESULT_SOURCE_BYTES) -> None:
        self.max_bytes = bounded_result_source_bytes(max_bytes)
        self.total_bytes = 0
        self._digest = hashlib.sha256()
        self._complete = bytearray()
        self._head = bytearray()
        self._tail = bytearray()
        self._truncated = False

    @property
    def truncated(self) -> bool:
        return self._truncated

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()

    @property
    def captured_bytes(self) -> int:
        return len(self.value_bytes())

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        value = bytes(chunk)
        self.total_bytes += len(value)
        self._digest.update(value)
        if not self._truncated and len(self._complete) + len(value) <= self.max_bytes:
            self._complete.extend(value)
            return
        if not self._truncated:
            self._begin_truncation(value)
            return
        self._append_tail(value)

    def value_bytes(self) -> bytes:
        if not self._truncated:
            return bytes(self._complete)
        return bytes(self._head) + _CAPTURE_MARKER + bytes(self._tail)

    def text(self) -> str:
        return self.value_bytes().decode("utf-8", errors="replace")

    def metadata(self) -> dict[str, Any]:
        return {
            "source_truncated": self.truncated,
            "source_original_bytes": self.total_bytes,
            "source_original_bytes_known": True,
            "source_captured_bytes": self.captured_bytes,
            "source_sha256": self.sha256,
        }

    def _begin_truncation(self, chunk: bytes) -> None:
        self._truncated = True
        retained_bytes = max(0, self.max_bytes - len(_CAPTURE_MARKER))
        head_limit = retained_bytes // 2
        tail_limit = retained_bytes - head_limit
        existing = bytes(self._complete)
        self._complete.clear()
        if len(existing) >= head_limit:
            self._head.extend(existing[:head_limit])
        else:
            self._head.extend(existing)
            self._head.extend(chunk[: head_limit - len(existing)])
        if tail_limit <= 0:
            return
        if len(chunk) >= tail_limit:
            self._tail.extend(chunk[-tail_limit:])
            return
        missing = tail_limit - len(chunk)
        self._tail.extend(existing[-missing:])
        self._tail.extend(chunk)

    def _append_tail(self, chunk: bytes) -> None:
        tail_limit = max(0, self.max_bytes - len(_CAPTURE_MARKER) - len(self._head))
        if tail_limit <= 0:
            return
        if len(chunk) >= tail_limit:
            self._tail[:] = chunk[-tail_limit:]
            return
        excess = max(0, len(self._tail) + len(chunk) - tail_limit)
        if excess:
            del self._tail[:excess]
        self._tail.extend(chunk)


def bounded_result_source_bytes(value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = DEFAULT_MAX_RESULT_SOURCE_BYTES
    return max(1024, min(parsed, MAX_RESULT_SOURCE_BYTES_HARD_LIMIT))


def bound_text_source(
    text: str,
    max_bytes: int = DEFAULT_MAX_RESULT_SOURCE_BYTES,
) -> tuple[str, dict[str, Any]]:
    capture = BoundedByteCapture(max_bytes)
    encoded = str(text).encode("utf-8", errors="replace")
    for start in range(0, len(encoded), _PIPE_CHUNK_BYTES):
        capture.feed(encoded[start : start + _PIPE_CHUNK_BYTES])
    return capture.text(), capture.metadata()


def run_command(
    args: list[str],
    *,
    cwd: Path,
    timeout: int,
    input_text: str | None = None,
    shell: bool = False,
    env: dict[str, str] | None = None,
    max_output_bytes: int = DEFAULT_MAX_RESULT_SOURCE_BYTES,
    max_input_bytes: int = DEFAULT_MAX_RESULT_SOURCE_BYTES,
) -> ToolResult:
    started = time.monotonic()
    output_limit = bounded_result_source_bytes(max_output_bytes)
    input_limit = bounded_result_source_bytes(max_input_bytes)
    encoded_input = input_text.encode("utf-8") if input_text is not None else None
    if encoded_input is not None and len(encoded_input) > input_limit:
        return ToolResult(
            False,
            "",
            f"command input exceeds {input_limit} bytes",
            data={
                "source_truncated": True,
                "source_original_bytes": len(encoded_input),
                "source_original_bytes_known": True,
                "source_captured_bytes": 0,
            },
            duration_ms=elapsed_ms(started),
        )
    try:
        process = subprocess.Popen(
            args if not shell else " ".join(args),
            cwd=str(cwd),
            stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=shell,
            env=env,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        return ToolResult(False, "", f"command not found: {exc}", duration_ms=elapsed_ms(started))

    stdout_capture = BoundedByteCapture(output_limit)
    stderr_capture = BoundedByteCapture(output_limit)
    assert process.stdout is not None
    assert process.stderr is not None
    drain_threads = [
        threading.Thread(
            target=_drain_pipe,
            args=(process.stdout, stdout_capture),
            daemon=True,
            name="deep-agent-stdout",
        ),
        threading.Thread(
            target=_drain_pipe,
            args=(process.stderr, stderr_capture),
            daemon=True,
            name="deep-agent-stderr",
        ),
    ]
    for thread in drain_threads:
        thread.start()
    writer: threading.Thread | None = None
    if encoded_input is not None:
        assert process.stdin is not None
        writer = threading.Thread(
            target=_write_stdin,
            args=(process.stdin, encoded_input),
            daemon=True,
            name="deep-agent-stdin",
        )
        writer.start()

    timed_out = False
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_process_group(process)
    except BaseException:
        # Do not join pipe-drain threads while the child is still alive.  In
        # particular, Ctrl+C must terminate the complete process group, drain
        # the now-closed pipes, and then propagate the original interruption
        # so Runtime can synthesize paired interrupted ToolResults.
        _terminate_process_group(process)
        raise
    finally:
        if writer is not None:
            writer.join(timeout=1)
        for thread in drain_threads:
            thread.join()

    output, error, result_allocation_truncated = _render_command_streams(
        stdout_capture,
        stderr_capture,
        output_limit,
        success=not timed_out and process.returncode == 0,
    )
    output = output.strip()
    error = error.strip()
    source_data = _command_source_metadata(stdout_capture, stderr_capture)
    source_data["source_truncated"] = bool(source_data["source_truncated"] or result_allocation_truncated)
    source_data["source_captured_bytes"] = len(output.encode("utf-8")) + len(error.encode("utf-8"))
    source_data.update({"returncode": process.returncode, "args": args})
    if timed_out:
        timeout_error = f"timeout after {timeout}s"
        if error:
            timeout_error = f"{error}\n{timeout_error}"
        return ToolResult(
            False,
            output,
            timeout_error,
            data=source_data,
            duration_ms=elapsed_ms(started),
        )
    return ToolResult(
        process.returncode == 0,
        output,
        error,
        source_data,
        elapsed_ms(started),
    )


def _drain_pipe(stream: Any, capture: BoundedByteCapture) -> None:
    try:
        while True:
            chunk = stream.read(_PIPE_CHUNK_BYTES)
            if not chunk:
                break
            capture.feed(chunk)
    finally:
        stream.close()


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=2)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _write_stdin(stream: Any, content: bytes) -> None:
    try:
        stream.write(content)
        stream.flush()
    except (BrokenPipeError, OSError):
        pass
    finally:
        stream.close()


def _command_source_metadata(
    stdout_capture: BoundedByteCapture,
    stderr_capture: BoundedByteCapture,
) -> dict[str, Any]:
    original_bytes = stdout_capture.total_bytes + stderr_capture.total_bytes
    captured_bytes = stdout_capture.captured_bytes + stderr_capture.captured_bytes
    return {
        "source_truncated": stdout_capture.truncated or stderr_capture.truncated,
        "source_original_bytes": original_bytes,
        "source_original_bytes_known": True,
        "source_captured_bytes": captured_bytes,
        "source_stream_bytes": {
            "stdout": stdout_capture.total_bytes,
            "stderr": stderr_capture.total_bytes,
        },
        "source_stream_sha256": {
            "stdout": stdout_capture.sha256,
            "stderr": stderr_capture.sha256,
        },
    }


def _render_command_streams(
    stdout_capture: BoundedByteCapture,
    stderr_capture: BoundedByteCapture,
    max_bytes: int,
    *,
    success: bool,
) -> tuple[str, str, bool]:
    stdout = stdout_capture.value_bytes()
    stderr = stderr_capture.value_bytes()
    if len(stdout) + len(stderr) <= max_bytes:
        return (
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
            stdout_capture.truncated or stderr_capture.truncated,
        )

    primary_name = "stdout" if success else "stderr"
    primary = stdout if success else stderr
    secondary = stderr if success else stdout
    primary_budget = min(len(primary), max_bytes * 3 // 4)
    secondary_budget = min(len(secondary), max_bytes - primary_budget)
    remaining = max_bytes - primary_budget - secondary_budget
    if remaining:
        primary_extra = min(remaining, len(primary) - primary_budget)
        primary_budget += primary_extra
        remaining -= primary_extra
    if remaining:
        secondary_budget += min(remaining, len(secondary) - secondary_budget)
    primary_value = _head_tail_bytes(primary, primary_budget)
    secondary_value = _head_tail_bytes(secondary, secondary_budget)
    if primary_name == "stdout":
        stdout_value, stderr_value = primary_value, secondary_value
    else:
        stdout_value, stderr_value = secondary_value, primary_value
    return (
        stdout_value.decode("utf-8", errors="replace"),
        stderr_value.decode("utf-8", errors="replace"),
        True,
    )


def _head_tail_bytes(value: bytes, limit: int) -> bytes:
    if limit <= 0:
        return b""
    if len(value) <= limit:
        return value
    if limit <= len(_CAPTURE_MARKER):
        return value[:limit]
    available = limit - len(_CAPTURE_MARKER)
    head = available // 2
    tail = available - head
    return value[:head] + _CAPTURE_MARKER + value[-tail:]


def truncate_text(text: str, limit: int = 20000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def elapsed_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1000))
