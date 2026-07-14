from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
import pty
import select
import signal
import struct
import subprocess
import sys
import termios
import time
from pathlib import Path

from agent import cli
from agent.config import ensure_default_config
from agent.console import _cluster_width, _command_completer, _grapheme_clusters


def test_direct_prompt_and_command_escape(monkeypatch) -> None:
    calls: list[str] = []

    class SentinelConfig:
        @staticmethod
        def get(name, default=None):
            return default

    sentinel_config = SentinelConfig()
    monkeypatch.setattr(cli, "load_config", lambda: sentinel_config)
    monkeypatch.setattr(cli, "run_once", lambda config, prompt, **kwargs: calls.append(prompt) or 0)

    assert cli.main(["fix", "this", "project"]) == 0
    assert cli.main(["--", "doctor", "this", "code"]) == 0
    assert calls == ["fix this project", "doctor this code"]


def test_resume_without_prompt_enters_selected_interactive_session(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class SentinelConfig:
        @staticmethod
        def get(name, default=None):
            return default

    monkeypatch.setattr(cli, "load_config", SentinelConfig)

    def fake_repl(config, **kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "repl", fake_repl)

    assert cli.main(["resume", "--session", "session-123"]) == 0
    assert captured["initial_session"] == "session-123"


def test_top_level_help_separates_tasks_from_management_commands(capsys) -> None:
    assert cli.main(["--help"]) == 0

    output = capsys.readouterr().out
    assert "usage: agent [-h]" in output
    assert "[task ...]" in output
    assert "[commands]" not in output
    assert "Management commands: doctor, projects, init" in output


def test_interactive_command_completion() -> None:
    assert _command_completer("/sta", 0) == "/status"
    assert _command_completer("/sta", 1) is None
    assert _command_completer("/super", 0) == "/super-yolo"


def test_default_config_initialization_is_concurrency_safe(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _index: ensure_default_config(), range(24)))

    assert results == [None] * 24
    assert (tmp_path / "config" / "deep-agent" / "config.yaml").is_file()


def test_real_pty_enter_help_and_exit(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    master, slave = pty.openpty()
    # Exercise the GNU Readline ANSI prompt on a genuinely narrow terminal,
    # where unmarked control bytes otherwise corrupt wrapping and Enter.
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 40, 0, 0))
    env = {
        **os.environ,
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "TERM": "xterm-256color",
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "agent"],
        cwd=project,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env=env,
        close_fds=True,
    )
    os.close(slave)
    output = bytearray()

    def read_until(needle: bytes, timeout: float = 8.0) -> None:
        deadline = time.monotonic() + timeout
        while needle not in output and time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], 0.2)
            if ready:
                output.extend(os.read(master, 4096))
        assert needle in output, output.decode("utf-8", errors="replace")

    try:
        read_until("首次使用：agent config".encode())
        read_until("快速开始".encode())
        read_until("/resume 继续未完成任务".encode())
        read_until("总结所有 Word 文档".encode())
        read_until(b"> ")
        os.write(master, b"\r")
        read_until("未输入请求".encode())
        os.write(master, b"/help\r")
        read_until("Deep Agent 快速上手".encode())
        os.write(master, b"/exit\r")
        assert process.wait(timeout=8) == 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        os.close(master)


def test_resume_without_saved_session_reports_clean_error(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    env = {
        **os.environ,
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
        "TERM": "dumb",
    }

    result = subprocess.run(
        [sys.executable, "-m", "agent", "resume"],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 1
    assert "no saved session is available" in output
    assert "Traceback" not in output


def test_real_pty_event_progress_and_thinking_are_visible(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    probe = tmp_path / "progress_probe.py"
    probe.write_text(
        """
from agent.cli import repl
from agent.config import AppConfig, DEFAULT_CONFIG, deep_merge
from agent.deepseek import ChatResponse
import agent.cli as cli
import time

class FakeClient:
    def chat_stream(self, **kwargs):
        kwargs["on_reasoning"]("inspect ")
        time.sleep(0.05)
        kwargs["on_reasoning"]("bounded chunks")
        if not any(
            message.get("role") == "tool" and "finish-plan" in str(message.get("tool_call_id") or "")
            for message in kwargs["messages"]
        ):
            return ChatResponse(
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "finish-plan",
                            "type": "function",
                            "function": {
                                "name": "agent_update_plan",
                                "arguments": __import__("json").dumps(
                                    {
                                        "steps": [
                                            {"id": "scope", "title": "Scope", "status": "completed"},
                                            {"id": "inspect-chunks", "title": "Inspect", "status": "completed"},
                                            {"id": "implement", "title": "Implement", "status": "completed"},
                                            {"id": "verify", "title": "Verify", "status": "completed"},
                                        ]
                                    }
                                ),
                            },
                        }
                    ],
                },
                raw={},
            )
        return ChatResponse(
            message={"role": "assistant", "content": "PTY complete", "reasoning_content": "inspect bounded chunks"},
            raw={},
        )

original = cli.build_runtime
def build(*args, **kwargs):
    runtime = original(*args, **kwargs)
    runtime.client = FakeClient()
    return runtime
cli.build_runtime = build

values = deep_merge(
    DEFAULT_CONFIG,
    {
        "memory": {"vector_enabled": False},
        "events": {"jsonl_log": False},
        "runtime": {"task_mode": "standard"},
    },
)
raise SystemExit(repl(AppConfig(values=values, config_dir=__import__('pathlib').Path.cwd() / '.config', data_dir=__import__('pathlib').Path.cwd() / '.data'), yolo=True))
""".lstrip(),
        encoding="utf-8",
    )
    master, slave = pty.openpty()
    env = {
        **os.environ,
        "TERM": "xterm-256color",
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
    }
    process = subprocess.Popen(
        [sys.executable, str(probe)],
        cwd=project,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env=env,
        close_fds=True,
    )
    os.close(slave)
    output = bytearray()

    def read_until(needle: bytes, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while needle not in output and time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], 0.2)
            if ready:
                output.extend(os.read(master, 4096))
        assert needle in output, output.decode("utf-8", errors="replace")

    try:
        read_until(b"> ")
        os.write(master, "分析这个问题并给出结论\r".encode())
        read_until(b"DeepSeek Thinking")
        read_until(b"bounded chunks")
        read_until(b"PTY complete")
        rendered = output.decode("utf-8", errors="replace")
        assert "工具轮次软目标" in rendered
        assert "硬上限" in rendered
        assert "思考第" not in rendered
        assert "inspect bounded chunks" in _terminal_screen_text(rendered)
        os.write(master, b"/exit\r")
        assert process.wait(timeout=8) == 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        os.close(master)


def test_real_narrow_pty_progress_preserves_complex_graphemes(tmp_path: Path) -> None:
    probe = tmp_path / "narrow_progress_probe.py"
    probe.write_text(
        """
from agent.console import ConsoleUI
import threading
import time

ui = object.__new__(ConsoleUI)
ui.color = False
ui._progress_started = time.monotonic()
ui._progress_label = "🇨🇳👍🏽1️⃣👩‍💻e\u0301TAIL"
ui._progress_lock = threading.Lock()
ui._output_lock = threading.RLock()
ui._reasoning_stream_open = False
ui._progress_line_visible = False
ui._print_progress_line()
print()
""".lstrip(),
        encoding="utf-8",
    )
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 25, 0, 0))
    env = {
        **os.environ,
        "TERM": "xterm-256color",
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
    }
    process = subprocess.Popen(
        [sys.executable, str(probe)],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env=env,
        close_fds=True,
    )
    os.close(slave)
    output = bytearray()

    try:
        deadline = time.monotonic() + 8
        while "…".encode() not in output and time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], 0.2)
            if ready:
                output.extend(os.read(master, 4096))
        assert "…".encode() in output, output.decode("utf-8", errors="replace")
        assert process.wait(timeout=8) == 0

        rendered = _terminal_screen_text(output.decode("utf-8", errors="replace"))
        visible = next(line for line in rendered.splitlines() if "Thinking" in line)
        assert visible.endswith("🇨🇳👍🏽…")
        assert "1️⃣" not in visible
        assert "👩‍💻" not in visible
        assert not visible.endswith("\u200d")
        assert sum(_cluster_width(cluster) for cluster in _grapheme_clusters(visible)) <= 24
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        os.close(master)


def test_real_pty_ctrl_c_returns_to_recoverable_session_prompt(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    probe = tmp_path / "interrupt_probe.py"
    probe.write_text(
        """
from agent.cli import repl
from agent.config import AppConfig, DEFAULT_CONFIG, deep_merge
import agent.cli as cli
import time

class BlockingClient:
    def chat(self, **_kwargs):
        time.sleep(60)
        raise AssertionError("the PTY test should interrupt this request")

original = cli.build_runtime
def build(*args, **kwargs):
    runtime = original(*args, **kwargs)
    runtime.client = BlockingClient()
    return runtime
cli.build_runtime = build

values = deep_merge(
    DEFAULT_CONFIG,
    {
        "memory": {"vector_enabled": False},
        "events": {"jsonl_log": False},
        "runtime": {"task_mode": "simple", "show_reasoning_content": False},
    },
)
raise SystemExit(repl(AppConfig(values=values, config_dir=__import__('pathlib').Path.cwd() / '.config', data_dir=__import__('pathlib').Path.cwd() / '.data'), yolo=True))
""".lstrip(),
        encoding="utf-8",
    )
    master, slave = pty.openpty()
    env = {
        **os.environ,
        "TERM": "xterm-256color",
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
    }
    process = subprocess.Popen(
        [sys.executable, str(probe)],
        cwd=project,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env=env,
        close_fds=True,
    )
    os.close(slave)
    output = bytearray()

    def read_until(needle: bytes, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while needle not in output and time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], 0.2)
            if ready:
                output.extend(os.read(master, 4096))
        assert needle in output, output.decode("utf-8", errors="replace")

    try:
        read_until(b"> ")
        os.write(master, "等待中断并保存会话\r".encode())
        read_until("正在处理请求".encode())
        session_dir = project / ".project-agent" / "sessions"
        deadline = time.monotonic() + 8
        session_files: list[Path] = []
        while time.monotonic() < deadline:
            session_files = list(session_dir.glob("*.json"))
            if session_files:
                break
            time.sleep(0.05)
        assert len(session_files) == 1
        session_id = session_files[0].stem

        # SIGINT is the process-level effect produced by Ctrl+C on a controlling
        # terminal. Keep stdin/stdout attached to the real PTY while delivering
        # the signal directly so the test is deterministic under CI job control.
        os.kill(process.pid, signal.SIGINT)
        read_until("请求已中断".encode())
        read_until(session_id[-8:].encode())
        os.write(master, b"/status\r")
        read_until(f"Session: {session_id}".encode())
        os.write(master, b"/exit\r")
        assert process.wait(timeout=8) == 0

        payload = __import__("json").loads(session_files[0].read_text(encoding="utf-8"))
        assert payload["state"]["session_id"] == session_id
        assert payload["state"]["status"] == "running"
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        os.close(master)


def _terminal_screen_text(value: str) -> str:
    """Apply the small ANSI subset used by ConsoleUI and return visible lines."""
    import re

    lines = [""]
    cursor = 0
    index = 0
    while index < len(value):
        if value.startswith("\r\x1b[2K", index):
            lines[-1] = ""
            cursor = 0
            index += 5
            continue
        char = value[index]
        if char == "\r":
            cursor = 0
        elif char == "\n":
            lines.append("")
            cursor = 0
        elif char == "\x1b":
            match = re.match(r"\x1b\[[0-9;?]*[A-Za-z]", value[index:])
            if match:
                index += len(match.group(0)) - 1
        elif char not in {"\x01", "\x02"}:
            line = lines[-1]
            if cursor >= len(line):
                line += " " * (cursor - len(line)) + char
            else:
                line = line[:cursor] + char + line[cursor + 1 :]
            lines[-1] = line
            cursor += 1
        index += 1
    return "\n".join(lines)
