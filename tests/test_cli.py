from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import pty
import select
import subprocess
import sys
import time
from pathlib import Path

from agent import cli
from agent.config import ensure_default_config
from agent.console import _command_completer


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
        read_until(b"> ")
        os.write(master, b"\r")
        read_until("未输入请求".encode())
        os.write(master, b"/help\r")
        read_until(b"Interactive commands:")
        os.write(master, b"/exit\r")
        assert process.wait(timeout=8) == 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        os.close(master)


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

values = deep_merge(DEFAULT_CONFIG, {"memory": {"vector_enabled": False}, "events": {"jsonl_log": False}})
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
        os.write(master, "全面审计整个代码库并深度重构所有安全问题\r".encode())
        read_until(b"DeepSeek Thinking")
        read_until(b"bounded chunks")
        read_until(b"PTY complete")
        os.write(master, b"/exit\r")
        assert process.wait(timeout=8) == 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        os.close(master)
