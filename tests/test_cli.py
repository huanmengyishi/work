from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
