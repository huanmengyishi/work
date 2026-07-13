from __future__ import annotations

from agent import cli
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
