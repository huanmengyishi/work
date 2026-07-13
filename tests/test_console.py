from __future__ import annotations

from agent.console import ConsoleUI


def test_colored_readline_prompt_marks_ansi_sequences_as_nonprinting() -> None:
    ui = object.__new__(ConsoleUI)
    ui.color = True
    ui._readline = object()

    rendered = ui._style("agent> ", "1;32")

    assert rendered == "\001\033[1;32m\002agent> \001\033[0m\002"


def test_colored_non_readline_prompt_uses_plain_ansi_sequences() -> None:
    ui = object.__new__(ConsoleUI)
    ui.color = True
    ui._readline = None

    assert ui._style("agent> ", "1;32") == "\033[1;32magent> \033[0m"
