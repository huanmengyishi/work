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


def test_progress_renderer_shows_mode_round_and_reasoning(capsys) -> None:
    ui = object.__new__(ConsoleUI)
    ui.color = False
    ui._readline = None
    ui.show_thinking = True
    ui.show_reasoning_content = True
    ui._progress_started = 0.0
    ui._progress_label = ""
    ui._reasoning_stream_open = False
    import threading

    ui._progress_lock = threading.Lock()
    ui._print_progress_line = lambda: print(ui._progress_label)
    ui.update_progress(
        {
            "event": "strategy.selected",
            "mode": "deep",
            "strategy": {"max_tool_rounds": 24},
        }
    )
    ui.update_progress({"event": "thinking.content", "mode": "deep", "content": "inspect in chunks"})

    output = capsys.readouterr().out
    assert "深度任务图模式" in output
    assert "DeepSeek Thinking" in output
    assert "inspect in chunks" in output


def test_progress_renderer_streams_reasoning_chunks(capsys) -> None:
    ui = object.__new__(ConsoleUI)
    ui.color = False
    ui._readline = None
    ui.show_thinking = True
    ui.show_reasoning_content = True
    ui._reasoning_stream_open = False
    ui.update_progress({"event": "thinking.delta", "mode": "deep", "content": "first "})
    ui.update_progress({"event": "thinking.delta", "mode": "deep", "content": "chunk"})

    output = capsys.readouterr().out
    assert output.count("DeepSeek Thinking") == 1
    assert "first " in output
    assert "chunk" in output
