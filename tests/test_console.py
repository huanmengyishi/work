from __future__ import annotations

import re
import sys
import threading

from agent.console import ConsoleUI, _cluster_width, _grapheme_clusters, _truncate_display_width


def test_colored_readline_prompt_marks_ansi_sequences_as_nonprinting() -> None:
    ui = object.__new__(ConsoleUI)
    ui.color = True
    ui._readline = object()

    rendered = ui._style("agent> ", "1;32", readline_prompt=True)

    assert rendered == "\001\033[1;32m\002agent> \001\033[0m\002"


def test_colored_non_readline_prompt_uses_plain_ansi_sequences() -> None:
    ui = object.__new__(ConsoleUI)
    ui.color = True
    ui._readline = object()

    assert ui._style("agent> ", "1;32") == "\033[1;32magent> \033[0m"
    assert "\001" not in ui._style("ordinary output", "36")
    assert "\002" not in ui._style("ordinary output", "36")


def test_progress_renderer_shows_mode_tool_turn_limits_and_reasoning(capsys) -> None:
    ui = object.__new__(ConsoleUI)
    ui.color = False
    ui._readline = None
    ui.show_thinking = True
    ui.show_reasoning_content = True
    ui._progress_started = 0.0
    ui._progress_label = ""
    ui._reasoning_stream_open = False
    ui._progress_line_visible = False
    ui._progress_lock = threading.Lock()
    ui._output_lock = threading.RLock()
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
    assert "工具轮次软目标 24" in output
    assert "硬上限 32" in output
    assert "思考第" not in output
    assert "DeepSeek Thinking" in output
    assert "inspect in chunks" in output


def test_progress_renderer_distinguishes_tool_turn_and_recovery_phases(capsys) -> None:
    ui = object.__new__(ConsoleUI)
    ui.color = False
    ui._readline = None
    ui.show_thinking = True
    ui.show_reasoning_content = False
    ui.hard_tool_turn_limit = 32
    ui._progress_label = ""
    ui._reasoning_stream_open = False
    ui._progress_line_visible = False
    ui._progress_lock = threading.Lock()
    ui._output_lock = threading.RLock()
    ui._print_progress_line = lambda: print(ui._progress_label)

    ui.update_progress(
        {
            "event": "model.requested",
            "mode": "deep",
            "round": 7,
            "max_rounds": 24,
            "hard_limit": 32,
            "current_step": "implement",
        }
    )
    ui.update_progress(
        {
            "event": "context.overflow_recovered",
            "mode": "deep",
            "stage": "semantic_compact",
            "estimated_tokens": 48_000,
        }
    )
    ui.update_progress(
        {
            "event": "model.requested",
            "mode": "deep",
            "phase": "final_synthesis",
            "round": 25,
            "max_rounds": 25,
        }
    )

    output = capsys.readouterr().out
    assert "工具轮次 7；软目标 24；硬上限 32；步骤 implement" in output
    assert "恢复阶段：语义压缩完成；工具轮次不增加" in output
    assert "收口阶段：最终总结；工具轮次不增加" in output
    assert "7/24" not in output


def test_progress_renderer_streams_reasoning_chunks(capsys) -> None:
    ui = object.__new__(ConsoleUI)
    ui.color = False
    ui._readline = None
    ui.show_thinking = True
    ui.show_reasoning_content = True
    ui._reasoning_stream_open = False
    ui._progress_line_visible = False
    ui._output_lock = threading.RLock()
    ui.update_progress({"event": "thinking.delta", "mode": "deep", "content": "first "})
    ui.update_progress({"event": "thinking.delta", "mode": "deep", "content": "chunk"})

    output = capsys.readouterr().out
    assert output.count("DeepSeek Thinking") == 1
    assert "first " in output
    assert "chunk" in output


def test_tty_screen_keeps_reasoning_when_spinner_ticks_between_chunks(capsys, monkeypatch) -> None:
    ui = object.__new__(ConsoleUI)
    ui.color = False
    ui._readline = object()
    ui.show_thinking = True
    ui.show_reasoning_content = True
    ui.progress_interval_seconds = 10
    ui._progress_started = 0.0
    ui._progress_label = "tool turn"
    ui._progress_lock = threading.Lock()
    ui._output_lock = threading.RLock()
    ui._reasoning_stream_open = False
    ui._progress_line_visible = False
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

    ui._print_progress_line()
    ui.update_progress({"event": "thinking.delta", "content": "inspect "})
    ui._print_progress_line()  # A timer tick must not overwrite the open line.
    ui.update_progress({"event": "thinking.delta", "content": "bounded chunks"})
    ui.update_progress({"event": "model.responded"})
    ui._print_progress_line()

    raw = capsys.readouterr().out
    screen = _terminal_screen_text(raw)
    assert "inspect bounded chunks" in screen
    assert screen.count("DeepSeek Thinking") == 1


def test_progress_line_is_grapheme_safe_and_bounded_on_narrow_terminal(capsys, monkeypatch) -> None:
    ui = object.__new__(ConsoleUI)
    ui.color = False
    ui._progress_started = 0.0
    ui._progress_label = "深度任务图模式；工具轮次 12；软目标 24；硬上限 32；步骤 验证👩‍💻e\u0301"
    ui._progress_lock = threading.Lock()
    ui._output_lock = threading.RLock()
    ui._reasoning_stream_open = False
    ui._progress_line_visible = False
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(
        "agent.console.shutil.get_terminal_size", lambda fallback: __import__("os").terminal_size((32, 24))
    )

    ui._print_progress_line()

    raw = capsys.readouterr().out
    visible = _terminal_screen_text(raw).splitlines()[-1]
    assert sum(_cluster_width(item) for item in visible) <= 31
    assert visible.endswith("…")
    assert not visible.endswith("\u200d")
    assert _truncate_display_width("中e\u0301👩‍💻文", 5) == "中e\u0301…"


def test_grapheme_width_handles_flags_modifiers_keycaps_and_emoji_presentation() -> None:
    value = "🇨🇳👍🏽1️⃣©️"

    clusters = list(_grapheme_clusters(value))

    assert clusters == ["🇨🇳", "👍🏽", "1️⃣", "©️"]
    assert [_cluster_width(item) for item in clusters] == [2, 2, 2, 2]
    assert _truncate_display_width("🇨🇳X", 2) == "…"
    assert _truncate_display_width("🇨🇳X", 3) == "🇨🇳X"
    assert _truncate_display_width("👍🏽X", 2) == "…"


def _terminal_screen_text(value: str) -> str:
    """Reduce ConsoleUI's ANSI subset to the text left on the terminal screen."""
    lines = [""]
    cursor = 0
    index = 0
    while index < len(value):
        if value.startswith("\r\x1b[2K", index):
            lines[-1] = ""
            cursor = 0
            index += len("\r\x1b[2K")
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
                index += len(match.group(0))
                continue
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
