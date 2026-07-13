from __future__ import annotations

import atexit
import os
import sys
import threading
import time
from pathlib import Path

from . import __version__
from .project import Project
from .unicode_text import normalize_unicode_text


COMMANDS = (
    "/new",
    "/resume",
    "/sessions",
    "/status",
    "/undo",
    "/yolo",
    "/super-yolo",
    "/help",
    "/clear",
    "/exit",
)


class ConsoleUI:
    def __init__(
        self,
        project: Project,
        data_dir: Path,
        *,
        yolo: bool = False,
        super_yolo: bool = False,
        show_thinking: bool = True,
        show_reasoning_content: bool = True,
    ) -> None:
        self.project = project
        self.yolo = yolo
        self.super_yolo = super_yolo
        self.show_thinking = show_thinking
        self.show_reasoning_content = show_reasoning_content
        self.history_path = data_dir / "cache" / "repl_history"
        self.color = bool(sys.stdout.isatty() and os.environ.get("TERM") != "dumb" and "NO_COLOR" not in os.environ)
        self._readline = self._configure_readline()
        self._progress_stop = threading.Event()
        self._progress_thread: threading.Thread | None = None
        self._progress_started = 0.0
        self._progress_label = ""
        self._progress_lock = threading.Lock()
        self._reasoning_stream_open = False

    def banner(self) -> None:
        title = self._style(f"Deep Agent {__version__}", "1;36")
        print(f"\n{title}  |  project: {self.project.name}")
        print(f"workspace: {self.project.root}")
        if self.super_yolo:
            print(self._style("SUPER YOLO enabled: confirmations and permission policies are bypassed.", "1;31"))
        elif self.yolo:
            print(
                self._style("YOLO mode enabled: confirmations are skipped; hard safety policies remain active.", "1;31")
            )
        print("Type /help for commands. Use Up/Down for history and Tab for command completion.\n")

    def prompt(self, session_id: str | None) -> str:
        session = session_id[-8:] if session_id else "new"
        mode = ":SUPER-YOLO" if self.super_yolo else ":YOLO" if self.yolo else ""
        label = f"{self.project.name}:{session}{mode} > "
        return self._style(label, "1;31" if self.yolo else "1;32")

    def set_yolo(self, enabled: bool) -> None:
        self.yolo = enabled
        if enabled:
            self.info("YOLO mode enabled. Confirmation prompts are disabled; hard safety policies remain active.")
        else:
            self.info("YOLO mode disabled. Confirmation prompts are active.")

    def set_super_yolo(self, enabled: bool) -> None:
        self.super_yolo = enabled
        if enabled:
            self.error("SUPER YOLO enabled. sudo, external paths, privileged Docker, and destructive commands may run.")
        else:
            self.info("SUPER YOLO disabled. Permission Manager hard policies are active.")

    def read(self, session_id: str | None) -> str:
        # Flush before input so WSL terminals always render the prompt before waiting.
        sys.stdout.flush()
        return normalize_unicode_text(input(self.prompt(session_id))).strip()

    def answer(self, value: str) -> None:
        print(f"\n{self._style('assistant', '1;36')}\n{normalize_unicode_text(value)}\n")

    def info(self, value: str) -> None:
        print(self._style(value, "36"))

    def working(self) -> None:
        print(self._style("正在处理请求...（按 Ctrl+C 可返回交互界面）", "36"), flush=True)
        self.start_progress("分析任务并选择执行方式")

    def start_progress(self, label: str) -> None:
        if not self.show_thinking:
            return
        self.stop_progress(clear=False)
        with self._progress_lock:
            self._progress_label = label
            self._progress_started = time.monotonic()
        self._progress_stop.clear()
        self._progress_thread = threading.Thread(target=self._progress_loop, name="deep-agent-progress", daemon=True)
        self._progress_thread.start()

    def update_progress(self, value: dict) -> None:
        event = str(value.get("event") or "")
        mode = str(value.get("mode") or "standard")
        if event == "strategy.selected":
            strategy = value.get("strategy") if isinstance(value.get("strategy"), dict) else {}
            label = f"{self._mode_label(mode)}；准备 {strategy.get('max_tool_rounds', 8)} 轮以内的受控执行"
        elif event == "model.requested":
            step = value.get("current_step") or "当前任务"
            label = f"{self._mode_label(mode)}；思考第 {value.get('round')}/{value.get('max_rounds')} 轮；步骤 {step}"
        elif event == "tool.finished":
            outcome = "完成" if value.get("success") else "失败，正在调整"
            label = f"工具 {value.get('tool', 'unknown')} {outcome}"
        elif event == "thinking.content":
            if self.show_reasoning_content and bool(value.get("content")):
                self.show_reasoning(str(value["content"]))
            return
        elif event == "thinking.delta":
            content = str(value.get("content") or "")
            if self.show_reasoning_content and content:
                self.show_reasoning_delta(content)
            return
        else:
            return
        with self._progress_lock:
            self._progress_label = label
        self._print_progress_line()

    def show_reasoning(self, content: str) -> None:
        self._finish_tty_line()
        text = normalize_unicode_text(content).strip()
        if not text:
            return
        print(self._style("\n  ● DeepSeek Thinking", "1;35"))
        for line in text.splitlines():
            print(self._style(f"    {line}", "2;35"))
        print(flush=True)

    def show_reasoning_delta(self, content: str) -> None:
        text = normalize_unicode_text(content)
        if not text:
            return
        self._finish_tty_line()
        if not getattr(self, "_reasoning_stream_open", False):
            print(self._style("\n  ● DeepSeek Thinking", "1;35"))
            print("    ", end="")
            self._reasoning_stream_open = True
        print(self._style(text, "2;35"), end="", flush=True)

    def stop_progress(self, *, clear: bool = True) -> None:
        thread = getattr(self, "_progress_thread", None)
        stop = getattr(self, "_progress_stop", None)
        if stop is not None:
            stop.set()
        if thread and thread is not threading.current_thread():
            thread.join(timeout=0.2)
        self._progress_thread = None
        if getattr(self, "_reasoning_stream_open", False):
            print()
            self._reasoning_stream_open = False
        if clear:
            self._finish_tty_line()

    def _progress_loop(self) -> None:
        self._print_progress_line()
        while not self._progress_stop.wait(1.0):
            self._print_progress_line()

    def _print_progress_line(self) -> None:
        with self._progress_lock:
            elapsed = max(0, int(time.monotonic() - self._progress_started))
            label = self._progress_label
        rendered = self._style(f"  ◐ Thinking {elapsed}s · {label}", "35")
        if sys.stdout.isatty():
            print(f"\r\033[2K{rendered}", end="", flush=True)
        elif elapsed == 0 or elapsed % 10 == 0:
            print(rendered, flush=True)

    @staticmethod
    def _mode_label(mode: str) -> str:
        return {
            "simple": "轻量直答",
            "standard": "标准工程模式",
            "large": "大规模分块模式",
            "deep": "深度任务图模式",
        }.get(mode, "标准工程模式")

    @staticmethod
    def _finish_tty_line() -> None:
        if sys.stdout.isatty():
            print("\r\033[2K", end="", flush=True)

    def error(self, value: str) -> None:
        print(self._style(f"error: {value}", "31"), file=sys.stderr)

    def confirm(self, value: str) -> bool:
        print(f"\n{self._style('approval required', '1;33')}\n{value}")
        try:
            answer = normalize_unicode_text(input(self._style("Apply? [y/N] ", "1;33"))).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        return answer in {"y", "yes", "是", "确认"}

    def help(self) -> None:
        print(
            "\n".join(
                [
                    "Interactive commands:",
                    "  /new                  start a fresh conversation",
                    "  /resume [session-id]  resume the latest or selected session",
                    "  /sessions             list saved sessions",
                    "  /status                show project and active session",
                    "  /undo [snapshot-id]    undo the latest or selected file snapshot",
                    "  /yolo [on|off]         toggle confirmation-free high-risk mode",
                    "  /super-yolo [on|off]   toggle full permission-policy bypass",
                    "  /clear                 clear the terminal",
                    "  /exit                  leave Deep Agent",
                ]
            )
        )

    def clear(self) -> None:
        if sys.stdout.isatty():
            print("\033[2J\033[H", end="")
        else:
            print()
        self.banner()

    def close(self) -> None:
        self.stop_progress()
        if self._readline is None:
            return
        try:
            self._readline.write_history_file(self.history_path)
            self.history_path.chmod(0o600)
        except OSError:
            pass

    def _configure_readline(self):
        try:
            import readline
        except ImportError:
            return None

        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            readline.read_history_file(self.history_path)
        except FileNotFoundError:
            pass
        else:
            try:
                self.history_path.chmod(0o600)
            except OSError:
                pass
        readline.set_history_length(1000)
        readline.set_completer(_command_completer)
        readline.parse_and_bind("tab: complete")
        atexit.register(self.close)
        return readline

    def _style(self, value: str, code: str) -> str:
        if not self.color:
            return value
        start = f"\033[{code}m"
        end = "\033[0m"
        # GNU Readline counts raw ANSI bytes as printed characters unless they
        # are marked as non-printing. Without these markers its cursor and
        # newline calculations break in some WSL terminals.
        if self._readline is not None:
            return f"\001{start}\002{value}\001{end}\002"
        return f"{start}{value}{end}"


def _command_completer(text: str, state: int) -> str | None:
    matches = [command for command in COMMANDS if command.startswith(text)]
    return matches[state] if state < len(matches) else None
