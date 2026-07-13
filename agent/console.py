from __future__ import annotations

import atexit
import os
import sys
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
    ) -> None:
        self.project = project
        self.yolo = yolo
        self.super_yolo = super_yolo
        self.history_path = data_dir / "cache" / "repl_history"
        self.color = bool(sys.stdout.isatty() and os.environ.get("TERM") != "dumb" and "NO_COLOR" not in os.environ)
        self._readline = self._configure_readline()

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
        return normalize_unicode_text(input(self.prompt(session_id))).strip()

    def answer(self, value: str) -> None:
        print(f"\n{self._style('assistant', '1;36')}\n{normalize_unicode_text(value)}\n")

    def info(self, value: str) -> None:
        print(self._style(value, "36"))

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
        return f"\033[{code}m{value}\033[0m" if self.color else value


def _command_completer(text: str, state: int) -> str | None:
    matches = [command for command in COMMANDS if command.startswith(text)]
    return matches[state] if state < len(matches) else None
