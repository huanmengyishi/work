from __future__ import annotations

from pathlib import Path

from .base import ToolResult, run_command


class ShellTool:
    name = "shell.run"

    def __init__(self, cwd: Path, timeout: int = 120) -> None:
        self.cwd = cwd
        self.timeout = timeout

    def run(self, command: str, cwd: str | None = None, timeout: int | None = None) -> ToolResult:
        workdir = Path(cwd).expanduser().resolve() if cwd else self.cwd
        return run_command(["bash", "-lc", command], cwd=workdir, timeout=timeout or self.timeout)
