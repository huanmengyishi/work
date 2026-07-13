from __future__ import annotations

from pathlib import Path

from .base import ToolResult, run_command


class GitTool:
    def __init__(self, cwd: Path, timeout: int = 120) -> None:
        self.cwd = cwd
        self.timeout = timeout

    def status(self) -> ToolResult:
        return run_command(["git", "status", "--short", "--branch"], cwd=self.cwd, timeout=self.timeout)

    def diff(self, path: str | None = None) -> ToolResult:
        args = ["git", "diff", "--"]
        if path:
            args.append(path)
        return run_command(args, cwd=self.cwd, timeout=self.timeout)

    def log(self, limit: int = 10) -> ToolResult:
        return run_command(
            ["git", "log", f"--max-count={limit}", "--oneline", "--decorate"],
            cwd=self.cwd,
            timeout=self.timeout,
        )

    def add(self, paths: list[str]) -> ToolResult:
        return run_command(["git", "add", "--", *paths], cwd=self.cwd, timeout=self.timeout)

    def commit(self, message: str) -> ToolResult:
        return run_command(["git", "commit", "-m", message], cwd=self.cwd, timeout=self.timeout)

    def run(self, args: list[str]) -> ToolResult:
        return run_command(["git", *args], cwd=self.cwd, timeout=self.timeout)
