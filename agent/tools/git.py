from __future__ import annotations

from pathlib import Path

from .base import DEFAULT_MAX_RESULT_SOURCE_BYTES, ToolResult, run_command


class GitTool:
    def __init__(
        self,
        cwd: Path,
        timeout: int = 120,
        *,
        max_output_bytes: int = DEFAULT_MAX_RESULT_SOURCE_BYTES,
    ) -> None:
        self.cwd = cwd
        self.timeout = timeout
        self.max_output_bytes = max_output_bytes

    def status(self) -> ToolResult:
        return self._run(["git", "status", "--short", "--branch"])

    def diff(self, path: str | None = None) -> ToolResult:
        args = ["git", "diff", "--"]
        if path:
            args.append(path)
        return self._run(args)

    def log(self, limit: int = 10) -> ToolResult:
        return self._run(["git", "log", f"--max-count={limit}", "--oneline", "--decorate"])

    def add(self, paths: list[str]) -> ToolResult:
        return self._run(["git", "add", "--", *paths])

    def commit(self, message: str) -> ToolResult:
        return self._run(["git", "commit", "-m", message])

    def run(self, args: list[str]) -> ToolResult:
        return self._run(["git", *args])

    def _run(self, args: list[str]) -> ToolResult:
        return run_command(
            args,
            cwd=self.cwd,
            timeout=self.timeout,
            max_output_bytes=self.max_output_bytes,
        )
