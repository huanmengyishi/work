from __future__ import annotations

from pathlib import Path

from .base import ToolResult, run_command


class DockerTool:
    def __init__(self, cwd: Path, timeout: int = 180) -> None:
        self.cwd = cwd
        self.timeout = timeout

    def run(self, args: list[str], timeout: int | None = None) -> ToolResult:
        return run_command(["docker", *args], cwd=self.cwd, timeout=timeout or self.timeout)

    def ps(self) -> ToolResult:
        return self.run(["ps"])
