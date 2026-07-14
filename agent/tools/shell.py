from __future__ import annotations

from pathlib import Path

from .base import DEFAULT_MAX_RESULT_SOURCE_BYTES, ToolResult, run_command


class ShellTool:
    name = "shell.run"

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

    def run(self, command: str, cwd: str | None = None, timeout: int | None = None) -> ToolResult:
        workdir = Path(cwd).expanduser().resolve() if cwd else self.cwd
        return run_command(
            ["bash", "-o", "pipefail", "-lc", command],
            cwd=workdir,
            timeout=timeout or self.timeout,
            max_output_bytes=self.max_output_bytes,
        )
