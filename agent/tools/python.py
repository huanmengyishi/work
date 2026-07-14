from __future__ import annotations

import sys
from pathlib import Path

from .base import DEFAULT_MAX_RESULT_SOURCE_BYTES, ToolResult, run_command


class PythonTool:
    name = "python.run"

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

    def run(self, code: str, cwd: str | None = None, timeout: int | None = None) -> ToolResult:
        workdir = Path(cwd).expanduser().resolve() if cwd else self.cwd
        return run_command(
            [sys.executable, "-c", code],
            cwd=workdir,
            timeout=timeout or self.timeout,
            max_output_bytes=self.max_output_bytes,
        )
