from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class ToolRequest:
    tool: str
    action: str
    args: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid4()))
    model_name: str | None = None

    @property
    def capability(self) -> str:
        return f"{self.tool}.{self.action}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolResult:
    success: bool
    stdout: str
    stderr: str = ""
    data: dict[str, Any] | None = None
    duration_ms: int = 0
    request_id: str | None = None

    @property
    def ok(self) -> bool:
        return self.success

    @property
    def output(self) -> str:
        return self.stdout

    @property
    def error(self) -> str:
        return self.stderr

    def with_execution(self, *, request_id: str, duration_ms: int | None = None) -> "ToolResult":
        return replace(
            self,
            request_id=request_id,
            duration_ms=self.duration_ms if self.duration_ms else int(duration_ms or 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": self.duration_ms,
            "request_id": self.request_id,
            "data": self.data or {},
        }

    def as_text(self, limit: int = 12000) -> str:
        text = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        if len(text) > limit:
            return text[:limit] + "\n...[truncated]"
        return text


def run_command(
    args: list[str],
    *,
    cwd: Path,
    timeout: int,
    input_text: str | None = None,
    shell: bool = False,
    env: dict[str, str] | None = None,
) -> ToolResult:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            args if not shell else " ".join(args),
            cwd=str(cwd),
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout,
            shell=shell,
            check=False,
            env=env,
        )
    except FileNotFoundError as exc:
        return ToolResult(False, "", f"command not found: {exc}", duration_ms=elapsed_ms(started))
    except subprocess.TimeoutExpired as exc:
        return ToolResult(
            False,
            (exc.stdout or "") if isinstance(exc.stdout, str) else "",
            f"timeout after {timeout}s",
            duration_ms=elapsed_ms(started),
        )
    output = completed.stdout or ""
    error = completed.stderr or ""
    return ToolResult(
        completed.returncode == 0,
        output.strip(),
        error.strip(),
        {"returncode": completed.returncode, "args": args},
        elapsed_ms(started),
    )


def truncate_text(text: str, limit: int = 20000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def elapsed_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1000))
