from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .base import ToolResult
from .pathsafe import resolve_project_path


SUPPORTED_SUFFIXES = {".py", ".js", ".jsx", ".ts", ".tsx"}
TSC_DIAGNOSTIC_RE = re.compile(
    r"^(?P<file>.+?)\((?P<line>\d+),(?P<column>\d+)\):\s+"
    r"(?P<severity>error|warning)\s+TS(?P<code>\d+):\s+(?P<message>.*)$"
)


@dataclass(frozen=True)
class Diagnostic:
    file: str
    line: int
    column: int
    severity: str
    message: str
    code: str = ""
    source: str = ""


class LSPManager:
    def __init__(self, project_root: Path, *, timeout: int = 60, max_diagnostics: int = 200) -> None:
        self.project_root = project_root
        self.timeout = max(1, min(timeout, 300))
        self.max_diagnostics = max(1, min(max_diagnostics, 1000))

    def diagnostics(self, path: str | None = None) -> ToolResult:
        try:
            target = resolve_project_path(self.project_root, path, require_file=True) if path else self.project_root
        except (FileNotFoundError, ValueError) as exc:
            return ToolResult(False, "", str(exc))
        suffix = target.suffix.lower() if target.is_file() else ""
        if suffix == ".py" or (target.is_dir() and self._has_suffix(target, {".py"})):
            return self._pyright(target)
        if suffix in {".js", ".jsx", ".ts", ".tsx"} or (
            target.is_dir() and self._has_suffix(target, {".js", ".jsx", ".ts", ".tsx"})
        ):
            return self._typescript(target)
        return ToolResult(False, "", "LSP diagnostics support Python, JavaScript, and TypeScript only")

    def available(self) -> tuple[bool, str]:
        pyright = shutil.which("pyright")
        tsc = shutil.which("tsc")
        if pyright or tsc:
            engines = [name for name, path in (("Pyright", pyright), ("TypeScript", tsc)) if path]
            return True, f"available diagnostics engines: {', '.join(engines)}"
        missing = [name for name, path in (("pyright", pyright), ("tsc", tsc)) if not path]
        return False, f"missing diagnostics engine: {', '.join(missing)}"

    def _pyright(self, target: Path) -> ToolResult:
        executable = shutil.which("pyright")
        if not executable:
            return ToolResult(False, "", "pyright is not installed")
        completed = self._run([executable, "--outputjson", str(target)])
        if completed is None:
            return ToolResult(False, "", f"pyright timeout after {self.timeout}s")
        try:
            payload = json.loads(completed.stdout or "{}")
        except json.JSONDecodeError:
            return ToolResult(False, "", (completed.stderr or completed.stdout or "invalid pyright output")[:8000])
        diagnostics = []
        for item in payload.get("generalDiagnostics", []):
            if not isinstance(item, dict):
                continue
            start = (item.get("range") or {}).get("start") or {}
            diagnostics.append(
                Diagnostic(
                    file=self._relative(str(item.get("file") or "")),
                    line=int(start.get("line") or 0) + 1,
                    column=int(start.get("character") or 0) + 1,
                    severity=str(item.get("severity") or "error").title(),
                    message=str(item.get("message") or ""),
                    code=str(item.get("rule") or ""),
                    source="pyright",
                )
            )
        return self._result(diagnostics, "pyright", completed.returncode)

    def _typescript(self, target: Path) -> ToolResult:
        executable = shutil.which("tsc")
        if not executable:
            return ToolResult(False, "", "TypeScript compiler is not installed")
        project_config = self._nearest_tsconfig(target)
        if project_config:
            args = [executable, "--pretty", "false", "--noEmit", "-p", str(project_config)]
        else:
            files = [target] if target.is_file() else self._source_files(target, {".js", ".jsx", ".ts", ".tsx"})
            if not files:
                return ToolResult(True, "No JavaScript/TypeScript files found.", data={"diagnostics": []})
            args = [
                executable,
                "--pretty",
                "false",
                "--noEmit",
                "--allowJs",
                "--checkJs",
                "--skipLibCheck",
                "--target",
                "ES2022",
                "--module",
                "NodeNext",
                "--moduleResolution",
                "NodeNext",
                *[str(path) for path in files[:500]],
            ]
        completed = self._run(args)
        if completed is None:
            return ToolResult(False, "", f"TypeScript diagnostics timeout after {self.timeout}s")
        diagnostics = []
        for line in (completed.stdout + "\n" + completed.stderr).splitlines():
            match = TSC_DIAGNOSTIC_RE.match(line.strip())
            if not match:
                continue
            diagnostics.append(
                Diagnostic(
                    file=self._relative(match.group("file")),
                    line=int(match.group("line")),
                    column=int(match.group("column")),
                    severity=match.group("severity").title(),
                    message=match.group("message"),
                    code=f"TS{match.group('code')}",
                    source="typescript",
                )
            )
        return self._result(diagnostics, "typescript", completed.returncode)

    def _result(self, diagnostics: list[Diagnostic], engine: str, returncode: int) -> ToolResult:
        limited = diagnostics[: self.max_diagnostics]
        data: dict[str, Any] = {
            "engine": engine,
            "diagnostics": [asdict(item) for item in limited],
            "error_count": sum(item.severity.lower() == "error" for item in limited),
            "warning_count": sum(item.severity.lower() == "warning" for item in limited),
            "truncated": len(diagnostics) > len(limited),
            "returncode": returncode,
        }
        lines = [
            f"{item.severity} {item.file}:{item.line}:{item.column} {item.code} {item.message}".strip()
            for item in limited
        ]
        output = "\n".join(lines) or f"{engine}: no diagnostics"
        return ToolResult(
            data["error_count"] == 0,
            output if data["error_count"] == 0 else "",
            output if data["error_count"] else "",
            data=data,
        )

    def _run(self, args: list[str]) -> subprocess.CompletedProcess[str] | None:
        try:
            return subprocess.run(
                args,
                cwd=self.project_root,
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None

    def _nearest_tsconfig(self, target: Path) -> Path | None:
        start = target.parent if target.is_file() else target
        for current in (start, *start.parents):
            if current == self.project_root.parent:
                break
            candidate = current / "tsconfig.json"
            if candidate.is_file():
                return candidate
            if current == self.project_root:
                break
        return None

    def _relative(self, value: str) -> str:
        path = Path(value)
        try:
            return path.resolve().relative_to(self.project_root.resolve()).as_posix()
        except (OSError, ValueError):
            return value

    @staticmethod
    def _has_suffix(root: Path, suffixes: set[str]) -> bool:
        return bool(LSPManager._source_files(root, suffixes, limit=1))

    @staticmethod
    def _source_files(root: Path, suffixes: set[str], *, limit: int = 500) -> list[Path]:
        ignored = {".git", ".project-agent", ".venv", "venv", "node_modules", "dist", "build", "__pycache__"}
        files: list[Path] = []
        for current, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(name for name in dirnames if name not in ignored)
            for name in sorted(filenames):
                path = Path(current) / name
                if path.suffix.lower() in suffixes:
                    files.append(path)
                    if len(files) >= limit:
                        return files
        return files
