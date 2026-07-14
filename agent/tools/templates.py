from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from .base import (
    DEFAULT_MAX_RESULT_SOURCE_BYTES,
    BoundedByteCapture,
    ToolResult,
    bounded_result_source_bytes,
    run_command,
)
from .pathsafe import resolve_project_path


class SafeTemplateTool:
    """Parameterised high-frequency operations that never interpolate a shell command."""

    def __init__(
        self,
        project_root: Path,
        timeout: int = 120,
        *,
        max_input_bytes: int = 64 * 1024 * 1024,
        max_result_bytes: int = DEFAULT_MAX_RESULT_SOURCE_BYTES,
    ) -> None:
        self.project_root = project_root
        self.timeout = timeout
        self.max_input_bytes = max(1024, min(int(max_input_bytes), 256 * 1024 * 1024))
        self.max_result_bytes = bounded_result_source_bytes(max_result_bytes)

    def list_dir(self, path: str = ".", depth: int = 2, max_entries: int = 500) -> ToolResult:
        root = resolve_project_path(self.project_root, path)
        if not root.exists() or not root.is_dir():
            return ToolResult(False, "", f"directory does not exist: {path}")
        depth = max(0, min(depth, 8))
        max_entries = max(1, min(max_entries, 5000))
        lines: list[str] = []
        root_parts = len(root.parts)
        for current, dirs, files in os.walk(root):
            current_path = Path(current)
            level = len(current_path.parts) - root_parts
            dirs[:] = sorted(item for item in dirs if item not in {".git", ".project-agent", "node_modules"})
            if level >= depth:
                dirs[:] = []
            prefix = "  " * level
            if level:
                lines.append(f"{prefix}{current_path.name}/")
            lines.extend(f"{prefix}  {name}" for name in sorted(files))
            if len(lines) >= max_entries:
                lines.append("...[entry limit reached]")
                break
        return ToolResult(True, "\n".join(lines[: max_entries + 1]), data={"root": str(root)})

    def search_code(
        self,
        query: str,
        path: str = ".",
        glob: str | None = None,
        max_results: int = 200,
    ) -> ToolResult:
        root = resolve_project_path(self.project_root, path)
        if not query:
            return ToolResult(False, "", "search query is empty")
        rg = shutil.which("rg")
        if not rg:
            return ToolResult(False, "", "rg is required for search_code")
        args = [rg, "--line-number", "--color", "never", "--max-count", str(max(1, min(max_results, 1000)))]
        if glob:
            args.extend(["--glob", glob])
        args.extend(["--", query, str(root)])
        result = run_command(
            args,
            cwd=self.project_root,
            timeout=self.timeout,
            max_output_bytes=self.max_result_bytes,
        )
        if not result.success and result.data and result.data.get("returncode") == 1:
            return ToolResult(True, "", data={"count": 0})
        if not result.success:
            return result
        lines = result.stdout.splitlines()[: max(1, min(max_results, 1000))]
        data = dict(result.data or {})
        data["count"] = len(lines)
        return ToolResult(True, "\n".join(lines), data=data)

    def read_file(self, path: str, start_line: int = 1, end_line: int = 240) -> ToolResult:
        target = resolve_project_path(self.project_root, path, require_file=True)
        if not target.exists():
            return ToolResult(False, "", f"file does not exist: {path}")
        if end_line < start_line:
            return ToolResult(False, "", "end_line must be greater than or equal to start_line")
        start = max(1, start_line)
        end = max(start, min(end_line, start + 2000))
        try:
            input_bytes = target.stat().st_size
        except OSError as exc:
            return ToolResult(False, "", f"could not inspect file: {path}: {exc}")
        if input_bytes > self.max_input_bytes:
            return ToolResult(False, "", f"file exceeds the {self.max_input_bytes} byte read limit: {path}")
        capture = BoundedByteCapture(self.max_result_bytes)
        line_count = 0
        try:
            with target.open("r", encoding="utf-8", errors="strict") as handle:
                for line_count, line in enumerate(handle, start=1):
                    if start <= line_count <= end:
                        # Keep the line-number boundary visually and mechanically
                        # unambiguous.  Two plain separator spaces are
                        # indistinguishable from source indentation and can be
                        # copied into file_diff old_text. The fixed reference
                        # likewise uses an explicit tab-or-arrow boundary
                        # rather than an ambiguous run of plain spaces.
                        rendered = f"{line_count:>6}→{line.rstrip(chr(13) + chr(10))}\n"
                        capture.feed(rendered.encode("utf-8"))
        except UnicodeDecodeError:
            return ToolResult(False, "", f"file is not UTF-8 text: {path}")
        data = {"path": path, "line_count": line_count, "input_bytes": input_bytes, **capture.metadata()}
        return ToolResult(True, capture.text().rstrip("\n"), data=data)

    def find_files(self, pattern: str = "*", path: str = ".", max_results: int = 500) -> ToolResult:
        root = resolve_project_path(self.project_root, path)
        rg = shutil.which("rg")
        if rg:
            result = run_command(
                [rg, "--files", "--glob", pattern, str(root)],
                cwd=self.project_root,
                timeout=self.timeout,
                max_output_bytes=self.max_result_bytes,
            )
            if result.success:
                lines = result.stdout.splitlines()[: max(1, min(max_results, 5000))]
                data = dict(result.data or {})
                data["count"] = len(lines)
                return ToolResult(True, "\n".join(lines), data=data)
            return result
        values = [str(item.relative_to(self.project_root)) for item in root.rglob(pattern) if item.is_file()]
        values = values[: max(1, min(max_results, 5000))]
        return ToolResult(True, "\n".join(values), data={"count": len(values)})

    def git_diff_staged(self, path: str | None = None) -> ToolResult:
        args = ["git", "diff", "--cached", "--"]
        if path:
            target = resolve_project_path(self.project_root, path)
            args.append(str(target.relative_to(self.project_root)))
        return run_command(
            args,
            cwd=self.project_root,
            timeout=self.timeout,
            max_output_bytes=self.max_result_bytes,
        )

    def run_tests(self, framework: str = "auto", path: str = ".") -> ToolResult:
        cwd = resolve_project_path(self.project_root, path)
        selected = self._detect_framework(cwd) if framework == "auto" else framework
        if selected == "npm":
            return self._run_package_script(cwd, requested=None)
        if selected.startswith("npm:"):
            return self._run_package_script(cwd, requested=selected.partition(":")[2])
        commands = {
            "pytest": [self._project_python(), "-m", "pytest"],
            "cargo": ["cargo", "test"],
            "go": ["go", "test", "./..."],
            "gradle": [str(cwd / "gradlew"), "test"],
            "maven": ["mvn", "test"],
        }
        args = commands.get(selected)
        if not args:
            return ToolResult(False, "", f"unsupported or undetected test framework: {selected}")
        return run_command(args, cwd=cwd, timeout=self.timeout, max_output_bytes=self.max_result_bytes)

    def _run_package_script(self, cwd: Path, requested: str | None) -> ToolResult:
        package_path = cwd / "package.json"
        try:
            if package_path.stat().st_size > 1_000_000:
                return ToolResult(False, "", "package.json exceeds the 1000000 byte validation limit")
            package = json.loads(package_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return ToolResult(False, "", "package.json does not exist")
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return ToolResult(False, "", f"package.json could not be parsed: {type(exc).__name__}")
        scripts = package.get("scripts") if isinstance(package, dict) else None
        if not isinstance(scripts, dict):
            scripts = {}
        allowed = ("test", "typecheck", "check", "lint", "build")
        available = [name for name in allowed if isinstance(scripts.get(name), str) and scripts[name].strip()]
        script = requested or (available[0] if available else "")
        if not script:
            return ToolResult(False, "", "no supported package validation script exists; available: none")
        if script not in allowed:
            return ToolResult(False, "", f"unsupported package validation script: {script or 'none'}")
        if script not in available:
            shown = ", ".join(available) or "none"
            return ToolResult(False, "", f"package validation script '{script}' is missing; available: {shown}")
        return run_command(
            ["npm", "run", script],
            cwd=cwd,
            timeout=self.timeout,
            max_output_bytes=self.max_result_bytes,
        )

    def _project_python(self) -> str:
        candidate = self.project_root / ".venv" / "bin" / "python"
        return str(candidate) if candidate.exists() else sys.executable

    @staticmethod
    def _detect_framework(cwd: Path) -> str:
        if (cwd / "pyproject.toml").exists() or (cwd / "pytest.ini").exists():
            return "pytest"
        if (cwd / "package.json").exists():
            return "npm"
        # A generic tests/ directory is not a Python project marker.  Many
        # JavaScript/TypeScript repositories use the same name, so consider it
        # only after all language-specific package manifest checks below.
        if (cwd / "Cargo.toml").exists():
            return "cargo"
        if (cwd / "go.mod").exists():
            return "go"
        if (cwd / "gradlew").exists():
            return "gradle"
        if (cwd / "pom.xml").exists():
            return "maven"
        if (cwd / "tests").exists():
            return "pytest"
        return "unknown"
