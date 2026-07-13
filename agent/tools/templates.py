from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from .base import ToolResult, run_command, truncate_text
from .pathsafe import resolve_project_path


class SafeTemplateTool:
    """Parameterised high-frequency operations that never interpolate a shell command."""

    def __init__(self, project_root: Path, timeout: int = 120) -> None:
        self.project_root = project_root
        self.timeout = timeout

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
        result = run_command(args, cwd=self.project_root, timeout=self.timeout)
        if not result.success and result.data and result.data.get("returncode") == 1:
            return ToolResult(True, "", data={"count": 0})
        if not result.success:
            return result
        lines = result.stdout.splitlines()[: max(1, min(max_results, 1000))]
        return ToolResult(True, "\n".join(lines), data={"count": len(lines)})

    def read_file(self, path: str, start_line: int = 1, end_line: int = 240) -> ToolResult:
        target = resolve_project_path(self.project_root, path, require_file=True)
        if not target.exists():
            return ToolResult(False, "", f"file does not exist: {path}")
        start = max(1, start_line)
        end = max(start, min(end_line, start + 2000))
        try:
            lines = target.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            return ToolResult(False, "", f"file is not UTF-8 text: {path}")
        rendered = "\n".join(f"{number:>6}  {lines[number - 1]}" for number in range(start, min(end, len(lines)) + 1))
        return ToolResult(True, truncate_text(rendered), data={"path": path, "line_count": len(lines)})

    def find_files(self, pattern: str = "*", path: str = ".", max_results: int = 500) -> ToolResult:
        root = resolve_project_path(self.project_root, path)
        rg = shutil.which("rg")
        if rg:
            result = run_command(
                [rg, "--files", "--glob", pattern, str(root)], cwd=self.project_root, timeout=self.timeout
            )
            if result.success:
                lines = result.stdout.splitlines()[: max(1, min(max_results, 5000))]
                return ToolResult(True, "\n".join(lines), data={"count": len(lines)})
            return result
        values = [str(item.relative_to(self.project_root)) for item in root.rglob(pattern) if item.is_file()]
        values = values[: max(1, min(max_results, 5000))]
        return ToolResult(True, "\n".join(values), data={"count": len(values)})

    def git_diff_staged(self, path: str | None = None) -> ToolResult:
        args = ["git", "diff", "--cached", "--"]
        if path:
            target = resolve_project_path(self.project_root, path)
            args.append(str(target.relative_to(self.project_root)))
        return run_command(args, cwd=self.project_root, timeout=self.timeout)

    def run_tests(self, framework: str = "auto", path: str = ".") -> ToolResult:
        cwd = resolve_project_path(self.project_root, path)
        selected = self._detect_framework(cwd) if framework == "auto" else framework
        commands = {
            "pytest": [self._project_python(), "-m", "pytest"],
            "npm": ["npm", "test"],
            "cargo": ["cargo", "test"],
            "go": ["go", "test", "./..."],
            "gradle": [str(cwd / "gradlew"), "test"],
            "maven": ["mvn", "test"],
        }
        args = commands.get(selected)
        if not args:
            return ToolResult(False, "", f"unsupported or undetected test framework: {selected}")
        return run_command(args, cwd=cwd, timeout=self.timeout)

    def _project_python(self) -> str:
        candidate = self.project_root / ".venv" / "bin" / "python"
        return str(candidate) if candidate.exists() else sys.executable

    @staticmethod
    def _detect_framework(cwd: Path) -> str:
        if (cwd / "pyproject.toml").exists() or (cwd / "pytest.ini").exists() or (cwd / "tests").exists():
            return "pytest"
        if (cwd / "package.json").exists():
            return "npm"
        if (cwd / "Cargo.toml").exists():
            return "cargo"
        if (cwd / "go.mod").exists():
            return "go"
        if (cwd / "gradlew").exists():
            return "gradle"
        if (cwd / "pom.xml").exists():
            return "maven"
        return "unknown"
