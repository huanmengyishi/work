from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from .project import Project
from .timeutil import utc_now_iso


class WorkspaceMemoryManager:
    def __init__(self, project: Project) -> None:
        self.project = project
        self.path = project.agent_dir / "workspace_memory.json"

    def refresh(self, records: list[dict[str, Any]], *, fingerprint: str) -> dict[str, Any]:
        current = self._read()
        if current.get("fingerprint") == fingerprint and isinstance(current.get("detected"), dict):
            current["cache_hit"] = True
            return current
        detected = self._detect(records)
        value = {
            "schema_version": 1,
            "project_id": self.project.id,
            "fingerprint": fingerprint,
            "updated_at": utc_now_iso(),
            "cache_hit": False,
            "detected": detected,
            "manual": current.get("manual") if isinstance(current.get("manual"), dict) else {},
        }
        self._write(value)
        return value

    @staticmethod
    def render(value: dict[str, Any]) -> str:
        detected = value.get("detected") if isinstance(value.get("detected"), dict) else {}
        manual = value.get("manual") if isinstance(value.get("manual"), dict) else {}
        merged = {**detected, **manual}
        lines = ["## Workspace Memory", ""]
        for key in (
            "language",
            "frameworks",
            "package_managers",
            "test_commands",
            "build_commands",
            "run_commands",
            "docker",
            "ci",
            "entry_files",
            "coding_conventions",
            "common_directories",
            "core_modules",
            "databases",
        ):
            item = merged.get(key)
            if item in (None, "", [], {}):
                continue
            rendered = (
                ", ".join(str(part) for part in item)
                if isinstance(item, list)
                else json.dumps(item, ensure_ascii=False)
            )
            lines.append(f"- {key}: {rendered}")
        lines.extend(["", f"Full workspace memory: `{value.get('path', 'workspace_memory.json')}`"])
        return "\n".join(lines)

    def _detect(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        paths = {str(item.get("path") or "") for item in records}
        frameworks: list[str] = []
        package_managers: list[str] = []
        test_commands: list[str] = []
        build_commands: list[str] = []
        run_commands: list[str] = []
        databases: list[str] = []
        conventions = [name for name in ("AGENTS.md", "CLAUDE.md", ".editorconfig", "ruff.toml") if name in paths]

        pyproject = self._text("pyproject.toml")
        package_json = self._json("package.json")
        if "pyproject.toml" in paths or "requirements.txt" in paths:
            package_managers.append("uv" if "uv.lock" in paths else "pip")
            test_commands.append("pytest")
        if "pytest" in pyproject:
            test_commands = ["pytest"]
        if "django" in pyproject.lower() or "manage.py" in paths:
            frameworks.append("Django")
            run_commands.append("python manage.py runserver")
        if "fastapi" in pyproject.lower():
            frameworks.append("FastAPI")
        if package_json:
            package_managers.append("pnpm" if "pnpm-lock.yaml" in paths else "yarn" if "yarn.lock" in paths else "npm")
            scripts = package_json.get("scripts") if isinstance(package_json.get("scripts"), dict) else {}
            for name, command in scripts.items():
                target = f"npm run {name}"
                lowered = str(name).lower()
                if "test" in lowered:
                    test_commands.append(target)
                elif lowered in {"build", "compile"}:
                    build_commands.append(target)
                elif lowered in {"start", "dev", "serve"}:
                    run_commands.append(target)
            dependencies_value = package_json.get("dependencies")
            dev_dependencies_value = package_json.get("devDependencies")
            dependencies = {
                **(dependencies_value if isinstance(dependencies_value, dict) else {}),
                **(dev_dependencies_value if isinstance(dev_dependencies_value, dict) else {}),
            }
            for dependency, framework in (
                ("react", "React"),
                ("next", "Next.js"),
                ("vue", "Vue"),
                ("express", "Express"),
            ):
                if dependency in dependencies:
                    frameworks.append(framework)
        if "Cargo.toml" in paths:
            package_managers.append("cargo")
            test_commands.append("cargo test")
            build_commands.append("cargo build")
        if "go.mod" in paths:
            package_managers.append("go")
            test_commands.append("go test ./...")
            build_commands.append("go build ./...")
        if "pom.xml" in paths:
            package_managers.append("maven")
            test_commands.append("mvn test")
            build_commands.append("mvn package")
        if "build.gradle" in paths or "gradlew" in paths:
            package_managers.append("gradle")
            test_commands.append("./gradlew test")
            build_commands.append("./gradlew build")
        if "project.godot" in paths:
            frameworks.append("Godot")
            run_commands.append("godot --path .")
        manifest_text = "\n".join([pyproject, json.dumps(package_json)]).lower()
        for marker, name in (
            ("postgres", "PostgreSQL"),
            ("mysql", "MySQL"),
            ("sqlite", "SQLite"),
            ("mongodb", "MongoDB"),
            ("redis", "Redis"),
        ):
            if marker in manifest_text:
                databases.append(name)
        directories = sorted({path.split("/", 1)[0] for path in paths if "/" in path and not path.startswith(".")})[:30]
        core_modules = [
            path for path in sorted(paths) if path.startswith(("src/", "app/", "agent/")) and path.count("/") <= 2
        ][:40]
        entries = [
            path
            for path in sorted(paths)
            if Path(path).name
            in {"main.py", "app.py", "manage.py", "index.js", "main.ts", "main.go", "main.rs", "project.godot"}
        ]
        return {
            "language": self.project.language,
            "frameworks": unique(frameworks),
            "package_managers": unique(package_managers),
            "test_commands": unique(test_commands),
            "build_commands": unique(build_commands),
            "run_commands": unique(run_commands),
            "docker": {"enabled": "Dockerfile" in paths or "docker-compose.yml" in paths or "compose.yaml" in paths},
            "ci": sorted(
                path
                for path in paths
                if path.startswith(".github/workflows/") or path in {".gitlab-ci.yml", "Jenkinsfile"}
            ),
            "entry_files": entries,
            "coding_conventions": conventions,
            "common_directories": directories,
            "core_modules": core_modules,
            "databases": unique(databases),
        }

    def _text(self, relative: str) -> str:
        path = self.project.root / relative
        if not path.is_file():
            return ""
        try:
            return path.read_text(encoding="utf-8", errors="replace")[:100_000]
        except OSError:
            return ""

    def _json(self, relative: str) -> dict[str, Any]:
        try:
            value = json.loads(self._text(relative))
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    def _read(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _write(self, value: dict[str, Any]) -> None:
        value["path"] = str(self.path)
        temp = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(self.path)


def unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
