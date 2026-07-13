from __future__ import annotations

from pathlib import Path


PRIVATE_AGENT_DIRS = {
    "cache",
    "sessions",
    "snapshots",
    "browser-sessions",
    "downloads",
    "queues",
    "parallel",
    "memory",
}


def resolve_project_path(
    project_root: Path,
    value: str,
    *,
    require_file: bool = False,
    allow_private_agent_data: bool = False,
) -> Path:
    """Resolve a user-provided path without allowing project-root escapes."""
    if not value or "\x00" in value:
        raise ValueError("path is empty or contains a NUL byte")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    resolved = candidate.resolve(strict=False)
    try:
        relative = resolved.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError(f"path is outside the current project: {candidate}") from exc
    if (
        not allow_private_agent_data
        and len(relative.parts) >= 2
        and relative.parts[0] == ".project-agent"
        and relative.parts[1] in PRIVATE_AGENT_DIRS
    ):
        raise ValueError(f"path is private Agent data and cannot be accessed by project tools: {relative}")
    if require_file and resolved.exists() and not resolved.is_file():
        raise ValueError(f"path is not a regular file: {resolved}")
    return resolved
