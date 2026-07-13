from __future__ import annotations

import os
import hashlib
import re
from pathlib import Path


APP_NAME = "deep-agent"
_SAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._-]+")


def program_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def xdg_config_home() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")).expanduser()


def xdg_data_home() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")).expanduser()


def config_dir() -> Path:
    return xdg_config_home() / APP_NAME


def data_dir() -> Path:
    return xdg_data_home() / APP_NAME


def memory_dir() -> Path:
    return data_dir() / "memory"


def vector_dir() -> Path:
    return data_dir() / "vector"


def sqlite_dir() -> Path:
    return data_dir() / "sqlite"


def cache_dir() -> Path:
    return data_dir() / "cache"


def logs_dir() -> Path:
    return data_dir() / "logs"


def backup_dir() -> Path:
    return data_dir() / "backup"


def daemon_dir() -> Path:
    return data_dir() / "daemon"


def projects_db_path() -> Path:
    return data_dir() / "projects.db"


def memory_db_path() -> Path:
    return sqlite_dir() / "memory.db"


def ensure_base_dirs() -> None:
    for path in (
        config_dir(),
        memory_dir(),
        vector_dir(),
        sqlite_dir(),
        cache_dir(),
        logs_dir(),
        backup_dir(),
        daemon_dir(),
    ):
        path.mkdir(parents=True, exist_ok=True)


def storage_key(value: str, *, maximum: int = 120) -> str:
    """Map an external identifier to one stable, traversal-safe path component."""

    raw = str(value).strip()
    if not raw:
        raise ValueError("storage identifier must not be empty")
    if len(raw) <= maximum and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", raw):
        return raw
    safe = _SAFE_COMPONENT_RE.sub("_", raw).strip("._-") or "id"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    prefix_limit = max(1, maximum - len(digest) - 1)
    return f"{safe[:prefix_limit]}-{digest}"
