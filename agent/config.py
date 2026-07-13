from __future__ import annotations

import os
import shlex
import fcntl
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from . import paths


DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com",
        "chat_path": "/chat/completions",
        "model": "deepseek-v4-pro",
        "temperature": 0.2,
        "max_tokens": 4096,
        "api_key_env": "DEEPSEEK_API_KEY",
        "reasoning_effort": None,
        "thinking": None,
        "timeout_seconds": 300,
        "network_retries": 2,
        "retry_base_seconds": 1.0,
        "routing": {
            "enabled": True,
            "tier": "auto",
            "fast_model": None,
            "standard_model": None,
            "deep_model": None,
        },
    },
    "runtime": {
        "max_tool_rounds": 8,
        "max_tool_rounds_hard_limit": 32,
        "task_mode": "auto",
        "adaptive_thinking": True,
        "large_project_source_files": 500,
        "large_project_files": 2000,
        "progress_interval_seconds": 10,
        "show_thinking": True,
        "show_reasoning_content": True,
        "max_reasoning_display_chars": 4000,
        "max_user_request_chars": 250000,
        "auto_summarize": True,
        "write_lessons": True,
        "checkpoint_each_tool": True,
        "queue_stop_on_failure": True,
        "parallel_min_tasks": 8,
        "parallel_max_workers": 4,
        "capability_failure_threshold": 3,
    },
    "project": {
        "agent_dir": ".project-agent",
        "ignore_file": "ignore",
        "id_strategy": "uuid",
    },
    "tools": {
        "shell": {"enabled": True, "timeout_seconds": 120},
        "python": {"enabled": True, "timeout_seconds": 120},
        "git": {"enabled": True, "timeout_seconds": 120},
        "document": {"enabled": True, "timeout_seconds": 180, "max_input_bytes": 25_000_000},
        "ocr": {"enabled": True, "timeout_seconds": 180},
        "docker": {"enabled": True, "timeout_seconds": 180},
        "browser": {"enabled": True, "timeout_seconds": 180, "max_download_bytes": 100_000_000},
        "file": {"enabled": True, "max_file_bytes": 2000000},
        "template": {"enabled": True, "timeout_seconds": 300},
        "http": {
            "enabled": False,
            "timeout_seconds": 30,
            "max_response_bytes": 1048576,
            "allowed_domains": [],
        },
        "lsp": {
            "enabled": True,
            "timeout_seconds": 60,
            "max_diagnostics": 200,
            "auto_after_file_apply": True,
        },
    },
    "memory": {
        "sqlite_path": str(paths.memory_db_path()),
        "vector_path": str(paths.vector_dir()),
        "retrieval_limit": 8,
        "vector_enabled": True,
        "smart_reflection": False,
        "dedupe_similarity": 0.94,
        "default_confidence": 0.7,
        "expiry_days": 365,
        "protect_kinds": ["Correction", "Decision"],
    },
    "daemon": {
        "enabled": False,
        "poll_interval_seconds": 10,
        "memory_maintenance_seconds": 3600,
        "queue_enabled": False,
        "queue_timeout_seconds": 3600,
    },
    "context": {
        "max_files": 5000,
        "max_index_file_bytes": 1000000,
        "max_symbol_files": 500,
        "max_prompt_chars": 32000,
        "max_context_file_chars": 8000,
        "max_user_request_chars": 32000,
        "package_limits": {
            "simple": 12000,
            "standard": 32000,
            "large": 48000,
            "deep": 64000,
        },
        "max_package_chars_hard_limit": 96000,
        "max_task_context_chars": 8000,
        "max_session_context_chars": 6000,
        "max_memory_context_chars": 8000,
        "max_capability_context_chars": 8000,
        "max_recovery_context_chars": 6000,
        "semantic_index_enabled": False,
        "semantic_languages": ["python", "javascript", "typescript", "tsx", "java", "go", "rust"],
    },
    "permissions": {
        "enforce": True,
        "restrict_cwd_to_project": True,
        "deny_capabilities": [],
        "auto_approve_capabilities": ["file.apply", "file.undo"],
        "yolo": False,
        "super_yolo": False,
    },
    "events": {
        "jsonl_log": True,
        "metrics_enabled": True,
    },
}


DEFAULT_TOOLS = {
    "tools": {
        "allow_shell": True,
        "allow_python": True,
        "allow_git": True,
        "allow_document": True,
        "allow_ocr": True,
        "allow_docker": True,
        "allow_browser": True,
        "allow_file": True,
        "allow_template": True,
        "capabilities": {
            "shell": {
                "run": {
                    "enabled": True,
                    "permissions": ["read", "write", "execute"],
                    "timeout_seconds": 120,
                    "supports_stream": False,
                    "requires_confirmation": True,
                }
            },
            "python": {
                "run": {
                    "enabled": True,
                    "permissions": ["read", "write", "execute"],
                    "timeout_seconds": 120,
                    "supports_stream": False,
                    "requires_confirmation": True,
                }
            },
            "git": {
                "status": {"enabled": True, "permissions": ["read"]},
                "diff": {"enabled": True, "permissions": ["read"]},
                "log": {"enabled": True, "permissions": ["read"]},
                "add": {"enabled": True, "permissions": ["write"], "requires_confirmation": True},
                "commit": {"enabled": True, "permissions": ["write"], "requires_confirmation": True},
            },
            "document": {
                "parse": {
                    "enabled": True,
                    "permissions": ["read"],
                    "timeout_seconds": 180,
                    "input": ["text", "pdf", "image", "word"],
                    "output": ["markdown"],
                }
            },
            "ocr": {
                "parse": {
                    "enabled": True,
                    "permissions": ["read"],
                    "timeout_seconds": 180,
                    "input": ["pdf", "png", "jpg", "jpeg", "tiff", "webp"],
                    "output": ["markdown"],
                }
            },
            "docker": {
                "run": {
                    "enabled": True,
                    "permissions": ["read", "write", "execute"],
                    "requires_confirmation": True,
                }
            },
            "browser": {
                "open_url": {"enabled": True, "permissions": ["network", "read"]},
                "download": {"enabled": True, "permissions": ["network", "write"]},
                "list_sessions": {"enabled": True, "permissions": ["read"]},
                "close_session": {
                    "enabled": True,
                    "permissions": ["write"],
                    "requires_confirmation": True,
                },
            },
            "file": {
                "diff": {"enabled": True, "permissions": ["read"]},
                "apply": {
                    "enabled": True,
                    "permissions": ["write"],
                    "requires_confirmation": True,
                },
                "undo": {
                    "enabled": True,
                    "permissions": ["write"],
                    "requires_confirmation": True,
                },
            },
            "template": {
                "list_dir": {"enabled": True, "permissions": ["read"]},
                "search_code": {"enabled": True, "permissions": ["read"]},
                "read_file": {"enabled": True, "permissions": ["read"]},
                "find_files": {"enabled": True, "permissions": ["read"]},
                "git_diff_staged": {"enabled": True, "permissions": ["read"]},
                "run_tests": {"enabled": True, "permissions": ["read", "execute"]},
            },
            "http": {
                "request": {
                    "permissions": ["network", "read", "write"],
                    "timeout_seconds": 30,
                    "requires_confirmation": True,
                }
            },
            "lsp": {
                "diagnostics": {
                    "enabled": True,
                    "permissions": ["read", "execute"],
                    "timeout_seconds": 60,
                }
            },
            "memory": {
                "search": {"enabled": True, "permissions": ["read"]},
                "add": {"enabled": True, "permissions": ["write"]},
            },
            "project": {
                "read_context": {"enabled": True, "permissions": ["read"]},
                "write_context": {
                    "enabled": True,
                    "permissions": ["write"],
                    "requires_confirmation": True,
                },
            },
            "agent": {
                "update_plan": {"enabled": True, "permissions": ["state"]},
                "update_step": {"enabled": True, "permissions": ["state"]},
            },
        },
    }
}


DEFAULT_MEMORY = {
    "memory": {
        "lesson_tags": ["lesson", "correction", "reflection", "bug", "decision", "knowledge"],
        "auto_index": True,
        "fts": True,
        "chroma_optional": True,
    }
}


DEFAULT_MCP = {
    "mcp": {
        "enabled": False,
        "startup_timeout_seconds": 15,
        "call_timeout_seconds": 120,
        "resource_timeout_seconds": 60,
        "max_servers": 10,
        "max_tools": 80,
        "servers": [
            {
                "name": "sqlite-example",
                "enabled": False,
                "transport": "stdio",
                "command": str(paths.program_dir() / ".venv" / "bin" / "python"),
                "args": [
                    str(paths.program_dir() / "scripts" / "mcp_sqlite_server.py"),
                    str(paths.data_dir() / "sqlite" / "mcp-example.db"),
                ],
                "tool_allowlist": ["sqlite_query", "sqlite_execute"],
                "env": {},
                "env_passthrough": [],
                "tool_overrides": {
                    "sqlite_query": {
                        "permissions": ["external", "read"],
                        "requires_confirmation": False,
                    },
                    "sqlite_execute": {
                        "permissions": ["external", "write"],
                        "requires_confirmation": True,
                    },
                },
            }
        ],
    }
}


@dataclass(frozen=True)
class AppConfig:
    values: dict[str, Any]
    config_dir: Path
    data_dir: Path

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.values
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    @property
    def api_key(self) -> str | None:
        return self.api_keys[0] if self.api_keys else None

    @property
    def api_keys(self) -> tuple[str, ...]:
        env_name = self.get("model.api_key_env", "DEEPSEEK_API_KEY")
        return parse_api_keys(os.environ.get(env_name) or self.get("model.api_key"))


def parse_api_keys(value: Any) -> tuple[str, ...]:
    """Parse a single key or a comma-separated Key pool without leaking values."""
    if isinstance(value, str):
        candidates = value.replace("，", ",").split(",")
    elif isinstance(value, (list, tuple)):
        candidates = [str(item) for item in value]
    else:
        return ()
    keys: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.strip()
        if key and key not in seen:
            keys.append(key)
            seen.add(key)
    return tuple(keys)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def remove_default_shadows(
    overlay: dict[str, Any],
    primary: dict[str, Any],
    defaults: dict[str, Any],
) -> dict[str, Any]:
    """Drop generated overlay defaults that would hide explicit primary values.

    `model.yaml` remains authoritative when it contains a non-default value.
    A generated default in that file, however, must not erase a user override
    already present in `config.yaml`.
    """
    cleaned: dict[str, Any] = {}
    missing = object()
    for key, value in overlay.items():
        primary_value = primary.get(key, missing)
        default_value = defaults.get(key, missing)
        if isinstance(value, dict):
            nested = remove_default_shadows(
                value,
                primary_value if isinstance(primary_value, dict) else {},
                default_value if isinstance(default_value, dict) else {},
            )
            if nested:
                cleaned[key] = nested
            continue
        if (
            primary_value is not missing
            and default_value is not missing
            and primary_value != default_value
            and value == default_value
        ):
            continue
        cleaned[key] = value
    return cleaned


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def write_yaml_if_missing(path: Path, data: dict[str, Any]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)


def merge_yaml_defaults(path: Path, defaults: dict[str, Any]) -> None:
    """Add new defaults to an existing config without replacing user values."""
    if not path.exists():
        write_yaml_if_missing(path, defaults)
        return
    current = read_yaml(path)
    merged = deep_merge(defaults, current)
    if merged == current:
        return
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temp.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(merged, fh, sort_keys=False, allow_unicode=True)
    temp.replace(path)


def ensure_default_config() -> None:
    paths.ensure_base_dirs()
    cfg = paths.config_dir()
    lock_path = cfg / ".config.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        merge_yaml_defaults(cfg / "config.yaml", DEFAULT_CONFIG)
        merge_yaml_defaults(cfg / "tools.yaml", DEFAULT_TOOLS)
        merge_yaml_defaults(cfg / "memory.yaml", DEFAULT_MEMORY)
        merge_yaml_defaults(cfg / "mcp.yaml", DEFAULT_MCP)
        merge_yaml_defaults(cfg / "model.yaml", {"model": DEFAULT_CONFIG["model"]})
        ensure_mcp_examples(cfg / "mcp.yaml")
        migrate_http_activation(cfg / "tools.yaml")
        try:
            (cfg / "mcp.yaml").chmod(0o600)
        except OSError:
            pass
        ensure_secrets_file(cfg / "secrets.env")


def migrate_http_activation(path: Path) -> None:
    """Remove obsolete HTTP activation flags; config.yaml is the single switch."""
    current = read_yaml(path)
    tools = current.get("tools")
    if not isinstance(tools, dict):
        return
    changed = tools.pop("allow_http", None) is not None
    capabilities = tools.get("capabilities")
    if isinstance(capabilities, dict):
        http = capabilities.get("http")
        request = http.get("request") if isinstance(http, dict) else None
        if isinstance(request, dict) and "enabled" in request:
            request.pop("enabled")
            changed = True
    if not changed:
        return
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(current, handle, sort_keys=False, allow_unicode=True)
    temp.replace(path)


def ensure_mcp_examples(path: Path) -> None:
    """Add disabled built-in examples without changing existing MCP servers."""
    current = read_yaml(path)
    mcp = current.get("mcp")
    if not isinstance(mcp, dict):
        return
    servers = mcp.get("servers")
    if not isinstance(servers, list):
        return
    example = DEFAULT_MCP["mcp"]["servers"][0]
    if any(isinstance(item, dict) and item.get("name") == example["name"] for item in servers):
        return
    updated = deep_merge({}, current)
    updated["mcp"]["servers"] = [*servers, example]
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temp.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(updated, fh, sort_keys=False, allow_unicode=True)
    temp.replace(path)


def ensure_secrets_file(path: Path) -> None:
    if not path.exists():
        path.write_text(
            "# Deep Agent secrets. Keep this file private and outside Git.\n"
            "# DEEPSEEK_API_KEY=replace_with_your_valid_key\n",
            encoding="utf-8",
        )
    path.chmod(0o600)


def load_secrets_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, separator, raw_value = line.partition("=")
        key = key.strip()
        if not separator or not key or not key.replace("_", "").isalnum():
            continue
        try:
            parts = shlex.split(raw_value, comments=True, posix=True)
        except ValueError:
            continue
        value = parts[0] if parts else ""
        os.environ[key] = value


def load_config() -> AppConfig:
    ensure_default_config()
    cfg = paths.config_dir()
    load_secrets_file(cfg / "secrets.env")
    values = dict(DEFAULT_CONFIG)
    primary = read_yaml(cfg / "config.yaml")
    values = deep_merge(values, primary)
    model_overlay = remove_default_shadows(read_yaml(cfg / "model.yaml"), primary, DEFAULT_CONFIG)
    values = deep_merge(values, model_overlay)
    for filename in ("tools.yaml", "memory.yaml", "mcp.yaml"):
        values = deep_merge(values, read_yaml(cfg / filename))
    return AppConfig(values=values, config_dir=cfg, data_dir=paths.data_dir())
