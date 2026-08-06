from __future__ import annotations

import difflib
import logging
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from . import paths
from .file_lock import lock_exclusive


logger = logging.getLogger(__name__)


DEFAULT_CONFIG: dict[str, Any] = {
    "model": {
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com",
        "chat_path": "/chat/completions",
        "model": "deepseek-v4-pro",
        "context_window_tokens": 65_536,
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
        "budget": {
            "enabled": True,
            "max_model_requests_per_turn": 64,
            "max_total_tokens_per_turn": 1_000_000,
            "max_elapsed_seconds_per_turn": 3_600,
        },
        "resilience": {
            "max_corrective_rounds": 2,
            "max_abnormal_finish_recoveries": 1,
            "capability_backoff_enabled": True,
            "max_capability_backoff_rounds": 8,
            "circuit_recovery_rounds": 4,
        },
        "queue_stop_on_failure": True,
        "parallel_min_tasks": 8,
        "parallel_max_workers": 4,
        "parallel_subprocess_timeout_seconds": 3_600,
        "capability_failure_threshold": 3,
        "convergence": {
            "enabled": True,
            "max_consecutive_exploration_rounds": 6,
            "reserved_tool_rounds": 4,
            "max_tool_calls_per_round": 16,
            "max_parallel_read_tools": 4,
            "max_length_continuations": 2,
            "max_implementation_evidence_reads": 2,
            "max_validation_attachment_reads": 2,
            "single_tool_result_chars": 12_000,
            "same_round_tool_result_chars": 48_000,
            "aggregate_tool_result_chars": 96_000,
            "output_reserve_chars": 24_000,
            "compacted_tool_result_chars": 1_200,
            "keep_recent_tool_results": 4,
            "compaction_failure_limit": 3,
            "history_snip_enabled": True,
            "history_snip_min_chars": 24_000,
            "history_snip_min_complete_rounds": 8,
            "history_snip_keep_recent_rounds": 4,
            "history_snip_marker_chars": 768,
            "auto_compaction_enabled": True,
            "auto_compaction_max_tokens": 2_048,
            "context_safety_buffer_tokens": 8_192,
        },
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
        "document_generator": {"enabled": True, "timeout_seconds": 180},
        "ocr": {"enabled": True, "timeout_seconds": 180},
        "docker": {"enabled": True, "timeout_seconds": 180},
        "browser": {"enabled": True, "timeout_seconds": 180, "max_download_bytes": 100_000_000},
        "file": {"enabled": True, "max_file_bytes": 2000000},
        "template": {
            "enabled": True,
            "timeout_seconds": 300,
            "max_input_bytes": 67_108_864,
        },
        "tool_result": {
            "enabled": True,
            "max_attachment_bytes": 8_388_608,
            "persist_threshold_bytes": 12_000,
            "preview_chars": 12_000,
            "max_read_chars": 32_000,
            "max_attachments_per_session": 512,
            "max_session_bytes": 268_435_456,
        },
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
        "search_max_query_chars": 4_096,
        "search_max_query_bytes": 16_384,
        "vector_enabled": False,
        "query_cache_max_entries": 128,
        "query_cache_ttl_seconds": 60,
        "smart_reflection": True,
        "smart_reflection_min_tool_calls": 5,
        "smart_reflection_max_input_chars": 12_000,
        "smart_reflection_max_output_tokens": 768,
        "smart_reflection_max_output_chars": 5_000,
        "dedupe_similarity": 0.94,
        "default_confidence": 0.7,
        "expiry_days": 365,
        "protect_kinds": ["Correction", "Decision"],
        "max_items": 5000,
        "max_storage_mb": 100,
        "capacity_scan_limit": 5000,
        "capacity_report_limit": 100,
        "transfer": {
            "max_file_bytes": 8_388_608,
            "max_records": 5_000,
            "max_path_chars": 4_096,
            "max_title_chars": 500,
            "max_title_bytes": 2_000,
            "max_content_chars": 50_000,
            "max_content_bytes": 200_000,
            "max_tags_per_record": 32,
            "max_tag_chars": 100,
            "max_tag_bytes": 400,
        },
        "confidence": {
            "use_bonus": 0.02,
            "contradiction_penalty": 0.15,
            "lower_bound": 0.1,
            "upper_bound": 0.95,
        },
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
        "performance_history_enabled": True,
        "performance_history_max_records": 200,
    },
    "optimizer": {
        "strategy_adjustment_enabled": False,
        "adaptive_convergence_enabled": False,
        "min_samples": 8,
        "history_limit": 100,
        "failure_upgrade_threshold": 0.4,
        "success_downgrade_threshold": 0.9,
        "experiments": {
            "enabled": False,
            "salt": "deep-agent",
            "definitions": [],
        },
    },
}


DEFAULT_TOOLS = {
    "tools": {
        "allow_shell": True,
        "allow_python": True,
        "allow_git": True,
        "allow_document": True,
        "allow_document_generator": True,
        "allow_ocr": True,
        "allow_docker": True,
        "allow_browser": True,
        "allow_file": True,
        "allow_template": True,
        "document": {
            "max_input_bytes": 25_000_000,
            "max_render_chars": 250_000,
        },
        "document_generator": {"timeout_seconds": 180},
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
                },
                "render_docx": {
                    "enabled": True,
                    "permissions": ["write"],
                    "timeout_seconds": 180,
                    "input": ["markdown"],
                    "output": ["docx-preview"],
                },
            },
            "document_generator": {
                "create_outline": {
                    "enabled": True,
                    "permissions": ["state"],
                    "timeout_seconds": 180,
                },
                "confirm_outline": {
                    "enabled": True,
                    "permissions": ["state"],
                    "timeout_seconds": 180,
                    "requires_confirmation": True,
                },
                "next_chapter": {
                    "enabled": True,
                    "permissions": ["state"],
                    "timeout_seconds": 180,
                },
                "save_chapter": {
                    "enabled": True,
                    "permissions": ["state"],
                    "timeout_seconds": 180,
                },
                "rollback_chapter": {
                    "enabled": True,
                    "permissions": ["state"],
                    "timeout_seconds": 180,
                },
                "status": {
                    "enabled": True,
                    "permissions": ["read"],
                    "timeout_seconds": 180,
                },
                "render": {
                    "enabled": True,
                    "permissions": ["read", "write"],
                    "timeout_seconds": 180,
                },
                "finalize": {
                    "enabled": True,
                    "permissions": ["read", "state"],
                    "timeout_seconds": 180,
                },
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
            "tool_result": {
                "read": {"enabled": True, "permissions": ["read"]},
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
                "command": "{python}",
                "args": [
                    "-m",
                    "agent.tools.mcp_sqlite_server",
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

    def get(self, dotted: str, default: Any = None, *, warn_on_missing: bool = False) -> Any:
        cur: Any = self.values
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                if warn_on_missing:
                    logger.warning("config_missing_key key=%s", dotted)
                return default
            cur = cur[part]
        return cur

    def get_int(
        self,
        dotted: str,
        default: int,
        *,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> int:
        """Read one external integer safely and clamp it to explicit bounds."""

        value = self.get(dotted, default)
        if isinstance(value, bool):
            parsed = default
            valid = False
        else:
            try:
                parsed = int(value)
                valid = True
            except (TypeError, ValueError, OverflowError):
                parsed = default
                valid = False
        if not valid:
            logger.warning("config_invalid_integer key=%s action=use_default", dotted)
        if minimum is not None and parsed < minimum:
            logger.warning("config_integer_below_minimum key=%s action=clamp", dotted)
            parsed = minimum
        if maximum is not None and parsed > maximum:
            logger.warning("config_integer_above_maximum key=%s action=clamp", dotted)
            parsed = maximum
        return parsed

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


def warn_unknown_config_keys(
    values: dict[str, Any],
    schema: dict[str, Any],
    *,
    source: str,
    prefix: str = "",
) -> None:
    """Warn once per unknown mapping key without logging its configured value."""

    for key, value in values.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if key not in schema:
            suggestion = difflib.get_close_matches(str(key), [str(item) for item in schema], n=1, cutoff=0.72)
            suffix = f" suggestion={suggestion[0]}" if suggestion else ""
            logger.warning("config_unknown_key source=%s key=%s%s", source, dotted, suffix)
            continue
        expected = schema[key]
        if isinstance(value, dict) and isinstance(expected, dict) and expected:
            warn_unknown_config_keys(value, expected, source=source, prefix=dotted)


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
        lock_exclusive(lock)
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
    """Add or safely migrate the exact disabled built-in SQLite example."""
    current = read_yaml(path)
    mcp = current.get("mcp")
    if not isinstance(mcp, dict):
        return
    servers = mcp.get("servers")
    if not isinstance(servers, list):
        return
    example = DEFAULT_MCP["mcp"]["servers"][0]
    updated = deep_merge({}, current)
    replacement = list(servers)
    matching_indexes = [
        index for index, item in enumerate(servers) if isinstance(item, dict) and item.get("name") == example["name"]
    ]
    if matching_indexes:
        index = matching_indexes[0]
        item = servers[index]
        if not isinstance(item, dict) or not _is_managed_sqlite_example(item, example):
            return
        if item == example:
            return
        replacement[index] = example
    else:
        replacement.append(example)
    updated["mcp"]["servers"] = replacement
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temp.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(updated, fh, sort_keys=False, allow_unicode=True)
    temp.replace(path)


def _is_managed_sqlite_example(value: dict[str, Any], current: dict[str, Any]) -> bool:
    if (
        value.get("name") != current.get("name")
        or value.get("enabled") is not False
        or value.get("transport") != "stdio"
        or value.get("tool_allowlist") != current.get("tool_allowlist")
        or value.get("env", {}) != {}
        or value.get("env_passthrough", []) != []
        or value.get("tool_overrides") != current.get("tool_overrides")
    ):
        return False
    args = value.get("args")
    if not isinstance(args, list):
        return False
    database = str(current.get("args", ["", "", ""])[-1])
    packaged = len(args) == 3 and args[:2] == ["-m", "agent.tools.mcp_sqlite_server"]
    legacy_script = (
        len(args) == 2
        and Path(str(args[0])).name == "mcp_sqlite_server.py"
        and Path(str(args[0])).parent.name == "scripts"
    )
    return (packaged or legacy_script) and str(args[-1]) == database


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


def _config_schemas() -> dict[str, dict[str, Any]]:
    all_known = deep_merge(DEFAULT_CONFIG, DEFAULT_TOOLS)
    all_known = deep_merge(all_known, DEFAULT_MEMORY)
    all_known = deep_merge(all_known, DEFAULT_MCP)
    all_known = deep_merge(all_known, {"model": {"api_key": None}})
    model_schema = deep_merge({"model": DEFAULT_CONFIG["model"]}, {"model": {"api_key": None}})
    tools_schema = deep_merge({"tools": DEFAULT_CONFIG["tools"]}, DEFAULT_TOOLS)
    memory_schema = deep_merge({"memory": DEFAULT_CONFIG["memory"]}, DEFAULT_MEMORY)
    return {
        "config.yaml": all_known,
        "model.yaml": model_schema,
        "tools.yaml": tools_schema,
        "memory.yaml": memory_schema,
        "mcp.yaml": DEFAULT_MCP,
    }


def load_config() -> AppConfig:
    ensure_default_config()
    cfg = paths.config_dir()
    load_secrets_file(cfg / "secrets.env")
    raw_files = {
        filename: read_yaml(cfg / filename)
        for filename in ("config.yaml", "model.yaml", "tools.yaml", "memory.yaml", "mcp.yaml")
    }
    for filename, raw in raw_files.items():
        warn_unknown_config_keys(raw, _config_schemas()[filename], source=filename)
    values = dict(DEFAULT_CONFIG)
    primary = raw_files["config.yaml"]
    values = deep_merge(values, primary)
    model_overlay = remove_default_shadows(raw_files["model.yaml"], primary, DEFAULT_CONFIG)
    values = deep_merge(values, model_overlay)
    for filename in ("tools.yaml", "memory.yaml", "mcp.yaml"):
        values = deep_merge(values, raw_files[filename])
    return AppConfig(values=values, config_dir=cfg, data_dir=paths.data_dir())
