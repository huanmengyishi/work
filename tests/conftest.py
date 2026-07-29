from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent.config import AppConfig, DEFAULT_CONFIG, deep_merge


@pytest.fixture
def make_config(tmp_path: Path):
    def factory(overrides: dict[str, Any] | None = None) -> AppConfig:
        isolated = {
            "memory": {
                "sqlite_path": str(tmp_path / "data" / "sqlite" / "memory.db"),
                "vector_path": str(tmp_path / "data" / "vector"),
                "retrieval_limit": 8,
                "vector_enabled": False,
                # Most tests exercise unrelated Runtime protocol behavior with
                # finite fake response queues. Smart reflection has dedicated
                # boundary tests and remains enabled in production defaults.
                "smart_reflection": False,
            },
            "events": {"jsonl_log": False},
        }
        values = deep_merge(DEFAULT_CONFIG, isolated)
        if overrides:
            values = deep_merge(values, overrides)
        return AppConfig(
            values=values,
            config_dir=tmp_path / "config",
            data_dir=tmp_path / "data",
        )

    return factory
