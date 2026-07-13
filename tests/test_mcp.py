from __future__ import annotations

import sys
from pathlib import Path

from agent.memory import MemoryStore
from agent.project import ProjectManager
from agent.tools import ToolManager
from agent.tools.mcp import MCPManager


def mcp_overrides(database: Path) -> dict:
    script = Path(__file__).resolve().parents[1] / "scripts" / "mcp_sqlite_server.py"
    return {
        "mcp": {
            "enabled": True,
            "startup_timeout_seconds": 5,
            "call_timeout_seconds": 5,
            "servers": [
                {
                    "name": "sqlite",
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": [str(script), str(database)],
                    "tool_allowlist": ["sqlite_*"],
                }
            ],
        }
    }


def build_manager(root: Path, database: Path, make_config, **kwargs) -> ToolManager:
    config = make_config(mcp_overrides(database))
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    memory.sync_project(project)
    return ToolManager(config, project, memory, **kwargs)


def test_mcp_stdio_discovery_call_and_confirmation(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    database = tmp_path / "example.db"
    manager = build_manager(root, database, make_config)
    try:
        names = {item.model_name for item in manager.capabilities(enabled_only=True)}
        assert "mcp_sqlite_sqlite_query" in names
        assert "mcp_sqlite_sqlite_execute" in names
        assert manager.mcp.summary() == "1/1 servers connected (2 tools)"

        _, queried = manager.execute_model_call("mcp_sqlite_sqlite_query", {"query": "select 42 as answer"})
        _, denied = manager.execute_model_call(
            "mcp_sqlite_sqlite_execute",
            {"query": "create table items (id integer primary key, name text)"},
        )
        assert queried.success is True
        assert '"answer": 42' in queried.stdout
        assert denied.success is False
        assert "requires user confirmation" in denied.stderr
    finally:
        manager.close()

    approved = build_manager(root, database, make_config, yolo=True)
    try:
        _, created = approved.execute_model_call(
            "mcp_sqlite_sqlite_execute",
            {"query": "create table items (id integer primary key, name text)"},
        )
        _, inserted = approved.execute_model_call(
            "mcp_sqlite_sqlite_execute",
            {"query": "insert into items(name) values (?)", "params": ["demo"]},
        )
        _, rows = approved.execute_model_call("mcp_sqlite_sqlite_query", {"query": "select * from items"})
        assert created.success is True
        assert inserted.success is True
        assert rows.success is True
        assert "demo" in rows.stdout
    finally:
        approved.close()


def test_mcp_allowlist_filters_remote_tools(tmp_path: Path, make_config) -> None:
    root = tmp_path / "project"
    root.mkdir()
    overrides = mcp_overrides(tmp_path / "example.db")
    overrides["mcp"]["servers"][0]["tool_allowlist"] = ["sqlite_query"]
    config = make_config(overrides)
    project = ProjectManager(config).resolve_project(root)
    memory = MemoryStore(config)
    memory.sync_project(project)
    manager = ToolManager(config, project, memory)
    try:
        names = {item.model_name for item in manager.capabilities(enabled_only=True)}
        assert "mcp_sqlite_sqlite_query" in names
        assert "mcp_sqlite_sqlite_execute" not in names
    finally:
        manager.close()


def test_mcp_environment_does_not_inherit_deepseek_key(monkeypatch, make_config, tmp_path: Path) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "must-not-leak")
    config = make_config()
    manager = MCPManager(config, tmp_path)

    safe = manager._safe_environment({}, [])
    explicit = manager._safe_environment({}, ["DEEPSEEK_API_KEY"])

    assert "DEEPSEEK_API_KEY" not in safe
    assert explicit["DEEPSEEK_API_KEY"] == "must-not-leak"
