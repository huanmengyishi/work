#!/usr/bin/env python3
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any


WRITE_ACTIONS = {
    sqlite3.SQLITE_INSERT,
    sqlite3.SQLITE_UPDATE,
    sqlite3.SQLITE_DELETE,
    sqlite3.SQLITE_CREATE_INDEX,
    sqlite3.SQLITE_CREATE_TABLE,
    sqlite3.SQLITE_CREATE_TEMP_INDEX,
    sqlite3.SQLITE_CREATE_TEMP_TABLE,
    sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
    sqlite3.SQLITE_CREATE_TEMP_VIEW,
    sqlite3.SQLITE_CREATE_TRIGGER,
    sqlite3.SQLITE_CREATE_VIEW,
    sqlite3.SQLITE_DROP_INDEX,
    sqlite3.SQLITE_DROP_TABLE,
    sqlite3.SQLITE_DROP_TEMP_INDEX,
    sqlite3.SQLITE_DROP_TEMP_TABLE,
    sqlite3.SQLITE_DROP_TEMP_TRIGGER,
    sqlite3.SQLITE_DROP_TEMP_VIEW,
    sqlite3.SQLITE_DROP_TRIGGER,
    sqlite3.SQLITE_DROP_VIEW,
    sqlite3.SQLITE_ALTER_TABLE,
    sqlite3.SQLITE_REINDEX,
}


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: mcp_sqlite_server.py DATABASE", file=sys.stderr)
        return 2
    database = Path(sys.argv[1]).expanduser().resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        for line in sys.stdin:
            request: dict[str, Any] = {}
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    continue
                response = dispatch(connection, request)
            except Exception as exc:
                request_id = request.get("id") if isinstance(request, dict) else None
                response = error_response(request_id, -32603, str(exc))
            if response is not None:
                print(json.dumps(response, ensure_ascii=False, separators=(",", ":")), flush=True)
    return 0


def dispatch(connection: sqlite3.Connection, request: dict[str, Any]) -> dict[str, Any] | None:
    request_id = request.get("id")
    method = str(request.get("method") or "")
    params = request.get("params")
    params = params if isinstance(params, dict) else {}
    if method == "initialize":
        return result_response(
            request_id,
            {
                "protocolVersion": str(params.get("protocolVersion") or "2025-03-26"),
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "deep-agent-sqlite-example", "version": "1.0.0"},
            },
        )
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return result_response(request_id, {"tools": tool_definitions()})
    if method == "tools/call":
        name = str(params.get("name") or "")
        arguments = params.get("arguments")
        arguments = arguments if isinstance(arguments, dict) else {}
        if name == "sqlite_query":
            return result_response(request_id, query(connection, arguments))
        if name == "sqlite_execute":
            return result_response(request_id, execute(connection, arguments))
        return error_response(request_id, -32602, f"unknown tool: {name}")
    if request_id is None:
        return None
    return error_response(request_id, -32601, f"unknown method: {method}")


def tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "name": "sqlite_query",
            "description": "Run a read-only SQLite query and return JSON rows.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "params": {"type": "array"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
                },
                "required": ["query"],
            },
            "annotations": {"readOnlyHint": True, "destructiveHint": False},
        },
        {
            "name": "sqlite_execute",
            "description": "Execute a SQLite write statement and commit it.",
            "inputSchema": {
                "type": "object",
                "properties": {"query": {"type": "string"}, "params": {"type": "array"}},
                "required": ["query"],
            },
            "annotations": {"readOnlyHint": False, "destructiveHint": True},
        },
    ]


def query(connection: sqlite3.Connection, arguments: dict[str, Any]) -> dict[str, Any]:
    statement = str(arguments.get("query") or "")
    params = arguments.get("params")
    params = params if isinstance(params, list) else []
    limit = max(1, min(int(arguments.get("limit") or 200), 1000))

    def authorizer(action, arg1, arg2, database_name, trigger_name):
        return sqlite3.SQLITE_DENY if action in WRITE_ACTIONS else sqlite3.SQLITE_OK

    connection.set_authorizer(authorizer)
    try:
        cursor = connection.execute(statement, params)
        rows = [dict(row) for row in cursor.fetchmany(limit)]
    finally:
        connection.set_authorizer(None)
    return tool_result(rows, {"rows": rows, "row_count": len(rows)})


def execute(connection: sqlite3.Connection, arguments: dict[str, Any]) -> dict[str, Any]:
    statement = str(arguments.get("query") or "")
    params = arguments.get("params")
    params = params if isinstance(params, list) else []
    cursor = connection.execute(statement, params)
    connection.commit()
    data = {"row_count": cursor.rowcount, "last_row_id": cursor.lastrowid}
    return tool_result(data, data)


def tool_result(value: Any, structured: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False, indent=2)}],
        "structuredContent": structured,
        "isError": False,
    }


def result_response(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


if __name__ == "__main__":
    raise SystemExit(main())
