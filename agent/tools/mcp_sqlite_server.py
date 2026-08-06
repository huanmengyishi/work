from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any


READ_ACTIONS = {
    value
    for name in ("SQLITE_FUNCTION", "SQLITE_READ", "SQLITE_RECURSIVE", "SQLITE_SELECT")
    if isinstance((value := getattr(sqlite3, name, None)), int)
}
FORBIDDEN_DATABASE_ACTIONS = {
    value
    for name in ("SQLITE_ATTACH", "SQLITE_DETACH", "SQLITE_PRAGMA")
    if isinstance((value := getattr(sqlite3, name, None)), int)
}
MAX_REQUEST_BYTES = 1_048_576
MAX_SQL_CHARS = 100_000
MAX_SQL_BYTES = MAX_SQL_CHARS * 4
MAX_PARAMS = 1_000
MAX_PARAMS_BYTES = 1_048_576
MAX_CELL_CHARS = 100_000
MAX_RESULT_CHARS = 1_048_576
MAX_QUERY_SECONDS = 5.0
MAX_PROGRESS_STEPS = 5_000_000


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m agent.tools.mcp_sqlite_server DATABASE", file=sys.stderr)
        return 2
    source = Path(sys.argv[1]).expanduser()
    if source.is_symlink():
        print("database path must not be a symbolic link", file=sys.stderr)
        return 2
    database = source.resolve()
    stream = sys.stdin.buffer
    while True:
        raw = stream.readline(MAX_REQUEST_BYTES + 1)
        if not raw:
            break
        if len(raw) > MAX_REQUEST_BYTES:
            while raw and not raw.endswith(b"\n"):
                raw = stream.readline(MAX_REQUEST_BYTES + 1)
            response = error_response(None, -32600, "request exceeds the bounded input size")
        else:
            request: dict[str, Any] = {}
            try:
                request = json.loads(raw.decode("utf-8"))
                if not isinstance(request, dict):
                    response = error_response(None, -32600, "request must be a JSON object")
                else:
                    response = dispatch(database, request)
            except (UnicodeDecodeError, json.JSONDecodeError):
                response = error_response(None, -32700, "request contains invalid JSON")
            except Exception as exc:
                request_id = request.get("id") if isinstance(request, dict) else None
                response = error_response(request_id, -32603, f"request failed ({type(exc).__name__})")
        if response is not None:
            print(json.dumps(response, ensure_ascii=False, separators=(",", ":")), flush=True)
    return 0


def dispatch(database: Path, request: dict[str, Any]) -> dict[str, Any] | None:
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
            return result_response(request_id, query(database, arguments))
        if name == "sqlite_execute":
            return result_response(request_id, execute(database, arguments))
        return error_response(request_id, -32602, f"unknown tool: {name[:100]}")
    if request_id is None:
        return None
    return error_response(request_id, -32601, f"unknown method: {method[:100]}")


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


def query(database: Path, arguments: dict[str, Any]) -> dict[str, Any]:
    statement, params = validated_statement(arguments)
    limit = bounded_int(arguments.get("limit"), default=200, minimum=1, maximum=1_000)

    def authorizer(action, arg1, arg2, database_name, trigger_name):
        return sqlite3.SQLITE_OK if action in READ_ACTIONS else sqlite3.SQLITE_DENY

    with readonly_connection(database) as connection:
        connection.row_factory = sqlite3.Row
        connection.set_authorizer(authorizer)
        install_progress_limit(connection)
        try:
            cursor = connection.execute(statement, params)
            rows, truncated = bounded_rows(cursor, limit)
        finally:
            connection.set_authorizer(None)
            connection.set_progress_handler(None, 0)
    return tool_result(rows, {"rows": rows, "row_count": len(rows), "truncated": truncated})


def execute(database: Path, arguments: dict[str, Any]) -> dict[str, Any]:
    statement, params = validated_statement(arguments)

    def authorizer(action, arg1, arg2, database_name, trigger_name):
        if action in FORBIDDEN_DATABASE_ACTIONS:
            return sqlite3.SQLITE_DENY
        if action == getattr(sqlite3, "SQLITE_FUNCTION", -1) and str(arg2 or arg1 or "").casefold() in {
            "load_extension",
            "writefile",
        }:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    database.parent.mkdir(parents=True, exist_ok=True)
    if database.exists() and (database.is_symlink() or not database.is_file()):
        raise ValueError("database path must be a regular file")
    with sqlite3.connect(database) as connection:
        install_sqlite_limits(connection)
        connection.set_authorizer(authorizer)
        install_progress_limit(connection)
        try:
            cursor = connection.execute(statement, params)
            connection.commit()
        finally:
            connection.set_authorizer(None)
            connection.set_progress_handler(None, 0)
        data = {"row_count": cursor.rowcount, "last_row_id": cursor.lastrowid}
    return tool_result(data, data)


def readonly_connection(database: Path) -> sqlite3.Connection:
    if not database.exists():
        connection = sqlite3.connect(":memory:")
    else:
        if database.is_symlink() or not database.is_file():
            raise ValueError("database path must be a regular file")
        connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    install_sqlite_limits(connection)
    return connection


def install_sqlite_limits(connection: sqlite3.Connection) -> None:
    limits = {
        "SQLITE_LIMIT_ATTACHED": 0,
        "SQLITE_LIMIT_COLUMN": 1_000,
        "SQLITE_LIMIT_COMPOUND_SELECT": 100,
        "SQLITE_LIMIT_LENGTH": MAX_RESULT_CHARS,
        "SQLITE_LIMIT_LIKE_PATTERN_LENGTH": 10_000,
        "SQLITE_LIMIT_SQL_LENGTH": MAX_SQL_BYTES,
        "SQLITE_LIMIT_VARIABLE_NUMBER": MAX_PARAMS,
    }
    for name, limit in limits.items():
        category = getattr(sqlite3, name, None)
        if isinstance(category, int):
            connection.setlimit(category, limit)


def validated_statement(arguments: dict[str, Any]) -> tuple[str, list[Any]]:
    statement = arguments.get("query")
    if (
        not isinstance(statement, str)
        or not statement.strip()
        or len(statement) > MAX_SQL_CHARS
        or len(statement.encode("utf-8", errors="replace")) > MAX_SQL_BYTES
    ):
        raise ValueError("query must be a non-empty bounded string")
    params = arguments.get("params", [])
    if not isinstance(params, list) or len(params) > MAX_PARAMS:
        raise ValueError("params must be a bounded array")
    try:
        encoded = json.dumps(params, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("params must be JSON serializable") from exc
    if len(encoded) > MAX_PARAMS_BYTES:
        raise ValueError("params exceed the bounded input size")
    return statement.strip(), params


def bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError("integer value must not be boolean")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("integer value is invalid") from exc
    return max(minimum, min(parsed, maximum))


def install_progress_limit(connection: sqlite3.Connection) -> None:
    deadline = time.monotonic() + MAX_QUERY_SECONDS
    progress_steps = 0

    def progress() -> int:
        nonlocal progress_steps
        progress_steps += 1_000
        return int(progress_steps > MAX_PROGRESS_STEPS or time.monotonic() >= deadline)

    connection.set_progress_handler(progress, 1_000)


def bounded_rows(cursor: sqlite3.Cursor, limit: int) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    rendered_chars = 2
    truncated = False
    for row in cursor.fetchmany(limit + 1):
        if len(rows) >= limit:
            truncated = True
            break
        normalized = {str(key)[:200]: bounded_cell(value) for key, value in dict(row).items()}
        rendered = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
        if rendered_chars + len(rendered) > MAX_RESULT_CHARS:
            truncated = True
            break
        rendered_chars += len(rendered) + 1
        rows.append(normalized)
    return rows, truncated


def bounded_cell(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, bytes):
        return {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    text = str(value)
    return text if len(text) <= MAX_CELL_CHARS else text[:MAX_CELL_CHARS] + "...[truncated]"


def tool_result(value: Any, structured: dict[str, Any]) -> dict[str, Any]:
    result = {
        "content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False, indent=2)}],
        "structuredContent": structured,
        "isError": False,
    }
    if len(json.dumps(result, ensure_ascii=False, separators=(",", ":"))) > MAX_RESULT_CHARS * 2:
        raise ValueError("SQLite result exceeds the bounded output size")
    return result


def result_response(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


if __name__ == "__main__":
    raise SystemExit(main())
