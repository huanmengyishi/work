from __future__ import annotations

import atexit
import fnmatch
import hashlib
import ipaddress
import json
import os
import queue
import re
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .. import __version__
from ..config import AppConfig
from .base import ToolResult, truncate_text
from .registry import ToolCapability


MCP_PROTOCOL_VERSION = "2025-03-26"
MAX_TOOL_LIST_PAGES = 100
MAX_REMOTE_TOOLS = 2_000
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_]+")
SAFE_INHERITED_ENV = {
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "TEMP",
    "TMP",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
}


class RejectRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def validate_http_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("MCP HTTP transport requires an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ValueError("credentials embedded in MCP URLs are not allowed")
    return parsed.geturl()


def open_url(request: urllib.request.Request, *, timeout: int):
    hostname = urllib.parse.urlparse(request.full_url).hostname or ""
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    proxy = (
        urllib.request.ProxyHandler({})
        if hostname == "localhost" or (address and address.is_loopback)
        else urllib.request.ProxyHandler()
    )
    return urllib.request.build_opener(proxy, RejectRedirect()).open(request, timeout=timeout)


def parse_sse_message(value: str) -> dict[str, Any]:
    data_lines = [line[5:].lstrip() for line in value.splitlines() if line.startswith("data:")]
    payload = "\n".join(data_lines).strip()
    if not payload:
        raise RuntimeError("MCP SSE response did not contain a data event")
    message = json.loads(payload)
    if not isinstance(message, dict):
        raise RuntimeError("MCP response must be a JSON object")
    return message


def rpc_result(server_name: str, message: dict[str, Any]) -> dict[str, Any]:
    error = message.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        detail = str(error.get("message") or "MCP request failed")
        raise RuntimeError(f"MCP {server_name} JSON-RPC {code}: {detail}")
    result = message.get("result")
    return result if isinstance(result, dict) else {}


def list_remote_tools(client: Any) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()
    for _page in range(MAX_TOOL_LIST_PAGES):
        params = {"cursor": cursor} if cursor else {}
        result = client.request("tools/list", params, timeout=client.startup_timeout)
        values = result.get("tools") if isinstance(result, dict) else None
        if not isinstance(values, list):
            raise RuntimeError(f"MCP server {client.name} returned an invalid tools/list result")
        tools.extend(item for item in values if isinstance(item, dict))
        if len(tools) > MAX_REMOTE_TOOLS:
            raise RuntimeError(f"MCP server {client.name} returned more than {MAX_REMOTE_TOOLS} tools")
        cursor = str(result.get("nextCursor") or "")
        if not cursor:
            return tools
        if cursor in seen_cursors:
            raise RuntimeError(f"MCP server {client.name} repeated tools/list cursor")
        seen_cursors.add(cursor)
    raise RuntimeError(f"MCP server {client.name} exceeded {MAX_TOOL_LIST_PAGES} tools/list pages")


def tool_call_result(server_name: str, tool_name: str, result: dict[str, Any]) -> ToolResult:
    text_parts: list[str] = []
    content_meta: list[dict[str, Any]] = []
    content = result.get("content")
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "unknown")
            if item_type == "text":
                text_parts.append(str(item.get("text") or ""))
            else:
                content_meta.append({"type": item_type, "mimeType": item.get("mimeType")})
    structured = result.get("structuredContent")
    if structured is not None:
        text_parts.append(json.dumps(structured, ensure_ascii=False, indent=2))
    output = truncate_text("\n".join(part for part in text_parts if part).strip())
    is_error = bool(result.get("isError", False))
    data = {"server": server_name, "tool": tool_name, "content": content_meta}
    if structured is not None:
        data["structuredContent"] = structured
    return ToolResult(not is_error, output if not is_error else "", output if is_error else "", data=data)


def resource_result(server_name: str, uri: str, result: dict[str, Any]) -> ToolResult:
    contents = result.get("contents")
    if not isinstance(contents, list):
        return ToolResult(False, "", f"MCP {server_name} returned an invalid resources/read result")
    text_parts: list[str] = []
    metadata: list[dict[str, Any]] = []
    for item in contents:
        if not isinstance(item, dict):
            continue
        if "text" in item:
            text_parts.append(str(item.get("text") or ""))
        metadata.append(
            {
                "uri": item.get("uri"),
                "mimeType": item.get("mimeType"),
                "has_blob": "blob" in item,
            }
        )
    output = truncate_text("\n\n".join(part for part in text_parts if part).strip())
    return ToolResult(True, output, data={"server": server_name, "uri": uri, "contents": metadata})


@dataclass(frozen=True)
class MCPServerStatus:
    name: str
    enabled: bool
    connected: bool
    tool_count: int = 0
    error: str = ""


class MCPClient:
    def __init__(
        self,
        *,
        name: str,
        command: str,
        args: list[str],
        cwd: Path,
        env: dict[str, str],
        startup_timeout: int,
        call_timeout: int,
        protocol_version: str = MCP_PROTOCOL_VERSION,
    ) -> None:
        self.name = name
        self.command = command
        self.args = args
        self.cwd = cwd
        self.env = env
        self.startup_timeout = startup_timeout
        self.call_timeout = call_timeout
        self.protocol_version = protocol_version
        self.process: subprocess.Popen[str] | None = None
        self._messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self._pending: dict[int, dict[str, Any]] = {}
        self._stderr: list[str] = []
        self._next_id = 1
        self._request_lock = threading.Lock()

    def start(self) -> None:
        if self.process and self.process.poll() is None:
            return
        try:
            self.process = subprocess.Popen(
                [self.command, *self.args],
                cwd=self.cwd,
                env=self.env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            raise RuntimeError(f"could not start MCP server {self.name}: {exc}") from exc
        threading.Thread(target=self._read_stdout, daemon=True, name=f"mcp-{self.name}-stdout").start()
        threading.Thread(target=self._read_stderr, daemon=True, name=f"mcp-{self.name}-stderr").start()
        try:
            response = self.request(
                "initialize",
                {
                    "protocolVersion": self.protocol_version,
                    "capabilities": {},
                    "clientInfo": {"name": "deep-agent", "version": __version__},
                },
                timeout=self.startup_timeout,
            )
            if not isinstance(response, dict) or not response.get("protocolVersion"):
                raise RuntimeError("initialize response did not include protocolVersion")
            self.notify("notifications/initialized", {})
        except Exception:
            self.close()
            raise

    def list_tools(self) -> list[dict[str, Any]]:
        return list_remote_tools(self)

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        try:
            result = self.request(
                "tools/call",
                {"name": tool_name, "arguments": arguments},
                timeout=self.call_timeout,
            )
        except Exception as exc:
            return ToolResult(False, "", f"MCP {self.name}.{tool_name} failed: {exc}")
        if not isinstance(result, dict):
            return ToolResult(False, "", f"MCP {self.name}.{tool_name} returned an invalid result")
        text_parts: list[str] = []
        content_meta: list[dict[str, Any]] = []
        content = result.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get("type") or "unknown")
                if item_type == "text":
                    text_parts.append(str(item.get("text") or ""))
                else:
                    content_meta.append({"type": item_type, "mimeType": item.get("mimeType")})
        structured = result.get("structuredContent")
        if structured is not None:
            text_parts.append(json.dumps(structured, ensure_ascii=False, indent=2))
        output = truncate_text("\n".join(part for part in text_parts if part).strip())
        is_error = bool(result.get("isError", False))
        data = {"server": self.name, "tool": tool_name, "content": content_meta}
        if structured is not None:
            data["structuredContent"] = structured
        return ToolResult(not is_error, output if not is_error else "", output if is_error else "", data=data)

    def read_resource(self, uri: str) -> ToolResult:
        try:
            result = self.request("resources/read", {"uri": uri}, timeout=self.call_timeout)
        except Exception as exc:
            return ToolResult(False, "", f"MCP {self.name} resource read failed: {exc}")
        return resource_result(self.name, uri, result)

    def request(self, method: str, params: dict[str, Any], *, timeout: int) -> dict[str, Any]:
        with self._request_lock:
            request_id = self._next_id
            self._next_id += 1
            self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
            message = self._wait_for_response(request_id, timeout)
        error = message.get("error")
        if isinstance(error, dict):
            code = error.get("code")
            detail = str(error.get("message") or "MCP request failed")
            raise RuntimeError(f"JSON-RPC {code}: {detail}")
        result = message.get("result")
        return result if isinstance(result, dict) else {}

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def close(self) -> None:
        process = self.process
        self.process = None
        if not process:
            return
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                if stream:
                    stream.close()
            except OSError:
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

    def _send(self, message: dict[str, Any]) -> None:
        process = self.process
        if not process or process.poll() is not None or not process.stdin:
            raise RuntimeError(f"MCP server {self.name} is not running; inspect local logs for server diagnostics")
        payload = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
        try:
            process.stdin.write(payload + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(f"MCP server {self.name} closed its input") from exc

    def _wait_for_response(self, request_id: int, timeout: int) -> dict[str, Any]:
        if request_id in self._pending:
            return self._pending.pop(request_id)
        try:
            while True:
                message = self._messages.get(timeout=timeout)
                message_id = message.get("id")
                if message_id == request_id:
                    return message
                if isinstance(message_id, int):
                    self._pending[message_id] = message
        except queue.Empty as exc:
            raise TimeoutError(f"MCP request timed out after {timeout}s") from exc

    def _read_stdout(self) -> None:
        process = self.process
        if not process or not process.stdout:
            return
        for line in process.stdout:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                self._messages.put(value)

    def _read_stderr(self) -> None:
        process = self.process
        if not process or not process.stderr:
            return
        for line in process.stderr:
            text = line.strip()
            if text:
                self._stderr.append(text[:1000])
                del self._stderr[:-20]


class MCPHttpClient:
    def __init__(
        self,
        *,
        name: str,
        url: str,
        headers: dict[str, str],
        startup_timeout: int,
        call_timeout: int,
        protocol_version: str = MCP_PROTOCOL_VERSION,
    ) -> None:
        self.name = name
        self.url = validate_http_url(url)
        self.headers = headers
        self.startup_timeout = startup_timeout
        self.call_timeout = call_timeout
        self.protocol_version = protocol_version
        self.session_id: str | None = None
        self._next_id = 1
        self._request_lock = threading.Lock()

    def start(self) -> None:
        response = self.request(
            "initialize",
            {
                "protocolVersion": self.protocol_version,
                "capabilities": {},
                "clientInfo": {"name": "deep-agent", "version": __version__},
            },
            timeout=self.startup_timeout,
        )
        if not response.get("protocolVersion"):
            raise RuntimeError("initialize response did not include protocolVersion")
        self.notify("notifications/initialized", {})

    def list_tools(self) -> list[dict[str, Any]]:
        return list_remote_tools(self)

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        try:
            result = self.request("tools/call", {"name": tool_name, "arguments": arguments}, timeout=self.call_timeout)
        except Exception as exc:
            return ToolResult(False, "", f"MCP {self.name}.{tool_name} failed: {exc}")
        return tool_call_result(self.name, tool_name, result)

    def read_resource(self, uri: str) -> ToolResult:
        try:
            result = self.request("resources/read", {"uri": uri}, timeout=self.call_timeout)
        except Exception as exc:
            return ToolResult(False, "", f"MCP {self.name} resource read failed: {exc}")
        return resource_result(self.name, uri, result)

    def request(self, method: str, params: dict[str, Any], *, timeout: int) -> dict[str, Any]:
        with self._request_lock:
            request_id = self._next_id
            self._next_id += 1
            message = self._post(
                {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
                timeout=timeout,
            )
        return rpc_result(self.name, message)

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self._post({"jsonrpc": "2.0", "method": method, "params": params}, timeout=self.startup_timeout)

    def close(self) -> None:
        if not self.session_id:
            return
        headers = self._request_headers()
        request = urllib.request.Request(self.url, headers=headers, method="DELETE")
        try:
            open_url(request, timeout=2).close()
        except Exception:
            pass
        self.session_id = None

    def _post(self, payload: dict[str, Any], *, timeout: int) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        headers = self._request_headers()
        headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        try:
            with open_url(request, timeout=timeout) as response:
                session_id = response.headers.get("Mcp-Session-Id")
                if session_id:
                    self.session_id = session_id
                raw = response.read(4_194_305)
                if len(raw) > 4_194_304:
                    raise RuntimeError("MCP HTTP response exceeds 4 MiB")
                if not raw.strip():
                    return {}
                content_type = response.headers.get_content_type()
                return (
                    parse_sse_message(raw.decode("utf-8", errors="replace"))
                    if content_type == "text/event-stream"
                    else json.loads(raw)
                )
        except urllib.error.HTTPError as exc:
            detail = exc.read(8192).decode("utf-8", errors="replace")
            raise RuntimeError(f"MCP HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"MCP HTTP request failed: {exc}") from exc

    def _request_headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json, text/event-stream", **self.headers}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        return headers


class MCPSseClient(MCPHttpClient):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.endpoint: str | None = None
        self._messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stream = None
        self._reader: threading.Thread | None = None

    def start(self) -> None:
        request = urllib.request.Request(
            self.url, headers={"Accept": "text/event-stream", **self.headers}, method="GET"
        )
        try:
            self._stream = open_url(request, timeout=self.startup_timeout)
            self._disable_stream_timeout()
        except urllib.error.URLError as exc:
            raise RuntimeError(f"could not open MCP SSE stream: {exc}") from exc
        self._reader = threading.Thread(target=self._read_stream, daemon=True, name=f"mcp-{self.name}-sse")
        self._reader.start()
        deadline = threading.Event()
        for _ in range(max(1, self.startup_timeout * 20)):
            if self.endpoint:
                break
            if self._reader and not self._reader.is_alive():
                break
            deadline.wait(0.05)
        if not self.endpoint:
            self.close()
            raise RuntimeError("MCP SSE server did not provide an endpoint event")
        response = self.request(
            "initialize",
            {
                "protocolVersion": self.protocol_version,
                "capabilities": {},
                "clientInfo": {"name": "deep-agent", "version": __version__},
            },
            timeout=self.startup_timeout,
        )
        if not response.get("protocolVersion"):
            raise RuntimeError("initialize response did not include protocolVersion")
        self.notify("notifications/initialized", {})

    def request(self, method: str, params: dict[str, Any], *, timeout: int) -> dict[str, Any]:
        with self._request_lock:
            request_id = self._next_id
            self._next_id += 1
            self._send_sse({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}, timeout)
            message = self._wait_sse(request_id, timeout)
        return rpc_result(self.name, message)

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self._send_sse({"jsonrpc": "2.0", "method": method, "params": params}, self.startup_timeout)

    def close(self) -> None:
        stream = self._stream
        self._stream = None
        if stream:
            try:
                stream.close()
            except OSError:
                pass
        if self._reader and self._reader is not threading.current_thread():
            self._reader.join(timeout=1)
        self._reader = None

    def _send_sse(self, payload: dict[str, Any], timeout: int) -> None:
        if not self.endpoint:
            raise RuntimeError("MCP SSE endpoint is unavailable")
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json", **self.headers}
        request = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
        try:
            with open_url(request, timeout=timeout):
                pass
        except urllib.error.URLError as exc:
            raise RuntimeError(f"MCP SSE send failed: {exc}") from exc

    def _disable_stream_timeout(self) -> None:
        stream = self._stream
        try:
            stream.fp.raw._sock.settimeout(None)
        except (AttributeError, OSError):
            pass

    def _wait_sse(self, request_id: int, timeout: int) -> dict[str, Any]:
        try:
            while True:
                message = self._messages.get(timeout=timeout)
                if message.get("id") == request_id:
                    return message
        except queue.Empty as exc:
            raise TimeoutError(f"MCP SSE request timed out after {timeout}s") from exc

    def _read_stream(self) -> None:
        stream = self._stream
        if not stream:
            return
        event = "message"
        data_lines: list[str] = []
        try:
            for raw in stream:
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line:
                    data = "\n".join(data_lines)
                    if event == "endpoint" and data:
                        self.endpoint = urllib.parse.urljoin(self.url, data.strip())
                    elif data:
                        try:
                            value = json.loads(data)
                        except json.JSONDecodeError:
                            value = None
                        if isinstance(value, dict):
                            self._messages.put(value)
                    event, data_lines = "message", []
                elif line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
        except (OSError, ValueError, AttributeError):
            return


class MCPManager:
    def __init__(self, config: AppConfig, project_root: Path) -> None:
        self.config = config
        self.project_root = project_root
        self.clients: dict[str, Any] = {}
        self.statuses: list[MCPServerStatus] = []
        self._closed = False
        atexit.register(self.close)

    def discover(self) -> list[tuple[ToolCapability, Callable[..., ToolResult]]]:
        registrations: list[tuple[ToolCapability, Callable[..., ToolResult]]] = []
        servers = self.config.get("mcp.servers", [])
        if not bool(self.config.get("mcp.enabled", False)):
            for spec in servers if isinstance(servers, list) else []:
                if isinstance(spec, dict):
                    self.statuses.append(MCPServerStatus(str(spec.get("name") or "unnamed"), False, False))
            return registrations
        if not isinstance(servers, list):
            self.statuses.append(MCPServerStatus("configuration", True, False, error="mcp.servers must be a list"))
            return registrations

        max_servers = max(0, int(self.config.get("mcp.max_servers", 10)))
        max_tools = max(0, int(self.config.get("mcp.max_tools", 80)))
        enabled_specs = [spec for spec in servers if isinstance(spec, dict) and bool(spec.get("enabled", True))]
        if len(enabled_specs) > max_servers:
            self.statuses.append(
                MCPServerStatus(
                    "policy",
                    True,
                    False,
                    error=f"enabled MCP server count exceeds limit: {len(enabled_specs)} > {max_servers}",
                )
            )
            return registrations

        for spec in servers:
            if not isinstance(spec, dict):
                continue
            name = str(spec.get("name") or "").strip()
            enabled = bool(spec.get("enabled", True))
            if not name:
                self.statuses.append(MCPServerStatus("unnamed", enabled, False, error="server name is required"))
                continue
            if not enabled:
                self.statuses.append(MCPServerStatus(name, False, False))
                continue
            transport = str(spec.get("transport") or "stdio")
            if transport not in {"stdio", "streamable_http", "sse"}:
                self.statuses.append(
                    MCPServerStatus(name, True, False, error=f"unsupported MCP transport: {transport}")
                )
                continue
            try:
                client = self._build_client(name, spec)
                client.start()
                tools = client.list_tools() if bool(spec.get("tools_enabled", True)) else []
                selected = [tool for tool in tools if self._allowed(str(tool.get("name") or ""), spec)]
                resource_count = 1 if bool(spec.get("resources_enabled", False)) else 0
                if len(registrations) + len(selected) + resource_count > max_tools:
                    client.close()
                    self.statuses.append(
                        MCPServerStatus(
                            name,
                            True,
                            False,
                            error=f"MCP tool count would exceed limit: {max_tools}",
                        )
                    )
                    continue
                self.clients[name] = client
                registrations.extend(self._registrations(name, client, selected, spec))
                if bool(spec.get("resources_enabled", False)):
                    registrations.append(self._resource_registration(name, client, spec))
                self.statuses.append(MCPServerStatus(name, True, True, tool_count=len(selected) + resource_count))
            except Exception as exc:
                self.statuses.append(MCPServerStatus(name, True, False, error=str(exc)))
        return registrations

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for client in self.clients.values():
            client.close()
        self.clients.clear()

    def summary(self) -> str:
        if not self.statuses:
            return "disabled" if not bool(self.config.get("mcp.enabled", False)) else "no servers configured"
        connected = sum(1 for item in self.statuses if item.connected)
        enabled = sum(1 for item in self.statuses if item.enabled)
        tools = sum(item.tool_count for item in self.statuses)
        return f"{connected}/{enabled} servers connected ({tools} tools)"

    def _build_client(self, name: str, spec: dict[str, Any]) -> Any:
        transport = str(spec.get("transport") or "stdio")
        startup_timeout = int(spec.get("startup_timeout_seconds") or self.config.get("mcp.startup_timeout_seconds", 15))
        call_timeout = int(spec.get("call_timeout_seconds") or self.config.get("mcp.call_timeout_seconds", 120))
        protocol_version = str(spec.get("protocol_version") or MCP_PROTOCOL_VERSION)
        if transport in {"streamable_http", "sse"}:
            url = str(spec.get("url") or "").strip()
            if not url:
                raise ValueError(f"{transport} MCP server URL is required")
            headers_value = spec.get("headers", {})
            if not isinstance(headers_value, dict):
                raise ValueError("MCP HTTP headers must be a mapping")
            client_type = MCPHttpClient if transport == "streamable_http" else MCPSseClient
            return client_type(
                name=name,
                url=url,
                headers={str(key): str(value) for key, value in headers_value.items()},
                startup_timeout=startup_timeout,
                call_timeout=call_timeout,
                protocol_version=protocol_version,
            )
        command = str(spec.get("command") or "").strip()
        if not command:
            raise ValueError("stdio MCP server command is required")
        args_value = spec.get("args", [])
        if not isinstance(args_value, list):
            raise ValueError("MCP server args must be a list")
        env_value = spec.get("env", {})
        if not isinstance(env_value, dict):
            raise ValueError("MCP server env must be a mapping")
        passthrough_value = spec.get("env_passthrough", [])
        if not isinstance(passthrough_value, list):
            raise ValueError("MCP server env_passthrough must be a list")
        cwd_value = str(spec.get("cwd") or self.project_root)
        cwd = Path(cwd_value).expanduser()
        if not cwd.is_absolute():
            cwd = self.project_root / cwd
        return MCPClient(
            name=name,
            command=command,
            args=[str(item) for item in args_value],
            cwd=cwd.resolve(),
            env=self._safe_environment(env_value, passthrough_value),
            startup_timeout=startup_timeout,
            call_timeout=call_timeout,
            protocol_version=protocol_version,
        )

    def _registrations(
        self,
        server_name: str,
        client: Any,
        tools: list[dict[str, Any]],
        spec: dict[str, Any],
    ) -> list[tuple[ToolCapability, Callable[..., ToolResult]]]:
        result: list[tuple[ToolCapability, Callable[..., ToolResult]]] = []
        overrides = spec.get("tool_overrides", {})
        if not isinstance(overrides, dict):
            overrides = {}
        for remote in tools:
            tool_name = str(remote.get("name") or "").strip()
            if not tool_name:
                continue
            input_schema = remote.get("inputSchema")
            if not isinstance(input_schema, dict):
                input_schema = {}
            properties = input_schema.get("properties")
            required = input_schema.get("required")
            annotations = remote.get("annotations")
            annotations = annotations if isinstance(annotations, dict) else {}
            override = overrides.get(tool_name)
            override = override if isinstance(override, dict) else {}
            read_only = bool(annotations.get("readOnlyHint", False))
            default_permissions = ["external", "network", "read" if read_only else "write"]
            permissions = override.get("permissions", default_permissions)
            if not isinstance(permissions, list):
                permissions = default_permissions
            requires_confirmation = bool(
                override.get(
                    "requires_confirmation",
                    annotations.get("destructiveHint", False) or not read_only,
                )
            )
            capability = ToolCapability(
                "mcp",
                f"{server_name}.{tool_name}",
                self._model_name(server_name, tool_name),
                f"MCP {server_name}: {str(remote.get('description') or tool_name)}",
                properties if isinstance(properties, dict) else {},
                tuple(str(item) for item in required) if isinstance(required, list) else (),
                tuple(str(item) for item in permissions),
                client.call_timeout,
                requires_confirmation=requires_confirmation,
            )

            def handler(_client=client, _tool_name=tool_name, **kwargs):
                return _client.call_tool(_tool_name, kwargs)

            result.append((capability, handler))
        return result

    def _resource_registration(
        self,
        server_name: str,
        client: Any,
        spec: dict[str, Any],
    ) -> tuple[ToolCapability, Callable[..., ToolResult]]:
        timeout = int(spec.get("resource_timeout_seconds") or self.config.get("mcp.resource_timeout_seconds", 60))
        capability = ToolCapability(
            "mcp",
            f"{server_name}.resources.read",
            self._model_name(server_name, "resource_read"),
            f"Read one URI from MCP server {server_name} through resources/read.",
            {"uri": {"type": "string"}},
            ("uri",),
            ("external", "network", "read"),
            timeout,
        )

        def handler(uri: str, _client=client, _spec=spec):
            allowlist = _spec.get("resource_uri_allowlist", ["*"])
            if not isinstance(allowlist, list) or not any(
                fnmatch.fnmatchcase(uri, str(pattern)) for pattern in allowlist
            ):
                return ToolResult(False, "", f"MCP resource URI is not allowlisted: {uri}")
            return _client.read_resource(uri)

        return capability, handler

    @staticmethod
    def _allowed(tool_name: str, spec: dict[str, Any]) -> bool:
        allowlist = spec.get("tool_allowlist", ["*"])
        if not isinstance(allowlist, list) or not allowlist:
            return False
        return any(fnmatch.fnmatchcase(tool_name, str(pattern)) for pattern in allowlist)

    @staticmethod
    def _model_name(server_name: str, tool_name: str) -> str:
        base = SAFE_NAME_RE.sub("_", f"mcp_{server_name}_{tool_name}").strip("_") or "mcp_tool"
        if len(base) <= 64:
            return base
        digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:8]
        return f"{base[:55]}_{digest}"

    @staticmethod
    def _safe_environment(explicit: dict[str, Any], passthrough: list[Any]) -> dict[str, str]:
        inherited = {
            key: value for key, value in os.environ.items() if key in SAFE_INHERITED_ENV or key.startswith("LC_")
        }
        for item in passthrough:
            name = str(item)
            if name in os.environ:
                inherited[name] = os.environ[name]
        inherited.update({str(key): str(value) for key, value in explicit.items()})
        return inherited
