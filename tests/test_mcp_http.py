from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agent.tools.mcp import MCPManager


class MCPHandler(BaseHTTPRequestHandler):
    session_id = "test-session"

    def do_POST(self) -> None:
        size = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(size))
        method = request.get("method")
        if "id" not in request:
            self.send_response(202)
            self.end_headers()
            return
        if method == "initialize":
            result = {"protocolVersion": "2025-03-26", "capabilities": {}}
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "echo",
                        "description": "Echo text",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                        "annotations": {"readOnlyHint": True},
                    }
                ]
            }
        elif method == "tools/call":
            result = {"content": [{"type": "text", "text": request["params"]["arguments"]["text"]}]}
        elif method == "resources/read":
            result = {
                "contents": [{"uri": request["params"]["uri"], "mimeType": "text/plain", "text": "resource body"}]
            }
        else:
            result = {}
        body = json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Mcp-Session-Id", self.session_id)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_DELETE(self) -> None:
        self.send_response(204)
        self.end_headers()

    def log_message(self, format: str, *args) -> None:
        pass


def test_mcp_streamable_http_tools_and_resources(tmp_path: Path, make_config) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), MCPHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    config = make_config(
        {
            "mcp": {
                "enabled": True,
                "servers": [
                    {
                        "name": "remote",
                        "enabled": True,
                        "transport": "streamable_http",
                        "url": f"http://127.0.0.1:{server.server_port}/mcp",
                        "tool_allowlist": ["echo"],
                        "resources_enabled": True,
                        "resource_uri_allowlist": ["file://allowed/*"],
                    }
                ],
            }
        }
    )
    manager = MCPManager(config, tmp_path)
    try:
        registrations = manager.discover()
        by_name = {capability.name: handler for capability, handler in registrations}
        assert by_name["mcp.remote.echo"](text="hello").stdout == "hello"
        assert by_name["mcp.remote.resources.read"](uri="file://allowed/readme").stdout == "resource body"
        denied = by_name["mcp.remote.resources.read"](uri="file://denied/secret")
        assert denied.success is False
        assert manager.clients["remote"].session_id == MCPHandler.session_id
    finally:
        manager.close()
        server.shutdown()
        server.server_close()
