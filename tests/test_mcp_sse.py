from __future__ import annotations

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agent.tools.mcp import MCPSseClient


responses: queue.Queue[dict] = queue.Queue()


class SSEHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self.path != "/events":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(b"event: endpoint\ndata: /messages\n\n")
        self.wfile.flush()
        while True:
            try:
                message = responses.get(timeout=3)
            except queue.Empty:
                return
            data = json.dumps(message).encode()
            try:
                self.wfile.write(b"event: message\ndata: " + data + b"\n\n")
                self.wfile.flush()
            except OSError:
                return

    def do_POST(self) -> None:
        size = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(size))
        if "id" in request:
            method = request.get("method")
            if method == "initialize":
                result = {"protocolVersion": "2025-03-26", "capabilities": {}}
            elif method == "tools/list":
                result = {"tools": []}
            elif method == "resources/read":
                result = {"contents": [{"uri": request["params"]["uri"], "text": "legacy resource"}]}
            else:
                result = {}
            responses.put({"jsonrpc": "2.0", "id": request["id"], "result": result})
        self.send_response(202)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args) -> None:
        pass


def test_legacy_sse_initialize_and_resource_read() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), SSEHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    client = MCPSseClient(
        name="legacy",
        url=f"http://127.0.0.1:{server.server_port}/events",
        headers={},
        startup_timeout=3,
        call_timeout=3,
    )
    try:
        client.start()
        assert client.list_tools() == []
        result = client.read_resource("file://legacy/readme")
        assert result.success is True
        assert result.stdout == "legacy resource"
    finally:
        client.close()
        server.shutdown()
        server.server_close()
