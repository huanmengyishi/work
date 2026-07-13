from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agent.tools.http import MAX_HEADER_COUNT, MAX_REQUEST_BYTES, HttpTool


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "http://example.com/")
            self.end_headers()
            return
        body = json.dumps({"ok": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        pass


def test_http_tool_allowlist_and_response_limit(tmp_path: Path) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        tool = HttpTool(tmp_path, allowed_domains=["127.0.0.1"], max_response_bytes=1024)
        result = tool.request(f"http://127.0.0.1:{server.server_port}/")
        assert result.success is True
        assert result.data["json"] == {"ok": True}
        denied = HttpTool(tmp_path, allowed_domains=["example.com"]).request(f"http://127.0.0.1:{server.server_port}/")
        assert denied.success is False
        assert "not allowed" in denied.stderr
        tiny = HttpTool(tmp_path, allowed_domains=["127.0.0.1"], max_response_bytes=2).request(
            f"http://127.0.0.1:{server.server_port}/"
        )
        assert tiny.success is False
        assert "exceeds" in tiny.stderr
        redirected = tool.request(f"http://127.0.0.1:{server.server_port}/redirect")
        assert redirected.success is False
        assert "HTTP 302" in redirected.stderr
        oversized_body = tool.request(
            f"http://127.0.0.1:{server.server_port}/",
            method="POST",
            json_body={"value": "x" * MAX_REQUEST_BYTES},
        )
        assert oversized_body.success is False
        assert "request body exceeds" in oversized_body.stderr
        too_many_headers = tool.request(
            f"http://127.0.0.1:{server.server_port}/",
            headers={f"X-Test-{index}": "value" for index in range(MAX_HEADER_COUNT + 1)},
        )
        assert too_many_headers.success is False
        assert "at most" in too_many_headers.stderr
    finally:
        server.shutdown()
        server.server_close()
