from __future__ import annotations

import json
import socket
import ipaddress
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .base import ToolResult


SENSITIVE_HEADERS = {"authorization", "cookie", "proxy-authorization", "x-api-key", "api-key"}


class RejectRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class HttpTool:
    def __init__(
        self,
        cwd: Path,
        *,
        allowed_domains: list[str],
        timeout: int = 30,
        max_response_bytes: int = 1_048_576,
    ) -> None:
        self.cwd = cwd
        self.allowed_domains = tuple(domain.strip().lower().rstrip(".") for domain in allowed_domains if domain.strip())
        self.timeout = min(max(timeout, 1), 30)
        self.max_response_bytes = min(max(max_response_bytes, 1), 1_048_576)

    def request(
        self,
        url: str,
        method: str = "GET",
        json_body: Any = None,
        headers: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> ToolResult:
        try:
            parsed = self._validate_url(url)
            verb = method.upper().strip()
            if verb not in {"GET", "POST"}:
                raise ValueError("http_request supports only GET and POST")
            request_headers = {str(key): str(value) for key, value in (headers or {}).items()}
            if any(key.lower() in SENSITIVE_HEADERS for key in request_headers):
                raise ValueError("sensitive authentication headers are not accepted by http_request")
            body = None
            if json_body is not None:
                if verb != "POST":
                    raise ValueError("json_body is supported only with POST")
                body = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
                request_headers.setdefault("Content-Type", "application/json; charset=utf-8")
            request_headers.setdefault("Accept", "application/json, text/plain;q=0.9, */*;q=0.1")
            effective_timeout = min(max(int(timeout or self.timeout), 1), 30)
            request = urllib.request.Request(parsed.geturl(), data=body, headers=request_headers, method=verb)
            with self._open(request, parsed.hostname, effective_timeout) as response:
                raw = self._read_limited(response)
                content_type = response.headers.get_content_type()
                charset = response.headers.get_content_charset() or "utf-8"
                text = raw.decode(charset, errors="replace")
                data: dict[str, Any] = {
                    "url": parsed.geturl(),
                    "status": response.status,
                    "content_type": content_type,
                    "size": len(raw),
                }
                if content_type == "application/json":
                    try:
                        data["json"] = json.loads(text)
                    except json.JSONDecodeError:
                        pass
                return ToolResult(True, text, data=data)
        except urllib.error.HTTPError as exc:
            detail = exc.read(min(self.max_response_bytes, 8192)).decode("utf-8", errors="replace")
            return ToolResult(False, "", f"HTTP {exc.code}: {detail[:8192]}")
        except (urllib.error.URLError, socket.timeout, TimeoutError, ValueError) as exc:
            return ToolResult(False, "", str(exc))

    def _validate_url(self, url: str) -> urllib.parse.ParseResult:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("http_request requires an absolute HTTP(S) URL")
        if parsed.username or parsed.password:
            raise ValueError("credentials embedded in URLs are not allowed")
        host = parsed.hostname.lower().rstrip(".")
        if not self.allowed_domains:
            raise ValueError("http_request has no allowed domains; configure tools.http.allowed_domains")
        if not any(host == domain or host.endswith("." + domain) for domain in self.allowed_domains):
            raise ValueError(f"domain is not allowed: {host}")
        return parsed

    def _read_limited(self, response) -> bytes:
        raw = response.read(self.max_response_bytes + 1)
        if len(raw) > self.max_response_bytes:
            raise ValueError(f"HTTP response exceeds {self.max_response_bytes} bytes")
        return raw

    @staticmethod
    def _open(request: urllib.request.Request, hostname: str, timeout: int):
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
