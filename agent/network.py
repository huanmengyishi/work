from __future__ import annotations

import os
from urllib.parse import urlsplit, urlunsplit


PROXY_ENV_NAMES = (
    "HTTPS_PROXY",
    "https_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "ALL_PROXY",
    "all_proxy",
)


def proxy_url_from_env() -> str | None:
    for name in PROXY_ENV_NAMES:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def redacted_proxy_url(value: str | None) -> str:
    if not value:
        return "not configured"
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.netloc:
        return "configured"
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    port = f":{parsed.port}" if parsed.port else ""
    return urlunsplit((parsed.scheme, f"{hostname}{port}", parsed.path, parsed.query, parsed.fragment))
