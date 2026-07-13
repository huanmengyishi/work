from __future__ import annotations

from agent.network import proxy_url_from_env, redacted_proxy_url


def test_proxy_selection_and_redaction(monkeypatch) -> None:
    for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTP_PROXY", "http://user:secret@172.19.80.1:7897")

    value = proxy_url_from_env()

    assert value == "http://user:secret@172.19.80.1:7897"
    assert redacted_proxy_url(value) == "http://172.19.80.1:7897"
