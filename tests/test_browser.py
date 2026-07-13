from __future__ import annotations

import contextlib
import functools
import http.server
import threading
from pathlib import Path

from agent.tools.browser import BrowserTool


@contextlib.contextmanager
def local_server(root: Path):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_persistent_session_and_download(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "project"
    site = tmp_path / "site"
    root.mkdir()
    site.mkdir()
    (site / "set.html").write_text(
        "<script>localStorage.setItem('agent-state', 'persisted')</script><body>state written</body>",
        encoding="utf-8",
    )
    (site / "read.html").write_text(
        "<body><script>document.body.textContent=localStorage.getItem('agent-state') || 'missing'</script></body>",
        encoding="utf-8",
    )
    (site / "artifact.txt").write_text("download payload\n", encoding="utf-8")
    (site / "download.html").write_text(
        '<a id="download" href="artifact.txt" download="artifact.txt">download</a>',
        encoding="utf-8",
    )
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)

    browser = BrowserTool(root, timeout=30)
    with local_server(site) as base_url:
        written = browser.open_url(f"{base_url}/set.html", session_name="test-user")
        restored = browser.open_url(f"{base_url}/read.html", session_name="test-user")
        downloaded = browser.download(
            f"{base_url}/download.html",
            "#download",
            session_name="test-user",
        )

    assert written.success is True
    assert restored.success is True
    assert "persisted" in restored.stdout
    assert downloaded.success is True
    target = Path(downloaded.data["absolute_path"])
    assert target.read_text(encoding="utf-8") == "download payload\n"
    assert downloaded.data["mime_type"] == "text/plain"
    assert downloaded.data["path"].startswith(".project-agent/downloads/test-user/")

    sessions = browser.list_sessions()
    assert sessions.success is True
    assert any(item["name"] == "test-user" for item in sessions.data["sessions"])
    cleared = browser.close_session("test-user", clear_data=True)
    assert cleared.success is True
    assert not (root / ".project-agent" / "browser-sessions" / "test-user").exists()


def test_browser_rejects_unsafe_url_and_session_name(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    browser = BrowserTool(root)

    assert browser.open_url("file:///etc/passwd").success is False
    assert browser.open_url("https://user:password@example.com").success is False
    result = browser.open_url("https://example.com", session_name="../escape")
    assert result.success is False
    assert "session_name" in result.stderr
