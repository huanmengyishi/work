from __future__ import annotations

import json
import mimetypes
import re
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..network import proxy_url_from_env
from .base import ToolResult, truncate_text


SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


class BrowserTool:
    """Playwright browser automation with optional project-local persistent contexts."""

    def __init__(self, cwd: Path, timeout: int = 180, max_download_bytes: int = 100_000_000) -> None:
        self.cwd = cwd
        self.timeout = timeout
        self.max_download_bytes = max(1, min(int(max_download_bytes), 500_000_000))
        self.session_root = cwd / ".project-agent" / "browser-sessions"
        self.download_root = cwd / ".project-agent" / "downloads"
        self.session_root.mkdir(parents=True, exist_ok=True)
        self.download_root.mkdir(parents=True, exist_ok=True)
        for path in (self.session_root, self.download_root):
            try:
                path.chmod(0o700)
            except OSError:
                pass

    def open_url(self, url: str, session_name: str | None = None) -> ToolResult:
        error = self._validate_url(url)
        if error:
            return ToolResult(False, "", error)
        try:
            with self._context(session_name) as context:
                page = context.pages[0] if context.pages else context.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=min(self.timeout * 1000, 60_000))
                title = page.title()
                body = page.locator("body").inner_text(timeout=min(self.timeout * 1000, 10_000))
                data = {
                    "url": page.url,
                    "title": title,
                    "session_name": session_name,
                    "persistent": bool(session_name),
                }
                return ToolResult(True, truncate_text(f"{title}\n{body}", 8000), data=data)
        except Exception as exc:
            return ToolResult(False, "", f"browser open failed: {exc}")

    def download(
        self,
        url: str,
        selector: str,
        session_name: str | None = None,
        filename: str | None = None,
    ) -> ToolResult:
        error = self._validate_url(url)
        if error:
            return ToolResult(False, "", error)
        if not selector:
            return ToolResult(False, "", "download selector is empty")
        try:
            safe_session = self._session_name(session_name or "default")
            destination_dir = self.download_root / safe_session
            destination_dir.mkdir(parents=True, exist_ok=True)
            try:
                destination_dir.chmod(0o700)
            except OSError:
                pass
            with self._context(session_name) as context:
                page = context.pages[0] if context.pages else context.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=min(self.timeout * 1000, 60_000))
                with page.expect_download(timeout=self.timeout * 1000) as download_info:
                    page.locator(selector).click()
                download = download_info.value
                selected_name = filename or download.suggested_filename or "download.bin"
                target = self._unique_download_path(destination_dir, selected_name)
                download.save_as(target)
                size = target.stat().st_size
                if size > self.max_download_bytes:
                    target.unlink(missing_ok=True)
                    return ToolResult(False, "", f"browser download exceeds {self.max_download_bytes} bytes")
                mime_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                relative = target.relative_to(self.cwd).as_posix()
                return ToolResult(
                    True,
                    f"downloaded {relative}",
                    data={
                        "path": relative,
                        "absolute_path": str(target),
                        "filename": target.name,
                        "mime_type": mime_type,
                        "size_bytes": size,
                        "url": download.url,
                        "session_name": session_name,
                    },
                )
        except Exception as exc:
            return ToolResult(False, "", f"browser download failed: {exc}")

    def close_session(self, session_name: str, clear_data: bool = False) -> ToolResult:
        try:
            name = self._session_name(session_name)
        except ValueError as exc:
            return ToolResult(False, "", str(exc))
        session_dir = self.session_root / name
        if clear_data:
            if session_dir.exists():
                shutil.rmtree(session_dir)
            return ToolResult(
                True, f"cleared browser session data: {name}", data={"session_name": name, "cleared": True}
            )
        return ToolResult(
            True,
            f"browser sessions close after every tool call; persisted data remains for {name}",
            data={"session_name": name, "cleared": False, "path": str(session_dir)},
        )

    def list_sessions(self) -> ToolResult:
        sessions: list[dict[str, Any]] = []
        for path in sorted(self.session_root.iterdir()):
            if not path.is_dir() or not SESSION_RE.fullmatch(path.name):
                continue
            size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
            sessions.append({"name": path.name, "path": str(path), "size_bytes": size})
        return ToolResult(True, json.dumps(sessions, ensure_ascii=False, indent=2), data={"sessions": sessions})

    def _context(self, session_name: str | None):
        from playwright.sync_api import sync_playwright

        playwright = sync_playwright().start()
        launch_options: dict[str, Any] = {"headless": True, "accept_downloads": True}
        proxy = proxy_url_from_env()
        if proxy:
            launch_options["proxy"] = {"server": proxy}
        try:
            if session_name:
                name = self._session_name(session_name)
                user_data_dir = self.session_root / name
                user_data_dir.mkdir(parents=True, exist_ok=True)
                try:
                    user_data_dir.chmod(0o700)
                except OSError:
                    pass
                context = playwright.chromium.launch_persistent_context(str(user_data_dir), **launch_options)
            else:
                browser = playwright.chromium.launch(
                    **{key: value for key, value in launch_options.items() if key != "accept_downloads"}
                )
                context = browser.new_context(accept_downloads=True)
        except Exception:
            playwright.stop()
            raise
        return _BrowserContext(playwright, context)

    @staticmethod
    def _validate_url(url: str) -> str:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return "browser URL must use http or https"
        if parsed.username is not None or parsed.password is not None:
            return "browser URL must not contain credentials; use a named browser session for authentication"
        return ""

    @staticmethod
    def _session_name(value: str) -> str:
        if not SESSION_RE.fullmatch(value):
            raise ValueError("session_name must match [A-Za-z0-9][A-Za-z0-9_-]{0,63}")
        return value

    @staticmethod
    def _unique_download_path(directory: Path, filename: str) -> Path:
        safe = SAFE_FILENAME_RE.sub("_", Path(filename).name).strip("._") or "download.bin"
        candidate = directory / safe
        counter = 1
        while candidate.exists():
            candidate = directory / f"{Path(safe).stem}-{counter}{Path(safe).suffix}"
            counter += 1
        return candidate


class _BrowserContext:
    def __init__(self, playwright, context) -> None:
        self.playwright = playwright
        self.context = context

    def __enter__(self):
        return self.context

    def __exit__(self, exc_type, exc, traceback) -> bool:
        try:
            self.context.close()
        finally:
            self.playwright.stop()
        return False
