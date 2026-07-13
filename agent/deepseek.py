from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .config import AppConfig
from .unicode_text import normalize_unicode_data


@dataclass(frozen=True)
class ChatResponse:
    message: dict[str, Any]
    raw: dict[str, Any]


class DeepSeekClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.base_url = str(config.get("model.base_url", "https://api.deepseek.com")).rstrip("/")
        self.chat_path = str(config.get("model.chat_path", "/chat/completions"))
        self.model = str(config.get("model.model", "deepseek-v4-pro"))
        self.api_keys = config.api_keys
        self._next_key_index = 0

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        max_tokens: int | None = None,
    ) -> ChatResponse:
        if not self.api_keys:
            env_name = self.config.get("model.api_key_env", "DEEPSEEK_API_KEY")
            raise RuntimeError(
                f"DeepSeek API key is missing. Set environment variable {env_name}, "
                f"or configure model.api_key in {self.config.config_dir / 'model.yaml'}."
            )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config.get("model.temperature", 0.2),
            "max_tokens": max_tokens if max_tokens is not None else self.config.get("model.max_tokens", 4096),
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        reasoning_effort = self.config.get("model.reasoning_effort")
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        thinking = self.config.get("model.thinking")
        if thinking is not None:
            payload["thinking"] = thinking

        data = normalize_unicode_data(self._request_with_key_pool(normalize_unicode_data(payload)))
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"DeepSeek API returned no choices: {data}")
        message = choices[0].get("message") or {}
        if not isinstance(message, dict):
            raise RuntimeError(f"DeepSeek API returned invalid message: {data}")
        return ChatResponse(message=message, raw=data)

    def check_key_pool(self) -> int:
        """Verify each configured key once for `agent doctor --online`."""
        ready = 0
        failures: list[int] = []
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Reply with OK."},
                {"role": "user", "content": "Connection check."},
            ],
            "temperature": 0,
            "max_tokens": 8,
        }
        for key in self.api_keys:
            try:
                self._request(payload, key)
            except urllib.error.HTTPError as exc:
                if exc.code in {401, 403, 429}:
                    failures.append(exc.code)
                    continue
                raise RuntimeError(f"DeepSeek API HTTP {exc.code}") from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(f"DeepSeek API request failed: {exc}") from exc
            ready += 1
        if failures:
            status_text = ", ".join(str(code) for code in failures)
            raise RuntimeError(
                f"DeepSeek Key pool check: {ready}/{len(self.api_keys)} key(s) ready; failed statuses: {status_text}. "
                f"Update DEEPSEEK_API_KEY in {self.config.config_dir / 'secrets.env'}."
            )
        return ready

    def _request_with_key_pool(self, payload: dict[str, Any]) -> dict[str, Any]:
        failures: list[int] = []
        key_count = len(self.api_keys)
        for offset in range(key_count):
            key_index = (self._next_key_index + offset) % key_count
            key = self.api_keys[key_index]
            try:
                data = self._request(payload, key)
            except urllib.error.HTTPError as exc:
                if exc.code in {401, 403, 429}:
                    failures.append(exc.code)
                    continue
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"DeepSeek API HTTP {exc.code}: {body}") from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(f"DeepSeek API request failed: {exc}") from exc
            self._next_key_index = (key_index + 1) % key_count
            return data
        status_text = ", ".join(str(code) for code in failures) or "unknown"
        raise RuntimeError(
            f"DeepSeek Key pool failed after trying {key_count} key(s); HTTP statuses: {status_text}. "
            f"Update DEEPSEEK_API_KEY in {self.config.config_dir / 'secrets.env'}."
        )

    def _request(self, payload: dict[str, Any], key: str) -> dict[str, Any]:
        req = urllib.request.Request(
            self.base_url + self.chat_path,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode("utf-8"))
