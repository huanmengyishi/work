from __future__ import annotations

import json
import socket
import time
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


class DeepSeekStreamInterrupted(RuntimeError):
    """A streamed request emitted partial output and must not be replayed automatically."""


class DeepSeekClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        provider = str(config.get("model.provider", "deepseek")).strip().lower()
        if provider != "deepseek":
            raise ValueError("Deep Agent supports only the DeepSeek provider")
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
        thinking: dict[str, Any] | bool | None = None,
        reasoning_effort: str | None = None,
        model: str | None = None,
    ) -> ChatResponse:
        if not self.api_keys:
            env_name = self.config.get("model.api_key_env", "DEEPSEEK_API_KEY")
            raise RuntimeError(
                f"DeepSeek API key is missing. Set environment variable {env_name}, "
                f"or configure model.api_key in {self.config.config_dir / 'model.yaml'}."
            )
        payload = self._chat_payload(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            model=model,
        )

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

    def chat_stream(
        self,
        *,
        messages: list[dict[str, Any]],
        on_reasoning: Any = None,
        on_content: Any = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = "auto",
        max_tokens: int | None = None,
        thinking: dict[str, Any] | bool | None = None,
        reasoning_effort: str | None = None,
        model: str | None = None,
    ) -> ChatResponse:
        """Stream visible DeepSeek thinking/text while returning one normal assistant message."""
        if not self.api_keys:
            return self.chat(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
                thinking=thinking,
                reasoning_effort=reasoning_effort,
                model=model,
            )
        payload = self._chat_payload(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            model=model,
        )
        payload["stream"] = True
        emitted = False

        def emit_reasoning(value: str) -> None:
            nonlocal emitted
            emitted = True
            if on_reasoning:
                on_reasoning(value)

        def emit_content(value: str) -> None:
            nonlocal emitted
            emitted = True
            if on_content:
                on_content(value)

        try:
            message, raw = self._request_stream_with_key_pool(
                normalize_unicode_data(payload),
                on_reasoning=emit_reasoning,
                on_content=emit_content,
            )
        except RuntimeError:
            if emitted:
                raise DeepSeekStreamInterrupted(
                    "DeepSeek stream was interrupted after partial output; resume the saved Session to avoid "
                    "duplicating an in-flight tool call."
                ) from None
            # A stream can fail before any useful chunk reaches the client. The regular
            # request path keeps the task recoverable; callers still have elapsed-time progress.
            return self.chat(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
                thinking=thinking,
                reasoning_effort=reasoning_effort,
                model=model,
            )
        return ChatResponse(message=normalize_unicode_data(message), raw=raw)

    def _chat_payload(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        max_tokens: int | None,
        thinking: dict[str, Any] | bool | None,
        reasoning_effort: str | None,
        model: str | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": str(model or self.model),
            "messages": messages,
            "temperature": self.config.get("model.temperature", 0.2),
            "max_tokens": max_tokens if max_tokens is not None else self.config.get("model.max_tokens", 4096),
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        effective_effort = (
            reasoning_effort if reasoning_effort is not None else self.config.get("model.reasoning_effort")
        )
        if effective_effort:
            payload["reasoning_effort"] = effective_effort
        effective_thinking = thinking if thinking is not None else self.config.get("model.thinking")
        if isinstance(effective_thinking, bool):
            effective_thinking = {"type": "enabled" if effective_thinking else "disabled"}
        if effective_thinking is not None:
            payload["thinking"] = effective_thinking
        if self._thinking_enabled(effective_thinking):
            payload.pop("temperature", None)
        return payload

    def _request_with_key_pool(self, payload: dict[str, Any]) -> dict[str, Any]:
        failures: list[int] = []
        key_count = len(self.api_keys)
        retries = max(0, min(int(self.config.get("model.network_retries", 2)), 5))
        base_delay = max(0.0, min(float(self.config.get("model.retry_base_seconds", 1.0)), 10.0))
        for offset in range(key_count):
            key_index = (self._next_key_index + offset) % key_count
            key = self.api_keys[key_index]
            for attempt in range(retries + 1):
                try:
                    data = self._request(payload, key)
                except urllib.error.HTTPError as exc:
                    if exc.code in {401, 403, 429}:
                        failures.append(exc.code)
                        break
                    if exc.code in {408, 500, 502, 503, 504} and attempt < retries:
                        time.sleep(base_delay * (2**attempt))
                        continue
                    body = exc.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"DeepSeek API HTTP {exc.code}: {body}") from exc
                except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                    if attempt < retries:
                        time.sleep(base_delay * (2**attempt))
                        continue
                    raise RuntimeError(f"DeepSeek API request failed after {attempt + 1} attempt(s): {exc}") from exc
                self._next_key_index = (key_index + 1) % key_count
                return data
        status_text = ", ".join(str(code) for code in failures) or "unknown"
        raise RuntimeError(
            f"DeepSeek Key pool failed after trying {key_count} key(s); HTTP statuses: {status_text}. "
            f"Update DEEPSEEK_API_KEY in {self.config.config_dir / 'secrets.env'}."
        )

    def _request_stream_with_key_pool(
        self,
        payload: dict[str, Any],
        *,
        on_reasoning: Any,
        on_content: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        failures: list[int] = []
        key_count = len(self.api_keys)
        for offset in range(key_count):
            key_index = (self._next_key_index + offset) % key_count
            key = self.api_keys[key_index]
            try:
                message, raw = self._request_stream(
                    payload,
                    key,
                    on_reasoning=on_reasoning,
                    on_content=on_content,
                )
            except urllib.error.HTTPError as exc:
                if exc.code in {401, 403, 429}:
                    failures.append(exc.code)
                    continue
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"DeepSeek API HTTP {exc.code}: {body}") from exc
            except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                raise RuntimeError(f"DeepSeek streaming request failed: {exc}") from exc
            self._next_key_index = (key_index + 1) % key_count
            return message, raw
        status_text = ", ".join(str(code) for code in failures) or "unknown"
        raise RuntimeError(f"DeepSeek Key pool failed after trying {key_count} key(s); HTTP statuses: {status_text}.")

    def _request_stream(
        self,
        payload: dict[str, Any],
        key: str,
        *,
        on_reasoning: Any,
        on_content: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        req = urllib.request.Request(
            self.base_url + self.chat_path,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        timeout = max(30, min(int(self.config.get("model.timeout_seconds", 300)), 1800))
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        last_chunk: dict[str, Any] = {}
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data_text = line[5:].strip()
                if data_text == "[DONE]":
                    break
                if not data_text:
                    continue
                chunk = json.loads(data_text)
                last_chunk = chunk
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                reasoning = str(delta.get("reasoning_content") or "")
                content = str(delta.get("content") or "")
                if reasoning:
                    reasoning_parts.append(reasoning)
                    if on_reasoning:
                        on_reasoning(reasoning)
                if content:
                    content_parts.append(content)
                    if on_content:
                        on_content(content)
                for item in delta.get("tool_calls") or []:
                    index = int(item.get("index") or 0)
                    current = tool_calls.setdefault(
                        index,
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    if item.get("id"):
                        current["id"] = str(item["id"])
                    if item.get("type"):
                        current["type"] = str(item["type"])
                    function = item.get("function") or {}
                    if function.get("name"):
                        current["function"]["name"] += str(function["name"])
                    if function.get("arguments"):
                        current["function"]["arguments"] += str(function["arguments"])
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(content_parts) or None,
        }
        reasoning_content = "".join(reasoning_parts)
        if reasoning_content:
            message["reasoning_content"] = reasoning_content
        if tool_calls:
            message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
        return message, last_chunk

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
        timeout = max(30, min(int(self.config.get("model.timeout_seconds", 300)), 1800))
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _thinking_enabled(value: Any) -> bool:
        if isinstance(value, dict):
            return str(value.get("type") or "").lower() == "enabled"
        if isinstance(value, bool):
            return value
        return str(value or "").lower() in {"enabled", "true", "on", "1"}
