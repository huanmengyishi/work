from __future__ import annotations

import errno
import http.client
import json
import re
import socket
import ssl
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
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    http_attempt_count: int = 0


class DeepSeekContextOverflow(RuntimeError):
    """DeepSeek rejected a request because it exceeds the model context window."""

    def __init__(self, message: str, *, http_attempt_count: int = 0) -> None:
        super().__init__(message)
        self.http_attempt_count = max(0, int(http_attempt_count))


class _DeepSeekRequestError(RuntimeError):
    """Internal request failure carrying attempts made before it escaped."""

    def __init__(self, message: str, *, http_attempt_count: int) -> None:
        super().__init__(message)
        self.http_attempt_count = max(0, int(http_attempt_count))


class DeepSeekStreamInterrupted(RuntimeError):
    """A streamed request emitted partial output and must not be replayed automatically."""

    def __init__(self, message: str, *, http_attempt_count: int = 0) -> None:
        super().__init__(message)
        self.http_attempt_count = max(0, int(http_attempt_count))


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

        data, http_attempt_count = self._request_with_key_pool(normalize_unicode_data(payload))
        data = normalize_unicode_data(data)
        choices = data.get("choices") or []
        if not choices:
            raise _DeepSeekRequestError(
                f"DeepSeek API returned no choices: {data}",
                http_attempt_count=http_attempt_count,
            )
        message = choices[0].get("message") or {}
        if not isinstance(message, dict):
            raise _DeepSeekRequestError(
                f"DeepSeek API returned invalid message: {data}",
                http_attempt_count=http_attempt_count,
            )
        finish_reason = choices[0].get("finish_reason")
        usage = data.get("usage")
        return ChatResponse(
            message=message,
            raw=data,
            finish_reason=str(finish_reason) if finish_reason is not None else None,
            usage=usage if isinstance(usage, dict) else None,
            http_attempt_count=http_attempt_count,
        )

    def check_key_pool(self) -> int:
        """Verify each configured key once for `agent doctor --online`."""
        ready = 0
        failures: list[int] = []
        http_attempt_count = 0
        retries = max(0, min(int(self.config.get("model.network_retries", 2)), 5))
        base_delay = max(0.0, min(float(self.config.get("model.retry_base_seconds", 1.0)), 10.0))
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
            for attempt in range(retries + 1):
                try:
                    http_attempt_count += 1
                    self._request(payload, key)
                except urllib.error.HTTPError as exc:
                    if exc.code in {401, 403, 429}:
                        failures.append(exc.code)
                        break
                    if exc.code in {408, 500, 502, 503, 504} and attempt < retries:
                        time.sleep(base_delay * (2**attempt))
                        continue
                    raise _DeepSeekRequestError(
                        f"DeepSeek API HTTP {exc.code}",
                        http_attempt_count=http_attempt_count,
                    ) from exc
                except Exception as exc:
                    if not _is_transient_connection_error(exc):
                        raise _DeepSeekRequestError(
                            str(exc),
                            http_attempt_count=http_attempt_count,
                        ) from exc
                    if attempt < retries:
                        time.sleep(base_delay * (2**attempt))
                        continue
                    raise _DeepSeekRequestError(
                        f"DeepSeek API request failed after {attempt + 1} attempt(s): {exc}",
                        http_attempt_count=http_attempt_count,
                    ) from exc
                ready += 1
                break
        if failures:
            status_text = ", ".join(str(code) for code in failures)
            raise _DeepSeekRequestError(
                f"DeepSeek Key pool check: {ready}/{len(self.api_keys)} key(s) ready; failed statuses: {status_text}. "
                f"Update DEEPSEEK_API_KEY in {self.config.config_dir / 'secrets.env'}.",
                http_attempt_count=http_attempt_count,
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
        payload["stream_options"] = {"include_usage": True}
        emitted = False
        stream_http_attempt_count = 0

        def mark_stream_started() -> None:
            nonlocal emitted
            # Any valid assistant delta, including an in-progress tool call,
            # means replaying the request could duplicate model work or a
            # future side effect. Only a pre-delta failure is retryable.
            emitted = True

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
            message, raw, finish_reason, usage, http_attempt_count = self._request_stream_with_key_pool(
                normalize_unicode_data(payload),
                on_reasoning=emit_reasoning,
                on_content=emit_content,
                on_valid_data=mark_stream_started,
            )
        except DeepSeekContextOverflow:
            raise
        except RuntimeError as exc:
            stream_http_attempt_count = max(0, int(getattr(exc, "http_attempt_count", 0)))
            if emitted:
                raise DeepSeekStreamInterrupted(
                    "DeepSeek stream was interrupted after partial output; resume the saved Session to avoid "
                    "duplicating an in-flight tool call.",
                    http_attempt_count=stream_http_attempt_count,
                ) from None
            # A stream can fail before any useful chunk reaches the client. The regular
            # request path keeps the task recoverable; callers still have elapsed-time progress.
            try:
                response = self.chat(
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    max_tokens=max_tokens,
                    thinking=thinking,
                    reasoning_effort=reasoning_effort,
                    model=model,
                )
            except RuntimeError as fallback_error:
                _add_http_attempts(fallback_error, stream_http_attempt_count)
                raise
            return ChatResponse(
                message=response.message,
                raw=response.raw,
                finish_reason=response.finish_reason,
                usage=response.usage,
                http_attempt_count=stream_http_attempt_count + response.http_attempt_count,
            )
        return ChatResponse(
            message=normalize_unicode_data(message),
            raw=raw,
            finish_reason=finish_reason,
            usage=usage,
            http_attempt_count=http_attempt_count,
        )

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

    def _request_with_key_pool(self, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        failures: list[int] = []
        http_attempt_count = 0
        key_count = len(self.api_keys)
        retries = max(0, min(int(self.config.get("model.network_retries", 2)), 5))
        base_delay = max(0.0, min(float(self.config.get("model.retry_base_seconds", 1.0)), 10.0))
        for offset in range(key_count):
            key_index = (self._next_key_index + offset) % key_count
            key = self.api_keys[key_index]
            for attempt in range(retries + 1):
                try:
                    http_attempt_count += 1
                    data = self._request(payload, key)
                except urllib.error.HTTPError as exc:
                    if exc.code in {401, 403, 429}:
                        failures.append(exc.code)
                        break
                    if exc.code in {408, 500, 502, 503, 504} and attempt < retries:
                        time.sleep(base_delay * (2**attempt))
                        continue
                    body = _bounded_http_error_body(exc)
                    if _is_context_overflow_http_error(exc.code, body):
                        raise DeepSeekContextOverflow(
                            f"DeepSeek API HTTP {exc.code}: {body}",
                            http_attempt_count=http_attempt_count,
                        ) from exc
                    raise _DeepSeekRequestError(
                        f"DeepSeek API HTTP {exc.code}: {body}",
                        http_attempt_count=http_attempt_count,
                    ) from exc
                except Exception as exc:
                    if not _is_transient_connection_error(exc):
                        raise _DeepSeekRequestError(
                            str(exc),
                            http_attempt_count=http_attempt_count,
                        ) from exc
                    if attempt < retries:
                        time.sleep(base_delay * (2**attempt))
                        continue
                    raise _DeepSeekRequestError(
                        f"DeepSeek API request failed after {attempt + 1} attempt(s): {exc}",
                        http_attempt_count=http_attempt_count,
                    ) from exc
                self._next_key_index = (key_index + 1) % key_count
                return data, http_attempt_count
        status_text = ", ".join(str(code) for code in failures) or "unknown"
        raise _DeepSeekRequestError(
            f"DeepSeek Key pool failed after trying {key_count} key(s); HTTP statuses: {status_text}. "
            f"Update DEEPSEEK_API_KEY in {self.config.config_dir / 'secrets.env'}.",
            http_attempt_count=http_attempt_count,
        )

    def _request_stream_with_key_pool(
        self,
        payload: dict[str, Any],
        *,
        on_reasoning: Any,
        on_content: Any,
        on_valid_data: Any = None,
    ) -> tuple[dict[str, Any], dict[str, Any], str | None, dict[str, Any] | None, int]:
        failures: list[int] = []
        http_attempt_count = 0
        key_count = len(self.api_keys)
        retries = max(0, min(int(self.config.get("model.network_retries", 2)), 5))
        base_delay = max(0.0, min(float(self.config.get("model.retry_base_seconds", 1.0)), 10.0))
        for offset in range(key_count):
            key_index = (self._next_key_index + offset) % key_count
            key = self.api_keys[key_index]
            switch_key = False
            for attempt in range(retries + 1):
                attempt_started = False

                def mark_attempt_started() -> None:
                    nonlocal attempt_started
                    attempt_started = True
                    if on_valid_data:
                        on_valid_data()

                try:
                    http_attempt_count += 1
                    message, raw, finish_reason, usage = self._request_stream(
                        payload,
                        key,
                        on_reasoning=on_reasoning,
                        on_content=on_content,
                        on_valid_data=mark_attempt_started,
                    )
                except (json.JSONDecodeError, UnicodeError, TypeError, ValueError) as exc:
                    if attempt_started:
                        raise _DeepSeekRequestError(
                            f"DeepSeek streaming response became invalid: {exc}",
                            http_attempt_count=http_attempt_count,
                        ) from exc
                    if attempt < retries:
                        time.sleep(base_delay * (2**attempt))
                        continue
                    raise _DeepSeekRequestError(
                        f"DeepSeek streaming response was invalid before the first response data "
                        f"after {attempt + 1} attempt(s): {exc}",
                        http_attempt_count=http_attempt_count,
                    ) from exc
                except urllib.error.HTTPError as exc:
                    if exc.code in {401, 403, 429} and not attempt_started:
                        failures.append(exc.code)
                        switch_key = True
                        break
                    if exc.code in {408, 500, 502, 503, 504} and not attempt_started and attempt < retries:
                        time.sleep(base_delay * (2**attempt))
                        continue
                    body = _bounded_http_error_body(exc)
                    if not attempt_started and _is_context_overflow_http_error(exc.code, body):
                        raise DeepSeekContextOverflow(
                            f"DeepSeek API HTTP {exc.code}: {body}",
                            http_attempt_count=http_attempt_count,
                        ) from exc
                    raise _DeepSeekRequestError(
                        f"DeepSeek API HTTP {exc.code}: {body}",
                        http_attempt_count=http_attempt_count,
                    ) from exc
                except Exception as exc:
                    if not _is_transient_connection_error(exc):
                        raise _DeepSeekRequestError(
                            str(exc),
                            http_attempt_count=http_attempt_count,
                        ) from exc
                    if attempt_started:
                        raise _DeepSeekRequestError(
                            f"DeepSeek streaming request was interrupted: {exc}",
                            http_attempt_count=http_attempt_count,
                        ) from exc
                    if attempt < retries:
                        time.sleep(base_delay * (2**attempt))
                        continue
                    raise _DeepSeekRequestError(
                        f"DeepSeek streaming request failed before the first response data "
                        f"after {attempt + 1} attempt(s): {exc}",
                        http_attempt_count=http_attempt_count,
                    ) from exc
                self._next_key_index = (key_index + 1) % key_count
                return message, raw, finish_reason, usage, http_attempt_count
            if switch_key:
                continue
        status_text = ", ".join(str(code) for code in failures) or "unknown"
        raise _DeepSeekRequestError(
            f"DeepSeek Key pool failed after trying {key_count} key(s); HTTP statuses: {status_text}.",
            http_attempt_count=http_attempt_count,
        )

    def _request_stream(
        self,
        payload: dict[str, Any],
        key: str,
        *,
        on_reasoning: Any,
        on_content: Any,
        on_valid_data: Any = None,
    ) -> tuple[dict[str, Any], dict[str, Any], str | None, dict[str, Any] | None]:
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
        finish_reason: str | None = None
        usage: dict[str, Any] | None = None
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
                if not isinstance(chunk, dict):
                    continue
                chunk_usage = chunk.get("usage")
                if isinstance(chunk_usage, dict):
                    usage = chunk_usage
                choices = chunk.get("choices") or []
                if not choices:
                    last_chunk = chunk
                    continue
                choice = choices[0]
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    continue
                choice_finish_reason = choice.get("finish_reason")
                has_valid_delta = bool(delta)
                if has_valid_delta and on_valid_data:
                    on_valid_data()
                last_chunk = chunk
                if choice_finish_reason is not None:
                    finish_reason = str(choice_finish_reason)
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
        return message, last_chunk, finish_reason, usage

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


_TRANSIENT_ERRNOS = frozenset(
    {
        errno.ECONNABORTED,
        errno.ECONNREFUSED,
        errno.ECONNRESET,
        errno.EHOSTUNREACH,
        errno.EINTR,
        errno.ENETDOWN,
        errno.ENETRESET,
        errno.ENETUNREACH,
        errno.EPIPE,
        errno.ETIMEDOUT,
    }
)
_TRANSIENT_OSERROR_MARKERS = (
    "broken pipe",
    "connection aborted",
    "connection reset",
    "eof occurred in violation",
    "incomplete read",
    "record layer failure",
    "remote end closed",
    "remote disconnected",
    "temporary failure",
    "timed out",
    "tls",
)

_HTTP_ERROR_BODY_LIMIT = 4_096
_SECRET_FIELD_RE = re.compile(
    r'(?i)(["\']?(?:api[_-]?key|authorization|cookie|password|secret|token)["\']?\s*[:=]\s*)'
    r'("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|[^\s,;}]+)'
)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+\-/=]+")
_CONTEXT_OVERFLOW_MARKERS = (
    "context length exceeded",
    "context_length_exceeded",
    "context window exceeded",
    "context_window_exceeded",
    "context window is too long",
    "context length is too long",
    "context_length is too long",
    "exceeds the context window",
    "exceeds the context length",
    "maximum context length",
    "max context length",
    "prompt is too long",
    "request is too large for the model",
    "too many tokens",
)
_CONTEXT_OVERFLOW_STRUCTURED_CODES = frozenset(
    {
        "context_length_exceeded",
        "context_window_exceeded",
        "max_context_length_exceeded",
    }
)


def _add_http_attempts(error: RuntimeError, previous_attempts: int) -> None:
    """Preserve an error's type/message while adding attempts from an earlier request path."""
    current_attempts = max(0, int(getattr(error, "http_attempt_count", 0)))
    error.http_attempt_count = max(0, int(previous_attempts)) + current_attempts


def _bounded_http_error_body(exc: urllib.error.HTTPError) -> str:
    """Read a bounded error body and redact common credential-shaped values."""
    try:
        raw = exc.read(_HTTP_ERROR_BODY_LIMIT + 1)
    except TypeError:
        raw = exc.read()
    if not isinstance(raw, bytes):
        raw = bytes(str(raw), "utf-8")
    truncated = len(raw) > _HTTP_ERROR_BODY_LIMIT
    body = raw[:_HTTP_ERROR_BODY_LIMIT].decode("utf-8", errors="replace")
    body = _BEARER_RE.sub("Bearer [REDACTED]", body)
    body = _SECRET_FIELD_RE.sub(lambda match: f"{match.group(1)}[REDACTED]", body)
    if truncated:
        body += "...[truncated]"
    return body


def _is_context_overflow_http_error(status: int, body: str) -> bool:
    """Strictly classify only HTTP 400/413 responses with an overflow marker."""
    if status not in {400, 413}:
        return False
    normalized = " ".join(str(body).lower().split())
    if any(marker in normalized for marker in _CONTEXT_OVERFLOW_MARKERS):
        return True
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False
    candidates: list[Any] = [data]
    if isinstance(data, dict) and isinstance(data.get("error"), dict):
        candidates.append(data["error"])
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for field in ("code", "type"):
            if str(candidate.get(field) or "").strip().lower() in _CONTEXT_OVERFLOW_STRUCTURED_CODES:
                return True
    return False


def _is_transient_connection_error(exc: BaseException) -> bool:
    """Return whether an exception represents a retryable transport failure."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(
            current,
            (
                ssl.SSLError,
                http.client.IncompleteRead,
                http.client.RemoteDisconnected,
                ConnectionResetError,
                BrokenPipeError,
                TimeoutError,
                socket.timeout,
            ),
        ):
            return True
        if isinstance(current, urllib.error.URLError):
            reason = current.reason
            if isinstance(reason, BaseException):
                current = reason
                continue
            return True
        if isinstance(current, OSError):
            if current.errno in _TRANSIENT_ERRNOS:
                return True
            message = str(current).lower()
            if any(marker in message for marker in _TRANSIENT_OSERROR_MARKERS):
                return True
        next_error = current.__cause__ or current.__context__
        current = next_error if isinstance(next_error, BaseException) else None
    return False
