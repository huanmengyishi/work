from __future__ import annotations

import io
import json
import urllib.error

import pytest

from agent.config import parse_api_keys
from agent.deepseek import DeepSeekClient, DeepSeekStreamInterrupted


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __iter__(self):
        return iter(self.payload)


def http_error(url: str, code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, "failure", {}, io.BytesIO(b'{"error":"redacted"}'))


def test_parse_api_keys_accepts_comma_whitespace_and_duplicates() -> None:
    assert parse_api_keys(" key-one, key-two ,,key-one， key-three ") == ("key-one", "key-two", "key-three")


def test_client_rejects_non_deepseek_provider(make_config) -> None:
    with pytest.raises(ValueError, match="only the DeepSeek provider"):
        DeepSeekClient(make_config({"model": {"provider": "other"}}))


def test_key_pool_rotates_on_auth_and_rate_limit(monkeypatch, make_config) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "first, second, third")
    config = make_config({"model": {"api_key": "first, second, third"}})
    client = DeepSeekClient(config)
    seen_keys: list[str] = []
    responses = [http_error("https://api.deepseek.com", 401), http_error("https://api.deepseek.com", 429)]

    def fake_urlopen(request, timeout):
        seen_keys.append(request.get_header("Authorization"))
        if responses:
            raise responses.pop(0)
        return FakeResponse({"choices": [{"message": {"role": "assistant", "content": "OK"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    response = client.chat(messages=[{"role": "user", "content": "test"}])

    assert response.message["content"] == "OK"
    assert seen_keys == ["Bearer first", "Bearer second", "Bearer third"]


def test_request_repairs_surrogate_pair_before_utf8_encoding(monkeypatch, make_config) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = DeepSeekClient(make_config())
    sent_payload: dict = {}

    def fake_urlopen(request, timeout):
        sent_payload.update(json.loads(request.data.decode("utf-8")))
        return FakeResponse({"choices": [{"message": {"role": "assistant", "content": "OK"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client.chat(messages=[{"role": "user", "content": "状态：\ud83d\ude80"}])

    assert sent_payload["messages"][0]["content"] == "状态：🚀"


def test_key_pool_error_does_not_expose_keys(monkeypatch, make_config) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-one, secret-two")
    config = make_config({"model": {"api_key": "secret-one, secret-two"}})
    client = DeepSeekClient(config)

    def fake_urlopen(request, timeout):
        raise http_error("https://api.deepseek.com", 403)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="Key pool failed after trying 2 key") as error:
        client.chat(messages=[{"role": "user", "content": "test"}])
    assert "secret-one" not in str(error.value)
    assert "secret-two" not in str(error.value)


def test_key_pool_check_reports_partial_failure(monkeypatch, make_config) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "first, second")
    config = make_config({"model": {"api_key": "first, second"}})
    client = DeepSeekClient(config)
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise http_error("https://api.deepseek.com", 401)
        return FakeResponse({"choices": [{"message": {"role": "assistant", "content": "OK"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match=r"1/2 key\(s\) ready"):
        client.check_key_pool()


def test_thinking_request_uses_reasoning_fields_and_omits_temperature(monkeypatch, make_config) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = DeepSeekClient(make_config())
    sent_payload: dict = {}

    def fake_urlopen(request, timeout):
        sent_payload.update(json.loads(request.data.decode("utf-8")))
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "done",
                            "reasoning_content": "bounded reasoning",
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    response = client.chat(
        messages=[{"role": "user", "content": "hard task"}],
        thinking=True,
        reasoning_effort="max",
    )

    assert sent_payload["thinking"] == {"type": "enabled"}
    assert sent_payload["reasoning_effort"] == "max"
    assert "temperature" not in sent_payload
    assert response.message["reasoning_content"] == "bounded reasoning"


def test_model_override_is_request_scoped(monkeypatch, make_config) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = DeepSeekClient(make_config({"model": {"model": "deepseek-base"}}))
    sent_models: list[str] = []

    def fake_urlopen(request, timeout):
        sent_models.append(json.loads(request.data.decode("utf-8"))["model"])
        return FakeResponse({"choices": [{"message": {"role": "assistant", "content": "OK"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    client.chat(messages=[{"role": "user", "content": "fast"}], model="deepseek-fast")
    client.chat(messages=[{"role": "user", "content": "default"}])

    assert sent_models == ["deepseek-fast", "deepseek-base"]
    assert client.model == "deepseek-base"


def test_network_timeout_retries_same_key_with_backoff(monkeypatch, make_config) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = DeepSeekClient(make_config({"model": {"network_retries": 2, "retry_base_seconds": 0}}))
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise urllib.error.URLError(TimeoutError("temporary timeout"))
        return FakeResponse({"choices": [{"message": {"role": "assistant", "content": "OK"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert client.chat(messages=[{"role": "user", "content": "test"}]).message["content"] == "OK"
    assert calls == 3


def test_streaming_chat_emits_reasoning_and_reassembles_tool_call(monkeypatch, make_config) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = DeepSeekClient(make_config())
    lines = [
        b'data: {"choices":[{"delta":{"role":"assistant","reasoning_content":"inspect "}}]}\n',
        b'data: {"choices":[{"delta":{"reasoning_content":"first","tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"search_","arguments":"{\\"query\\":\\""}}]}}]}\n',
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"code","arguments":"bug\\"}"}}]}}]}\n',
        b"data: [DONE]\n",
    ]

    class StreamResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def __iter__(self):
            return iter(lines)

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: StreamResponse())
    chunks: list[str] = []

    response = client.chat_stream(
        messages=[{"role": "user", "content": "find bug"}],
        thinking=True,
        reasoning_effort="high",
        on_reasoning=chunks.append,
    )

    assert chunks == ["inspect ", "first"]
    assert response.message["reasoning_content"] == "inspect first"
    assert response.message["tool_calls"][0]["function"] == {
        "name": "search_code",
        "arguments": '{"query":"bug"}',
    }


def test_partial_stream_failure_is_not_replayed(monkeypatch, make_config) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = DeepSeekClient(make_config())
    calls = 0

    class BrokenStream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"reasoning_content":"partial"}}]}\n'
            raise urllib.error.URLError("connection reset")

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        return BrokenStream()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(DeepSeekStreamInterrupted, match="partial output"):
        client.chat_stream(
            messages=[{"role": "user", "content": "hard task"}],
            thinking=True,
            on_reasoning=lambda _chunk: None,
        )
    assert calls == 1
