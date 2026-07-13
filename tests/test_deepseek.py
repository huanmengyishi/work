from __future__ import annotations

import io
import json
import urllib.error

import pytest

from agent.config import parse_api_keys
from agent.deepseek import DeepSeekClient


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def http_error(url: str, code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, "failure", {}, io.BytesIO(b'{"error":"redacted"}'))


def test_parse_api_keys_accepts_comma_whitespace_and_duplicates() -> None:
    assert parse_api_keys(" key-one, key-two ,,key-one， key-three ") == ("key-one", "key-two", "key-three")


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
