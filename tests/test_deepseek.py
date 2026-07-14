from __future__ import annotations

import http.client
import io
import json
import ssl
import urllib.error

import pytest

from agent.config import parse_api_keys
from agent.deepseek import (
    ChatResponse,
    DeepSeekClient,
    DeepSeekContextOverflow,
    DeepSeekStreamInterrupted,
)


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


def http_error(url: str, code: int, body: bytes = b'{"error":"redacted"}') -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, "failure", {}, io.BytesIO(body))


def test_chat_response_protocol_fields_are_backward_compatible() -> None:
    response = ChatResponse(message={"role": "assistant", "content": "OK"}, raw={})

    assert response.finish_reason is None
    assert response.usage is None
    assert response.http_attempt_count == 0


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
    assert error.value.http_attempt_count == 2
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

    with pytest.raises(RuntimeError, match=r"1/2 key\(s\) ready") as error:
        client.check_key_pool()

    assert error.value.http_attempt_count == 2


def test_key_pool_check_wraps_ssl_transport_failure(monkeypatch, make_config) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = DeepSeekClient(make_config({"model": {"network_retries": 1, "retry_base_seconds": 0}}))
    calls = 0

    def fail_ssl(request, timeout):
        nonlocal calls
        calls += 1
        raise ssl.SSLError("[SSL: RECORD_LAYER_FAILURE] record layer failure (_ssl.c:2711)")

    monkeypatch.setattr("urllib.request.urlopen", fail_ssl)

    with pytest.raises(RuntimeError, match=r"failed after 2 attempt\(s\)") as error:
        client.check_key_pool()
    assert calls == 2
    assert error.value.http_attempt_count == 2


def test_key_pool_check_wraps_non_transient_failure_with_attempt_count(monkeypatch, make_config) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = DeepSeekClient(make_config())

    def fail_request(payload, key):
        raise LookupError("key probe protocol failure")

    monkeypatch.setattr(client, "_request", fail_request)

    with pytest.raises(RuntimeError, match="key probe protocol failure") as error:
        client.check_key_pool()

    assert error.value.http_attempt_count == 1
    assert isinstance(error.value.__cause__, LookupError)
    assert str(error.value) == str(error.value.__cause__)


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


def test_non_stream_response_exposes_finish_reason_usage_and_http_attempts(monkeypatch, make_config) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = DeepSeekClient(make_config({"model": {"network_retries": 1, "retry_base_seconds": 0}}))
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.URLError(TimeoutError("temporary timeout"))
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "complete"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    response = client.chat(messages=[{"role": "user", "content": "test"}])

    assert response.finish_reason == "stop"
    assert response.usage == {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
    assert response.http_attempt_count == 2


def test_non_stream_final_http_error_is_bounded_redacted_and_counts_all_retries(monkeypatch, make_config) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = DeepSeekClient(make_config({"model": {"network_retries": 1, "retry_base_seconds": 0}}))
    calls = 0
    secret = "sk-final-http-secret"
    body = json.dumps({"error": {"token": secret, "message": "temporary " + ("x" * 8_000)}}).encode()

    def fail_http(request, timeout):
        nonlocal calls
        calls += 1
        raise http_error(request.full_url, 503, body)

    monkeypatch.setattr("urllib.request.urlopen", fail_http)

    with pytest.raises(RuntimeError, match="DeepSeek API HTTP 503") as error:
        client.chat(messages=[{"role": "user", "content": "test"}])

    assert calls == 2
    assert error.value.http_attempt_count == 2
    assert secret not in str(error.value)
    assert "[truncated]" in str(error.value)
    assert len(str(error.value)) < 4_300


def test_non_stream_final_transport_error_counts_all_attempts(monkeypatch, make_config) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = DeepSeekClient(make_config({"model": {"network_retries": 2, "retry_base_seconds": 0}}))
    calls = 0

    def fail_transport(request, timeout):
        nonlocal calls
        calls += 1
        raise ssl.SSLError("record layer failure")

    monkeypatch.setattr("urllib.request.urlopen", fail_transport)

    with pytest.raises(RuntimeError, match=r"failed after 3 attempt\(s\)") as error:
        client.chat(messages=[{"role": "user", "content": "test"}])

    assert calls == 3
    assert error.value.http_attempt_count == 3


def test_non_stream_invalid_json_is_wrapped_with_attempt_count(monkeypatch, make_config) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = DeepSeekClient(make_config({"model": {"network_retries": 3, "retry_base_seconds": 0}}))
    calls = 0

    class InvalidJsonResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return b"{invalid-json"

    def invalid_json(request, timeout):
        nonlocal calls
        calls += 1
        return InvalidJsonResponse()

    monkeypatch.setattr("urllib.request.urlopen", invalid_json)

    with pytest.raises(RuntimeError, match="Expecting property name") as error:
        client.chat(messages=[{"role": "user", "content": "test"}])

    assert calls == 1
    assert error.value.http_attempt_count == 1
    assert isinstance(error.value.__cause__, json.JSONDecodeError)
    assert str(error.value) == str(error.value.__cause__)


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_non_stream_does_not_wrap_process_control_exceptions(monkeypatch, make_config, exception_type) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = DeepSeekClient(make_config())

    def interrupt(request, timeout):
        raise exception_type("stop now")

    monkeypatch.setattr("urllib.request.urlopen", interrupt)

    with pytest.raises(exception_type, match="stop now"):
        client.chat(messages=[{"role": "user", "content": "test"}])


@pytest.mark.parametrize("status", [400, 413])
def test_context_overflow_http_error_is_typed_bounded_and_redacted(monkeypatch, make_config, status) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = DeepSeekClient(make_config())
    secret = "sk-should-never-escape"
    body = json.dumps(
        {
            "error": {
                "token": secret,
                "message": "Maximum context length exceeded for this request " + ("x" * 8_000),
            }
        }
    ).encode("utf-8")

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: (_ for _ in ()).throw(http_error(request.full_url, status, body)),
    )

    with pytest.raises(DeepSeekContextOverflow) as error:
        client.chat(messages=[{"role": "user", "content": "too large"}])

    assert error.value.http_attempt_count == 1
    assert secret not in str(error.value)
    assert "[truncated]" in str(error.value)
    assert len(str(error.value)) < 4_300


def test_context_overflow_structured_error_code_is_typed(monkeypatch, make_config) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = DeepSeekClient(make_config())
    body = b'{"error":{"code":"context_length_exceeded","message":"invalid input"}}'
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: (_ for _ in ()).throw(http_error(request.full_url, 400, body)),
    )

    with pytest.raises(DeepSeekContextOverflow):
        client.chat(messages=[{"role": "user", "content": "too large"}])


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (400, b'{"error":{"message":"invalid request body"}}'),
        (413, b'{"error":{"message":"uploaded file too large"}}'),
        (422, b'{"error":{"message":"maximum context length exceeded"}}'),
    ],
)
def test_unrelated_http_errors_are_not_misclassified_as_context_overflow(
    monkeypatch, make_config, status, body
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = DeepSeekClient(make_config())
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: (_ for _ in ()).throw(http_error(request.full_url, status, body)),
    )

    with pytest.raises(RuntimeError) as error:
        client.chat(messages=[{"role": "user", "content": "bad request"}])

    assert not isinstance(error.value, DeepSeekContextOverflow)


@pytest.mark.parametrize(
    "failure",
    [
        ssl.SSLError("[SSL: RECORD_LAYER_FAILURE] record layer failure (_ssl.c:2711)"),
        http.client.IncompleteRead(b"partial", 100),
        http.client.RemoteDisconnected("remote end closed connection"),
        ConnectionResetError("connection reset by peer"),
        BrokenPipeError("broken pipe"),
        OSError("TLS connection aborted temporarily"),
    ],
)
def test_non_stream_transport_failures_are_bounded_and_retryable(monkeypatch, make_config, failure) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = DeepSeekClient(make_config({"model": {"network_retries": 1, "retry_base_seconds": 0}}))
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise failure
        return FakeResponse({"choices": [{"message": {"role": "assistant", "content": "OK"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert client.chat(messages=[{"role": "user", "content": "test"}]).message["content"] == "OK"
    assert calls == 2


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


def test_stream_response_exposes_finish_reason_usage_and_http_attempts(monkeypatch, make_config) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = DeepSeekClient(make_config({"model": {"network_retries": 1, "retry_base_seconds": 0}}))
    calls = 0
    sent_payload: dict = {}
    lines = [
        b'data: {"choices":[{"delta":{"content":"done"},"finish_reason":null}]}\n',
        b'data: {"choices":[{"delta":{},"finish_reason":"length"}],'
        b'"usage":{"prompt_tokens":9,"completion_tokens":4,"total_tokens":13}}\n',
        b"data: [DONE]\n",
    ]

    class StreamResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def __iter__(self):
            return iter(lines)

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        sent_payload.update(json.loads(request.data.decode("utf-8")))
        if calls == 1:
            raise ConnectionResetError("connection reset before headers")
        return StreamResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    response = client.chat_stream(messages=[{"role": "user", "content": "test"}])

    assert response.message["content"] == "done"
    assert response.finish_reason == "length"
    assert response.usage == {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13}
    assert response.http_attempt_count == 2
    assert sent_payload["stream_options"] == {"include_usage": True}


def test_stream_context_overflow_is_typed_and_never_falls_back(monkeypatch, make_config) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = DeepSeekClient(make_config({"model": {"network_retries": 2, "retry_base_seconds": 0}}))
    calls = 0
    body = b'{"error":{"message":"context_length_exceeded: request has too many tokens"}}'

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        raise http_error(request.full_url, 400, body)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(DeepSeekContextOverflow) as error:
        client.chat_stream(messages=[{"role": "user", "content": "too large"}])

    assert error.value.http_attempt_count == 1
    assert calls == 1


def test_stream_non_transient_pre_delta_failure_is_wrapped_with_attempt_count(monkeypatch, make_config) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = DeepSeekClient(make_config({"model": {"network_retries": 3, "retry_base_seconds": 0}}))
    calls = 0

    class BrokenStream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def __iter__(self):
            raise LookupError("unexpected pre-delta stream failure")

    def broken_stream(request, timeout):
        nonlocal calls
        calls += 1
        return BrokenStream()

    monkeypatch.setattr("urllib.request.urlopen", broken_stream)

    with pytest.raises(RuntimeError, match="unexpected pre-delta stream failure") as error:
        client._request_stream_with_key_pool({}, on_reasoning=None, on_content=None)

    assert calls == 1
    assert error.value.http_attempt_count == 1
    assert isinstance(error.value.__cause__, LookupError)
    assert str(error.value) == str(error.value.__cause__)


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


def test_record_layer_failure_before_first_stream_data_retries_stream(monkeypatch, make_config) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = DeepSeekClient(make_config({"model": {"network_retries": 1, "retry_base_seconds": 0}}))
    calls = 0

    class HealthyStream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"reasoning_content":"recovered "}}]}\n'
            yield b'data: {"choices":[{"delta":{"content":"done"}}]}\n'
            yield b"data: [DONE]\n"

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ssl.SSLError("[SSL: RECORD_LAYER_FAILURE] record layer failure (_ssl.c:2711)")
        return HealthyStream()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    reasoning: list[str] = []

    response = client.chat_stream(
        messages=[{"role": "user", "content": "hard task"}],
        thinking=True,
        on_reasoning=reasoning.append,
    )

    assert calls == 2
    assert reasoning == ["recovered "]
    assert response.message["content"] == "done"


def test_invalid_sse_before_first_valid_delta_retries_stream(monkeypatch, make_config) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = DeepSeekClient(make_config({"model": {"network_retries": 1, "retry_base_seconds": 0}}))
    calls = 0

    class Stream:
        def __init__(self, lines):
            self.lines = lines

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def __iter__(self):
            return iter(self.lines)

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            return Stream([b"data: {invalid-json}\n"])
        return Stream([b'data: {"choices":[{"delta":{"content":"recovered"}}]}\n', b"data: [DONE]\n"])

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    response = client.chat_stream(messages=[{"role": "user", "content": "test"}], thinking=True)

    assert calls == 2
    assert response.message["content"] == "recovered"


def test_record_layer_failure_before_first_data_can_fallback_to_non_stream(monkeypatch, make_config) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = DeepSeekClient(make_config({"model": {"network_retries": 1, "retry_base_seconds": 0}}))
    stream_calls = 0
    non_stream_calls = 0

    def fake_urlopen(request, timeout):
        nonlocal stream_calls, non_stream_calls
        payload = json.loads(request.data.decode("utf-8"))
        if payload.get("stream"):
            stream_calls += 1
            raise ssl.SSLError("[SSL: RECORD_LAYER_FAILURE] record layer failure (_ssl.c:2711)")
        non_stream_calls += 1
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "fallback complete",
                            "reasoning_content": "bounded fallback",
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    response = client.chat_stream(
        messages=[{"role": "user", "content": "hard task"}],
        thinking=True,
    )

    assert stream_calls == 2
    assert non_stream_calls == 1
    assert response.message["content"] == "fallback complete"
    assert response.http_attempt_count == 3


def test_stream_auth_key_rotation_attempts_are_included_in_fallback_count(monkeypatch, make_config) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "first, second")
    client = DeepSeekClient(
        make_config({"model": {"api_key": "first, second", "network_retries": 0, "retry_base_seconds": 0}})
    )
    stream_calls = 0
    non_stream_calls = 0

    def fake_urlopen(request, timeout):
        nonlocal stream_calls, non_stream_calls
        payload = json.loads(request.data.decode("utf-8"))
        if payload.get("stream"):
            stream_calls += 1
            raise http_error(request.full_url, 401)
        non_stream_calls += 1
        return FakeResponse({"choices": [{"message": {"role": "assistant", "content": "fallback"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    response = client.chat_stream(messages=[{"role": "user", "content": "test"}])

    assert stream_calls == 2
    assert non_stream_calls == 1
    assert response.http_attempt_count == 3


def test_stream_fallback_final_failure_accumulates_both_request_paths(monkeypatch, make_config) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = DeepSeekClient(make_config({"model": {"network_retries": 1, "retry_base_seconds": 0}}))
    stream_calls = 0
    non_stream_calls = 0

    def fake_urlopen(request, timeout):
        nonlocal stream_calls, non_stream_calls
        payload = json.loads(request.data.decode("utf-8"))
        if payload.get("stream"):
            stream_calls += 1
            raise ConnectionResetError("stream reset before first delta")
        non_stream_calls += 1
        raise http_error(request.full_url, 503, b'{"error":{"message":"temporarily unavailable"}}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="DeepSeek API HTTP 503") as error:
        client.chat_stream(messages=[{"role": "user", "content": "test"}])

    assert stream_calls == 2
    assert non_stream_calls == 2
    assert error.value.http_attempt_count == 4


def test_stream_fallback_overflow_accumulates_both_request_paths(monkeypatch, make_config) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = DeepSeekClient(make_config({"model": {"network_retries": 1, "retry_base_seconds": 0}}))
    stream_calls = 0
    non_stream_calls = 0

    def fake_urlopen(request, timeout):
        nonlocal stream_calls, non_stream_calls
        payload = json.loads(request.data.decode("utf-8"))
        if payload.get("stream"):
            stream_calls += 1
            raise ConnectionResetError("stream reset before first delta")
        non_stream_calls += 1
        body = b'{"error":{"code":"context_length_exceeded","message":"too many tokens"}}'
        raise http_error(request.full_url, 400, body)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(DeepSeekContextOverflow) as error:
        client.chat_stream(messages=[{"role": "user", "content": "too large"}])

    assert stream_calls == 2
    assert non_stream_calls == 1
    assert error.value.http_attempt_count == 3


def test_record_layer_failure_after_reasoning_is_resumable_and_never_replayed(monkeypatch, make_config) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = DeepSeekClient(make_config({"model": {"network_retries": 3, "retry_base_seconds": 0}}))
    calls = 0
    chunks: list[str] = []

    class BrokenStream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"reasoning_content":"partial reasoning"}}]}\n'
            raise ssl.SSLError("[SSL: RECORD_LAYER_FAILURE] record layer failure (_ssl.c:2711)")

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        return BrokenStream()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(DeepSeekStreamInterrupted, match="resume the saved Session") as error:
        client.chat_stream(
            messages=[{"role": "user", "content": "hard task"}],
            thinking=True,
            on_reasoning=chunks.append,
        )

    assert chunks == ["partial reasoning"]
    assert calls == 1
    assert error.value.http_attempt_count == 1


def test_valid_delta_after_a_pre_delta_retry_is_never_replayed(monkeypatch, make_config) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = DeepSeekClient(make_config({"model": {"network_retries": 3, "retry_base_seconds": 0}}))
    calls = 0

    class BrokenAfterDelta:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"content":"started"}}]}\n'
            raise ConnectionResetError("connection reset after valid delta")

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionResetError("connection reset before first delta")
        return BrokenAfterDelta()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(DeepSeekStreamInterrupted, match="partial output") as error:
        client.chat_stream(messages=[{"role": "user", "content": "test"}])

    assert calls == 2
    assert error.value.http_attempt_count == 2


def test_partial_tool_call_delta_marks_stream_non_replayable(monkeypatch, make_config) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = DeepSeekClient(make_config({"model": {"network_retries": 2, "retry_base_seconds": 0}}))
    calls = 0

    class BrokenToolStream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def __iter__(self):
            yield (
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1",'
                b'"function":{"name":"shell_","arguments":"{"}}]}}]}\n'
            )
            raise ConnectionResetError("connection reset")

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        return BrokenToolStream()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(DeepSeekStreamInterrupted, match="partial output"):
        client.chat_stream(messages=[{"role": "user", "content": "run something"}], thinking=True)
    assert calls == 1


def test_empty_finish_chunk_does_not_make_later_transport_failure_non_replayable(monkeypatch, make_config) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = DeepSeekClient(make_config({"model": {"network_retries": 1, "retry_base_seconds": 0}}))
    calls = 0

    class Stream:
        def __init__(self, lines, failure=None):
            self.lines = lines
            self.failure = failure

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def __iter__(self):
            yield from self.lines
            if self.failure:
                raise self.failure

    def fake_urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            return Stream(
                [b'data: {"choices":[{"delta":{},"finish_reason":null}]}\n'],
                ConnectionResetError("connection reset"),
            )
        return Stream([b'data: {"choices":[{"delta":{"content":"recovered"},"finish_reason":"stop"}]}\n'])

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    response = client.chat_stream(messages=[{"role": "user", "content": "test"}])

    assert response.message["content"] == "recovered"
    assert response.finish_reason == "stop"
    assert response.http_attempt_count == 2
