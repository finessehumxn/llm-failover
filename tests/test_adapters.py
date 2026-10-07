"""Adapter tests. No network: the SDK client is replaced with a stub, and SDK
exceptions are constructed directly from the installed SDK."""

from types import SimpleNamespace

import pytest

from llm_failover import ErrorClass, ProviderError, Request

anthropic = pytest.importorskip("anthropic")
openai = pytest.importorskip("openai")


def _http(sdk):
    # Both SDKs expose the HTTP library they were built on; use whichever it is.
    import importlib

    for mod in ("httpx2", "httpx"):
        try:
            return importlib.import_module(mod)
        except ImportError:
            continue
    raise RuntimeError("no httpx available")


def status_error(sdk, cls, status, headers=None):
    httpx = _http(sdk)
    request = httpx.Request("POST", "https://example.invalid/v1")
    response = httpx.Response(status, request=request, headers=headers or {})
    return cls(f"status {status}", response=response, body=None)


def connection_error(sdk, cls):
    httpx = _http(sdk)
    return cls(request=httpx.Request("POST", "https://example.invalid/v1"))


class Recorder:
    """Stands in for client.messages / client.chat.completions."""

    def __init__(self, result=None, exc=None):
        self.result, self.exc, self.kwargs = result, exc, None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        if self.exc is not None:
            raise self.exc
        return self.result


# -- Anthropic --------------------------------------------------------------------


def anthropic_with(result=None, exc=None):
    from llm_failover.providers.anthropic import AnthropicProvider

    rec = Recorder(result, exc)
    p = AnthropicProvider(client=SimpleNamespace(messages=rec))
    return p, rec


def anthropic_message(text="hello"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        model="claude-sonnet-5-5",
        usage=SimpleNamespace(input_tokens=11, output_tokens=3),
        stop_reason="end_turn",
    )


def test_anthropic_defaults():
    from llm_failover import AnthropicProvider

    p = AnthropicProvider(api_key="test-key-not-real")
    assert p.model == "claude-sonnet-5-5"
    assert p.name == "anthropic"
    assert p._client.max_retries == 0  # the router owns retries


async def test_anthropic_request_and_response_mapping():
    p, rec = anthropic_with(anthropic_message("hi there"))
    resp = await p.complete(Request.user("q", system="be brief", max_tokens=50))
    assert rec.kwargs == {
        "model": "claude-sonnet-5-5",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "q"}],
        "system": "be brief",
    }
    assert "temperature" not in rec.kwargs  # unset means provider default
    assert resp.text == "hi there"  # non-text blocks ignored
    assert resp.usage.input_tokens == 11 and resp.usage.output_tokens == 3
    assert resp.stop_reason == "end_turn"


async def test_anthropic_sends_temperature_only_when_set():
    p, rec = anthropic_with(anthropic_message())
    await p.complete(Request.user("q", temperature=0.2))
    assert rec.kwargs["temperature"] == 0.2


@pytest.mark.parametrize(
    "cls_name, status, expected",
    [
        ("AuthenticationError", 401, ErrorClass.FATAL_CONFIG),
        ("PermissionDeniedError", 403, ErrorClass.FATAL_CONFIG),
        ("NotFoundError", 404, ErrorClass.FATAL_CONFIG),
        ("BadRequestError", 400, ErrorClass.NON_RETRYABLE),
        ("UnprocessableEntityError", 422, ErrorClass.NON_RETRYABLE),
        ("RateLimitError", 429, ErrorClass.RETRYABLE),
        ("InternalServerError", 500, ErrorClass.RETRYABLE),
        ("APIStatusError", 529, ErrorClass.RETRYABLE),
    ],
)
async def test_anthropic_status_errors_are_classified(cls_name, status, expected):
    exc = status_error(anthropic, getattr(anthropic, cls_name), status)
    p, _ = anthropic_with(exc=exc)
    with pytest.raises(ProviderError) as ei:
        await p.complete(Request.user("q"))
    assert ei.value.error_class is expected
    assert ei.value.status_code == status
    assert ei.value.__cause__ is exc


async def test_anthropic_retry_after_header_is_read():
    exc = status_error(anthropic, anthropic.RateLimitError, 429, headers={"retry-after": "7"})
    p, _ = anthropic_with(exc=exc)
    with pytest.raises(ProviderError) as ei:
        await p.complete(Request.user("q"))
    assert ei.value.retry_after == 7.0


@pytest.mark.parametrize("cls_name", ["APIConnectionError", "APITimeoutError"])
async def test_anthropic_connection_errors_are_retryable(cls_name):
    p, _ = anthropic_with(exc=connection_error(anthropic, getattr(anthropic, cls_name)))
    with pytest.raises(ProviderError) as ei:
        await p.complete(Request.user("q"))
    assert ei.value.error_class is ErrorClass.RETRYABLE


async def test_anthropic_non_sdk_exceptions_pass_through():
    p, _ = anthropic_with(exc=ValueError("adapter bug"))
    with pytest.raises(ValueError):
        await p.complete(Request.user("q"))


# -- OpenAI ------------------------------------------------------------------------


def openai_with(result=None, exc=None, model="test-model"):
    from llm_failover.providers.openai import OpenAIProvider

    rec = Recorder(result, exc)
    p = OpenAIProvider(model=model, client=SimpleNamespace(chat=SimpleNamespace(completions=rec)))
    return p, rec


def openai_completion(text="hello"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")],
        model="test-model",
        usage=SimpleNamespace(prompt_tokens=9, completion_tokens=2),
    )


def test_openai_requires_a_model_and_disables_sdk_retries():
    from llm_failover import OpenAIProvider

    with pytest.raises(TypeError):
        OpenAIProvider(api_key="x")  # type: ignore[call-arg]
    p = OpenAIProvider(model="m", api_key="test-key-not-real")
    assert p._client.max_retries == 0


async def test_openai_request_and_response_mapping():
    p, rec = openai_with(openai_completion("yo"))
    resp = await p.complete(Request.user("q", system="sys", max_tokens=20))
    assert rec.kwargs == {
        "model": "test-model",
        "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}],
        "max_completion_tokens": 20,
    }
    assert resp.text == "yo"
    assert resp.usage.input_tokens == 9 and resp.usage.output_tokens == 2
    assert resp.stop_reason == "stop"


async def test_openai_none_content_becomes_empty_string():
    p, _ = openai_with(openai_completion(None))
    assert (await p.complete(Request.user("q"))).text == ""


@pytest.mark.parametrize(
    "cls_name, status, expected",
    [
        ("AuthenticationError", 401, ErrorClass.FATAL_CONFIG),
        ("PermissionDeniedError", 403, ErrorClass.FATAL_CONFIG),
        ("NotFoundError", 404, ErrorClass.FATAL_CONFIG),
        ("BadRequestError", 400, ErrorClass.NON_RETRYABLE),
        ("RateLimitError", 429, ErrorClass.RETRYABLE),
        ("InternalServerError", 503, ErrorClass.RETRYABLE),
    ],
)
async def test_openai_status_errors_are_classified(cls_name, status, expected):
    exc = status_error(openai, getattr(openai, cls_name), status)
    p, _ = openai_with(exc=exc)
    with pytest.raises(ProviderError) as ei:
        await p.complete(Request.user("q"))
    assert ei.value.error_class is expected


async def test_openai_timeout_is_retryable():
    p, _ = openai_with(exc=connection_error(openai, openai.APITimeoutError))
    with pytest.raises(ProviderError) as ei:
        await p.complete(Request.user("q"))
    assert ei.value.error_class is ErrorClass.RETRYABLE


async def test_error_messages_do_not_leak_keys():
    httpx = _http(anthropic)
    request = httpx.Request("POST", "https://example.invalid/v1")
    response = httpx.Response(401, request=request)
    exc = anthropic.AuthenticationError("invalid x-api-key sk-ant-api03-SECRETSECRETSECRET", response=response, body=None)
    p, _ = anthropic_with(exc=exc)
    with pytest.raises(ProviderError) as ei:
        await p.complete(Request.user("q"))
    assert "SECRET" not in str(ei.value)
