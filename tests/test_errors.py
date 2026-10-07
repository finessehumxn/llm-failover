import asyncio

import pytest

from llm_failover import ErrorClass, ProviderError, classify, classify_status
from llm_failover.errors import scrub


@pytest.mark.parametrize(
    "status, expected",
    [
        (400, ErrorClass.NON_RETRYABLE),
        (413, ErrorClass.NON_RETRYABLE),
        (422, ErrorClass.NON_RETRYABLE),
        (418, ErrorClass.NON_RETRYABLE),
        (401, ErrorClass.FATAL_CONFIG),
        (402, ErrorClass.FATAL_CONFIG),
        (403, ErrorClass.FATAL_CONFIG),
        (404, ErrorClass.FATAL_CONFIG),
        (408, ErrorClass.RETRYABLE),
        (409, ErrorClass.RETRYABLE),
        (429, ErrorClass.RETRYABLE),
        (500, ErrorClass.RETRYABLE),
        (502, ErrorClass.RETRYABLE),
        (503, ErrorClass.RETRYABLE),
        (529, ErrorClass.RETRYABLE),
        (302, ErrorClass.UNKNOWN),
    ],
)
def test_classify_status(status, expected):
    assert classify_status(status) is expected


def test_provider_error_from_status_carries_fields():
    e = ProviderError.from_status(429, "slow down", retry_after=2.5)
    c = classify(e)
    assert c.error_class is ErrorClass.RETRYABLE
    assert c.status_code == 429
    assert c.retry_after == 2.5
    assert "slow down" in c.message


def test_timeouts_and_connection_errors_are_retryable():
    assert classify(asyncio.TimeoutError()).error_class is ErrorClass.RETRYABLE
    assert classify(TimeoutError()).error_class is ErrorClass.RETRYABLE
    assert classify(ConnectionResetError("reset")).error_class is ErrorClass.RETRYABLE


def test_unrecognised_exceptions_are_unknown_not_retryable():
    c = classify(KeyError("choices"))
    assert c.error_class is ErrorClass.UNKNOWN
    assert "KeyError" in c.message


def test_os_errors_that_are_not_connection_errors_are_unknown():
    assert classify(FileNotFoundError("x")).error_class is ErrorClass.UNKNOWN


def test_scrub_removes_credential_shaped_strings():
    text = "invalid key sk-ant-api03-abcdefghijklmnop and token " + "a" * 40
    out = scrub(text)
    assert "sk-ant" not in out
    assert "a" * 40 not in out
    assert out.count("[redacted]") == 2
    assert "invalid key" in out


def test_classify_scrubs_messages():
    c = classify(ProviderError.from_status(401, "bad key sk-proj-1234567890abcdef"))
    assert "1234567890abcdef" not in c.message
