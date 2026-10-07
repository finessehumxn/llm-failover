"""Error taxonomy.

The whole library turns on one question asked of every failure: *if I send
the same request again, to this provider or another one, could the answer be
different?*

* RETRYABLE      -- yes, probably. Timeouts, 429, 5xx, dropped connections.
                    Retry here with backoff, then fail over.
* FATAL_CONFIG   -- not from this provider, not until a human fixes something.
                    401/402/403, revoked key, unknown model. Fail over (another
                    provider has its own credentials) but say so loudly.
* NON_RETRYABLE  -- no. The request itself is malformed (400, 413, 422). Every
                    provider will reject it, so raise immediately instead of
                    spending the rest of the list on a guaranteed failure.
* UNKNOWN        -- an exception we cannot classify, usually a bug in an
                    adapter. Do not hammer the same provider; try the next one.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .events import Event


class ErrorClass(str, Enum):
    RETRYABLE = "retryable"
    FATAL_CONFIG = "fatal_config"
    NON_RETRYABLE = "non_retryable"
    UNKNOWN = "unknown"


_RETRYABLE_STATUS = {408, 409, 425, 429}
_CONFIG_STATUS = {401, 402, 403, 404}
_REQUEST_STATUS = {400, 413, 422}


def classify_status(status: int) -> ErrorClass:
    """Map an HTTP status from an LLM API to an error class.

    404 is treated as configuration, not a bad request: against a fixed
    chat-completions endpoint it nearly always means the configured model
    name does not exist (or the account cannot see it), which is a deploy
    problem, not something the caller's payload caused.
    """
    if status in _CONFIG_STATUS:
        return ErrorClass.FATAL_CONFIG
    if status in _RETRYABLE_STATUS or status >= 500:
        return ErrorClass.RETRYABLE
    if status in _REQUEST_STATUS or 400 <= status < 500:
        return ErrorClass.NON_RETRYABLE
    return ErrorClass.UNKNOWN


class ProviderError(Exception):
    """An error raised by an adapter, already classified.

    Adapters translate their SDK's exceptions into this so the router can
    make decisions without importing any SDK.
    """

    def __init__(
        self,
        message: str,
        error_class: ErrorClass,
        *,
        status_code: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.error_class = error_class
        self.status_code = status_code
        self.retry_after = retry_after

    @classmethod
    def from_status(
        cls, status: int, message: str = "", *, retry_after: float | None = None
    ) -> "ProviderError":
        return cls(
            message or f"HTTP {status}",
            classify_status(status),
            status_code=status,
            retry_after=retry_after,
        )


@dataclass(frozen=True)
class Classified:
    error_class: ErrorClass
    status_code: int | None
    retry_after: float | None
    message: str


def classify(exc: BaseException) -> Classified:
    """Classify any exception raised while calling a provider."""
    if isinstance(exc, ProviderError):
        return Classified(exc.error_class, exc.status_code, exc.retry_after, scrub(str(exc)))
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return Classified(ErrorClass.RETRYABLE, None, None, "timed out")
    if isinstance(exc, ConnectionError):
        return Classified(ErrorClass.RETRYABLE, None, None, scrub(str(exc)) or type(exc).__name__)
    return Classified(
        ErrorClass.UNKNOWN, None, None, f"{type(exc).__name__}: {scrub(str(exc))}"
    )


# Error messages are upstream text we do not control, and the one secret
# guaranteed to be near a provider call is its API key. Anything shaped like a
# credential is replaced before it reaches a log line or an event.
_CREDENTIAL_SHAPED = re.compile(r"(?:sk|rk|pk)-[A-Za-z0-9_\-]{8,}|[A-Za-z0-9_\-]{32,}")


def scrub(text: str, limit: int = 300) -> str:
    return _CREDENTIAL_SHAPED.sub("[redacted]", text or "")[:limit]


class RouterError(Exception):
    """Base class for errors raised by the router. Carries the attempt trail."""

    def __init__(self, message: str, trail: "list[Event]") -> None:
        super().__init__(message)
        self.trail = trail


class RequestRejected(RouterError):
    """A provider rejected the request itself (400-class). Not retried anywhere."""

    def __init__(self, message: str, trail: "list[Event]", *, provider: str, status_code: int | None):
        super().__init__(message, trail)
        self.provider = provider
        self.status_code = status_code


class AllProvidersFailed(RouterError):
    """Every provider was tried (or skipped) and none produced a response."""

    @property
    def config_errors(self) -> "list[Event]":
        from .events import EventType

        return [e for e in self.trail if e.type is EventType.CONFIG_ERROR]


class DeadlineExceeded(AllProvidersFailed):
    """The overall deadline ran out before any provider succeeded."""


class ConfigurationError(RouterError):
    """Raised instead of failing over when ``strict_config=True``."""

    def __init__(self, message: str, trail: "list[Event]", *, provider: str, status_code: int | None):
        super().__init__(message, trail)
        self.provider = provider
        self.status_code = status_code
