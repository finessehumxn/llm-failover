from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..errors import ErrorClass, ProviderError, classify_status, scrub
from ..types import Request, Response


@runtime_checkable
class Provider(Protocol):
    """Anything with a name and an async ``complete``.

    Implementations should raise :class:`ProviderError` with a classification
    when they can. Anything else they raise is classified by the router
    (timeouts and connection errors as retryable, the rest as unknown).
    """

    name: str

    async def complete(self, request: Request) -> Response: ...


def _retry_after(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None  # HTTP-date form; not worth parsing for a hint


def map_sdk_exception(exc: BaseException, sdk: Any) -> BaseException:
    """Translate an exception from an OpenAI-style SDK into a ProviderError.

    The ``anthropic`` and ``openai`` Python SDKs share a hierarchy
    (``APIStatusError`` with ``status_code``, ``APIConnectionError`` with
    ``APITimeoutError`` beneath it), so one mapping serves both. Exceptions
    from outside the SDK are returned unchanged for the router to classify.
    """
    if isinstance(exc, sdk.APIStatusError):
        status = int(exc.status_code)
        return ProviderError(
            f"HTTP {status}: {scrub(str(exc), 200)}",
            classify_status(status),
            status_code=status,
            retry_after=_retry_after(exc),
        )
    if isinstance(exc, sdk.APIConnectionError):  # includes APITimeoutError
        return ProviderError(type(exc).__name__, ErrorClass.RETRYABLE)
    # Missing or unloadable credentials surface before any HTTP call. Matched
    # by name so a missing class in an older SDK version is not an ImportError.
    if type(exc).__name__ in {"CredentialsError", "WorkloadIdentityError", "IdentityTokenFileError", "OAuthError"}:
        return ProviderError(f"{type(exc).__name__}: {scrub(str(exc), 200)}", ErrorClass.FATAL_CONFIG)
    return exc
