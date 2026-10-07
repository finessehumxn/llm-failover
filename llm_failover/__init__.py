"""Multi-provider LLM routing with fallback that never hides a configuration failure."""

from .backoff import Backoff
from .breaker import BreakerState, CircuitBreaker
from .errors import (
    AllProvidersFailed,
    ConfigurationError,
    DeadlineExceeded,
    ErrorClass,
    ProviderError,
    RequestRejected,
    RouterError,
    classify,
    classify_status,
)
from .events import CollectingHook, Event, EventType, Hook, LoggingHook, Outcome
from .providers import FakeProvider, Provider
from .router import Route, Router, RouterResult
from .types import Message, Request, Response, Usage

__version__ = "0.1.0"

__all__ = [
    "AllProvidersFailed",
    "AnthropicProvider",
    "Backoff",
    "BreakerState",
    "CircuitBreaker",
    "CollectingHook",
    "ConfigurationError",
    "DeadlineExceeded",
    "ErrorClass",
    "Event",
    "EventType",
    "FakeProvider",
    "Hook",
    "LoggingHook",
    "Message",
    "OpenAIProvider",
    "Outcome",
    "Provider",
    "ProviderError",
    "Request",
    "RequestRejected",
    "Response",
    "Route",
    "Router",
    "RouterError",
    "RouterResult",
    "Usage",
    "classify",
    "classify_status",
]


def __getattr__(name: str):
    if name in ("AnthropicProvider", "OpenAIProvider"):
        from . import providers

        return getattr(providers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
