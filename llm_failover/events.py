"""Structured events and the hooks that receive them.

Every decision the router makes produces an event: each attempt, each
configuration failure, each circuit transition, each skipped provider. The
same events are returned to the caller on ``RouterResult.trail`` and on the
``trail`` of any exception, so "did we fall back, and why?" always has an
answer.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable

from .errors import ErrorClass

logger = logging.getLogger("llm_failover")


class EventType(str, Enum):
    ATTEMPT = "attempt"
    CONFIG_ERROR = "config_error"
    CIRCUIT_OPENED = "circuit_opened"
    SKIPPED = "skipped"


class Outcome(str, Enum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class Event:
    type: EventType
    provider: str
    attempt: int | None = None
    outcome: Outcome | None = None
    error_class: ErrorClass | None = None
    error: str | None = None
    status_code: int | None = None
    latency_ms: float | None = None
    detail: str | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
        return {k: v for k, v in d.items() if v is not None}


Hook = Callable[[Event], None]


class LoggingHook:
    """Default hook: one log line per event, with the event dict attached.

    Levels are chosen so that a healthy system is quiet and a misconfigured
    one is not: successes are DEBUG, transient failures WARNING, and
    configuration errors ERROR -- the level most alerting is wired to.
    """

    def __init__(self, log: logging.Logger | None = None) -> None:
        self.log = log or logger

    def __call__(self, event: Event) -> None:
        level = self._level(event)
        fields = " ".join(f"{k}={v}" for k, v in event.to_dict().items() if k != "timestamp")
        self.log.log(level, "llm_failover %s", fields, extra={"llm_failover": event.to_dict()})

    @staticmethod
    def _level(event: Event) -> int:
        if event.type is EventType.CONFIG_ERROR:
            return logging.ERROR
        if event.type is EventType.CIRCUIT_OPENED:
            return logging.WARNING
        if event.type is EventType.SKIPPED:
            return logging.INFO
        if event.outcome is Outcome.OK:
            return logging.DEBUG
        return logging.WARNING


class CollectingHook:
    """Keeps every event in a list. Useful in tests and in the demo."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def __call__(self, event: Event) -> None:
        self.events.append(event)
