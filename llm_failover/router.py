"""The router: ordered providers, classified failures, honest fallback."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Iterable, Sequence

from .backoff import Backoff
from .breaker import BreakerState, CircuitBreaker
from .errors import (
    AllProvidersFailed,
    ConfigurationError,
    DeadlineExceeded,
    ErrorClass,
    RequestRejected,
    classify,
)
from .events import Event, EventType, Hook, LoggingHook, Outcome
from .providers.base import Provider
from .types import Request, Response

logger = logging.getLogger("llm_failover")

Sleep = Callable[[float], Awaitable[None]]
Clock = Callable[[], float]


@dataclass
class Route:
    """A provider plus its per-provider settings. ``None`` means "use the router default"."""

    provider: Provider
    timeout: float | None = None
    max_attempts: int | None = None
    breaker: CircuitBreaker | None = None


@dataclass
class _Slot:
    provider: Provider
    timeout: float | None
    max_attempts: int
    breaker: CircuitBreaker

    @property
    def name(self) -> str:
        return self.provider.name


@dataclass
class RouterResult:
    response: Response
    trail: list[Event] = field(default_factory=list)
    primary: str = ""

    @property
    def provider(self) -> str:
        return self.response.provider

    @property
    def fell_back(self) -> bool:
        """True if the answer came from anyone other than the first-listed provider."""
        return self.response.provider != self.primary

    @property
    def attempts(self) -> list[Event]:
        return [e for e in self.trail if e.type is EventType.ATTEMPT]

    @property
    def config_errors(self) -> list[Event]:
        return [e for e in self.trail if e.type is EventType.CONFIG_ERROR]


class Router:
    """Try providers in order until one answers.

    Per provider: up to ``max_attempts`` tries, each bounded by ``timeout``,
    separated by full-jitter backoff, gated by a circuit breaker. Across all
    providers: an optional ``deadline`` in seconds.

    ``strict_config=True`` raises :class:`ConfigurationError` on the first
    auth/permission/model-not-found failure instead of failing over. The
    default fails over (so users are served) but emits a ``config_error``
    event at ERROR level and counts it on :attr:`config_error_counts`.
    """

    def __init__(
        self,
        providers: Sequence[Provider | Route],
        *,
        timeout: float | None = 30.0,
        max_attempts: int = 2,
        deadline: float | None = None,
        backoff: Backoff | None = None,
        failure_threshold: int = 5,
        cooldown: float = 30.0,
        strict_config: bool = False,
        hooks: Iterable[Hook] | None = None,
        sleep: Sleep = asyncio.sleep,
        clock: Clock = time.monotonic,
    ) -> None:
        if not providers:
            raise ValueError("Router needs at least one provider")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._clock = clock
        self._sleep = sleep
        self.backoff = backoff or Backoff()
        self.deadline = deadline
        self.strict_config = strict_config
        self.hooks: list[Hook] = [LoggingHook()] if hooks is None else list(hooks)
        self.config_error_counts: Counter[str] = Counter()

        self._slots: list[_Slot] = []
        seen: set[str] = set()
        for item in providers:
            route = item if isinstance(item, Route) else Route(item)
            name = route.provider.name
            if name in seen:
                raise ValueError(f"duplicate provider name {name!r}; names identify breakers and events")
            seen.add(name)
            self._slots.append(
                _Slot(
                    provider=route.provider,
                    timeout=route.timeout if route.timeout is not None else timeout,
                    max_attempts=route.max_attempts or max_attempts,
                    breaker=route.breaker
                    or CircuitBreaker(failure_threshold=failure_threshold, cooldown=cooldown, clock=clock),
                )
            )

    # -- introspection -------------------------------------------------------

    @property
    def provider_names(self) -> list[str]:
        return [s.name for s in self._slots]

    def breaker(self, name: str) -> CircuitBreaker:
        for s in self._slots:
            if s.name == name:
                return s.breaker
        raise KeyError(name)

    def health(self) -> dict[str, dict[str, object]]:
        return {
            s.name: {
                "circuit": s.breaker.state.value,
                "consecutive_failures": s.breaker.consecutive_failures,
                "config_errors": self.config_error_counts[s.name],
            }
            for s in self._slots
        }

    # -- the work ------------------------------------------------------------

    async def complete(self, request: Request, *, deadline: float | None = None) -> RouterResult:
        budget = deadline if deadline is not None else self.deadline
        deadline_at = self._clock() + budget if budget is not None else None
        trail: list[Event] = []
        primary = self._slots[0].name

        def remaining() -> float | None:
            return None if deadline_at is None else deadline_at - self._clock()

        def deadline_exceeded() -> DeadlineExceeded:
            return DeadlineExceeded(
                f"deadline of {budget}s exhausted after {self._count_attempts(trail)} attempt(s)", trail
            )

        for slot in self._slots:
            for attempt in range(1, slot.max_attempts + 1):
                left = remaining()
                if left is not None and left <= 0:
                    raise deadline_exceeded()

                if not slot.breaker.allow():
                    if attempt == 1:
                        self._emit(trail, Event(
                            EventType.SKIPPED, slot.name,
                            detail=f"circuit {slot.breaker.state.value}",
                        ))
                    break

                # The attempt timeout is the provider's own, cut short by the
                # deadline if the deadline is closer. Remember which one bound
                # it: a timeout caused by our deadline says nothing about the
                # provider's health and must not count against its breaker.
                timeout = slot.timeout
                deadline_bound = False
                if left is not None and (timeout is None or left < timeout):
                    timeout, deadline_bound = left, True

                started = self._clock()
                try:
                    if timeout is None:
                        response = await slot.provider.complete(request)
                    else:
                        response = await asyncio.wait_for(slot.provider.complete(request), timeout)
                except asyncio.CancelledError:
                    slot.breaker.release()
                    raise
                except Exception as exc:
                    latency = self._ms_since(started)
                    c = classify(exc)
                    timed_out = isinstance(exc, (asyncio.TimeoutError, TimeoutError))
                    self._emit(trail, Event(
                        EventType.ATTEMPT, slot.name, attempt=attempt,
                        outcome=Outcome.TIMEOUT if timed_out else Outcome.ERROR,
                        error_class=c.error_class, error=c.message,
                        status_code=c.status_code, latency_ms=latency,
                    ))

                    if c.error_class is ErrorClass.NON_RETRYABLE:
                        slot.breaker.release()
                        raise RequestRejected(
                            f"{slot.name} rejected the request ({c.message}); "
                            "not retried because every provider would reject it",
                            trail, provider=slot.name, status_code=c.status_code,
                        ) from exc

                    if c.error_class is ErrorClass.FATAL_CONFIG:
                        self.config_error_counts[slot.name] += 1
                        self._emit(trail, Event(
                            EventType.CONFIG_ERROR, slot.name, attempt=attempt,
                            error_class=c.error_class, error=c.message, status_code=c.status_code,
                            detail=f"{slot.name} refused our configuration "
                                   f"(count={self.config_error_counts[slot.name]}); "
                                   "this will not fix itself",
                        ))
                        if slot.breaker.trip():
                            self._circuit_opened(trail, slot, "config error")
                        if self.strict_config:
                            raise ConfigurationError(
                                f"{slot.name} configuration rejected: {c.message}",
                                trail, provider=slot.name, status_code=c.status_code,
                            ) from exc
                        break  # fail over; retrying the same key is pointless

                    if timed_out and deadline_bound:
                        slot.breaker.release()
                        raise deadline_exceeded() from exc

                    if slot.breaker.record_failure():
                        self._circuit_opened(trail, slot, f"{slot.breaker.failure_threshold} consecutive failures")

                    if c.error_class is ErrorClass.UNKNOWN or attempt == slot.max_attempts:
                        break

                    # RETRYABLE with attempts left: back off, unless the wait
                    # is pointless (server asked for longer than we would ever
                    # wait, or the deadline would pass while sleeping).
                    delay = self.backoff.delay(attempt)
                    if c.retry_after is not None:
                        if c.retry_after > self.backoff.max_delay:
                            break
                        delay = max(delay, c.retry_after)
                    left = remaining()
                    if left is not None and delay >= left:
                        break
                    await self._sleep(delay)
                    continue
                else:
                    latency = self._ms_since(started)
                    slot.breaker.record_success()
                    response = dataclasses.replace(response, provider=slot.name, latency_ms=latency)
                    self._emit(trail, Event(
                        EventType.ATTEMPT, slot.name, attempt=attempt,
                        outcome=Outcome.OK, latency_ms=latency,
                    ))
                    return RouterResult(response=response, trail=trail, primary=primary)

        left = remaining()
        if left is not None and left <= 0:
            raise deadline_exceeded()
        n = self._count_attempts(trail)
        configs = sum(1 for e in trail if e.type is EventType.CONFIG_ERROR)
        msg = f"all {len(self._slots)} provider(s) failed after {n} attempt(s)"
        if n == 0:
            msg = f"all {len(self._slots)} provider(s) skipped: every circuit is open"
        if configs:
            msg += f"; {configs} configuration error(s) -- check credentials and model names"
        raise AllProvidersFailed(msg, trail)

    # -- helpers -------------------------------------------------------------

    def _circuit_opened(self, trail: list[Event], slot: _Slot, why: str) -> None:
        self._emit(trail, Event(
            EventType.CIRCUIT_OPENED, slot.name,
            detail=f"{why}; cooling down {slot.breaker.cooldown}s",
        ))

    def _emit(self, trail: list[Event], event: Event) -> None:
        trail.append(event)
        for hook in self.hooks:
            try:
                hook(event)
            except Exception:  # a broken hook must never break routing
                logger.exception("llm_failover hook %r raised", hook)

    def _ms_since(self, started: float) -> float:
        return round((self._clock() - started) * 1000.0, 3)

    @staticmethod
    def _count_attempts(trail: list[Event]) -> int:
        return sum(1 for e in trail if e.type is EventType.ATTEMPT)


__all__ = ["Route", "Router", "RouterResult", "BreakerState"]
