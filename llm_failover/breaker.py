"""A per-provider circuit breaker.

closed     -- requests flow. Consecutive failures are counted.
open       -- requests are refused without calling the provider, until the
              cool-down has elapsed.
half_open  -- one probe request is let through. Success closes the circuit,
              failure re-opens it for another full cool-down.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Callable


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if cooldown < 0:
            raise ValueError("cooldown must be >= 0")
        self.failure_threshold = failure_threshold
        self.cooldown = cooldown
        self._clock = clock
        self._state = BreakerState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._probe_in_flight = False

    @property
    def state(self) -> BreakerState:
        """Current state, promoting open -> half_open once the cool-down is over."""
        if self._state is BreakerState.OPEN and self._clock() - self._opened_at >= self.cooldown:
            self._state = BreakerState.HALF_OPEN
            self._probe_in_flight = False
        return self._state

    @property
    def consecutive_failures(self) -> int:
        return self._failures

    def allow(self) -> bool:
        """May a request go to this provider right now?

        In half-open, exactly one caller gets ``True`` until that probe
        reports back; concurrent callers are refused rather than stampeding
        a provider that has only just started answering again.
        """
        state = self.state
        if state is BreakerState.CLOSED:
            return True
        if state is BreakerState.HALF_OPEN and not self._probe_in_flight:
            self._probe_in_flight = True
            return True
        return False

    def record_success(self) -> None:
        self._state = BreakerState.CLOSED
        self._failures = 0
        self._probe_in_flight = False

    def record_failure(self) -> bool:
        """Count a failure. Returns True if this call opened the circuit."""
        self._probe_in_flight = False
        if self._state is BreakerState.HALF_OPEN:
            self._open()
            return True
        if self._state is BreakerState.OPEN:
            return False
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._open()
            return True
        return False

    def trip(self) -> bool:
        """Open immediately, regardless of the count. Returns True if it was not already open."""
        self._probe_in_flight = False
        was_open = self._state is BreakerState.OPEN
        self._open()
        return not was_open

    def release(self) -> None:
        """An attempt ended without saying anything about provider health
        (bad request, caller cancellation, deadline). Free the probe slot."""
        self._probe_in_flight = False

    def _open(self) -> None:
        self._state = BreakerState.OPEN
        self._opened_at = self._clock()
