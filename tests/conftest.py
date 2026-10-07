from __future__ import annotations

import random

import pytest

from llm_failover import Backoff, CollectingHook, ProviderError, Request, Response


class FakeClock:
    """A monotonic clock that only moves when told to, plus a matching sleep."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class SlowFailProvider:
    """Takes ``cost`` seconds of fake time per call, then fails with ``status``."""

    def __init__(self, name: str, clock: FakeClock, cost: float, status: int = 503) -> None:
        self.name = name
        self.clock = clock
        self.cost = cost
        self.status = status
        self.calls = 0

    async def complete(self, request: Request) -> Response:
        self.calls += 1
        self.clock.advance(self.cost)
        raise ProviderError.from_status(self.status)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def hook() -> CollectingHook:
    return CollectingHook()


@pytest.fixture
def req() -> Request:
    return Request.user("hello")


@pytest.fixture
def backoff() -> Backoff:
    return Backoff(base_delay=0.5, max_delay=8.0, rng=random.Random(1234))
