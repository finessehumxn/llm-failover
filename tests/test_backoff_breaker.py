import random

import pytest

from llm_failover import Backoff, BreakerState, CircuitBreaker


# -- backoff ----------------------------------------------------------------


def test_ceiling_doubles_then_caps():
    b = Backoff(base_delay=0.5, max_delay=3.0)
    assert [b.ceiling(n) for n in range(1, 6)] == [0.5, 1.0, 2.0, 3.0, 3.0]


def test_huge_retry_numbers_do_not_overflow():
    assert Backoff(base_delay=1, max_delay=10).ceiling(10_000) == 10


def test_full_jitter_stays_in_window_and_is_deterministic_with_seed():
    a = Backoff(base_delay=1.0, max_delay=8.0, rng=random.Random(42))
    b = Backoff(base_delay=1.0, max_delay=8.0, rng=random.Random(42))
    for n in range(1, 8):
        da, db = a.delay(n), b.delay(n)
        assert da == db
        assert 0.0 <= da <= a.ceiling(n)


def test_full_jitter_actually_spreads():
    b = Backoff(base_delay=1.0, max_delay=1.0, rng=random.Random(0))
    samples = [b.delay(1) for _ in range(200)]
    assert min(samples) < 0.1 and max(samples) > 0.9


def test_backoff_validation():
    with pytest.raises(ValueError):
        Backoff(base_delay=-1)
    with pytest.raises(ValueError):
        Backoff().ceiling(0)


# -- breaker ----------------------------------------------------------------


def test_opens_after_threshold_consecutive_failures(clock):
    br = CircuitBreaker(failure_threshold=3, cooldown=10, clock=clock)
    assert br.record_failure() is False
    assert br.record_failure() is False
    assert br.state is BreakerState.CLOSED
    assert br.record_failure() is True
    assert br.state is BreakerState.OPEN
    assert br.allow() is False


def test_success_resets_the_count(clock):
    br = CircuitBreaker(failure_threshold=2, cooldown=10, clock=clock)
    br.record_failure()
    br.record_success()
    br.record_failure()
    assert br.state is BreakerState.CLOSED
    assert br.consecutive_failures == 1


def test_half_open_after_cooldown_allows_exactly_one_probe(clock):
    br = CircuitBreaker(failure_threshold=1, cooldown=10, clock=clock)
    br.record_failure()
    clock.advance(9.99)
    assert br.allow() is False
    clock.advance(0.01)
    assert br.state is BreakerState.HALF_OPEN
    assert br.allow() is True
    assert br.allow() is False  # second concurrent caller is refused


def test_half_open_probe_success_closes(clock):
    br = CircuitBreaker(failure_threshold=1, cooldown=5, clock=clock)
    br.record_failure()
    clock.advance(5)
    assert br.allow()
    br.record_success()
    assert br.state is BreakerState.CLOSED
    assert br.allow() and br.allow()


def test_half_open_probe_failure_reopens_for_full_cooldown(clock):
    br = CircuitBreaker(failure_threshold=1, cooldown=5, clock=clock)
    br.record_failure()
    clock.advance(5)
    assert br.allow()
    assert br.record_failure() is True
    assert br.state is BreakerState.OPEN
    clock.advance(4.9)
    assert br.allow() is False
    clock.advance(0.1)
    assert br.allow() is True


def test_release_frees_the_probe_slot(clock):
    br = CircuitBreaker(failure_threshold=1, cooldown=1, clock=clock)
    br.record_failure()
    clock.advance(1)
    assert br.allow()
    br.release()
    assert br.state is BreakerState.HALF_OPEN
    assert br.allow()


def test_trip_opens_immediately_and_reports_transition(clock):
    br = CircuitBreaker(failure_threshold=5, cooldown=1, clock=clock)
    assert br.trip() is True
    assert br.state is BreakerState.OPEN
    assert br.trip() is False


def test_breaker_validation():
    with pytest.raises(ValueError):
        CircuitBreaker(failure_threshold=0)
    with pytest.raises(ValueError):
        CircuitBreaker(cooldown=-1)
