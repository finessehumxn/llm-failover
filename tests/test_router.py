import asyncio
import logging

import pytest

from llm_failover import (
    AllProvidersFailed,
    BreakerState,
    CircuitBreaker,
    ConfigurationError,
    DeadlineExceeded,
    ErrorClass,
    EventType,
    FakeProvider,
    LoggingHook,
    Outcome,
    ProviderError,
    Request,
    RequestRejected,
    Route,
    Router,
)
from llm_failover.providers import Ok, Raise, ScriptExhausted, Sleep

from .conftest import SlowFailProvider


def err(status, **kw):
    return Raise(ProviderError.from_status(status, **kw))


def make(providers, clock, hook, backoff, **kw):
    return Router(providers, clock=clock, sleep=clock.sleep, hooks=[hook], backoff=backoff, **kw)


def kinds(trail):
    out = []
    for e in trail:
        if e.type is EventType.ATTEMPT:
            out.append(f"{e.provider}:{e.outcome.value}")
        else:
            out.append(f"{e.provider}:{e.type.value}")
    return out


# -- happy path ---------------------------------------------------------------


async def test_primary_success_is_not_a_fallback(clock, hook, backoff, req):
    a, b = FakeProvider("a", [Ok("hi")]), FakeProvider("b")
    r = make([a, b], clock, hook, backoff)
    res = await r.complete(req)
    assert res.response.text == "hi"
    assert res.provider == "a"
    assert res.fell_back is False
    assert kinds(res.trail) == ["a:ok"]
    assert b.call_count == 0
    assert hook.events == res.trail


async def test_router_stamps_provider_name_and_latency(clock, hook, backoff, req):
    class Slowish:
        name = "slowish"

        async def complete(self, request):
            clock.advance(0.25)
            from llm_failover import Response

            return Response(text="x", model="m", provider="WRONG", latency_ms=-1)

    res = await make([Slowish()], clock, hook, backoff).complete(req)
    assert res.response.provider == "slowish"
    assert res.response.latency_ms == pytest.approx(250.0)
    assert res.attempts[0].latency_ms == pytest.approx(250.0)


# -- retryable ------------------------------------------------------------------


async def test_retryable_error_is_retried_on_same_provider_with_backoff(clock, hook, backoff, req):
    a = FakeProvider("a", [err(503), Ok("second time")])
    r = make([a, FakeProvider("b")], clock, hook, backoff, max_attempts=3)
    res = await r.complete(req)
    assert res.provider == "a"
    assert res.fell_back is False
    assert kinds(res.trail) == ["a:error", "a:ok"]
    assert len(clock.sleeps) == 1
    assert 0 <= clock.sleeps[0] <= backoff.ceiling(1)


async def test_backoff_sleeps_follow_the_jitter_schedule(clock, hook, req):
    import random

    from llm_failover import Backoff

    seeded = Backoff(base_delay=1, max_delay=100, rng=random.Random(99))
    expected_rng = random.Random(99)
    expected = [expected_rng.uniform(0, c) for c in (1, 2, 4)]
    a = FakeProvider("a", [err(500), err(500), err(500), Ok()])
    r = make([a], clock, hook, seeded, max_attempts=4)
    await r.complete(req)
    assert clock.sleeps == pytest.approx(expected)


async def test_exhausted_retries_fall_over_to_next_provider(clock, hook, backoff, req):
    a = FakeProvider("a", [err(429), err(429)])
    b = FakeProvider("b", [Ok("from b")])
    res = await make([a, b], clock, hook, backoff, max_attempts=2).complete(req)
    assert res.provider == "b"
    assert res.fell_back is True
    assert kinds(res.trail) == ["a:error", "a:error", "b:ok"]
    assert [e.error_class for e in res.attempts[:2]] == [ErrorClass.RETRYABLE] * 2
    assert len(clock.sleeps) == 1  # no sleep after the final attempt on a provider


async def test_timeout_is_classified_and_fails_over(hook, req):
    a = FakeProvider("a", [Sleep(10)])
    b = FakeProvider("b", [Ok()])
    r = Router([Route(a, timeout=0.02), b], max_attempts=1, hooks=[hook])
    res = await r.complete(req)
    assert res.provider == "b"
    first = res.attempts[0]
    assert first.outcome is Outcome.TIMEOUT
    assert first.error_class is ErrorClass.RETRYABLE
    assert first.latency_ms < 1000


async def test_retry_after_is_honoured_when_reasonable(clock, hook, backoff, req):
    a = FakeProvider("a", [err(429, retry_after=3.0), Ok()])
    res = await make([a], clock, hook, backoff).complete(req)
    assert res.provider == "a"
    assert clock.sleeps == [3.0]


async def test_retry_after_longer_than_max_delay_moves_on_instead_of_waiting(clock, hook, backoff, req):
    a = FakeProvider("a", [err(429, retry_after=120.0)])
    b = FakeProvider("b", [Ok()])
    res = await make([a, b], clock, hook, backoff, max_attempts=3).complete(req)
    assert res.provider == "b"
    assert a.call_count == 1
    assert clock.sleeps == []


async def test_connection_error_from_custom_provider_is_retryable(clock, hook, backoff, req):
    a = FakeProvider("a", [Raise(ConnectionResetError("reset")), Ok()])
    res = await make([a], clock, hook, backoff).complete(req)
    assert res.provider == "a"
    assert res.attempts[0].error_class is ErrorClass.RETRYABLE


# -- non-retryable ----------------------------------------------------------------


async def test_bad_request_raises_immediately_without_burning_other_providers(clock, hook, backoff, req):
    a = FakeProvider("a", [err(400, message="roles must alternate")])
    b = FakeProvider("b", default=Ok())
    r = make([a, b], clock, hook, backoff, max_attempts=3)
    with pytest.raises(RequestRejected) as ei:
        await r.complete(req)
    assert ei.value.provider == "a"
    assert ei.value.status_code == 400
    assert b.call_count == 0
    assert a.call_count == 1
    assert kinds(ei.value.trail) == ["a:error"]
    assert isinstance(ei.value.__cause__, ProviderError)


async def test_bad_request_does_not_count_against_the_breaker(clock, hook, backoff, req):
    a = FakeProvider("a", default=err(422))
    r = make([a], clock, hook, backoff, failure_threshold=1)
    for _ in range(3):
        with pytest.raises(RequestRejected):
            await r.complete(req)
    assert r.breaker("a").state is BreakerState.CLOSED


# -- config errors -----------------------------------------------------------------


@pytest.mark.parametrize("status", [401, 402, 403, 404])
async def test_config_error_fails_over_loudly(status, clock, hook, backoff, req):
    a = FakeProvider("a", [err(status)])
    b = FakeProvider("b", [Ok()])
    r = make([a, b], clock, hook, backoff, max_attempts=3)
    res = await r.complete(req)
    assert res.provider == "b"
    assert res.fell_back is True
    assert a.call_count == 1  # never retried with the same broken credentials
    assert kinds(res.trail) == ["a:error", "a:config_error", "a:circuit_opened", "b:ok"]
    assert len(res.config_errors) == 1
    assert res.config_errors[0].status_code == status
    assert r.config_error_counts["a"] == 1
    assert r.health()["a"]["config_errors"] == 1


async def test_config_error_trips_breaker_so_later_calls_skip_but_still_report(clock, hook, backoff, req):
    a = FakeProvider("a", [err(401), err(401)])
    b = FakeProvider("b", default=Ok())
    r = make([a, b], clock, hook, backoff, cooldown=60)
    await r.complete(req)
    res2 = await r.complete(req)
    assert a.call_count == 1
    assert kinds(res2.trail) == ["a:skipped", "b:ok"]
    clock.advance(60)  # cool-down over: one probe, still broken, still loud
    res3 = await r.complete(req)
    assert a.call_count == 2
    assert len(res3.config_errors) == 1
    assert r.config_error_counts["a"] == 2


async def test_config_error_is_logged_at_error_level(clock, backoff, req, caplog):
    a = FakeProvider("a", [err(401, message="invalid x-api-key")])
    b = FakeProvider("b", [Ok()])
    r = Router([a, b], clock=clock, sleep=clock.sleep, backoff=backoff)  # default hooks
    with caplog.at_level(logging.DEBUG, logger="llm_failover"):
        await r.complete(req)
    errors = [rec for rec in caplog.records if rec.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "config_error" in errors[0].getMessage()
    assert errors[0].llm_failover["provider"] == "a"


async def test_strict_config_raises_instead_of_failing_over(clock, hook, backoff, req):
    a = FakeProvider("a", [err(403)])
    b = FakeProvider("b", default=Ok())
    r = make([a, b], clock, hook, backoff, strict_config=True)
    with pytest.raises(ConfigurationError) as ei:
        await r.complete(req)
    assert ei.value.provider == "a"
    assert b.call_count == 0
    assert r.config_error_counts["a"] == 1


# -- unknown -------------------------------------------------------------------------


async def test_unknown_exception_is_not_retried_on_same_provider(clock, hook, backoff, req):
    a = FakeProvider("a", [Raise(KeyError("choices"))])
    b = FakeProvider("b", [Ok()])
    res = await make([a, b], clock, hook, backoff, max_attempts=3).complete(req)
    assert res.provider == "b"
    assert a.call_count == 1
    assert res.attempts[0].error_class is ErrorClass.UNKNOWN


# -- total failure -----------------------------------------------------------------


async def test_all_failed_carries_trail_and_names_config_errors(clock, hook, backoff, req):
    a = FakeProvider("a", [err(401)])
    b = FakeProvider("b", [err(503), err(503)])
    with pytest.raises(AllProvidersFailed) as ei:
        await make([a, b], clock, hook, backoff).complete(req)
    e = ei.value
    assert not isinstance(e, DeadlineExceeded)
    assert "configuration error" in str(e)
    assert len(e.config_errors) == 1
    assert kinds(e.trail) == ["a:error", "a:config_error", "a:circuit_opened", "b:error", "b:error"]


async def test_all_circuits_open_raises_without_calling_anyone(clock, hook, backoff, req):
    a, b = FakeProvider("a"), FakeProvider("b")
    r = make([a, b], clock, hook, backoff)
    r.breaker("a").trip()
    r.breaker("b").trip()
    with pytest.raises(AllProvidersFailed, match="every circuit is open"):
        await r.complete(req)
    assert a.call_count == b.call_count == 0


# -- circuit breaker integration ------------------------------------------------------


async def test_circuit_opens_after_threshold_and_provider_is_skipped(clock, hook, backoff, req):
    a = FakeProvider("a", default=err(503))
    b = FakeProvider("b", default=Ok())
    r = make([a, b], clock, hook, backoff, max_attempts=2, failure_threshold=3, cooldown=30)
    await r.complete(req)  # a fails twice
    res2 = await r.complete(req)  # a fails once more -> opens, retry abandoned
    assert kinds(res2.trail) == ["a:error", "a:circuit_opened", "b:ok"]
    res3 = await r.complete(req)
    assert kinds(res3.trail) == ["a:skipped", "b:ok"]
    assert a.call_count == 3
    assert r.health()["a"]["circuit"] == "open"


async def test_half_open_probe_success_restores_primary(clock, hook, backoff, req):
    a = FakeProvider("a", [err(500), Ok("back")])
    b = FakeProvider("b", default=Ok())
    r = make([a, b], clock, hook, backoff, max_attempts=1, failure_threshold=1, cooldown=10)
    assert (await r.complete(req)).provider == "b"
    assert (await r.complete(req)).provider == "b"  # still cooling down
    clock.advance(10)
    res = await r.complete(req)
    assert res.provider == "a" and res.fell_back is False
    assert r.breaker("a").state is BreakerState.CLOSED


async def test_half_open_probe_failure_does_not_retry_same_provider(clock, hook, backoff, req):
    a = FakeProvider("a", [err(500), err(500)])
    b = FakeProvider("b", default=Ok())
    r = make([a, b], clock, hook, backoff, max_attempts=3, failure_threshold=1, cooldown=10)
    await r.complete(req)
    clock.advance(10)
    res = await r.complete(req)
    assert a.call_count == 2  # one probe, not three attempts
    assert kinds(res.trail) == ["a:error", "a:circuit_opened", "b:ok"]


async def test_breakers_are_per_provider(clock, hook, backoff, req):
    a = FakeProvider("a", default=err(503))
    b = FakeProvider("b", default=Ok())
    r = make([a, b], clock, hook, backoff, max_attempts=1, failure_threshold=1)
    await r.complete(req)
    assert r.breaker("a").state is BreakerState.OPEN
    assert r.breaker("b").state is BreakerState.CLOSED


async def test_route_can_supply_its_own_breaker(clock, hook, backoff, req):
    custom = CircuitBreaker(failure_threshold=1, cooldown=999, clock=clock)
    a = FakeProvider("a", [err(500)])
    b = FakeProvider("b", default=Ok())
    r = make([Route(a, breaker=custom), b], clock, hook, backoff, max_attempts=1)
    await r.complete(req)
    assert r.breaker("a") is custom
    assert custom.state is BreakerState.OPEN


# -- deadline ------------------------------------------------------------------------


async def test_deadline_stops_further_attempts(clock, hook, backoff, req):
    a = SlowFailProvider("a", clock, cost=4.0)
    b = SlowFailProvider("b", clock, cost=4.0)
    c = FakeProvider("c")  # would succeed, but the budget is spent before we reach it
    r = make([a, b, c], clock, hook, backoff, max_attempts=1, deadline=8.0)
    with pytest.raises(DeadlineExceeded) as ei:
        await r.complete(req)
    assert c.call_count == 0
    assert len([e for e in ei.value.trail if e.type is EventType.ATTEMPT]) == 2


async def test_backoff_is_skipped_when_it_would_overrun_the_deadline(clock, hook, req):
    from llm_failover import Backoff

    class Fixed:
        def uniform(self, lo, hi):
            return hi

    a = SlowFailProvider("a", clock, cost=1.0)
    b = FakeProvider("b", [Ok()])
    r = make([a, b], clock, hook, Backoff(base_delay=5, max_delay=5, rng=Fixed()), max_attempts=3, deadline=4.0)
    res = await r.complete(req)
    assert res.provider == "b"
    assert a.calls == 1
    assert clock.sleeps == []


async def test_per_call_deadline_overrides_router_default(clock, hook, backoff, req):
    a = SlowFailProvider("a", clock, cost=2.0)
    r = make([a], clock, hook, backoff, max_attempts=5, deadline=None)
    with pytest.raises(DeadlineExceeded):
        await r.complete(req, deadline=1.0)
    assert a.calls == 1


async def test_deadline_bound_timeout_does_not_blame_the_provider(hook, req):
    a = FakeProvider("a", [Sleep(10)])
    r = Router([Route(a, timeout=30)], hooks=[hook], failure_threshold=1)
    with pytest.raises(DeadlineExceeded):
        await r.complete(req, deadline=0.05)
    assert r.breaker("a").state is BreakerState.CLOSED
    assert r.breaker("a").consecutive_failures == 0


# -- hooks, cancellation, construction ----------------------------------------------


async def test_broken_hook_does_not_break_routing(clock, backoff, req, caplog):
    def bad_hook(event):
        raise RuntimeError("hook bug")

    a = FakeProvider("a", [Ok()])
    r = Router([a], clock=clock, sleep=clock.sleep, hooks=[bad_hook], backoff=backoff)
    with caplog.at_level(logging.ERROR, logger="llm_failover"):
        res = await r.complete(req)
    assert res.provider == "a"
    assert "hook" in caplog.text


async def test_cancellation_propagates_and_frees_half_open_probe(req):
    a = FakeProvider("a", [Sleep(10)])
    r = Router([a], hooks=[], timeout=None, failure_threshold=1, cooldown=0)
    r.breaker("a").trip()
    task = asyncio.create_task(r.complete(req))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert r.breaker("a").allow() is True  # the probe slot was released


def test_logging_hook_levels():
    from llm_failover import Event

    lvl = LoggingHook._level
    assert lvl(Event(EventType.CONFIG_ERROR, "a")) == logging.ERROR
    assert lvl(Event(EventType.CIRCUIT_OPENED, "a")) == logging.WARNING
    assert lvl(Event(EventType.SKIPPED, "a")) == logging.INFO
    assert lvl(Event(EventType.ATTEMPT, "a", outcome=Outcome.OK)) == logging.DEBUG
    assert lvl(Event(EventType.ATTEMPT, "a", outcome=Outcome.TIMEOUT)) == logging.WARNING


def test_event_to_dict_is_json_friendly():
    import json

    from llm_failover import Event

    d = Event(EventType.ATTEMPT, "a", attempt=1, outcome=Outcome.ERROR,
              error_class=ErrorClass.RETRYABLE, status_code=503).to_dict()
    assert d["type"] == "attempt" and d["error_class"] == "retryable"
    assert "latency_ms" not in d  # None fields dropped
    json.dumps(d)


def test_construction_validation():
    with pytest.raises(ValueError):
        Router([])
    with pytest.raises(ValueError, match="duplicate"):
        Router([FakeProvider("x"), FakeProvider("x")])
    with pytest.raises(ValueError):
        Router([FakeProvider("x")], max_attempts=0)


async def test_route_overrides_attempts(clock, hook, backoff, req):
    a = FakeProvider("a", [err(503)] * 4 + [Ok()])
    r = make([Route(a, max_attempts=5)], clock, hook, backoff, max_attempts=1)
    res = await r.complete(req)
    assert res.provider == "a" and a.call_count == 5


async def test_fake_provider_flags_unexpected_calls(req):
    with pytest.raises(ScriptExhausted):
        await FakeProvider("x").complete(req)


def test_request_validation_and_dict_messages():
    r = Request([{"role": "user", "content": "hi"}], system="s")
    assert r.messages[0].role == "user"
    with pytest.raises(ValueError):
        Request([])
    with pytest.raises(ValueError):
        Request.user("x", max_tokens=0)
    with pytest.raises(ValueError):
        Request([{"role": "system", "content": "nope"}])
