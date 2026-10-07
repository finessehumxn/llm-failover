"""Offline walkthrough of the router's failure handling.

    python -m llm_failover.demo

Uses scripted fake providers only: no network, no API keys.
"""

from __future__ import annotations

import asyncio
import random

from . import (
    AllProvidersFailed,
    Backoff,
    FakeProvider,
    ProviderError,
    Request,
    RequestRejected,
    Route,
    Router,
    RouterResult,
)
from .events import Event, EventType
from .providers import Ok, Raise, Sleep


def _fmt(e: Event) -> str:
    if e.type is EventType.ATTEMPT:
        s = f"attempt  {e.provider:<10} #{e.attempt}  {e.outcome.value:<7}"
        if e.error_class:
            s += f"  {e.error_class.value}"
        if e.status_code:
            s += f" (HTTP {e.status_code})"
        if e.latency_ms is not None:
            s += f"  {e.latency_ms:.0f}ms"
        return s
    if e.type is EventType.CONFIG_ERROR:
        return f"CONFIG   {e.provider:<10} !! {e.detail}"
    if e.type is EventType.CIRCUIT_OPENED:
        return f"circuit  {e.provider:<10} OPENED ({e.detail})"
    return f"skipped  {e.provider:<10} ({e.detail})"


def _print_hook(e: Event) -> None:
    print("   ", _fmt(e))


def _result(r: RouterResult) -> None:
    print(f"    => answered by {r.provider!r}; fell_back={r.fell_back}; "
          f"attempts={len(r.attempts)}; config_errors={len(r.config_errors)}")


def _router(providers, **kw) -> Router:
    kw.setdefault("hooks", [_print_hook])
    kw.setdefault("backoff", Backoff(base_delay=0.01, max_delay=0.05, rng=random.Random(7)))
    return Router(providers, **kw)


async def main() -> None:
    req = Request.user("Summarize this contract in one paragraph.")

    print("\n1. Primary times out, secondary answers")
    primary = FakeProvider("primary", [Sleep(5), Sleep(5)])
    secondary = FakeProvider("secondary", [Ok("Here is the summary...")])
    r = _router([Route(primary, timeout=0.05), secondary])
    _result(await r.complete(req))

    print("\n2. Primary's key was revoked: fall back, but loudly")
    primary = FakeProvider("primary", [Raise(ProviderError.from_status(401, "invalid x-api-key"))])
    secondary = FakeProvider("secondary", [Ok()])
    r = _router([primary, secondary])
    res = await r.complete(req)
    _result(res)
    print(f"    router.health()['primary'] = {r.health()['primary']}")

    print("\n3. Repeated 503s open the primary's circuit; later calls skip it")
    primary = FakeProvider("primary", default=Raise(ProviderError.from_status(503, "overloaded")))
    secondary = FakeProvider("secondary", default=Ok())
    r = _router([primary, secondary], max_attempts=2, failure_threshold=3, cooldown=60)
    for i in range(1, 4):
        print(f"  call {i}:")
        await r.complete(req)
    print(f"    primary was called {primary.call_count} times across 3 requests "
          f"(circuit is {r.breaker('primary').state.value})")

    print("\n4. A malformed request (400) is raised at once, not shopped around")
    primary = FakeProvider("primary", [Raise(ProviderError.from_status(400, "messages: roles must alternate"))])
    secondary = FakeProvider("secondary", default=Ok())
    r = _router([primary, secondary])
    try:
        await r.complete(req)
    except RequestRejected as e:
        print(f"    => RequestRejected from {e.provider!r}; secondary was called {secondary.call_count} times")

    print("\n5. Everything down: the exception carries the trail and names the config error")
    primary = FakeProvider("primary", [Raise(ProviderError.from_status(403, "model access denied"))])
    secondary = FakeProvider("secondary", default=Raise(ConnectionResetError("peer reset")))
    r = _router([primary, secondary])
    try:
        await r.complete(req)
    except AllProvidersFailed as e:
        print(f"    => {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
