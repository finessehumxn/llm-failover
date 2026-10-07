# llm-failover

Multi-provider LLM routing for Python, with retries, circuit breakers and a deadline, built so that falling back never hides a configuration failure.

## Why

If your product calls one LLM vendor, that vendor's outage is your outage. Putting a second provider behind the first fixes that, and it creates a quieter problem: a fallback that catches every exception will also catch `401 invalid api key`. The key gets revoked, traffic moves to the backup, every request still succeeds, and nothing tells you the primary is dead until the backup also has a bad day or the bill arrives. A fallback that hides a revoked key is worse than an outage, because an outage at least gets noticed.

This library keeps the fallback and loses the silence. Every failure is classified, and the class decides what happens next:

- transient failures are retried, then failed over;
- configuration failures fail over immediately but are reported loudly, counted, and trip that provider's circuit;
- malformed requests are raised at once, because every provider would reject them too.

The core is a handful of small modules (router, error taxonomy, breaker, backoff, events) with no runtime dependencies. The Anthropic and OpenAI adapters are optional extras.

## Quickstart

```bash
pip install "llm-failover[anthropic,openai]"   # or just llm-failover for the core
python -m llm_failover.demo                    # offline walkthrough, no keys needed
```

```python
import asyncio
from llm_failover import Router, Route, Request, AllProvidersFailed
from llm_failover import AnthropicProvider, OpenAIProvider

router = Router(
    [
        Route(AnthropicProvider(), timeout=20),          # default model: claude-sonnet-5-5
        OpenAIProvider(model="your-openai-model-id"),
    ],
    max_attempts=2,     # per provider
    deadline=45,        # seconds, across all attempts and providers
)

async def main():
    result = await router.complete(Request.user("Summarize this in one line: ..."))
    print(result.response.text)
    if result.fell_back:
        print("served by", result.provider, [e.to_dict() for e in result.trail])
    if result.config_errors:
        ...  # page someone: a provider is rejecting our credentials

asyncio.run(main())
```

API keys come from the usual environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) or the `api_key=` argument. Any object with a `name` and an `async complete(request) -> Response` is a provider, so adding another vendor is one small class.

## What the router does with each failure

| Error class | Examples | Same provider | Other providers | Circuit breaker | Reported as |
|---|---|---|---|---|---|
| `RETRYABLE` | timeout, connection reset, 408, 409, 429, 5xx, 529 | retried up to `max_attempts` with full-jitter backoff; honours `Retry-After` up to `max_delay` | tried next once retries run out | counts toward `failure_threshold` | `attempt` event, WARNING |
| `FATAL_CONFIG` | 401, 402, 403, 404 (unknown model), missing credentials | not retried | tried next (or `ConfigurationError` raised if `strict_config=True`) | tripped open immediately | `attempt` + `config_error` event at ERROR; `router.config_error_counts` |
| `NON_RETRYABLE` | 400, 413, 422 | not retried | **not tried**: `RequestRejected` is raised | not counted | `attempt` event, then the exception |
| `UNKNOWN` | any other exception (usually an adapter bug) | not retried | tried next | counts toward `failure_threshold` | `attempt` event, WARNING |
| deadline | overall budget spent | stops | stops | a timeout caused by the deadline is not counted | `DeadlineExceeded` |

If nothing succeeds you get `AllProvidersFailed`, whose message says how many configuration errors were involved. Every exception and every `RouterResult` carries `trail`: the ordered list of events (attempts, config errors, circuit openings, skips), each with provider, attempt number, outcome, error class, status code and latency. Error text is scrubbed of anything credential-shaped before it reaches an event or a log line.

Hooks are plain callables that receive each event. The default hook logs to the `llm_failover` logger with the event dict attached as `record.llm_failover`; pass `hooks=[...]` to send events to metrics or tracing instead, or `hooks=[]` to turn them off. A hook that raises is logged and ignored.

## Decisions and what I rejected

**Not retrying 400s anywhere.** A 400 means the request is malformed: roles out of order, a parameter the API does not accept. Sending it to the next provider spends latency and money on a near-certain second rejection, and in the meantime hides a bug in the caller behind what looks like an outage. The cost of this rule is real: a few 400s are provider-specific (one model rejects a parameter another accepts, or one context window is smaller). I would rather surface those as errors and fix the request than have the router quietly paper over them.

**Config errors fail over, but loudly.** The two obvious options are both wrong. Raising on a 401 turns one revoked key into a user-facing outage even though a healthy backup is sitting there. Failing over silently is how a dead key goes unnoticed for weeks. So the router serves the user from the next provider and, in the same moment, emits a `config_error` event at ERROR level, increments a per-provider counter you can alert on, and trips that provider's circuit so it is not hammered with doomed requests. When the cool-down ends, one probe goes through; if the key is still bad, the event fires again. Teams that prefer to fail closed can set `strict_config=True`.

**Full jitter, not fixed or plain exponential backoff.** When a provider has a blip, many clients fail at the same instant. Fixed or purely exponential delays bring them back at the same instant too. Full jitter draws each delay uniformly from `[0, min(max_delay, base * 2^n)]`, which spreads the retries across the window. The sleep function and the random generator are injectable, so the tests assert exact delays.

**One circuit breaker per provider, not one global breaker.** The question a breaker answers is "is this provider healthy?", and that is a per-provider fact. A global breaker would stop traffic to a healthy backup because the primary is failing, which defeats the purpose. Bad requests do not count against a breaker (they say nothing about provider health), and neither do timeouts caused by the caller's own deadline.

**Retry ownership belongs to the router.** Both vendor SDKs retry internally by default. The adapters set `max_retries=0` when they build the client, because otherwise attempts multiply (router retries times SDK retries) and the SDK's retries never appear in the trail.

**Zero required dependencies.** The core is the part people need to read and trust, so it is the standard library only. SDKs are imported lazily when you construct an adapter, which also means installing the router does not pin your vendor SDK versions.

**Why not LiteLLM?** LiteLLM is the right choice if you need breadth: a hundred-plus providers behind one API, a proxy server, spend tracking, and a large community keeping all of that current. It also has fallbacks and retries. This is a different trade. It is a small core you can read in one sitting, focused entirely on failure semantics: the error taxonomy, the per-provider breaker, the deadline, and the rule that a credential failure is never silent. If you already run LiteLLM, the taxonomy and the "loud config error" rule are ideas you can apply there; if you want something minimal to embed and audit, this is that.

## Limits

- **No streaming yet.** `complete` returns a whole response. Failing over mid-stream needs its own rules (what to do with tokens already sent), and I would rather not ship that half-done.
- **No cost- or latency-aware routing.** Order is the routing policy. Choosing providers by price, observed latency or load is out of scope.
- **Thin adapters.** They map text messages, a system prompt, `max_tokens` and `temperature`. Tools, images, structured output and vendor-specific features are not mapped; use `response.raw` or write a richer adapter. Anthropic's `refusal` stop reason is passed through as `stop_reason` and not treated as an error.
- **Breaker state is in-process.** Each process learns provider health on its own; there is no shared state across replicas.
- **No hedging.** Requests are sequential across providers, never raced in parallel.
- **Status-code classification is a policy, not a law.** 404 is treated as a configuration error because, against a fixed endpoint, it nearly always means an unknown model name. If a provider uses codes differently, raise `ProviderError` with your own `ErrorClass` from a custom adapter.

## Development

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev,anthropic,openai]"
.venv/bin/python -m pytest
```

The tests make no network calls. Time is controlled with an injected clock and sleep, randomness with a seeded generator, and the SDK adapters are tested against the SDKs' real exception classes with a stubbed client.

## License

MIT
