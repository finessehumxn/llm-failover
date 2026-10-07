"""Provider-neutral request and response shapes.

Every adapter translates to and from these, so the router never has to know
which vendor it is talking to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got {self.role!r}")


@dataclass(frozen=True)
class Request:
    """One chat completion request, independent of provider.

    ``system`` is kept separate from ``messages`` because providers disagree
    about where it goes (Anthropic: a top-level field; OpenAI: a message).
    ``temperature=None`` means "use the provider default" and is not sent.
    """

    messages: tuple[Message, ...]
    system: str | None = None
    max_tokens: int = 1024
    temperature: float | None = None

    def __init__(
        self,
        messages: Sequence[Message | Mapping[str, str]],
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> None:
        normalized = tuple(
            m if isinstance(m, Message) else Message(role=m["role"], content=m["content"])  # type: ignore[arg-type]
            for m in messages
        )
        if not normalized:
            raise ValueError("a request needs at least one message")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        object.__setattr__(self, "messages", normalized)
        object.__setattr__(self, "system", system)
        object.__setattr__(self, "max_tokens", max_tokens)
        object.__setattr__(self, "temperature", temperature)

    @classmethod
    def user(cls, text: str, **kwargs: Any) -> "Request":
        """Shorthand for a single-turn request."""
        return cls([Message("user", text)], **kwargs)


@dataclass(frozen=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class Response:
    """A completed response.

    ``provider`` and ``latency_ms`` are overwritten by the router with the
    route name and the latency it measured, so adapters may leave them at
    their defaults. ``raw`` keeps the SDK object for callers who need
    something this shape does not carry; it is excluded from repr and
    equality.
    """

    text: str
    model: str
    provider: str = ""
    latency_ms: float = 0.0
    usage: Usage | None = None
    stop_reason: str | None = None
    raw: Any = field(default=None, repr=False, compare=False)
