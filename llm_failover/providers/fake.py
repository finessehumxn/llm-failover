"""A scriptable provider for tests and the offline demo."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Iterable, Union

from ..types import Request, Response, Usage


@dataclass(frozen=True)
class Ok:
    text: str = "ok"


@dataclass(frozen=True)
class Raise:
    exc: BaseException


@dataclass(frozen=True)
class Sleep:
    """Sleep (really, so ``asyncio.wait_for`` can cancel it), then succeed."""

    seconds: float
    text: str = "slow ok"


Outcome = Union[Ok, Raise, Sleep]


class ScriptExhausted(AssertionError):
    pass


class FakeProvider:
    """Plays back a fixed sequence of outcomes, one per call.

    If ``default`` is given it is used once the script runs out; otherwise an
    unexpected extra call raises :class:`ScriptExhausted`, which makes tests
    fail loudly when the router calls a provider it should have skipped.
    """

    def __init__(
        self,
        name: str,
        script: Iterable[Outcome] = (),
        *,
        default: Outcome | None = None,
        model: str = "fake-model",
    ) -> None:
        self.name = name
        self.model = model
        self._script = list(script)
        self._default = default
        self.calls: list[Request] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    async def complete(self, request: Request) -> Response:
        self.calls.append(request)
        if self._script:
            step = self._script.pop(0)
        elif self._default is not None:
            step = self._default
        else:
            raise ScriptExhausted(f"{self.name} was called more times than scripted")

        if isinstance(step, Raise):
            raise step.exc
        if isinstance(step, Sleep):
            await asyncio.sleep(step.seconds)
            text = step.text
        else:
            text = step.text
        return Response(
            text=text,
            model=self.model,
            usage=Usage(input_tokens=sum(len(m.content.split()) for m in request.messages), output_tokens=len(text.split())),
            stop_reason="end_turn",
        )
