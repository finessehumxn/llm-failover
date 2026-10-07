"""Anthropic adapter. Requires the ``anthropic`` extra."""

from __future__ import annotations

from typing import Any

from ..types import Request, Response, Usage
from .base import map_sdk_exception


class AnthropicProvider:
    def __init__(
        self,
        *,
        model: str = "claude-sonnet-5-5",
        api_key: str | None = None,
        name: str = "anthropic",
        client: Any = None,
        **client_kwargs: Any,
    ) -> None:
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "AnthropicProvider needs the anthropic SDK: pip install 'llm-failover[anthropic]'"
            ) from e
        self._sdk = anthropic
        self.name = name
        self.model = model
        # The router owns retries and timeouts. Leaving the SDK's own retries on
        # would multiply attempts (router retries x SDK retries) and hide them
        # from the attempt trail.
        if client is None:
            client_kwargs.setdefault("max_retries", 0)
            client = anthropic.AsyncAnthropic(api_key=api_key, **client_kwargs)
        self._client = client

    async def complete(self, request: Request) -> Response:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        }
        if request.system is not None:
            params["system"] = request.system
        if request.temperature is not None:
            params["temperature"] = request.temperature
        try:
            msg = await self._client.messages.create(**params)
        except Exception as exc:
            mapped = map_sdk_exception(exc, self._sdk)
            if mapped is exc:
                raise
            raise mapped from exc

        text = "".join(
            getattr(block, "text", "") for block in msg.content if getattr(block, "type", None) == "text"
        )
        usage = getattr(msg, "usage", None)
        return Response(
            text=text,
            model=getattr(msg, "model", self.model),
            usage=Usage(getattr(usage, "input_tokens", None), getattr(usage, "output_tokens", None))
            if usage is not None
            else None,
            stop_reason=getattr(msg, "stop_reason", None),
            raw=msg,
        )
