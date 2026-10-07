"""OpenAI adapter (Chat Completions). Requires the ``openai`` extra.

``model`` is required: there is no default here because a default model name
baked into a library goes stale, and a stale name is a 404 -- exactly the
kind of configuration failure this library exists to make loud.
"""

from __future__ import annotations

from typing import Any

from ..types import Request, Response, Usage
from .base import map_sdk_exception


class OpenAIProvider:
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        name: str = "openai",
        client: Any = None,
        **client_kwargs: Any,
    ) -> None:
        try:
            import openai
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "OpenAIProvider needs the openai SDK: pip install 'llm-failover[openai]'"
            ) from e
        self._sdk = openai
        self.name = name
        self.model = model
        if client is None:
            client_kwargs.setdefault("max_retries", 0)  # the router owns retries
            client = openai.AsyncOpenAI(api_key=api_key, **client_kwargs)
        self._client = client

    async def complete(self, request: Request) -> Response:
        messages: list[dict[str, str]] = []
        if request.system is not None:
            messages.append({"role": "system", "content": request.system})
        messages.extend({"role": m.role, "content": m.content} for m in request.messages)
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": request.max_tokens,
        }
        if request.temperature is not None:
            params["temperature"] = request.temperature
        try:
            completion = await self._client.chat.completions.create(**params)
        except Exception as exc:
            mapped = map_sdk_exception(exc, self._sdk)
            if mapped is exc:
                raise
            raise mapped from exc

        choice = completion.choices[0] if completion.choices else None
        text = (choice.message.content or "") if choice is not None else ""
        usage = getattr(completion, "usage", None)
        return Response(
            text=text,
            model=getattr(completion, "model", self.model),
            usage=Usage(getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None))
            if usage is not None
            else None,
            stop_reason=getattr(choice, "finish_reason", None),
            raw=completion,
        )
