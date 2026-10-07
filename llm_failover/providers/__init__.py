from .base import Provider, map_sdk_exception
from .fake import FakeProvider, Ok, Raise, ScriptExhausted, Sleep

__all__ = [
    "AnthropicProvider",
    "FakeProvider",
    "Ok",
    "OpenAIProvider",
    "Provider",
    "Raise",
    "ScriptExhausted",
    "Sleep",
    "map_sdk_exception",
]


def __getattr__(name: str):
    # SDK adapters are imported on first use so the core stays dependency-free.
    if name == "AnthropicProvider":
        from .anthropic import AnthropicProvider

        return AnthropicProvider
    if name == "OpenAIProvider":
        from .openai import OpenAIProvider

        return OpenAIProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
