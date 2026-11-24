from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

from .messages import Message


class LLMClient(Protocol):
    """Contrato mínimo de un cliente de modelo de lenguaje."""

    def complete(self, messages: Sequence[Message], *, temperature: float = 0.2) -> str: ...


class LiteLLMClient:
    """Cliente real usando litellm (multi-proveedor: OpenAI, Anthropic, Gemini, Ollama...)."""

    def __init__(self, model: str, api_key: str | None = None, **kwargs: Any) -> None:
        self.model = model
        self.api_key = api_key
        self.kwargs = kwargs
        self.calls: list[list[Message]] = []

    def complete(self, messages: Sequence[Message], *, temperature: float = 0.2) -> str:
        import litellm

        self.calls.append(list(messages))
        kwargs = dict(self.kwargs)
        if self.api_key:
            kwargs["api_key"] = self.api_key
        response = litellm.completion(
            model=self.model,
            messages=[m.to_openai() for m in messages],
            temperature=temperature,
            **kwargs,
        )
        content = response.choices[0].message.content
        return content or ""


Handler = Callable[[list[Message]], str]


class FakeLLM:
    """Cliente falso para tests y demos sin clave de API.

    Acepta un `handler` (messages -> str) o una cola de `responses`.
    Registra todas las llamadas en `calls` para aserciones.
    """

    def __init__(self, handler: Handler | None = None, responses: Sequence[str] = ()) -> None:
        self.handler = handler
        self.responses = list(responses)
        self.calls: list[list[Message]] = []

    def complete(self, messages: Sequence[Message], *, temperature: float = 0.2) -> str:
        self.calls.append(list(messages))
        if self.handler is not None:
            return self.handler(list(messages))
        if not self.responses:
            raise AssertionError("FakeLLM sin respuestas disponibles")
        return self.responses.pop(0)
