from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

from .messages import Message


class LLMClient(Protocol):
    """Minimal contract of a language-model client.

    `run_id` groups the calls of one supervisor run for usage/cost tracking.
    """

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.2,
        run_id: str | None = None,
    ) -> str: ...


class LiteLLMClient:
    """Real client using litellm (multi-provider: OpenAI, Anthropic, Gemini, Ollama...)."""

    def __init__(self, model: str, api_key: str | None = None, **kwargs: Any) -> None:
        self.model = model
        self.api_key = api_key
        self.kwargs = kwargs
        self.calls: list[list[Message]] = []

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.2,
        run_id: str | None = None,
    ) -> str:
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
    """Fake client for tests and demos without an API key.

    Accepts a `handler` (messages -> str) or a queue of `responses`.
    Records every call in `calls` (and its run_id in `run_ids`) for assertions.
    """

    def __init__(self, handler: Handler | None = None, responses: Sequence[str] = ()) -> None:
        self.handler = handler
        self.responses = list(responses)
        self.calls: list[list[Message]] = []
        self.run_ids: list[str | None] = []

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.2,
        run_id: str | None = None,
    ) -> str:
        self.calls.append(list(messages))
        self.run_ids.append(run_id)
        if self.handler is not None:
            return self.handler(list(messages))
        if not self.responses:
            raise AssertionError("FakeLLM with no responses available")
        return self.responses.pop(0)
