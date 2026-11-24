from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Protocol

from .messages import Message


class LLMClient(Protocol):
    """Minimal synchronous contract of a language-model client.

    `run_id` groups the calls of one supervisor run for usage/cost tracking.
    Real calls go through `ModelGateway` (see gateway.py); this protocol is
    implemented by `GatewayLLMClient` (production) and `FakeLLM` (offline).
    """

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.2,
        run_id: str | None = None,
    ) -> str: ...


class AsyncLLMClient(Protocol):
    """Asynchronous contract (native async path; see `Agent.arun`/`ModelGateway.acomplete`)."""

    async def acomplete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.2,
        run_id: str | None = None,
    ) -> str: ...


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
        self.acomplete_calls: list[list[Message]] = []
        self.run_ids: list[str | None] = []

    def _next(self, messages: list[Message]) -> str:
        if self.handler is not None:
            return self.handler(messages)
        if not self.responses:
            raise AssertionError("FakeLLM with no responses available")
        return self.responses.pop(0)

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.2,
        run_id: str | None = None,
    ) -> str:
        self.calls.append(list(messages))
        self.run_ids.append(run_id)
        return self._next(list(messages))

    async def acomplete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.2,
        run_id: str | None = None,
    ) -> str:
        self.acomplete_calls.append(list(messages))
        self.run_ids.append(run_id)
        result = self._next(list(messages))
        if inspect.isawaitable(result):  # handlers may be async for concurrency tests
            result = await result
        return result
