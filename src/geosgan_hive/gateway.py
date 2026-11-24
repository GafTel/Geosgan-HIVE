from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .errors import (
    AuthenticationError as HiveAuthenticationError,
)
from .errors import (
    BadRequestError as HiveBadRequestError,
)
from .errors import (
    GatewayConnectionError,
    GatewayError,
    GatewayTimeoutError,
    PermanentGatewayError,
    RateLimitError,
    ServiceUnavailableError,
)
from .messages import Message
from .providers import Credentials, Provider, parse_model, resolve_credentials
from .retry import RetryPolicy, is_retryable
from .usage import UsageRecord, UsageTracker


@dataclass
class LLMResponse:
    """Full result of one model call: content plus its usage record."""

    content: str
    model: str
    usage: UsageRecord


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Extract a Retry-After hint (seconds) from a provider exception, if any."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if not headers:
        headers = getattr(exc, "headers", None)
    if not headers:
        return None
    try:
        value = headers.get("retry-after") or headers.get("Retry-After")
    except AttributeError:
        return None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _map_litellm_error(exc: BaseException) -> GatewayError:
    """Map a LiteLLM exception to the HIVE error hierarchy."""
    if isinstance(exc, GatewayError):
        return exc
    import litellm.exceptions as litellm_exc

    if isinstance(exc, litellm_exc.RateLimitError):
        return RateLimitError(str(exc))
    if isinstance(exc, (litellm_exc.ServiceUnavailableError, litellm_exc.InternalServerError)):
        return ServiceUnavailableError(str(exc))
    if isinstance(exc, litellm_exc.Timeout):
        return GatewayTimeoutError(str(exc))
    if isinstance(exc, litellm_exc.APIConnectionError):
        return GatewayConnectionError(str(exc))
    if isinstance(exc, litellm_exc.AuthenticationError):
        return HiveAuthenticationError(str(exc))
    if isinstance(exc, litellm_exc.BadRequestError):
        return HiveBadRequestError(str(exc))
    if isinstance(exc, (ConnectionError, TimeoutError)):  # builtin network errors
        return GatewayConnectionError(str(exc))
    if isinstance(exc, litellm_exc.APIError):
        return PermanentGatewayError(str(exc))
    return PermanentGatewayError(f"{type(exc).__name__}: {exc}")


class ModelGateway:
    """Central gateway for every model call (sync and async) through LiteLLM.

    This is the single choke point for provider/credential resolution, retries,
    timeouts and usage/cost capture. LiteLLM's own retry mechanism is disabled
    (`num_retries=0`) so retries exist only here, in one place.
    """

    def __init__(
        self,
        tracker: UsageTracker | None = None,
        *,
        retry_policy: RetryPolicy | None = None,
        default_timeout_seconds: float = 60.0,
        environ: Mapping[str, str] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        async_sleep: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        self.tracker = tracker if tracker is not None else UsageTracker()
        self.retry_policy = retry_policy if retry_policy is not None else RetryPolicy()
        self.default_timeout_seconds = default_timeout_seconds
        self._environ = environ
        self._sleep = sleep
        self._async_sleep = async_sleep
        self._credentials: dict[Provider, Credentials] = {}

    def bind(
        self,
        model: str,
        *,
        agent: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> GatewayLLMClient:
        """Create an LLMClient bound to one agent's model, backed by this gateway."""
        parse_model(model)  # fail fast on an invalid model string
        return GatewayLLMClient(
            self,
            model=model,
            agent=agent,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )

    def complete(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        agent: str | None = None,
        run_id: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> str:
        return self.complete_response(
            messages,
            model=model,
            agent=agent,
            run_id=run_id,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        ).content

    def complete_response(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        agent: str | None = None,
        run_id: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> LLMResponse:
        kwargs = self._build_kwargs(
            messages, model=model, temperature=temperature, max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )
        policy = self.retry_policy
        for attempt in range(1, policy.max_attempts + 1):
            started = time.perf_counter()
            try:
                import litellm

                response = litellm.completion(**kwargs)
            except Exception as exc:
                error = _map_litellm_error(exc)
                if is_retryable(error) and attempt < policy.max_attempts:
                    self._sleep(policy.delay_for_attempt(attempt, _retry_after_seconds(exc)))
                    continue
                self._record_failure(
                    model=model, agent=agent, run_id=run_id, attempts=attempt,
                    error=type(error).__name__, latency_ms=(time.perf_counter() - started) * 1000,
                )
                raise error from exc
            latency_ms = (time.perf_counter() - started) * 1000
            return self._to_response(
                response, model=model, agent=agent, run_id=run_id,
                latency_ms=latency_ms, attempts=attempt,
            )
        raise PermanentGatewayError("unreachable: retry loop exhausted")  # pragma: no cover

    async def acomplete(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        agent: str | None = None,
        run_id: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> str:
        return (
            await self.acomplete_response(
                messages,
                model=model,
                agent=agent,
                run_id=run_id,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
            )
        ).content

    async def acomplete_response(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        agent: str | None = None,
        run_id: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> LLMResponse:
        kwargs = self._build_kwargs(
            messages, model=model, temperature=temperature, max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )
        policy = self.retry_policy
        for attempt in range(1, policy.max_attempts + 1):
            started = time.perf_counter()
            try:
                import litellm

                response = await litellm.acompletion(**kwargs)
            except asyncio.CancelledError:
                self._record_failure(
                    model=model, agent=agent, run_id=run_id, attempts=attempt,
                    error="cancelled", latency_ms=(time.perf_counter() - started) * 1000,
                )
                raise
            except Exception as exc:
                error = _map_litellm_error(exc)
                if is_retryable(error) and attempt < policy.max_attempts:
                    delay = policy.delay_for_attempt(attempt, _retry_after_seconds(exc))
                    await self._async_sleep(delay)
                    continue
                self._record_failure(
                    model=model, agent=agent, run_id=run_id, attempts=attempt,
                    error=type(error).__name__, latency_ms=(time.perf_counter() - started) * 1000,
                )
                raise error from exc
            latency_ms = (time.perf_counter() - started) * 1000
            return self._to_response(
                response, model=model, agent=agent, run_id=run_id,
                latency_ms=latency_ms, attempts=attempt,
            )
        raise PermanentGatewayError("unreachable: retry loop exhausted")  # pragma: no cover

    def _credentials_for(self, provider: Provider) -> Credentials:
        if provider not in self._credentials:
            self._credentials[provider] = resolve_credentials(provider, self._environ)
        return self._credentials[provider]

    def _build_kwargs(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float,
        max_tokens: int | None,
        timeout_seconds: float | None,
    ) -> dict[str, Any]:
        provider, _ = parse_model(model)
        creds = self._credentials_for(provider)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [m.to_openai() for m in messages],
            "temperature": temperature,
            "timeout": timeout_seconds
            if timeout_seconds is not None
            else self.default_timeout_seconds,
            "num_retries": 0,  # retries are handled exclusively by this gateway
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if creds.api_key:
            kwargs["api_key"] = creds.api_key
        if creds.api_base:
            kwargs["api_base"] = creds.api_base
        return kwargs

    def _to_response(
        self,
        response: Any,
        *,
        model: str,
        agent: str | None,
        run_id: str | None,
        latency_ms: float,
        attempts: int,
    ) -> LLMResponse:
        provider, _ = parse_model(model)
        content = response.choices[0].message.content or ""
        usage = response.usage
        record = UsageRecord(
            agent=agent,
            model=model,
            provider=provider.value,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
            cost_usd=self._safe_cost(response),
            latency_ms=latency_ms,
            attempts=attempts,
            timestamp=datetime.now(UTC).isoformat(timespec="milliseconds"),
            run_id=run_id,
        )
        self.tracker.record(record)
        return LLMResponse(content=content, model=model, usage=record)

    def _record_failure(
        self, *, model: str, agent: str | None, run_id: str | None,
        attempts: int, error: str, latency_ms: float,
    ) -> None:
        """Record a failed/cancelled call so physical provider requests (retry
        attempts, timeouts, cancellations) are accounted, never silently dropped.
        cost_usd stays None on failure (unknown, never 0)."""
        provider, _ = parse_model(model)
        self.tracker.record(
            UsageRecord(
                agent=agent, model=model, provider=provider.value,
                prompt_tokens=0, completion_tokens=0, total_tokens=0,
                cost_usd=None, latency_ms=latency_ms, attempts=attempts,
                timestamp=datetime.now(UTC).isoformat(timespec="milliseconds"),
                run_id=run_id, error=error,
            )
        )

    @staticmethod
    def _safe_cost(response: Any) -> float | None:
        """Actual cost from LiteLLM when known; None otherwise (never 0)."""
        try:
            import litellm

            value = litellm.completion_cost(completion_response=response)
            cost = float(value)
        except Exception:
            return None
        if not math.isfinite(cost) or cost <= 0:
            return None  # unknown or free-local model: report None, not 0
        return cost


class GatewayLLMClient:
    """LLMClient bound to a single agent's model; delegates to the shared gateway."""

    def __init__(
        self,
        gateway: ModelGateway,
        *,
        model: str,
        agent: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.gateway = gateway
        self.model = model
        self.agent = agent
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        run_id: str | None = None,
    ) -> str:
        return self.gateway.complete(
            messages,
            model=self.model,
            agent=self.agent,
            run_id=run_id,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout_seconds,
        )

    async def acomplete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        run_id: str | None = None,
    ) -> str:
        return await self.gateway.acomplete(
            messages,
            model=self.model,
            agent=self.agent,
            run_id=run_id,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout_seconds,
        )
