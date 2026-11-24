from __future__ import annotations

import random
from dataclasses import dataclass

from .errors import TransientGatewayError


def is_retryable(exc: BaseException) -> bool:
    """Only transient gateway errors are retried; permanent ones fail fast."""
    return isinstance(exc, TransientGatewayError)


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded retries with exponential backoff and jitter.

    `max_attempts` bounds the total number of attempts (including the first),
    so retries are always finite. Delays grow as base * 2**(attempt-1), plus a
    random jitter of up to `jitter_ratio` of the delay, and are capped.
    A server-provided Retry-After hint is respected when present.
    """

    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    jitter_ratio: float = 0.5

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay_seconds < 0 or self.max_delay_seconds < 0:
            raise ValueError("delays must be >= 0")
        if self.jitter_ratio < 0:
            raise ValueError("jitter_ratio must be >= 0")

    def delay_for_attempt(self, attempt: int, retry_after: float | None = None) -> float:
        """Delay before the retry that follows the failed `attempt` (1-based)."""
        backoff = min(self.max_delay_seconds, self.base_delay_seconds * (2 ** (attempt - 1)))
        delay = backoff * (1.0 + random.uniform(0.0, self.jitter_ratio))
        if retry_after is not None:
            delay = max(delay, retry_after)
        cap = self.max_delay_seconds * (1.0 + self.jitter_ratio)
        return min(delay, cap) if self.max_delay_seconds > 0 else 0.0
