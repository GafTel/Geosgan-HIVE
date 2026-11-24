from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from .providers import Provider


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Per-key circuit breaker. In-memory only (Phase 2.5 decision): cooldowns use
    a monotonic clock in-process; `opened_at_utc` is stored for observability and
    restart-safe scheduling by the recovery worker (via persisted checkpoint
    timestamps, not this registry)."""

    key: str
    state: CircuitState = CircuitState.CLOSED
    failures: int = 0
    opened_at_mono: float | None = None
    opened_at_utc: str | None = None
    probe_in_flight: bool = False


class HealthRegistry:
    """Thread-safe, async-compatible provider/model health registry.

    - Keys: provider-wide ``p:<provider>`` and model-specific ``m:<model>``.
    - CLOSED -> OPEN after `failure_threshold` recorded failures.
    - OPEN -> HALF_OPEN automatically once the cooldown has elapsed (monotonic clock).
    - HALF_OPEN: at most `half_open_max_probes` concurrent probes, single-flight
      (claimed under a lock; concurrent callers simply see the circuit as unavailable).
    - HALF_OPEN probe success -> CLOSED; probe failure -> OPEN again (reopened).
    - No polling: state changes only on observed outcomes or cooldown expiry.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        cooldown_seconds: float = 60.0,
        half_open_max_probes: int = 1,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be >= 0")
        if half_open_max_probes < 1:
            raise ValueError("half_open_max_probes must be >= 1")
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.half_open_max_probes = half_open_max_probes
        self._clock = clock
        self._lock = threading.Lock()
        self._breakers: dict[str, CircuitBreaker] = {}

    @staticmethod
    def provider_key(provider: Provider | str) -> str:
        value = provider.value if isinstance(provider, Provider) else str(provider)
        return f"p:{value}"

    @staticmethod
    def model_key(model: str) -> str:
        return f"m:{model}"

    def _breaker(self, key: str) -> CircuitBreaker:
        breaker = self._breakers.get(key)
        if breaker is None:
            breaker = CircuitBreaker(key=key)
            self._breakers[key] = breaker
        return breaker

    def _current_state(self, breaker: CircuitBreaker) -> CircuitState:
        """Apply cooldown expiry: OPEN -> HALF_OPEN once the cooldown elapsed."""
        if (
            breaker.state == CircuitState.OPEN
            and breaker.opened_at_mono is not None
            and self._clock() - breaker.opened_at_mono >= self.cooldown_seconds
        ):
            breaker.state = CircuitState.HALF_OPEN
            breaker.probe_in_flight = False
        return breaker.state

    def state(self, key: str) -> CircuitState:
        with self._lock:
            return self._current_state(self._breaker(key))

    def cooldown_remaining(self, key: str) -> float:
        with self._lock:
            breaker = self._breaker(key)
            if breaker.state != CircuitState.OPEN or breaker.opened_at_mono is None:
                return 0.0
            return max(0.0, self.cooldown_seconds - (self._clock() - breaker.opened_at_mono))

    def is_available(self, key: str) -> bool:
        """True when dispatch may proceed: CLOSED always; HALF_OPEN only if a probe
        slot is free; OPEN only after cooldown (which flips it to HALF_OPEN)."""
        with self._lock:
            breaker = self._breaker(key)
            state = self._current_state(breaker)
            if state == CircuitState.CLOSED:
                return True
            if state == CircuitState.HALF_OPEN:
                return not breaker.probe_in_flight or self.half_open_max_probes > 1
            return False  # OPEN, cooldown not elapsed

    def record_failure(self, key: str, *, open_circuit: bool = True) -> CircuitState:
        """Record a failure. `open_circuit=False` for credential-specific errors
        (auth): they must not disable unrelated authorized credentials."""
        with self._lock:
            breaker = self._breaker(key)
            self._current_state(breaker)
            breaker.failures += 1
            if open_circuit and breaker.failures >= self.failure_threshold:
                breaker.state = CircuitState.OPEN
                breaker.opened_at_mono = self._clock()
                breaker.opened_at_utc = datetime.now(UTC).isoformat(timespec="milliseconds")
                breaker.probe_in_flight = False
            return breaker.state

    def record_success(self, key: str) -> None:
        with self._lock:
            breaker = self._breaker(key)
            breaker.state = CircuitState.CLOSED
            breaker.failures = 0
            breaker.opened_at_mono = None
            breaker.opened_at_utc = None
            breaker.probe_in_flight = False

    def try_begin_probe(self, key: str) -> bool:
        """Single-flight recovery probe claim. Only in HALF_OPEN; at most
        `half_open_max_probes` concurrent. Returns False when another probe is
        already in flight (callers then treat the key as unavailable)."""
        with self._lock:
            breaker = self._breaker(key)
            state = self._current_state(breaker)
            if state != CircuitState.HALF_OPEN:
                return False
            if breaker.probe_in_flight and self.half_open_max_probes <= 1:
                return False
            breaker.probe_in_flight = True
            return True

    def end_probe(self, key: str, *, success: bool) -> CircuitState:
        """HALF_OPEN probe result: success closes the circuit, failure reopens it."""
        with self._lock:
            breaker = self._breaker(key)
            breaker.probe_in_flight = False
            if success:
                breaker.state = CircuitState.CLOSED
                breaker.failures = 0
                breaker.opened_at_mono = None
                breaker.opened_at_utc = None
            else:
                breaker.state = CircuitState.OPEN
                breaker.opened_at_mono = self._clock()
                breaker.opened_at_utc = datetime.now(UTC).isoformat(timespec="milliseconds")
            return breaker.state

    def snapshot(self) -> dict[str, str]:
        with self._lock:
            return {k: self._current_state(b).value for k, b in self._breakers.items()}
