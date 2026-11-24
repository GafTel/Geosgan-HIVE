import asyncio
import threading

import pytest

from geosgan_hive.errors import (
    AuthenticationError,
    BudgetExceededError,
    ContextWindowExceededError,
    GatewayTimeoutError,
    QuotaExhaustedError,
    RateLimitError,
    ServiceUnavailableError,
    classify_error,
)
from geosgan_hive.health import CircuitState, HealthRegistry


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


# ---------- error taxonomy ----------

def test_classify_error_matrix():
    assert classify_error(RateLimitError("429")) == ("rate_limit", True, True, False)
    assert classify_error(QuotaExhaustedError("q"))[0] == "quota"
    assert classify_error(QuotaExhaustedError("q"))[1:] == (False, True, False)
    assert classify_error(ServiceUnavailableError("503"))[0] == "outage"
    assert classify_error(GatewayTimeoutError("t"))[0] == "timeout"
    assert classify_error(AuthenticationError("401")) == ("auth", False, False, False)
    assert classify_error(ContextWindowExceededError("cw"))[0] == "context_window"
    assert classify_error(ContextWindowExceededError("cw"))[1:] == (False, True, False)
    assert classify_error(BudgetExceededError("b"))[0] == "budget"
    assert classify_error(BudgetExceededError("b"))[3] is True  # terminal
    assert classify_error(asyncio.CancelledError()) == ("cancelled", False, False, True)
    assert classify_error(ValueError("x"))[0] == "unknown"  # conservative permanent
    assert classify_error(ValueError("x"))[1:] == (False, False, False)


# ---------- health registry ----------

def test_circuit_opens_after_threshold_and_blocks():
    clock = FakeClock()
    reg = HealthRegistry(failure_threshold=3, cooldown_seconds=30.0, clock=clock)
    key = reg.model_key("openai/gpt-4o-mini")
    assert reg.state(key) == CircuitState.CLOSED
    assert reg.is_available(key) is True
    reg.record_failure(key)
    reg.record_failure(key)
    assert reg.state(key) == CircuitState.CLOSED
    reg.record_failure(key)
    assert reg.state(key) == CircuitState.OPEN
    assert reg.is_available(key) is False
    assert reg.cooldown_remaining(key) == 30.0


def test_open_circuit_flips_to_half_open_after_cooldown():
    clock = FakeClock()
    reg = HealthRegistry(failure_threshold=1, cooldown_seconds=30.0, clock=clock)
    key = reg.provider_key("openai")
    reg.record_failure(key)
    assert reg.state(key) == CircuitState.OPEN
    clock.advance(29.0)
    assert reg.state(key) == CircuitState.OPEN
    clock.advance(1.0)
    assert reg.state(key) == CircuitState.HALF_OPEN
    assert reg.is_available(key) is True  # probe slot free


def test_half_open_probe_is_single_flight():
    clock = FakeClock()
    reg = HealthRegistry(
        failure_threshold=1, cooldown_seconds=10.0, half_open_max_probes=1, clock=clock
    )
    key = reg.model_key("mistral/x")
    reg.record_failure(key)
    clock.advance(10.0)
    assert reg.state(key) == CircuitState.HALF_OPEN
    assert reg.try_begin_probe(key) is True
    assert reg.try_begin_probe(key) is False  # second probe deduplicated
    # concurrent claims from threads are also deduplicated
    results = []
    def worker():
        results.append(reg.try_begin_probe(key))
    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(r is False for r in results)


def test_probe_success_closes_and_probe_failure_reopens():
    clock = FakeClock()
    reg = HealthRegistry(failure_threshold=1, cooldown_seconds=10.0, clock=clock)
    key = reg.model_key("gemini/x")
    reg.record_failure(key)
    clock.advance(10.0)
    assert reg.try_begin_probe(key) is True
    assert reg.end_probe(key, success=False) == CircuitState.OPEN  # failure reopens
    assert reg.is_available(key) is False
    clock.advance(10.0)
    assert reg.try_begin_probe(key) is True
    assert reg.end_probe(key, success=True) == CircuitState.CLOSED  # success closes
    assert reg.is_available(key) is True
    reg.record_success(key)
    assert reg.snapshot()[key] == "closed"


def test_auth_failure_with_open_circuit_false_does_not_open():
    reg = HealthRegistry(failure_threshold=2, cooldown_seconds=60.0)
    key = reg.provider_key("openai")
    reg.record_failure(key, open_circuit=False)  # credential-specific: no circuit
    reg.record_failure(key, open_circuit=False)
    assert reg.state(key) == CircuitState.CLOSED
    assert reg.is_available(key) is True


def test_provider_and_model_keys_are_independent():
    reg = HealthRegistry(failure_threshold=1, cooldown_seconds=60.0)
    reg.record_failure(reg.model_key("openai/a"))
    assert reg.state(reg.model_key("openai/a")) == CircuitState.OPEN
    assert reg.state(reg.model_key("openai/b")) == CircuitState.CLOSED
    assert reg.state(reg.provider_key("openai")) == CircuitState.CLOSED
    reg.record_failure(reg.provider_key("openai"))
    assert reg.state(reg.provider_key("openai")) == CircuitState.OPEN
    assert reg.state(reg.provider_key("mistral")) == CircuitState.CLOSED


def test_registry_validates_config():
    with pytest.raises(ValueError):
        HealthRegistry(failure_threshold=0)
    with pytest.raises(ValueError):
        HealthRegistry(cooldown_seconds=-1)
    with pytest.raises(ValueError):
        HealthRegistry(half_open_max_probes=0)


def test_is_available_async_safe_under_threads():
    reg = HealthRegistry(failure_threshold=5, cooldown_seconds=0.0)
    key = reg.model_key("ollama/llama3.2")
    outcomes = []
    lock = threading.Lock()

    def worker():
        ok = reg.is_available(key)
        with lock:
            outcomes.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(outcomes)
