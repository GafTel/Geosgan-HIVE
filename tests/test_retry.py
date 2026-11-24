import random

import pytest

from geosgan_hive.errors import (
    AuthenticationError,
    PermanentGatewayError,
    RateLimitError,
    TransientGatewayError,
)
from geosgan_hive.retry import RetryPolicy, is_retryable


def test_policy_validates_bounds():
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError):
        RetryPolicy(base_delay_seconds=-1)


def test_exponential_backoff_without_jitter():
    policy = RetryPolicy(
        max_attempts=5, base_delay_seconds=1.0, max_delay_seconds=100.0, jitter_ratio=0.0
    )
    assert policy.delay_for_attempt(1) == 1.0
    assert policy.delay_for_attempt(2) == 2.0
    assert policy.delay_for_attempt(3) == 4.0
    assert policy.delay_for_attempt(4) == 8.0


def test_backoff_is_capped_by_max_delay():
    policy = RetryPolicy(
        max_attempts=10, base_delay_seconds=1.0, max_delay_seconds=5.0, jitter_ratio=0.0
    )
    assert policy.delay_for_attempt(10) == 5.0


def test_jitter_stays_within_bounds():
    policy = RetryPolicy(
        max_attempts=3, base_delay_seconds=2.0, max_delay_seconds=100.0, jitter_ratio=0.5
    )
    for _ in range(200):
        delay = policy.delay_for_attempt(1)
        assert 2.0 <= delay <= 3.0


def test_retry_after_hint_is_respected_but_bounded():
    policy = RetryPolicy(
        max_attempts=3, base_delay_seconds=1.0, max_delay_seconds=10.0, jitter_ratio=0.0
    )
    assert policy.delay_for_attempt(1, retry_after=7.0) == 7.0
    assert policy.delay_for_attempt(1, retry_after=999.0) == 10.0  # still bounded


def test_only_transient_errors_are_retryable():
    assert is_retryable(RateLimitError("429"))
    assert is_retryable(TransientGatewayError("x"))
    assert not is_retryable(AuthenticationError("401"))
    assert not is_retryable(PermanentGatewayError("400"))
    assert not is_retryable(ValueError("nope"))


def test_random_is_used_for_jitter(monkeypatch):
    monkeypatch.setattr(random, "uniform", lambda a, b: b)  # max jitter
    policy = RetryPolicy(
        max_attempts=2, base_delay_seconds=2.0, max_delay_seconds=100.0, jitter_ratio=0.25
    )
    assert policy.delay_for_attempt(1) == 2.5
