import asyncio

import litellm
import litellm.exceptions as litellm_exc
import pytest
from test_gateway import fake_response

from geosgan_hive.errors import AuthenticationError, GatewayTimeoutError, ServiceUnavailableError
from geosgan_hive.gateway import ModelGateway
from geosgan_hive.messages import Message
from geosgan_hive.retry import RetryPolicy
from geosgan_hive.usage import UsageTracker


@pytest.fixture
def openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")


def test_acomplete_success_records_usage(monkeypatch, openai_key):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return fake_response(content="async ok")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.0015)
    tracker = UsageTracker()
    gateway = ModelGateway(tracker=tracker)
    out = asyncio.run(
        gateway.acomplete(
            [Message(role="user", content="hi")],
            model="openai/gpt-4o-mini",
            agent="analyst",
            run_id="run-async",
        )
    )
    assert out == "async ok"
    assert calls[0]["num_retries"] == 0  # LiteLLM retries stay disabled on the async path too
    record = tracker.records()[0]
    assert record.agent == "analyst"
    assert record.run_id == "run-async"
    assert record.cost_usd == 0.0015
    assert record.total_tokens == 15


def test_acomplete_response_returns_llm_response(monkeypatch, openai_key):
    async def fake_acompletion(**kwargs):
        return fake_response(content="full")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.001)
    gateway = ModelGateway(tracker=UsageTracker())
    response = asyncio.run(
        gateway.acomplete_response([Message("user", "hi")], model="openai/gpt-4o-mini")
    )
    assert response.content == "full"
    assert response.model == "openai/gpt-4o-mini"
    assert response.usage.total_tokens == 15


def test_acomplete_retries_transient_errors_with_async_backoff(monkeypatch, openai_key):
    attempts = []

    async def fake_acompletion(**kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise litellm_exc.ServiceUnavailableError(
                "503", llm_provider="openai", model="gpt-4o-mini"
            )
        return fake_response(content="recovered")

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    sleeps = []
    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.001)
    policy = RetryPolicy(
        max_attempts=4, base_delay_seconds=1.0, max_delay_seconds=30.0, jitter_ratio=0.0
    )
    tracker = UsageTracker()
    gateway = ModelGateway(tracker=tracker, retry_policy=policy, async_sleep=fake_sleep)
    out = asyncio.run(
        gateway.acomplete([Message("user", "hi")], model="openai/gpt-4o-mini")
    )
    assert out == "recovered"
    assert len(attempts) == 3
    assert sleeps == [1.0, 2.0]
    assert tracker.records()[0].attempts == 3


def test_acomplete_timeout_is_transient_and_retried(monkeypatch, openai_key):
    attempts = []

    async def fake_acompletion(**kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise litellm_exc.Timeout("timed out", model="gpt-4o-mini", llm_provider="openai")
        return fake_response(content="after timeout")

    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.001)
    policy = RetryPolicy(
        max_attempts=2, base_delay_seconds=0.5, max_delay_seconds=10.0, jitter_ratio=0.0
    )
    gateway = ModelGateway(tracker=UsageTracker(), retry_policy=policy, async_sleep=fake_sleep)
    out = asyncio.run(
        gateway.acomplete([Message("user", "hi")], model="openai/gpt-4o-mini")
    )
    assert out == "after timeout"
    assert sleeps == [0.5]


def test_acomplete_permanent_error_fails_fast(monkeypatch, openai_key):
    attempts = []

    async def fake_acompletion(**kwargs):
        attempts.append(1)
        raise litellm_exc.AuthenticationError(
            "bad key", llm_provider="openai", model="gpt-4o-mini"
        )

    async def fake_sleep(seconds):
        raise AssertionError("must not sleep on permanent errors")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    gateway = ModelGateway(tracker=UsageTracker(), async_sleep=fake_sleep)
    with pytest.raises(AuthenticationError):
        asyncio.run(gateway.acomplete([Message("user", "hi")], model="openai/gpt-4o-mini"))
    assert len(attempts) == 1


def test_acomplete_exhaustion_raises_transient_error(monkeypatch, openai_key):
    async def fake_acompletion(**kwargs):
        raise litellm_exc.ServiceUnavailableError(
            "503", llm_provider="openai", model="gpt-4o-mini"
        )

    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    policy = RetryPolicy(
        max_attempts=3, base_delay_seconds=1.0, max_delay_seconds=30.0, jitter_ratio=0.0
    )
    gateway = ModelGateway(tracker=UsageTracker(), retry_policy=policy, async_sleep=fake_sleep)
    with pytest.raises(ServiceUnavailableError):
        asyncio.run(gateway.acomplete([Message("user", "hi")], model="openai/gpt-4o-mini"))
    assert len(sleeps) == 2


def test_async_timeout_mapping(monkeypatch, openai_key):
    async def fake_acompletion(**kwargs):
        raise litellm_exc.Timeout("timed out", model="gpt-4o-mini", llm_provider="openai")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    gateway = ModelGateway(
        tracker=UsageTracker(),
        retry_policy=RetryPolicy(max_attempts=1),  # no retries: surface the mapped error
    )
    with pytest.raises(GatewayTimeoutError):
        asyncio.run(gateway.acomplete([Message("user", "hi")], model="openai/gpt-4o-mini"))


def test_bound_client_acomplete_delegates_to_gateway(monkeypatch, openai_key):
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return fake_response(content="bound async ok", model=kwargs["model"])

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.002)
    tracker = UsageTracker()
    gateway = ModelGateway(tracker=tracker)
    client = gateway.bind(
        "openai/gpt-4o-mini", agent="developer", temperature=0.4, timeout_seconds=9.0
    )
    out = asyncio.run(client.acomplete([Message("user", "code")], run_id="run-b"))
    assert out == "bound async ok"
    assert calls[0]["model"] == "openai/gpt-4o-mini"
    assert calls[0]["temperature"] == 0.4
    assert calls[0]["timeout"] == 9.0
    assert calls[0]["num_retries"] == 0
    record = tracker.records()[0]
    assert record.agent == "developer" and record.run_id == "run-b" and record.cost_usd == 0.002
