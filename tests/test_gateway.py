from types import SimpleNamespace

import litellm
import litellm.exceptions as litellm_exc
import pytest

from geosgan_hive.errors import (
    AuthenticationError,
    MissingCredentialsError,
    RateLimitError,
    UnknownProviderError,
)
from geosgan_hive.gateway import GatewayLLMClient, ModelGateway, _retry_after_seconds
from geosgan_hive.messages import Message
from geosgan_hive.retry import RetryPolicy
from geosgan_hive.usage import UsageTracker


def fake_response(content="ok", model="openai/gpt-4o-mini", prompt=10, completion=5):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
        ),
        model=model,
    )


def rate_limit_error(headers=None):
    return litellm_exc.RateLimitError(
        "rate limited", llm_provider="openai", model="gpt-4o-mini", headers=headers
    )


@pytest.fixture
def openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.delenv("OLLAMA_API_BASE", raising=False)


def test_complete_success_records_usage_and_cost(monkeypatch, openai_key):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return fake_response(content="hello")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.0021)

    tracker = UsageTracker()
    gateway = ModelGateway(tracker=tracker)
    out = gateway.complete(
        [Message(role="user", content="hi")],
        model="openai/gpt-4o-mini",
        agent="researcher",
        run_id="run-1",
    )
    assert out == "hello"
    kwargs = calls[0]
    assert kwargs["model"] == "openai/gpt-4o-mini"
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert kwargs["num_retries"] == 0  # LiteLLM's own retries disabled: gateway owns retries
    assert kwargs["timeout"] == 60.0  # gateway default
    assert kwargs["api_key"] == "sk-test-key"
    record = tracker.records()[0]
    assert record.agent == "researcher"
    assert record.run_id == "run-1"
    assert record.provider == "openai"
    assert record.prompt_tokens == 10
    assert record.completion_tokens == 5
    assert record.total_tokens == 15
    assert record.cost_usd == 0.0021
    assert record.attempts == 1
    assert record.latency_ms >= 0


def test_timeout_and_max_tokens_are_passed(monkeypatch, openai_key):
    calls = []
    def fake_completion(**kwargs):
        calls.append(kwargs)
        return fake_response()

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.001)
    gateway = ModelGateway(tracker=UsageTracker(), default_timeout_seconds=45.0)
    gateway.complete([Message("user", "hi")], model="openai/gpt-4o-mini",
                     timeout_seconds=5.0, max_tokens=128)
    assert calls[0]["timeout"] == 5.0
    assert calls[0]["max_tokens"] == 128
    gateway.complete([Message("user", "hi")], model="openai/gpt-4o-mini")
    assert calls[1]["timeout"] == 45.0
    assert "max_tokens" not in calls[1]


def test_unknown_or_zero_cost_is_recorded_as_none(monkeypatch, openai_key):
    monkeypatch.setattr(litellm, "completion", lambda **kwargs: fake_response())
    tracker = UsageTracker()
    gateway = ModelGateway(tracker=tracker)

    def boom(completion_response):
        raise ValueError("model not in cost map")

    monkeypatch.setattr(litellm, "completion_cost", boom)
    gateway.complete([Message("user", "hi")], model="openai/gpt-4o-mini")
    assert tracker.records()[0].cost_usd is None

    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.0)
    gateway.complete([Message("user", "hi")], model="openai/gpt-4o-mini")
    assert tracker.records()[1].cost_usd is None  # never 0 for unknown costs
    assert tracker.records()[1].total_tokens == 15  # tokens still tracked


def test_transient_error_is_retried_with_backoff_then_succeeds(monkeypatch, openai_key):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise rate_limit_error()
        return fake_response(content="recovered")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.001)
    sleeps = []
    policy = RetryPolicy(
        max_attempts=4, base_delay_seconds=1.0, max_delay_seconds=30.0, jitter_ratio=0.0
    )
    tracker = UsageTracker()
    gateway = ModelGateway(tracker=tracker, retry_policy=policy, sleep=sleeps.append)
    out = gateway.complete([Message("user", "hi")], model="openai/gpt-4o-mini", agent="writer")
    assert out == "recovered"
    assert len(calls) == 3
    assert sleeps == [1.0, 2.0]  # exponential backoff, bounded attempts
    assert tracker.records()[0].attempts == 3  # only the successful call is recorded


def test_retry_exhaustion_raises_last_transient_error(monkeypatch, openai_key):
    def always_rate_limited(**kwargs):
        raise rate_limit_error()

    monkeypatch.setattr(litellm, "completion", always_rate_limited)
    sleeps = []
    policy = RetryPolicy(
        max_attempts=3, base_delay_seconds=1.0, max_delay_seconds=30.0, jitter_ratio=0.0
    )
    gateway = ModelGateway(tracker=UsageTracker(), retry_policy=policy, sleep=sleeps.append)
    with pytest.raises(RateLimitError):
        gateway.complete([Message("user", "hi")], model="openai/gpt-4o-mini")
    assert len(sleeps) == 2  # no sleep after the final attempt


def test_permanent_error_is_not_retried(monkeypatch, openai_key):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        raise litellm_exc.AuthenticationError("bad key", llm_provider="openai", model="gpt-4o-mini")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    sleeps = []
    gateway = ModelGateway(tracker=UsageTracker(), sleep=sleeps.append)
    with pytest.raises(AuthenticationError):
        gateway.complete([Message("user", "hi")], model="openai/gpt-4o-mini")
    assert len(calls) == 1 and sleeps == []


def test_retry_after_header_is_respected(monkeypatch, openai_key):
    def fake_completion(**kwargs):
        raise rate_limit_error(headers={"retry-after": "7"})

    monkeypatch.setattr(litellm, "completion", fake_completion)
    sleeps = []
    policy = RetryPolicy(
        max_attempts=2, base_delay_seconds=1.0, max_delay_seconds=30.0, jitter_ratio=0.0
    )
    gateway = ModelGateway(tracker=UsageTracker(), retry_policy=policy, sleep=sleeps.append)
    with pytest.raises(RateLimitError):
        gateway.complete([Message("user", "hi")], model="openai/gpt-4o-mini")
    assert sleeps == [7.0]


def test_missing_credentials_raise_before_any_call(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    called = []
    monkeypatch.setattr(litellm, "completion", lambda **kwargs: called.append(kwargs))
    gateway = ModelGateway(tracker=UsageTracker())
    with pytest.raises(MissingCredentialsError) as exc_info:
        gateway.complete([Message("user", "hi")], model="openai/gpt-4o-mini")
    assert called == []  # credentials are only required for providers actually used
    assert "OPENAI_API_KEY" in str(exc_info.value)


def test_ollama_needs_no_key_and_uses_default_base(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_API_BASE", raising=False)
    calls = []
    def fake_completion(**kwargs):
        calls.append(kwargs)
        return fake_response(model="ollama/llama3.2")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.0)
    tracker = UsageTracker()
    gateway = ModelGateway(tracker=tracker)
    out = gateway.complete([Message("user", "hi")], model="ollama/llama3.2", agent="reviewer")
    assert out == "ok"
    assert calls[0]["api_base"] == "http://localhost:11434"
    assert "api_key" not in calls[0]
    record = tracker.records()[0]
    assert record.provider == "ollama"
    assert record.cost_usd is None  # local model: cost unknown, not zero


def test_bind_returns_per_agent_client_with_own_model(monkeypatch, openai_key):
    monkeypatch.setenv("MISTRAL_API_KEY", "mistral-key")
    calls = []
    def fake_completion(**kwargs):
        calls.append(kwargs)
        return fake_response(model=kwargs["model"])

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.002)
    tracker = UsageTracker()
    gateway = ModelGateway(tracker=tracker)
    client = gateway.bind(
        "mistral/mistral-large-latest", agent="developer", temperature=0.5, timeout_seconds=12.0
    )
    assert isinstance(client, GatewayLLMClient)
    out = client.complete([Message("user", "code")], run_id="run-9")
    assert out == "ok"
    assert calls[0]["model"] == "mistral/mistral-large-latest"
    assert calls[0]["api_key"] == "mistral-key"
    assert calls[0]["temperature"] == 0.5
    assert calls[0]["timeout"] == 12.0
    record = tracker.records()[0]
    assert record.agent == "developer"
    assert record.run_id == "run-9"
    assert record.provider == "mistral"


def test_bind_rejects_invalid_model():
    gateway = ModelGateway(tracker=UsageTracker())
    with pytest.raises(UnknownProviderError):
        gateway.bind("cohere/command-r")
    with pytest.raises(UnknownProviderError):
        gateway.bind("gpt-4o-mini")


def test_retry_after_seconds_helper():
    exc = SimpleNamespace(response=SimpleNamespace(headers={"retry-after": "3.5"}))
    assert _retry_after_seconds(exc) == 3.5
    assert _retry_after_seconds(SimpleNamespace()) is None
    assert _retry_after_seconds(ValueError("x")) is None
