import threading
from types import SimpleNamespace

import litellm
import litellm.exceptions as litellm_exc
import pytest

from geosgan_hive.collaboration import CollaborationEngine
from geosgan_hive.errors import BudgetExceededError
from geosgan_hive.gateway import ModelGateway
from geosgan_hive.messages import Message
from geosgan_hive.retry import RetryPolicy
from geosgan_hive.run_context import CollaborationPolicy, RunContext
from geosgan_hive.usage import UsageTracker


def ok_response(model="ollama/llama3.2"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        model=model,
    )


def make_gateway(monkeypatch, tracker, policy=None, with_ctx=True):
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.001)
    engine = CollaborationEngine(policy=policy or CollaborationPolicy(allow_agent_to_agent=True),
                                 tracker=tracker)
    gateway = ModelGateway(tracker=tracker)
    if with_ctx:
        gateway.set_context_lookup(engine.context_for)
    return gateway, engine


def test_every_retry_attempt_reserves_a_physical_request(monkeypatch):
    attempts = []

    def fake_completion(**kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise litellm_exc.RateLimitError("429", llm_provider="ollama", model="llama3.2")
        return ok_response()

    monkeypatch.setattr(litellm, "completion", fake_completion)
    tracker = UsageTracker()
    gateway, engine = make_gateway(monkeypatch, tracker)
    gateway.retry_policy = RetryPolicy(max_attempts=3, base_delay_seconds=0, max_delay_seconds=0,
                                       jitter_ratio=0)
    out = gateway.complete([Message("user", "hi")], model="ollama/llama3.2", run_id="r1")
    assert out == "ok"
    ctx = engine._contexts["r1"]
    assert len(attempts) == 3
    assert ctx.physical_reserved == 3  # one reservation per physical dispatch (incl. retries)
    assert ctx.model_calls == 0  # logical reservation belongs to the agent, not the gateway
    assert tracker.physical_requests("r1") == 3  # confirmed attempts match reservations


def test_physical_cap_blocks_dispatch_before_request(monkeypatch):
    monkeypatch.setattr(litellm, "completion", lambda **kwargs: ok_response())
    tracker = UsageTracker()
    policy = CollaborationPolicy(allow_agent_to_agent=True, max_provider_requests_per_run=2)
    gateway, engine = make_gateway(monkeypatch, tracker, policy=policy)
    gateway.complete([Message("user", "a")], model="ollama/llama3.2", run_id="r2")
    gateway.complete([Message("user", "b")], model="ollama/llama3.2", run_id="r2")
    with pytest.raises(BudgetExceededError):
        gateway.complete([Message("user", "c")], model="ollama/llama3.2", run_id="r2")
    ctx = engine._contexts["r2"]
    assert ctx.physical_reserved == 2  # the blocked dispatch reserved nothing
    assert len(tracker.records()) == 2  # only dispatched requests were confirmed


def test_reserve_physical_request_is_atomic_under_threads():
    ctx = RunContext("r", CollaborationPolicy(max_provider_requests_per_run=50))
    outcomes = []
    lock = threading.Lock()

    def worker():
        ok = ctx.reserve_physical_request()
        with lock:
            outcomes.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(200)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert ctx.physical_reserved == 50
    assert sum(outcomes) == 50
    assert ctx.reserve_physical_request() is False


def test_unlimited_physical_reservation_when_cap_is_none():
    ctx = RunContext("r", CollaborationPolicy())  # max_provider_requests_per_run=None
    assert all(ctx.reserve_physical_request() for _ in range(1000))
    assert ctx.physical_reserved == 1000
    assert ctx.physical_requests == 0  # reserved != confirmed; nothing dispatched yet


def test_gateway_without_context_lookup_dispatches_without_reservation(monkeypatch):
    monkeypatch.setattr(litellm, "completion", lambda **kwargs: ok_response())
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.001)
    gateway = ModelGateway(tracker=UsageTracker())  # no context_lookup
    assert gateway.complete([Message("user", "hi")], model="ollama/llama3.2") == "ok"


def test_counters_snapshot_and_restore_without_double_counting():
    ctx = RunContext("r", CollaborationPolicy())
    ctx.reserve_model_call_for("researcher", None, 10)
    ctx.reserve_model_call_for("researcher", None, 10)
    ctx.reserve_physical_request()
    snap = ctx.counters_snapshot()
    assert snap == {"model_calls": 2, "physical_reserved": 1, "handoffs": 0,
                    "agent_calls": {"researcher": 2},
                    "tool_calls": {"total": 0, "per_agent": {}, "per_tool": {}}}
    # simulate process restart: fresh context restored from the snapshot
    ctx2 = RunContext("r", CollaborationPolicy())
    ctx2.restore_counters(snap)
    assert ctx2.model_calls == 2 and ctx2.physical_reserved == 1
    assert ctx2.agent_calls("researcher") == 2
    # a resumed logical call is NOT counted again
    ctx2.restore_counters(snap)
    assert ctx2.model_calls == 2
