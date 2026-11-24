"""Phase 2 audit — edge-case regression tests (items 1-4 of the final audit).

E1: nested parallel delegation must never exceed max_parallel_agents in ACTIVE
    model requests (semaphore is taken per model call, never while waiting).
E2: per-agent limit check+reservation must be atomic under concurrency.
E3: physical provider request accounting for retries / failures / timeouts /
    cancellation.
E4: optional hard cap on physical provider requests vs the hard logical budget.
"""

import asyncio
import json
from types import SimpleNamespace

import litellm
import litellm.exceptions as litellm_exc
import pytest

from geosgan_hive.agent_config import Limits, Permissions
from geosgan_hive.agents import Agent
from geosgan_hive.collaboration import CollaborationEngine
from geosgan_hive.errors import RateLimitError
from geosgan_hive.gateway import ModelGateway
from geosgan_hive.llm import FakeLLM
from geosgan_hive.messages import Message
from geosgan_hive.retry import RetryPolicy
from geosgan_hive.run_context import CollaborationPolicy, RunContext, TaskStatus
from geosgan_hive.usage import UsageTracker


def policy(**kwargs) -> CollaborationPolicy:
    kwargs.setdefault("allow_agent_to_agent", True)
    return CollaborationPolicy(**kwargs)


def make_agent(name, llm, engine, *, allow_delegation=False, allowed_agents=(), limits=None):
    return Agent(
        name=name,
        description=f"{name} agent",
        system_prompt=f"You are {name}.",
        llm=llm,
        permissions=Permissions(
            allow_delegation=allow_delegation, allowed_agents=list(allowed_agents)
        ),
        limits=limits or Limits(),
        collaboration=engine,
    )


def delegated(handler):
    """Wrap a final-answer handler: first call emits a delegation batch."""
    async def wrapper(messages, plan=None):
        if any("Delegation result from" in m.content for m in messages):
            return await handler(messages)
        return plan

    return wrapper


# ---------- E1: active model requests never exceed the concurrency limit ----------

def test_nested_parallel_never_exceeds_active_model_request_limit(monkeypatch):
    """max_parallel_agents bounds ACTIVE model requests at the gateway dispatch
    boundary, even with nested parallel delegation (no slot is held while an
    agent waits for its subtasks, so this can also never deadlock)."""
    from types import SimpleNamespace

    import litellm

    from geosgan_hive.gateway import ModelGateway

    peak = {"cur": 0, "max": 0}
    plans = {
        "root": ["b", "c"],
        "b": ["d", "e"],
        "c": ["f", "g"],
    }

    async def fake_acompletion(**kwargs):
        name = kwargs["model"].split("/", 1)[1]
        peak["cur"] += 1
        peak["max"] = max(peak["max"], peak["cur"])
        await asyncio.sleep(0.02)
        peak["cur"] -= 1
        msgs = kwargs["messages"]
        if any("Delegation result from" in m["content"] for m in msgs):
            content = f"{name} final"
        elif name in plans:
            content = json.dumps({"delegations": [
                {"target": t, "task": f"{name}->{t}"} for t in plans[name]
            ]})
        else:
            content = f"{name} ok"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            model=kwargs["model"],
        )

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.001)

    engine = CollaborationEngine(policy=policy(max_parallel_agents=2, max_delegation_depth=2))
    gateway = ModelGateway(tracker=engine.tracker)
    gateway.set_context_lookup(engine.context_for)
    names = ["root", "b", "c", "d", "e", "f", "g"]
    allowed = {"root": ["b", "c"], "b": ["d", "e"], "c": ["f", "g"]}
    agents = {
        name: make_agent(
            name, gateway.bind(f"ollama/{name}", agent=name), engine,
            allow_delegation=name in allowed, allowed_agents=allowed.get(name, []),
        )
        for name in names
    }
    engine.register(agents)
    ctx = engine.context_for("e1")
    ctx.create_task(
        task_id="t0", parent_task_id=None, agent="supervisor", description="root", depth=0
    )

    async def main():
        return await asyncio.wait_for(
            agents["root"].arun("tree", run_id="e1", task_id="t0", run_context=ctx),
            timeout=10.0,
        )

    result = asyncio.run(main())
    assert result.answer == "root final"
    assert peak["max"] <= 2, f"active model requests peaked at {peak['max']} > 2"
    assert peak["max"] == 2  # parallelism was actually used
    leaves = [t for t in ctx.tasks() if t.agent in ("d", "e", "f", "g")]
    assert len(leaves) == 4 and all(t.status == TaskStatus.COMPLETED for t in leaves)
    # root/b/c make 2 calls each (delegate, then final), leaves 1 each: 3*2+4 = 10
    assert ctx.physical_reserved == 10


# ---------- E2: atomic per-agent check+reserve under concurrency ----------

def test_per_agent_limit_atomic_under_concurrent_delegations():
    engine = CollaborationEngine(policy=policy(max_handoffs_per_run=10, max_parallel_agents=8))

    async def slow(messages):
        await asyncio.sleep(0.02)
        return "analyst answer"

    agents = {
        "researcher": make_agent("researcher", FakeLLM(responses=["r"]), engine,
                                 allow_delegation=True, allowed_agents=["analyst"]),
        "analyst": make_agent("analyst", FakeLLM(handler=slow), engine,
                              limits=Limits(max_calls_per_run=2)),
    }
    engine.register(agents)

    async def main():
        return await engine.adelegate_many(
            [{"target": "analyst", "task": f"q{i}"} for i in range(5)],
            run_id="e2", source_agent="researcher",
        )

    results = asyncio.run(main())
    ctx = engine._contexts.get("e2")
    assert ctx.agent_calls("analyst") == 2  # exactly the limit, never exceeded
    answered = [r for r in results if r.result == "analyst answer"]
    limited = [r for r in results if "Per-agent limit" in (r.result or "")]
    assert len(answered) == 2 and len(limited) == 3
    engine.close_run("e2")


# ---------- E3: physical accounting for retries, failures, timeouts, cancellation ----------

def _ok_response(model="openai/gpt-4o-mini"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        model=model,
    )


def test_physical_requests_count_failed_retry_attempts(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-audit-test")
    monkeypatch.setattr(
        litellm, "completion",
        lambda **kwargs: (_ for _ in ()).throw(
            litellm_exc.RateLimitError("429", llm_provider="openai", model="gpt-4o-mini")
        ),
    )
    tracker = UsageTracker()
    gateway = ModelGateway(
        tracker=tracker,
        retry_policy=RetryPolicy(max_attempts=3, base_delay_seconds=0, max_delay_seconds=0,
                                 jitter_ratio=0),
    )
    run_ctx = RunContext("e3-fail", CollaborationPolicy(), tracker)
    with pytest.raises(RateLimitError):
        gateway.complete([Message(role="user", content="hi")], model="openai/gpt-4o-mini",
                         agent="researcher", run_id="e3-fail")
    # the failed logical call still produced 3 physical provider requests
    assert tracker.physical_requests("e3-fail") == 3
    assert run_ctx.physical_requests == 3
    (record,) = tracker.records()
    assert record.attempts == 3
    assert record.error == "RateLimitError"  # failure is recorded, not silently dropped
    assert record.cost_usd is None  # unknown on failure: None, never 0


def test_physical_requests_count_timeout_retries_then_success(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-audit-test")
    calls = []

    def fake_completion(**kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise litellm_exc.Timeout("t/o", model="gpt-4o-mini", llm_provider="openai")
        return _ok_response()

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.001)
    tracker = UsageTracker()
    gateway = ModelGateway(
        tracker=tracker,
        retry_policy=RetryPolicy(max_attempts=3, base_delay_seconds=0, max_delay_seconds=0,
                                 jitter_ratio=0),
    )
    out = gateway.complete([Message(role="user", content="hi")], model="openai/gpt-4o-mini",
                           run_id="e3-timeout")
    assert out == "ok"
    assert len(calls) == 2
    (record,) = tracker.records()
    assert record.attempts == 2 and record.error is None
    assert tracker.physical_requests("e3-timeout") == 2  # 1 logical call, 2 physical


def test_physical_requests_count_async_cancellation(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-audit-test")

    async def slow_acompletion(**kwargs):
        await asyncio.sleep(30)
        return _ok_response()

    monkeypatch.setattr(litellm, "acompletion", slow_acompletion)
    tracker = UsageTracker()
    gateway = ModelGateway(tracker=tracker)

    async def main():
        task = asyncio.create_task(
            gateway.acomplete([Message(role="user", content="hi")], model="openai/gpt-4o-mini",
                              run_id="e3-cancel")
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    (record,) = tracker.records()
    assert record.attempts == 1
    assert record.error == "cancelled"
    assert tracker.physical_requests("e3-cancel") == 1


# ---------- E4: hard logical budget vs optional physical-request cap ----------

def test_hard_logical_budget_and_optional_physical_cap_are_distinct(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-audit-test")
    monkeypatch.setattr(litellm, "completion", lambda **kwargs: _ok_response())
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.001)
    tracker = UsageTracker()
    gateway = ModelGateway(tracker=tracker)
    # physical cap = 2, logical budget = 200 (default): the physical cap binds first
    engine = CollaborationEngine(
        policy=policy(max_provider_requests_per_run=2), tracker=tracker
    )
    agent = make_agent("researcher", gateway.bind("openai/gpt-4o-mini", agent="researcher"), engine)
    engine.register({"researcher": agent})
    ctx = engine.context_for("e4")
    ctx.create_task(
        task_id="t0", parent_task_id=None, agent="supervisor", description="root", depth=0
    )
    # agent emits tool calls; with tools allowed each loop step is one model call
    from geosgan_hive.tools import CalculatorTool, ToolRegistry
    agent.tools = ToolRegistry([CalculatorTool()])
    agent.permissions = Permissions(allow_tools=True, allowed_tools=["calculator"])
    agent.llm = gateway.bind("openai/gpt-4o-mini", agent="researcher")
    import json as _json
    agent.llm.handler = None  # gateway-bound client, not FakeLLM
    # drive 3 logical calls via a wrapping fake that delegates to the gateway client
    calls = {"n": 0}
    real_complete = agent.llm.complete

    def counting_complete(messages, *, temperature=0.2, run_id=None):
        calls["n"] += 1
        return real_complete(messages, temperature=temperature, run_id=run_id)

    # the gateway client returns "ok" (plain text) so the agent would stop after 1 call;
    # use max_tool_steps loop instead: patch the client to always emit a tool call
    def tool_complete(messages, *, temperature=0.2, run_id=None):
        calls["n"] += 1
        tracker.record  # noqa
        return _json.dumps({"tool": "calculator", "args": {"expression": "1+1"}})

    agent.llm = FakeLLM(handler=tool_complete)
    # but FakeLLM doesn't record usage; emulate gateway physical records manually per call
    orig = agent.llm.complete

    def wrapped(messages, *, temperature=0.2, run_id=None):
        out = orig(messages, temperature=temperature, run_id=run_id)
        from datetime import UTC, datetime

        from geosgan_hive.usage import UsageRecord as _UR
        tracker.record(_UR(
            agent="researcher", model="openai/gpt-4o-mini", provider="openai",
            prompt_tokens=1, completion_tokens=1, total_tokens=2, cost_usd=0.001,
            latency_ms=1.0, attempts=1, timestamp=datetime.now(UTC).isoformat(), run_id=run_id,
        ))
        return out

    agent.llm.complete = wrapped
    result = agent.run("loop", run_id="e4", task_id="t0", run_context=ctx)
    assert "Provider-request limit" in result.answer or "provider-request" in result.answer.lower()
    assert ctx.model_calls == 2  # logical calls made
    assert ctx.physical_requests == 2  # physical cap stopped the run
    assert calls["n"] == 2
