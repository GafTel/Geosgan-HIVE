import asyncio
import json
import time

import pytest

from geosgan_hive.agent_config import Permissions
from geosgan_hive.agents import Agent
from geosgan_hive.collaboration import CollaborationEngine, DelegationStatus
from geosgan_hive.llm import FakeLLM
from geosgan_hive.orchestrator import Supervisor
from geosgan_hive.run_context import CollaborationPolicy, EventType, TaskStatus
from geosgan_hive.usage import UsageTracker


def make_policy(**kwargs) -> CollaborationPolicy:
    kwargs.setdefault("allow_agent_to_agent", True)
    return CollaborationPolicy(**kwargs)


def make_agent(name, llm, *, allow_delegation=False, allowed_agents=(), engine=None):
    return Agent(
        name=name,
        description=f"{name} agent",
        system_prompt=f"You are {name}.",
        llm=llm,
        permissions=Permissions(
            allow_delegation=allow_delegation, allowed_agents=list(allowed_agents)
        ),
        collaboration=engine,
    )


def make_async_team(policy=None, **llms):
    policy = policy or make_policy()
    engine = CollaborationEngine(policy=policy, tracker=UsageTracker())
    agents = {
        name: make_agent(
            name, llms.get(name, FakeLLM(responses=[f"{name} answer"])),
            allow_delegation=(name == "researcher"),
            allowed_agents=["analyst", "reviewer", "developer"] if name == "researcher" else [],
            engine=engine,
        )
        for name in ("researcher", "analyst", "reviewer", "developer")
    }
    engine.register(agents)
    return engine, agents


def test_adelegate_uses_native_async_path():
    engine, agents = make_async_team()
    result = asyncio.run(
        engine.adelegate({"target": "analyst", "task": "x"}, run_id="r", source_agent="researcher")
    )
    assert result.status == DelegationStatus.COMPLETED and result.result == "analyst answer"
    assert len(agents["analyst"].llm.acomplete_calls) == 1  # native acomplete used
    assert agents["analyst"].llm.calls == []  # sync path untouched


def test_adelegate_many_parallel_deterministic_order_and_concurrency_cap(monkeypatch):
    """Concurrency is bounded at the gateway dispatch boundary: peak ACTIVE model
    requests never exceed max_parallel_agents, and results keep input order."""
    from types import SimpleNamespace

    import litellm

    from geosgan_hive.gateway import ModelGateway

    peak = {"current": 0, "max": 0}
    delays = {"analyst": 0.08, "reviewer": 0.02, "developer": 0.05}

    async def fake_acompletion(**kwargs):
        agent = kwargs["model"].split("/", 1)[1]
        peak["current"] += 1
        peak["max"] = max(peak["max"], peak["current"])
        await asyncio.sleep(delays[agent])  # inside the dispatched request
        peak["current"] -= 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=f"{agent} done"))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            model=kwargs["model"],
        )

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.001)

    policy = make_policy(max_parallel_agents=2)
    engine = CollaborationEngine(policy=policy, tracker=UsageTracker())
    gateway = ModelGateway(tracker=engine.tracker)
    gateway.set_context_lookup(engine.context_for)  # dispatch reserves physical + slot
    agents = {}
    for name in ("researcher", "analyst", "reviewer", "developer"):
        agents[name] = make_agent(
            name, gateway.bind(f"ollama/{name}", agent=name), engine=engine,
            allow_delegation=(name == "researcher"),
            allowed_agents=["analyst", "reviewer", "developer"] if name == "researcher" else [],
        )
    engine.register(agents)
    proposals = [
        {"target": "analyst", "task": "slow"},
        {"target": "reviewer", "task": "fast"},
        {"target": "developer", "task": "mid"},
    ]
    started = time.perf_counter()
    results = asyncio.run(
        engine.adelegate_many(proposals, run_id="r", source_agent="researcher")
    )
    elapsed = time.perf_counter() - started
    # deterministic input ordering despite different completion times
    assert [r.target_agent for r in results] == ["analyst", "reviewer", "developer"]
    assert [r.result for r in results] == ["analyst done", "reviewer done", "developer done"]
    assert all(r.status == DelegationStatus.COMPLETED for r in results)
    # bounded concurrency AND parallelism proven by observed overlap:
    # peak == 2 means two dispatched requests were in flight concurrently
    assert peak["max"] == 2
    assert elapsed < 1.0  # generous bound; sequential would take 0.15s+ of model time
    # each dispatched request reserved one physical attempt
    ctx = engine._contexts.get("r")
    assert ctx.physical_reserved == 3


def test_adelegate_many_dedupes_identical_proposals():
    calls = []

    async def handler(messages):
        calls.append(1)
        await asyncio.sleep(0)
        return "once"

    engine, agents = make_async_team(analyst=FakeLLM(handler=handler))
    proposals = [
        {"target": "analyst", "task": "same"},
        {"target": "analyst", "task": "same"},
        {"target": "reviewer", "task": "other"},
    ]
    results = asyncio.run(
        engine.adelegate_many(proposals, run_id="r", source_agent="researcher")
    )
    assert len(calls) == 1  # duplicate executed only once
    assert results[0] is results[1]
    assert [r.target_agent for r in results] == ["analyst", "analyst", "reviewer"]


def test_adelegate_timeout_returns_cancelled_result():
    async def slow(messages):
        await asyncio.sleep(5)
        return "never"

    engine, _ = make_async_team(analyst=FakeLLM(handler=slow))
    ctx = engine.context_for("r")
    result = asyncio.run(
        engine.adelegate({"target": "analyst", "task": "x"}, run_id="r",
                         source_agent="researcher", timeout=0.05)
    )
    assert result.status == DelegationStatus.CANCELLED
    assert "timed out" in result.error
    (node,) = [t for t in ctx.tasks() if t.agent == "analyst"]
    assert node.status == TaskStatus.CANCELLED
    assert ctx.events(EventType.TASK_FAILED)


def test_cancelled_delegation_marks_task_and_reraises():
    async def slow(messages):
        await asyncio.sleep(10)
        return "never"

    engine, _ = make_async_team(analyst=FakeLLM(handler=slow))
    ctx = engine.context_for("r")

    async def main():
        task = asyncio.create_task(
            engine.adelegate({"target": "analyst", "task": "x"}, run_id="r",
                             source_agent="researcher")
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    (node,) = [t for t in ctx.tasks() if t.agent == "analyst"]
    assert node.status == TaskStatus.CANCELLED
    failed = ctx.events(EventType.TASK_FAILED)
    assert failed and failed[0].metadata.get("status") == "cancelled"


def test_legacy_sync_only_adapter_uses_thread_fallback():
    class SyncOnlyLLM:
        def __init__(self):
            self.calls = 0

        def complete(self, messages, *, temperature=0.2, run_id=None):
            self.calls += 1
            return "sync-only answer"

    sync_llm = SyncOnlyLLM()
    policy = make_policy()
    engine = CollaborationEngine(policy=policy, tracker=UsageTracker())
    analyst = make_agent("analyst", sync_llm, engine=engine)
    researcher = make_agent(
        "researcher", FakeLLM(responses=["x"]), allow_delegation=True,
        allowed_agents=["analyst"], engine=engine,
    )
    engine.register({"analyst": analyst, "researcher": researcher})
    result = asyncio.run(
        engine.adelegate({"target": "analyst", "task": "x"}, run_id="r", source_agent="researcher")
    )
    assert result.status == DelegationStatus.COMPLETED
    assert result.result == "sync-only answer"
    assert sync_llm.calls == 1  # ran in a thread, but the native path is used when available


def test_event_loop_not_blocked_during_parallel_delegation():
    ticks = {"n": 0}

    async def ticker():
        while True:
            await asyncio.sleep(0.01)
            ticks["n"] += 1

    async def slow(messages):
        await asyncio.sleep(0.1)
        return "done"

    engine, _ = make_async_team(analyst=FakeLLM(handler=slow), reviewer=FakeLLM(handler=slow))

    async def main():
        t = asyncio.create_task(ticker())
        results = await engine.adelegate_many(
            [{"target": "analyst", "task": "a"}, {"target": "reviewer", "task": "b"}],
            run_id="r", source_agent="researcher",
        )
        t.cancel()
        return results

    results = asyncio.run(main())
    assert len(results) == 2
    assert ticks["n"] >= 5  # the loop kept ticking while delegations ran


def test_agent_arun_batch_delegation_parallel_ordered():
    async def researcher_handler(messages):
        if any("Delegation result from" in m.content for m in messages):
            return "researcher final"
        return json.dumps({"delegations": [
            {"target": "analyst", "task": "A"},
            {"target": "reviewer", "task": "B"},
        ]})

    engine, agents = make_async_team(researcher=FakeLLM(handler=researcher_handler))
    ctx = engine.context_for("r")
    ctx.create_task(task_id="t0", parent_task_id=None, agent="supervisor",
                    description="root", depth=0)
    result = asyncio.run(
        agents["researcher"].arun("research", run_id="r", task_id="t0", run_context=ctx)
    )
    assert result.answer == "researcher final"
    assert [h.target_agent for h in result.handoffs] == ["analyst", "reviewer"]  # ordered
    assert all(h.status == DelegationStatus.COMPLETED for h in result.handoffs)
    # both delegation results were fed back before the final answer
    second_call = agents["researcher"].llm.acomplete_calls[1]
    assert any("Delegation result from 'analyst'" in m.content for m in second_call)
    assert any("Delegation result from 'reviewer'" in m.content for m in second_call)
    # native async path used
    assert len(agents["researcher"].llm.acomplete_calls) == 2
    assert agents["researcher"].llm.calls == []


def test_supervisor_arun_end_to_end_and_cleanup():
    def sup_handler(messages):
        if any(m.agent == "researcher" for m in messages):
            return json.dumps({"next": "FINISH", "answer": "async final"})
        return json.dumps({"next": "researcher", "reason": "go"})

    engine, agents = make_async_team()
    ctx = engine.context_for("ra")
    supervisor = Supervisor(agents, FakeLLM(handler=sup_handler), max_steps=5,
                            collaboration=engine)
    result = asyncio.run(supervisor.arun("task", run_id="ra"))
    assert result.route == ["researcher"]
    assert result.final_answer == "async final"
    assert engine.active_runs() == 0  # cleaned up after successful async run
    assert {t.agent for t in ctx.tasks()} == {"supervisor", "researcher"}
    assert ctx.model_calls == 3  # supervisor(2) + researcher(1)
    # native async used on both levels
    assert len(supervisor.llm.acomplete_calls) == 2
    assert len(agents["researcher"].llm.acomplete_calls) == 1


def test_supervisor_arun_cancellation_cleans_up():
    async def slow_sup(messages):
        await asyncio.sleep(10)
        return json.dumps({"next": "FINISH", "answer": "never"})

    engine, agents = make_async_team()
    ctx = engine.context_for("rc")
    supervisor = Supervisor(agents, FakeLLM(handler=slow_sup), max_steps=5,
                            collaboration=engine)

    async def main():
        task = asyncio.create_task(supervisor.arun("task", run_id="rc"))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    assert engine.active_runs() == 0  # cleaned up on the cancellation path too
    root = [t for t in ctx.tasks() if t.agent == "supervisor"][0]
    assert root.status == TaskStatus.CANCELLED
    assert ctx.events(EventType.TASK_FAILED)


def test_shared_budget_is_atomic_under_concurrent_delegations():
    policy = make_policy(max_model_calls_per_run=4, max_parallel_agents=3)
    engine = CollaborationEngine(policy=policy, tracker=UsageTracker())

    def busy_llm(name):
        async def handler(messages):
            await asyncio.sleep(0.01)  # force interleaving
            return f"{name} ok"

        return FakeLLM(handler=handler)

    agents = {}
    for name in ("researcher", "analyst", "reviewer"):
        agents[name] = make_agent(
            name, busy_llm(name),
            allow_delegation=(name == "researcher"),
            allowed_agents=["analyst", "reviewer", "developer"] if name == "researcher" else [],
            engine=engine,
        )
    engine.register(agents)
    ctx = engine.context_for("r")
    proposals = [{"target": t, "task": f"task-{i}"} for i, t in enumerate(
        ["analyst", "reviewer", "analyst", "reviewer", "analyst"]
    )]
    results = asyncio.run(
        engine.adelegate_many(proposals, run_id="r", source_agent="researcher")
    )
    # every agent reserves exactly one call before its request; budget caps at 4
    assert ctx.model_calls == 4  # atomic: no overcount, no double count under concurrency
    budgeted = [r for r in results if "model-call limit" in (r.result or "")]
    answered = [r for r in results if r.status == DelegationStatus.COMPLETED
                and "model-call limit" not in (r.result or "")]
    assert len(answered) == 4 and len(budgeted) == 1  # 4 ran, 1 stopped by the shared budget
