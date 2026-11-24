"""Phase 2 final audit: regression probes for concurrency, accounting and safety.

Each test targets one audit requirement. Written failing-first against the
pre-audit implementation, then fixed in the same branch.
"""

import asyncio
import json
import time
from datetime import UTC, datetime
from types import SimpleNamespace

import litellm
import litellm.exceptions as litellm_exc
import pytest
from pydantic import ValidationError

from geosgan_hive.agent_config import Limits, Permissions
from geosgan_hive.agents import Agent
from geosgan_hive.collaboration import (
    CollaborationEngine,
    DelegationStatus,
    HandoffRequest,
)
from geosgan_hive.errors import GatewayError
from geosgan_hive.llm import FakeLLM
from geosgan_hive.run_context import CollaborationPolicy, EventType, TaskStatus
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


def chain_team(engine, handlers):
    agents = {}
    allowed = {
        "researcher": ["analyst"],
        "analyst": ["reviewer"],
        "reviewer": [],
    }
    for name, handler in handlers.items():
        agents[name] = make_agent(
            name,
            FakeLLM(handler=handler),
            engine,
            allow_delegation=name in allowed and bool(allowed[name]),
            allowed_agents=allowed.get(name, []),
        )
    engine.register(agents)
    return agents


# ---------- AUDIT 2: semaphore deadlock when all slots wait on children ----------

def test_nested_delegation_with_single_slot_completes():
    """max_parallel_agents=1 + nested delegation must not deadlock/stall:
    a child delegation inherits the ancestor's slot instead of needing a new one."""
    engine = CollaborationEngine(policy=policy(max_parallel_agents=1, max_delegation_depth=2,
                                                default_delegation_timeout_seconds=5.0))

    async def researcher(messages):
        if any("Delegation result from" in m.content for m in messages):
            return "researcher final"
        return json.dumps({"delegate": {"target": "analyst", "task": "step 2"}})

    async def analyst(messages):
        if any("Delegation result from" in m.content for m in messages):
            return "analyst final"
        return json.dumps({"delegate": {"target": "reviewer", "task": "step 3"}})

    async def reviewer(messages):
        return "reviewer answer"

    agents = chain_team(
        engine, {"researcher": researcher, "analyst": analyst, "reviewer": reviewer}
    )
    ctx = engine.context_for("deadlock-1")
    ctx.create_task(
        task_id="t0", parent_task_id=None, agent="supervisor", description="root", depth=0
    )

    async def main():
        return await asyncio.wait_for(
            agents["researcher"].arun("chain", run_id="deadlock-1", task_id="t0", run_context=ctx),
            timeout=8.0,
        )

    started = time.perf_counter()
    result = asyncio.run(main())
    elapsed = time.perf_counter() - started
    assert result.answer == "researcher final"
    assert [h.status for h in result.handoffs] == [DelegationStatus.COMPLETED]
    assert elapsed < 5.0  # no delegation-timeout stall
    # the nested reviewer task also completed
    statuses = {t.agent: t.status for t in ctx.tasks()}
    assert statuses["reviewer"] == TaskStatus.COMPLETED


def test_batch_where_every_slot_waits_on_its_own_child():
    """max_parallel=2: A delegates to B and C (both slots); B and C each delegate
    further. All four leaf subtasks must still run (children inherit slots)."""
    engine = CollaborationEngine(policy=policy(max_parallel_agents=2, max_delegation_depth=2))

    async def researcher(messages):
        if any("Delegation result from" in m.content for m in messages):
            return "researcher final"
        return json.dumps({"delegations": [
            {"target": "analyst", "task": "b-work"},
            {"target": "reviewer", "task": "c-work"},
        ]})

    async def analyst(messages):
        if any("Delegation result from" in m.content for m in messages):
            return "analyst final"
        await asyncio.sleep(0.02)
        return json.dumps({"delegate": {"target": "developer", "task": "b-child"}})

    async def reviewer(messages):
        if any("Delegation result from" in m.content for m in messages):
            return "reviewer final"
        await asyncio.sleep(0.02)
        return json.dumps({"delegate": {"target": "developer", "task": "c-child"}})

    async def developer(messages):
        await asyncio.sleep(0.02)
        return "developer answer"

    agents = {}
    for name, handler, deleg, targets in [
        ("researcher", researcher, True, ["analyst", "reviewer"]),
        ("analyst", analyst, True, ["developer"]),
        ("reviewer", reviewer, True, ["developer"]),
        ("developer", developer, False, []),
    ]:
        agents[name] = make_agent(name, FakeLLM(handler=handler), engine,
                                  allow_delegation=deleg, allowed_agents=targets)
    engine.register(agents)

    ctx = engine.context_for("deadlock-2")
    ctx.create_task(
        task_id="t0", parent_task_id=None, agent="supervisor", description="root", depth=0
    )

    async def main():
        return await asyncio.wait_for(
            agents["researcher"].arun("batch", run_id="deadlock-2", task_id="t0", run_context=ctx),
            timeout=8.0,
        )

    result = asyncio.run(main())
    assert result.answer == "researcher final"
    assert len(result.handoffs) == 2
    assert all(h.status == DelegationStatus.COMPLETED for h in result.handoffs)
    # both nested child tasks ran to completion
    dev_tasks = [t for t in ctx.tasks() if t.agent == "developer"]
    assert len(dev_tasks) == 2 and all(t.status == TaskStatus.COMPLETED for t in dev_tasks)


# ---------- AUDIT 5: cancellation must not leave orphaned coroutines ----------

def test_cancelled_batch_cancels_all_children_no_orphans():
    engine = CollaborationEngine(policy=policy(max_parallel_agents=4,
                                                default_delegation_timeout_seconds=30.0))

    async def fast(messages):
        await asyncio.sleep(0.01)
        return "fast ok"

    async def hang(messages):
        await asyncio.sleep(60)
        return "never"

    agents = {
        "researcher": make_agent("researcher", FakeLLM(responses=["x"]), engine,
                                 allow_delegation=True,
                                 allowed_agents=["analyst", "reviewer", "developer"]),
        "analyst": make_agent("analyst", FakeLLM(handler=hang), engine),
        "reviewer": make_agent("reviewer", FakeLLM(handler=fast), engine),
        "developer": make_agent("developer", FakeLLM(handler=fast), engine),
    }
    engine.register(agents)
    ctx = engine.context_for("cancel-batch")

    async def main():
        task = asyncio.create_task(
            engine.adelegate_many(
                [{"target": t, "task": f"t-{t}"} for t in ("analyst", "reviewer", "developer")],
                run_id="cancel-batch", source_agent="researcher",
            )
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.05)  # let cancellations settle
        # no orphaned coroutines besides the current task
        pending = [
            t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()
        ]
        assert pending == []

    asyncio.run(main())
    statuses = {t.agent: t.status for t in ctx.tasks()}
    # the hanging subtask was cancelled; the fast ones had already completed
    assert statuses["analyst"] == TaskStatus.CANCELLED
    assert set(statuses.values()) <= {TaskStatus.COMPLETED, TaskStatus.CANCELLED}
    assert not any(t.status in (TaskStatus.RUNNING, TaskStatus.PENDING) for t in ctx.tasks())
    assert len(ctx.events(EventType.TASK_FAILED)) >= 1


def test_nested_cancellation_marks_every_level():
    engine = CollaborationEngine(policy=policy(max_parallel_agents=4,
                                                max_delegation_depth=3,
                                                default_delegation_timeout_seconds=30.0))

    async def hang(messages):
        await asyncio.sleep(60)
        return "never"

    async def middle(messages):
        if any("Delegation result from" in m.content for m in messages):
            return "middle final"
        return json.dumps({"delegate": {"target": "leaf", "task": "deep"}})

    async def researcher(messages):
        if any("Delegation result from" in m.content for m in messages):
            return "researcher final"
        return json.dumps({"delegate": {"target": "analyst", "task": "mid"}})

    agents = {
        "researcher": make_agent("researcher", FakeLLM(handler=researcher), engine,
                                 allow_delegation=True, allowed_agents=["analyst"]),
        "analyst": make_agent("analyst", FakeLLM(handler=middle), engine,
                              allow_delegation=True, allowed_agents=["leaf"]),
        "leaf": make_agent("leaf", FakeLLM(handler=hang), engine),
    }
    engine.register(agents)
    ctx = engine.context_for("cancel-nested")
    ctx.create_task(
        task_id="t0", parent_task_id=None, agent="supervisor", description="root", depth=0
    )

    async def main():
        task = asyncio.create_task(
            agents["researcher"].arun("deep", run_id="cancel-nested", task_id="t0", run_context=ctx)
        )
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.05)
        pending = [
            t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()
        ]
        assert pending == []

    asyncio.run(main())
    # every delegated task in the chain was marked CANCELLED, not left RUNNING
    # (t0 was created manually by the test and is not engine-managed)
    for node in ctx.tasks():
        if node.task_id == "t0":
            continue
        assert node.status == TaskStatus.CANCELLED, node


# ---------- AUDIT 4: per-agent limits cannot be bypassed via repeated delegation ----------

def test_per_agent_call_limit_holds_across_repeated_delegations():
    engine = CollaborationEngine(policy=policy(max_handoffs_per_run=10))
    analyst_llm = FakeLLM(responses=["a1", "a2", "a3"])
    agents = {
        "researcher": make_agent("researcher", FakeLLM(responses=["r"]), engine,
                                 allow_delegation=True, allowed_agents=["analyst"]),
        "analyst": make_agent("analyst", analyst_llm, engine,
                              limits=Limits(max_calls_per_run=2)),
    }
    engine.register(agents)
    ctx = engine.context_for("per-agent")
    for i in range(3):
        result = engine.delegate({"target": "analyst", "task": f"q{i}"}, run_id="per-agent",
                                 source_agent="researcher")
        assert result.status == DelegationStatus.COMPLETED
    # analyst was allowed only 2 model calls in the whole run, not 2 per delegation
    assert ctx.agent_calls("analyst") == 2
    assert len(analyst_llm.calls) == 2
    assert "Per-agent limit" in result.result


# ---------- AUDIT 3: logical calls vs physical provider requests (retries) ----------

def test_logical_calls_vs_physical_provider_requests(monkeypatch):
    from geosgan_hive.gateway import ModelGateway
    from geosgan_hive.retry import RetryPolicy

    monkeypatch.setenv("OPENAI_API_KEY", "sk-audit-test")
    attempts = []

    def fake_completion(**kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise litellm_exc.ServiceUnavailableError(
                "503", llm_provider="openai", model="gpt-4o-mini"
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            model="openai/gpt-4o-mini",
        )

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.001)
    tracker = UsageTracker()
    gateway = ModelGateway(tracker=tracker, retry_policy=RetryPolicy(
        max_attempts=3, base_delay_seconds=0.0, max_delay_seconds=0.0, jitter_ratio=0.0
    ))
    engine = CollaborationEngine(policy=policy(), tracker=tracker)
    agent = make_agent("researcher", gateway.bind("openai/gpt-4o-mini", agent="researcher"), engine)
    engine.register({"researcher": agent})
    ctx = engine.context_for("retry-run")
    ctx.create_task(
        task_id="t0", parent_task_id=None, agent="supervisor", description="root", depth=0
    )
    result = agent.run("hi", run_id="retry-run", task_id="t0", run_context=ctx)
    assert result.answer == "ok"
    # logical model calls reserved by the run: exactly 1
    assert ctx.model_calls == 1
    # physical provider requests (including retries): 3
    assert len(attempts) == 3
    assert ctx.physical_requests == 3
    assert tracker.records()[0].attempts == 3


# ---------- AUDIT 7: concurrent independent runs are isolated ----------

def test_concurrent_runs_do_not_share_context():
    from geosgan_hive.orchestrator import Supervisor

    def sup_handler_factory(tag):
        def handler(messages):
            if any(m.agent == "researcher" for m in messages):
                return json.dumps({"next": "FINISH", "answer": f"done-{tag}"})
            return json.dumps({"next": "researcher", "reason": "go"})

        return handler

    engine = CollaborationEngine(policy=policy())
    agents_a = {"researcher": make_agent("researcher", FakeLLM(responses=["a"]), engine)}
    agents_b = {"researcher": make_agent("researcher", FakeLLM(responses=["b"]), engine)}
    engine.register(agents_a)
    engine.register(agents_b)
    sup_a = Supervisor(agents_a, FakeLLM(handler=sup_handler_factory("a")), collaboration=engine)
    sup_b = Supervisor(agents_b, FakeLLM(handler=sup_handler_factory("b")), collaboration=engine)

    async def main():
        return await asyncio.gather(
            sup_a.arun("task A", run_id="run-A"),
            sup_b.arun("task B", run_id="run-B"),
        )

    (res_a, res_b) = asyncio.run(main())
    assert res_a.final_answer == "done-a" and res_b.final_answer == "done-b"
    assert engine.active_runs() == 0
    # contexts were distinct and are gone now; verify via fresh probes that
    # run-A and run-B never shared a context object
    ctx_probe_a = engine.context_for("run-A")
    ctx_probe_b = engine.context_for("run-B")
    assert ctx_probe_a is not ctx_probe_b
    engine.close_run("run-A")
    engine.close_run("run-B")


# ---------- AUDIT 8: atomic handoff limit + task creation under concurrency ----------

def test_handoff_limit_atomic_under_concurrency():
    engine = CollaborationEngine(policy=policy(max_handoffs_per_run=5, max_parallel_agents=8))

    async def fast(messages):
        await asyncio.sleep(0.01)
        return "ok"

    agents = {"researcher": make_agent("researcher", FakeLLM(responses=["x"]), engine,
                                       allow_delegation=True,
                                       allowed_agents=["analyst", "reviewer"])}
    for name in ("analyst", "reviewer"):
        agents[name] = make_agent(name, FakeLLM(handler=fast), engine)
    engine.register(agents)

    async def main():
        proposals = []
        for i in range(20):
            target = "analyst" if i % 2 == 0 else "reviewer"
            proposals.append({"target": target, "task": f"unique-{i}"})
        return await engine.adelegate_many(proposals, run_id="atomic", source_agent="researcher")

    results = asyncio.run(main())
    ctx = engine.context_for("atomic") if engine.active_runs() else None
    # re-fetch: adelegate_many used context "atomic"; it is still cached
    ctx = engine._contexts.get("atomic")
    assert ctx is not None
    assert ctx.handoffs == 5  # exactly the limit, no over-reservation
    approved = [r for r in results if r.status == DelegationStatus.COMPLETED]
    rejected = [r for r in results if r.status == DelegationStatus.REJECTED]
    assert len(approved) == 5 and len(rejected) == 15
    assert all("max handoffs" in r.error for r in rejected)
    task_ids = [t.task_id for t in ctx.tasks()]
    assert len(task_ids) == len(set(task_ids)) == 5  # unique, atomic task creation
    engine.close_run("atomic")


# ---------- AUDIT 9: runtime identities cannot be overridden ----------

def test_handoff_request_forbids_extra_keys():
    with pytest.raises(ValidationError):
        HandoffRequest.model_validate({
            "request_id": "q", "run_id": "r", "task_id": "t", "parent_task_id": None,
            "source_agent": "researcher", "target_agent": "analyst",
            "task_description": "x", "delegation_depth": 1,
            "source_agent_override": "supervisor",  # extra key must be rejected
        })


def test_model_metadata_cannot_override_identity_or_depth():
    engine = CollaborationEngine(policy=policy(max_delegation_depth=2))
    agents = {
        "researcher": make_agent("researcher", FakeLLM(responses=["r"]), engine,
                                 allow_delegation=True, allowed_agents=["analyst"]),
        "analyst": make_agent("analyst", FakeLLM(responses=["a"]), engine),
    }
    engine.register(agents)
    ctx = engine.context_for("meta")
    ctx.create_task(
        task_id="t0", parent_task_id=None, agent="supervisor", description="root", depth=0
    )
    malicious = {
        "target": "analyst", "task": "x",
        "request_id": "forged", "run_id": "forged", "task_id": "forged",
        "parent_task_id": "t0", "source_agent": "supervisor", "delegation_depth": 99,
    }
    result = engine.delegate(
        malicious, run_id="meta", parent_task_id="t0", source_agent="researcher"
    )
    assert result.request_id != "forged"
    assert result.run_id == "meta"
    assert result.source_agent == "researcher"  # not "supervisor"
    assert result.delegation_depth == 1  # not 99
    assert result.status == DelegationStatus.COMPLETED


# ---------- AUDIT 10: malformed / mixed actions ----------

def test_mixed_tool_and_delegation_json_is_safe():
    from geosgan_hive.tools import CalculatorTool, ToolRegistry

    engine = CollaborationEngine(policy=policy())
    llm = FakeLLM(responses=[
        # mixed action: tool wins, the delegation part is ignored (safe)
        json.dumps({"tool": "calculator", "args": {"expression": "2+2"},
                    "delegate": {"target": "analyst", "task": "x"}}),
        "done",
    ])
    agent = Agent("researcher", "d", "s", llm, ToolRegistry([CalculatorTool()]),
                  permissions=Permissions(allow_tools=True, allowed_tools=["calculator"],
                                          allow_delegation=True, allowed_agents=["analyst"]),
                  collaboration=engine)
    engine.register({
        "researcher": agent,
        "analyst": make_agent("analyst", FakeLLM(responses=["a"]), engine),
    })
    ctx = engine.context_for("mixed")
    result = agent.run("calc", run_id="mixed", run_context=ctx)
    assert result.answer == "done"
    assert len(result.tool_calls) == 1 and result.tool_calls[0].result == "4"
    assert result.handoffs == []  # delegation part was not executed


def test_malformed_delegate_scalar_is_safe():
    engine = CollaborationEngine(policy=policy())
    llm = FakeLLM(responses=[json.dumps({"delegate": 123}), "plain"])
    agent = make_agent("researcher", llm, engine, allow_delegation=True, allowed_agents=["analyst"])
    engine.register({"researcher": agent})
    ctx = engine.context_for("malformed")
    result = agent.run("x", run_id="malformed", run_context=ctx)
    # not parsed as delegation -> treated as plain text, no handoff, no crash
    assert result.handoffs == []
    assert ctx.events(EventType.DELEGATION_REQUESTED) == []


# ---------- AUDIT 11: context sharing leaks neither credentials nor sibling data ----------

def test_scoped_context_has_no_credentials_and_no_sibling_data(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-audit-secret-123")
    engine = CollaborationEngine(policy=policy())
    agents = {
        "researcher": make_agent("researcher", FakeLLM(responses=["r"]), engine,
                                 allow_delegation=True, allowed_agents=["analyst"]),
        "analyst": make_agent("analyst", FakeLLM(responses=["a"]), engine),
        "reviewer": make_agent(
            "reviewer", FakeLLM(responses=["top secret sibling result"]), engine
        ),
    }
    engine.register(agents)
    ctx = engine.context_for("leak")
    ctx.create_task(
        task_id="t0", parent_task_id=None, agent="supervisor", description="root", depth=0
    )
    ctx.create_task(
        task_id="t1", parent_task_id="t0", agent="researcher", description="child A", depth=1
    )
    ctx.create_task(task_id="t2", parent_task_id="t0", agent="reviewer",
                    description="sibling B", depth=1)
    ctx.update_task("t2", status=TaskStatus.COMPLETED, result="top secret sibling result")
    ctx.update_task("t1", status=TaskStatus.RUNNING)
    result = engine.delegate(
        {"target": "analyst", "task": "x", "context_hints": ["hint about task"]},
        run_id="leak", parent_task_id="t1", source_agent="researcher",
    )
    # find the request's relevant_context via the created task's child... use engine internals:
    # rebuild what _build_request produced: scoped context of t1
    scoped = ctx.scoped_context("t1")
    all_text = " ".join(m.content for m in scoped)
    assert "sk-audit-secret-123" not in all_text
    assert "api_key" not in all_text.lower()
    assert "top secret sibling result" not in all_text  # sibling branch not shared
    assert "child A" in all_text  # own ancestor chain IS shared
    assert result.status == DelegationStatus.COMPLETED


# ---------- AUDIT 12: partial failures — statuses, event ordering, usage ----------

def test_partial_failure_batch_statuses_and_events():
    engine = CollaborationEngine(policy=policy())
    tracker = UsageTracker()

    class RecordingLLM:
        def __init__(self, agent, answer=None, error=None):
            self.agent, self.answer, self.error = agent, answer, error

        def _run(self, run_id):
            from geosgan_hive.usage import UsageRecord
            if self.error:
                raise self.error
            tracker.record(UsageRecord(
                agent=self.agent, model="fake/m", provider="fake", prompt_tokens=1,
                completion_tokens=1, total_tokens=2, cost_usd=0.001, latency_ms=1.0,
                attempts=1, timestamp=datetime.now(UTC).isoformat(), run_id=run_id,
            ))
            return self.answer

        def complete(self, messages, *, temperature=0.2, run_id=None):
            return self._run(run_id)

    agents = {
        "researcher": make_agent("researcher", FakeLLM(responses=["r"]), engine,
                                 allow_delegation=True, allowed_agents=["analyst", "reviewer"]),
        "analyst": make_agent(
            "analyst", RecordingLLM("analyst", error=GatewayError("boom")), engine
        ),
        "reviewer": make_agent("reviewer", RecordingLLM("reviewer", answer="fine"), engine),
    }
    engine.register(agents)

    async def main():
        return await engine.adelegate_many(
            [{"target": "analyst", "task": "will fail"},
             {"target": "reviewer", "task": "will pass"}],
            run_id="partial", source_agent="researcher",
        )

    results = asyncio.run(main())
    assert [r.status for r in results] == [DelegationStatus.FAILED, DelegationStatus.COMPLETED]
    assert "boom" in results[0].error
    assert results[1].result == "fine"
    ctx = engine._contexts.get("partial")
    failed = ctx.events(EventType.TASK_FAILED)
    completed = ctx.events(EventType.TASK_COMPLETED)
    returned = ctx.events(EventType.RESULT_RETURNED)
    assert len(failed) == 1 and len(completed) == 1 and len(returned) == 2
    statuses = {t.agent: t.status for t in ctx.tasks()}
    assert statuses["analyst"] == TaskStatus.FAILED and statuses["reviewer"] == TaskStatus.COMPLETED
    # usage: only the successful call recorded; failed subtask recorded nothing
    assert tracker.summary(agent="reviewer").calls == 1
    assert tracker.summary(agent="analyst").calls == 0
    assert tracker.cost_for_run("partial") == 0.001
    engine.close_run("partial")


# ---------- AUDIT 6: legacy sync adapter timeout limitation ----------

def test_legacy_thread_adapter_timeout_cancels_wrapper_but_thread_finishes():
    class SlowSyncLLM:
        def __init__(self):
            self.calls = 0

        def complete(self, messages, *, temperature=0.2, run_id=None):
            time.sleep(0.3)  # cannot be interrupted once in a thread
            self.calls += 1
            return "slow sync answer"

    slow = SlowSyncLLM()
    engine = CollaborationEngine(policy=policy(default_delegation_timeout_seconds=0.05))
    agents = {
        "researcher": make_agent("researcher", FakeLLM(responses=["r"]), engine,
                                 allow_delegation=True, allowed_agents=["analyst"]),
        "analyst": make_agent("analyst", slow, engine),
    }
    engine.register(agents)
    ctx = engine.context_for("legacy-timeout")

    async def main():
        started = time.perf_counter()
        result = await engine.adelegate({"target": "analyst", "task": "x"}, run_id="legacy-timeout",
                                        source_agent="researcher")
        elapsed = time.perf_counter() - started
        assert result.status == DelegationStatus.CANCELLED
        assert elapsed < 0.25  # wrapper cancelled promptly...
        await asyncio.sleep(0.4)  # ...but the thread runs to completion
        assert slow.calls == 1  # the physical call still happened and is counted
        return result

    result = asyncio.run(main())
    assert "timed out" in result.error
    node = [t for t in ctx.tasks() if t.agent == "analyst"][0]
    assert node.status == TaskStatus.CANCELLED
    assert ctx.model_calls == 1  # reserved and counted even though the wrapper was cancelled
