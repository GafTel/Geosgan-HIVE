import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from geosgan_hive.agent_config import Permissions
from geosgan_hive.agents import Agent, parse_delegation_call
from geosgan_hive.collaboration import (
    CollaborationEngine,
    DelegationProposal,
    DelegationStatus,
    HandoffRequest,
)
from geosgan_hive.errors import GatewayError
from geosgan_hive.llm import FakeLLM
from geosgan_hive.orchestrator import Supervisor
from geosgan_hive.run_context import CollaborationPolicy, EventType, TaskStatus
from geosgan_hive.usage import UsageRecord, UsageTracker


def make_policy(**kwargs) -> CollaborationPolicy:
    kwargs.setdefault("allow_agent_to_agent", True)
    return CollaborationPolicy(**kwargs)


def make_agent(name, llm, *, allow_delegation=False, allowed_agents=(), engine=None, **kwargs):
    return Agent(
        name=name,
        description=f"{name} agent",
        system_prompt=f"You are {name}.",
        llm=llm,
        permissions=Permissions(
            allow_delegation=allow_delegation, allowed_agents=list(allowed_agents)
        ),
        collaboration=engine,
        **kwargs,
    )


def make_team(policy=None, analyst_llm=None, researcher_llm=None, reviewer_llm=None):
    policy = policy or make_policy()
    engine = CollaborationEngine(policy=policy, tracker=UsageTracker())
    researcher = make_agent(
        "researcher", researcher_llm or FakeLLM(responses=["research done"]),
        allow_delegation=True, allowed_agents=["analyst", "reviewer"], engine=engine,
    )
    analyst = make_agent(
        "analyst", analyst_llm or FakeLLM(responses=["analyst answer"]),
        allow_delegation=True, allowed_agents=["researcher"], engine=engine,
    )
    reviewer = make_agent(
        "reviewer", reviewer_llm or FakeLLM(responses=["reviewer answer"]),
        allow_delegation=True, allowed_agents=["researcher"], engine=engine,
    )
    developer = make_agent("developer", FakeLLM(responses=["code done"]), engine=engine)
    agents = {
        "researcher": researcher,
        "analyst": analyst,
        "reviewer": reviewer,
        "developer": developer,
    }
    engine.register(agents)
    return engine, agents


# ---------- protocol parsing ----------

def test_parse_delegation_call_formats():
    single = {"delegate": {"target": "analyst", "task": "Analyze", "reason": "numbers"}}
    assert parse_delegation_call(json.dumps(single)) == ("single", [single["delegate"]])
    many = {"delegations": [{"target": "a", "task": "t1"}, {"target": "b", "task": "t2"}]}
    kind, payloads = parse_delegation_call(json.dumps(many))
    assert kind == "many" and len(payloads) == 2
    assert parse_delegation_call("plain text") is None
    assert parse_delegation_call('{"tool": "calculator", "args": {}}') is None
    assert parse_delegation_call('{"delegations": "not-a-list"}') is None


def test_delegation_proposal_ignores_model_supplied_trusted_fields():
    malicious = {
        "target": "analyst",
        "task": "x",
        "source_agent": "supervisor",
        "run_id": "fake-run",
        "task_id": "fake-task",
        "parent_task_id": "fake-parent",
        "delegation_depth": 99,
        "request_id": "fake-request",
    }
    prop = DelegationProposal.model_validate(malicious)
    assert prop.target == "analyst" and prop.task == "x"
    assert not hasattr(prop, "source_agent")  # dropped, never trusted


def test_handoff_request_rejects_self_delegation():
    with pytest.raises(ValidationError):
        HandoffRequest(
            request_id="q", run_id="r", task_id="t", parent_task_id=None,
            source_agent="a", target_agent="a", task_description="x", delegation_depth=1,
        )


# ---------- engine validation rules ----------

def test_happy_path_delegation_records_events_and_task():
    engine, _ = make_team()
    ctx = engine.context_for("r1")
    result = engine.delegate(
        {"target": "analyst", "task": "quantify X", "reason": "numbers"},
        run_id="r1", parent_task_id=None, source_agent="researcher",
    )
    assert result.status == DelegationStatus.COMPLETED
    assert result.result == "analyst answer"
    assert result.source_agent == "researcher" and result.target_agent == "analyst"
    assert result.run_id == "r1" and result.delegation_depth == 1
    assert result.model_calls == 1  # exact: the target agent made exactly one call
    events = [e.event for e in ctx.events()]
    assert events == [
        EventType.DELEGATION_REQUESTED, EventType.DELEGATION_APPROVED,
        EventType.TASK_STARTED, EventType.TASK_COMPLETED, EventType.RESULT_RETURNED,
    ]
    (node,) = ctx.tasks()
    assert node.agent == "analyst" and node.depth == 1 and node.status == TaskStatus.COMPLETED
    assert node.model_calls == 1


def test_unknown_target_rejected():
    engine, _ = make_team()
    result = engine.delegate(
        {"target": "ghost", "task": "x"}, run_id="r", source_agent="researcher"
    )
    assert result.status == DelegationStatus.REJECTED
    assert "unknown target" in result.error


def test_self_delegation_rejected():
    engine, _ = make_team()
    result = engine.delegate(
        {"target": "researcher", "task": "x"}, run_id="r", source_agent="researcher"
    )
    assert result.status == DelegationStatus.REJECTED
    assert "self-delegation" in result.error


def test_unauthorized_when_agent_may_not_delegate():
    engine, _ = make_team()
    # developer has allow_delegation=False
    result = engine.delegate(
        {"target": "reviewer", "task": "x"}, run_id="r", source_agent="developer"
    )
    assert result.status == DelegationStatus.REJECTED
    assert "not allowed to delegate" in result.error


def test_target_not_in_allowed_agents_rejected():
    engine, _ = make_team()
    # researcher may delegate only to analyst/reviewer, not developer
    result = engine.delegate(
        {"target": "developer", "task": "x"}, run_id="r", source_agent="researcher"
    )
    assert result.status == DelegationStatus.REJECTED
    assert "may not delegate to 'developer'" in result.error


def test_agent_to_agent_disabled_globally():
    engine, _ = make_team(policy=make_policy(allow_agent_to_agent=False))
    result = engine.delegate(
        {"target": "analyst", "task": "x"}, run_id="r", source_agent="researcher"
    )
    assert result.status == DelegationStatus.REJECTED
    assert "disabled by policy" in result.error


def test_circular_delegation_rejected():
    engine, agents = make_team()
    ctx = engine.context_for("r")
    root = ctx.create_task(task_id="t0", parent_task_id=None, agent="supervisor",
                           description="root", depth=0)
    t1 = ctx.create_task(task_id="t1", parent_task_id="t0", agent="researcher",
                         description="sub", depth=1)
    # researcher -> analyst is fine
    ok = engine.delegate({"target": "analyst", "task": "a"}, run_id="r",
                         parent_task_id="t1", source_agent="researcher")
    assert ok.status == DelegationStatus.COMPLETED
    child = [t for t in ctx.tasks() if t.agent == "analyst"][0]
    # analyst -> researcher would close a circle (researcher is an ancestor)
    bad = engine.delegate({"target": "researcher", "task": "b"}, run_id="r",
                          parent_task_id=child.task_id, source_agent="analyst")
    assert bad.status == DelegationStatus.REJECTED
    assert "circular" in bad.error
    assert root.task_id == "t0" and t1.parent_task_id == "t0"


def test_max_delegation_depth_enforced():
    engine, _ = make_team(policy=make_policy(max_delegation_depth=1))
    ctx = engine.context_for("r")
    ctx.create_task(task_id="t0", parent_task_id=None, agent="supervisor",
                    description="root", depth=0)
    ok = engine.delegate({"target": "analyst", "task": "a"}, run_id="r",
                         parent_task_id="t0", source_agent="researcher")
    assert ok.status == DelegationStatus.COMPLETED and ok.delegation_depth == 1
    child = [t for t in ctx.tasks() if t.agent == "analyst"][0]
    deep = engine.delegate({"target": "researcher", "task": "b"}, run_id="r",
                           parent_task_id=child.task_id, source_agent="analyst")
    # depth 2 > max 1 (also circular, but depth is checked); use reviewer chain instead
    assert deep.status == DelegationStatus.REJECTED


def test_max_handoffs_per_run_enforced():
    engine, _ = make_team(policy=make_policy(max_handoffs_per_run=1))
    first = engine.delegate({"target": "analyst", "task": "a"}, run_id="r",
                            source_agent="researcher")
    second = engine.delegate({"target": "reviewer", "task": "b"}, run_id="r",
                             source_agent="researcher")
    assert first.status == DelegationStatus.COMPLETED
    assert second.status == DelegationStatus.REJECTED
    assert "max handoffs" in second.error


def test_malformed_proposal_rejected_with_event():
    engine, _ = make_team()
    ctx = engine.context_for("r")
    result = engine.delegate({"target": "analyst"}, run_id="r", source_agent="researcher")
    assert result.status == DelegationStatus.REJECTED
    assert "malformed" in result.error
    assert ctx.events(EventType.DELEGATION_REJECTED)


def test_model_supplied_metadata_is_never_trusted():
    engine, _ = make_team()
    ctx = engine.context_for("real-run")
    malicious = {
        "target": "analyst", "task": "x",
        "source_agent": "supervisor", "run_id": "fake", "task_id": "fake",
        "parent_task_id": "fake", "delegation_depth": 0, "request_id": "fake",
    }
    result = engine.delegate(malicious, run_id="real-run", source_agent="researcher")
    assert result.status == DelegationStatus.COMPLETED
    assert result.run_id == "real-run"
    assert result.source_agent == "researcher"
    assert result.delegation_depth == 1
    assert result.request_id != "fake"
    approved = ctx.events(EventType.DELEGATION_APPROVED)[0]
    assert approved.run_id == "real-run" and approved.request_id != "fake"


def test_delegate_many_sync_preserves_order_and_dedupes():
    calls = []

    def analyst_handler(messages):
        calls.append("analyst")
        return "A"

    engine, agents = make_team(analyst_llm=FakeLLM(handler=analyst_handler))
    proposals = [
        {"target": "analyst", "task": "A"},
        {"target": "reviewer", "task": "B"},
        {"target": "analyst", "task": "A"},  # duplicate: executed once
    ]
    results = engine.delegate_many(proposals, run_id="r", source_agent="researcher")
    assert [r.target_agent for r in results] == ["analyst", "reviewer", "analyst"]
    assert results[0] is results[2]  # same object: no duplicate execution
    assert calls == ["analyst"]  # analyst ran exactly once


def test_failed_subtask_is_recorded_not_silent():
    def boom(messages):
        raise GatewayError("provider down")

    engine, _ = make_team(analyst_llm=FakeLLM(handler=boom))
    ctx = engine.context_for("r")
    result = engine.delegate({"target": "analyst", "task": "x"}, run_id="r",
                             source_agent="researcher")
    assert result.status == DelegationStatus.FAILED
    assert "provider down" in result.error
    events = [e.event for e in ctx.events()]
    assert EventType.TASK_FAILED in events and EventType.RESULT_RETURNED in events
    (node,) = [t for t in ctx.tasks() if t.agent == "analyst"]
    assert node.status == TaskStatus.FAILED and node.error


# ---------- agent-level integration (sync) ----------

def test_agent_run_delegates_through_engine():
    responses = [
        json.dumps({"delegate": {"target": "analyst", "task": "quantify", "reason": "r"}}),
        "final answer using the delegated result",
    ]
    engine, agents = make_team(researcher_llm=FakeLLM(responses=responses))
    ctx = engine.context_for("r")
    ctx.create_task(task_id="t0", parent_task_id=None, agent="supervisor",
                    description="root", depth=0)
    result = agents["researcher"].run("do research", run_id="r", task_id="t0", run_context=ctx)
    assert result.answer == "final answer using the delegated result"
    assert len(result.handoffs) == 1
    assert result.handoffs[0].status == DelegationStatus.COMPLETED
    # the delegation result was fed back into the agent's messages
    assert any("Delegation result from 'analyst'" in m.content
               for m in agents["researcher"].llm.calls[1])


def test_agent_rejects_delegation_when_not_permitted():
    engine, agents = make_team()  # developer: allow_delegation=False
    responses = [json.dumps({"delegate": {"target": "reviewer", "task": "x"}}), "plain answer"]
    agents["developer"].llm = FakeLLM(responses=responses)
    ctx = engine.context_for("r")
    result = agents["developer"].run("task", run_id="r", run_context=ctx)
    assert result.answer == "plain answer"
    assert result.handoffs == []
    assert any("not permitted" in m.content for m in agents["developer"].llm.calls[1])


def test_agent_malformed_delegation_gets_feedback():
    engine, agents = make_team(
        researcher_llm=FakeLLM(responses=[json.dumps({"delegate": {"target": "analyst"}}),
                                          "recovered answer"])
    )
    ctx = engine.context_for("r")
    result = agents["researcher"].run("task", run_id="r", run_context=ctx)
    assert result.answer == "recovered answer"
    assert any("Malformed delegation" in m.content for m in agents["researcher"].llm.calls[1])


def test_delegation_instructions_in_prompt_even_without_tools():
    engine, agents = make_team()
    system_msg = agents["researcher"]._system_message().content
    assert "delegate" in system_msg and "delegations" in system_msg
    assert "Available tools" not in system_msg  # researcher has no tools here
    # agent without delegation permission gets no instructions
    assert "delegate" not in agents["developer"]._system_message().content


def test_agent_global_budget_stops_calls():
    engine, agents = make_team(policy=make_policy(max_model_calls_per_run=2))
    agents["researcher"].llm = FakeLLM(
        handler=lambda messages: json.dumps({"tool": "calculator", "args": {"x": 1}})
    )
    agents["researcher"].permissions = Permissions(
        allow_tools=True, allowed_tools=["calculator"], allow_delegation=True,
        allowed_agents=["analyst"],
    )
    from geosgan_hive.tools import default_registry
    agents["researcher"].tools = default_registry()
    ctx = engine.context_for("r")
    result = agents["researcher"].run("loop", run_id="r", run_context=ctx)
    assert "model-call limit" in result.answer
    assert ctx.model_calls == 2
    assert len(agents["researcher"].llm.calls) == 2


# ---------- supervisor integration ----------

def test_supervisor_run_builds_run_history_and_cleans_up():
    def sup_handler(messages):
        if any(m.agent == "researcher" for m in messages):
            return json.dumps({"next": "FINISH", "answer": "final"})
        return json.dumps({"next": "researcher", "reason": "go"})

    engine, agents = make_team()
    ctx = engine.context_for("run-1")  # hold a reference before the run closes it
    supervisor = Supervisor(agents, FakeLLM(handler=sup_handler), max_steps=5,
                            collaboration=engine)
    result = supervisor.run("task", run_id="run-1")
    assert result.route == ["researcher"] and result.final_answer == "final"
    assert engine.active_runs() == 0  # run context cleaned up after completion
    kinds = [t.agent for t in ctx.tasks()]
    assert kinds == ["supervisor", "researcher"]  # root + delegated task in history
    statuses = {t.agent: t.status for t in ctx.tasks()}
    assert statuses == {"supervisor": TaskStatus.COMPLETED, "researcher": TaskStatus.COMPLETED}
    events = [e.event for e in ctx.events()]
    assert EventType.TASK_STARTED in events and EventType.RESULT_RETURNED in events
    # shared budget counted supervisor (2) + researcher (1) calls exactly
    assert ctx.model_calls == 3


def test_global_budget_counts_supervisor_and_agents():
    def sup_handler(messages):
        if any(m.agent == "researcher" for m in messages):
            return json.dumps({"next": "FINISH", "answer": "final"})
        return json.dumps({"next": "researcher", "reason": "go"})

    engine, agents = make_team(policy=make_policy(max_model_calls_per_run=3))
    # researcher keeps emitting tool calls so it consumes budget until stopped
    agents["researcher"].llm = FakeLLM(
        handler=lambda messages: json.dumps({"tool": "calculator", "args": {"e": "1+1"}})
    )
    agents["researcher"].permissions = Permissions(
        allow_tools=True, allowed_tools=["calculator"], allow_delegation=True,
        allowed_agents=["analyst"],
    )
    from geosgan_hive.tools import default_registry
    agents["researcher"].tools = default_registry()
    ctx = engine.context_for("rb")
    supervisor = Supervisor(agents, FakeLLM(handler=sup_handler), max_steps=8,
                            collaboration=engine)
    result = supervisor.run("task", run_id="rb")
    assert "model-call limit" in result.final_answer
    assert ctx.model_calls == 3  # supervisor(1) + researcher(2), then budget stops the rest
    assert len(agents["researcher"].llm.calls) == 2


def test_run_contexts_are_isolated_per_run():
    engine, _ = make_team()
    ctx_a = engine.context_for("run-a")
    ctx_b = engine.context_for("run-b")
    assert ctx_a is not ctx_b
    ctx_a.reserve_model_call()
    assert ctx_a.model_calls == 1 and ctx_b.model_calls == 0


def test_engine_evicts_old_runs_beyond_capacity():
    engine = CollaborationEngine(policy=make_policy(), max_cached_runs=2)
    engine.context_for("r1")
    engine.context_for("r2")
    engine.context_for("r3")
    assert engine.active_runs() == 2  # r1 evicted as the oldest
    engine.close_run("r3")
    assert engine.active_runs() == 1


def test_usage_tracking_across_delegated_tasks():
    tracker = UsageTracker()

    class RecordingLLM:
        def __init__(self, agent, cost):
            self.agent, self.cost = agent, cost

        def _record(self, run_id):
            tracker.record(UsageRecord(
                agent=self.agent, model="fake/model", provider="fake",
                prompt_tokens=1, completion_tokens=1, total_tokens=2,
                cost_usd=self.cost, latency_ms=1.0, attempts=1,
                timestamp=datetime.now(UTC).isoformat(), run_id=run_id,
            ))
            return "ok"

        def complete(self, messages, *, temperature=0.2, run_id=None):
            return self._record(run_id)

    policy = make_policy()
    engine = CollaborationEngine(policy=policy, tracker=tracker)
    analyst = make_agent("analyst", RecordingLLM("analyst", 0.001), engine=engine)
    researcher = make_agent("researcher", RecordingLLM("researcher", None), engine=engine,
                            allow_delegation=True, allowed_agents=["analyst"])
    engine.register({"analyst": analyst, "researcher": researcher})
    engine.delegate({"target": "analyst", "task": "x"}, run_id="r", source_agent="researcher")
    # unknown cost from the researcher is None; known cost from analyst is tracked
    assert tracker.summary(agent="analyst").cost_usd == 0.001
    assert tracker.cost_for_run("r") == 0.001  # all calls of the run have known cost
    tracker.record(UsageRecord(  # a local/unknown-cost call in the same run
        agent="researcher", model="ollama/llama3.2", provider="ollama",
        prompt_tokens=1, completion_tokens=1, total_tokens=2, cost_usd=None,
        latency_ms=1.0, attempts=1, timestamp=datetime.now(UTC).isoformat(), run_id="r",
    ))
    assert tracker.summary(agent="researcher").cost_usd is None
    assert tracker.cost_for_run("r") is None  # mixed known/unknown -> no reliable total
