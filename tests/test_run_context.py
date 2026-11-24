import threading

import pytest
from pydantic import ValidationError

from geosgan_hive.run_context import (
    CollaborationPolicy,
    EventType,
    RunContext,
    TaskStatus,
)


def test_policy_has_the_approved_conservative_defaults():
    policy = CollaborationPolicy()
    assert policy.max_delegation_depth == 2
    assert policy.max_handoffs_per_run == 10
    assert policy.max_parallel_agents == 3
    assert policy.default_delegation_timeout_seconds == 120.0
    assert policy.max_model_calls_per_run == 200
    assert policy.allow_agent_to_agent is False


def test_policy_rejects_invalid_values():
    with pytest.raises(ValidationError):
        CollaborationPolicy(max_delegation_depth=0)
    with pytest.raises(ValidationError):
        CollaborationPolicy(max_parallel_agents=0)
    with pytest.raises(ValidationError):
        CollaborationPolicy(max_handoffs_per_run=-1)
    with pytest.raises(ValidationError):
        CollaborationPolicy(default_delegation_timeout_seconds=0)
    with pytest.raises(ValidationError):
        CollaborationPolicy(max_model_calls_per_run=0)


def test_model_call_reservation_is_atomic_under_threads():
    ctx = RunContext("r1", CollaborationPolicy(max_model_calls_per_run=50))
    outcomes: list[bool] = []
    lock = threading.Lock()

    def worker():
        ok = ctx.reserve_model_call()
        with lock:
            outcomes.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(200)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert ctx.model_calls == 50  # exactly the budget, no double counting
    assert sum(outcomes) == 50
    assert ctx.reserve_model_call() is False


def test_handoff_budget_is_bounded():
    ctx = RunContext("r", CollaborationPolicy(max_handoffs_per_run=2))
    assert ctx.reserve_handoff() is True
    assert ctx.reserve_handoff() is True
    assert ctx.reserve_handoff() is False
    assert ctx.handoffs == 2


def test_task_tree_ancestors_and_scoped_context():
    ctx = RunContext("r", CollaborationPolicy())
    root = ctx.create_task(
        task_id="t0", parent_task_id=None, agent="supervisor", description="root", depth=0
    )
    child = ctx.create_task(
        task_id="t1", parent_task_id="t0", agent="researcher", description="sub", depth=1
    )
    sibling = ctx.create_task(
        task_id="t2", parent_task_id="t0", agent="analyst", description="other", depth=1
    )
    ctx.update_task("t1", status=TaskStatus.COMPLETED, result="R" * 1000)

    chain = ctx.ancestors("t1")
    assert [n.task_id for n in chain] == ["t0", "t1"]
    assert ctx.ancestors(None) == []

    scoped = ctx.scoped_context("t1")
    assert len(scoped) == 2
    assert scoped[0].agent == "system"
    assert "t0" in scoped[0].content and "root" in scoped[0].content
    # results are truncated, never the full text
    assert "Result: " + "R" * 500 in scoped[1].content
    assert "R" * 1000 not in scoped[1].content
    # the sibling branch is NOT part of the scoped context
    assert all("t2" not in m.content and "other" not in m.content for m in scoped)
    assert root.status == TaskStatus.PENDING
    assert sibling.agent == "analyst"
    assert child.depth == 1


def test_events_are_recorded_with_identifiers():
    ctx = RunContext("r", CollaborationPolicy())
    ctx.record_event(
        EventType.DELEGATION_REQUESTED, request_id="q1", source_agent="a", target_agent="b"
    )
    ctx.record_event(EventType.TASK_STARTED, task_id="t1")
    events = ctx.events()
    assert [e.event for e in events] == [EventType.DELEGATION_REQUESTED, EventType.TASK_STARTED]
    assert events[0].request_id == "q1" and events[0].run_id == "r"
    assert ctx.events(EventType.TASK_STARTED)[0].task_id == "t1"
    assert ctx.events(EventType.TASK_COMPLETED) == []


def test_result_store_supports_idempotency():
    ctx = RunContext("r", CollaborationPolicy())
    assert ctx.get_result("q1") is None
    ctx.set_result("q1", {"status": "completed"})
    assert ctx.get_result("q1") == {"status": "completed"}


def test_close_marks_context_closed():
    ctx = RunContext("r", CollaborationPolicy())
    assert ctx.closed is False
    ctx.close()
    assert ctx.closed is True
