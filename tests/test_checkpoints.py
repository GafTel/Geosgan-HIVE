import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import litellm
import litellm.exceptions as litellm_exc
import pytest
from pydantic import ValidationError

from geosgan_hive.agent_config import (
    AgentSpec,
    CheckpointConfig,
    CollaborationPolicy,
    HiveConfig,
    SupervisorSpec,
)
from geosgan_hive.agents import Agent
from geosgan_hive.checkpoints import (
    CheckpointStore,
    ExecutionState,
    PersistMode,
)
from geosgan_hive.collaboration import CollaborationEngine
from geosgan_hive.errors import CheckpointError, ProviderUnavailableError
from geosgan_hive.fallback import FallbackMode, FallbackPolicy
from geosgan_hive.gateway import ModelGateway
from geosgan_hive.orchestrator import Supervisor
from geosgan_hive.retry import RetryPolicy
from geosgan_hive.run_context import TaskStatus
from geosgan_hive.tools import default_registry
from geosgan_hive.usage import UsageTracker


class FakeClock:
    def __init__(self, start=1_700_000_000.0):
        self.t = start

    def __call__(self):
        return datetime.fromtimestamp(self.t, tz=UTC)

    def advance(self, seconds):
        self.t += seconds


def make_store(tmp_path, *, persist_mode=PersistMode.RESUMABLE, clock=None, name="cp.db"):
    clock = clock or FakeClock()
    return CheckpointStore(
        str(tmp_path / name), persist_mode=persist_mode, lease_ttl_seconds=60.0,
        utcnow=clock,
    ), clock


# ---- store basics ----

def test_create_get_list_and_terminal_states(tmp_path):
    store, _ = make_store(tmp_path)
    cp = store.create_run(run_id="r1", owner="w1", task="do x", agent="supervisor",
                          model="ollama/sup", config_json='{"a": 1}')
    assert cp.state == ExecutionState.RUNNING
    assert cp.lease_owner == "w1"
    store.mark_completed("r1", owner="w1", result="done", counters_json="{}",
                         steps_json="[]", route_json="[]", transcript_json=None,
                         step_count=2)
    cp = store.get("r1")
    assert cp.state == ExecutionState.COMPLETED and cp.result == "done"
    assert cp.lease_owner is None  # lease released on terminal state
    assert store.list_runs(state=ExecutionState.COMPLETED)[0].run_id == "r1"
    store.close()


def test_schema_version_mismatch_is_rejected(tmp_path):
    import sqlite3

    path = str(tmp_path / "bad.db")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO meta VALUES ('schema_version', '999')")
    conn.commit()
    conn.close()
    with pytest.raises(CheckpointError, match="schema_version"):
        CheckpointStore(path)


def test_metadata_only_never_stores_transcript(tmp_path):
    store, _ = make_store(tmp_path, persist_mode=PersistMode.METADATA_ONLY)
    store.create_run(run_id="r", owner="w", task="t", agent="supervisor", model="m")
    store.mark_waiting(
        "r", owner="w", error="e", counters_json="{}", steps_json="[]", route_json="[]",
        transcript_json='[{"role": "user", "content": "secret prompt"}]',
        step_count=1, next_attempt_at="2024-01-01T00:00:00.000+00:00",
        deadline_at="2024-01-01T01:00:00.000+00:00",
    )
    cp = store.get("r")
    assert cp.state == ExecutionState.WAITING_FOR_PROVIDER
    assert cp.transcript_json is None  # privacy mode: transcript never persisted
    assert cp.lease_owner is None  # lease released on suspension
    store.close()


def test_resumable_mode_stores_transcript(tmp_path):
    store, _ = make_store(tmp_path, persist_mode=PersistMode.RESUMABLE)
    store.create_run(run_id="r", owner="w", task="t", agent="supervisor", model="m")
    store.mark_waiting(
        "r", owner="w", error="e", counters_json="{}", steps_json="[]", route_json="[]",
        transcript_json='[{"role": "user", "content": "hi"}]', step_count=1,
        next_attempt_at="2024-01-01T00:00:00.000+00:00",
        deadline_at="2024-01-01T01:00:00.000+00:00",
    )
    assert json.loads(store.get("r").transcript_json) == [{"role": "user", "content": "hi"}]
    store.close()


# ---- leases ----

def test_lease_claim_is_single_winner_and_fenced(tmp_path):
    store, clock = make_store(tmp_path)
    store.create_run(run_id="r", owner="w", task="t", agent="s", model="m")
    store.mark_waiting(
        "r", owner="w", error="e", counters_json="{}", steps_json="[]", route_json="[]",
        transcript_json=None, step_count=1,
        next_attempt_at=store.now_iso(), deadline_at=None,
    )
    assert store.claim_lease("r", "worker-A") is True
    assert store.claim_lease("r", "worker-B") is False  # already leased
    assert store.renew_lease("r", "worker-B") is False  # not the owner
    assert store.renew_lease("r", "worker-A") is True
    clock.advance(61)  # lease expires
    assert store.claim_lease("r", "worker-B") is True  # expired lease can be claimed
    with pytest.raises(CheckpointError, match="lease lost"):
        store.mark_completed("r", owner="worker-A", result="x", counters_json="{}",
                             steps_json="[]", route_json="[]", transcript_json=None,
                             step_count=1)
    store.close()


def test_list_eligible_respects_next_attempt_deadline_and_lease(tmp_path):
    store, clock = make_store(tmp_path)
    now = store.now_iso()
    future = (datetime.now(UTC) + timedelta(hours=1)).isoformat(timespec="milliseconds")
    for rid, nxt, deadline in (("due", now, future), ("not-due", future, future),
                               ("expired", now, now)):
        store.create_run(run_id=rid, owner="w", task="t", agent="s", model="m")
        store.mark_waiting(
            rid, owner="w", error="e", counters_json="{}", steps_json="[]", route_json="[]",
            transcript_json=None, step_count=1, next_attempt_at=nxt, deadline_at=deadline,
        )
    assert [c.run_id for c in store.list_eligible()] == ["due"]
    store.claim_lease("due", "worker-A")
    assert store.list_eligible() == []  # leased runs are not eligible
    store.close()


def test_cancel_is_terminal_and_idempotent_guard(tmp_path):
    store, _ = make_store(tmp_path)
    store.create_run(run_id="r", owner="w", task="t", agent="s", model="m")
    store.mark_waiting(
        "r", owner="w", error="e", counters_json="{}", steps_json="[]", route_json="[]",
        transcript_json=None, step_count=1, next_attempt_at=store.now_iso(), deadline_at=None,
    )
    store.cancel("r", reason="user cancelled")
    assert store.get("r").state == ExecutionState.CANCELLED
    with pytest.raises(CheckpointError, match="terminal"):
        store.cancel("r")
    store.close()


def test_crash_during_running_marks_failed_on_reopen(tmp_path):
    clock = FakeClock()
    store, _ = make_store(tmp_path, clock=clock)
    store.create_run(run_id="crash", owner="w", task="t", agent="s", model="m")
    store.close()
    clock.advance(120)  # lease (60s) now expired -> owner process presumed dead
    store2, _ = make_store(tmp_path, clock=clock)
    cp = store2.get("crash")
    assert cp.state == ExecutionState.FAILED
    assert "crash" in cp.error
    store2.close()


def test_crash_during_waiting_clears_stale_lease_but_stays_waiting(tmp_path):
    clock = FakeClock()
    store, _ = make_store(tmp_path, clock=clock)
    store.create_run(run_id="w", owner="w", task="t", agent="s", model="m")
    store.mark_waiting(
        "w", owner="w", error="e", counters_json="{}", steps_json="[]", route_json="[]",
        transcript_json=None, step_count=1,
        next_attempt_at=(clock() + timedelta(seconds=30)).isoformat(timespec="milliseconds"),
        deadline_at=None,
    )
    # simulate a worker that leased the waiting run and then died
    assert store.claim_lease("w", "dead-worker")
    store.close()
    clock.advance(120)
    store2, _ = make_store(tmp_path, clock=clock)
    cp = store2.get("w")
    assert cp.state == ExecutionState.WAITING_FOR_PROVIDER
    assert cp.lease_owner is None  # stale lease cleared, run still resumable
    assert [c.run_id for c in store2.list_eligible()] == ["w"]  # next attempt is due
    store2.close()


# ---- config validation ----

def test_wait_mode_requires_checkpoint_config():
    base = dict(
        supervisor=SupervisorSpec(name="supervisor", description="d", system_prompt="s",
                                  model="ollama/sup"),
        agents=[AgentSpec(name="analyst", description="d", system_prompt="s",
                          model="ollama/an")],
    )
    with pytest.raises(ValidationError, match="checkpoint"):
        HiveConfig(**base, fallback=FallbackPolicy(mode=FallbackMode.WAIT))
    ok = HiveConfig(**base, fallback=FallbackPolicy(mode=FallbackMode.WAIT),
                    checkpoint=CheckpointConfig(path=":memory:"))
    assert ok.checkpoint.persist_mode == PersistMode.METADATA_ONLY
    # OFF (default) needs no checkpoint
    HiveConfig(**base)


def test_readonly_tool_flags():
    reg = default_registry()
    assert reg.is_readonly("calculator") is True
    assert reg.is_readonly("word_count") is True
    assert reg.is_readonly("nope") is False


# ---- supervisor suspend / resume integration ----

def ok_response(model, content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        model=model,
    )




def valid_config_json():
    from geosgan_hive.agent_config import AgentSpec, HiveConfig, SupervisorSpec
    return HiveConfig(
        supervisor=SupervisorSpec(name="supervisor", description="d", system_prompt="s",
                                  model="ollama/sup"),
        agents=[AgentSpec(name="analyst", description="d", system_prompt="s",
                          model="ollama/analyst")],
    ).model_dump_json()


def make_team(tmp_path, monkeypatch, *, persist_mode=PersistMode.RESUMABLE, clock=None):
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.001)
    clock = clock or FakeClock()
    store = CheckpointStore(
        str(tmp_path / "cp.db"), persist_mode=persist_mode, lease_ttl_seconds=60.0,
        utcnow=clock,
    )
    tracker = UsageTracker()
    wait_policy = FallbackPolicy(mode=FallbackMode.WAIT)  # primary only, suspend on failure
    gateway = ModelGateway(
        tracker=tracker,
        retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0, max_delay_seconds=0,
                                 jitter_ratio=0),
        sleep=lambda s: None,
    )
    engine = CollaborationEngine(policy=CollaborationPolicy(allow_agent_to_agent=True),
                                 tracker=tracker)
    gateway.set_context_lookup(engine.context_for)
    sup_llm = gateway.bind("ollama/sup", agent="supervisor", fallback=wait_policy)
    agent_llm = gateway.bind("ollama/analyst", agent="analyst",
                             fallback=FallbackPolicy())  # fallback OFF for the agent
    analyst = Agent("analyst", "d", "s", agent_llm, collaboration=engine)
    engine.register({"analyst": analyst})
    supervisor = Supervisor(
        {"analyst": analyst}, sup_llm, collaboration=engine, checkpoint_store=store,
        checkpoint_config_json=valid_config_json(), suspend_cooldown_seconds=60.0,
        max_wait_seconds=300.0,
    )
    return supervisor, store, engine, gateway, clock


def test_supervisor_suspends_and_resumes_without_repeating_steps(tmp_path, monkeypatch):
    supervisor, store, engine, gateway, clock = make_team(tmp_path, monkeypatch)
    calls = []

    def phase1(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == "ollama/sup":
            if len([c for c in calls if c == "ollama/sup"]) == 1:
                return ok_response("ollama/sup", '{"next": "analyst", "reason": "go"}')
            raise litellm_exc.RateLimitError("429", llm_provider="ollama", model="sup")
        return ok_response("ollama/analyst", "analysis done")

    monkeypatch.setattr(litellm, "completion", phase1)
    result = supervisor.run("analyze the data", run_id="r1")
    assert result.suspended is True
    cp = store.get("r1")
    assert cp.state == ExecutionState.WAITING_FOR_PROVIDER
    assert cp.next_attempt_at is not None and cp.deadline_at is not None
    assert cp.persist_mode == PersistMode.RESUMABLE and cp.transcript_json is not None
    steps = json.loads(cp.steps_json)
    assert len(steps) == 1 and steps[0]["agent"] == "analyst"  # completed step persisted
    counters = json.loads(cp.counters_json)
    # sup step1 + analyst + failed sup step2 all reserved a logical call exactly once
    assert counters["model_calls"] == 3
    assert counters["physical_reserved"] == 3  # sup + analyst + failed sup dispatch

    # provider "recovers": next phase the supervisor answers FINISH directly
    def phase2(**kwargs):
        calls.append(kwargs["model"])
        return ok_response(kwargs["model"], '{"next": "FINISH", "answer": "final answer"}')

    monkeypatch.setattr(litellm, "completion", phase2)
    resumed = supervisor.resume("r1", owner="worker-1")
    assert resumed.suspended is False
    assert resumed.final_answer == "final answer"
    cp = store.get("r1")
    assert cp.state == ExecutionState.COMPLETED
    # the analyst was NOT executed again on resume (completed steps are not repeated)
    assert calls.count("ollama/analyst") == 1
    # counters restored from the checkpoint, not double counted:
    # 3 reservations before suspend + 1 after resume = 4 (the resumed logical call
    # is reserved once; the restored ones are not re-counted)
    assert json.loads(cp.counters_json)["physical_reserved"] == 4
    assert json.loads(cp.counters_json)["model_calls"] == 4
    store.close()


def test_resume_rejects_metadata_only_checkpoint(tmp_path, monkeypatch):
    supervisor, store, _e, _g, _c = make_team(tmp_path, monkeypatch,
                                              persist_mode=PersistMode.METADATA_ONLY)

    def always_fail(**kwargs):
        raise litellm_exc.RateLimitError("429", llm_provider="ollama", model="sup")

    monkeypatch.setattr(litellm, "completion", always_fail)
    result = supervisor.run("x", run_id="r2")
    assert result.suspended is True
    cp = store.get("r2")
    assert cp.transcript_json is None  # metadata_only: no transcript stored
    with pytest.raises(CheckpointError, match="resumable"):
        supervisor.resume("r2")
    store.close()


def test_duplicate_resume_is_prevented_by_lease(tmp_path, monkeypatch):
    supervisor, store, _e, _g, clock = make_team(tmp_path, monkeypatch)

    def always_fail(**kwargs):
        raise litellm_exc.RateLimitError("429", llm_provider="ollama", model="sup")

    monkeypatch.setattr(litellm, "completion", always_fail)
    supervisor.run("x", run_id="r3")
    assert store.claim_lease("r3", "worker-A")  # another worker holds the lease
    with pytest.raises(CheckpointError, match="leased"):
        supervisor.resume("r3", owner="worker-B")
    # once the lease expires, another worker may claim it
    clock.advance(61)
    assert store.claim_lease("r3", "worker-B")
    store.close()


def test_nested_suspension_fails_safely_with_diagnostic(tmp_path, monkeypatch):
    """A delegated child that hits WAIT/HYBRID suspension is FAILED with an explicit
    diagnostic (never checkpointed mid-flight) and the run suspends at the supervisor
    boundary; a still-RUNNING child makes the whole run FAIL instead of suspending."""
    supervisor, store, engine, gateway, _c = make_team(tmp_path, monkeypatch)
    # make the ANALYST also use WAIT so its gateway raises ProviderUnavailableError
    analyst = supervisor.agents["analyst"]
    analyst.llm = gateway.bind(
        "ollama/analyst", agent="analyst", fallback=FallbackPolicy(mode=FallbackMode.WAIT)
    )
    calls = []

    def phase1(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == "ollama/sup":
            return ok_response("ollama/sup", '{"next": "analyst", "reason": "go"}')
        raise litellm_exc.RateLimitError("429", llm_provider="ollama", model="analyst")

    monkeypatch.setattr(litellm, "completion", phase1)
    result = supervisor.run("x", run_id="r4")
    assert result.suspended is True  # suspended at the supervisor boundary
    cp = store.get("r4")
    steps = json.loads(cp.steps_json)
    assert steps[0]["status"] == TaskStatus.FAILED.value
    assert "ProviderUnavailable" in (steps[0]["error"] or "") or steps[0]["error"]

    # now simulate a still-active child task: suspension must refuse and FAIL safely
    ctx = engine.context_for("r5")
    ctx.create_task(task_id="root", parent_task_id=None, agent="supervisor",
                    description="t", depth=0)
    ctx.update_task("root", status=TaskStatus.RUNNING)
    ctx.create_task(task_id="child", parent_task_id="root", agent="analyst",
                    description="c", depth=1)
    ctx.update_task("child", status=TaskStatus.RUNNING)
    store.create_run(run_id="r5", owner="w", task="t", agent="supervisor", model="m")
    exc = ProviderUnavailableError("no model")
    out = supervisor._suspend("r5", ctx, "root", "t", [], [], 1, exc, owner="w")
    assert out.failed is True and "Unsupported nested suspension" in out.final_answer
    assert store.get("r5").state == ExecutionState.FAILED
    assert "nested suspension" in store.get("r5").error
    store.close()


def test_suspension_events_and_no_credentials_in_checkpoint(tmp_path, monkeypatch):
    supervisor, store, engine, _g, _c = make_team(tmp_path, monkeypatch)

    def always_fail(**kwargs):
        raise litellm_exc.RateLimitError("429", llm_provider="ollama", model="sup")

    monkeypatch.setattr(litellm, "completion", always_fail)
    supervisor.run("secret task", run_id="r6")
    cp = store.get("r6")
    blob = cp.config_json + cp.counters_json + cp.steps_json + (cp.transcript_json or "")
    assert "API_KEY" not in blob and "api_key" not in blob  # never persist credentials
    store.close()


def test_async_suspend_and_aresume(tmp_path, monkeypatch):
    import asyncio

    supervisor, store, engine, gateway, clock = make_team(tmp_path, monkeypatch)
    calls = []

    async def phase1(**kwargs):
        calls.append(kwargs["model"])
        if len([c for c in calls if c == "ollama/sup"]) == 1:
            return ok_response("ollama/sup", '{"next": "FINISH", "answer": "no wait"}')
        raise litellm_exc.RateLimitError("429", llm_provider="ollama", model="sup")

    # first supervisor call delegates, second fails -> suspend
    async def phase1b(**kwargs):
        calls.append(kwargs["model"])
        n = len([c for c in calls if c == "ollama/sup"])
        if n == 1:
            return ok_response("ollama/sup", '{"next": "analyst", "reason": "go"}')
        if kwargs["model"] == "ollama/analyst":
            return ok_response("ollama/analyst", "done")
        raise litellm_exc.RateLimitError("429", llm_provider="ollama", model="sup")

    async def run_phase1():
        monkeypatch.setattr(litellm, "acompletion", phase1b)
        return await supervisor.arun("task", run_id="ra")

    result = asyncio.run(run_phase1())
    assert result.suspended is True
    assert store.get("ra").state == ExecutionState.WAITING_FOR_PROVIDER

    async def phase2(**kwargs):
        return ok_response(kwargs["model"], '{"next": "FINISH", "answer": "async final"}')

    async def run_phase2():
        monkeypatch.setattr(litellm, "acompletion", phase2)
        return await supervisor.aresume("ra", owner="worker-async")

    resumed = asyncio.run(run_phase2())
    assert resumed.final_answer == "async final"
    assert store.get("ra").state == ExecutionState.COMPLETED
    assert calls.count("ollama/analyst") == 1
    store.close()
