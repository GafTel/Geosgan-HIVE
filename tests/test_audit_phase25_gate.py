"""Phase 2.5 APPROVAL GATE — failing-first regression tests (run against db4313a
before the fixes landed):

- G1/G2: periodic lease renewal keeps a long model call owned; keeper stops on
  completion/cancellation and leaves no orphan tasks
- G3: stale workers cannot renew/commit/dispatch (ownership fencing)
- G4: every physical dispatch reservation is durably committed to SQLite
  BEFORE the provider request is sent
- G5: subprocess crash between reservation and dispatch retains the reservation
- G6: multiple dispatches within one step / nested delegation all persist;
  counters never decrease
- G7: resume reconciles persisted physical reservations (no double count)
- G8: credential/provider scope of health keys (quota is provider-account scope,
  auth opens nothing provider-wide, other providers unaffected)
- G9: resumed runs are governed by the CURRENT authorization (a modified current
  config cannot be overridden by the checkpoint snapshot)
"""

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from datetime import timedelta
from types import SimpleNamespace

import litellm
import litellm.exceptions as litellm_exc
import pytest

from geosgan_hive.agents import Agent
from geosgan_hive.checkpoints import CheckpointStore, ExecutionState, PersistMode
from geosgan_hive.collaboration import CollaborationEngine
from geosgan_hive.errors import CheckpointError
from geosgan_hive.fallback import FallbackMode, FallbackPolicy
from geosgan_hive.gateway import ModelGateway
from geosgan_hive.health import CircuitState, HealthRegistry
from geosgan_hive.orchestrator import Supervisor
from geosgan_hive.retry import RetryPolicy
from geosgan_hive.run_context import CollaborationPolicy
from geosgan_hive.usage import UsageTracker

PRIMARY = "ollama/primary"
BACKUP = "ollama/backup"


def ok_response(model, content="ok"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        model=model,
    )


def make_gateway(monkeypatch, **kwargs):
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.001)
    tracker = UsageTracker()
    gateway = ModelGateway(
        tracker=tracker,
        retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0, max_delay_seconds=0,
                                 jitter_ratio=0),
        sleep=lambda s: None,
        async_sleep=lambda s: asyncio.sleep(0),
        **kwargs,
    )
    return gateway, tracker


def valid_config_json():
    from geosgan_hive.agent_config import AgentSpec, HiveConfig, SupervisorSpec
    return HiveConfig(
        supervisor=SupervisorSpec(name="supervisor", description="d", system_prompt="s",
                                  model=PRIMARY),
        agents=[AgentSpec(name="a", description="d", system_prompt="s", model=BACKUP)],
    ).model_dump_json()


def make_supervisor(tmp_path, monkeypatch, *, mode=FallbackMode.WAIT, lease_ttl=60.0,
                    renew_interval=None, name="cp.db"):
    gateway, tracker = make_gateway(monkeypatch)
    engine = CollaborationEngine(policy=CollaborationPolicy(allow_agent_to_agent=True),
                                 tracker=tracker)
    gateway.set_context_lookup(engine.context_for)
    store = CheckpointStore(str(tmp_path / name), persist_mode=PersistMode.RESUMABLE,
                            lease_ttl_seconds=lease_ttl)
    gateway.set_durable_store(store)  # G4: durable reservations before dispatch
    fb = FallbackPolicy(mode=mode, cooldown_seconds=60.0, max_wait_seconds=600.0)
    sup_llm = gateway.bind(PRIMARY, agent="supervisor", fallback=fb)
    agent = Agent("a", "d", "s", gateway.bind(BACKUP, agent="a", fallback=FallbackPolicy()),
                  collaboration=engine)
    engine.register({"a": agent})
    kwargs = {}
    if renew_interval is not None:
        kwargs["lease_renew_interval_seconds"] = renew_interval
    sup = Supervisor({"a": agent}, sup_llm, collaboration=engine, checkpoint_store=store,
                     checkpoint_config_json=valid_config_json(),
                     suspend_cooldown_seconds=60.0, max_wait_seconds=600.0, **kwargs)
    return sup, store, engine, gateway


# ---- G1/G2: periodic lease renewal during a long model call ----

def test_g1_long_model_call_keeps_lease_via_periodic_renewal(tmp_path, monkeypatch):
    # lease ttl 0.3s, model call takes 0.45s -> without periodic renewal the lease
    # expires mid-call and the fenced terminal commit would fail
    sup, store, engine, gateway = make_supervisor(
        tmp_path, monkeypatch, lease_ttl=0.3, renew_interval=0.05
    )

    def slow_finish(**kwargs):
        time.sleep(0.45)
        return ok_response(kwargs["model"], json.dumps({"next": "FINISH", "answer": "ok"}))

    monkeypatch.setattr(litellm, "completion", slow_finish)
    result = sup.run("task", run_id="g1")
    assert result.final_answer == "ok"
    cp = store.get("g1")
    assert cp.state == ExecutionState.COMPLETED  # commit succeeded: lease was renewed
    assert cp.lease_owner is None  # released on terminal state
    store.close()


def test_g2_lease_keeper_stops_on_completion_and_cancellation(tmp_path, monkeypatch):
    sup, store, engine, gateway = make_supervisor(
        tmp_path, monkeypatch, lease_ttl=0.3, renew_interval=0.05, name="g2.db"
    )
    monkeypatch.setattr(litellm, "completion", lambda **kw: ok_response(
        kw["model"], json.dumps({"next": "FINISH", "answer": "ok"})))
    sup.run("task", run_id="g2-sync")
    keepers = [t for t in threading.enumerate() if t.name.startswith("lease-keeper")]
    assert keepers == []  # no orphaned renewal threads after completion

    # async: cancellation stops the keeper task too
    entered = asyncio.Event()

    async def slow(**kwargs):
        entered.set()
        await asyncio.sleep(30)
        return ok_response(kwargs["model"])

    monkeypatch.setattr(litellm, "acompletion", slow)

    async def go():
        task = asyncio.create_task(sup.arun("task", run_id="g2-async"))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)  # let the keeper task finish cancelling
        leftover = [t for t in asyncio.all_tasks() if "lease-keeper" in (t.get_name())]
        return leftover

    leftover = asyncio.run(go())
    assert leftover == []  # no orphaned renewal tasks after cancellation
    store.close()


# ---- G3: ownership fencing ----

def test_g3_stale_worker_cannot_renew_commit_or_dispatch(tmp_path):
    store = CheckpointStore(str(tmp_path / "g3.db"), persist_mode=PersistMode.RESUMABLE,
                            lease_ttl_seconds=60.0)
    store.create_run(run_id="g3", owner="owner-A", task="t", agent="s", model="m",
                     root_task_id="r")
    assert store.renew_lease("g3", "owner-B") is False  # not the owner
    assert store.renew_lease("g3", "owner-A") is True
    with pytest.raises(CheckpointError, match="lease lost"):
        store.mark_completed("g3", owner="owner-B", result="x", counters_json="{}",
                             steps_json="[]", route_json="[]", transcript_json=None,
                             step_count=1)
    # durable physical reservation only while RUNNING: after suspension the
    # crashed/leased-out worker can no longer dispatch
    now = store.utcnow()
    store.mark_waiting("g3", owner="owner-A", error="e", counters_json="{}", steps_json="[]",
                       route_json="[]", transcript_json=None, step_count=1,
                       next_attempt_at=now.isoformat(timespec="milliseconds"),
                       deadline_at=(now + timedelta(hours=1)).isoformat(timespec="milliseconds"))
    assert store.reserve_physical_durable("g3") is False  # not RUNNING -> no dispatch
    store.close()


# ---- G4: durable reservation committed BEFORE the provider request ----

def test_g4_physical_reservation_persisted_before_dispatch(tmp_path, monkeypatch):
    sup, store, engine, gateway = make_supervisor(tmp_path, monkeypatch, name="g4.db")
    seen_at_dispatch = {}

    def fake_completion(**kwargs):
        # inside the provider call: the reservation must ALREADY be in SQLite
        seen_at_dispatch["physical_reserved"] = store.get("g4").physical_reserved
        return ok_response(kwargs["model"], json.dumps({"next": "FINISH", "answer": "ok"}))

    monkeypatch.setattr(litellm, "completion", fake_completion)
    sup.run("task", run_id="g4")
    assert seen_at_dispatch["physical_reserved"] == 1  # durable BEFORE the request
    assert store.get("g4").physical_reserved == 1
    store.close()


# ---- G5: subprocess crash between reservation and dispatch ----

CRASH_SCRIPT = """
import os, sys
import litellm
from types import SimpleNamespace
sys.path.insert(0, "src")
from geosgan_hive.checkpoints import CheckpointStore, PersistMode
from geosgan_hive.collaboration import CollaborationEngine
from geosgan_hive.gateway import ModelGateway
from geosgan_hive.retry import RetryPolicy
from geosgan_hive.run_context import CollaborationPolicy
from geosgan_hive.usage import UsageTracker

db, mode = sys.argv[1], sys.argv[2]
store = CheckpointStore(db, persist_mode=PersistMode.RESUMABLE, lease_ttl_seconds=60.0)
store.create_run(run_id="crash", owner="w", task="t", agent="supervisor", model="ollama/p",
                 root_task_id="r")
tracker = UsageTracker()
gw = ModelGateway(tracker=tracker,
                  retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0,
                                           max_delay_seconds=0, jitter_ratio=0),
                  sleep=lambda s: None)
gw.set_durable_store(store)
engine = CollaborationEngine(policy=CollaborationPolicy(), tracker=tracker)
gw.set_context_lookup(engine.context_for)
ctx = engine.context_for("crash")
ctx.lease_owner = "w"  # worker-owned identity in the execution context
ctx.lease_generation = 0

def crash(**kwargs):
    os._exit(9)  # hard crash: after the durable reservation, at the provider call

litellm.completion = crash
try:
    gw.complete([__import__("geosgan_hive.messages", fromlist=["Message"]).Message("user", "x")],
                model="ollama/p", run_id="crash")
except BaseException:
    pass
"""


def test_g5_crash_after_reservation_before_dispatch_retains_it(tmp_path):
    db = str(tmp_path / "g5.db")
    store = CheckpointStore(db, persist_mode=PersistMode.RESUMABLE, lease_ttl_seconds=60.0)
    store.create_run(run_id="probe", owner="w", task="t", agent="s", model="m",
                     root_task_id="r")
    store.close()
    repo = os.getcwd()
    proc = subprocess.run(
        [sys.executable, "-c", CRASH_SCRIPT, db, "crash"],
        cwd=repo, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 9  # the subprocess really crashed
    store2 = CheckpointStore(db, persist_mode=PersistMode.RESUMABLE, lease_ttl_seconds=60.0)
    cp = store2.get("crash")
    assert cp.physical_reserved == 1  # consumed reservation conservatively retained
    assert cp.state == ExecutionState.RUNNING  # crash during RUNNING, lease still live
    store2.close()


# ---- G6: multiple dispatches in one step + nested delegation all persist ----

def test_g6_all_dispatches_persist_and_counters_never_decrease(tmp_path, monkeypatch):
    sup, store, engine, gateway = make_supervisor(tmp_path, monkeypatch, name="g6.db")
    dispatched = []

    def fake(**kwargs):
        dispatched.append(kwargs["model"])
        n = len(dispatched)
        if kwargs["model"] == PRIMARY and n == 1:
            return ok_response(PRIMARY, json.dumps({"next": "a", "reason": "go"}))
        if kwargs["model"] == BACKUP and len([d for d in dispatched if d == BACKUP]) == 1:
            raise litellm_exc.RateLimitError("429", llm_provider="ollama", model="backup")
        if kwargs["model"] == PRIMARY:
            return ok_response(PRIMARY, json.dumps({"next": "FINISH", "answer": "done"}))
        return ok_response(BACKUP, "agent answer")

    monkeypatch.setattr(litellm, "completion", fake)
    gateway.retry_policy = RetryPolicy(max_attempts=2, base_delay_seconds=0,
                                       max_delay_seconds=0, jitter_ratio=0)
    result = sup.run("task", run_id="g6")
    assert result.final_answer == "done"
    # sup + agent(fail) + agent(retry ok) + sup final = 4 physical dispatches
    assert len(dispatched) == 4
    cp = store.get("g6")
    assert cp.physical_reserved == 4  # every dispatch durably persisted
    ctx_physical = json.loads(cp.counters_json)["physical_reserved"]
    assert ctx_physical == 4  # in-memory and durable agree; neither decreased
    store.close()


# ---- G7: resume reconciles durable reservations without double counting ----

def test_g7_resume_reconciles_persisted_physical_reservations(tmp_path, monkeypatch):
    sup, store, engine, gateway = make_supervisor(tmp_path, monkeypatch, name="g7.db")

    def fail(**kwargs):
        raise litellm_exc.RateLimitError("429", llm_provider="ollama", model="primary")

    monkeypatch.setattr(litellm, "completion", fail)
    sup.run("task", run_id="g7")  # suspends after 1 durable physical reservation
    cp = store.get("g7")
    assert cp.physical_reserved == 1
    # a suspended (non-RUNNING) run can no longer reserve: dispatch is blocked
    assert store.reserve_physical_durable("g7") is False

    # resume with the provider "recovered" (same store = same process view)
    monkeypatch.setattr(litellm, "completion", lambda **kw: ok_response(
        kw["model"], json.dumps({"next": "FINISH", "answer": "resumed"})))
    r = sup.resume("g7", owner="w2")
    assert r.final_answer == "resumed"
    cp = store.get("g7")
    assert cp.state == ExecutionState.COMPLETED
    # 1 (before suspend) + 1 (resumed call) = 2, never reset, never double counted
    assert cp.physical_reserved == 2
    assert json.loads(cp.counters_json)["physical_reserved"] == 2
    store.close()


# ---- G8: credential/provider scope of health keys ----

def test_g8_health_scopes_do_not_leak_across_providers_or_credentials(monkeypatch):
    health = HealthRegistry(failure_threshold=1, cooldown_seconds=60.0)
    # quota exhaustion on the openai ACCOUNT opens only the openai provider scope
    for _ in range(1):
        health.record_failure(HealthRegistry.provider_key("openai"))
    assert health.state(HealthRegistry.provider_key("openai")) == CircuitState.OPEN
    assert health.state(HealthRegistry.provider_key("ollama")) == CircuitState.CLOSED
    # an auth failure (credential-specific) opens NO provider-wide circuit
    health.record_failure(HealthRegistry.model_key("openai/gpt-4o"), open_circuit=False)
    health.record_failure(HealthRegistry.provider_key("mistral"), open_circuit=False)
    assert health.state(HealthRegistry.provider_key("mistral")) == CircuitState.CLOSED
    # NOTE: the architecture configures exactly ONE credential per provider
    # (first matching env var), so provider scope == that credential's scope.
    # Multi-credential isolation per provider is NOT claimed (documented).
    # Health keys/logs must never contain raw credential material:
    assert all("sk-" not in k and "key" not in k.lower() for k in health.snapshot())


# ---- G9: resumed runs follow CURRENT authorization, not the snapshot ----

def test_g9_resume_uses_current_policy_not_checkpoint_snapshot(tmp_path, monkeypatch):
    sup, store, engine, gateway = make_supervisor(
        tmp_path, monkeypatch, mode=FallbackMode.WAIT, name="g9.db"
    )

    def fail(**kwargs):
        raise litellm_exc.RateLimitError("429", llm_provider="ollama", model="primary")

    monkeypatch.setattr(litellm, "completion", fail)
    r1 = sup.run("task", run_id="g9")
    assert r1.suspended is True
    # the CURRENT config is changed to fallback OFF (snapshot still says WAIT):
    sup.llm = gateway.bind(PRIMARY, agent="supervisor", fallback=FallbackPolicy())
    dispatched = []

    def still_failing(**kwargs):
        dispatched.append(kwargs["model"])
        raise litellm_exc.RateLimitError("429", llm_provider="ollama", model="primary")

    monkeypatch.setattr(litellm, "completion", still_failing)
    r2 = sup.resume("g9", owner="w")
    # current policy (OFF) governs: NO fallback model was ever dispatched
    assert set(dispatched) == {PRIMARY}
    assert r2.suspended is False and r2.failed is True
    assert store.get("g9").state == ExecutionState.FAILED  # not re-suspended
    store.close()
