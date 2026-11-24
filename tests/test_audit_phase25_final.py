"""Phase 2.5 FINAL adversarial audit — failing-first regression tests.

Each test targets a specific audit requirement and was written to FAIL against
commit 2b21f31 before the corresponding fix landed on the same branch:

- A1: root task identifier preserved across suspend/resume (checkpoint field)
- A2: provider Retry-After hint extends the PERSISTED next_attempt_at
- A3: resumable checkpoint with missing transcript is rejected (no silent
      restart of an incomplete task)
- A4: resume validates stored config integrity
- A5: a second CheckpointStore instance must not kill a live-lease RUNNING run;
      only expired/lease-less RUNNING rows are recovered as FAILED
- A6: CONTINUITY never suspends/persists as WAITING (it fails the run)
- A7: checkpoint DB file permissions are owner-only where supported
- A8: progress counters are persisted per step and fenced by ownership
- A9: two recovery workers cannot duplicate-resume the same run
- A10: cancellation mid-run leaves a consistent, recoverable store row
"""

import asyncio
import json
import os
import stat
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
from geosgan_hive.orchestrator import Supervisor
from geosgan_hive.recovery import RecoveryWorker
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


def make_supervisor(tmp_path, monkeypatch, *, mode=FallbackMode.WAIT,
                    persist=PersistMode.RESUMABLE, lease_ttl=60.0, name="cp.db"):
    from geosgan_hive.agent_config import AgentSpec, HiveConfig, SupervisorSpec

    gateway, tracker = make_gateway(monkeypatch)
    engine = CollaborationEngine(policy=CollaborationPolicy(allow_agent_to_agent=True),
                                 tracker=tracker)
    gateway.set_context_lookup(engine.context_for)
    store = CheckpointStore(str(tmp_path / name), persist_mode=persist,
                            lease_ttl_seconds=lease_ttl)
    fb = FallbackPolicy(mode=mode, cooldown_seconds=60.0, max_wait_seconds=600.0)
    sup_llm = gateway.bind(PRIMARY, agent="supervisor", fallback=fb)
    agent = Agent("a", "d", "s", gateway.bind(BACKUP, agent="a", fallback=FallbackPolicy()),
                  collaboration=engine)
    engine.register({"a": agent})
    config_json = HiveConfig(
        supervisor=SupervisorSpec(name="supervisor", description="d", system_prompt="s",
                                  model=PRIMARY),
        agents=[AgentSpec(name="a", description="d", system_prompt="s", model=BACKUP)],
    ).model_dump_json()
    sup = Supervisor({"a": agent}, sup_llm, collaboration=engine, checkpoint_store=store,
                     checkpoint_config_json=config_json, suspend_cooldown_seconds=60.0,
                     max_wait_seconds=600.0)
    return sup, store, engine, gateway


# ---- A1: root task id preserved across resume ----

def test_a1_root_task_id_preserved_across_suspend_and_resume(tmp_path, monkeypatch):
    sup, store, engine, gateway = make_supervisor(tmp_path, monkeypatch)

    def fail(**kwargs):
        raise litellm_exc.RateLimitError("429", llm_provider="ollama", model="primary")

    monkeypatch.setattr(litellm, "completion", fail)
    r1 = sup.run("task", run_id="a1")
    assert r1.suspended is True
    cp = store.get("a1")
    assert cp.root_task_id  # persisted
    assert cp.root_task_id == r1.root_task_id  # same id in result and checkpoint

    def finish(**kwargs):
        return ok_response(kwargs["model"], json.dumps({"next": "FINISH", "answer": "ok"}))

    monkeypatch.setattr(litellm, "completion", finish)
    r2 = sup.resume("a1", owner="w")
    assert r2.root_task_id == cp.root_task_id  # resume keeps the ORIGINAL root task id
    assert r2.run_id == "a1"
    store.close()


# ---- A2: Retry-After extends the persisted next_attempt_at ----

def test_a2_retry_after_hint_extends_persisted_next_attempt(tmp_path, monkeypatch):
    sup, store, engine, gateway = make_supervisor(tmp_path, monkeypatch)

    def limited(**kwargs):
        exc = litellm_exc.RateLimitError("429", llm_provider="ollama", model="primary")
        exc.response = SimpleNamespace(headers={"retry-after": "120"})
        raise exc

    from datetime import datetime

    monkeypatch.setattr(litellm, "completion", limited)
    before = store.utcnow()
    sup.run("task", run_id="a2")
    cp = store.get("a2")

    def parse(s):
        return datetime.fromisoformat(s)

    delta = (parse(cp.next_attempt_at) - before).total_seconds()
    # cooldown is 60s but the provider asked for 120s -> persisted hint wins
    assert 115 <= delta <= 125, f"next_attempt delta was {delta}s (cooldown 60 ignored Retry-After)"
    # and it survives a "restart": a fresh store instance honors it
    store.close()
    store2 = CheckpointStore(str(tmp_path / "cp.db"), persist_mode=PersistMode.RESUMABLE)
    eligible = store2.list_eligible()
    assert eligible == []  # next attempt is 120s in the future, not due yet
    store2.close()


# ---- A3: resumable checkpoint with missing transcript is rejected ----

def test_a3_resumable_checkpoint_without_transcript_is_rejected(tmp_path, monkeypatch):
    sup, store, engine, gateway = make_supervisor(tmp_path, monkeypatch)

    def fail(**kwargs):
        raise litellm_exc.RateLimitError("429", llm_provider="ollama", model="primary")

    monkeypatch.setattr(litellm, "completion", fail)
    sup.run("task", run_id="a3")
    # simulate an incomplete checkpoint: resumable mode but no transcript stored
    with store._lock, store._conn:
        store._conn.execute("UPDATE runs SET transcript_json=NULL WHERE run_id='a3'")
    with pytest.raises(CheckpointError, match="INCOMPLETE"):
        sup.resume("a3", owner="w")
    store.close()


# ---- A4: resume validates stored config integrity ----

def test_a4_resume_rejects_corrupt_config_snapshot(tmp_path, monkeypatch):
    sup, store, engine, gateway = make_supervisor(tmp_path, monkeypatch)

    def fail(**kwargs):
        raise litellm_exc.RateLimitError("429", llm_provider="ollama", model="primary")

    monkeypatch.setattr(litellm, "completion", fail)
    sup.run("task", run_id="a4")
    with store._lock, store._conn:
        store._conn.execute("UPDATE runs SET config_json='{not json' WHERE run_id='a4'")
    with pytest.raises(CheckpointError, match="config"):
        sup.resume("a4", owner="w")
    store.close()


# ---- A5: second store instance must not kill a live RUNNING run ----

def test_a5_live_lease_protects_running_run_across_instances(tmp_path):
    db = str(tmp_path / "cp.db")
    s1 = CheckpointStore(db, persist_mode=PersistMode.RESUMABLE, lease_ttl_seconds=60.0)
    s1.create_run(run_id="live", owner="proc-A", task="t", agent="supervisor", model="m",
                  root_task_id="root-1")
    # a second connection/process opens the same DB while the lease is live
    s2 = CheckpointStore(db, persist_mode=PersistMode.RESUMABLE, lease_ttl_seconds=60.0)
    assert s2.get("live").state == ExecutionState.RUNNING  # NOT failed
    s2.close()
    # short-ttl store: after expiry, reopening recovers the abandoned run
    s3 = CheckpointStore(db, persist_mode=PersistMode.RESUMABLE, lease_ttl_seconds=1.0)
    s3.create_run(run_id="dead", owner="proc-A", task="t", agent="s", model="m",
                  root_task_id="root-2")
    s3.close()
    time.sleep(1.2)
    s4 = CheckpointStore(db, persist_mode=PersistMode.RESUMABLE, lease_ttl_seconds=1.0)
    assert s4.get("dead").state == ExecutionState.FAILED  # abandoned -> recovered
    assert s4.get("live").state == ExecutionState.RUNNING  # live lease still protects it
    s1.close()
    s4.close()


# ---- A6: CONTINUITY never suspends; it fails the run in the store ----

def test_a6_continuity_exhaustion_marks_failed_not_waiting(tmp_path, monkeypatch):
    sup, store, engine, gateway = make_supervisor(tmp_path, monkeypatch,
                                                  mode=FallbackMode.CONTINUITY)

    def fail(**kwargs):
        raise litellm_exc.RateLimitError("429", llm_provider="ollama", model="primary")

    monkeypatch.setattr(litellm, "completion", fail)
    r = sup.run("task", run_id="a6")
    assert r.suspended is False
    assert isinstance(r.final_answer, str) and "error" in r.final_answer.lower()
    cp = store.get("a6")
    assert cp.state == ExecutionState.FAILED  # not WAITING: continuity never suspends
    assert cp.next_attempt_at is None
    store.close()


# ---- A7: checkpoint DB file is owner-only where the OS supports it ----

@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions only")
def test_a7_checkpoint_db_permissions_are_owner_only(tmp_path):
    db = str(tmp_path / "cp.db")
    store = CheckpointStore(db, persist_mode=PersistMode.RESUMABLE)
    store.create_run(run_id="p", owner="w", task="t", agent="s", model="m", root_task_id="r")
    store.close()
    mode = stat.S_IMODE(os.stat(db).st_mode)
    assert mode & 0o077 == 0, f"checkpoint db must not be group/world accessible, got {oct(mode)}"


# ---- A8: progress counters persisted per step, fenced by ownership ----

def test_a8_progress_update_persists_counters_and_is_fenced(tmp_path):
    store = CheckpointStore(str(tmp_path / "cp.db"), persist_mode=PersistMode.RESUMABLE)
    store.create_run(run_id="a8", owner="w1", task="t", agent="s", model="m",
                     root_task_id="r")
    store.update_progress("a8", owner="w1", counters_json='{"model_calls": 3}',
                          step_count=2)
    assert json.loads(store.get("a8").counters_json) == {"model_calls": 3}
    assert store.get("a8").step_count == 2
    with pytest.raises(CheckpointError, match="lease"):
        store.update_progress("a8", owner="w2", counters_json='{"model_calls": 99}',
                              step_count=9)  # stale/wrong owner fenced out
    assert json.loads(store.get("a8").counters_json) == {"model_calls": 3}
    store.close()


# ---- A9: two workers, one run — exactly one resumption ----

def test_a9_two_workers_cannot_duplicate_resume(tmp_path):
    store = CheckpointStore(str(tmp_path / "cp.db"), persist_mode=PersistMode.RESUMABLE,
                            lease_ttl_seconds=60.0)
    store.create_run(run_id="a9", owner="w", task="t", agent="s", model="m",
                     root_task_id="r")
    now = store.utcnow()
    store.mark_waiting("a9", owner="w", error="e", counters_json="{}", steps_json="[]",
                       route_json="[]", transcript_json=None, step_count=1,
                       next_attempt_at=now.isoformat(timespec="milliseconds"),
                       deadline_at=(now + timedelta(hours=1)).isoformat(timespec="milliseconds"))
    calls = []

    async def resume_fn(run_id, owner):
        # mimic supervisor.resume: claim lease, then fenced terminal commit
        if not store.claim_lease(run_id, owner):
            raise CheckpointError("leased")
        await asyncio.sleep(0.02)  # widen the race window
        store.mark_completed(run_id, owner=owner, result="done", counters_json="{}",
                             steps_json="[]", route_json="[]", transcript_json=None,
                             step_count=2)
        calls.append(owner)

    w1 = RecoveryWorker(store, resume_fn, owner="worker-1")
    w2 = RecoveryWorker(store, resume_fn, owner="worker-2")

    async def go():
        await asyncio.gather(w1.run_once(), w2.run_once())

    asyncio.run(go())
    assert len(calls) == 1  # exactly one worker resumed
    assert store.get("a9").state == ExecutionState.COMPLETED
    assert w1.resumed + w2.resumed == 1
    assert w1.skipped + w2.skipped <= 1  # loser either skipped or never saw it eligible
    store.close()


# ---- A10: cancellation mid-run leaves a consistent recoverable row ----

def test_a10_cancelled_async_run_leaves_recoverable_row(tmp_path, monkeypatch):
    sup, store, engine, gateway = make_supervisor(tmp_path, monkeypatch, lease_ttl=0.5)
    entered = asyncio.Event()

    async def slow_acompletion(**kwargs):
        entered.set()
        await asyncio.sleep(30)
        return ok_response(kwargs["model"])

    monkeypatch.setattr(litellm, "acompletion", slow_acompletion)

    async def go():
        task = asyncio.create_task(sup.arun("task", run_id="a10"))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())
    cp = store.get("a10")
    assert cp is not None and cp.state == ExecutionState.RUNNING  # consistent, not orphaned
    assert cp.lease_owner is not None  # lease visible; expires -> recoverable as FAILED
    store.close()
    # after the persisted lease expires, reopening recovers the abandoned run
    time.sleep(0.6)
    store3 = CheckpointStore(str(tmp_path / "cp.db"), persist_mode=PersistMode.RESUMABLE,
                             lease_ttl_seconds=0.5)
    assert store3.get("a10").state == ExecutionState.FAILED
    assert "crash" in (store3.get("a10").error or "")
    store3.close()
