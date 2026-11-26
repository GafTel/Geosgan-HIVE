"""Phase 3 T4: durable approvals, execution ledger, atomic persistent tool budgets.

Covers: approval lifecycle (PENDING->APPROVED->CONSUMED etc.), replay + argument
substitution rejection, concurrent single-winner consumption, operator-only deciders,
ledger ownership fencing, duplicate execution ids, UNKNOWN outcome + operator
resolution, durable tool_reservations (monotonic, RUNNING-only, visible across
connections), fail-closed gateway paths, full sync/async approval flow through the
gateway, and subprocess crash tests before/after the side effect."""

import asyncio
import os
import subprocess
import sys
import threading
import time

import pytest
from pydantic import BaseModel, ConfigDict, Field

from geosgan_hive.approvals import ApprovalManager
from geosgan_hive.checkpoints import (
    ApprovalState,
    CheckpointStore,
    ExecutionState,
    PersistMode,
    ToolExecutionStatus,
)
from geosgan_hive.errors import ApprovalError
from geosgan_hive.run_context import CollaborationPolicy, EventType, RunContext
from geosgan_hive.tool_gateway import ToolExecutionContext, ToolGateway
from geosgan_hive.tool_policy import build_policy
from geosgan_hive.tool_registry import ToolCatalog
from geosgan_hive.tool_spec import SideEffect, ToolSpec, args_sha256


class WriteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=200)
    content: str = Field(max_length=10_000)


def write_spec(**overrides) -> ToolSpec:
    kwargs = dict(
        name="fake_write", description="test mutating tool", category="test",
        side_effect=SideEffect.LOCAL_MUTATION, idempotent=False,
        requires_approval=True, input_model=WriteArgs, timeout_seconds=5,
    )
    kwargs.update(overrides)
    return ToolSpec(**kwargs)


class MutatingAdapter:
    def __init__(self, spec, side_effect=None):
        self.spec = spec
        self.calls = 0
        self.side_effect = side_effect or (lambda: None)

    def execute(self, args, ctx):
        self.calls += 1
        self.side_effect()
        return f"wrote {args['path']}"

    def approval_context(self, args):
        return {
            "operation": "write_file",
            "target": args["path"],
            "preconditions": {"expected_sha256": None},
        }


def make_store(tmp_path, name="t.db", lease_ttl=60.0) -> CheckpointStore:
    store = CheckpointStore(str(tmp_path / name), persist_mode=PersistMode.RESUMABLE,
                            lease_ttl_seconds=lease_ttl)
    store.create_run(run_id="r1", owner="w1", task="t", agent="alice", model="m",
                     root_task_id="root")
    return store


def make_gateway(store, adapter=None, *, approval_manager="default", strict=True,
                 spec=None):
    spec = spec or write_spec()
    adapter = adapter or MutatingAdapter(spec)
    catalog = ToolCatalog()
    catalog.register(spec, adapter)
    policy = build_policy(global_enabled=[spec.name],
                          agent_permissions={"alice": (True, [spec.name])})
    mgr = None
    if approval_manager == "default":
        mgr = ApprovalManager(store, expiry_seconds=30.0, preview_bytes=500)
    elif approval_manager is not None:
        mgr = approval_manager
    gw = ToolGateway(catalog, policy, store=store, approval_manager=mgr, strict=strict)
    return gw, adapter


def ctx(agent="alice", run_id="r1"):
    return ToolExecutionContext(agent=agent, run_id=run_id, task_id="t1")


def make_rc(run_id="r1"):
    return RunContext(run_id, CollaborationPolicy())


# ---- approval lifecycle ----

def test_approval_lifecycle_create_preview_decide_consume(tmp_path):
    store = make_store(tmp_path)
    mgr = ApprovalManager(store, expiry_seconds=60.0, preview_bytes=200)
    spec = write_spec()
    req = mgr.create(ctx=ctx(), spec=spec, args={"path": "a.txt", "content": "hello"},
                     operation="write_file", target="a.txt",
                     preconditions={"expected_sha256": "abc"})
    row = store.get_approval(req.approval_id)
    assert row["status"] == ApprovalState.PENDING.value
    assert row["agent"] == "alice" and row["run_id"] == "r1" and row["task_id"] == "t1"
    assert row["tool"] == "fake_write" and row["operation"] == "write_file"
    assert row["target"] == "a.txt"
    assert row["args_sha256"] == args_sha256({"path": "a.txt", "content": "hello"})
    # bounded, meaningful preview (not just a hash)
    assert "fake_write" in req.preview and "write_file" in req.preview
    assert "a.txt" in req.preview and len(req.preview) <= 200 + 200
    assert mgr.decide(req.approval_id, approve=True, operator="operator-1") is True
    assert store.get_approval(req.approval_id)["status"] == ApprovalState.APPROVED.value
    assert store.get_approval(req.approval_id)["decided_by"] == "operator-1"
    assert mgr.consume(req.approval_id,
                       expected_args_sha256=args_sha256({"path": "a.txt",
                                                          "content": "hello"}))
    assert store.get_approval(req.approval_id)["status"] == ApprovalState.CONSUMED.value
    store.close()


def test_consume_is_single_use_replay_rejected(tmp_path):
    store = make_store(tmp_path)
    mgr = ApprovalManager(store)
    spec = write_spec()
    args = {"path": "a.txt", "content": "x"}
    req = mgr.create(ctx=ctx(), spec=spec, args=args, operation="write_file",
                     target="a.txt")
    mgr.decide(req.approval_id, approve=True, operator="op")
    h = args_sha256(args)
    assert mgr.consume(req.approval_id, expected_args_sha256=h) is True
    assert mgr.consume(req.approval_id, expected_args_sha256=h) is False  # replay
    store.close()


def test_argument_substitution_rejected_by_hash_binding(tmp_path):
    store = make_store(tmp_path)
    mgr = ApprovalManager(store)
    spec = write_spec()
    req = mgr.create(ctx=ctx(), spec=spec, args={"path": "a.txt", "content": "x"},
                     operation="write_file", target="a.txt")
    mgr.decide(req.approval_id, approve=True, operator="op")
    # attacker substitutes the arguments after approval: hash mismatch -> rejected
    assert mgr.consume(req.approval_id, expected_args_sha256=args_sha256(
        {"path": "../../etc/evil", "content": "x"})) is False
    assert store.get_approval(req.approval_id)["status"] == ApprovalState.APPROVED.value
    store.close()


def test_concurrent_consumption_has_exactly_one_winner(tmp_path):
    store = make_store(tmp_path)
    mgr = ApprovalManager(store)
    spec = write_spec()
    args = {"path": "a.txt", "content": "x"}
    req = mgr.create(ctx=ctx(), spec=spec, args=args, operation="write_file",
                     target="a.txt")
    mgr.decide(req.approval_id, approve=True, operator="op")
    h = args_sha256(args)
    results = []
    lock = threading.Lock()

    def worker():
        ok = store.consume_approval(req.approval_id, expected_args_sha256=h)
        with lock:
            results.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(results) == 1
    assert store.get_approval(req.approval_id)["status"] == ApprovalState.CONSUMED.value
    store.close()


def test_model_or_agent_cannot_be_a_decider(tmp_path):
    store = make_store(tmp_path)
    mgr = ApprovalManager(store)
    spec = write_spec()
    req = mgr.create(ctx=ctx(), spec=spec, args={"path": "a.txt", "content": "x"},
                     operation="write_file", target="a.txt")
    for bad in ("model", "agent", "llm", "assistant", "", "  ", "agent:alice",
                "AGENT:bob"):
        with pytest.raises(ApprovalError):
            mgr.decide(req.approval_id, approve=True, operator=bad)
    assert store.get_approval(req.approval_id)["status"] == ApprovalState.PENDING.value
    store.close()


def test_decide_only_once_and_expired_cannot_be_decided_or_consumed(tmp_path):
    store = make_store(tmp_path)
    mgr = ApprovalManager(store, expiry_seconds=0.3)
    spec = write_spec()
    args = {"path": "a.txt", "content": "x"}
    req = mgr.create(ctx=ctx(), spec=spec, args=args, operation="write_file",
                     target="a.txt")
    mgr.decide(req.approval_id, approve=True, operator="op")
    with pytest.raises(ApprovalError):
        mgr.decide(req.approval_id, approve=False, operator="op")  # already decided
    time.sleep(0.4)
    assert mgr.sweep_expired() >= 1
    row = store.get_approval(req.approval_id)
    assert row["status"] == ApprovalState.EXPIRED.value
    with pytest.raises(ApprovalError):
        mgr.decide(req.approval_id, approve=True, operator="op")  # expired: no decide
    assert mgr.consume(req.approval_id, expected_args_sha256=args_sha256(args)) is False
    store.close()


def test_cancel_state_via_operator(tmp_path):
    store = make_store(tmp_path)
    mgr = ApprovalManager(store)
    spec = write_spec()
    req = mgr.create(ctx=ctx(), spec=spec, args={"path": "a.txt", "content": "x"},
                     operation="write_file", target="a.txt")
    mgr.decide(req.approval_id, approve=True, operator="op", cancel=True)
    assert store.get_approval(req.approval_id)["status"] == ApprovalState.CANCELLED.value
    store.close()


# ---- ledger ----

def test_ledger_duplicate_execution_id_rejected(tmp_path):
    store = make_store(tmp_path)
    ok1 = store.ledger_start(execution_id="e1", run_id="r1", agent="alice",
                             tool="fake_write", args_sha256="h", status="started",
                             lease_owner="w1", lease_generation=0)
    ok2 = store.ledger_start(execution_id="e1", run_id="r1", agent="alice",
                             tool="fake_write", args_sha256="h", status="started",
                             lease_owner="w1", lease_generation=0)
    assert ok1 is True and ok2 is False  # never double-dispatch
    store.close()


def test_ledger_update_fenced_by_owner_and_generation(tmp_path):
    store = make_store(tmp_path)
    store.ledger_start(execution_id="e1", run_id="r1", agent="alice", tool="fake_write",
                       args_sha256="h", status="started", lease_owner="w1",
                       lease_generation=0)
    assert store.ledger_update("e1", status="completed", lease_owner="w2",
                               lease_generation=0) is False  # wrong owner
    assert store.ledger_update("e1", status="completed", lease_owner="w1",
                               lease_generation=1) is False  # wrong generation
    assert store.ledger_update("e1", status="completed", lease_owner="w1",
                               lease_generation=0) is True
    assert store.ledger_get("e1")["status"] == "completed"
    store.close()


def test_unknown_outcome_requires_operator_resolution(tmp_path):
    store = make_store(tmp_path)
    store.ledger_start(execution_id="e1", run_id="r1", agent="alice", tool="fake_write",
                       args_sha256="h", status="started", lease_owner="w1",
                       lease_generation=0)
    store.ledger_update("e1", status="unknown", error="timeout; outcome uncertain",
                        lease_owner="w1", lease_generation=0)
    row = store.ledger_get("e1")
    assert row["status"] == ToolExecutionStatus.UNKNOWN.value
    # not auto-replayed; only an operator resolves, recording decision + evidence
    assert store.ledger_resolve("e1", decision="completed", operator="op-9",
                                evidence="verified file content on disk") is True
    row = store.ledger_get("e1")
    assert row["status"] == ToolExecutionStatus.COMPLETED.value
    assert "operator-resolved by op-9" in row["error"]
    assert "verified file content on disk" in row["error"]
    assert store.ledger_resolve("e1", decision="failed", operator="op-9",
                                evidence="x") is False  # no longer UNKNOWN
    store.close()


def test_ledger_list_filters(tmp_path):
    store = make_store(tmp_path)
    store.ledger_start(execution_id="e1", run_id="r1", agent="a", tool="t",
                       args_sha256="h", status="completed")
    store.ledger_start(execution_id="e2", run_id="r1", agent="a", tool="t",
                       args_sha256="h", status="unknown")
    assert len(store.ledger_list(run_id="r1")) == 2
    assert len(store.ledger_list(status="unknown")) == 1
    assert store.ledger_list(run_id="other") == []
    store.close()


# ---- durable tool reservations ----

def test_reserve_tool_durable_monotonic_running_only_and_visible_across_connections(tmp_path):
    db = str(tmp_path / "d.db")
    store = CheckpointStore(db, persist_mode=PersistMode.RESUMABLE, lease_ttl_seconds=60.0)
    store.create_run(run_id="r1", owner="w1", task="t", agent="a", model="m",
                     root_task_id="root")
    assert store.reserve_tool_durable("r1") is True
    assert store.reserve_tool_durable("r1") is True
    assert store.reserve_tool_durable("missing") is False
    store.close()
    store2 = CheckpointStore(db, persist_mode=PersistMode.RESUMABLE, lease_ttl_seconds=60.0)
    assert store2.get("r1").tool_reservations == 2  # durable across connections
    store2.cancel("r1", reason="done")
    assert store2.reserve_tool_durable("r1") is False  # terminal state -> fail closed
    assert store2.get("r1").tool_reservations == 2  # never decreases
    store2.close()


def test_gateway_rolls_back_memory_reservation_when_durable_reservation_fails(tmp_path):
    store = make_store(tmp_path)
    store.cancel("r1", reason="run ended")  # not RUNNING anymore
    gw, adapter = make_gateway(store)
    rc = make_rc()
    r = gw.execute("fake_write", {"path": "a.txt", "content": "x"}, ctx=ctx(),
                   run_context=rc)
    assert r.ok is False and "no longer active" in r.error
    assert rc.tool_calls == 0  # in-memory reservation rolled back
    assert adapter.calls == 0
    store.close()


# ---- fail-closed gateway paths ----

def test_mutating_tool_fails_closed_without_durable_store(tmp_path):
    catalog = ToolCatalog()
    spec = write_spec(requires_approval=False)  # reach the ledger fail-closed path
    adapter = MutatingAdapter(spec)
    catalog.register(spec, adapter)
    policy = build_policy(global_enabled=["fake_write"],
                          agent_permissions={"alice": (True, ["fake_write"])})
    gw = ToolGateway(catalog, policy, store=None, approval_manager=None, strict=False)
    r = gw.execute("fake_write", {"path": "a.txt", "content": "x"}, ctx=ctx())
    assert r.ok is False and "no durable ledger" in r.error
    assert adapter.calls == 0


def test_approval_required_tool_fails_closed_without_manager(tmp_path):
    store = make_store(tmp_path)
    gw, adapter = make_gateway(store, approval_manager=None)
    r = gw.execute("fake_write", {"path": "a.txt", "content": "x"}, ctx=ctx(),
                   run_context=make_rc())
    assert r.ok is False and "no ApprovalManager" in r.error
    assert adapter.calls == 0
    store.close()


def test_strict_gateway_fails_closed_without_run_id(tmp_path):
    store = make_store(tmp_path)
    gw, _ = make_gateway(store)
    r = gw.execute("fake_write", {"path": "a.txt", "content": "x"},
                   ctx=ToolExecutionContext(agent="alice"),  # run_id missing
                   run_context=make_rc("other"))
    assert r.ok is False and "failing closed" in r.error
    store.close()


# ---- full gateway approval flow ----

def _run_in_thread(gw, args, rc):
    box = {}

    def go():
        box["result"] = gw.execute("fake_write", args, ctx=ctx(), run_context=rc)

    t = threading.Thread(target=go)
    t.start()
    return t, box


def test_gateway_full_approval_flow_sync(tmp_path):
    store = make_store(tmp_path)
    gw, adapter = make_gateway(store)
    rc = make_rc()
    args = {"path": "notes.txt", "content": "hello"}
    t, box = _run_in_thread(gw, args, rc)
    mgr = gw.approval_manager
    # wait for the PENDING approval (no semaphore is held during the wait)
    deadline = time.time() + 5
    while time.time() < deadline:
        pending = mgr.list_pending(run_id="r1")
        if pending:
            break
        time.sleep(0.02)
    pending = mgr.list_pending(run_id="r1")
    assert len(pending) == 1
    ap = pending[0]
    assert ap["status"] == ApprovalState.PENDING.value
    assert "write_file" in ap["preview"] and "notes.txt" in ap["preview"]
    assert rc.sync_tool_semaphore().acquire(blocking=False) is True  # slot free during wait
    rc.sync_tool_semaphore().release()
    mgr.decide(ap["approval_id"], approve=True, operator="operator-1")
    t.join(timeout=10)
    assert not t.is_alive()
    r = box["result"]
    assert r.ok is True and r.status == "completed"
    assert adapter.calls == 1
    assert store.get_approval(ap["approval_id"])["status"] == ApprovalState.CONSUMED.value
    row = store.ledger_get(r.execution_id)
    assert row["status"] == ToolExecutionStatus.COMPLETED.value
    assert row["approval_id"] == ap["approval_id"]
    assert row["lease_owner"] == "w1" and row["lease_generation"] == 0
    assert store.get("r1").tool_reservations == 1  # durable reservation committed
    events = [e.event for e in rc.events()]
    assert EventType.APPROVAL_REQUESTED in events and EventType.TOOL_EXECUTED in events
    store.close()


def test_gateway_denied_approval_blocks_execution(tmp_path):
    store = make_store(tmp_path)
    gw, adapter = make_gateway(store)
    rc = make_rc()
    t, box = _run_in_thread(gw, {"path": "a.txt", "content": "x"}, rc)
    mgr = gw.approval_manager
    deadline = time.time() + 5
    ap = None
    while time.time() < deadline and ap is None:
        pending = mgr.list_pending(run_id="r1")
        if pending:
            ap = pending[0]
        else:
            time.sleep(0.02)
    mgr.decide(ap["approval_id"], approve=False, operator="operator-1")
    t.join(timeout=10)
    r = box["result"]
    assert r.ok is False and "denied" in r.error
    assert adapter.calls == 0
    assert store.ledger_list(run_id="r1") == []  # never reached dispatch
    store.close()


def test_gateway_approval_expiry_fails_closed(tmp_path):
    store = make_store(tmp_path)
    mgr = ApprovalManager(store, expiry_seconds=0.4, preview_bytes=500)
    gw, adapter = make_gateway(store, approval_manager=mgr)
    rc = make_rc()
    r = gw.execute("fake_write", {"path": "a.txt", "content": "x"}, ctx=ctx(),
                   run_context=rc)
    assert r.ok is False and "expired" in r.error
    assert adapter.calls == 0
    store.close()


def test_gateway_revalidates_policy_before_consuming_approval(tmp_path):
    store = make_store(tmp_path)
    gw, adapter = make_gateway(store)
    rc = make_rc()
    t, box = _run_in_thread(gw, {"path": "a.txt", "content": "x"}, rc)
    mgr = gw.approval_manager
    deadline = time.time() + 5
    ap = None
    while time.time() < deadline and ap is None:
        pending = mgr.list_pending(run_id="r1")
        if pending:
            ap = pending[0]
        else:
            time.sleep(0.02)
    # policy revoked WHILE the approval was pending -> must not execute
    gw.policy.per_agent["alice"] = frozenset()
    mgr.decide(ap["approval_id"], approve=True, operator="operator-1")
    t.join(timeout=10)
    r = box["result"]
    assert r.ok is False and "re-denied" in r.error
    assert adapter.calls == 0
    store.close()


def test_gateway_async_approval_flow(tmp_path):
    store = make_store(tmp_path)
    gw, adapter = make_gateway(store)
    rc = make_rc()

    async def go():
        task = asyncio.create_task(gw.aexecute(
            "fake_write", {"path": "b.txt", "content": "y"}, ctx=ctx(), run_context=rc))
        mgr = gw.approval_manager
        ap = None
        for _ in range(250):
            pending = mgr.list_pending(run_id="r1")
            if pending:
                ap = pending[0]
                break
            await asyncio.sleep(0.02)
        assert ap is not None
        mgr.decide(ap["approval_id"], approve=True, operator="operator-async")
        return await task, ap

    (r, ap), = [asyncio.run(go())]
    assert r.ok is True and adapter.calls == 1
    assert store.get_approval(ap["approval_id"])["status"] == ApprovalState.CONSUMED.value
    assert store.ledger_get(r.execution_id)["status"] == ToolExecutionStatus.COMPLETED.value
    store.close()


def test_mutating_timeout_marks_ledger_unknown(tmp_path):
    store = make_store(tmp_path)
    spec = write_spec(timeout_seconds=0.05)

    class SlowMutating(MutatingAdapter):
        def execute(self, args, ctx):
            time.sleep(0.4)  # side effect may or may not have happened -> UNKNOWN
            self.calls += 1
            return "late"

    adapter = SlowMutating(spec)
    gw, _ = make_gateway(store, adapter=adapter, spec=spec)
    rc = make_rc()
    t, box = _run_in_thread(gw, {"path": "a.txt", "content": "x"}, rc)
    mgr = gw.approval_manager
    deadline = time.time() + 5
    ap = None
    while time.time() < deadline and ap is None:
        pending = mgr.list_pending(run_id="r1")
        if pending:
            ap = pending[0]
        else:
            time.sleep(0.02)
    mgr.decide(ap["approval_id"], approve=True, operator="op")
    t.join(timeout=10)
    r = box["result"]
    assert r.ok is False and r.status == "timeout"
    row = store.ledger_get(r.execution_id)
    assert row["status"] == ToolExecutionStatus.UNKNOWN.value  # never auto-replayed
    # occupancy held until the underlying thread terminates
    time.sleep(0.5)
    store.close()


# ---- schema migration v4 -> v5 -> v6 ----

def test_schema_v4_migrates_to_v6_with_ledger_and_quota_tables(tmp_path):
    db = str(tmp_path / "m.db")
    store = CheckpointStore(db, persist_mode=PersistMode.RESUMABLE, lease_ttl_seconds=60.0)
    store.close()
    import sqlite3

    conn = sqlite3.connect(db)
    conn.execute("DROP TABLE tool_executions")
    conn.execute("DROP TABLE approvals")
    conn.execute("UPDATE meta SET value='4' WHERE key='schema_version'")
    conn.commit()
    conn.close()
    store2 = CheckpointStore(db, persist_mode=PersistMode.RESUMABLE, lease_ttl_seconds=60.0)
    with store2._lock:
        tables = {r[0] for r in store2._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        v = store2._conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
    assert v == "13" and {"tool_executions", "approvals", "quota_reservations", "api_reservations",
                         "quota_usage", "quota_policies", "quota_policy_audit",
                         "benchmark_evidence", "routing_decisions"} <= tables
    store2.close()


def test_schema_version_above_current_is_rejected(tmp_path):
    from geosgan_hive.errors import CheckpointError

    db = str(tmp_path / "x.db")
    store = CheckpointStore(db, persist_mode=PersistMode.RESUMABLE, lease_ttl_seconds=60.0)
    store.close()
    import sqlite3

    conn = sqlite3.connect(db)
    conn.execute("UPDATE meta SET value='999' WHERE key='schema_version'")
    conn.commit()
    conn.close()
    with pytest.raises(CheckpointError):
        CheckpointStore(db, persist_mode=PersistMode.RESUMABLE, lease_ttl_seconds=60.0)


# ---- subprocess crash tests ----

CRASH_BEFORE_SIDE_EFFECT = """
import os, sys
sys.path.insert(0, "src")
from geosgan_hive.checkpoints import CheckpointStore, PersistMode

db = sys.argv[1]
store = CheckpointStore(db, persist_mode=PersistMode.RESUMABLE, lease_ttl_seconds=60.0)
store.create_run(run_id="crash", owner="w", task="t", agent="alice", model="m",
                 root_task_id="r")
store.reserve_tool_durable("crash")
store.ledger_start(execution_id="e-crash", run_id="crash", agent="alice",
                   tool="fake_write", args_sha256="h", status="started",
                   lease_owner="w", lease_generation=0)
os._exit(9)  # hard crash AFTER ledger start, BEFORE the side effect
"""

CRASH_AFTER_SIDE_EFFECT = """
import os, sys
sys.path.insert(0, "src")
from geosgan_hive.checkpoints import CheckpointStore, PersistMode

db, marker = sys.argv[1], sys.argv[2]
store = CheckpointStore(db, persist_mode=PersistMode.RESUMABLE, lease_ttl_seconds=60.0)
store.create_run(run_id="crash", owner="w", task="t", agent="alice", model="m",
                 root_task_id="r")
store.reserve_tool_durable("crash")
store.ledger_start(execution_id="e-crash", run_id="crash", agent="alice",
                   tool="fake_write", args_sha256="h", status="started",
                   lease_owner="w", lease_generation=0)
with open(marker, "w") as f:  # the side effect actually happened
    f.write("mutated")
os._exit(9)  # hard crash AFTER the side effect, BEFORE ledger finish
"""


def _run_crash(script, *args):
    proc = subprocess.run([sys.executable, "-c", script, *args],
                          cwd=os.getcwd(), capture_output=True, text=True, timeout=60)
    assert proc.returncode == 9


def test_crash_after_ledger_start_before_side_effect(tmp_path):
    db = str(tmp_path / "c1.db")
    store = CheckpointStore(db, persist_mode=PersistMode.RESUMABLE, lease_ttl_seconds=60.0)
    store.close()
    _run_crash(CRASH_BEFORE_SIDE_EFFECT, db)
    store2 = CheckpointStore(db, persist_mode=PersistMode.RESUMABLE, lease_ttl_seconds=60.0)
    cp = store2.get("crash")
    assert cp.tool_reservations == 1  # consumed reservation conservatively retained
    assert cp.state == ExecutionState.RUNNING  # lease still live; crash during RUNNING
    row = store2.ledger_get("e-crash")
    assert row["status"] == ToolExecutionStatus.STARTED.value  # NOT auto-completed
    assert store2.ledger_list(run_id="crash", status="completed") == []  # no auto-replay
    store2.close()


def test_crash_after_side_effect_yields_unknown_resolved_by_operator(tmp_path):
    db = str(tmp_path / "c2.db")
    marker = str(tmp_path / "marker.txt")
    store = CheckpointStore(db, persist_mode=PersistMode.RESUMABLE, lease_ttl_seconds=60.0)
    store.close()
    _run_crash(CRASH_AFTER_SIDE_EFFECT, db, marker)
    assert open(marker).read() == "mutated"  # side effect happened exactly once
    store2 = CheckpointStore(db, persist_mode=PersistMode.RESUMABLE, lease_ttl_seconds=60.0)
    row = store2.ledger_get("e-crash")
    assert row["status"] == ToolExecutionStatus.STARTED.value
    # operator marks it UNKNOWN first (outcome uncertain), then resolves explicitly
    assert store2.ledger_update("e-crash", status="unknown", error="crashed",
                                lease_owner="w", lease_generation=0) is True
    assert store2.ledger_resolve("e-crash", decision="completed", operator="op",
                                 evidence="marker file verified on disk") is True
    assert store2.ledger_get("e-crash")["status"] == ToolExecutionStatus.COMPLETED.value
    # recovery does not re-execute: the ledger still has exactly ONE row
    assert len(store2.ledger_list(run_id="crash")) == 1
    store2.close()
