"""Phase 3 final-approval gate tests: platform fail-closed, ownership/generation
fencing of every durable dispatch, complete approval binding (incl. task_id),
preview integrity, metadata-only payload policy, and recovery non-replay."""

import time
from datetime import timedelta

import pytest

from geosgan_hive.approvals import ApprovalManager
from geosgan_hive.checkpoints import (
    ApprovalState,
    CheckpointStore,
    ExecutionState,
    PersistMode,
    ToolExecutionStatus,
)
from geosgan_hive.errors import CheckpointError
from geosgan_hive.run_context import CollaborationPolicy, RunContext
from geosgan_hive.tool_gateway import ToolExecutionContext, ToolGateway
from geosgan_hive.tool_policy import build_policy
from geosgan_hive.tool_registry import ToolCatalog
from geosgan_hive.tool_spec import SideEffect, ToolSpec, args_sha256


def make_store(tmp_path, run_id="r1", ttl=60.0, mode=PersistMode.RESUMABLE):
    store = CheckpointStore(str(tmp_path / "cp.db"), persist_mode=mode,
                            lease_ttl_seconds=ttl)
    store.create_run(run_id=run_id, owner="w1", task="t", agent="alice", model="m",
                     root_task_id="root")
    return store


def takeover(store, run_id="r1", owner="w2", gen=1):
    """Simulate a competing worker acquiring a fresh valid lease (same atomic effect
    as claim_lease+mark_running): new owner, bumped generation, unexpired lease."""
    future = (store.utcnow() + timedelta(seconds=60)).isoformat(timespec="milliseconds")
    with store._lock, store._conn:
        store._conn.execute(
            "UPDATE runs SET lease_owner=?, lease_generation=?, lease_expires_at=? "
            "WHERE run_id=?", (owner, gen, future, run_id)
        )


def ctx(agent="alice", run_id="r1", task_id="t1"):
    return ToolExecutionContext(agent=agent, run_id=run_id, task_id=task_id)


# ================= §2 ownership + generation fencing =================

def test_stale_worker_cannot_reserve_after_lease_takeover(tmp_path):
    store = make_store(tmp_path)
    # w1 (gen 0) is the current owner: its reserves succeed
    assert store.reserve_tool_durable("r1", lease_owner="w1", lease_generation=0) is True
    assert store.reserve_physical_durable("r1", lease_owner="w1", lease_generation=0) is True
    takeover(store, owner="w2", gen=1)  # competing worker acquires a fresh lease
    # stale worker w1/gen0 is now fenced out of BOTH durable dispatch reservations
    assert store.reserve_tool_durable("r1", lease_owner="w1", lease_generation=0) is False
    assert store.reserve_physical_durable("r1", lease_owner="w1", lease_generation=0) is False
    # wrong generation with the right owner is fenced too
    assert store.reserve_tool_durable("r1", lease_owner="w2", lease_generation=0) is False
    # the fresh owner+generation can dispatch
    assert store.reserve_tool_durable("r1", lease_owner="w2", lease_generation=1) is True
    assert store.reserve_physical_durable("r1", lease_owner="w2", lease_generation=1) is True
    # counters only ever increased
    cp = store.get("r1")
    assert cp.tool_reservations == 2 and cp.physical_reserved == 2
    store.close()


def test_execution_start_fenced_by_owner_and_generation(tmp_path):
    store = make_store(tmp_path)
    mgr = ApprovalManager(store)
    spec = ToolSpec(name="mut", description="d", side_effect=SideEffect.LOCAL_MUTATION,
                    requires_approval=True)
    args = {"path": "a.txt", "content": "x"}
    req = mgr.create(ctx=ctx(), spec=spec, args=args, operation="create_file",
                     target="a.txt")
    mgr.decide(req.approval_id, approve=True, operator="op")
    h = args_sha256(args)
    takeover(store, owner="w2", gen=1)
    # stale owner/generation: no ledger row, approval NOT consumed (atomic)
    ok = store.start_execution(execution_id="e1", run_id="r1", agent="alice", tool="mut",
                               tool_version="1", args_sha256=h, status="started",
                               approval_id=req.approval_id, expected_args_sha256=h,
                               lease_owner="w1", lease_generation=0, expected_task_id="t1")
    assert ok is False
    assert store.ledger_list(run_id="r1") == []
    assert store.get_approval(req.approval_id)["status"] == ApprovalState.APPROVED.value
    # ledger_start (non-approval path) is fenced the same way
    assert store.ledger_start(execution_id="e2", run_id="r1", agent="alice", tool="mut",
                              args_sha256=h, status="started", lease_owner="w1",
                              lease_generation=0) is False
    # fresh owner+generation succeeds atomically
    ok = store.start_execution(execution_id="e1", run_id="r1", agent="alice", tool="mut",
                               tool_version="1", args_sha256=h, status="started",
                               approval_id=req.approval_id, expected_args_sha256=h,
                               lease_owner="w2", lease_generation=1, expected_task_id="t1")
    assert ok is True
    assert store.get_approval(req.approval_id)["status"] == ApprovalState.CONSUMED.value
    assert len(store.ledger_list(run_id="r1")) == 1
    store.close()


def test_terminal_commits_fenced_by_owner_and_generation(tmp_path):
    store = make_store(tmp_path)
    takeover(store, owner="w2", gen=1)
    with pytest.raises(CheckpointError):
        store.mark_completed("r1", owner="w1", result="x", counters_json="{}",
                             steps_json="[]", route_json="[]", transcript_json=None,
                             step_count=1)  # stale owner
    with pytest.raises(CheckpointError):
        store.mark_completed("r1", owner="w2", lease_generation=0, result="x",
                             counters_json="{}", steps_json="[]", route_json="[]",
                             transcript_json=None, step_count=1)  # stale generation
    store.mark_completed("r1", owner="w2", lease_generation=1, result="x",
                         counters_json="{}", steps_json="[]", route_json="[]",
                         transcript_json=None, step_count=1)  # fresh: accepted
    assert store.get("r1").state == ExecutionState.COMPLETED
    store.close()


def test_gateway_passes_owner_and_generation_on_durable_reserve(tmp_path):
    store = make_store(tmp_path)
    spec = ToolSpec(name="echo", description="d", side_effect=SideEffect.READ_ONLY,
                    timeout_seconds=5)

    class A:
        def __init__(self):
            self.spec = spec

        def execute(self, args, ctx):
            return "ok"

    catalog = ToolCatalog()
    catalog.register(spec, A())
    gw = ToolGateway(catalog, build_policy(global_enabled=["echo"],
                                           agent_permissions={"alice": (True, ["echo"])}),
                     store=store, strict=True)
    captured = {}
    orig = store.reserve_tool_durable

    def spy(run_id, **kwargs):
        captured.update(kwargs)
        return orig(run_id, **kwargs)

    store.reserve_tool_durable = spy
    rc = RunContext("r1", CollaborationPolicy())
    r = gw.execute("echo", {}, ctx=ctx(), run_context=rc)
    assert r.ok, r.error
    assert captured.get("lease_owner") == "w1" and captured.get("lease_generation") == 0
    store.close()


# ================= §3 complete approval binding (incl. task_id) =================

def test_approval_cannot_authorize_a_different_task_in_same_run(tmp_path):
    store = make_store(tmp_path)
    mgr = ApprovalManager(store)
    spec = ToolSpec(name="mut", description="d", side_effect=SideEffect.LOCAL_MUTATION,
                    requires_approval=True)
    args = {"path": "a.txt", "content": "x"}
    req = mgr.create(ctx=ctx(task_id="t1"), spec=spec, args=args, operation="create_file",
                     target="a.txt")
    assert store.get_approval(req.approval_id)["task_id"] == "t1"
    mgr.decide(req.approval_id, approve=True, operator="op")
    h = args_sha256(args)
    # task substitution: same run, same agent, same args, DIFFERENT task -> rejected
    ok = store.start_execution(execution_id="e1", run_id="r1", agent="alice", tool="mut",
                               tool_version="1", args_sha256=h, status="started",
                               approval_id=req.approval_id, expected_args_sha256=h,
                               lease_owner="w1", lease_generation=0, expected_task_id="t2")
    assert ok is False
    assert store.ledger_list(run_id="r1") == []
    assert store.get_approval(req.approval_id)["status"] == ApprovalState.APPROVED.value
    # consume_approval with task binding also rejects substitution
    assert mgr.consume(req.approval_id, expected_args_sha256=h, expected_task_id="t2") is False
    # the correct task binding consumes atomically with the ledger start
    ok = store.start_execution(execution_id="e1", run_id="r1", agent="alice", tool="mut",
                               tool_version="1", args_sha256=h, status="started",
                               approval_id=req.approval_id, expected_args_sha256=h,
                               lease_owner="w1", lease_generation=0, expected_task_id="t1")
    assert ok is True
    store.close()


def test_preview_shows_full_hash_and_truncation_note(tmp_path):
    store = make_store(tmp_path)
    mgr = ApprovalManager(store, expiry_seconds=60.0, preview_bytes=120)
    spec = ToolSpec(name="mut", description="d", side_effect=SideEffect.LOCAL_MUTATION,
                    requires_approval=True)
    args = {"path": "a.txt", "content": "x" * 500}
    req = mgr.create(ctx=ctx(), spec=spec, args=args, operation="create_file",
                     target="a.txt")
    preview = req.preview
    full_hash = args_sha256(args)
    assert full_hash in preview  # FULL 64-char hash, not a truncated prefix
    assert "truncated" in preview  # operator is told the args preview is bounded
    assert len(full_hash) == 64
    # the stored row carries the same full hash + bounded payload
    row = store.get_approval(req.approval_id)
    assert row["args_sha256"] == full_hash
    assert len(row["args_json"]) <= 8_000
    store.close()


# ================= §4 metadata-only payloads + recovery =================

def test_metadata_only_store_keeps_hash_only_tool_payloads(tmp_path):
    store = make_store(tmp_path, mode=PersistMode.METADATA_ONLY)
    mgr = ApprovalManager(store)
    spec = ToolSpec(name="mut", description="d", side_effect=SideEffect.LOCAL_MUTATION,
                    requires_approval=True)
    args = {"path": "a.txt", "content": "secret-content"}
    req = mgr.create(ctx=ctx(), spec=spec, args=args, operation="create_file",
                     target="a.txt")
    row = store.get_approval(req.approval_id)
    assert row["args_json"] == ""  # no payload retained in metadata_only mode
    assert row["args_sha256"] == args_sha256(args)  # hash still binds the action
    assert row["preview"]  # operator-facing bounded preview is still there
    store.ledger_start(execution_id="e1", run_id="r1", agent="alice", tool="mut",
                       args_sha256="h", args_json="SHOULD-NOT-BE-STORED", status="started")
    # (the gateway sends "" in metadata_only; direct store calls store what they get —
    #  the gateway-level test below is the guarantee)
    store.close()
    # gateway level: metadata_only -> ledger row has empty args_json, hash present
    store2 = make_store(tmp_path / "sub2", mode=PersistMode.METADATA_ONLY) if False else None
    store_r = CheckpointStore(str(tmp_path / "cp2.db"), persist_mode=PersistMode.METADATA_ONLY,
                              lease_ttl_seconds=60.0)
    store_r.create_run(run_id="r1", owner="w1", task="t", agent="alice", model="m",
                       root_task_id="root")
    spec2 = ToolSpec(name="echo", description="d", side_effect=SideEffect.READ_ONLY,
                     timeout_seconds=5)

    class A:
        def __init__(self):
            self.spec = spec2

        def execute(self, args, ctx):
            return "ok"

    catalog = ToolCatalog()
    catalog.register(spec2, A())
    gw = ToolGateway(catalog, build_policy(global_enabled=["echo"],
                                           agent_permissions={"alice": (True, ["echo"])}),
                     store=store_r, strict=True)
    rc = RunContext("r1", CollaborationPolicy())
    r = gw.execute("echo", {"payload": "secret-content"}, ctx=ctx(), run_context=rc)
    assert r.ok
    row = store_r.ledger_get(r.execution_id)
    assert row["args_json"] == ""
    assert row["args_sha256"] == args_sha256({"payload": "secret-content"})
    store_r.close()
    # resumable mode DOES persist a bounded, redacted payload (documented)
    store2 = make_store(tmp_path / "sub2")
    gw2_spec = spec2
    catalog2 = ToolCatalog()
    catalog2.register(gw2_spec, A())
    gw2 = ToolGateway(catalog2, build_policy(global_enabled=["echo"],
                                             agent_permissions={"alice": (True, ["echo"])}),
                      store=store2, strict=True)
    rc2 = RunContext("r1", CollaborationPolicy())
    r2 = gw2.execute("echo", {"payload": "visible"}, ctx=ctx(), run_context=rc2)
    assert r2.ok
    row2 = store2.ledger_get(r2.execution_id)
    assert "visible" in row2["args_json"] and len(row2["args_json"]) <= 8_000
    store2.close()


def test_recovery_never_replays_and_unknown_waits_for_operator(tmp_path):
    db = str(tmp_path / "rec.db")
    store = make_store(tmp_path)
    store.close()
    # crash a RUNNING run: expire its lease, then reopen -> recovered as FAILED
    store = CheckpointStore(db, persist_mode=PersistMode.RESUMABLE, lease_ttl_seconds=0.3)
    store.create_run(run_id="r1", owner="w1", task="t", agent="a", model="m",
                     root_task_id="root")
    store.ledger_start(execution_id="done1", run_id="r1", agent="a", tool="t",
                       args_sha256="h", status="completed")
    store.ledger_start(execution_id="unk1", run_id="r1", agent="a", tool="t",
                       args_sha256="h", status="started")
    store.ledger_update("unk1", status="unknown", error="crash")
    store.reserve_tool_durable("r1")
    store.close()
    time.sleep(0.4)
    store2 = CheckpointStore(db, persist_mode=PersistMode.RESUMABLE, lease_ttl_seconds=0.3)
    cp = store2.get("r1")
    assert cp.state == ExecutionState.FAILED  # crashed RUNNING run recovered as FAILED
    assert store2.reserve_tool_durable("r1") is False  # no dispatch after recovery
    # completed row is untouched (never replayed), unknown stays unknown
    assert store2.ledger_get("done1")["status"] == ToolExecutionStatus.COMPLETED.value
    assert store2.ledger_get("unk1")["status"] == ToolExecutionStatus.UNKNOWN.value
    # only the operator moves it
    assert store2.ledger_resolve("unk1", decision="failed", operator="op",
                                 evidence="checked") is True
    assert store2.ledger_get("unk1")["status"] == ToolExecutionStatus.FAILED.value
    assert len(store2.ledger_list(run_id="r1")) == 2  # still exactly 2 rows: no replay
    store2.close()
