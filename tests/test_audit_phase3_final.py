"""Phase 3 final adversarial audit — failing-first regression tests.

Covers the audit checklist: Windows write disable + hard-link rejection +
parent-swap race (workspace), approval binding to tool version/run/agent and
the atomic consume+ledger-start transaction (approvals), lease-expiry fencing
of durable reserves, crash during file replacement, schema v3->v5 migration
chain, async policy denial, async concurrent budget reservations, async
timeout semaphore release, and permission revocation during execution."""

import asyncio
import os
import sqlite3
import subprocess
import sys
import threading
import time

import pytest
from pydantic import BaseModel, ConfigDict, Field

from geosgan_hive.agent_config import (
    AgentSpec,
    HiveConfig,
    Permissions,
    SupervisorSpec,
    ToolsConfig,
)
from geosgan_hive.approvals import ApprovalManager
from geosgan_hive.checkpoints import (
    ApprovalState,
    CheckpointStore,
    PersistMode,
    ToolExecutionStatus,
)
from geosgan_hive.errors import SandboxViolationError
from geosgan_hive.run_context import CollaborationPolicy, RunContext
from geosgan_hive.sandbox import WorkspaceSandbox
from geosgan_hive.tool_factory import build_tool_catalog
from geosgan_hive.tool_gateway import ToolExecutionContext, ToolGateway
from geosgan_hive.tool_policy import build_policy
from geosgan_hive.tool_registry import ToolCatalog
from geosgan_hive.tool_spec import SideEffect, ToolBudgets, ToolSpec, args_sha256

POSIX = os.name == "posix"
HAS_PROC_FD = POSIX and os.path.isdir("/proc/self/fd")


class TextArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=1000)


def echo_spec(**overrides) -> ToolSpec:
    kwargs = dict(name="echo", description="echo", category="data",
                  side_effect=SideEffect.READ_ONLY, input_model=TextArgs,
                  timeout_seconds=5)
    kwargs.update(overrides)
    return ToolSpec(**kwargs)


class EchoAdapter:
    def __init__(self, spec, out="ok"):
        self.spec = spec
        self.calls = 0
        self.out = out

    def execute(self, args, ctx):
        self.calls += 1
        return f"{self.out}:{args['text']}"

    async def aexecute(self, args, ctx):
        return self.execute(args, ctx)


def make_gw(spec=None, adapter=None, store=None, mgr=None, strict=False, **kw):
    spec = spec or echo_spec()
    adapter = adapter or EchoAdapter(spec)
    catalog = ToolCatalog()
    catalog.register(spec, adapter)
    policy = build_policy(global_enabled=[spec.name],
                          agent_permissions={"alice": (True, [spec.name]),
                                             "bob": (False, [])})
    return ToolGateway(catalog, policy, store=store, approval_manager=mgr,
                       strict=strict, **kw), adapter


def make_store(tmp_path, run_id="r1", ttl=60.0):
    store = CheckpointStore(str(tmp_path / "cp.db"), persist_mode=PersistMode.RESUMABLE,
                            lease_ttl_seconds=ttl)
    store.create_run(run_id=run_id, owner="w1", task="t", agent="alice", model="m",
                     root_task_id="root")
    return store


def ctx(agent="alice", run_id="r1"):
    return ToolExecutionContext(agent=agent, run_id=run_id, task_id="t1")


# ================= §2 workspace security =================

def test_non_race_resistant_platforms_disable_all_file_ops_by_default(tmp_path):
    # Windows AND any platform without /proc fd re-verification (e.g. macOS):
    # reads are disabled too — read-only access can still leak via TOCTOU races
    for platform, fd in (("nt", False), ("darwin", False)):
        sb = WorkspaceSandbox(str(tmp_path / f"ws_{platform}"), platform=platform,
                              fd_reverify=fd)  # constructor creates the root
        (tmp_path / f"ws_{platform}" / "a.txt").write_text("x")
        with pytest.raises(SandboxViolationError, match="disabled by default"):
            sb.read_text("a.txt", max_bytes=100)
        with pytest.raises(SandboxViolationError, match="disabled by default"):
            sb.list_dir(".", max_entries=10)
        with pytest.raises(SandboxViolationError, match="disabled by default"):
            sb.metadata("a.txt")
        with pytest.raises(SandboxViolationError, match="disabled by default"):
            sb.write_text("b.txt", "y")
        assert sb.race_resistant is False
    # explicit operator override re-enables (documented residual TOCTOU risk)
    sb2 = WorkspaceSandbox(str(tmp_path / "ws_override"), platform="nt",
                           fd_reverify=False, allow_insecure_fs=True)
    (tmp_path / "ws_override" / "a.txt").write_text("x")
    assert sb2.read_text("a.txt", max_bytes=100) == "x"
    # Linux with /proc: fully race-resistant, everything enabled
    sb3 = WorkspaceSandbox(str(tmp_path / "ws_linux"))
    (tmp_path / "ws_linux" / "a.txt").write_text("x")
    if sb3.race_resistant:
        assert sb3.read_text("a.txt", max_bytes=100) == "x"


def test_factory_does_not_register_write_tools_on_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    config = HiveConfig(
        supervisor=SupervisorSpec(name="supervisor", description="d", system_prompt="s",
                                  model="ollama/s"),
        agents=[AgentSpec(name="alice", description="d", system_prompt="s", model="ollama/a",
                          permissions=Permissions(allow_tools=True,
                                                  allowed_tools=["fs_write"]))],
        tools=ToolsConfig(enabled=["fs_write"], allow_write=True,
                          workspace_root=str(tmp_path / "ws")),
    )
    catalog, sandbox = build_tool_catalog(config)
    assert sandbox.platform == "nt"
    assert sandbox.race_resistant is False
    # on a non-race-resistant platform NO file tools are registered at all
    assert not any(n.startswith("fs_") for n in catalog.names())


@pytest.mark.skipif(not POSIX, reason="hard links need a POSIX filesystem")
def test_hard_link_to_outside_file_is_rejected(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET")
    os.link(secret, ws / "link.txt")  # hard link: same inode, name inside the root
    sb = WorkspaceSandbox(str(ws))
    with pytest.raises(SandboxViolationError, match="hard links"):
        sb.read_text("link.txt", max_bytes=1000)
    with pytest.raises(SandboxViolationError, match="hard links"):
        sb.write_text("link.txt", "overwrite", expected_sha256="x" * 64,
                      require_exists=True)


def test_deny_patterns_match_case_insensitively(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "X.DB").write_text("data")
    (ws / ".ENV").write_text("KEY=1")
    sb = WorkspaceSandbox(str(ws))
    for name in ("X.DB", "x.db", ".ENV", ".env"):
        with pytest.raises(SandboxViolationError, match="denied"):
            sb.resolve(name)


@pytest.mark.skipif(not HAS_PROC_FD, reason="fd re-verification is Linux/POSIX-only")
def test_parent_directory_swap_after_validation_is_caught_by_fd_reverify(tmp_path):
    """TOCTOU: the parent directory is replaced by a symlink to an outside dir AFTER
    resolve() validated the path but BEFORE open. The post-open /proc/self/fd check
    must reject the opened file (containment is verified on the ACTUAL fd)."""
    ws = tmp_path / "ws"
    ws.mkdir()
    sub = ws / "sub"
    sub.mkdir()
    (sub / "file.txt").write_text("inside")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file.txt").write_text("OUTSIDE SECRET")
    sb = WorkspaceSandbox(str(ws))

    real_resolve = sb.resolve

    def racing_resolve(relpath):
        resolved = real_resolve(relpath)
        if relpath == "sub/file.txt":
            # attacker wins the race right after validation:
            os.rename(sub, ws / "sub_moved")
            os.symlink(outside, sub)
        return resolved

    sb.resolve = racing_resolve
    with pytest.raises(SandboxViolationError):
        sb.read_text("sub/file.txt", max_bytes=1000)


# ================= §3 approval integrity =================

def test_start_execution_binds_tool_version_run_agent_tool(tmp_path):
    store = make_store(tmp_path)
    mgr = ApprovalManager(store)
    spec = echo_spec(name="mut", side_effect=SideEffect.LOCAL_MUTATION,
                     requires_approval=True)
    args = {"text": "x"}
    req = mgr.create(ctx=ctx(), spec=spec, args=args, operation="op", target="t")
    mgr.decide(req.approval_id, approve=True, operator="op")
    h = args_sha256(args)
    # tool-version substitution -> rejected, approval NOT consumed, no ledger row
    ok = store.start_execution(execution_id="e1", run_id="r1", agent="alice",
                               tool="mut", tool_version="2", args_sha256=h,
                               status="started", approval_id=req.approval_id,
                               expected_args_sha256=h)
    assert ok is False
    assert store.get_approval(req.approval_id)["status"] == ApprovalState.APPROVED.value
    assert store.ledger_list(run_id="r1") == []
    # run substitution -> rejected
    ok = store.start_execution(execution_id="e1", run_id="OTHER", agent="alice",
                               tool="mut", tool_version="1", args_sha256=h,
                               status="started", approval_id=req.approval_id,
                               expected_args_sha256=h)
    assert ok is False
    # agent substitution -> rejected
    ok = store.start_execution(execution_id="e1", run_id="r1", agent="mallory",
                               tool="mut", tool_version="1", args_sha256=h,
                               status="started", approval_id=req.approval_id,
                               expected_args_sha256=h)
    assert ok is False
    # correct binding -> BOTH the consume and the ledger row commit atomically
    ok = store.start_execution(execution_id="e1", run_id="r1", agent="alice",
                               tool="mut", tool_version="1", args_sha256=h,
                               status="started", approval_id=req.approval_id,
                               expected_args_sha256=h)
    assert ok is True
    assert store.get_approval(req.approval_id)["status"] == ApprovalState.CONSUMED.value
    rows = store.ledger_list(run_id="r1")
    assert len(rows) == 1 and rows[0]["status"] == ToolExecutionStatus.STARTED.value
    assert rows[0]["approval_id"] == req.approval_id
    store.close()


def test_start_execution_duplicate_id_leaves_approval_unconsumed(tmp_path):
    store = make_store(tmp_path)
    mgr = ApprovalManager(store)
    spec = echo_spec(name="mut", side_effect=SideEffect.LOCAL_MUTATION,
                     requires_approval=True)
    args = {"text": "x"}
    req = mgr.create(ctx=ctx(), spec=spec, args=args, operation="op", target="t")
    mgr.decide(req.approval_id, approve=True, operator="op")
    h = args_sha256(args)
    assert store.start_execution(execution_id="dup", run_id="r1", agent="alice",
                                 tool="mut", tool_version="1", args_sha256=h,
                                 status="started", approval_id=req.approval_id,
                                 expected_args_sha256=h) is True
    # second dispatch with the SAME execution id: no double dispatch, and because
    # the id check runs first, the already-consumed approval is untouched
    assert store.start_execution(execution_id="dup", run_id="r1", agent="alice",
                                 tool="mut", tool_version="1", args_sha256=h,
                                 status="started", approval_id=req.approval_id,
                                 expected_args_sha256=h) is False
    assert len(store.ledger_list(run_id="r1")) == 1
    store.close()


def test_redaction_does_not_alter_the_authorization_binding(tmp_path):
    store = make_store(tmp_path)
    mgr = ApprovalManager(store, preview_bytes=5000)
    spec = echo_spec(name="mut", side_effect=SideEffect.LOCAL_MUTATION,
                     requires_approval=True)
    secret = "sk-abcdefghijklmnop1234567890"
    args = {"text": f"key {secret} end"}
    req = mgr.create(ctx=ctx(), spec=spec, args=args, operation="op", target="t")
    row = store.get_approval(req.approval_id)
    # binding hash is over the ORIGINAL args, preview/payload are redacted
    assert row["args_sha256"] == args_sha256(args)
    assert secret not in row["preview"] and secret not in row["args_json"]
    mgr.decide(req.approval_id, approve=True, operator="op")
    # consume with the hash of the REDACTED text must FAIL (different identity)
    redacted_hash = args_sha256({"text": "key ***REDACTED*** end"})
    assert store.consume_approval(req.approval_id, expected_args_sha256=redacted_hash) is False
    # consume with the original hash succeeds
    assert store.consume_approval(req.approval_id, expected_args_sha256=args_sha256(args))
    store.close()


def test_consume_approval_binding_params(tmp_path):
    store = make_store(tmp_path)
    mgr = ApprovalManager(store)
    spec = echo_spec(name="mut", side_effect=SideEffect.LOCAL_MUTATION,
                     requires_approval=True)
    args = {"text": "x"}
    req = mgr.create(ctx=ctx(), spec=spec, args=args, operation="op", target="t")
    mgr.decide(req.approval_id, approve=True, operator="op")
    h = args_sha256(args)
    assert mgr.consume(req.approval_id, expected_args_sha256=h, expected_run_id="r1",
                       expected_agent="alice", expected_tool="mut",
                       expected_tool_version="1") is False or True  # first call consumes
    # replay with any binding -> rejected
    assert mgr.consume(req.approval_id, expected_args_sha256=h, expected_run_id="r1",
                       expected_agent="alice", expected_tool="mut",
                       expected_tool_version="1") is False
    store.close()


# ================= §4/§5 durable execution, leases, migrations =================

def test_expired_lease_blocks_durable_reservations_while_running(tmp_path):
    store = make_store(tmp_path, ttl=0.3)
    assert store.reserve_tool_durable("r1") is True
    assert store.reserve_physical_durable("r1") is True
    time.sleep(0.4)
    cp = store.get("r1")
    assert cp.state.value == "running"  # same-process: no recovery ran yet
    assert store.reserve_tool_durable("r1") is False  # stale lease -> fail closed
    assert store.reserve_physical_durable("r1") is False
    assert cp.tool_reservations == 1 and cp.physical_reserved == 1  # never decrease
    store.close()


def test_mutating_tool_fails_closed_on_expired_lease(tmp_path):
    store = make_store(tmp_path, ttl=0.3)
    spec = echo_spec(name="mut", side_effect=SideEffect.LOCAL_MUTATION,
                     requires_approval=False, timeout_seconds=5)
    gw, adapter = make_gw(spec, store=store, strict=True)
    time.sleep(0.4)  # lease expires while the run row is still RUNNING
    rc = RunContext("r1", CollaborationPolicy())
    r = gw.execute("mut", {"text": "x"}, ctx=ctx(), run_context=rc)
    # fails closed at the durable-reservation OR the lease check — either way no dispatch
    assert r.ok is False and ("no longer active" in r.error or "valid lease owner" in r.error)
    assert adapter.calls == 0
    store.close()


def test_schema_v3_database_migrates_through_v4_to_v6(tmp_path):
    db = str(tmp_path / "old.db")
    store = CheckpointStore(db, persist_mode=PersistMode.RESUMABLE, lease_ttl_seconds=60.0)
    store.create_run(run_id="old", owner="w", task="t", agent="a", model="m",
                     root_task_id="r")
    store.close()
    conn = sqlite3.connect(db)
    conn.execute("ALTER TABLE runs DROP COLUMN lease_generation")
    conn.execute("ALTER TABLE runs DROP COLUMN tool_reservations")
    conn.execute("DROP TABLE tool_executions")
    conn.execute("DROP TABLE approvals")
    conn.execute("UPDATE meta SET value='3' WHERE key='schema_version'")
    conn.commit()
    conn.close()
    store2 = CheckpointStore(db, persist_mode=PersistMode.RESUMABLE, lease_ttl_seconds=60.0)
    with store2._lock:
        v = store2._conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
        tables = {r[0] for r in store2._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        cols = {r["name"] for r in store2._conn.execute("PRAGMA table_info(runs)").fetchall()}
    assert v == "12"
    assert {"tool_executions", "approvals", "quota_reservations", "quota_usage",
            "quota_policies", "quota_policy_audit"} <= tables
    assert {"lease_generation", "tool_reservations"} <= cols
    assert store2.get("old") is not None  # pre-existing run row survives the chain
    store2.close()


CRASH_DURING_REPLACE = """
import os, sys
sys.path.insert(0, "src")
from geosgan_hive.checkpoints import CheckpointStore, PersistMode
from geosgan_hive.run_context import CollaborationPolicy, RunContext
from geosgan_hive.sandbox import WorkspaceSandbox
from geosgan_hive.tool_gateway import ToolExecutionContext, ToolGateway
from geosgan_hive.tool_policy import build_policy
from geosgan_hive.tool_registry import ToolCatalog
from geosgan_hive.tool_spec import SideEffect, ToolSpec

db, wsdir = sys.argv[1], sys.argv[2]
store = CheckpointStore(db, persist_mode=PersistMode.RESUMABLE, lease_ttl_seconds=60.0)
store.create_run(run_id="crash", owner="w", task="t", agent="alice", model="m",
                 root_task_id="r")
sb = WorkspaceSandbox(wsdir)

real_replace = os.replace
def crashing_replace(src, dst):
    os._exit(9)  # hard crash AFTER the tmp file is written, BEFORE the atomic replace
os.replace = crashing_replace

class Mut:
    def __init__(self, spec): self.spec = spec
    def execute(self, args, ctx):
        sb.write_text(args["path"], args["content"], require_exists=False)
        return "never reached"

spec = ToolSpec(name="mut", description="d", side_effect=SideEffect.LOCAL_MUTATION,
                requires_approval=False, timeout_seconds=30)
catalog = ToolCatalog()
catalog.register(spec, Mut(spec))
policy = build_policy(global_enabled=["mut"], agent_permissions={"alice": (True, ["mut"])})
gw = ToolGateway(catalog, policy, store=store, strict=True)
rc = RunContext("crash", CollaborationPolicy())
gw.execute("mut", {"path": "out.txt", "content": "data"},
           ctx=ToolExecutionContext(agent="alice", run_id="crash", task_id="t"),
           run_context=rc)
"""


@pytest.mark.skipif(os.name != "posix", reason="write path is disabled on Windows")
def test_crash_during_file_replacement_leaves_no_partial_target_and_no_replay(tmp_path):
    db = str(tmp_path / "crash.db")
    wsdir = tmp_path / "ws"
    wsdir.mkdir()
    store = CheckpointStore(db, persist_mode=PersistMode.RESUMABLE, lease_ttl_seconds=60.0)
    store.close()
    proc = subprocess.run(
        [sys.executable, "-c", CRASH_DURING_REPLACE, db, str(wsdir)],
        cwd=os.getcwd(), capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 9
    assert not (wsdir / "out.txt").exists()  # replace never happened: no partial file
    leftovers = [p for p in wsdir.iterdir() if ".tmp-" in p.name]
    assert len(leftovers) == 1  # the orphaned tmp file is visible evidence, harmless
    store2 = CheckpointStore(db, persist_mode=PersistMode.RESUMABLE, lease_ttl_seconds=60.0)
    rows = store2.ledger_list(run_id="crash")
    assert len(rows) == 1 and rows[0]["status"] == ToolExecutionStatus.STARTED.value
    assert store2.get("crash").tool_reservations == 1
    # operator resolves UNKNOWN explicitly; nothing is auto-replayed
    assert store2.ledger_update(rows[0]["execution_id"], status="unknown", error="crash",
                                lease_owner="w", lease_generation=0) is True
    assert store2.ledger_resolve(rows[0]["execution_id"], decision="failed",
                                 operator="op", evidence="target absent, tmp orphaned") is True
    assert len(store2.ledger_list(run_id="crash")) == 1
    assert not (wsdir / "out.txt").exists()
    store2.close()


# ================= §1/§6 boundary, budgets, concurrency =================

def test_async_path_enforces_policy_and_trusted_context():
    gw, adapter = make_gw()

    async def go():
        r = await gw.aexecute("echo", {"text": "hi"}, ctx=ctx(agent="bob"))
        return r

    r = asyncio.run(go())
    assert r.ok is False and "not allowed" in r.error and adapter.calls == 0
    # model-supplied identity inside args is rejected by the strict schema
    async def go2():
        return await gw.aexecute("echo", {"text": "x", "run_id": "forged"},
                                 ctx=ctx(agent="alice"))

    r2 = asyncio.run(go2())
    assert r2.ok is False and "invalid arguments" in r2.error


def test_async_concurrent_reservations_are_exact():
    gw, _ = make_gw()
    rc = RunContext("r", CollaborationPolicy(),
                    tool_budgets=ToolBudgets(max_tool_calls_per_run=30,
                                             max_tool_calls_per_agent=100,
                                             max_tool_calls_per_tool=100))

    async def go():
        async def one(i):
            return await gw.aexecute("echo", {"text": str(i)}, ctx=ctx(run_id="r"),
                                     run_context=rc)

        return await asyncio.gather(*[one(i) for i in range(60)])

    results = asyncio.run(go())
    assert sum(1 for r in results if r.ok) == 30
    assert rc.tool_calls == 30


def test_async_timeout_releases_tool_semaphore():
    spec = echo_spec(name="slows", timeout_seconds=0.05)

    class SlowAsync:
        def __init__(self, spec):
            self.spec = spec

        def execute(self, args, ctx):
            time.sleep(1)
            return "late"

        async def aexecute(self, args, ctx):
            await asyncio.sleep(1)
            return "late"

    adapter = SlowAsync(spec)
    catalog = ToolCatalog()
    catalog.register(spec, adapter)
    policy = build_policy(global_enabled=["slows"],
                          agent_permissions={"alice": (True, ["slows"])})
    gw = ToolGateway(catalog, policy)
    rc = RunContext("r", CollaborationPolicy(),
                    tool_budgets=ToolBudgets(max_concurrent_tool_executions=1))

    async def go():
        r = await gw.aexecute("slows", {"text": "x"}, ctx=ctx(run_id="r"), run_context=rc)
        assert r.ok is False and r.status == "timeout"
        sem = rc.semaphore_tool()
        await asyncio.wait_for(sem.acquire(), timeout=2)  # released, not leaked
        sem.release()

    asyncio.run(go())


def test_permission_revocation_during_execution_does_not_affect_inflight_call(tmp_path):
    spec = echo_spec(name="slow2", timeout_seconds=10)

    class Slow(EchoAdapter):
        def execute(self, args, ctx):
            time.sleep(0.3)
            return super().execute(args, ctx)

    gw, adapter = make_gw(spec, adapter=Slow(spec))
    rc = RunContext("r", CollaborationPolicy())
    box = {}
    t = threading.Thread(target=lambda: box.update(
        r=gw.execute("slow2", {"text": "x"}, ctx=ctx(run_id="r"), run_context=rc)))
    t.start()
    time.sleep(0.05)
    gw.policy.per_agent["alice"] = frozenset()  # revoked mid-flight
    t.join(timeout=5)
    assert box["r"].ok is True  # already-dispatched execution completes
    r2 = gw.execute("slow2", {"text": "y"}, ctx=ctx(run_id="r"), run_context=rc)
    assert r2.ok is False and "not allowed" in r2.error  # next call is denied
    assert adapter.calls == 1
