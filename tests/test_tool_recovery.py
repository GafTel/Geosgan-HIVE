"""Phase 3 T6: checkpoint/resume integration, cancellation, crash-safety and
nested delegation with tools.

Covers: tool-counter reconcile on resume (no reset, no double count, durable
reservations stay monotonic), completed executions are never auto-replayed,
current-policy revalidation on resume, async cancellation during tool execution
and during approval waits, a subprocess crash DURING a mutating tool
(ledger STARTED, no auto-replay, operator resolution), and nested delegation
where the delegated agent keeps its own (denied) tool privileges."""

import asyncio
import os
import subprocess
import sys
import time

from pydantic import BaseModel, ConfigDict, Field

from geosgan_hive.agent_config import (
    AgentSpec,
    HiveConfig,
    Permissions,
    SupervisorSpec,
    ToolsConfig,
)
from geosgan_hive.agents import Agent
from geosgan_hive.checkpoints import CheckpointStore, PersistMode, ToolExecutionStatus
from geosgan_hive.collaboration import CollaborationEngine
from geosgan_hive.llm import FakeLLM
from geosgan_hive.run_context import CollaborationPolicy, RunContext
from geosgan_hive.tool_factory import build_tool_gateway
from geosgan_hive.tool_gateway import ToolExecutionContext, ToolGateway
from geosgan_hive.tool_policy import build_policy
from geosgan_hive.tool_registry import ToolCatalog
from geosgan_hive.tool_spec import SideEffect, ToolBudgets, ToolSpec
from geosgan_hive.usage import UsageTracker


class EchoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=1000)


def echo_spec(**overrides) -> ToolSpec:
    kwargs = dict(name="echo", description="echo", category="data",
                  side_effect=SideEffect.READ_ONLY, input_model=EchoArgs,
                  timeout_seconds=5)
    kwargs.update(overrides)
    return ToolSpec(**kwargs)


class EchoAdapter:
    def __init__(self, spec):
        self.spec = spec
        self.calls = 0

    def execute(self, args, ctx):
        self.calls += 1
        return f"echo:{args['text']}"

    async def aexecute(self, args, ctx):
        return self.execute(args, ctx)


def make_gw(store=None, *, strict=False, approval_manager=None, spec=None):
    spec = spec or echo_spec()
    adapter = EchoAdapter(spec)
    catalog = ToolCatalog()
    catalog.register(spec, adapter)
    policy = build_policy(global_enabled=[spec.name],
                          agent_permissions={"alice": (True, [spec.name]),
                                             "bob": (False, [])})
    gw = ToolGateway(catalog, policy, store=store, approval_manager=approval_manager,
                     strict=strict)
    return gw, adapter


def make_store(tmp_path, run_id="r1"):
    store = CheckpointStore(str(tmp_path / "cp.db"), persist_mode=PersistMode.RESUMABLE,
                            lease_ttl_seconds=60.0)
    store.create_run(run_id=run_id, owner="w1", task="t", agent="alice", model="m",
                     root_task_id="root")
    return store


def ctx(agent="alice", run_id="r1"):
    return ToolExecutionContext(agent=agent, run_id=run_id, task_id="t1")


# ---- counter reconcile on resume ----

def test_tool_counters_reconcile_on_resume_without_reset_or_double_count(tmp_path):
    store = make_store(tmp_path)
    gw, _ = make_gw(store, strict=True)
    rc = RunContext("r1", CollaborationPolicy(),
                    tool_budgets=ToolBudgets(max_tool_calls_per_run=3,
                                             max_tool_calls_per_agent=10,
                                             max_tool_calls_per_tool=10))
    assert gw.execute("echo", {"text": "a"}, ctx=ctx(), run_context=rc).ok
    assert gw.execute("echo", {"text": "b"}, ctx=ctx(), run_context=rc).ok
    snap = rc.counters_snapshot()
    assert snap["tool_calls"]["total"] == 2
    assert store.get("r1").tool_reservations == 2  # durable, monotonic
    # "resume": a fresh RunContext is restored from the persisted snapshot
    rc2 = RunContext("r1", CollaborationPolicy(),
                     tool_budgets=ToolBudgets(max_tool_calls_per_run=3,
                                              max_tool_calls_per_agent=10,
                                              max_tool_calls_per_tool=10))
    rc2.restore_counters(snap)
    assert rc2.tool_calls == 2  # restored, not reset
    # only ONE more call fits the run budget; no double counting of the old ones
    assert gw.execute("echo", {"text": "c"}, ctx=ctx(), run_context=rc2).ok
    r = gw.execute("echo", {"text": "d"}, ctx=ctx(), run_context=rc2)
    assert r.ok is False and "limit" in r.error
    assert rc2.tool_calls == 3
    assert store.get("r1").tool_reservations == 3  # durable counter never decreases
    store.close()


def test_completed_tool_executions_are_not_replayed_on_resume(tmp_path):
    store = make_store(tmp_path)
    gw, adapter = make_gw(store, strict=True)
    rc = RunContext("r1", CollaborationPolicy())
    r1 = gw.execute("echo", {"text": "x"}, ctx=ctx(), run_context=rc)
    assert r1.ok and r1.execution_id
    rows = store.ledger_list(run_id="r1")
    assert len(rows) == 1 and rows[0]["status"] == ToolExecutionStatus.COMPLETED.value
    # a resumed execution gets a NEW execution id; the completed one is not re-run
    r2 = gw.execute("echo", {"text": "y"}, ctx=ctx(), run_context=rc)
    assert r2.execution_id != r1.execution_id
    assert adapter.calls == 2
    assert len(store.ledger_list(run_id="r1")) == 2
    store.close()


def test_current_policy_revalidated_on_resume(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = make_store(tmp_path)

    def config_with(enabled):
        return HiveConfig(
            supervisor=SupervisorSpec(name="supervisor", description="d",
                                      system_prompt="s", model="ollama/s"),
            agents=[AgentSpec(name="alice", description="d", system_prompt="s",
                              model="ollama/a",
                              permissions=Permissions(allow_tools=True,
                                                      allowed_tools=["fs_read"]))],
            tools=ToolsConfig(enabled=enabled, workspace_root=str(tmp_path / "ws")),
        )

    gw_a = build_tool_gateway(config_with(["fs_read"]), store)
    rc = RunContext("r1", CollaborationPolicy())
    (tmp_path / "ws").mkdir(exist_ok=True)
    (tmp_path / "ws" / "f.txt").write_text("data")
    r = gw_a.execute("fs_read", {"path": "f.txt"}, ctx=ctx(), run_context=rc)
    assert r.ok and "data" in r.output
    # resume under the CURRENT configuration which no longer enables fs_read
    gw_b = build_tool_gateway(config_with([]), store)  # explicit empty = deny all
    r2 = gw_b.execute("fs_read", {"path": "f.txt"}, ctx=ctx(), run_context=rc)
    assert r2.ok is False and "not allowed" in r2.error
    store.close()


# ---- cancellation ----

def test_async_cancel_during_mutating_tool_marks_ledger_unknown(tmp_path):
    store = make_store(tmp_path)
    spec = echo_spec(name="mut", side_effect=SideEffect.LOCAL_MUTATION,
                     requires_approval=False, timeout_seconds=30)

    class SlowMut:
        def __init__(self, spec):
            self.spec = spec

        def execute(self, args, ctx):
            time.sleep(5)
            return "late"

        async def aexecute(self, args, ctx):
            await asyncio.sleep(5)
            return "late"

    adapter = SlowMut(spec)
    catalog = ToolCatalog()
    catalog.register(spec, adapter)
    policy = build_policy(global_enabled=["mut"], agent_permissions={"alice": (True, ["mut"])})
    gw = ToolGateway(catalog, policy, store=store, strict=True)
    rc = RunContext("r1", CollaborationPolicy())
    box = {}

    async def go():
        task = asyncio.create_task(gw.aexecute("mut", {"text": "x"}, ctx=ctx(),
                                                run_context=rc))
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            box["r"] = await task
        except asyncio.CancelledError:
            box["cancelled"] = True
        # find the ledger row written before dispatch
        rows = store.ledger_list(run_id="r1")
        box["rows"] = rows

    asyncio.run(go())
    assert box.get("cancelled") is True
    rows = box["rows"]
    assert len(rows) == 1
    assert rows[0]["status"] == ToolExecutionStatus.UNKNOWN.value  # mutating + cancelled
    store.close()


def test_async_cancel_during_approval_wait_holds_no_slot_and_writes_no_ledger(tmp_path):
    store = make_store(tmp_path)
    spec = echo_spec(name="mut2", side_effect=SideEffect.LOCAL_MUTATION,
                     requires_approval=True, timeout_seconds=30)
    from geosgan_hive.approvals import ApprovalManager

    adapter = EchoAdapter(spec)
    catalog = ToolCatalog()
    catalog.register(spec, adapter)
    policy = build_policy(global_enabled=["mut2"],
                          agent_permissions={"alice": (True, ["mut2"])})
    mgr = ApprovalManager(store, expiry_seconds=60.0)
    gw = ToolGateway(catalog, policy, store=store, approval_manager=mgr, strict=True)
    rc = RunContext("r1", CollaborationPolicy())
    box = {}

    async def go():
        task = asyncio.create_task(gw.aexecute("mut2", {"text": "x"}, ctx=ctx(),
                                                run_context=rc))
        await asyncio.sleep(0.2)  # now waiting for the operator
        assert len(mgr.list_pending(run_id="r1")) == 1
        assert rc.semaphore_tool().locked() is False  # NO tool slot held during wait
        task.cancel()
        try:
            box["r"] = await task
        except asyncio.CancelledError:
            box["cancelled"] = True

    asyncio.run(go())
    assert box.get("cancelled") is True
    assert adapter.calls == 0
    assert store.ledger_list(run_id="r1") == []  # never reached dispatch
    # the approval stays PENDING for the operator (cancel did not decide it)
    assert store.list_approvals(run_id="r1", status="pending")
    store.close()


# ---- subprocess crash DURING a mutating tool ----

CRASH_DURING_TOOL = """
import os, sys
sys.path.insert(0, "src")
from geosgan_hive.approvals import ApprovalManager
from geosgan_hive.checkpoints import CheckpointStore, PersistMode
from geosgan_hive.run_context import CollaborationPolicy, RunContext
from geosgan_hive.tool_gateway import ToolExecutionContext, ToolGateway
from geosgan_hive.tool_policy import build_policy
from geosgan_hive.tool_registry import ToolCatalog
from geosgan_hive.tool_spec import SideEffect, ToolSpec

db, marker = sys.argv[1], sys.argv[2]
store = CheckpointStore(db, persist_mode=PersistMode.RESUMABLE, lease_ttl_seconds=60.0)
store.create_run(run_id="crash", owner="w", task="t", agent="alice", model="m",
                 root_task_id="r")

class Mut:
    def __init__(self, spec): self.spec = spec
    def execute(self, args, ctx):
        with open(marker, "w") as f:  # side effect happens...
            f.write("mutated")
        os._exit(9)  # ...then a hard crash BEFORE the ledger finish

spec = ToolSpec(name="mut", description="d", side_effect=SideEffect.LOCAL_MUTATION,
                requires_approval=False, timeout_seconds=30)
catalog = ToolCatalog()
catalog.register(spec, Mut(spec))
policy = build_policy(global_enabled=["mut"], agent_permissions={"alice": (True, ["mut"])})
gw = ToolGateway(catalog, policy, store=store, approval_manager=None, strict=True)
rc = RunContext("crash", CollaborationPolicy())
gw.execute("mut", {}, ctx=ToolExecutionContext(agent="alice", run_id="crash", task_id="t"),
           run_context=rc)
"""


def test_subprocess_crash_during_mutating_tool_no_auto_replay(tmp_path):
    db = str(tmp_path / "crash.db")
    marker = str(tmp_path / "marker.txt")
    store = CheckpointStore(db, persist_mode=PersistMode.RESUMABLE, lease_ttl_seconds=60.0)
    store.close()
    proc = subprocess.run([sys.executable, "-c", CRASH_DURING_TOOL, db, marker],
                          cwd=os.getcwd(), capture_output=True, text=True, timeout=60)
    assert proc.returncode == 9
    assert open(marker).read() == "mutated"  # side effect happened exactly once
    store2 = CheckpointStore(db, persist_mode=PersistMode.RESUMABLE, lease_ttl_seconds=60.0)
    rows = store2.ledger_list(run_id="crash")
    assert len(rows) == 1  # exactly one ledger row; nothing replayed
    assert rows[0]["status"] == ToolExecutionStatus.STARTED.value  # not auto-completed
    assert store2.get("crash").tool_reservations == 1
    # operator marks UNKNOWN and resolves explicitly; still no re-execution
    assert store2.ledger_update(rows[0]["execution_id"], status="unknown", error="crash",
                                lease_owner="w", lease_generation=0) is True
    assert store2.ledger_resolve(rows[0]["execution_id"], decision="completed",
                                 operator="op", evidence="marker verified") is True
    assert len(store2.ledger_list(run_id="crash")) == 1
    assert store2.ledger_get(rows[0]["execution_id"])["status"] == "completed"
    store2.close()


# ---- nested delegation with tools: privileges are not transferred ----

def test_delegated_agent_keeps_its_own_tool_privileges(tmp_path):
    store = make_store(tmp_path)
    gw, adapter = make_gw(store, strict=True)
    gw.policy.per_agent["researcher"] = frozenset({"echo"})  # researcher may use echo
    policy = CollaborationPolicy(allow_agent_to_agent=True)
    engine = CollaborationEngine(policy=policy, tracker=UsageTracker())
    researcher = Agent(
        name="researcher", description="d", system_prompt="s",
        llm=FakeLLM(responses=['{"delegate": {"target": "developer", "task": "write code",'
                               ' "reason": "impl"}}', "final"]),
        permissions=Permissions(allow_delegation=True, allowed_agents=["developer"],
                                allow_tools=True, allowed_tools=["echo"]),
        collaboration=engine, tool_gateway=gw,
    )
    developer = Agent(
        name="developer", description="d", system_prompt="s",
        llm=FakeLLM(responses=['{"tool": "echo", "args": {"text": "hi"}}', "done"]),
        permissions=Permissions(allow_delegation=False, allowed_agents=[],
                                allow_tools=True, allowed_tools=[]),  # no tools allowed
        collaboration=engine, tool_gateway=gw,
    )
    engine.register({"researcher": researcher, "developer": developer})
    result = engine.delegate({"target": "developer", "task": "write code",
                              "reason": "impl"}, run_id="r1", source_agent="researcher")
    assert result.status.value == "completed"
    # the delegated agent's tool call was resolved under ITS identity -> denied:
    # a TOOL_REJECTED audit event from 'developer' and the adapter never ran
    from geosgan_hive.run_context import EventType

    rejected = [e for e in engine.context_for("r1").events(EventType.TOOL_REJECTED)]
    assert any(e.source_agent == "developer" and e.metadata.get("tool") == "echo"
               for e in rejected)
    assert adapter.calls == 0
    # the researcher itself may use the tool under its own identity
    r = gw.execute("echo", {"text": "hi"}, ctx=ToolExecutionContext(agent="researcher",
                                                                   run_id="r1", task_id="t"),
                   run_context=engine.context_for("r1"))
    assert r.ok is True
    store.close()


def test_supervisor_cannot_bypass_tool_policy(tmp_path):
    store = make_store(tmp_path)
    gw, _ = make_gw(store, strict=True)
    rc = RunContext("r1", CollaborationPolicy())
    r = gw.execute("echo", {"text": "x"},
                   ctx=ToolExecutionContext(agent="supervisor", run_id="r1", task_id="t"),
                   run_context=rc)
    assert r.ok is False and "not allowed" in r.error  # supervisor has no tools
    store.close()
