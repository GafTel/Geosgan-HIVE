"""Offline demo of the Phase 3 TOOL ENGINE (no API keys needed).

Shows, fully offline with a scripted fake LLM:

1. Every tool call goes through the single ToolGateway (policy + strict
   validation + budgets); the legacy registry path is not used by agents.
2. A read-only file tool (fs_read) inside the sandboxed workspace.
3. A mutating tool (fs_write) requires a durable human approval: the gateway
   parks the call (no semaphore held), the CLI-style operator preview is
   printed, the operator approves via ApprovalManager, the approval is
   consumed atomically bound to the exact argument hash, the file is written
   atomically, and the execution ledger records COMPLETED.
4. A second write that the operator DENIES never touches the filesystem.
5. The durable ledger + approvals are listed at the end (what the CLI
   --tool-executions / --pending-approvals commands show).

Run from the repo root:  python examples/demo_tool_engine.py
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path

from geosgan_hive.agent_config import (
    AgentSpec,
    HiveConfig,
    Permissions,
    SupervisorSpec,
    ToolsConfig,
)
from geosgan_hive.agents import Agent
from geosgan_hive.checkpoints import CheckpointStore, PersistMode
from geosgan_hive.factory import build_tool_gateway
from geosgan_hive.llm import FakeLLM
from geosgan_hive.run_context import CollaborationPolicy, RunContext
from geosgan_hive.tool_gateway import ToolExecutionContext


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="hive-tools-demo-"))
    ws = tmp / "workspace"
    ws.mkdir()
    (ws / "notes.md").write_text("# Plan\n- step 1\n- step 2\n", encoding="utf-8")
    db = tmp / "checkpoints.db"

    config = HiveConfig(
        supervisor=SupervisorSpec(name="supervisor", description="d",
                                  system_prompt="s", model="ollama/s"),
        agents=[AgentSpec(
            name="researcher", description="d", system_prompt="s", model="ollama/r",
            permissions=Permissions(allow_tools=True,
                                    allowed_tools=["fs_read", "fs_write", "calculator"]),
        )],
        tools=ToolsConfig(
            enabled=["fs_read", "fs_write", "calculator"],
            workspace_root=str(ws),
            allow_write=True,
        ),
    )

    store = CheckpointStore(str(db), persist_mode=PersistMode.RESUMABLE,
                            lease_ttl_seconds=60.0)
    store.create_run(run_id="demo-run", owner="demo-worker", task="demo",
                     agent="researcher", model="ollama/r", root_task_id="root")
    gateway = build_tool_gateway(config, store)  # ApprovalManager auto-wired
    rc = RunContext("demo-run", CollaborationPolicy())
    mgr = gateway.approval_manager
    assert mgr is not None

    # 1) an agent executes a tool through the gateway (prompt-JSON transport)
    agent = Agent(
        "researcher", "d", "You are a researcher.",
        FakeLLM(responses=[
            json.dumps({"tool": "fs_read", "args": {"path": "notes.md"}}),
            json.dumps({"tool": "calculator", "args": {"expression": "6*7"}}),
            "done",
        ]),
        permissions=Permissions(allow_tools=True,
                                allowed_tools=["fs_read", "fs_write", "calculator"]),
        tool_gateway=gateway,
    )
    result = agent.run("read notes and compute", run_id="demo-run", run_context=rc)
    print("== agent tool calls (all via ToolGateway) ==")
    for call in result.tool_calls:
        print(f"  {call.tool}: {call.result[:80]!r}")

    # 2) a mutating call parks until the human operator decides
    def call_write():
        return gateway.execute(
            "fs_write", {"path": "summary.txt", "content": "approved summary"},
            ctx=ToolExecutionContext(agent="researcher", run_id="demo-run",
                                     task_id="t-write"),
            run_context=rc,
        )

    box: dict = {}
    t = threading.Thread(target=lambda: box.update(r=call_write()))
    t.start()
    deadline = time.time() + 5
    pending = []
    while time.time() < deadline and not pending:
        pending = mgr.list_pending(run_id="demo-run")
        time.sleep(0.02)
    print("\n== pending approval (CLI preview) ==")
    ap = pending[0]
    print(mgr.preview(ap["approval_id"]))
    mgr.decide(ap["approval_id"], approve=True, operator="operator-alice")
    t.join(timeout=10)
    r = box["r"]
    print(f"\n== approved write executed: ok={r.ok} status={r.status} "
          f"execution_id={r.execution_id[:12]}... ==")
    print(f"   file content: {(ws / 'summary.txt').read_text()!r}")

    # 3) a denied write never happens
    box2: dict = {}
    t2 = threading.Thread(
        target=lambda: box2.update(r=gateway.execute(
            "fs_write", {"path": "evil.txt", "content": "nope"},
            ctx=ToolExecutionContext(agent="researcher", run_id="demo-run",
                                     task_id="t-evil"),
            run_context=rc)))
    t2.start()
    deadline = time.time() + 5
    pending = []
    while time.time() < deadline and not pending:
        pending = [p for p in mgr.list_pending(run_id="demo-run")
                   if p["target"] == "evil.txt"]
        time.sleep(0.02)
    mgr.decide(pending[0]["approval_id"], approve=False, operator="operator-alice")
    t2.join(timeout=10)
    print(f"\n== denied write: ok={box2['r'].ok} error={box2['r'].error}")
    print(f"   evil.txt exists: {(ws / 'evil.txt').exists()}")

    # 4) durable ledger + approvals (what the CLI management commands show)
    print("\n== durable execution ledger ==")
    for row in store.ledger_list(run_id="demo-run"):
        ap = row["approval_id"][:8] if row["approval_id"] else "-"
        print(f"  {row['execution_id'][:12]}... tool={row['tool']} "
              f"status={row['status']} approval={ap}")
    print("== approvals ==")
    for row in store.list_approvals(run_id="demo-run"):
        print(f"  {row['approval_id'][:8]}... {row['operation']} {row['target']} "
              f"-> {row['status']} (by {row['decided_by']})")
    print(f"\n== durable tool reservations on the run: "
          f"{store.get('demo-run').tool_reservations} ==")
    store.close()
    print(f"\nDemo workspace: {ws}")


if __name__ == "__main__":
    main()
