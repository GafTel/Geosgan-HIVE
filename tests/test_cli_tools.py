"""Phase 3 T7: CLI management commands for tool approvals and the execution ledger
(CLI-only trusted operator interface; model/agent deciders are rejected)."""

import textwrap

from geosgan_hive.approvals import ApprovalManager
from geosgan_hive.checkpoints import (
    ApprovalState,
    CheckpointStore,
    PersistMode,
    ToolExecutionStatus,
)
from geosgan_hive.cli import main
from geosgan_hive.tool_gateway import ToolExecutionContext
from geosgan_hive.tool_spec import SideEffect, ToolSpec

CONFIG = """
supervisor:
  name: supervisor
  description: d
  system_prompt: s
  model: ollama/s
agents:
  - name: alice
    description: d
    system_prompt: s
    model: ollama/a
    permissions: {allow_tools: true, allowed_tools: [fake_write]}
tools:
  enabled: [fake_write]
  allow_write: true
  workspace_root: {ws}
checkpoint:
  path: {db}
  persist_mode: resumable
  lease_ttl_seconds: 60
"""


def make_env(tmp_path):
    db = tmp_path / "cp.db"
    ws = tmp_path / "ws"
    ws.mkdir()
    cfg = tmp_path / "agents.yaml"
    cfg.write_text(CONFIG.replace("{db}", str(db)).replace("{ws}", str(ws)))
    store = CheckpointStore(str(db), persist_mode=PersistMode.RESUMABLE,
                            lease_ttl_seconds=60.0)
    store.create_run(run_id="r1", owner="w1", task="t", agent="alice", model="m",
                     root_task_id="root")
    return store, str(cfg), str(db)


def make_approval(store, path="a.txt"):
    mgr = ApprovalManager(store, expiry_seconds=600.0, preview_bytes=2000)
    spec = ToolSpec(name="fake_write", description="d", category="test",
                    side_effect=SideEffect.LOCAL_MUTATION, requires_approval=True)
    req = mgr.create(ctx=ToolExecutionContext(agent="alice", run_id="r1", task_id="t1"),
                     spec=spec, args={"path": path, "content": "hello"},
                     operation="create_file", target=path,
                     preconditions={"target_exists": False})
    return mgr, req


def test_cli_pending_approvals_shows_bounded_preview(tmp_path, capsys):
    store, cfg, db = make_env(tmp_path)
    _mgr, req = make_approval(store)
    rc = main(["--pending-approvals", "--agents-config", cfg, "--checkpoint", db])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Pending approvals: 1" in out
    assert req.approval_id in out and "fake_write" in out and "create_file" in out
    assert "a.txt" in out  # meaningful preview, not just a hash
    store.close()


def test_cli_approve_and_deny_with_operator(tmp_path, capsys):
    store, cfg, db = make_env(tmp_path)
    mgr, req = make_approval(store)
    rc = main(["--approve", req.approval_id, "--operator", "operator-1",
               "--agents-config", cfg, "--checkpoint", db])
    assert rc == 0
    assert store.get_approval(req.approval_id)["status"] == ApprovalState.APPROVED.value
    assert "APPROVED" in capsys.readouterr().out
    # already decided -> error
    rc = main(["--deny", req.approval_id, "--operator", "operator-1",
               "--agents-config", cfg, "--checkpoint", db])
    assert rc == 1
    store.close()


def test_cli_model_or_agent_operator_is_rejected(tmp_path, capsys):
    store, cfg, db = make_env(tmp_path)
    _mgr, req = make_approval(store)
    for bad in ("model", "agent", "AGENT:alice"):
        rc = main(["--approve", req.approval_id, "--operator", bad,
                   "--agents-config", cfg, "--checkpoint", db])
        assert rc == 1
    assert store.get_approval(req.approval_id)["status"] == ApprovalState.PENDING.value
    # missing operator -> usage error
    rc = main(["--approve", req.approval_id, "--agents-config", cfg, "--checkpoint", db])
    assert rc == 2
    store.close()


def test_cli_show_approval_and_cancel(tmp_path, capsys):
    store, cfg, db = make_env(tmp_path)
    mgr, req = make_approval(store)
    rc = main(["--show-approval", req.approval_id, "--agents-config", cfg,
               "--checkpoint", db])
    assert rc == 0
    out = capsys.readouterr().out
    assert req.approval_id in out and "args_sha256" in out and "preview" in out
    rc = main(["--cancel-approval", req.approval_id, "--operator", "op",
               "--agents-config", cfg, "--checkpoint", db])
    assert rc == 0
    assert store.get_approval(req.approval_id)["status"] == ApprovalState.CANCELLED.value
    store.close()


def test_cli_tool_executions_and_resolve_unknown(tmp_path, capsys):
    store, cfg, db = make_env(tmp_path)
    store.ledger_start(execution_id="e1", run_id="r1", agent="alice", tool="fake_write",
                       args_sha256="h", status="started", lease_owner="w1",
                       lease_generation=0)
    rc = main(["--tool-executions", "--run", "r1", "--agents-config", cfg,
               "--checkpoint", db])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Tool executions: 1" in out and "e1" in out and "started" in out
    # not UNKNOWN -> resolution refused
    rc = main(["--resolve-execution", "e1", "--decision", "completed",
               "--operator", "op", "--evidence", "x", "--agents-config", cfg,
               "--checkpoint", db])
    assert rc == 1
    store.ledger_update("e1", status="unknown", error="crash", lease_owner="w1",
                        lease_generation=0)
    rc = main(["--resolve-execution", "e1", "--decision", "completed",
               "--operator", "op", "--evidence", "verified on disk",
               "--agents-config", cfg, "--checkpoint", db])
    assert rc == 0
    row = store.ledger_get("e1")
    assert row["status"] == ToolExecutionStatus.COMPLETED.value
    assert "verified on disk" in row["error"]
    store.close()


def test_cli_management_requires_checkpoint_store(tmp_path, capsys):
    cfg = tmp_path / "no_cp.yaml"
    cfg.write_text(textwrap.dedent("""
        supervisor: {name: supervisor, description: d, system_prompt: s, model: ollama/s}
        agents:
          - {name: alice, description: d, system_prompt: s, model: ollama/a}
        """))
    rc = main(["--pending-approvals", "--agents-config", str(cfg)])
    assert rc == 2
    assert "no checkpoint store configured" in capsys.readouterr().err
