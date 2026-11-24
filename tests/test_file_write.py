"""Phase 3 T5: approved file writes \u2014 fs_write/fs_update through the sandbox with
atomic replace, extension allowlist, approval-bound content-hash preconditions
(fs_update), and full gateway approval+ledger integration."""

import hashlib
import json
import os
import threading
import time

import pytest

from geosgan_hive.agent_config import (
    AgentSpec,
    HiveConfig,
    Permissions,
    SupervisorSpec,
    ToolsConfig,
    apply_env_overrides,
)
from geosgan_hive.checkpoints import (
    ApprovalState,
    CheckpointStore,
    PersistMode,
    ToolExecutionStatus,
)
from geosgan_hive.errors import SandboxViolationError
from geosgan_hive.run_context import CollaborationPolicy, EventType, RunContext
from geosgan_hive.sandbox import WorkspaceSandbox
from geosgan_hive.tool_factory import build_tool_catalog, build_tool_gateway
from geosgan_hive.tool_gateway import ToolExecutionContext

posix_writes = pytest.mark.skipif(
    os.name != "posix", reason="file writes are disabled by default on Windows"
)


def sha(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def make_sandbox(tmp_path, **kwargs) -> WorkspaceSandbox:
    return WorkspaceSandbox(str(tmp_path / "ws"), **kwargs)


# ---- sandbox write primitives ----

@posix_writes
def test_write_creates_file_atomically_and_cleans_tmp(tmp_path):
    sb = make_sandbox(tmp_path)
    (tmp_path / "ws" / "notes").mkdir()
    out = sb.write_text("notes/a.txt", "hello", require_exists=False)
    assert out == sha("hello")
    assert (tmp_path / "ws" / "notes" / "a.txt").read_text() == "hello"
    leftovers = [p for p in (tmp_path / "ws" / "notes").iterdir() if ".tmp-" in p.name]
    assert leftovers == []  # tmp file replaced, not left behind


@posix_writes
def test_write_refuses_existing_file_and_update_requires_existing(tmp_path):
    sb = make_sandbox(tmp_path)
    sb.write_text("a.txt", "v1")
    with pytest.raises(SandboxViolationError, match="already exists"):
        sb.write_text("a.txt", "v2", require_exists=False)
    with pytest.raises(SandboxViolationError, match="does not exist"):
        sb.write_text("b.txt", "x", expected_sha256=sha("x"), require_exists=True)
    assert (tmp_path / "ws" / "a.txt").read_text() == "v1"


@posix_writes
def test_update_binds_to_expected_sha256(tmp_path):
    sb = make_sandbox(tmp_path)
    sb.write_text("a.txt", "v1")
    sb.write_text("a.txt", "v2", expected_sha256=sha("v1"), require_exists=True)
    assert (tmp_path / "ws" / "a.txt").read_text() == "v2"
    # file changed after approval -> hash mismatch -> rejected, file untouched
    (tmp_path / "ws" / "a.txt").write_text("tampered")
    with pytest.raises(SandboxViolationError, match="changed after approval"):
        sb.write_text("a.txt", "v3", expected_sha256=sha("v1"), require_exists=True)
    assert (tmp_path / "ws" / "a.txt").read_text() == "tampered"


@pytest.mark.skipif(os.name != "posix", reason="symlink creation is POSIX-privileged")
def test_write_rejects_traversal_outside_denied_names_extensions_symlinks(tmp_path):
    sb = make_sandbox(tmp_path)
    (tmp_path / "outside.txt").write_text("secret")
    with pytest.raises(SandboxViolationError):
        sb.write_text("../outside.txt", "x")
    with pytest.raises(SandboxViolationError, match="denied"):
        sb.write_text(".env", "KEY=1")
    with pytest.raises(SandboxViolationError, match="denied"):
        sb.write_text("sub/x.db", "x")
    with pytest.raises(SandboxViolationError, match="allowlist"):
        sb.write_text("evil.py", "import os")
    # symlink final component -> refuse to write through it
    (tmp_path / "ws" / "real.txt").write_text("real")
    os.symlink(str(tmp_path / "ws" / "real.txt"), str(tmp_path / "ws" / "link.txt"))
    with pytest.raises(SandboxViolationError, match="symlink"):
        sb.write_text("link.txt", "through")
    assert (tmp_path / "ws" / "real.txt").read_text() == "real"


@posix_writes
def test_write_size_cap_and_extension_allowlist_configurable(tmp_path):
    sb = make_sandbox(tmp_path, write_extensions=[".txt"])
    with pytest.raises(SandboxViolationError, match="allowlist"):
        sb.write_text("a.md", "x")
    sb.write_text("a.txt", "x")
    big = make_sandbox(tmp_path / "ws2")
    with pytest.raises(SandboxViolationError, match="write limit"):
        big.write_text("b.txt", "x" * (1024 * 1024 + 1))


# ---- registration gating ----

def _config(tools: ToolsConfig | None) -> HiveConfig:
    return HiveConfig(
        supervisor=SupervisorSpec(name="supervisor", description="d", system_prompt="s",
                                  model="ollama/s"),
        agents=[AgentSpec(name="alice", description="d", system_prompt="s", model="ollama/a",
                          permissions=Permissions(allow_tools=True,
                                                  allowed_tools=["fs_write", "fs_update",
                                                                 "fs_read"]))],
        tools=tools,
    )


def test_write_tools_not_registered_unless_allow_write(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    catalog, _ = build_tool_catalog(_config(ToolsConfig(enabled=["fs_write", "fs_update",
                                                                "fs_read"],
                                                        allow_write=False,
                                                        workspace_root=str(tmp_path / "ws"))))
    assert "fs_write" not in catalog.names() and "fs_update" not in catalog.names()
    assert "fs_read" in catalog.names()
    catalog2, sb2 = build_tool_catalog(_config(ToolsConfig(enabled=["fs_write", "fs_update"],
                                                            allow_write=True,
                                                            workspace_root=str(tmp_path / "ws"))))
    if os.name == "posix":
        assert "fs_write" in catalog2.names() and "fs_update" in catalog2.names()
    else:  # Windows: writes are disabled by default, so they stay unregistered
        assert sb2.platform == "nt"
        assert "fs_write" not in catalog2.names() and "fs_update" not in catalog2.names()
    if os.name == "posix":
        spec = catalog2.get("fs_update").spec
        assert spec.side_effect.value == "local_mutation" and spec.approval_required is True


def test_legacy_mode_denies_all_non_builtin_tools(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    gw = build_tool_gateway(_config(None))
    assert gw.policy.allows("alice", "fs_read") is False  # legacy: built-ins only
    assert gw.policy.allows("alice", "fs_write") is False
    assert gw.catalog.get("fs_write") is None  # mutating tools not even registered
    from geosgan_hive.tool_factory import BUILTIN_TOOL_NAMES

    assert set(gw.policy.global_enabled) == set(BUILTIN_TOOL_NAMES)  # built-ins only


def test_write_extensions_env_override_and_validation():
    data = {"agents": [], "supervisor": {"name": "s"}}
    out = apply_env_overrides(data, {"HIVE_TOOLS_WRITE_EXTENSIONS": ".txt, .MD ,.json"})
    assert out["tools"]["write_extensions"] == [".txt", ".MD", ".json"]
    with pytest.raises(ValueError):
        ToolsConfig(write_extensions=["txt"])  # must start with a dot
    assert ToolsConfig(write_extensions=[".MD"]).write_extensions == [".md"]


# ---- gateway end-to-end with approvals + ledger ----

def make_env(tmp_path):
    store = CheckpointStore(str(tmp_path / "cp.db"), persist_mode=PersistMode.RESUMABLE,
                            lease_ttl_seconds=60.0)
    store.create_run(run_id="r1", owner="w1", task="t", agent="alice", model="m",
                     root_task_id="root")
    config = _config(ToolsConfig(enabled=["fs_write", "fs_update"], allow_write=True,
                                 workspace_root=str(tmp_path / "ws")))
    gw = build_tool_gateway(config, store)
    assert gw.approval_manager is not None  # auto-wired from store + tools block
    rc = RunContext("r1", CollaborationPolicy())
    return store, gw, rc


def ctx(agent="alice", run_id="r1"):
    return ToolExecutionContext(agent=agent, run_id=run_id, task_id="t1")


def _wait_pending(gw, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        pending = gw.approval_manager.list_pending(run_id="r1")
        if pending:
            return pending[0]
        time.sleep(0.02)
    raise AssertionError("no pending approval appeared")


@posix_writes
def test_fs_write_full_approval_flow_writes_file(tmp_path):
    store, gw, rc = make_env(tmp_path)
    box = {}

    def go():
        box["r"] = gw.execute("fs_write", {"path": "out/notes.md", "content": "# Hi"},
                              ctx=ctx(), run_context=rc)

    os.makedirs(tmp_path / "ws" / "out")
    t = threading.Thread(target=go)
    t.start()
    ap = _wait_pending(gw)
    assert ap["operation"] == "create_file" and ap["target"] == "out/notes.md"
    assert "create_file" in ap["preview"] and "out/notes.md" in ap["preview"]
    gw.approval_manager.decide(ap["approval_id"], approve=True, operator="op-1")
    t.join(timeout=10)
    r = box["r"]
    assert r.ok is True, r.error
    payload = json.loads(r.output)
    assert payload["created"] is True and payload["sha256"] == sha("# Hi")
    assert (tmp_path / "ws" / "out" / "notes.md").read_text() == "# Hi"
    assert store.get_approval(ap["approval_id"])["status"] == ApprovalState.CONSUMED.value
    row = store.ledger_get(r.execution_id)
    assert row["status"] == ToolExecutionStatus.COMPLETED.value
    assert row["approval_id"] == ap["approval_id"]
    assert store.get("r1").tool_reservations == 1
    kinds = [e.event for e in rc.events()]
    assert EventType.APPROVAL_REQUESTED in kinds and EventType.TOOL_EXECUTED in kinds
    store.close()


@posix_writes
def test_fs_write_denied_means_no_file(tmp_path):
    store, gw, rc = make_env(tmp_path)
    box = {}
    t = threading.Thread(target=lambda: box.update(
        r=gw.execute("fs_write", {"path": "x.txt", "content": "x"}, ctx=ctx(),
                     run_context=rc)))
    t.start()
    ap = _wait_pending(gw)
    gw.approval_manager.decide(ap["approval_id"], approve=False, operator="op-1")
    t.join(timeout=10)
    assert box["r"].ok is False and "denied" in box["r"].error
    assert not (tmp_path / "ws" / "x.txt").exists()
    assert store.ledger_list(run_id="r1") == []
    store.close()


@posix_writes
def test_fs_update_rejected_when_file_changed_after_approval(tmp_path):
    store, gw, rc = make_env(tmp_path)
    target = tmp_path / "ws" / "doc.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("v1")
    args = {"path": "doc.txt", "content": "v2", "expected_sha256": sha("v1")}
    box = {}
    t = threading.Thread(target=lambda: box.update(
        r=gw.execute("fs_update", args, ctx=ctx(), run_context=rc)))
    t.start()
    ap = _wait_pending(gw)
    # operator approves, but the file changes BEFORE execution -> precondition fails
    target.write_text("tampered-by-someone-else")
    gw.approval_manager.decide(ap["approval_id"], approve=True, operator="op-1")
    t.join(timeout=10)
    r = box["r"]
    assert r.ok is False and "changed after approval" in r.error
    assert target.read_text() == "tampered-by-someone-else"  # tool did NOT write
    row = store.ledger_get(r.execution_id)
    assert row["status"] == ToolExecutionStatus.FAILED.value  # side effect never happened
    store.close()


@posix_writes
def test_fs_update_happy_path_with_hash_precondition(tmp_path):
    store, gw, rc = make_env(tmp_path)
    target = tmp_path / "ws" / "doc.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("v1")
    args = {"path": "doc.txt", "content": "v2", "expected_sha256": sha("v1")}
    box = {}
    t = threading.Thread(target=lambda: box.update(
        r=gw.execute("fs_update", args, ctx=ctx(), run_context=rc)))
    t.start()
    ap = _wait_pending(gw)
    assert ap["preconditions_json"].find(sha("v1")) != -1
    gw.approval_manager.decide(ap["approval_id"], approve=True, operator="op-1")
    t.join(timeout=10)
    assert box["r"].ok is True, box["r"].error
    assert target.read_text() == "v2"
    store.close()


@posix_writes
def test_fs_write_strict_arg_validation_rejects_extra_args(tmp_path):
    store, gw, rc = make_env(tmp_path)
    r = gw.execute("fs_write", {"path": "a.txt", "content": "x", "mode": "777"},
                   ctx=ctx(), run_context=rc)
    assert r.ok is False and "invalid arguments" in r.error
    r = gw.execute("fs_update", {"path": "a.txt", "content": "x"},  # missing hash
                   ctx=ctx(), run_context=rc)
    assert r.ok is False and "invalid arguments" in r.error
    store.close()


@posix_writes
def test_policy_requires_approval_for_mutating_file_tools(tmp_path):
    store, gw, rc = make_env(tmp_path)
    spec = gw.catalog.get("fs_write").spec
    assert gw.policy.requires_approval(spec, "alice", "fs_write") is True
    # supervisor has no tools configured -> denied even though globally enabled
    r = gw.execute("fs_write", {"path": "a.txt", "content": "x"},
                   ctx=ToolExecutionContext(agent="supervisor", run_id="r1", task_id="t"),
                   run_context=rc)
    assert r.ok is False and "not allowed" in r.error
    store.close()
