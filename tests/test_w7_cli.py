"""W7 (v0.8.0): CLI + operator workflow tests.

Cross-process tests drive the REAL CLI as a subprocess (python -m geosgan_hive)
against a temp config/workspace/database: start run -> prepare -> pending list ->
show -> approve -> publish -> status/ledger/source inspection, plus rejection,
expiry, replay, stale-lease, usage errors, exit codes, JSON output, capabilities
report, and secret redaction. In-process tests cover error paths via cli.main().
No live network, no models, zero cost.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
POSIX = os.name == "posix"
posix_only = pytest.mark.skipif(not POSIX, reason="staging/publication are POSIX-only")

CONFIG = """
supervisor:
  name: supervisor
  description: s
  system_prompt: s
  model: ollama/llama3.2
agents:
  - name: analyst
    description: a
    system_prompt: a
    model: ollama/llama3.2
    permissions:
      allow_tools: true
      allowed_tools: [doc_read, doc_generate, doc_publish, source_list, source_get, source_excerpt]
tools:
  enabled: [doc_read, doc_generate, doc_publish, source_list, source_get, source_excerpt]
  workspace_root: {ws}
  allow_write: false
  approval:
    expiry_seconds: {expiry}
checkpoint:
  path: {db}
  persist_mode: metadata_only
  lease_ttl_seconds: 60
documents:
  enabled: {docs}
"""

SPEC = {"title": "Informe ñ", "sheets": [
    {"name": "d", "header": ["k", "v"], "rows": [["ventas", 100], ["f", "=1+1"]]}]}


def make_workspace(tmp_path: Path, *, docs: bool = True, expiry: float = 600.0):
    ws = tmp_path / "ws"
    (ws / "out").mkdir(parents=True)
    db = tmp_path / "cp.db"
    cfg = tmp_path / "agents.yaml"
    cfg.write_text(CONFIG.format(ws=ws, db=db, expiry=expiry, docs=str(docs).lower()))
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(SPEC), encoding="utf-8")
    return cfg, ws, db, spec


def cli(cfg: Path, *args: str, env_extra: dict | None = None):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-m", "geosgan_hive", "--agents-config", str(cfg), *args],
        capture_output=True, text=True, env=env, cwd=cfg.parent, timeout=120)


def cli_json(cfg: Path, *args: str, env_extra: dict | None = None):
    r = cli(cfg, *args, "--json", env_extra=env_extra)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout), r


# ---- full cross-process lifecycle (each step is a SEPARATE CLI process) ----------

@posix_only
def test_cli_cross_process_approval_and_publication(tmp_path):
    cfg, ws, db, spec = make_workspace(tmp_path)
    # 1. start an authorized run (process 1)
    r = cli(cfg, "--start-run", "demo task", "--run-id", "r1", "--operator", "op1")
    assert r.returncode == 0, r.stderr
    # 2. prepare a document (process 2) -> pending exact-byte approval
    prep, _ = cli_json(cfg, "--doc-prepare", "--doc-format", "csv",
                       "--doc-destination", "out/rep.csv", "--spec-file", str(spec),
                       "--run", "r1", "--operator", "op1")
    assert prep["status"] == "pending_approval"
    aid = prep["approval_id"]
    assert not (ws / "out" / "rep.csv").exists()
    # 3. inspect pending in another process (process 3)
    pending, _ = cli_json(cfg, "--pending-approvals")
    assert any(a["approval_id"] == aid for a in pending)
    r = cli(cfg, "--show-approval", aid)  # human-readable: full sha before deciding
    assert r.returncode == 0 and prep["artifact_sha256"] in r.stdout
    assert "byte_count" in r.stdout and "destination" in r.stdout and "expires_at" in r.stdout
    # publish before approval must fail (process 4)
    r = cli(cfg, "--doc-publish", aid, "--run", "r1", "--operator", "op1")
    assert r.returncode == 1 and "not approved" in r.stderr
    assert not (ws / "out" / "rep.csv").exists()
    # 4. human approves (process 5)
    r = cli(cfg, "--approve", aid, "--operator", "op1")
    assert r.returncode == 0, r.stderr
    # 5. publish the exact approved bytes (process 6)
    res, _ = cli_json(cfg, "--doc-publish", aid, "--run", "r1", "--operator", "op1")
    assert res["status"] == "published"
    assert res["artifact_sha256"] == prep["artifact_sha256"]
    data = (ws / "out" / "rep.csv").read_bytes()
    assert hashlib.sha256(data).hexdigest() == prep["artifact_sha256"]
    assert b"'=1+1" in data  # formula stays literal
    # 6. inspect final execution + provenance (processes 7-8)
    status, _ = cli_json(cfg, "--doc-status", aid)
    assert status["lifecycle"] == "published"
    assert status["artifact"]["staged_file_present"] is False  # staging cleaned
    assert status["source_ids"]
    execs, _ = cli_json(cfg, "--tool-executions", "--run", "r1")
    assert any(e["status"] == "completed" and e["approval_id"] == aid for e in execs)
    src, _ = cli_json(cfg, "--source-id", status["source_ids"][0])
    assert src["provider"] == "hive-docgen"
    prov = json.loads(src["provenance_json"])
    assert prov["generated"] is True and prov["output_sha256"] == prep["artifact_sha256"]
    # 7. replay is rejected (process 9)
    r = cli(cfg, "--doc-publish", aid, "--run", "r1", "--operator", "op1")
    assert r.returncode == 1 and "consumed" in r.stderr
    # 8. doc-read of the published artifact via CLI (process 10)
    r = cli(cfg, "--doc-read", "out/rep.csv", "--json")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["format"] == "csv" and "ventas" in out["text"]
    # sources list json
    srcs, _ = cli_json(cfg, "--sources")
    assert any(s["source_id"] == status["source_ids"][0] for s in srcs)


@posix_only
def test_cli_rejection_and_expiry_no_publication(tmp_path):
    cfg, ws, db, spec = make_workspace(tmp_path, expiry=3.0)
    assert cli(cfg, "--start-run", "t", "--run-id", "r1", "--operator", "op1").returncode == 0
    prep, _ = cli_json(cfg, "--doc-prepare", "--doc-format", "csv",
                       "--doc-destination", "out/a.csv", "--spec-file", str(spec),
                       "--run", "r1", "--operator", "op1")
    aid = prep["approval_id"]
    # rejection
    r = cli(cfg, "--deny", aid, "--operator", "op1")
    assert r.returncode == 0, r.stderr
    r = cli(cfg, "--doc-publish", aid, "--run", "r1", "--operator", "op1")
    assert r.returncode == 1 and "denied" in r.stderr
    assert not (ws / "out" / "a.csv").exists()
    # expiry: approve a second artifact, wait past expiry, publish -> rejected
    prep2, _ = cli_json(cfg, "--doc-prepare", "--doc-format", "csv",
                        "--doc-destination", "out/b.csv", "--spec-file", str(spec),
                        "--run", "r1", "--operator", "op1")
    aid2 = prep2["approval_id"]
    assert cli(cfg, "--approve", aid2, "--operator", "op1").returncode == 0
    time.sleep(3.5)
    r = cli(cfg, "--doc-publish", aid2, "--run", "r1", "--operator", "op1")
    assert r.returncode == 1 and ("expired" in r.stderr or "not approved" in r.stderr)
    assert not (ws / "out" / "b.csv").exists()
    # lifecycle shows terminal state
    st, _ = cli_json(cfg, "--doc-status", aid2)
    assert st["status"] in ("expired", "denied") or "expired" in st["lifecycle"]


@posix_only
def test_cli_stale_lease_and_operator_mismatch(tmp_path):
    cfg, ws, db, spec = make_workspace(tmp_path)
    assert cli(cfg, "--start-run", "t", "--run-id", "r1", "--operator", "op1").returncode == 0
    prep, _ = cli_json(cfg, "--doc-prepare", "--doc-format", "csv",
                       "--doc-destination", "out/c.csv", "--spec-file", str(spec),
                       "--run", "r1", "--operator", "op1")
    aid = prep["approval_id"]
    assert cli(cfg, "--approve", aid, "--operator", "op1").returncode == 0
    # a DIFFERENT operator cannot renew/steal the live lease of cli:op1
    r = cli(cfg, "--doc-publish", aid, "--run", "r1", "--operator", "op2")
    assert r.returncode == 1 and "live worker" in r.stderr
    assert not (ws / "out" / "c.csv").exists()
    # wrong run binding
    r = cli(cfg, "--doc-publish", aid, "--run", "rX", "--operator", "op1")
    assert r.returncode == 1


# ---- usage errors, exit codes, capabilities, compat, secrets -----------------------

def test_cli_usage_errors_and_exit_codes(tmp_path):
    cfg, ws, db, spec = make_workspace(tmp_path)
    # missing args -> usage error 2
    assert cli(cfg, "--doc-prepare").returncode == 2
    assert cli(cfg, "--doc-publish", "0" * 32).returncode == 2  # needs --run/--operator
    # unknown run -> 1
    r = cli(cfg, "--doc-prepare", "--doc-format", "csv", "--doc-destination", "out/x.csv",
            "--spec-file", str(spec), "--run", "nope", "--operator", "op1")
    assert r.returncode == 1 and "no run" in r.stderr
    # invalid spec (extra field, extra=forbid) -> 2
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"title": "x", "evil": 1}), encoding="utf-8")
    assert cli(cfg, "--start-run", "t", "--run-id", "r1", "--operator", "op1").returncode == 0
    r = cli(cfg, "--doc-prepare", "--doc-format", "csv", "--doc-destination", "out/x.csv",
            "--spec-file", str(bad), "--run", "r1", "--operator", "op1")
    assert r.returncode == 2 and "invalid specification" in r.stderr
    # unknown approval -> 1
    r = cli(cfg, "--doc-status", "f" * 32)
    assert r.returncode == 1
    # doc-read of a missing file -> 1
    r = cli(cfg, "--doc-read", "nope.txt")
    assert r.returncode == 1 and "Error:" in r.stderr
    # existing commands still work (compat)
    assert cli(cfg, "--list-suspended").returncode == 0
    assert cli(cfg, "--pending-approvals").returncode == 0
    assert cli(cfg, "--tool-executions").returncode == 0
    assert cli(cfg, "--sources").returncode == 0
    r = cli(cfg, "--show", "nonexistent")
    assert r.returncode == 1
    # no task + no management command -> parser error (exit 2)
    assert cli(cfg).returncode == 2


def test_cli_documents_disabled_fail_closed(tmp_path):
    cfg, ws, db, spec = make_workspace(tmp_path, docs=False)
    r = cli(cfg, "--doc-read", "x.txt")
    assert r.returncode == 2 and "disabled" in r.stderr
    cap, _ = cli_json(cfg, "--capabilities")
    assert cap["documents"]["enabled"] is False
    assert cap["zero_cost_mode"]["non_local_cloud_inference"] == "blocked in zero-cost mode"


def test_cli_capabilities_report_and_no_secret_leak(tmp_path):
    cfg, ws, db, spec = make_workspace(tmp_path)
    secret = "sk-test-secret-value-9876543210"
    cap, _ = cli_json(cfg, "--capabilities", env_extra={"ANALYST_API_KEY": secret,
                                                         "HIVE_TEST_SECRET": secret})
    assert cap["documents"]["enabled"] is True
    assert cap["documents"]["optional_dependencies"]["openpyxl"] is True
    assert cap["platform"]["parser_worker_available"] == POSIX
    assert cap["workspace"]["root"] == str(ws)
    analyst = next(a for a in cap["agents"] if a["name"] == "analyst")
    assert "doc_generate" in analyst["effective_tools"]
    assert cap["approvals"]["human_operator_only"] is True
    # no secret values anywhere in the report
    r = cli(cfg, "--capabilities")
    assert secret not in r.stdout and secret not in r.stderr
    r = cli(cfg, "--capabilities", "--json", env_extra={"ANALYST_API_KEY": secret})
    assert secret not in r.stdout


@posix_only
def test_cli_no_clobber_destination_exists(tmp_path):
    cfg, ws, db, spec = make_workspace(tmp_path)
    assert cli(cfg, "--start-run", "t", "--run-id", "r1", "--operator", "op1").returncode == 0
    (ws / "out" / "rep.csv").write_text("intruder", encoding="utf-8")
    r = cli(cfg, "--doc-prepare", "--doc-format", "csv", "--doc-destination", "out/rep.csv",
            "--spec-file", str(spec), "--run", "r1", "--operator", "op1")
    assert r.returncode == 1 and "creation-only" in r.stderr
    assert (ws / "out" / "rep.csv").read_text(encoding="utf-8") == "intruder"


@posix_only
def test_cli_missing_parent_dir_fails_clean_without_burning_approval(tmp_path):
    """W7 residual (failing-first): a destination whose parent directory is missing
    must produce a clean typed error (exit 1, no traceback leak) and must NOT consume
    or burn the approval \u2014 the operator can fix the workspace and publish the SAME
    approved bytes afterwards."""
    import shutil

    cfg, ws, db, spec = make_workspace(tmp_path)
    assert cli(cfg, "--start-run", "t", "--run-id", "r1", "--operator", "op1").returncode == 0
    # prepare into a MISSING parent dir: exit 1, clean Error line, no approval created
    r = cli(cfg, "--doc-prepare", "--doc-format", "csv",
            "--doc-destination", "nodir/rep.csv", "--spec-file", str(spec),
            "--run", "r1", "--operator", "op1")
    assert r.returncode == 1, r.stdout
    assert "Error:" in r.stderr and "Traceback" not in r.stderr
    assert "parent" in r.stderr
    pending, _ = cli_json(cfg, "--pending-approvals")
    assert pending == []
    # prepare OK + approve, then REMOVE the parent between approval and publish:
    # publish must fail clean BEFORE consuming the approval (status stays approved)
    prep, _ = cli_json(cfg, "--doc-prepare", "--doc-format", "csv",
                       "--doc-destination", "out/rep.csv", "--spec-file", str(spec),
                       "--run", "r1", "--operator", "op1")
    aid = prep["approval_id"]
    assert cli(cfg, "--approve", aid, "--operator", "op1").returncode == 0
    shutil.rmtree(ws / "out")
    r = cli(cfg, "--doc-publish", aid, "--run", "r1", "--operator", "op1")
    assert r.returncode == 1, r.stdout
    assert "Error:" in r.stderr and "Traceback" not in r.stderr
    assert "parent" in r.stderr
    st, _ = cli_json(cfg, "--doc-status", aid)
    assert st["status"] == "approved"  # NOT consumed -> not burned into 'uncertain'
    assert "awaiting publication" in st["lifecycle"]
    assert not (ws / "out" / "rep.csv").exists()
    # operator fixes the workspace: publish now succeeds with the SAME approval/bytes
    (ws / "out").mkdir()
    res, _ = cli_json(cfg, "--doc-publish", aid, "--run", "r1", "--operator", "op1")
    assert res["status"] == "published"
    assert res["artifact_sha256"] == prep["artifact_sha256"]
    data = (ws / "out" / "rep.csv").read_bytes()
    assert hashlib.sha256(data).hexdigest() == prep["artifact_sha256"]
