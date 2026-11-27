"""W7 (v0.8.0) offline demo: operator CLI workflow for secure document generation.

Drives the REAL `python -m geosgan-hive` CLI as separate subprocesses against a
temporary workspace + checkpoint DB (simulated operator session; no models, no
network, zero cost):

  start run -> prepare (pending exact-byte approval) -> list/show pending ->
  approve -> publish exact bytes -> status/ledger/source inspection,
  plus a REJECTED artifact and a replay attempt (both refused).

All document writes go through the ToolGateway-side writer, the exact-byte
human approval, the durable execution ledger and the sandboxed no-clobber
publication. Temporary workspace/DB only.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"

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
    permissions: {allow_tools: true, allowed_tools: [doc_read, doc_generate, doc_publish]}
tools:
  enabled: [doc_read, doc_generate, doc_publish, source_list, source_get, source_excerpt]
  workspace_root: {ws}
  allow_write: false
checkpoint:
  path: {db}
  persist_mode: metadata_only
  lease_ttl_seconds: 60
documents:
  enabled: true
"""

SPEC = {"title": "Informe ZZDEMO-ñ", "sheets": [
    {"name": "datos", "header": ["concepto", "importe"],
     "rows": [["ventas", 100], ["iva", "=B2*0.21"], ["nota", None]]}]}


def cli(cfg: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-m", "geosgan_hive", "--agents-config",
                           str(cfg), *args], capture_output=True, text=True, env=env,
                          cwd=cfg.parent, timeout=120)


def show(label: str, r: subprocess.CompletedProcess, lines: int = 6) -> None:
    print(f"\n$ geosgan-hive {' '.join(a for a in r.args[4:] if not a.startswith('--'))}"
          f"  # {label} -> exit={r.returncode}")
    out = (r.stdout or r.stderr).strip().splitlines()
    for line in out[:lines]:
        print("  " + line)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="hive-w7-demo-"))
    ws = tmp / "ws"
    (ws / "out").mkdir(parents=True)
    cfg = tmp / "agents.yaml"
    cfg.write_text(CONFIG.replace("{ws}", str(ws)).replace("{db}", str(tmp / "cp.db")),
                    encoding="utf-8")
    spec_file = tmp / "spec.json"
    spec_file.write_text(json.dumps(SPEC), encoding="utf-8")
    print(f"temp workspace: {tmp}  (simulated operator session; offline; zero cost)")

    r = cli(cfg, "--capabilities", "--json")
    cap = json.loads(r.stdout)
    print(f"capabilities: docs enabled={cap['documents']['enabled']} "
          f"deps={cap['documents']['optional_dependencies']} "
          f"parser_worker={cap['platform']['parser_worker_available']} "
          f"zero-cost local-only={cap['zero_cost_mode']['release_gate_local_providers']}")

    show("start an authorized run", cli(cfg, "--start-run", "informe de ventas",
                                         "--run-id", "demo", "--operator", "alice"))
    r = cli(cfg, "--doc-prepare", "--doc-format", "csv", "--doc-destination", "out/reporte.csv",
            "--spec-file", str(spec_file), "--run", "demo", "--operator", "alice", "--json")
    show("prepare document (structured spec, validated)", r, lines=3)
    prep = json.loads(r.stdout)
    aid = prep["approval_id"]
    print(f"  -> pending approval {aid}")
    print(f"     artifact_sha256={prep['artifact_sha256']}")
    print(f"     byte_count={prep['byte_count']} destination={prep['destination']}")

    show("list pending approvals (another process)",
         cli(cfg, "--pending-approvals"), lines=3)
    show("inspect approval (full exact-byte binding shown before deciding)",
         cli(cfg, "--show-approval", aid), lines=22)

    # rejected artifact: prepare + deny + publish refused
    r = cli(cfg, "--doc-prepare", "--doc-format", "csv", "--doc-destination", "out/otro.csv",
            "--spec-file", str(spec_file), "--run", "demo", "--operator", "alice", "--json")
    aid2 = json.loads(r.stdout)["approval_id"]
    show("human DENIES the second artifact", cli(cfg, "--deny", aid2, "--operator", "alice"))
    r = cli(cfg, "--doc-publish", aid2, "--run", "demo", "--operator", "alice")
    show("publish of denied artifact refused", r, lines=2)
    assert not (ws / "out" / "otro.csv").exists()

    show("publish BEFORE approval refused",
         cli(cfg, "--doc-publish", aid, "--run", "demo", "--operator", "alice"), lines=2)
    show("human APPROVES the exact artifact", cli(cfg, "--approve", aid, "--operator", "alice"))
    r = cli(cfg, "--doc-publish", aid, "--run", "demo", "--operator", "alice", "--json")
    show("publish the exact approved bytes", r, lines=3)
    res = json.loads(r.stdout)
    assert res["status"] == "published"
    data = (ws / "out" / "reporte.csv").read_bytes()
    assert hashlib.sha256(data).hexdigest() == prep["artifact_sha256"]
    print("  -> published file hash matches the approved artifact exactly; content:")
    print("     " + data.decode("utf-8").replace("\n", "\n     ").rstrip())

    show("replay of consumed approval refused",
         cli(cfg, "--doc-publish", aid, "--run", "demo", "--operator", "alice"), lines=2)

    r = cli(cfg, "--doc-status", aid, "--json")
    st = json.loads(r.stdout)
    show("final lifecycle + execution + provenance", r, lines=3)
    print(f"  lifecycle={st['lifecycle']} source={st['source_ids']} "
          f"staged_present={st['artifact']['staged_file_present']}")
    r = cli(cfg, "--tool-executions", "--run", "demo", "--json")
    print(f"  ledger: {[(e['tool'], e['status']) for e in json.loads(r.stdout)]}")
    r = cli(cfg, "--source-id", st["source_ids"][0])
    print("  source provenance (generated artifact, never factual evidence):")
    for line in r.stdout.splitlines()[-6:]:
        print("   " + line)
    r = cli(cfg, "--doc-read", "out/reporte.csv", "--json")
    out = json.loads(r.stdout)
    print(f"  doc-read of the published artifact: format={out['format']} "
          f"raw_sha256={out['raw_sha256'][:16]}… truncated={out['truncated']}")

    db = (tmp / "cp.db").read_bytes()
    assert b"ZZDEMO" not in db and b"ventas,100" not in db  # artifact content not persisted
    print("\nmetadata-only OK: no document content persisted in the checkpoint DB")
    print("Demo finished. Fully offline; real CLI subprocesses; exact-byte approval; "
          "no-clobber publication; zero cost.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
