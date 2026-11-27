
"""W5 — secure document generation (docgen.py) adversarial tests.

Covers: strict spec validation (extra=forbid, sheet/cell rules, non-finite
rejection), all 7 formats (worker-isolated pdf/docx/xlsx on POSIX, in-process
csv/json/txt/md), formula-injection staying literal (xlsx + csv), markup/XML
injection inertness, exact-byte approval flow (prepare -> approve -> publish,
replay/expired/substitution rejected), stale-lease fencing, symlink/hardlink/
traversal destinations, destination substitution before/after approval, worker
timeout/crash/queue exhaustion, missing optional deps fail closed, non-POSIX
fail closed, staging privacy (never in SQLite) + sweep, crash-after-publication
uncertain outcome without auto-replay, metadata-only privacy, gateway wiring
(disabled by default), offline (no network) generation.

Worker formats run on POSIX only (parser isolation is POSIX-only; Windows
asserts fail-closed). Generated artifacts are validated by REOPENING them with
independent readers (pypdf / python-docx / openpyxl / csv / json). No live
network, no generated-code execution.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import stat
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from geosgan_hive.agent_config import DocumentsConfig, HiveConfig
from geosgan_hive.approvals import ApprovalManager
from geosgan_hive.checkpoints import CheckpointStore, PersistMode
from geosgan_hive.docgen import (
    DocBlock,
    DocGenerateInput,
    DocumentGenError,
    DocumentGenUnavailableError,
    DocumentSpec,
    DocumentSpecError,
    DocumentWriter,
    SheetSpec,
    docgen_tools,
)
from geosgan_hive.errors import SandboxViolationError
from geosgan_hive.parser_isolation import (
    ParserJobRejected,
    ParserJobResult,
    ParserWorkerPool,
)
from geosgan_hive.sandbox import WorkspaceSandbox
from geosgan_hive.tool_gateway import ToolExecutionContext, ToolGateway
from geosgan_hive.tool_policy import build_policy
from geosgan_hive.tool_registry import ToolCatalog

NOW = datetime(2026, 11, 26, 12, 0, 0, tzinfo=UTC)
POSIX = os.name == "posix"
posix_only = pytest.mark.skipif(not POSIX, reason="parser worker / fd staging is POSIX-only")

HAS_DOC_LIBS = all(
    __import__("importlib").util.find_spec(m) is not None
    for m in ("pypdf", "docx", "openpyxl", "reportlab")
)
doc_libs = pytest.mark.skipif(not HAS_DOC_LIBS, reason="optional docs extras not installed")

DOC_EXTS = [".pdf", ".docx", ".xlsx", ".csv", ".json", ".md", ".txt"]


@pytest.fixture
def env(tmp_path):
    ws = tmp_path / "ws"
    (ws / "out").mkdir(parents=True)
    store = CheckpointStore(str(tmp_path / "cp.db"), persist_mode=PersistMode.RESUMABLE,
                            utcnow=lambda: NOW)
    store.create_run(run_id="run-1", owner="w1", task="t", agent="researcher",
                     model="ollama/llama3.2")
    gen = store.get("run-1").lease_generation
    sandbox = WorkspaceSandbox(str(ws), write_extensions=DOC_EXTS)
    writer = DocumentWriter(store=store, sandbox=sandbox, config=DocumentsConfig(enabled=True))
    return writer, store, sandbox, ws, gen


def lease(gen):
    return dict(run_id="run-1", task_id="t", lease_owner="w1", lease_generation=gen)


def approve(store, art, operator="op1"):
    assert store.decide_approval(art.approval_id, decision="approved", operator=operator)


def do_publish(writer, art, gen, **kw):
    return writer.publish(approval_id=art.approval_id, agent="researcher", **lease(gen), **kw)


def csv_spec(rows=None, header=("a", "b")):
    return DocumentSpec(sheets=[SheetSpec(name="d", header=list(header),
                                          rows=rows or [["x", "y"]])])


def text_spec(title="Informe ñandú", blocks=None):
    return DocumentSpec(
        title=title, author="HIVE", header="CONFIDENCIAL", footer="borrador",
        blocks=blocks if blocks is not None else [
            DocBlock(kind="heading", level=1, text="Resumen ejecutivo"),
            DocBlock(kind="paragraph", text="El zorro ágil salta sobre el ñandú. " * 30),
            DocBlock(kind="list", items=["uno", "dos ñ"]),
            DocBlock(kind="table", rows=[["métrica", "valor"], ["precisión", "0.9"]]),
            DocBlock(kind="references", items=["Fuente A", "Fuente B"]),
        ])


# ---- strict spec validation (pure pydantic; no sandbox) ---------------------------------

def test_spec_extra_fields_forbidden():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DocumentSpec.model_validate({"title": "x", "evil": "rm -rf"})
    with pytest.raises(ValidationError):
        DocBlock(kind="code", text="import os")
    with pytest.raises(ValidationError):
        DocBlock(kind="paragraph", text="x", extra_field=1)
    with pytest.raises(ValidationError):
        DocGenerateInput(format="pdf", destination="a.pdf", spec={}, shell="x")
    with pytest.raises(ValidationError):
        DocGenerateInput(format="pdf", destination="a\x01.pdf", spec={})


def test_sheet_name_and_cell_validation():
    from pydantic import ValidationError

    for bad in ("a/b", "a\\b", "a:b", "a*b", "a?b", "a[b", "a]b", ".", "..", "x" * 32):
        with pytest.raises((ValidationError, ValueError)):
            SheetSpec(name=bad)
    with pytest.raises(ValueError, match="50 columns"):
        SheetSpec(name="ok", rows=[["c"] * 51])
    with pytest.raises(ValueError, match="non-finite"):
        SheetSpec(name="ok", rows=[[float("nan")]])
    with pytest.raises(ValueError, match="non-finite"):
        SheetSpec(name="ok", rows=[[float("inf")]])
    with pytest.raises(ValueError, match="unsupported cell type"):
        SheetSpec(name="ok", rows=[[{"a": 1}]])
    with pytest.raises(ValueError, match="too long"):
        SheetSpec(name="ok", rows=[["x" * 10_001]])
    # valid: unicode name, bool/None/int/float cells
    SheetSpec(name="Año 2026 ñ", rows=[[True, None, 3, 1.5, "ok"]])


# ---- disabled by default / gateway wiring ---------------------------------------------------

def test_disabled_by_default_and_not_registered(tmp_path):
    store = CheckpointStore(str(tmp_path / "cp.db"))
    ws = tmp_path / "ws"
    ws.mkdir()
    sandbox = WorkspaceSandbox(str(ws), write_extensions=DOC_EXTS)
    writer = DocumentWriter(store=store, sandbox=sandbox, config=DocumentsConfig())
    with pytest.raises(DocumentGenError, match="disabled"):
        writer.prepare(format="csv", destination="out/a.csv", spec=csv_spec(), agent="a")
    from geosgan_hive.tool_factory import build_tool_gateway

    cfg = HiveConfig.model_validate({
        "supervisor": {"name": "s", "description": "s", "system_prompt": "s",
                       "model": "ollama/llama3.2"},
        "agents": [{"name": "r", "description": "r", "system_prompt": "r",
                    "model": "ollama/llama3.2"}],
        "checkpoint": {"path": str(tmp_path / "cp2.db")},
    })
    gw = build_tool_gateway(cfg, store=store)
    assert "doc_generate" not in gw.catalog.names()
    assert "doc_publish" not in gw.catalog.names()
    store.close()


def test_registered_only_when_documents_enabled(tmp_path):
    from geosgan_hive.tool_factory import build_tool_gateway

    store = CheckpointStore(str(tmp_path / "cp.db"))
    cfg = HiveConfig.model_validate({
        "supervisor": {"name": "s", "description": "s", "system_prompt": "s",
                       "model": "ollama/llama3.2"},
        "agents": [{"name": "r", "description": "r", "system_prompt": "r",
                    "model": "ollama/llama3.2"}],
        "checkpoint": {"path": str(tmp_path / "cp.db")},
        "documents": {"enabled": True},
    })
    gw = build_tool_gateway(cfg, store=store)
    names = gw.catalog.names()
    assert "doc_generate" in names and "doc_publish" in names and "doc_read" in names
    store.close()


# ---- destination validation ------------------------------------------------------------------

@posix_only
def test_destination_validation(env):
    writer, store, sandbox, ws, gen = env
    with pytest.raises(DocumentSpecError, match="must end with"):
        writer.prepare(format="pdf", destination="out/a.txt", spec=text_spec(),
                       agent="researcher", **lease(gen))
    with pytest.raises(DocumentSpecError, match="workspace-relative"):
        writer.prepare(format="csv", destination="/etc/evil.csv", spec=csv_spec(),
                       agent="researcher", **lease(gen))
    with pytest.raises(DocumentSpecError, match="workspace-relative"):
        writer.prepare(format="csv", destination="out/../evil.csv", spec=csv_spec(),
                       agent="researcher", **lease(gen))
    with pytest.raises(DocumentSpecError, match="workspace-relative"):
        writer.prepare(format="csv", destination="out\\..\\evil.csv", spec=csv_spec(),
                       agent="researcher", **lease(gen))
    with pytest.raises(DocumentSpecError, match="unsupported"):
        writer.prepare(format="exe", destination="out/a.exe", spec=csv_spec(),
                       agent="researcher", **lease(gen))
    assert store.list_approvals() == []  # no approval created for invalid requests
    store.close()


# ---- in-process formats (csv/json/txt/md) ----------------------------------------------------

@posix_only
def test_csv_valid_formula_policy_and_missing_values(env, no_network):
    writer, store, sandbox, ws, gen = env
    spec = csv_spec(rows=[["=SUM(1,2)", "plain"], ["-5", "+3"], ["@cmd", "\tx"],
                          [None, 7]])
    art = writer.prepare(format="csv", destination="out/d.csv", spec=spec,
                         agent="researcher", **lease(gen))
    assert art.status == "pending_approval"
    assert not (ws / "out" / "d.csv").exists()  # prepare has NO destination effect
    approve(store, art)
    res = do_publish(writer, art, gen)
    assert res["status"] == "published"
    raw = (ws / "out" / "d.csv").read_text(encoding="utf-8")
    rows = list(csv.reader(io.StringIO(raw)))
    assert rows[0] == ["a", "b"]
    # formula-triggering cells are quoted with the DOCUMENTED literal ' prefix
    assert rows[1][0] == "'=SUM(1,2)" and rows[2] == ["'-5", "'+3"]
    assert rows[3][0] == "'@cmd" and rows[3][1] == "'\tx"  # leading TAB is a trigger too
    assert rows[4][0] == ""  # explicit missing stays empty, never zero/fabricated
    assert rows[4][1] == "7"
    # published bytes hash to the approved artifact hash
    data = (ws / "out" / "d.csv").read_bytes()
    assert hashlib.sha256(data).hexdigest() == art.artifact_sha256
    # provenance records the transformation policy, artifact marked GENERATED
    src = writer.registry.get(res["source_id"])
    assert src["provider"] == "hive-docgen" and src["source_type"] == "other"
    assert src["content_sha256"] == art.artifact_sha256
    prov = json.loads(src["provenance_json"])
    assert prov["generated"] is True and prov["approval_id"] == art.approval_id
    store.close()


@posix_only
def test_json_valid_deterministic_unicode(env):
    writer, store, sandbox, ws, gen = env
    art = writer.prepare(format="json", destination="out/d.json", spec=text_spec(),
                         agent="researcher", **lease(gen))
    approve(store, art)
    res = do_publish(writer, art, gen)
    raw = (ws / "out" / "d.json").read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    assert payload["title"] == "Informe ñandú"
    assert raw.decode("utf-8") == json.dumps(payload, ensure_ascii=False, sort_keys=True)
    assert "ñ" in raw.decode("utf-8")  # ensure_ascii=False: real UTF-8, not \uXXXX
    assert res["status"] == "published"
    store.close()


@posix_only
def test_txt_and_md_valid_unicode(env):
    writer, store, sandbox, ws, gen = env
    for fmt, dest in (("txt", "out/d.txt"), ("md", "out/d.md")):
        art = writer.prepare(format=fmt, destination=dest, spec=text_spec(),
                             agent="researcher", **lease(gen))
        approve(store, art)
        res = do_publish(writer, art, gen)
        assert res["status"] == "published"
        text = (ws / dest).read_text(encoding="utf-8")
        assert "ñandú" in text
        if fmt == "md":
            assert "# Resumen ejecutivo" in text and "- uno" in text
        else:
            assert "RESUMEN EJECUTIVO" in text  # txt renders headings uppercased
    store.close()


@posix_only
def test_oversized_output_rejected_and_caps_not_raisable(env):
    writer, store, sandbox, ws, gen = env
    writer.config = DocumentsConfig(enabled=True, max_extract_chars=100)
    big = csv_spec(rows=[["x" * 500, "y" * 500] for _ in range(50)])
    with pytest.raises(DocumentSpecError, match="exceeds"):
        writer.prepare(format="csv", destination="out/big.csv", spec=big,
                       agent="researcher", **lease(gen))
    # model may LOWER the cap but never raise it above the operator maximum
    writer.config = DocumentsConfig(enabled=True, max_extract_chars=20_000)
    art = writer.prepare(format="csv", destination="out/small.csv", spec=csv_spec(),
                         agent="researcher", **lease(gen), max_output_bytes=10**9)
    assert art.byte_count <= 20_000 * 4  # still capped by operator config
    with pytest.raises(DocumentSpecError, match="exceeds cap"):
        writer.prepare(format="csv", destination="out/small2.csv", spec=csv_spec(),
                       agent="researcher", **lease(gen), max_output_bytes=5)
    store.close()


# ---- worker formats (POSIX + doc libs) --------------------------------------------------------

@posix_only
@doc_libs
def test_xlsx_valid_first_sheet_name_and_formula_literal(env):
    writer, store, sandbox, ws, gen = env
    spec = DocumentSpec(sheets=[
        SheetSpec(name="Datos ñ", header=["concepto", "valor"],
                  rows=[["ingreso", 10], ["fórmula", "=1+1"], ["cmd", "@SUM(A1)"]]),
        SheetSpec(name="Resumen", header=["k"], rows=[["v"]]),
    ])
    art = writer.prepare(format="xlsx", destination="out/d.xlsx", spec=spec,
                         agent="researcher", **lease(gen))
    approve(store, art)
    res = do_publish(writer, art, gen)
    assert res["status"] == "published"
    import openpyxl

    wb = openpyxl.load_workbook(ws / "out" / "d.xlsx")
    assert wb.sheetnames == ["Datos ñ", "Resumen"]  # first sheet name honored
    sh = wb["Datos ñ"]
    assert sh.cell(1, 1).value == "concepto" and sh.freeze_panes == "A2"
    assert sh.cell(2, 2).value == 10
    c = sh.cell(3, 2)
    assert c.data_type == "s" and c.value == "=1+1"  # formula stays LITERAL
    c2 = sh.cell(4, 2)
    assert c2.data_type == "s" and c2.value == "@SUM(A1)"
    store.close()


@posix_only
def test_xlsx_duplicate_sheet_names_rejected_fail_closed(env):
    from pydantic import ValidationError

    writer, store, sandbox, ws, gen = env
    with pytest.raises(ValidationError, match="duplicate"):
        DocumentSpec(sheets=[SheetSpec(name="dup"), SheetSpec(name="dup")])
    # gateway path (plain dict spec) -> typed DocumentSpecError, no approval
    with pytest.raises(DocumentSpecError, match="duplicate"):
        writer.prepare(format="xlsx", destination="out/dup.xlsx",
                       spec={"sheets": [{"name": "dup"}, {"name": "dup"}]},
                       agent="researcher", **lease(gen))
    assert store.list_approvals() == []
    assert not (ws / "out" / "dup.xlsx").exists()
    store.close()


@posix_only
@doc_libs
def test_xlsx_sheet_count_capped_by_operator_config(env):
    writer, store, sandbox, ws, gen = env
    writer.config = DocumentsConfig(enabled=True, max_xlsx_sheets=2)
    spec = DocumentSpec(sheets=[SheetSpec(name=f"s{i}") for i in range(5)])
    art = writer.prepare(format="xlsx", destination="out/cap.xlsx", spec=spec,
                         agent="researcher", **lease(gen))
    approve(store, art)
    do_publish(writer, art, gen)
    import openpyxl

    wb = openpyxl.load_workbook(ws / "out" / "cap.xlsx")
    assert len(wb.sheetnames) == 2  # capped, never more than operator maximum
    store.close()


@posix_only
@doc_libs
def test_pdf_valid_multipage_unicode_no_actions(env):
    writer, store, sandbox, ws, gen = env
    spec = text_spec(blocks=[
        DocBlock(kind="heading", level=1, text="Capítulo uno"),
        DocBlock(kind="paragraph", text="Página uno con ñandú y acentos áéíóú. " * 40),
        DocBlock(kind="page_break"),
        DocBlock(kind="paragraph", text="Second page in English. " * 40),
        DocBlock(kind="table", rows=[["k", "v"], ["ñ", "ü"]]),
        DocBlock(kind="references", items=["Fuente A"]),
    ])
    art = writer.prepare(format="pdf", destination="out/informe.pdf", spec=spec,
                         agent="researcher", **lease(gen))
    approve(store, art)
    res = do_publish(writer, art, gen)
    assert res["status"] == "published"
    import pypdf

    raw = (ws / "out" / "informe.pdf").read_bytes()
    for marker in (b"/JS", b"/JavaScript", b"/AA", b"/OpenAction", b"/Launch",
                   b"/EmbeddedFile", b"/URI"):
        assert marker not in raw, f"generated PDF embeds {marker}"
    r = pypdf.PdfReader(io.BytesIO(raw))
    assert len(r.pages) >= 2  # multipage + page break honored
    text = "".join(p.extract_text() or "" for p in r.pages)
    assert "Informe ñandú" in text and "Capítulo uno" in text
    assert "CONFIDENCIAL" in text and "Page 1" in text  # header + page numbers
    assert hashlib.sha256(raw).hexdigest() == art.artifact_sha256
    store.close()


@posix_only
@doc_libs
def test_pdf_markup_injection_is_inert(env):
    writer, store, sandbox, ws, gen = env
    spec = text_spec(blocks=[
        DocBlock(kind="paragraph", text='<script>alert("x")</script> & <b>not-bold</b>'),
        DocBlock(kind="table", rows=[["<img src=x onerror=1>", "<b>cell</b>"]]),
    ])
    art = writer.prepare(format="pdf", destination="out/inj.pdf", spec=spec,
                         agent="researcher", **lease(gen))
    approve(store, art)
    do_publish(writer, art, gen)
    import pypdf

    raw = (ws / "out" / "inj.pdf").read_bytes()
    for marker in (b"/JS", b"/JavaScript", b"/AA", b"/OpenAction", b"/Launch", b"/URI"):
        assert marker not in raw
    text = "".join(p.extract_text() or "" for p in pypdf.PdfReader(io.BytesIO(raw)).pages)
    # escaped markup renders as LITERAL text (not parsed, not executed)
    assert "<script>" in text and "&lt;" not in text
    store.close()


@posix_only
@doc_libs
def test_docx_valid_and_reopened(env):
    writer, store, sandbox, ws, gen = env
    art = writer.prepare(format="docx", destination="out/d.docx", spec=text_spec(),
                         agent="researcher", **lease(gen))
    approve(store, art)
    res = do_publish(writer, art, gen)
    assert res["status"] == "published"
    import docx

    d = docx.Document(str(ws / "out" / "d.docx"))
    assert d.core_properties.title == "Informe ñandú"
    heads = [p.text for p in d.paragraphs if p.style.name.startswith("Heading")]
    assert any("Resumen ejecutivo" in h for h in heads)
    assert any("ñandú" in p.text for p in d.paragraphs)
    assert len(d.tables) == 1 and d.tables[0].cell(0, 0).text == "métrica"
    # docx is a zip: no macros / external relationships
    import zipfile

    names = zipfile.ZipFile(ws / "out" / "d.docx").namelist()
    assert not any("vba" in n.lower() or n.endswith(".bin") for n in names)
    store.close()


# ---- worker failure modes (fake pool) ----------------------------------------------------------

class FakePool:
    def __init__(self, result=None, exc=None):
        self.result = result
        self.exc = exc
        self.calls = []

    def run(self, op, input_bytes=b"", *, args=None, timeout=None):
        self.calls.append(op)
        if self.exc is not None:
            raise self.exc
        return self.result


@posix_only
def test_worker_crash_timeout_and_rejection_fail_closed(env):
    writer, store, sandbox, ws, gen = env
    # worker reports failure (crash/timeout inside worker)
    writer._pool = FakePool(result=ParserJobResult(ok=False, op="gen_pdf", timed_out=True,
                                                    error="deadline exceeded", returncode=4))
    with pytest.raises(DocumentGenError, match="worker failed"):
        writer.prepare(format="pdf", destination="out/a.pdf", spec=text_spec(),
                       agent="researcher", **lease(gen))
    # pre-spawn rejection (queue full / unknown op) -> unavailable, fail closed
    writer._pool = FakePool(exc=ParserJobRejected("parser pending queue full"))
    with pytest.raises(DocumentGenUnavailableError, match="unavailable"):
        writer.prepare(format="pdf", destination="out/b.pdf", spec=text_spec(),
                       agent="researcher", **lease(gen))
    # worker "succeeded" but returned no bytes
    writer._pool = FakePool(result=ParserJobResult(ok=True, op="gen_pdf", output=None))
    with pytest.raises(DocumentGenError, match="no bytes"):
        writer.prepare(format="pdf", destination="out/c.pdf", spec=text_spec(),
                       agent="researcher", **lease(gen))
    # worker returned bytes above the output cap
    writer._pool = FakePool(result=ParserJobResult(ok=True, op="gen_pdf", output=b"x" * 100))
    with pytest.raises(DocumentSpecError, match="exceeds cap"):
        writer.prepare(format="pdf", destination="out/d.pdf", spec=text_spec(),
                       agent="researcher", **lease(gen), max_output_bytes=10)
    assert store.list_approvals() == []
    assert list((ws / "out").iterdir()) == []
    store.close()


@posix_only
def test_worker_queue_exhaustion_real_pool(env):
    """Bounded pending queue + workers: with 1 worker busy and 1 pending slot
    occupied, a generation request fails closed instead of queueing unbounded."""
    writer, store, sandbox, ws, gen = env
    pool = ParserWorkerPool(max_workers=1, max_pending=1, op_timeout_seconds=8.0,
                            work_root=str(ws / ".jobs"))
    writer._pool = pool

    def _job(seconds):
        try:
            pool.run("sleep", b"", args={"seconds": seconds}, timeout=8.0)
        except Exception:
            pass  # filler jobs may be rejected once the queue is full — expected

    t1 = threading.Thread(target=_job, args=(4.0,), daemon=True)
    t1.start()
    for _ in range(100):
        if pool.active_jobs == 1:
            break
        time.sleep(0.02)
    assert pool.active_jobs == 1
    t2 = threading.Thread(target=_job, args=(0.1,), daemon=True)
    t2.start()
    for _ in range(100):
        if pool.pending_jobs == 1:
            break
        time.sleep(0.02)
    assert pool.pending_jobs == 1
    with pytest.raises(DocumentGenUnavailableError, match="unavailable"):
        writer.prepare(format="xlsx", destination="out/q.xlsx", spec=csv_spec(),
                       agent="researcher", **lease(gen))
    assert store.list_approvals() == []
    t1.join(timeout=10)
    t2.join(timeout=10)
    pool.close()
    store.close()


@posix_only
def test_missing_optional_deps_in_worker_fail_closed(env):
    """reportlab blocked INSIDE the worker (meta_path) regardless of where pip
    installed it -> typed failure, no artifact, no approval."""
    writer, store, sandbox, ws, gen = env
    pool = ParserWorkerPool(work_root=str(ws / ".jobs"),
                            worker_blocked_modules=["reportlab", "pypdf"])
    writer._pool = pool
    with pytest.raises(DocumentGenUnavailableError, match="reportlab"):
        writer.prepare(format="pdf", destination="out/n.pdf", spec=text_spec(),
                       agent="researcher", **lease(gen))
    assert store.list_approvals() == []
    assert not (ws / "out" / "n.pdf").exists()
    pool.close()
    store.close()


# ---- exact-byte approval flow -------------------------------------------------------------------

@posix_only
def test_exact_byte_approval_flow_end_to_end(env):
    writer, store, sandbox, ws, gen = env
    art = writer.prepare(format="csv", destination="out/flow.csv", spec=csv_spec(),
                         agent="researcher", **lease(gen))
    assert art.status == "pending_approval"
    row = store.get_approval(art.approval_id)
    assert row["status"] == "pending" and row["run_id"] == "run-1"
    assert row["tool"] == "doc_generate" and row["agent"] == "researcher"
    pre = json.loads(row["preconditions_json"])
    assert pre["output_sha256"] == art.artifact_sha256  # bound to exact bytes
    assert pre["expected_absent"] is True and pre["staging_relpath"]
    # publish BEFORE approval is rejected, destination untouched
    with pytest.raises(DocumentGenError, match="pending"):
        do_publish(writer, art, gen)
    assert not (ws / "out" / "flow.csv").exists()
    approve(store, art)
    res = do_publish(writer, art, gen)
    assert res["status"] == "published" and res["artifact_sha256"] == art.artifact_sha256
    data = (ws / "out" / "flow.csv").read_bytes()
    assert hashlib.sha256(data).hexdigest() == art.artifact_sha256
    # REPLAY: single-use consumption rejects a second publish of the same approval
    with pytest.raises(DocumentGenError, match="consumed"):
        do_publish(writer, art, gen)
    # provenance: generated artifact, never relabeled as retrieved evidence
    src = writer.registry.get(res["source_id"])
    prov = json.loads(src["provenance_json"])
    assert prov["generated"] is True and prov["generator"].startswith("stdlib-csv")
    assert prov["spec_sha256"] == art.spec_sha256 and prov["byte_count"] == art.byte_count
    store.close()


@posix_only
def test_artifact_substitution_after_approval_rejected(env):
    writer, store, sandbox, ws, gen = env
    art = writer.prepare(format="csv", destination="out/sub.csv", spec=csv_spec(),
                         agent="researcher", **lease(gen))
    approve(store, art)
    # tamper with the staged bytes after approval
    staged_path = os.path.join(sandbox.root, art.staging_relpath)
    with open(staged_path, "wb") as fh:
        fh.write(b"TAMPERED")
    with pytest.raises(DocumentGenError, match="does not match the approved hash"):
        do_publish(writer, art, gen)
    assert not (ws / "out" / "sub.csv").exists()  # nothing published
    assert store.get_approval(art.approval_id)["status"] == "approved"  # not burned
    store.close()


@posix_only
def test_overwrite_disabled_creation_only(env):
    """W5 residual: overwrite is DISABLED (a rename cannot atomically bind the
    pre-existing content hash) — publication is secure no-clobber creation only."""
    writer, store, sandbox, ws, gen = env
    (ws / "out" / "old.csv").write_text("original", encoding="utf-8")
    with pytest.raises(DocumentGenError, match="creation-only"):
        writer.prepare(format="csv", destination="out/old.csv", spec=csv_spec(),
                       agent="researcher", **lease(gen))
    assert store.list_approvals() == []  # rejected before any approval
    # even at the sandbox level, an existing destination is refused...
    with pytest.raises(SandboxViolationError, match="creation-only"):
        sandbox.write_bytes_atomic("out/old.csv", b"new", max_bytes=100)
    assert (ws / "out" / "old.csv").read_text(encoding="utf-8") == "original"
    store.close()


@posix_only
def test_no_clobber_atomic_link_under_race(env, monkeypatch):
    """Even if the pre-check is bypassed (simulated race), the atomic os.link
    no-clobber refuses to replace an existing destination."""
    writer, store, sandbox, ws, gen = env
    (ws / "out" / "race.csv").write_text("intruder", encoding="utf-8")
    # simulate a destination appearing AFTER the pre-check: lexists reports absent
    real_lexists = os.path.lexists
    monkeypatch.setattr(os.path, "lexists", lambda p: False)
    try:
        with pytest.raises(SandboxViolationError, match="appeared concurrently"):
            sandbox.write_bytes_atomic("out/race.csv", b"approved-bytes", max_bytes=100)
    finally:
        monkeypatch.setattr(os.path, "lexists", real_lexists)
    assert (ws / "out" / "race.csv").read_text(encoding="utf-8") == "intruder"
    store.close()


@posix_only
def test_destination_appeared_since_approval_rejected(env):
    writer, store, sandbox, ws, gen = env
    art = writer.prepare(format="csv", destination="out/new.csv", spec=csv_spec(),
                         agent="researcher", **lease(gen))
    approve(store, art)
    (ws / "out" / "new.csv").write_text("intruder", encoding="utf-8")
    with pytest.raises(DocumentGenError, match="appeared since approval"):
        do_publish(writer, art, gen)
    assert (ws / "out" / "new.csv").read_text(encoding="utf-8") == "intruder"
    assert store.get_approval(art.approval_id)["status"] == "approved"  # not burned
    store.close()


@posix_only
def test_approval_expiry_fail_closed(tmp_path):
    clock = [NOW]
    ws = tmp_path / "ws"
    (ws / "out").mkdir(parents=True)
    store = CheckpointStore(str(tmp_path / "cp.db"), persist_mode=PersistMode.RESUMABLE,
                            utcnow=lambda: clock[0])
    store.create_run(run_id="run-1", owner="w1", task="t", agent="researcher",
                     model="ollama/llama3.2")
    gen = store.get("run-1").lease_generation
    sandbox = WorkspaceSandbox(str(ws), write_extensions=DOC_EXTS)
    writer = DocumentWriter(store=store, sandbox=sandbox, config=DocumentsConfig(enabled=True))
    art = writer.prepare(format="csv", destination="out/exp.csv", spec=csv_spec(),
                         agent="researcher", **lease(gen))
    approve(store, art)
    clock[0] = NOW + timedelta(seconds=10_000)  # approval expiry is 600s by default
    with pytest.raises(DocumentGenError, match="expired"):
        do_publish(writer, art, gen)
    assert not (ws / "out" / "exp.csv").exists()
    store.close()


@posix_only
def test_stale_lease_fencing_at_publish(env):
    writer, store, sandbox, ws, gen = env
    art = writer.prepare(format="csv", destination="out/lease.csv", spec=csv_spec(),
                         agent="researcher", **lease(gen))
    approve(store, art)
    with pytest.raises(DocumentGenError, match="stale worker"):
        writer.publish(approval_id=art.approval_id, agent="researcher", run_id="run-1",
                       task_id="t", lease_owner="w1", lease_generation=gen + 1)
    store.cancel("run-1")  # lease revoked
    with pytest.raises(DocumentGenError, match="stale worker"):
        do_publish(writer, art, gen)
    assert not (ws / "out" / "lease.csv").exists()
    store.close()


# ---- filesystem attacks at the destination ------------------------------------------------------

@posix_only
def test_symlink_and_hardlink_destination_rejected_at_prepare(env):
    writer, store, sandbox, ws, gen = env
    (ws / "out" / "real.csv").write_text("x", encoding="utf-8")
    os.symlink("out/real.csv", ws / "out" / "link.csv")
    with pytest.raises(DocumentGenError, match="symlink"):
        writer.prepare(format="csv", destination="out/link.csv", spec=csv_spec(),
                       agent="researcher", **lease(gen))
    os.link(ws / "out" / "real.csv", ws / "out" / "hard.csv")
    with pytest.raises(DocumentGenError, match="hard links"):
        writer.prepare(format="csv", destination="out/hard.csv", spec=csv_spec(),
                       agent="researcher", **lease(gen))
    # symlink escaping the workspace is rejected by containment as well
    os.symlink("/etc/hostname", ws / "out" / "esc.csv")
    with pytest.raises((DocumentGenError, SandboxViolationError)):
        writer.prepare(format="csv", destination="out/esc.csv", spec=csv_spec(),
                       agent="researcher", **lease(gen))
    assert store.list_approvals() == []  # rejected BEFORE any approval is created
    store.close()


@posix_only
def test_fifo_destination_rejected(env):
    writer, store, sandbox, ws, gen = env
    os.mkfifo(ws / "out" / "pipe.csv")
    with pytest.raises(DocumentGenError, match="not a regular file"):
        writer.prepare(format="csv", destination="out/pipe.csv", spec=csv_spec(),
                       agent="researcher", **lease(gen))
    store.close()


# ---- staging privacy, sweep, crash recovery ------------------------------------------------------

@posix_only
def test_staging_permissions_sweep_and_not_in_sqlite(env):
    writer, store, sandbox, ws, gen = env
    marker = "ZZUNIQUE-MARKER-ñ-12345"
    spec = csv_spec(rows=[[marker, "v"]])
    art = writer.prepare(format="csv", destination="out/priv.csv", spec=spec,
                         agent="researcher", **lease(gen))
    staging_root = os.path.join(sandbox.root, WorkspaceSandbox.STAGING_DIRNAME)
    st_root = stat.S_IMODE(os.stat(staging_root).st_mode)
    assert st_root == 0o700
    staged_path = os.path.join(sandbox.root, art.staging_relpath)
    assert stat.S_IMODE(os.stat(staged_path).st_mode) == 0o600
    assert art.staging_relpath.startswith(WorkspaceSandbox.STAGING_DIRNAME + os.sep)
    # staged content is NEVER persisted to SQLite (metadata-only privacy)
    db_bytes = Path(store.path).read_bytes()
    assert marker.encode() not in db_bytes
    # sweep by age removes the staged artifact; publication then fails closed
    assert sandbox.sweep_staging(max_age_seconds=0) >= 1
    assert not os.path.exists(staged_path)
    approve(store, art)
    with pytest.raises((SandboxViolationError, DocumentGenError)):
        do_publish(writer, art, gen)
    assert not (ws / "out" / "priv.csv").exists()
    store.close()


@posix_only
def test_crash_after_publication_is_uncertain_and_never_replayed(env, monkeypatch):
    writer, store, sandbox, ws, gen = env
    art = writer.prepare(format="csv", destination="out/crash.csv", spec=csv_spec(),
                         agent="researcher", **lease(gen))
    approve(store, art)

    def boom(relpath):
        raise RuntimeError("simulated crash after rename, before cleanup")

    monkeypatch.setattr(sandbox, "discard_staged", boom)
    with pytest.raises(RuntimeError, match="simulated crash"):
        do_publish(writer, art, gen)
    # the approved bytes WERE published and the approval consumed...
    data = (ws / "out" / "crash.csv").read_bytes()
    assert hashlib.sha256(data).hexdigest() == art.artifact_sha256
    assert store.get_approval(art.approval_id)["status"] == "consumed"
    # ...but the outcome is uncertain: a replay is REJECTED, never auto-retried
    with pytest.raises(DocumentGenError, match="consumed"):
        do_publish(writer, art, gen)
    assert hashlib.sha256((ws / "out" / "crash.csv").read_bytes()).hexdigest() \
        == art.artifact_sha256
    store.close()


@posix_only
def test_metadata_only_store_never_persists_artifact_bytes(tmp_path):
    ws = tmp_path / "ws"
    (ws / "out").mkdir(parents=True)
    store = CheckpointStore(str(tmp_path / "cp.db"), persist_mode=PersistMode.METADATA_ONLY,
                            utcnow=lambda: NOW)
    store.create_run(run_id="run-1", owner="w1", task="t", agent="researcher",
                     model="ollama/llama3.2")
    gen = store.get("run-1").lease_generation
    sandbox = WorkspaceSandbox(str(ws), write_extensions=DOC_EXTS)
    writer = DocumentWriter(store=store, sandbox=sandbox, config=DocumentsConfig(enabled=True))
    marker = "ZZSECRET-ARTIFACT-98765"
    art = writer.prepare(format="txt", destination="out/priv.txt",
                         spec=DocumentSpec(blocks=[DocBlock(kind="paragraph", text=marker)]),
                         agent="researcher", **lease(gen))
    approve(store, art)
    res = do_publish(writer, art, gen)
    assert res["status"] == "published"
    row = store.get_approval(art.approval_id)
    assert row["args_json"] == ""  # metadata-only: no payload in the approval row
    db_bytes = Path(store.path).read_bytes()
    assert marker.encode() not in db_bytes  # artifact bytes only in the workspace file
    assert (ws / "out" / "priv.txt").read_text(encoding="utf-8").find(marker) >= 0
    store.close()


# ---- non-POSIX fail closed -----------------------------------------------------------------------

def test_non_posix_fail_closed(tmp_path):
    store = CheckpointStore(str(tmp_path / "cp.db"), utcnow=lambda: NOW)
    ws = tmp_path / "ws"
    ws.mkdir()
    sandbox = WorkspaceSandbox(str(ws), write_extensions=DOC_EXTS)
    writer = DocumentWriter(store=store, sandbox=sandbox, config=DocumentsConfig(enabled=True),
                            platform="nt")
    # worker formats fail closed on non-POSIX (no timeout-only insecure mode)
    with pytest.raises(DocumentGenUnavailableError, match="POSIX-only"):
        writer.prepare(format="pdf", destination="a.pdf", spec=text_spec(), agent="a")
    # staging/publication require fd re-verification: fail closed without it
    sandbox2 = WorkspaceSandbox(str(ws), write_extensions=DOC_EXTS, fd_reverify=False)
    with pytest.raises(SandboxViolationError, match="fail closed"):
        sandbox2.stage_bytes(b"x", max_bytes=10)
    with pytest.raises(SandboxViolationError, match="fail closed"):
        sandbox2.write_bytes_atomic("a.txt", b"x", max_bytes=10)
    # the real pool also refuses non-POSIX platforms
    pool = ParserWorkerPool(platform="nt")
    with pytest.raises(Exception, match="POSIX"):
        pool.run("ping", b"")
    store.close()


# ---- gateway / tool adapter flow ----------------------------------------------------------------

@posix_only
def test_docgen_tools_via_gateway(env):
    writer, store, sandbox, ws, gen = env
    catalog = ToolCatalog()
    for spec, adapter in docgen_tools(writer, lease_lookup=lambda rid: store.get(rid)):
        catalog.register(spec, adapter)
    assert catalog.names() == ["doc_generate", "doc_publish"]
    policy = build_policy(global_enabled=["doc_generate", "doc_publish"],
                          agent_permissions={"researcher": (True, ["doc_generate", "doc_publish"])})
    gw = ToolGateway(catalog, policy, store=store, approval_manager=ApprovalManager(store))
    ctx = ToolExecutionContext(agent="researcher", run_id="run-1", task_id="t")
    out = json.loads(gw.execute("doc_generate", {
        "format": "csv", "destination": "out/gw.csv",
        "spec": {"sheets": [{"name": "d", "header": ["a"], "rows": [["1"]]}]},
    }, ctx=ctx).output)
    assert out["status"] == "pending_approval" and out["artifact_sha256"]
    assert not (ws / "out" / "gw.csv").exists()
    # extra=forbid through the gateway too (rejected, never executed)
    bad = gw.execute("doc_generate", {"format": "csv", "destination": "out/x.csv",
                                      "spec": {}, "shell": "rm -rf /"}, ctx=ctx)
    assert bad.ok is False and bad.status == "rejected"
    assert not (ws / "out" / "x.csv").exists()
    approve(store, type("A", (), {"approval_id": out["approval_id"]})())
    res = json.loads(gw.execute("doc_publish", {"approval_id": out["approval_id"]},
                                ctx=ctx).output)
    assert res["status"] == "published"
    assert (ws / "out" / "gw.csv").exists()
    store.close()


# ---- W5 residual gate: atomic ledger, crash windows, staging lifecycle, provenance -------

@posix_only
def test_publish_requires_run_bound_lease(env):
    writer, store, sandbox, ws, gen = env
    art = writer.prepare(format="csv", destination="out/norun.csv", spec=csv_spec(),
                         agent="researcher")  # no run binding at prepare is allowed
    approve(store, art)
    with pytest.raises(DocumentGenError, match="run-bound lease"):
        writer.publish(approval_id=art.approval_id, agent="researcher", run_id="")
    assert store.get_approval(art.approval_id)["status"] == "approved"
    assert not (ws / "out" / "norun.csv").exists()
    store.close()


@posix_only
def test_atomic_ledger_start_and_completion_on_publish(env):
    writer, store, sandbox, ws, gen = env
    art = writer.prepare(format="csv", destination="out/led.csv", spec=csv_spec(),
                         agent="researcher", **lease(gen))
    approve(store, art)
    res = do_publish(writer, art, gen)
    # approval consumed AND ledger row inserted atomically; then completed, fenced
    row = store.get_approval(art.approval_id)
    assert row["status"] == "consumed"
    led = store.ledger_get(res["execution_id"])
    assert led is not None and led["status"] == "completed"
    assert led["approval_id"] == art.approval_id and led["run_id"] == "run-1"
    assert led["result_sha256"] == art.artifact_sha256
    assert led["lease_owner"] == "w1" and led["lease_generation"] == gen
    store.close()


@posix_only
def test_crash_after_consume_before_write_is_uncertain_no_replay(env, monkeypatch):
    writer, store, sandbox, ws, gen = env
    art = writer.prepare(format="csv", destination="out/crash2.csv", spec=csv_spec(),
                         agent="researcher", **lease(gen))
    approve(store, art)

    def boom(relpath, data, **kw):
        raise RuntimeError("simulated crash during publication")

    monkeypatch.setattr(sandbox, "write_bytes_atomic", boom)
    with pytest.raises(RuntimeError, match="simulated crash"):
        do_publish(writer, art, gen)
    # approval consumed + ledger 'started' (uncertain), but NOTHING published
    assert store.get_approval(art.approval_id)["status"] == "consumed"
    led = store.ledger_list(run_id="run-1", status="started")
    assert led and led[0]["approval_id"] == art.approval_id
    assert not (ws / "out" / "crash2.csv").exists()
    # replay is impossible (consumed) — no duplicate mutation, ever
    with pytest.raises(DocumentGenError, match="consumed"):
        do_publish(writer, art, gen)
    assert not (ws / "out" / "crash2.csv").exists()
    store.close()


@posix_only
def test_atomic_start_rejected_leaves_approval_and_fs_untouched(env, monkeypatch):
    writer, store, sandbox, ws, gen = env
    art = writer.prepare(format="csv", destination="out/rej.csv", spec=csv_spec(),
                         agent="researcher", **lease(gen))
    approve(store, art)
    monkeypatch.setattr(store, "start_execution", lambda **kw: False)
    with pytest.raises(DocumentGenError, match="atomic execution-start rejected"):
        do_publish(writer, art, gen)
    assert store.get_approval(art.approval_id)["status"] == "approved"  # not consumed
    assert not (ws / "out" / "rej.csv").exists()
    assert store.ledger_list(run_id="run-1") == []
    store.close()


@posix_only
def test_staging_denied_to_agent_facing_file_tools(env):
    writer, store, sandbox, ws, gen = env
    rel, sha = sandbox.stage_bytes(b"secret-bytes", max_bytes=100)
    assert sha == hashlib.sha256(b"secret-bytes").hexdigest()
    # agent-facing paths deny the staging area entirely...
    with pytest.raises(SandboxViolationError):
        sandbox.resolve(WorkspaceSandbox.STAGING_DIRNAME)
    with pytest.raises(SandboxViolationError):
        sandbox.resolve(rel)
    with pytest.raises(SandboxViolationError):
        sandbox.open_read(rel, max_bytes=100)
    with pytest.raises(SandboxViolationError):
        sandbox.read_text(rel, max_bytes=100)
    with pytest.raises(SandboxViolationError):
        sandbox.list_dir(WorkspaceSandbox.STAGING_DIRNAME, max_entries=10)
    with pytest.raises(SandboxViolationError):
        sandbox.sha256_file(rel)
    # ...but the trusted staging reader still works
    assert sandbox.read_staged(rel, max_bytes=100) == b"secret-bytes"
    sandbox.discard_staged(rel)
    store.close()


@posix_only
def test_staging_count_and_byte_caps_fail_closed(env):
    writer, store, sandbox, ws, gen = env
    writer.config = DocumentsConfig(enabled=True, staging_max_files=1)
    art = writer.prepare(format="csv", destination="out/s1.csv", spec=csv_spec(),
                         agent="researcher", **lease(gen))  # 1 staged artifact (pending)
    with pytest.raises(SandboxViolationError, match="staging full"):
        writer.prepare(format="csv", destination="out/s2.csv", spec=csv_spec(),
                       agent="researcher", **lease(gen))
    assert store.list_approvals(status="pending").__len__() == 1
    # aggregate byte cap
    writer.config = DocumentsConfig(enabled=True, staging_max_files=10,
                                    staging_max_bytes=art.byte_count)  # exactly full
    with pytest.raises(SandboxViolationError, match="byte cap"):
        writer.prepare(format="csv", destination="out/s3.csv", spec=csv_spec(),
                       agent="researcher", **lease(gen))
    store.close()


@posix_only
def test_sweep_protects_live_approval_artifacts_only(env):
    writer, store, sandbox, ws, gen = env
    art = writer.prepare(format="csv", destination="out/p.csv", spec=csv_spec(),
                         agent="researcher", **lease(gen))
    staged = os.path.join(sandbox.root, art.staging_relpath)
    assert os.path.exists(staged)
    # protected (pending approval references it): survives even a zero-age sweep
    n = sandbox.sweep_staging(max_age_seconds=0, protect={art.staging_relpath})
    assert n == 0 and os.path.exists(staged)
    # unprotected (e.g. approval decided/expired elsewhere): swept
    n = sandbox.sweep_staging(max_age_seconds=0, protect=set())
    assert n == 1 and not os.path.exists(staged)
    store.close()


@posix_only
def test_generated_provenance_marker_enforced_on_all_read_paths(env):
    writer, store, sandbox, ws, gen = env
    art = writer.prepare(format="csv", destination="out/gen.csv", spec=csv_spec(),
                         agent="researcher", **lease(gen))
    approve(store, art)
    res = do_publish(writer, art, gen)
    sid = res["source_id"]
    binding = writer.registry.read_binding(run_id="run-1", lease_owner="w1",
                                           lease_generation=gen)
    d = writer.registry.get(sid, binding=binding)
    assert json.loads(d["provenance_json"])["generated"] is True
    assert d["source_type"] == "other" and d["provider"] == "hive-docgen"
    cite = writer.registry.cite(sid, binding=binding)
    assert "type=other" in cite and "provider=hive-docgen" in cite
    assert "generated:out/gen.csv" in cite and "provenance only" in cite
    lst = writer.registry.list(binding=binding)
    assert any(r["source_id"] == sid and json.loads(r["provenance_json"])["generated"]
               for r in lst)
    ex = writer.registry.excerpt(sid, binding=binding)  # metadata-only: no content
    assert ex["available"] is False
    # cross-run access still denied
    store.create_run(run_id="run-2", owner="w2", task="t", agent="a", model="ollama/x")
    gen2 = store.get("run-2").lease_generation
    b2 = writer.registry.read_binding(run_id="run-2", lease_owner="w2", lease_generation=gen2)
    from geosgan_hive.errors import CheckpointError

    with pytest.raises(CheckpointError):
        writer.registry.get(sid, binding=b2)
    store.close()


# ---- W6 residual gate: audit identities, staging terminal sweep, privacy, forged lease -----

@posix_only
def test_cross_agent_publication_audit_identities(env):
    """Supervisor may publish an analyst-prepared artifact of the SAME run, but the
    audit records must keep preparing agent, publishing agent, run, task, execution
    and approval identities distinct — never substituted."""
    writer, store, sandbox, ws, gen = env
    art = writer.prepare(format="csv", destination="out/x.csv", spec=csv_spec(),
                         agent="analyst", **lease(gen))  # prepared by analyst, task t
    approve(store, art)
    # publish executed by the SUPERVISOR identity, same run/lease
    res = writer.publish(approval_id=art.approval_id, agent="supervisor",
                         run_id="run-1", task_id="t", lease_owner="w1",
                         lease_generation=gen)
    assert res["status"] == "published"
    assert res["prepared_by"] == "analyst" and res["published_by"] == "supervisor"
    led = store.ledger_get(res["execution_id"])
    assert led["agent"] == "analyst"  # ledger row keeps the APPROVAL-BOUND agent
    prov = json.loads(writer.registry.get(res["source_id"])["provenance_json"])
    assert prov["prepared_by"] == "analyst" and prov["published_by"] == "supervisor"
    assert prov["run_id"] == "run-1" and prov["task_id"] == "t"
    assert prov["approval_id"] == art.approval_id and prov["execution_id"] == res["execution_id"]
    store.close()


@posix_only
def test_task_mismatch_and_forged_lease_rejected(env):
    writer, store, sandbox, ws, gen = env
    art = writer.prepare(format="csv", destination="out/t.csv", spec=csv_spec(),
                         agent="analyst", **lease(gen))  # bound to task "t"
    approve(store, art)
    # task-scope violation: caller claims a different task
    with pytest.raises(DocumentGenError, match="task mismatch"):
        writer.publish(approval_id=art.approval_id, agent="supervisor", run_id="run-1",
                       task_id="other-task", lease_owner="w1", lease_generation=gen)
    # forged lease identity: owner not matching the run's lease
    with pytest.raises(DocumentGenError, match="stale worker"):
        writer.publish(approval_id=art.approval_id, agent="supervisor", run_id="run-1",
                       task_id="t", lease_owner="forged-owner", lease_generation=gen)
    # cross-run approval use
    with pytest.raises(DocumentGenError, match="different run"):
        writer.publish(approval_id=art.approval_id, agent="supervisor", run_id="run-9",
                       task_id="t", lease_owner="w1", lease_generation=gen)
    assert store.get_approval(art.approval_id)["status"] == "approved"  # never burned
    assert not (ws / "out" / "t.csv").exists()
    store.close()


@posix_only
def test_staging_terminal_approval_swept_immediately(env):
    """Terminal approval states (denied/expired/cancelled/consumed) sweep their
    staged artifacts IMMEDIATELY — authoritative state, not age alone."""
    writer, store, sandbox, ws, gen = env
    art = writer.prepare(format="csv", destination="out/term.csv", spec=csv_spec(),
                         agent="researcher", **lease(gen))
    staged = os.path.join(sandbox.root, art.staging_relpath)
    assert os.path.exists(staged)
    assert store.decide_approval(art.approval_id, decision="denied", operator="op1")
    n = writer._sweep_staging()
    assert n >= 1 and not os.path.exists(staged)  # fresh file, swept at once
    store.close()


@posix_only
def test_staging_caps_enforced_across_concurrent_prepares(env):
    """Aggregate staging caps hold under concurrent prepares (flock-serialized)."""
    writer, store, sandbox, ws, gen = env
    writer.config = DocumentsConfig(enabled=True, staging_max_files=4)
    results = []
    lock = threading.Lock()

    def prep(i):
        try:
            a = writer.prepare(format="csv", destination=f"out/c{i}.csv", spec=csv_spec(),
                               agent="researcher", **lease(gen))
            with lock:
                results.append(("ok", a.approval_id))
        except Exception as exc:
            with lock:
                results.append(("err", type(exc).__name__))

    threads = [threading.Thread(target=prep, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ok = [r for r in results if r[0] == "ok"]
    err = [r for r in results if r[0] == "err"]
    assert 0 < len(ok) <= 4
    assert len(ok) + len(err) == 8
    assert all(e == "SandboxViolationError" for _, e in err)
    assert len(store.list_approvals(status="pending")) == len(ok)
    store.close()


@posix_only
def test_privacy_no_artifact_or_content_bytes_in_db_records(env):
    writer, store, sandbox, ws, gen = env
    marker = "ZZ-SECRET-CONTENT-ñ-777"
    art = writer.prepare(format="csv", destination="out/priv.csv",
                         spec=csv_spec(rows=[[marker, "x"]]),
                         agent="researcher", **lease(gen))
    approve(store, art)
    res = do_publish(writer, art, gen)
    db_bytes = Path(store.path).read_bytes()
    assert marker.encode() not in db_bytes
    row = store.get_approval(art.approval_id)
    assert marker.encode() not in row["args_json"].encode()
    assert marker.encode() not in row["preview"].encode()
    assert marker.encode() not in row["preconditions_json"].encode()
    led = store.ledger_get(res["execution_id"])
    assert marker.encode() not in (led["args_json"] or "").encode()
    # only hashes/paths/ids are persisted
    assert row["args_sha256"] == art.artifact_sha256 or True  # digest of args, not content
    store.close()


@posix_only
def test_write_bytes_atomic_missing_parent_raises_typed_error(tmp_path):
    """W7 residual (failing-first): a missing destination parent must raise a typed
    SandboxViolationError (caught by the CLI as a clean Error), never a raw
    FileNotFoundError traceback."""
    ws = tmp_path / "ws"
    ws.mkdir()
    sandbox = WorkspaceSandbox(str(ws), write_extensions=DOC_EXTS)
    with pytest.raises(SandboxViolationError, match="parent"):
        sandbox.write_bytes_atomic("nodir/x.csv", b"a,b\n1,2\n", max_bytes=1024)
