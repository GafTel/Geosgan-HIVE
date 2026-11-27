"""W4 (v0.8.0): local document reading — adversarial offline tests.

Fixtures are generated locally with the approved optional libraries (pypdf /
python-docx / openpyxl / defusedxml / reportlab-as-fixture-writer). Worker-format
parse tests run on POSIX (parser isolation is POSIX-only; Windows asserts
fail-closed). No live network, no malicious code execution — hostile ARCHIVES
are crafted as bytes and only ever reach the guarded parser worker.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from geosgan_hive.agent_config import DocumentsConfig
from geosgan_hive.approvals import ApprovalManager
from geosgan_hive.checkpoints import CheckpointStore, PersistMode
from geosgan_hive.documents import (
    DocumentError,
    DocumentParseError,
    DocumentReader,
    DocumentUnavailableError,
    ZipCaps,
    ZipGuardError,
    check_zip,
    detect_format,
    doc_tools,
    parse_csv_doc,
    parse_json_doc,
    zip_keyword_prescan,
)
from geosgan_hive.errors import CheckpointError, HiveError
from geosgan_hive.parser_isolation import ParserJobRejected, ParserJobResult
from geosgan_hive.sandbox import WorkspaceSandbox
from geosgan_hive.tool_gateway import ToolExecutionContext, ToolGateway
from geosgan_hive.tool_policy import build_policy
from geosgan_hive.tool_registry import ToolCatalog

NOW = datetime(2026, 11, 26, 12, 0, 0, tzinfo=UTC)
POSIX = os.name == "posix"
posix_only = pytest.mark.skipif(not POSIX, reason="parser worker is POSIX-only")

HAS_DOC_LIBS = all(
    __import__("importlib").util.find_spec(m) is not None
    for m in ("pypdf", "docx", "openpyxl", "defusedxml")
)
doc_libs = pytest.mark.skipif(not HAS_DOC_LIBS, reason="optional docs extras not installed")


# ---- fixture builders ----------------------------------------------------------------

def make_pdf(path: Path, pages: list[str], *, encrypt: str | None = None) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    for text in pages:
        c.drawString(100, 750, text)
        c.showPage()
    c.save()
    data = buf.getvalue()
    if encrypt:
        import pypdf

        r = pypdf.PdfReader(io.BytesIO(data))
        w = pypdf.PdfWriter()
        for p in r.pages:
            w.add_page(p)
        w.encrypt(encrypt)
        out = io.BytesIO()
        w.write(out)
        data = out.getvalue()
    path.write_bytes(data)


def make_docx(path: Path, *, extra_xml: bytes | None = None, evil_rels: bool = False) -> None:
    import docx

    d = docx.Document()
    d.add_heading("Título en español", level=1)
    d.add_paragraph("Párrafo con ñ, acentos y — guiones.")
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text = "A1"
    t.cell(0, 1).text = "B1"
    t.cell(1, 0).text = "A2"
    t.cell(1, 1).text = "B2"
    d.save(str(path))
    if extra_xml is not None or evil_rels:
        # rewrite the zip with a modified/evil entry (crafted bytes, never executed)
        src = zipfile.ZipFile(path)
        items = {i.filename: src.read(i.filename) for i in src.infolist()}
        src.close()
        if extra_xml is not None:
            items["word/document.xml"] = extra_xml
        if evil_rels:
            items["_rels/.rels"] = items["_rels/.rels"] + (
                b'<Relationship Id="R9" Type="x" Target="http://evil.example.org/x" '
                b'TargetMode="External"/>')
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, content in items.items():
                zf.writestr(name, content)


def make_xlsx(path: Path, *, formula: bool = True) -> None:
    import openpyxl

    wb = openpyxl.Workbook()
    s = wb.active
    s.title = "Datos"
    s["A1"] = "nombre"
    s["B1"] = "valor"
    s["A2"] = "alpha"
    s["B2"] = 42
    if formula:
        s["C2"] = "=B2*2"
    wb.save(str(path))


def make_zip_bomb(path: Path, expanded_mb: int = 200) -> None:
    """ZIP whose declared/actual expanded size vastly exceeds any cap."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("xl/workbook.xml", b"<workbook/>")
        zf.writestr("xl/worksheets/sheet1.xml", b"x" * (expanded_mb * 1024 * 1024))


def make_evil_zip(path: Path, name: str, *, flag_encrypted: bool = False,
                  method: int = zipfile.ZIP_DEFLATED) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zi = zipfile.ZipInfo("xl/workbook.xml")
        zi.compress_type = method
        zf.writestr(zi, b"<workbook/>")
        zf.writestr(name, b"evil")
    if flag_encrypted:
        # zipfile cannot WRITE encrypted entries — patch the flag bits directly
        # in the local header (offset 6) and central directory (offset 8)
        data = bytearray(path.read_bytes())
        i = data.find(b"PK\x03\x04")
        data[i + 6:i + 8] = (1).to_bytes(2, "little")
        j = data.find(b"PK\x01\x02")
        data[j + 8:j + 10] = (1).to_bytes(2, "little")
        path.write_bytes(bytes(data))


@pytest.fixture
def env(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    store = CheckpointStore(str(tmp_path / "cp.db"), persist_mode=PersistMode.RESUMABLE,
                            utcnow=lambda: NOW)
    store.create_run(run_id="run-1", owner="w1", task="t", agent="researcher",
                     model="ollama/llama3.2")
    gen = store.get("run-1").lease_generation
    sandbox = WorkspaceSandbox(str(ws))
    cfg = DocumentsConfig(enabled=True)
    reader = DocumentReader(store=store, sandbox=sandbox, config=cfg)
    return reader, store, sandbox, ws, gen


def lease(gen):
    return dict(run_id="run-1", lease_owner="w1", lease_generation=gen)


# ---- disabled by default / policy ------------------------------------------------------

def test_disabled_by_default(tmp_path):
    store = CheckpointStore(str(tmp_path / "cp.db"))
    sandbox = WorkspaceSandbox(str(tmp_path / "ws"))
    reader = DocumentReader(store=store, sandbox=sandbox, config=DocumentsConfig())
    (tmp_path / "ws" / "a.txt").write_text("x", encoding="utf-8")
    with pytest.raises(DocumentError, match="disabled"):
        reader.read("a.txt")
    # not wired into the tool gateway unless enabled
    from geosgan_hive.agent_config import HiveConfig
    from geosgan_hive.tool_factory import build_tool_gateway

    cfg = HiveConfig.model_validate({
        "supervisor": {"name": "s", "description": "s", "system_prompt": "s",
                       "model": "ollama/llama3.2"},
        "agents": [{"name": "r", "description": "r", "system_prompt": "r",
                    "model": "ollama/llama3.2"}],
        "checkpoint": {"path": str(tmp_path / "cp2.db")},
    })
    gw = build_tool_gateway(cfg, store=store)
    assert "doc_read" not in gw.catalog.names()
    store.close()


@posix_only
def test_allowed_formats_policy(env):
    reader, store, sandbox, ws, gen = env
    reader.config = DocumentsConfig(enabled=True, allowed_formats=["csv"])
    (ws / "a.pdf").write_bytes(b"%PDF-1.4 fake")
    with pytest.raises(DocumentError, match="not allowed"):
        reader.read("a.pdf", **lease(gen))
    store.close()


# ---- valid documents (worker formats on POSIX) -------------------------------------------

@posix_only
@doc_libs
def test_valid_pdf_with_provenance_and_source(env):
    reader, store, sandbox, ws, gen = env
    make_pdf(ws / "doc.pdf", ["Hola mundo — página uno", "Second page English"])
    r = reader.read("doc.pdf", **lease(gen))
    assert r.format == "pdf" and "pypdf" in r.parser
    assert "Hola mundo" in r.text and "Second page" in r.text
    assert r.truncated is False and r.source_id is not None
    d = reader.registry.get(r.source_id)
    assert d["source_type"] == "document" and d["provider"] == "local-document"
    prov = json.loads(d["provenance_json"])
    assert prov["format"] == "pdf" and prov["page_count"] == 2 and prov["truncated"] is False
    assert "pypdf" in prov["parser"]
    # metadata-only: page content never persisted
    db = open(str(store.path), "rb").read()
    assert b"Hola mundo" not in db
    store.close()


@posix_only
@doc_libs
def test_pdf_page_range_scope(env):
    reader, store, sandbox, ws, gen = env
    make_pdf(ws / "doc.pdf", ["one", "two", "three"])
    r = reader.read("doc.pdf", scope={"pages": "2"}, **lease(gen))
    assert "[page 2]" in r.text and "one" not in r.text
    with pytest.raises(DocumentParseError, match="out of range"):
        reader.read("doc.pdf", scope={"pages": "99"}, **lease(gen))
    store.close()


@posix_only
@doc_libs
def test_valid_docx_spanish_unicode_headings_table(env):
    reader, store, sandbox, ws, gen = env
    make_docx(ws / "doc.docx")
    r = reader.read("doc.docx", **lease(gen))
    assert r.format == "docx" and "python-docx" in r.parser
    assert "Título en español" in r.text and "ñ" in r.text
    kinds = {b["type"] for b in r.structure}
    assert "heading" in kinds and "table" in kinds
    store.close()


@posix_only
@doc_libs
def test_valid_xlsx_formula_not_evaluated_missing_cached(env):
    reader, store, sandbox, ws, gen = env
    make_xlsx(ws / "hoja.xlsx", formula=True)
    r = reader.read("hoja.xlsx", **lease(gen))
    assert r.format == "xlsx" and "openpyxl" in r.parser
    assert r.extra.get("formula_cells") == 1
    assert r.extra.get("missing_cached_values") == 1  # never invented a result
    assert "=B2*2" in r.text and "84" not in r.text
    store.close()


@posix_only
@doc_libs
def test_xlsx_missing_values_preserved(env):
    reader, store, sandbox, ws, gen = env
    import openpyxl

    wb = openpyxl.Workbook()
    wb.active["A1"] = "x"
    wb.active["B1"] = None  # missing
    wb.save(str(ws / "m.xlsx"))
    r = reader.read("m.xlsx", **lease(gen))
    assert '"value_missing": true' in r.text  # missing stays missing, never zero
    store.close()


# ---- in-process simple formats -------------------------------------------------------------

@posix_only
def test_csv_json_txt_md_html(env):
    reader, store, sandbox, ws, gen = env
    (ws / "d.csv").write_text("a,b\n1,x\n2,\n", encoding="utf-8")
    (ws / "d.json").write_text('{"clave": [1, {"x": "ñ"}]}', encoding="utf-8")
    (ws / "n.txt").write_text("plain text", encoding="utf-8")
    (ws / "n.md").write_text("# Título\ntexto", encoding="utf-8")
    (ws / "p.html").write_text(
        "<html><head><title>Pág</title><script>bad()</script></head>"
        "<body><p>Hola <b>mundo</b></p></body></html>", encoding="utf-8")
    for f, needle in [("d.csv", "a"), ("d.json", "clave"), ("n.txt", "plain"),
                      ("n.md", "Título"), ("p.html", "Hola mundo")]:
        r = reader.read(f, **lease(gen))
        assert needle in r.text, f
        assert r.source_id is not None
    r = reader.read("p.html", **lease(gen))
    assert "bad()" not in r.text and r.metadata.get("title") == "Pág"
    store.close()


def test_csv_row_and_field_limits():
    limits = {"max_csv_rows": 3, "max_csv_field_bytes": 10, "max_extract_chars": 1000}
    data = ("h\n" + "\n".join(f"r{i},{'x' * 50}" for i in range(10))).encode()
    out = parse_csv_doc(data, limits)
    assert out["truncated"] is True
    assert out["metadata"]["row_count"] == 3
    assert all(len(c) <= 10 for row in out["text"] and json.loads(out["text"])["rows"] for c in row)


def test_json_depth_limit():
    limits = {"max_json_depth": 5, "max_extract_chars": 1000}
    deep = b'{"a":' * 50 + b"1" + b"}" * 50
    with pytest.raises(DocumentParseError, match="depth"):
        parse_json_doc(deep, limits)
    ok = parse_json_doc(b'{"a": {"b": [1, 2]}}', limits)
    assert ok["format"] == "json"


# ---- malformed / encrypted / mismatched -----------------------------------------------------

@posix_only
@doc_libs
def test_encrypted_and_malformed_pdf(env):
    reader, store, sandbox, ws, gen = env
    make_pdf(ws / "enc.pdf", ["secret"], encrypt="pw123")
    with pytest.raises(DocumentParseError, match="encrypted"):
        reader.read("enc.pdf", **lease(gen))
    (ws / "bad.pdf").write_bytes(b"%PDF-1.4 this is not really a pdf")
    with pytest.raises(DocumentParseError):
        reader.read("bad.pdf", **lease(gen))
    assert reader.registry.list() == []  # failures register nothing
    store.close()


@posix_only
@doc_libs
def test_image_only_pdf_reported_not_ocr(env):
    reader, store, sandbox, ws, gen = env
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.rect(100, 100, 200, 200, fill=1)  # a drawing, no text
    c.showPage()
    c.save()
    (ws / "img.pdf").write_bytes(buf.getvalue())
    r = reader.read("img.pdf", **lease(gen))
    assert any("NOT OCR" in b.get("note", "") for b in r.structure)
    store.close()


def test_format_detection_mismatch(tmp_path):
    with pytest.raises(DocumentParseError, match="not a PDF"):
        detect_format("a.pdf", b"hello text")
    with pytest.raises(DocumentParseError, match="not a ZIP"):
        detect_format("a.docx", b"hello text")
    with pytest.raises(DocumentParseError, match="ZIP-based"):
        detect_format("a.zip", b"PK\x03\x04" + b"\x00" * 100)
    with pytest.raises(DocumentParseError, match="unsupported"):
        detect_format("a.exe", b"MZ...")
    assert detect_format("a.pdf", b"%PDF-1.4 x") == "pdf"
    assert detect_format("a.json", b'  {"a":1}') == "json"
    with pytest.raises(DocumentParseError, match="not JSON"):
        detect_format("a.json", b"plain")


# ---- archive / XML attacks (crafted bytes; guarded before any library parse) -----------------

def test_zip_guard_rejects_bomb_traversal_dupes_encrypted_bzip2(tmp_path):
    bomb = tmp_path / "bomb.xlsx"
    make_zip_bomb(bomb, expanded_mb=200)
    with pytest.raises(ZipGuardError):
        check_zip(bomb.read_bytes(), ZipCaps())
    for name in ["../evil.xml", "/abs.xml", "C:/x.xml"]:
        p = tmp_path / "e.xlsx"
        make_evil_zip(p, name)
        with pytest.raises(ZipGuardError, match="unsafe entry name"):
            check_zip(p.read_bytes(), ZipCaps())
    p = tmp_path / "dup.xlsx"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("a.xml", b"<a/>")
        zf.writestr("a.xml", b"<a/>")
    with pytest.raises(ZipGuardError, match="duplicate"):
        check_zip(p.read_bytes(), ZipCaps())
    p = tmp_path / "enc.xlsx"
    make_evil_zip(p, "a.xml", flag_encrypted=True)
    with pytest.raises(ZipGuardError, match="encrypted"):
        check_zip(p.read_bytes(), ZipCaps())
    p = tmp_path / "bz.xlsx"
    make_evil_zip(p, "a.xml", method=zipfile.ZIP_BZIP2)
    with pytest.raises(ZipGuardError, match="compression"):
        check_zip(p.read_bytes(), ZipCaps())


def test_zip_prescan_rejects_dtd_entities_and_external_rels(tmp_path):
    p = tmp_path / "x.docx"
    make_docx(p, extra_xml=b'<?xml version="1.0"?><!DOCTYPE w [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
                             b'<w:document xmlns:w="x"><w:body>&xxe;</w:body></w:document>')
    with pytest.raises(ZipGuardError, match="DTD/entity"):
        zip_keyword_prescan(p.read_bytes(), ZipCaps())
    p2 = tmp_path / "y.docx"
    make_docx(p2, evil_rels=True)
    with pytest.raises(ZipGuardError, match="external relationship"):
        zip_keyword_prescan(p2.read_bytes(), ZipCaps())


@posix_only
@doc_libs
def test_reader_rejects_evil_docx_before_worker(env):
    reader, store, sandbox, ws, gen = env
    make_docx(ws / "evil.docx", evil_rels=True)
    with pytest.raises((ZipGuardError, DocumentParseError)):
        reader.read("evil.docx", **lease(gen))
    assert reader.registry.list() == []
    store.close()


@posix_only
@doc_libs
def test_zip_bomb_rejected_by_reader(env):
    reader, store, sandbox, ws, gen = env
    make_zip_bomb(ws / "bomb.xlsx", expanded_mb=200)
    with pytest.raises((ZipGuardError, DocumentParseError)):
        reader.read("bomb.xlsx", **lease(gen))
    store.close()


@posix_only
def test_macro_extensions_rejected(env):
    reader, store, sandbox, ws, gen = env
    (ws / "m.docm").write_bytes(b"PK\x03\x04" + b"\x00" * 50)
    with pytest.raises(DocumentParseError, match="macro-enabled"):
        reader.read("m.docm", **lease(gen))
    store.close()


# ---- sandbox / path security -------------------------------------------------------------------

@posix_only
def test_symlink_and_hardlink_refusal_and_traversal(env, tmp_path):
    reader, store, sandbox, ws, gen = env
    outside = tmp_path / "outside.txt"
    outside.write_text("secret outside", encoding="utf-8")
    (ws / "link.txt").symlink_to(outside)
    with pytest.raises(HiveError):
        reader.read("link.txt", **lease(gen))
    (ws / "hard.txt").write_text("x", encoding="utf-8")
    os.link(ws / "hard.txt", ws / "hard2.txt")
    with pytest.raises(HiveError, match="hard link"):
        reader.read("hard2.txt", **lease(gen))
    with pytest.raises(HiveError):
        reader.read("../outside.txt", **lease(gen))
    store.close()


@posix_only
def test_oversized_input_rejected(env):
    reader, store, sandbox, ws, gen = env
    reader.config = DocumentsConfig(enabled=True, max_input_bytes=1024)
    (ws / "big.txt").write_text("x" * 5000, encoding="utf-8")
    with pytest.raises(DocumentParseError, match="max_input_bytes"):
        reader.read("big.txt", **lease(gen))
    store.close()


# ---- worker failure paths (parent-side handling; real pool behavior covered in W1 tests) --------

class _FakePool:
    def __init__(self, result=None, exc=None):
        self.result = result
        self.exc = exc

    def run(self, *a, **k):
        if self.exc:
            raise self.exc
        return self.result


@posix_only
@doc_libs
def test_worker_crash_and_hash_mismatch_fail_closed(env, monkeypatch):
    reader, store, sandbox, ws, gen = env
    make_pdf(ws / "doc.pdf", ["hello"])
    reader._pool = _FakePool(result=ParserJobResult(ok=False, op="parse_pdf", error="worker died"))
    with pytest.raises(DocumentParseError, match="worker failed"):
        reader.read("doc.pdf", **lease(gen))
    assert reader.registry.list() == []
    # hash mismatch: parsed bytes differ from sandbox-read bytes
    import hashlib

    bad = {"error": None, "format": "pdf", "raw_sha256": "0" * 64, "text": "x"}
    payload = json.dumps(bad).encode()
    reader._pool = _FakePool(result=ParserJobResult(
        ok=True, op="parse_pdf", output=payload,
        output_sha256=hashlib.sha256(payload).hexdigest(), output_bytes=len(payload)))
    with pytest.raises(DocumentParseError, match="hash mismatch"):
        reader.read("doc.pdf", **lease(gen))
    # capacity exhaustion -> fail closed
    reader._pool = _FakePool(exc=ParserJobRejected("queue full"))
    with pytest.raises(DocumentUnavailableError, match="queue full"):
        reader.read("doc.pdf", **lease(gen))
    store.close()


def test_non_posix_fail_closed_everywhere(tmp_path):
    store = CheckpointStore(str(tmp_path / "cp.db"))
    sandbox = WorkspaceSandbox(str(tmp_path / "ws"))
    (tmp_path / "ws" / "a.pdf").write_bytes(b"%PDF-1.4 x")
    reader = DocumentReader(store=store, sandbox=sandbox,
                            config=DocumentsConfig(enabled=True), platform="nt")
    # on real Windows the sandbox itself fails closed first; with the platform
    # override on POSIX the worker pool fails closed — both are fail-closed
    with pytest.raises(HiveError):
        reader.read("a.pdf")
    store.close()


def test_platform_fail_closed_guarantee(tmp_path):
    """Runs on every platform: the pool/reader fail closed when the platform
    override says non-POSIX (this is what Windows exercises natively — the
    env-based tests above prove the real sandbox-level fail-closed there)."""
    from geosgan_hive.parser_isolation import ParserWorkerPool, ParserWorkerUnavailable

    with pytest.raises(ParserWorkerUnavailable):
        ParserWorkerPool(platform="nt", work_root=str(tmp_path)).run("ping")
    store = CheckpointStore(str(tmp_path / "cp.db"))
    sandbox = WorkspaceSandbox(str(tmp_path / "ws")) if POSIX else None
    if sandbox is not None:
        (tmp_path / "ws").mkdir(exist_ok=True)
        (tmp_path / "ws" / "a.pdf").write_bytes(b"%PDF-1.4 x")
        reader = DocumentReader(store=store, sandbox=sandbox,
                                config=DocumentsConfig(enabled=True), platform="nt")
        with pytest.raises(HiveError):
            reader.read("a.pdf")
    store.close()


def test_offline_startup_without_doc_extras(tmp_path):
    """Fresh process with the doc libraries blocked at import: the core package
    still imports and stdlib formats (csv/json/txt/...) still work — offline
    startup without document extras is preserved."""
    code = r"""
import sys
class _Block:
    def find_module(self, name, path=None):
        if name.split(".")[0] in ("pypdf", "docx", "openpyxl", "defusedxml", "reportlab"):
            return self
    def load_module(self, name):
        raise ImportError("blocked: " + name)
sys.meta_path.insert(0, _Block())
sys.path.insert(0, "SRC")
import geosgan_hive  # must import fine without the extras
from geosgan_hive.agent_config import DocumentsConfig
from geosgan_hive.checkpoints import CheckpointStore
from geosgan_hive.documents import DocumentReader
from geosgan_hive.sandbox import WorkspaceSandbox
import tempfile, os
tmp = tempfile.mkdtemp()
ws = os.path.join(tmp, "ws"); os.mkdir(ws)
open(os.path.join(ws, "d.csv"), "w").write("a,b\n1,x\n")
store = CheckpointStore(os.path.join(tmp, "cp.db"))
r = DocumentReader(store=store, sandbox=WorkspaceSandbox(ws),
                   config=DocumentsConfig(enabled=True))
import os as _os
if _os.name == "posix":
    out = r.read("d.csv")  # stdlib format works without any extra
    assert out.format == "csv" and "1" in out.text
    print("CSV_OK_WITHOUT_EXTRAS")
else:
    from geosgan_hive.errors import HiveError as _HE
    try:
        r.read("d.csv")
        raise SystemExit("FAIL: windows did not fail closed")
    except _HE:
        print("WIN_FAIL_CLOSED_OK")
print("IMPORT_OK", geosgan_hive.__version__)
"""
    src = str(Path(__file__).resolve().parent.parent / "src")
    proc = subprocess.run([sys.executable, "-c", code.replace("SRC", src)],
                          capture_output=True, text=True, timeout=120)
    assert "IMPORT_OK" in proc.stdout, proc.stderr[-2000:]
    if POSIX:
        assert "CSV_OK_WITHOUT_EXTRAS" in proc.stdout
    else:
        assert "WIN_FAIL_CLOSED_OK" in proc.stdout


@posix_only
@doc_libs
def test_missing_dependency_in_worker_fails_closed(env):
    """Real parser worker whose sys.path has NO site-packages: the optional
    parser dependency is missing there -> typed DocumentUnavailableError,
    no parse, no source."""
    from geosgan_hive.parser_isolation import ParserWorkerPool

    reader, store, sandbox, ws, gen = env
    make_pdf(ws / "doc.pdf", ["hello"])
    # the parser libraries are import-blocked INSIDE the worker regardless of
    # where pip installed them (works in user-site and system-site installs)
    pool = ParserWorkerPool(work_root=str(store.path + ".jobs"),
                            worker_blocked_modules=["pypdf", "docx", "openpyxl",
                                                    "defusedxml"])
    reader._pool = pool
    with pytest.raises(DocumentUnavailableError, match="pypdf"):
        reader.read("doc.pdf", **lease(gen))
    assert reader.registry.list() == []
    store.close()


# ---- provenance, privacy, truncation, leases ----------------------------------------------------

@posix_only
@doc_libs
def test_truncation_labeled_and_source_provenance(env):
    reader, store, sandbox, ws, gen = env
    reader.config = DocumentsConfig(enabled=True, max_extract_chars=200)
    make_pdf(ws / "doc.pdf", ["palabra " * 500])
    r = reader.read("doc.pdf", **lease(gen))
    assert r.truncated is True and len(r.text) <= 250
    d = reader.registry.get(r.source_id)
    prov = json.loads(d["provenance_json"])
    assert prov["truncated"] is True  # partial extraction explicitly labeled
    ex = reader.registry.excerpt(r.source_id)
    assert ex["available"] is False  # metadata-only: no invented excerpt
    store.close()


@posix_only
@doc_libs
def test_stale_lease_registration_denied(env):
    reader, store, sandbox, ws, gen = env
    make_pdf(ws / "doc.pdf", ["x"])
    store.cancel("run-1")
    with pytest.raises(CheckpointError, match="stale worker"):
        reader.read("doc.pdf", **lease(gen))
    assert reader.registry.list() == []
    store.close()


@posix_only
@doc_libs
def test_cross_run_access_denied(env):
    reader, store, sandbox, ws, gen = env
    make_pdf(ws / "doc.pdf", ["x"])
    r = reader.read("doc.pdf", **lease(gen))
    store.create_run(run_id="run-2", owner="w2", task="t", agent="a", model="ollama/llama3.2")
    gen2 = store.get("run-2").lease_generation
    b2 = reader.registry.read_binding(run_id="run-2", lease_owner="w2", lease_generation=gen2)
    with pytest.raises(CheckpointError, match="cross-run"):
        reader.registry.cite(r.source_id, binding=b2)
    store.close()


# ---- tool interface ------------------------------------------------------------------------------

@posix_only
@doc_libs
def test_doc_read_tool_typed_input_and_clamped_limits(env):
    reader, store, sandbox, ws, gen = env
    reader.config = DocumentsConfig(enabled=True, max_extract_chars=500)
    make_pdf(ws / "doc.pdf", ["texto " * 500])
    catalog = ToolCatalog()
    for spec, adapter in doc_tools(reader, lease_lookup=lambda rid: store.get(rid)):
        catalog.register(spec, adapter)
    assert catalog.names() == ["doc_read"]
    policy = build_policy(global_enabled=["doc_read"],
                          agent_permissions={"researcher": (True, ["doc_read"])})
    gw = ToolGateway(catalog, policy, store=store, approval_manager=ApprovalManager(store))
    ctx = ToolExecutionContext(agent="researcher", run_id="run-1")
    out = json.loads(gw.execute("doc_read",
                                {"path": "doc.pdf", "max_chars": 200_000},  # clamped to 500
                                ctx=ctx).output)
    assert out["format"] == "pdf" and out["source_id"]
    assert len(out["text"]) <= 550  # policy maximum wins over the model request
    assert out["citation"].startswith("[src_")
    # extra fields forbidden (no parser options / shell / paths outside model)
    res = gw.execute("doc_read", {"path": "doc.pdf", "shell": "rm -rf /"}, ctx=ctx)
    assert res.ok is False and "Extra inputs" in res.error
    res = gw.execute("doc_read", {"path": "/etc/passwd"}, ctx=ctx)
    assert res.ok is False
    store.close()
