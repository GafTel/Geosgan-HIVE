"""W4 (v0.8.0) offline demo: local document reading with parser isolation.

Fully offline: fixtures are generated locally with the approved optional
libraries; PDF/DOCX/XLSX are parsed ONLY inside the isolated parser worker
(POSIX; fail closed elsewhere); CSV/JSON/TXT/MD/HTML via bounded stdlib
parsers. No network, no API keys, zero cost.
"""

from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

from geosgan_hive.agent_config import DocumentsConfig
from geosgan_hive.checkpoints import CheckpointStore
from geosgan_hive.documents import DocumentParseError, DocumentReader
from geosgan_hive.sandbox import WorkspaceSandbox


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="hive-docs-demo-"))
    ws = tmp / "ws"
    ws.mkdir()

    # ---- fixtures (local, deterministic) ----
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.drawString(100, 750, "Hola mundo — documento de demostración")
    c.showPage()
    c.drawString(100, 750, "Second page, English.")
    c.save()
    (ws / "informe.pdf").write_bytes(buf.getvalue())

    import docx

    d = docx.Document()
    d.add_heading("Informe Ñandú", level=1)
    d.add_paragraph("Párrafo con acentos y ñ.")
    d.save(str(ws / "informe.docx"))

    import openpyxl

    wb = openpyxl.Workbook()
    s = wb.active
    s.title = "datos"
    s["A1"] = "concepto"
    s["B1"] = "importe"
    s["A2"] = "venta"
    s["B2"] = 100
    s["C2"] = "=B2*0.21"  # formula: reported, never evaluated
    wb.save(str(ws / "tabla.xlsx"))

    (ws / "datos.csv").write_text("a,b\n1,x\n2,\n", encoding="utf-8")
    (ws / "nota.md").write_text("# Nota\ntexto", encoding="utf-8")

    # hostile archive (crafted bytes, only ever meets the zip guard)
    with zipfile.ZipFile(ws / "bomba.xlsx", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("xl/workbook.xml", b"<workbook/>")
        zf.writestr("xl/worksheets/sheet1.xml", b"x" * (150 * 1024 * 1024))

    store = CheckpointStore(str(tmp / "cp.db"))
    store.create_run(run_id="demo", owner="demo-w", task="d", agent="researcher",
                     model="ollama/llama3.2")
    gen = store.get("demo").lease_generation
    reader = DocumentReader(store=store, sandbox=WorkspaceSandbox(str(ws)),
                            config=DocumentsConfig(enabled=True))
    kw = dict(run_id="demo", lease_owner="demo-w", lease_generation=gen)

    for f in ["informe.pdf", "informe.docx", "tabla.xlsx", "datos.csv", "nota.md"]:
        r = reader.read(f, **kw)
        print(f"{f}: format={r.format} parser={r.parser} source={r.source_id} "
              f"truncated={r.truncated}")
        print(f"   text: {r.text[:90]!r}")
    r = reader.read("tabla.xlsx", **kw)
    print(f"   xlsx formulas={r.extra.get('formula_cells')} "
          f"missing_cached={r.extra.get('missing_cached_values')} (never evaluated)")
    print(f"   citation: {reader.registry.cite(r.source_id)[:110]}...")

    try:
        reader.read("bomba.xlsx", **kw)
        raise SystemExit("FAIL: zip bomb accepted")
    except DocumentParseError as exc:
        print(f"zip bomb rejected by the archive guard: {str(exc)[:80]}")
    assert reader.registry.list(), "successful parses must register sources"
    db = open(tmp / "cp.db", "rb").read()
    assert b"Hola mundo" not in db  # metadata-only: contents never persisted
    print("metadata-only OK: document contents are not persisted in the checkpoint DB")
    print("Demo finished. Fully offline; worker isolation on POSIX; zero cost.")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
