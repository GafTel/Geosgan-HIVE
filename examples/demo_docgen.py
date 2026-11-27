"""W5 (v0.8.0) offline demo: secure document generation with exact-byte approval.

Fully offline: documents are generated locally (PDF via ReportLab, DOCX via
python-docx, XLSX via openpyxl inside the isolated parser worker on POSIX;
CSV/JSON/TXT/MD via bounded stdlib in-process). The mandatory flow is
generate-once -> stage -> hash -> operator approval -> publish exact bytes,
with destination preconditions and lease fencing revalidated at publication.
Spreadsheet formula injection stays literal; PDF markup is escaped; generated
artifacts are reopened with independent readers to verify them. No network,
no API keys, zero cost.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import tempfile
from pathlib import Path

from geosgan_hive.agent_config import DocumentsConfig
from geosgan_hive.checkpoints import CheckpointStore
from geosgan_hive.docgen import (
    DocBlock,
    DocumentGenError,
    DocumentSpec,
    DocumentWriter,
    SheetSpec,
)
from geosgan_hive.sandbox import WorkspaceSandbox

DOC_EXTS = [".pdf", ".docx", ".xlsx", ".csv", ".json", ".md", ".txt"]


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="hive-docgen-demo-"))
    ws = tmp / "ws"
    (ws / "out").mkdir(parents=True)

    store = CheckpointStore(str(tmp / "cp.db"))
    store.create_run(run_id="demo", owner="demo-w", task="d", agent="researcher",
                     model="ollama/llama3.2")
    gen = store.get("demo").lease_generation
    sandbox = WorkspaceSandbox(str(ws), write_extensions=DOC_EXTS)
    writer = DocumentWriter(store=store, sandbox=sandbox,
                            config=DocumentsConfig(enabled=True))
    lease = dict(run_id="demo", task_id="d", lease_owner="demo-w", lease_generation=gen)

    spec = DocumentSpec(
        title="Informe de demostración ñandú",
        author="GEOSGAN HIVE",
        header="DEMO — CONFIDENCIAL",
        footer="borrador",
        blocks=[
            DocBlock(kind="heading", level=1, text="Resumen ejecutivo"),
            DocBlock(kind="paragraph",
                     text="El zorro ágil salta sobre el ñandú. " * 20),
            DocBlock(kind="table", rows=[["concepto", "importe"],
                                         ["ventas", "100"], ["iva", "=B2*0.21"]]),
            DocBlock(kind="page_break"),
            DocBlock(kind="references", items=["Fuente A (demo)", "Fuente B (demo)"]),
        ],
        sheets=[SheetSpec(name="Datos ñ", header=["concepto", "valor"],
                          rows=[["venta", 100], ["fórmula", "=1+1"], ["comando", "@cmd"]])],
    )

    # ---- 1. prepare: generate ONCE, stage, hash, bind an approval ----
    art = writer.prepare(format="xlsx", destination="out/reporte.xlsx", spec=spec,
                         agent="researcher", **lease)
    print(f"prepared: status={art.status} approval={art.approval_id[:12]}… "
          f"bytes={art.byte_count} sha256={art.artifact_sha256[:16]}…")
    assert not (ws / "out" / "reporte.xlsx").exists(), "prepare must not write"

    # ---- 2. publication before operator approval is rejected (fail closed) ----
    try:
        writer.publish(approval_id=art.approval_id, agent="researcher", **lease)
        raise SystemExit("FAIL: publish without approval succeeded")
    except DocumentGenError as exc:
        print(f"publish before approval rejected: {exc}")

    # ---- 3. operator approves the EXACT artifact bytes, then publish ----
    assert store.decide_approval(art.approval_id, decision="approved", operator="operator-1")
    res = writer.publish(approval_id=art.approval_id, agent="researcher", **lease)
    print(f"published: {res['destination']} sha256={res['artifact_sha256'][:16]}… "
          f"source={res['source_id']}")
    data = (ws / "out" / "reporte.xlsx").read_bytes()
    assert hashlib.sha256(data).hexdigest() == art.artifact_sha256, "exact-byte binding"

    # ---- 4. replay of the consumed approval is rejected ----
    try:
        writer.publish(approval_id=art.approval_id, agent="researcher", **lease)
        raise SystemExit("FAIL: replayed publish succeeded")
    except DocumentGenError as exc:
        print(f"replay rejected: {exc}")

    # ---- 5. verify by REOPENING with an independent reader ----
    import openpyxl

    wb = openpyxl.load_workbook(ws / "out" / "reporte.xlsx")
    sh = wb["Datos ñ"]  # first sheet keeps its spec name
    cell = sh.cell(row=3, column=2)
    assert cell.data_type == "s" and cell.value == "=1+1", "formula must stay literal"
    print(f"xlsx reopened OK: sheet={sh.title!r} formula cell literal: {cell.value!r}")
    cite = writer.registry.cite(res["source_id"])
    print(f"artifact provenance citation: {cite[:120]}")
    prov = json.loads(writer.registry.get(res["source_id"])["provenance_json"])
    assert prov["generated"] is True, "generated artifacts are never external evidence"

    # ---- 6. CSV with the documented spreadsheet-safe literal-quote policy ----
    csv_spec = DocumentSpec(sheets=[SheetSpec(name="d", header=["a", "b"],
                                              rows=[["=SUM(1,2)", "x"], [None, "ñ"]])])
    art2 = writer.prepare(format="csv", destination="out/export.csv", spec=csv_spec,
                          agent="researcher", **lease)
    assert store.decide_approval(art2.approval_id, decision="approved", operator="operator-1")
    writer.publish(approval_id=art2.approval_id, agent="researcher", **lease)
    rows = list(csv.reader(io.StringIO((ws / "out" / "export.csv").read_text("utf-8"))))
    assert rows[1][0] == "'=SUM(1,2)" and rows[2][0] == "", rows
    print(f"csv reopened OK: formula cell kept literal {rows[1][0]!r}, missing stays empty")

    # ---- 7. PDF (worker-isolated on POSIX) verified with pypdf ----
    if __import__("os").name == "posix":
        art3 = writer.prepare(format="pdf", destination="out/informe.pdf", spec=spec,
                              agent="researcher", **lease)
        assert store.decide_approval(art3.approval_id, decision="approved",
                                     operator="operator-1")
        writer.publish(approval_id=art3.approval_id, agent="researcher", **lease)
        import pypdf

        raw = (ws / "out" / "informe.pdf").read_bytes()
        assert b"/JS" not in raw and b"/OpenAction" not in raw and b"/Launch" not in raw
        text = "".join(p.extract_text() or "" for p in pypdf.PdfReader(io.BytesIO(raw)).pages)
        assert "Informe de demostración" in text and "Page 1" in text
        print(f"pdf reopened OK: {len(pypdf.PdfReader(io.BytesIO(raw)).pages)} pages, "
              "no JS/actions embedded, unicode + page numbers present")

    # ---- 8. metadata-only privacy: artifact bytes never persisted in SQLite ----
    db = open(tmp / "cp.db", "rb").read()
    assert "ñandú".encode() not in db and b"=1+1" not in db
    print("metadata-only OK: generated content is not persisted in the checkpoint DB")
    print("Demo finished. Fully offline; exact-byte approval; worker isolation on POSIX; "
          "zero cost.")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
