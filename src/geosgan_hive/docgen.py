"""W5 (v0.8.0): secure document GENERATION with exact-byte approval + publication.

Flow (mandatory acceptance gate):
  1. validate the structured DocumentSpec (extra=forbid, bounded) and clamp all
     limits to operator maxima (models may lower, never raise);
  2. validate destination (workspace-relative, extension matches format,
     containment via WorkspaceSandbox). Publication is CREATION-ONLY: an
     existing destination is rejected (overwrite disabled — a rename cannot
     atomically bind the pre-existing content hash);
  3. generate the document EXACTLY ONCE — pdf/docx/xlsx inside the trusted
     parser-worker (same isolation as W4: rlimits, watchdog, killpg, bounded
     IPC, sanitized env); csv/json/txt/md in-process via stdlib, bounded;
  4. stage the bytes securely (sandbox.stage_bytes: 0700 dir, 0600 file,
     bounded count + aggregate bytes, age-swept with live approvals protected;
     .hive-staging is denied to agent-facing file tools and never persisted to
     SQLite);
  5. compute exact sha256 + byte count;
  6. bind an ApprovalManager approval to run/task/agent/tool/version, canonical
     arg digest, destination absence, exact new artifact sha256 + byte count,
     staging id, expiry;
  7. publication revalidates approval (single-use, unexpired, unconsumed), run
     binding, current policy, destination preconditions and lease owner+
     generation, re-hashes the staged bytes (substitution -> reject), then
     ATOMICALLY consumes the approval AND inserts the durable execution-ledger
     row in ONE transaction (store.start_execution), publishes via
     sandbox.write_bytes_atomic (parent dir handle + staged-fd re-verification +
     ATOMIC NO-CLOBBER os.link, never rename/replace + post-publish re-open/
     re-hash; a hash mismatch is never reported as success), marks the ledger
     row completed, and registers the artifact provenance (clearly marked
     GENERATED, never as retrieved external evidence).

Never regenerate after approval; an approval for one artifact never authorizes
different bytes; replay/expired/substitution/stale-worker publication rejected.
Crash windows are documented and never auto-replayed (the approval is consumed
atomically with the ledger start, so a retry is impossible by construction).
Staging is in-workspace, never persisted to SQLite, age-swept, count/byte-bounded.

Formula injection: XLSX cells and CSV fields starting with = + - @ TAB CR are
kept LITERAL (never formulas; the CSV transformation is the documented
spreadsheet-safe `'` prefix policy). No macros, no external relationships, no
DDE, no executable formulas. PDF text is XML-escaped before ReportLab markup
components. No remote fonts/images, no JS/actions/attachments. Trusted local
fonts only. Unicode (es/en) supported.

Disabled by default (documents.enabled); optional deps lazy; core startup works
without them; non-POSIX fails closed for worker formats (Windows Job Objects =
documented future path; no timeout-only mode).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import stat
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4
from xml.sax.saxutils import escape as xml_escape

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .agent_config import DocumentsConfig
from .approvals import ApprovalManager
from .checkpoints import ToolExecutionStatus
from .errors import HiveError
from .parser_isolation import ParserIsolationError, ParserResourceLimits, ParserWorkerPool
from .sources import SourceRegistry, SourceType
from .tool_gateway import ToolExecutionContext, redact_secrets
from .tool_spec import SideEffect, ToolSpec, args_sha256, canonical_args

GEN_FORMATS = ("pdf", "docx", "xlsx", "csv", "json", "txt", "md")
WORKER_GEN_FORMATS = ("pdf", "docx", "xlsx")
FORMAT_EXT = {"pdf": ".pdf", "docx": ".docx", "xlsx": ".xlsx", "csv": ".csv",
              "json": ".json", "txt": ".txt", "md": ".md"}

#: spreadsheet formula trigger characters (CSV/XLSX export policy)
FORMULA_TRIGGERS = ("=", "+", "-", "@", "\t", "\r")
#: worksheet names: Excel-forbidden chars rejected, length bounded
_SHEET_NAME_BAD = re.compile(r"[\[\]:*?/\\]")


class DocumentGenError(HiveError):
    """Base for generation failures (all fail closed)."""


class DocumentGenUnavailableError(DocumentGenError):
    """Missing optional dependency / non-POSIX worker / disabled — fail closed."""


class DocumentSpecError(DocumentGenError):
    """Invalid structured specification or limits."""


# ---- structured, strictly validated specification (extra=forbid everywhere) --------

class DocBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["heading", "paragraph", "list", "table", "page_break", "references"]
    level: int = Field(default=1, ge=1, le=6)
    text: str = Field(default="", max_length=100_000)
    items: list[str] = Field(default_factory=list, max_length=1_000)
    rows: list[list[str]] = Field(default_factory=list, max_length=1_000)

    @classmethod
    def _bound_items(cls, v: list[str]) -> list[str]:
        return [x[:10_000] for x in v]


class SheetSpec(BaseModel):
    """One worksheet: bounded, sanitized name; typed cell values only."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=31)
    header: list[str] = Field(default_factory=list, max_length=50)
    rows: list[list[Any]] = Field(default_factory=list, max_length=1_000)

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        if _SHEET_NAME_BAD.search(self.name) or self.name in (".", ".."):
            raise ValueError(f"unsafe worksheet name {self.name!r}")
        for row in self.rows:
            if len(row) > 50:
                raise ValueError("row exceeds 50 columns")
            for v in row:
                if isinstance(v, float) and v != v or isinstance(v, float) and v in (
                        float("inf"), float("-inf")):
                    raise ValueError("non-finite numeric value rejected")
                if not isinstance(v, (str, int, float, bool, type(None))):
                    raise ValueError(f"unsupported cell type {type(v).__name__}")
                if isinstance(v, str) and len(v) > 10_000:
                    raise ValueError("cell string too long")


class DocumentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(default="", max_length=1_000)
    author: str = Field(default="", max_length=300)
    subject: str = Field(default="", max_length=300)
    page_size: Literal["A4", "letter"] = "A4"
    header: str = Field(default="", max_length=200)
    footer: str = Field(default="", max_length=200)
    blocks: list[DocBlock] = Field(default_factory=list, max_length=2_000)
    sheets: list[SheetSpec] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def _unique_sheet_names(self) -> DocumentSpec:
        names = [s.name for s in self.sheets]
        if len(set(names)) != len(names):
            raise ValueError("duplicate worksheet names are rejected (no silent renames)")
        return self


# ---- worker-side generators (trusted parser worker only) -------------------------------

def _spec_dict(spec: DocumentSpec) -> dict[str, Any]:
    return json.loads(spec.model_dump_json())


def worker_generate_pdf(spec: DocumentSpec, limits: dict[str, Any]) -> bytes:
    """ReportLab platypus; all untrusted text XML-escaped; local fonts only; no
    remote resources, no JS/actions. Validates the produced PDF by reopening it."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:
        raise DocumentGenUnavailableError("reportlab not installed") from exc
    max_blocks = int(limits.get("max_docx_blocks", 2000))
    max_pages = int(limits.get("max_pdf_pages", 500))
    styles = getSampleStyleSheet()
    page_size = letter if spec.page_size == "letter" else A4
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=page_size, title=xml_escape(spec.title)[:200],
                            author=xml_escape(spec.author)[:200])
    story: list[Any] = []
    if spec.title:
        story.append(Paragraph(xml_escape(spec.title), styles["Title"]))
    n = 0
    for b in spec.blocks:
        n += 1
        if n > max_blocks:
            break
        k = b.kind
        if k == "page_break":
            story.append(PageBreak())
        elif k == "heading":
            story.append(Paragraph(xml_escape(b.text), styles[f"Heading{min(b.level, 6)}"]))
        elif k == "paragraph":
            story.append(Paragraph(xml_escape(b.text), styles["BodyText"]))
        elif k == "list":
            for item in b.items[:200]:
                story.append(Paragraph("• " + xml_escape(item), styles["BodyText"]))
        elif k == "references":
            for i, item in enumerate(b.items[:500], 1):
                story.append(Paragraph(f"[{i}] {xml_escape(item)}", styles["BodyText"]))
        elif k == "table":
            # Table cells are plain strings: ReportLab does NOT parse markup in
            # them, so untrusted cell text is inert here (no escaping needed and
            # escaping would double-display entities). Paragraph-based blocks
            # above are the markup-aware path and are always xml_escaped.
            rows = [[str(c) for c in row[:20]] for row in b.rows[:200]]
            if rows:
                t = Table(rows, colWidths=[4 * cm] * len(rows[0]))
                t.setStyle(TableStyle([
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ]))
                story.append(t)
        story.append(Spacer(1, 6))

    def on_page(canv: Any, doc_: Any) -> None:
        canv.saveState()
        if spec.header:
            canv.drawString(2 * cm, page_size[1] - 1.2 * cm, xml_escape(spec.header)[:100])
        if spec.footer:
            canv.drawString(2 * cm, 1 * cm, xml_escape(spec.footer)[:100])
        canv.drawRightString(page_size[0] - 2 * cm, 1 * cm, f"Page {canv.getPageNumber()}")
        canv.restoreState()

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    data = buf.getvalue()
    # validate the artifact by reopening with an independent reader (pypdf)
    try:
        import pypdf

        r = pypdf.PdfReader(io.BytesIO(data))
        pages = len(r.pages)
        if pages == 0 or pages > max_pages:
            raise DocumentGenError(f"generated PDF has {pages} pages (cap {max_pages})")
    except ImportError:
        pass  # pypdf optional for generation; tests validate with it
    except DocumentGenError:
        raise
    except Exception as exc:
        raise DocumentGenError(f"generated PDF failed validation: {type(exc).__name__}") from exc
    return data


def worker_generate_docx(spec: DocumentSpec, limits: dict[str, Any]) -> bytes:
    try:
        import docx
    except ImportError as exc:
        raise DocumentGenUnavailableError("python-docx not installed") from exc
    max_blocks = int(limits.get("max_docx_blocks", 2000))
    d = docx.Document()
    cp = d.core_properties
    cp.title = spec.title[:300]
    cp.author = spec.author[:300]
    cp.subject = spec.subject[:300]
    if spec.title:
        d.add_heading(spec.title, level=1)
    n = 0
    for b in spec.blocks:
        n += 1
        if n > max_blocks:
            break
        k = b.kind
        if k == "page_break":
            d.add_page_break()
        elif k == "heading":
            d.add_heading(b.text, level=min(b.level, 6))
        elif k == "paragraph":
            d.add_paragraph(b.text)
        elif k == "list":
            for item in b.items[:200]:
                d.add_paragraph(item, style="List Bullet")
        elif k == "references":
            for i, item in enumerate(b.items[:500], 1):
                d.add_paragraph(f"[{i}] {item}")
        elif k == "table":
            rows = b.rows[:200]
            if rows:
                t = d.add_table(rows=len(rows), cols=min(len(rows[0]), 20))
                t.style = "Table Grid"
                for ri, row in enumerate(rows):
                    for ci, cell in enumerate(row[:20]):
                        t.cell(ri, ci).text = str(cell)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def _xlsx_cell_literal(ws: Any, row: int, col: int, value: Any) -> None:
    """Set a cell keeping formula-triggering strings LITERAL (never a formula)."""
    c = ws.cell(row=row, column=col)
    c.value = value
    if isinstance(value, str) and value[:1] in FORMULA_TRIGGERS:
        c.data_type = "s"  # force literal string; openpyxl would infer 'f' for '='
    return None


def worker_generate_xlsx(spec: DocumentSpec, limits: dict[str, Any]) -> bytes:
    try:
        import openpyxl
    except ImportError as exc:
        raise DocumentGenUnavailableError("openpyxl not installed") from exc
    max_sheets = int(limits.get("max_xlsx_sheets", 10))
    max_cells = int(limits.get("max_xlsx_cells", 20_000))
    wb = openpyxl.Workbook()
    first = wb.active
    first.title = "Sheet1"
    sheets = spec.sheets[:max_sheets] or [SheetSpec(name="Sheet1")]
    count = 0
    for si, sh in enumerate(sheets):
        ws = first if si == 0 else wb.create_sheet(title=sh.name[:31])
        ws.title = sh.name  # SheetSpec already validated chars/length; dupes raise here
        r = 1
        if sh.header:
            for ci, h in enumerate(sh.header[:50], 1):
                _xlsx_cell_literal(ws, r, ci, str(h))
            r += 1
        for row in sh.rows[:1000]:
            for ci, v in enumerate(row[:50], 1):
                count += 1
                if count > max_cells:
                    break
                _xlsx_cell_literal(ws, r, ci, v)
            r += 1
            if count > max_cells:
                break
        for col in range(1, min(len(sh.header or (sh.rows[0] if sh.rows else [])), 50) + 1):
            ws.column_dimensions[openpyxl.utils.get_column_letter(col)].width = 20
        ws.freeze_panes = "A2" if sh.header else None
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ---- in-process simple generators (stdlib, bounded) --------------------------------------

def gen_csv_bytes(spec: DocumentSpec, limits: dict[str, Any]) -> tuple[bytes, str]:
    """CSV export with the DOCUMENTED spreadsheet-safe policy: cells starting
    with a formula trigger get a leading apostrophe (kept as literal text by
    spreadsheet applications). The transformation is reported, never silent."""
    max_chars = int(limits.get("max_extract_chars", 20_000))
    sheet = spec.sheets[0] if spec.sheets else SheetSpec(name="data")
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    transformed = 0
    if sheet.header:
        w.writerow(sheet.header[:50])
    for row in sheet.rows[:1000]:
        out = []
        for v in row[:50]:
            if v is None:
                out.append("")  # explicit missing stays empty, never zero
            elif isinstance(v, str) and v[:1] in FORMULA_TRIGGERS:
                out.append("'" + v)  # documented literal-quote policy
                transformed += 1
            else:
                out.append(v)
        w.writerow(out)
    data = buf.getvalue().encode("utf-8")
    if len(data) > max_chars * 4:
        raise DocumentSpecError("CSV output exceeds bound")
    note = ("spreadsheet-safe policy applied: "
            f"{transformed} formula-triggering cell(s) prefixed with ' (kept literal)"
            if transformed else "no formula-triggering cells")
    return data, note


def gen_json_bytes(spec: DocumentSpec, limits: dict[str, Any]) -> bytes:
    max_chars = int(limits.get("max_extract_chars", 20_000))
    payload = _spec_dict(spec)
    try:
        text = json.dumps(payload, ensure_ascii=False, allow_nan=False,
                          sort_keys=True, default=_json_reject)
    except (TypeError, ValueError) as exc:
        raise DocumentSpecError(f"JSON serialization rejected: {exc}") from exc
    data = text.encode("utf-8")
    if len(data) > max_chars * 4:
        raise DocumentSpecError("JSON output exceeds bound")
    return data


def _json_reject(o: Any) -> Any:
    raise TypeError(f"unsupported type for JSON: {type(o).__name__}")


def gen_text_bytes(spec: DocumentSpec, fmt: str, limits: dict[str, Any]) -> bytes:
    max_chars = int(limits.get("max_extract_chars", 20_000))
    lines: list[str] = []
    if spec.title:
        lines.append(f"# {spec.title}" if fmt == "md" else spec.title)
    for b in spec.blocks[:2000]:
        if b.kind == "heading":
            lines.append("#" * min(b.level, 6) + " " + b.text if fmt == "md"
                         else b.text.upper())
        elif b.kind == "paragraph":
            lines.append(b.text)
        elif b.kind == "list":
            lines.extend(("- " + i) if fmt == "md" else i for i in b.items[:200])
        elif b.kind == "references":
            lines.extend(f"[{i}] {t}" for i, t in enumerate(b.items[:500], 1))
        elif b.kind == "table":
            for row in b.rows[:200]:
                lines.append(" | ".join(str(c) for c in row[:20]))
        elif b.kind == "page_break":
            lines.append("---" if fmt == "md" else "")
    text = "\n".join(lines)
    if len(text) > max_chars * 2:
        raise DocumentSpecError(f"{fmt} output exceeds bound")
    return text.encode("utf-8")


# ---- writer: prepare (generate+stage+approval) / publish (verify+publish) ----------------

@dataclass
class PreparedArtifact:
    approval_id: str
    staging_relpath: str
    artifact_sha256: str
    byte_count: int
    spec_sha256: str
    destination: str
    format: str
    generator: str
    status: str = "pending_approval"


class DocumentWriter:
    def __init__(self, *, store: Any, sandbox: Any, config: DocumentsConfig,
                 registry: SourceRegistry | None = None,
                 approval_manager: ApprovalManager | None = None,
                 pool: ParserWorkerPool | None = None,
                 platform: str | None = None,
                 tool_spec: ToolSpec | None = None) -> None:
        self.store = store
        self.sandbox = sandbox
        self.config = config
        self.registry = registry if registry is not None else SourceRegistry(store)
        self.approval_manager = approval_manager or ApprovalManager(store)
        self._pool = pool
        self._platform = platform or os.name
        self._tool_spec = tool_spec or ToolSpec(
            name="doc_generate", description="document generation (W5)", version="1",
            category="documents", side_effect=SideEffect.LOCAL_MUTATION)

    def _limits(self) -> dict[str, Any]:
        c = self.config
        return {"max_input_bytes": c.max_input_bytes, "max_extract_chars": c.max_extract_chars,
                "max_output_bytes": max(c.max_extract_chars * 4, 262_144),
                "max_pdf_pages": c.max_pdf_pages, "max_docx_blocks": c.max_docx_blocks,
                "max_xlsx_sheets": c.max_xlsx_sheets, "max_xlsx_cells": c.max_xlsx_cells}

    def _active_staging(self) -> set[str]:
        """Staging relpaths still referenced by a live PENDING/APPROVED approval."""
        out: set[str] = set()
        for status in ("pending", "approved"):
            for row in self.store.list_approvals(status=status):
                try:
                    pre = json.loads(row.get("preconditions_json") or "{}")
                except (ValueError, TypeError):
                    continue
                rel = pre.get("staging_relpath")
                if isinstance(rel, str) and rel:
                    out.add(rel)
        return out

    def _sweep_staging(self) -> int:
        """W5 residual staging lifecycle: expire stale approvals, then age-sweep the
        staging dir while PROTECTING artifacts referenced by live approvals (never
        clean up an active artifact owned by another valid worker). Staged bytes
        are sensitive and are never persisted to SQLite."""
        self.store.sweep_expired_approvals()
        return self.sandbox.sweep_staging(
            max_age_seconds=self.config.staging_max_age_seconds,
            protect=self._active_staging())

    def _get_pool(self) -> ParserWorkerPool:
        if self._platform != "posix":
            raise DocumentGenUnavailableError(
                f"document generation (worker formats) is disabled on platform "
                f"{self._platform!r} — parser-worker isolation is POSIX-only; fail closed")
        if self._pool is None:
            c = self.config
            self._pool = ParserWorkerPool(
                max_workers=c.worker_max_workers, max_pending=c.worker_max_pending,
                op_timeout_seconds=c.worker_op_timeout_seconds,
                limits=ParserResourceLimits(
                    max_memory_bytes=c.worker_max_memory_bytes,
                    max_cpu_seconds=int(c.worker_op_timeout_seconds),
                    max_output_file_bytes=max(c.max_extract_chars * 4, 262_144),
                    max_input_bytes=c.max_input_bytes))
        return self._pool

    def _generate(self, fmt: str, spec: DocumentSpec, limits: dict[str, Any]) -> tuple[bytes, str]:
        if fmt in WORKER_GEN_FORMATS:
            pool = self._get_pool()
            try:
                res = pool.run(f"gen_{fmt}", b"",
                               args={"spec": _spec_dict(spec), "limits": limits},
                               timeout=self.config.worker_op_timeout_seconds)
            except ParserIsolationError as exc:
                raise DocumentGenUnavailableError(f"generation worker unavailable: {exc}") from exc
            if not res.ok:
                if "not installed" in res.error or "ImportError" in res.error:
                    raise DocumentGenUnavailableError(
                        f"generation dependency unavailable in worker: {res.error}")
                raise DocumentGenError(f"generation worker failed: {res.error}")
            if not res.output:
                raise DocumentGenError("generation worker returned no bytes")
            gen_name = {"pdf": "reportlab", "docx": "python-docx", "xlsx": "openpyxl"}[fmt]
            return res.output, gen_name
        if fmt == "csv":
            data, note = gen_csv_bytes(spec, limits)
            return data, f"stdlib-csv ({note})"
        if fmt == "json":
            return gen_json_bytes(spec, limits), "stdlib-json"
        return gen_text_bytes(spec, fmt, limits), "stdlib-text"

    def prepare(self, *, format: str, destination: str, spec: DocumentSpec,
                agent: str, run_id: str | None = None, task_id: str | None = None,
                lease_owner: str | None = None, lease_generation: int | None = None,
                max_output_bytes: int | None = None) -> PreparedArtifact:
        """Validate -> generate once -> stage -> hash -> bind approval. No filesystem
        effect at the destination. Returns the pending artifact."""
        if not self.config.enabled:
            raise DocumentGenError("document generation is disabled (documents.enabled)")
        if not isinstance(spec, DocumentSpec):
            # gateway path: validate_args() model_dumps -> plain dict; revalidate strictly
            try:
                spec = DocumentSpec.model_validate(spec)
            except ValidationError as exc:
                raise DocumentSpecError(f"invalid document spec: {exc}") from exc
        if format not in GEN_FORMATS:
            raise DocumentSpecError(f"unsupported generation format {format!r}")
        ext = FORMAT_EXT[format]
        if not destination.lower().endswith(ext):
            raise DocumentSpecError(f"destination {destination!r} must end with {ext!r}")
        if destination.startswith(("/", "\\")) or ".." in destination.replace("\\", "/").split("/"):
            raise DocumentSpecError("destination must be a workspace-relative path")
        limits = self._limits()
        eff_max_out = min(max_output_bytes or limits["max_output_bytes"],
                          limits["max_output_bytes"])
        data, generator = self._generate(format, spec, limits)
        if len(data) > eff_max_out:
            raise DocumentSpecError(
                f"generated artifact {len(data)}B exceeds cap {eff_max_out}B")
        artifact_sha = hashlib.sha256(data).hexdigest()
        spec_sha = hashlib.sha256(canonical_args({"spec": _spec_dict(spec), "format": format,
                                                  "destination": destination}).encode()).hexdigest()
        # destination precondition (checked BEFORE generating/staging so approvals
        # are never burned on doomed destinations): publication is CREATION-ONLY —
        # an existing destination means overwrite, and overwrite is DISABLED (a
        # rename cannot atomically bind the pre-existing content hash); the
        # atomic no-clobber link in write_bytes_atomic is the authoritative check
        dest_resolved = self.sandbox.resolve(destination)
        lexical = os.path.normpath(os.path.join(self.sandbox.root, destination))
        if os.path.lexists(lexical):
            lst = os.lstat(lexical)
            if stat.S_ISLNK(lst.st_mode):
                raise DocumentGenError(f"destination {destination!r} is a symlink — rejected")
            if not stat.S_ISREG(lst.st_mode):
                raise DocumentGenError(f"destination {destination!r} is not a regular file")
            if lst.st_nlink > 1:
                raise DocumentGenError(
                    f"destination {destination!r} has {lst.st_nlink} hard links — rejected")
            raise DocumentGenError(
                f"destination {destination!r} already exists — publication is "
                "creation-only (overwrite disabled; secure no-clobber creation only)")
        if os.path.lexists(dest_resolved):
            raise DocumentGenError(
                f"destination {destination!r} already exists — publication is creation-only")
        # bounded staging: sweep expired orphans (protecting live approvals), then
        # enforce the operator caps on staged artifact count and aggregate bytes
        self._sweep_staging()
        staging_rel, stage_sha = self.sandbox.stage_bytes(
            data, max_bytes=eff_max_out, max_files=self.config.staging_max_files,
            max_total_bytes=self.config.staging_max_bytes,
            max_age_seconds=self.config.staging_max_age_seconds)
        if stage_sha != artifact_sha:
            self.sandbox.discard_staged(staging_rel)
            raise DocumentGenError("staged artifact hash mismatch — discarded")
        args = {"format": format, "destination": destination, "spec_sha256": spec_sha,
                "artifact_sha256": artifact_sha, "byte_count": len(data)}
        ctx = ToolExecutionContext(agent=agent, run_id=run_id, task_id=task_id)
        req = self.approval_manager.create(
            ctx=ctx, spec=self._tool_spec, args=args, operation="publish_artifact",
            target=destination,
            preconditions={
                "output_sha256": artifact_sha, "byte_count": len(data),
                "spec_sha256": spec_sha, "staging_relpath": staging_rel,
                "expected_absent": True, "format": format,
                "generator": generator,
            })
        return PreparedArtifact(
            approval_id=req.approval_id, staging_relpath=staging_rel,
            artifact_sha256=artifact_sha, byte_count=len(data), spec_sha256=spec_sha,
            destination=destination, format=format, generator=generator)

    def publish(self, *, approval_id: str, agent: str, run_id: str,
                task_id: str | None = None, lease_owner: str | None = None,
                lease_generation: int | None = None) -> dict[str, Any]:
        """W5 residual: revalidate everything, re-hash the staged artifact, then
        ATOMICALLY consume the approval AND start the durable execution-ledger row
        in ONE transaction (store.start_execution — never a standalone consume),
        publish the exact approved bytes with an atomic no-clobber creation,
        verify the published hash (a mismatch is NEVER reported as success), mark
        the ledger row completed, discard staging, register provenance.

        Crash windows (documented, never auto-replayed — the approval is consumed
        atomically with the ledger start, so a retry is impossible by construction):
        - before start_execution: nothing happened; approval still approved.
        - after start_execution, before/during publication: ledger row stays
          'started' (uncertain), approval consumed, destination may or may not
          exist with the approved bytes; recovery must NOT replay.
        - after publication, before ledger_update: same uncertain outcome."""
        if not self.config.enabled:
            raise DocumentGenError("document generation is disabled")
        if not run_id:
            raise DocumentGenError(
                "publication requires a run-bound lease (run_id) — fail closed")
        row = self.store.get_approval(approval_id)
        if row is None:
            raise DocumentGenError(f"unknown approval {approval_id!r}")
        if row["status"] != "approved":
            raise DocumentGenError(f"approval {approval_id!r} is {row['status']}, not approved")
        if row["expires_at"] <= self.store.now_iso():
            raise DocumentGenError("approval expired — regenerate and re-approve (fail closed)")
        if row["run_id"] != run_id:
            raise DocumentGenError("approval belongs to a different run — rejected")
        # the durable bindings (ledger + provenance) use the APPROVAL-BOUND agent;
        # who may invoke doc_publish at all is gated by ToolPolicy, and the run
        # lease owner+generation gates which worker may publish — so a supervisor
        # may publish an artifact prepared by another agent of the SAME run, but
        # never cross-run or cross-lease
        bound_agent = row["agent"] or agent
        pre = json.loads(row["preconditions_json"] or "{}")
        out_sha = pre.get("output_sha256")
        if not out_sha:
            raise DocumentGenError("approval is not bound to an exact artifact hash")
        # lease fencing: current owner+generation must hold the run
        with self.store._lock:
            if not self.store._verify_lease_locked(run_id, lease_owner, lease_generation):
                raise DocumentGenError("stale worker: lease fencing failed at publication")
        # re-hash the staged artifact: substitution after approval is rejected
        staged = self.sandbox.read_staged(pre["staging_relpath"],
                                          max_bytes=int(pre["byte_count"]) + 1)
        if hashlib.sha256(staged).hexdigest() != out_sha:
            self.sandbox.discard_staged(pre["staging_relpath"])  # corrupt: never publishable
            raise DocumentGenError("staged artifact does not match the approved hash — rejected")
        if len(staged) != int(pre["byte_count"]):
            raise DocumentGenError("staged artifact byte count mismatch — rejected")
        dest = row["target"]
        # destination preconditions re-checked at publication time (BEFORE the
        # approval is consumed, so a substitution never burns the approval)
        dest_resolved = self.sandbox.resolve(dest)
        lexical = os.path.normpath(os.path.join(self.sandbox.root, dest))
        if os.path.lexists(lexical):
            lst = os.lstat(lexical)
            if stat.S_ISLNK(lst.st_mode) or not stat.S_ISREG(lst.st_mode) or lst.st_nlink > 1:
                raise DocumentGenError(
                    f"destination {dest!r} is a symlink/hardlink/special file — rejected")
            raise DocumentGenError(
                f"destination {dest!r} appeared since approval — creation-only "
                "publication refuses to clobber (fail closed)")
        if os.path.lexists(dest_resolved):
            raise DocumentGenError(f"destination {dest!r} exists — creation-only (fail closed)")
        # canonical args digest binding
        args = {"format": pre.get("format"), "destination": dest,
                "spec_sha256": pre.get("spec_sha256"), "artifact_sha256": out_sha,
                "byte_count": int(pre["byte_count"])}
        if row["args_sha256"] != args_sha256(args):
            raise DocumentGenError("approval args digest mismatch — substitution rejected")
        # ATOMIC start-before-effect: consume the approval AND insert the ledger
        # row in ONE transaction, fenced by lease owner+generation, bound to the
        # exact args hash / tool version / run / agent / tool / task / expiry
        execution_id = uuid4().hex
        args_json = ("" if getattr(self.store.persist_mode, "value", "") == "metadata_only"
                     else redact_secrets(canonical_args(args))[:8_000])
        if not self.store.start_execution(
            execution_id=execution_id, run_id=run_id, agent=bound_agent, tool=row["tool"],
            args_sha256=row["args_sha256"], status=ToolExecutionStatus.STARTED.value,
            approval_id=approval_id, expected_args_sha256=row["args_sha256"],
            task_id=task_id, tool_version=row["tool_version"], args_json=args_json,
            lease_owner=lease_owner, lease_generation=lease_generation,
            expected_task_id=row["task_id"],
        ):
            raise DocumentGenError(
                "atomic execution-start rejected (replay, substitution, expiry or stale "
                "lease) — approval NOT consumed, nothing published")
        # publish the exact approved bytes (parent dir-fd + staged-fd re-verify +
        # ATOMIC NO-CLOBBER link + post-publish re-open/re-hash)
        published_sha = self.sandbox.write_bytes_atomic(dest, staged,
                                                        max_bytes=int(pre["byte_count"]))
        if published_sha != out_sha:
            raise DocumentGenError(
                "post-publication hash mismatch — NOT reported as successful completion")
        # durable completion, fenced by the same lease (a failure here leaves an
        # uncertain 'started' row; the consumed approval makes auto-replay impossible)
        if not self.store.ledger_update(
            execution_id, status=ToolExecutionStatus.COMPLETED.value,
            result_sha256=published_sha, output_bytes=len(staged),
            lease_owner=lease_owner, lease_generation=lease_generation,
        ):
            raise DocumentGenError(
                "published but durable completion failed — uncertain outcome, no auto-replay")
        self.sandbox.discard_staged(pre["staging_relpath"])
        # artifact provenance: clearly marked GENERATED (never external evidence)
        prov = {"generated": True, "generator": pre.get("generator", ""), "format": pre["format"],
                "spec_sha256": pre["spec_sha256"], "output_sha256": out_sha,
                "byte_count": int(pre["byte_count"]), "destination": dest,
                "approval_id": approval_id, "execution_id": execution_id,
                "note": "generated artifact — not retrieved evidence"}
        origin = self.registry.trusted_origin(
            run_id=run_id, lease_owner=lease_owner,  # type: ignore[arg-type]
            lease_generation=lease_generation,      # type: ignore[arg-type]
            agent_id=bound_agent, task_id=task_id)
        source_id = self.registry.register(
            origin=origin, source_type=SourceType.OTHER, provider="hive-docgen",
            locator=f"generated:{dest}", content_sha256=out_sha,
            extract_method=str(pre.get("generator", "")), extract_version="1",
            provenance=prov, license=None, attribution="")
        return {"status": "published", "destination": dest, "artifact_sha256": published_sha,
                "byte_count": len(staged), "approval_id": approval_id,
                "execution_id": execution_id, "source_id": source_id, "format": pre["format"]}


# ---- tools -----------------------------------------------------------------------------------

class DocGenerateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format: Literal["pdf", "docx", "xlsx", "csv", "json", "txt", "md"]
    destination: str = Field(pattern=r"^[^/\\\x00-\x1f][^\x00-\x1f]{0,511}$")
    spec: DocumentSpec
    max_output_bytes: int | None = Field(default=None, ge=1, le=100 << 20)


class DocPublishInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approval_id: str = Field(pattern=r"^[0-9a-f]{32}$")


class DocGenerateAdapter:
    def __init__(self, writer: DocumentWriter, spec: ToolSpec, *,
                 lease_lookup: Callable[[str], Any] | None = None) -> None:
        self.writer = writer
        self.spec = spec
        self._lease_lookup = lease_lookup

    def _lease(self, ctx: Any):
        run_id = getattr(ctx, "run_id", None)
        if run_id is None:
            return None, None, None
        rc = self._lease_lookup(run_id) if self._lease_lookup else None
        owner = getattr(rc, "lease_owner", None) if rc is not None else None
        gen = getattr(rc, "lease_generation", None) if rc is not None else None
        if owner is None or gen is None:
            from .errors import ToolPermissionError

            raise ToolPermissionError(f"run {run_id!r} has no lease identity — fail closed")
        return run_id, owner, gen

    def execute(self, args: dict[str, Any], ctx: Any) -> str:
        run_id, owner, gen = self._lease(ctx)
        art = self.writer.prepare(
            format=args["format"], destination=args["destination"], spec=args["spec"],
            agent=getattr(ctx, "agent", "system"), run_id=run_id,
            task_id=getattr(ctx, "task_id", None), lease_owner=owner, lease_generation=gen,
            max_output_bytes=args.get("max_output_bytes"))
        out = {"status": art.status, "approval_id": art.approval_id,
               "artifact_sha256": art.artifact_sha256, "byte_count": art.byte_count,
               "destination": art.destination, "format": art.format,
               "generator": art.generator,
               "note": ("artifact staged and bound to approval; publish with doc_publish "
                        "after operator approval; the approved bytes are never regenerated")}
        return redact_secrets(json.dumps(out, default=str))

    async def aexecute(self, args: dict[str, Any], ctx: Any) -> str:
        import asyncio

        return await asyncio.to_thread(self.execute, args, ctx)


class DocPublishAdapter:
    def __init__(self, writer: DocumentWriter, spec: ToolSpec, *,
                 lease_lookup: Callable[[str], Any] | None = None) -> None:
        self.writer = writer
        self.spec = spec
        self._lease_lookup = lease_lookup

    def execute(self, args: dict[str, Any], ctx: Any) -> str:
        # W5 residual: publication is a run-bound mutation — a full lease identity
        # (owner+generation) is mandatory, fail closed without it
        run_id = getattr(ctx, "run_id", None)
        if not run_id:
            raise DocumentGenError(
                "doc_publish requires a run-bound context (run_id) — fail closed")
        owner = gen = None
        if self._lease_lookup is not None:
            rc = self._lease_lookup(run_id)
            owner = getattr(rc, "lease_owner", None) if rc is not None else None
            gen = getattr(rc, "lease_generation", None) if rc is not None else None
        if owner is None or gen is None:
            from .errors import ToolPermissionError

            raise ToolPermissionError(f"run {run_id!r} has no lease identity — fail closed")
        res = self.writer.publish(
            approval_id=args["approval_id"], agent=getattr(ctx, "agent", "system"),
            run_id=run_id, task_id=getattr(ctx, "task_id", None),
            lease_owner=owner, lease_generation=gen)
        return redact_secrets(json.dumps(res, default=str))

    async def aexecute(self, args: dict[str, Any], ctx: Any) -> str:
        import asyncio

        return await asyncio.to_thread(self.execute, args, ctx)


def docgen_tools(writer: DocumentWriter, *, lease_lookup: Callable[[str], Any] | None = None
                 ) -> list[tuple[ToolSpec, Any]]:
    gen_spec = ToolSpec(
        name="doc_generate", version="1", category="documents",
        side_effect=SideEffect.LOCAL_MUTATION,
        # exact-byte approval is managed inside via the shared ApprovalManager
        # (bound to the generated artifact hash); the gateway still enforces
        # policy, budgets, ledger and lease fencing around the call
        requires_approval=False,
        description=("Prepare ONE generated document (structured spec, bounded) — "
                     "generates once, stages securely, binds an exact-byte approval. "
                     "No destination write happens here."),
        input_model=DocGenerateInput, max_output_bytes=200_000)
    pub_spec = ToolSpec(
        name="doc_publish", version="1", category="documents",
        side_effect=SideEffect.LOCAL_MUTATION, requires_approval=False,
        description=("Publish the exact artifact bytes bound to an approved, "
                     "unexpired, unconsumed approval (re-hashed before publication; "
                     "destination preconditions and lease revalidated)."),
        input_model=DocPublishInput)
    return [(gen_spec, DocGenerateAdapter(writer, gen_spec, lease_lookup=lease_lookup)),
            (pub_spec, DocPublishAdapter(writer, pub_spec, lease_lookup=lease_lookup))]
