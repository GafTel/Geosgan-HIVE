"""W4 (v0.8.0): local document READING engine.

Mandatory isolation: PDF / DOCX / XLSX (complex, untrusted, library-parsed) are
parsed ONLY inside the trusted parser-worker subprocesses
(`parser_isolation.ParserWorkerPool` + `_parser_worker` ops). The main process
sandbox-reads bounded bytes (WorkspaceSandbox: O_NOFOLLOW, hard-link refusal,
fd re-verification), hashes them, sends ONLY those bytes to the worker (never
filesystem paths), and parses only the bounded JSON result. CSV/JSON/TXT/MD/HTML
use existing bounded stdlib parsers in-process (sanctioned by the W4 brief).

Archive/XML security (DOCX/XLSX are ZIP+XML): before any library processing,
`check_zip` enforces archive size, entry count, per-entry and total expanded
size (ACTUAL decompression, not central-directory declarations), compression
ratios, canonical non-traversal entry names, duplicate rejection, encrypted and
unsupported-compression rejection; `xml_prescan` rejects DTD/ENTITY declarations
and external relationships and re-parses every XML entry with defusedxml
(defusedxml alone is NOT treated as proof — the keyword scan + caps + worker
rlimits are layered on top; lxml/openpyxl indirect paths are covered by the
prescan + rlimits). Nothing is extracted to the filesystem.

Provenance: successful parses register a W2 source (lease-fenced trusted
origin) binding raw sha256 (of the exact bytes parsed — the worker echoes it
back and the parent verifies), extracted-content sha256, format, parser
name+version, requested vs actual scope, page/block/sheet/cell provenance,
byte counts, truncation and unsupported-content indicators. Failed parses
register NOTHING. Metadata-only by default (document contents are never
persisted). Partial extraction is explicitly labeled truncated.

Fail closed: missing optional dependencies -> DocumentUnavailableError; non-POSIX
platforms (Windows) -> DocumentUnavailableError for worker formats (parser
isolation is POSIX-only; no timeout-only insecure mode); unsupported formats,
encrypted/malformed documents -> typed DocumentParseError. No OCR, no macro
execution, no formula evaluation, no external relationships, no embedded
objects, no resource fetching.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .agent_config import DocumentsConfig
from .errors import HiveError
from .parser_isolation import ParserIsolationError, ParserResourceLimits, ParserWorkerPool
from .sources import SourceRegistry, SourceType
from .tool_gateway import redact_secrets
from .tool_spec import SideEffect, ToolSpec

SUPPORTED_FORMATS = ("pdf", "docx", "xlsx", "csv", "json", "txt", "md", "html")
WORKER_FORMATS = ("pdf", "docx", "xlsx")  # parsed ONLY in the isolated worker


class DocumentError(HiveError):
    """Base for document-reading failures (all fail closed)."""


class DocumentUnavailableError(DocumentError):
    """Format unsupported here: missing optional dependency, non-POSIX parser
    isolation, or disabled by policy — fail closed."""


class DocumentParseError(DocumentError):
    """Malformed/encrypted/unsupported/mismatched document or hash mismatch."""


class ZipGuardError(DocumentParseError):
    """Archive violates the enforced zip/XML security caps."""


@dataclass(frozen=True)
class ZipCaps:
    max_archive_bytes: int = 50 << 20
    max_entries: int = 10_000
    max_entry_bytes: int = 100 << 20
    max_total_expanded: int = 200 << 20
    max_ratio: int = 100  # expanded/compressed per entry


_ENTRY_BAD = re.compile(r"(^/|\\|^[A-Za-z]:|[\x00-\x1f])")


def _entry_name_ok(name: str) -> bool:
    if not name or _ENTRY_BAD.search(name):
        return False
    return not any(seg == ".." for seg in name.replace("\\", "/").split("/"))


def check_zip(data: bytes, caps: ZipCaps) -> list[zipfile.ZipInfo]:
    """Enforce archive security with ACTUAL decompression bounds (never trust the
    central directory alone). Raises ZipGuardError on any violation."""
    if len(data) > caps.max_archive_bytes:
        raise ZipGuardError(f"archive {len(data)}B exceeds cap {caps.max_archive_bytes}B")
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ZipGuardError(f"malformed archive: {exc}") from exc
    with zf:
        infos = zf.infolist()
        if len(infos) > caps.max_entries:
            raise ZipGuardError(f"entry count {len(infos)} exceeds {caps.max_entries}")
        seen: set[str] = set()
        total = 0
        for info in infos:
            name = info.filename
            if not _entry_name_ok(name):
                raise ZipGuardError(f"unsafe entry name {name!r} (traversal/absolute/control)")
            if name in seen:
                raise ZipGuardError(f"duplicate entry name {name!r}")
            seen.add(name)
            if info.flag_bits & 0x1:
                raise ZipGuardError(f"encrypted entry {name!r} rejected")
            if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise ZipGuardError(
                    f"unsupported compression in {name!r}: {info.compress_type}")
            if info.file_size > caps.max_entry_bytes:
                raise ZipGuardError(f"entry {name!r} declared size {info.file_size} exceeds cap")
            if info.compress_size > 0 and info.file_size > info.compress_size * caps.max_ratio:
                raise ZipGuardError(f"entry {name!r} compression ratio exceeds {caps.max_ratio}x")
            total += info.file_size
            if total > caps.max_total_expanded:
                raise ZipGuardError("total expanded size exceeds cap (decompression bomb)")
        # ACTUAL decompression, bounded (declarations are not trusted alone)
        remaining = caps.max_total_expanded
        for info in infos:
            with zf.open(info) as fh:
                while True:
                    chunk = fh.read(min(1 << 20, remaining + 1))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    if remaining < 0:
                        raise ZipGuardError(f"entry {info.filename!r} exceeds actual expanded cap")
    return infos


_XML_FORBIDDEN = (b"<!DOCTYPE", b"<!doctype", b"<!ENTITY", b"<!entity")


def zip_keyword_prescan(data: bytes, caps: ZipCaps) -> None:
    """Pure-stdlib prescan (runs in the PARENT too, defense in depth): reject
    DTD/ENTITY declarations and external relationships in any XML entry."""
    zf = zipfile.ZipFile(io.BytesIO(data))
    with zf:
        for info in zf.infolist():
            low = info.filename.lower()
            if not (low.endswith(".xml") or low.endswith(".rels")):
                continue
            with zf.open(info) as fh:
                content = fh.read(caps.max_entry_bytes + 1)
            if len(content) > caps.max_entry_bytes:
                raise ZipGuardError(f"XML entry {info.filename!r} too large")
            for kw in _XML_FORBIDDEN:
                if kw in content:
                    raise ZipGuardError(
                        f"DTD/entity declaration forbidden in {info.filename!r}")
            if low.endswith(".rels") and b'TargetMode="External"' in content:
                # only actual external relationships are rejected; namespace
                # URIs (http://schemas...) are not relationships
                raise ZipGuardError(f"external relationship forbidden in {info.filename!r}")


def xml_defused_prescan(data: bytes, caps: ZipCaps) -> None:
    """Worker-side: keyword prescan + defusedxml parse of every XML entry.
    Installing defusedxml alone is not treated as proof; this is one layer among
    keyword scan + zip caps + worker rlimits."""
    try:
        from defusedxml import ElementTree as DET
        from defusedxml.common import (
            DefusedXmlException,
        )
    except ImportError as exc:
        raise DocumentUnavailableError(
            "defusedxml not installed — XML prescan unavailable") from exc
    zip_keyword_prescan(data, caps)
    zf = zipfile.ZipFile(io.BytesIO(data))
    with zf:
        for info in zf.infolist():
            if not info.filename.lower().endswith((".xml", ".rels")):
                continue
            with zf.open(info) as fh:
                content = fh.read(caps.max_entry_bytes + 1)
            try:
                DET.fromstring(content)
            except DefusedXmlException as exc:
                raise ZipGuardError(
                    f"defusedxml rejected {info.filename!r}: {exc}") from exc
            except Exception as exc:
                raise ZipGuardError(
                    f"malformed XML in {info.filename!r}: {type(exc).__name__}") from exc


def detect_format(filename: str, data: bytes) -> str:
    """Magic + extension detection; extension/content mismatch is a typed error."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext in ("docm", "xlsm", "pptm", "dotm", "xltm"):
        raise DocumentParseError(f"macro-enabled format .{ext} rejected (never executed)")
    if data[:5] == b"%PDF-":
        if ext not in ("", "pdf"):
            raise DocumentParseError(f"extension .{ext} but content is PDF")
        return "pdf"
    if data[:4] == b"PK\x03\x04":
        if ext not in ("", "docx", "xlsx"):
            raise DocumentParseError(f"extension .{ext} but content is a ZIP-based document")
        # distinguish by archive layout (central directory only, no decompression)
        try:
            names = set(zipfile.ZipFile(io.BytesIO(data)).namelist())
        except zipfile.BadZipFile as exc:
            raise DocumentParseError(f"malformed archive: {exc}") from exc
        if ext == "docx" or (not ext and "word/document.xml" in names):
            return "docx"
        if ext == "xlsx" or (not ext and "xl/workbook.xml" in names):
            return "xlsx"
        raise DocumentParseError("ZIP archive is neither DOCX nor XLSX")
    if ext in ("docx", "xlsx"):
        raise DocumentParseError(f"extension .{ext} but content is not a ZIP archive")
    if ext == "pdf":
        raise DocumentParseError("extension .pdf but content is not a PDF")
    if ext == "json":
        stripped = data.lstrip()[:1]
        if stripped not in (b"{", b"["):
            raise DocumentParseError("extension .json but content is not JSON")
        return "json"
    if ext in ("csv", "txt", "md", "html", ""):
        return ext or "txt"
    raise DocumentParseError(f"unsupported format extension .{ext!r}")


# ---- worker-side parsers (run ONLY inside the isolated parser worker) ---------------

def _missing(dep: str) -> dict[str, Any]:
    return {"error": "missing_dependency", "dep": dep}


def _parse_pages_spec(spec: str | None, n_pages: int, max_pages: int) -> list[int]:
    if not spec or spec == "all":
        return list(range(1, min(n_pages, max_pages) + 1))
    pages: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            lo, hi = int(a), int(b)
            if lo < 1 or hi < lo:
                raise DocumentParseError(f"invalid page range {part!r}")
            pages.extend(range(lo, min(hi, n_pages) + 1))
        else:
            p = int(part)
            if p < 1 or p > n_pages:
                raise DocumentParseError(f"page {p} out of range (1..{n_pages})")
            pages.append(p)
    if not pages:
        raise DocumentParseError("empty page selection")
    if len(pages) > max_pages:
        raise DocumentParseError(f"page selection exceeds max {max_pages}")
    return sorted(set(pages))


def worker_parse_pdf(data: bytes, scope: dict[str, Any], limits: dict[str, Any]) -> dict[str, Any]:
    """pypdf bounded text extraction. No JS/actions/attachments/links are executed;
    no OCR — image-only pages are reported as having no extractable text."""
    try:
        import pypdf
        from pypdf.errors import PdfReadError
    except ImportError:
        return _missing("pypdf")
    raw_sha = hashlib.sha256(data).hexdigest()
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            return {"error": "encrypted", "raw_sha256": raw_sha}
        n_pages = len(reader.pages)
        max_pages = int(limits.get("max_pdf_pages", 50))
        max_chars = int(limits.get("max_extract_chars", 20_000))
        selected = _parse_pages_spec(scope.get("pages"), n_pages, max_pages)
        page_prov: list[dict[str, Any]] = []
        parts: list[str] = []
        total = 0
        truncated = len(selected) < n_pages
        for pno in selected:
            try:
                txt = reader.pages[pno - 1].extract_text() or ""
            except Exception:
                txt = ""
                page_prov.append({"page": pno, "chars": 0, "note": "extraction error on this page"})
                continue
            if not txt.strip():
                page_prov.append({"page": pno, "chars": 0,
                                  "note": "no extractable text (image-only/empty; NOT OCR)"})
                continue
            take = txt if total + len(txt) <= max_chars else txt[: max(0, max_chars - total)]
            if len(take) < len(txt):
                truncated = True
            parts.append(f"[page {pno}]\n{take}")
            total += len(take)
            page_prov.append({"page": pno, "chars": len(take)})
            if total >= max_chars:
                truncated = True
                break
        meta: dict[str, Any] = {}
        try:
            md = reader.metadata or {}
            for k in ("/Title", "/Author", "/Subject", "/Creator", "/Producer"):
                v = md.get(k)
                if v is not None:
                    meta[str(k).lstrip("/")] = str(v)[:300]
        except Exception:
            pass
        text = "\n".join(parts)
        return {"format": "pdf", "parser": f"pypdf {pypdf.__version__}",
                "page_count": n_pages, "pages": page_prov, "pages_selected": selected,
                "text": text, "metadata": meta, "truncated": truncated,
                "unsupported": ["embedded_actions", "attachments", "ocr", "annotations"],
                "raw_sha256": raw_sha, "extract_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "bytes_in": len(data)}
    except PdfReadError as exc:
        return {"error": "malformed_pdf", "detail": str(exc)[:200], "raw_sha256": raw_sha}
    except DocumentParseError as exc:
        return {"error": "bad_scope", "detail": str(exc)[:200], "raw_sha256": raw_sha}
    except Exception as exc:
        return {"error": "parse_failed", "detail": type(exc).__name__, "raw_sha256": raw_sha}


def worker_parse_docx(data: bytes, scope: dict[str, Any], limits: dict[str, Any]) -> dict[str, Any]:
    """python-docx after zip guard + defusedxml prescan. Paragraphs, headings,
    tables in document order. Unsupported (documented, not extracted): text
    boxes, tracked changes, complex layout, embedded objects, macros (.docm
    rejected by extension/format)."""
    try:
        import docx  # noqa: F401
    except ImportError:
        return _missing("python-docx")
    raw_sha = hashlib.sha256(data).hexdigest()
    caps = ZipCaps(max_archive_bytes=int(limits.get("max_input_bytes", 10 << 20)) * 2,
                   max_total_expanded=400 << 20)
    try:
        check_zip(data, caps)
        xml_defused_prescan(data, caps)
    except (ZipGuardError, DocumentUnavailableError) as exc:
        return {"error": "archive_rejected", "detail": str(exc)[:300], "raw_sha256": raw_sha}
    try:
        import docx

        doc = docx.Document(io.BytesIO(data))
        max_blocks = int(limits.get("max_docx_blocks", 2000))
        max_chars = int(limits.get("max_extract_chars", 20_000))
        blocks: list[dict[str, Any]] = []
        parts: list[str] = []
        total = 0
        truncated = False
        n = 0
        for p in doc.paragraphs:
            n += 1
            if n > max_blocks:
                truncated = True
                break
            style = (p.style.name if p.style is not None else "") or ""
            txt = p.text or ""
            if not txt.strip() and not style.startswith("Heading"):
                continue
            kind = "heading" if style.startswith("Heading") else "paragraph"
            take = txt if total + len(txt) <= max_chars else txt[: max(0, max_chars - total)]
            if len(take) < len(txt):
                truncated = True
            blocks.append({"type": kind, "index": n, "style": style[:64], "chars": len(take)})
            parts.append(f"[{kind} {n}] {take}")
            total += len(take)
            if total >= max_chars:
                truncated = True
                break
        n_tables = 0
        for tbl in doc.tables:
            n_tables += 1
            if n_tables > max_blocks:
                truncated = True
                break
            rows_out = []
            for row in tbl.rows[:50]:
                cells = [c.text[:200] for c in row.cells[:20]]
                rows_out.append(cells)
                total += sum(len(c) for c in cells)
                if total >= max_chars:
                    truncated = True
                    break
            blocks.append({"type": "table", "index": n_tables, "rows": len(rows_out)})
            blob = json.dumps(rows_out, ensure_ascii=False)
            parts.append(f"[table {n_tables}] " + blob[: max(0, max_chars - total)])
            if truncated:
                break
        meta = {}
        try:
            cp = doc.core_properties
            for k in ("title", "author", "subject", "created"):
                v = getattr(cp, k, None)
                if v is not None:
                    meta[k] = str(v)[:300]
        except Exception:
            pass
        text = "\n".join(parts)
        docx_ver = getattr(docx, "__version__", "1.x")
        return {"format": "docx", "parser": f"python-docx {docx_ver}",
                "blocks": blocks[:500], "text": text, "metadata": meta, "truncated": truncated,
                "unsupported": ["text_boxes", "tracked_changes", "complex_layout",
                                "embedded_objects", "macros"],
                "raw_sha256": raw_sha, "extract_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "bytes_in": len(data)}
    except DocumentParseError as exc:
        return {"error": "bad_scope", "detail": str(exc)[:200], "raw_sha256": raw_sha}
    except Exception as exc:
        return {"error": "parse_failed", "detail": type(exc).__name__, "raw_sha256": raw_sha}


def worker_parse_xlsx(data: bytes, scope: dict[str, Any], limits: dict[str, Any]) -> dict[str, Any]:
    """openpyxl read-only, bounded. Formulas are NEVER evaluated: formula cells
    report the expression + cached value (or explicit missing-cached); literal
    cells report typed values; missing stays missing (never zero-filled).
    Declared dimensions are not trusted as a bound — iteration is capped."""
    try:
        import openpyxl
    except ImportError:
        return _missing("openpyxl")
    raw_sha = hashlib.sha256(data).hexdigest()
    caps = ZipCaps(max_archive_bytes=int(limits.get("max_input_bytes", 10 << 20)) * 2,
                   max_total_expanded=400 << 20)
    try:
        check_zip(data, caps)
        xml_defused_prescan(data, caps)
    except (ZipGuardError, DocumentUnavailableError) as exc:
        return {"error": "archive_rejected", "detail": str(exc)[:300], "raw_sha256": raw_sha}
    try:
        import openpyxl

        max_sheets = int(limits.get("max_xlsx_sheets", 10))
        max_cells = int(limits.get("max_xlsx_cells", 20_000))
        max_chars = int(limits.get("max_extract_chars", 20_000))
        wb_f = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=False)
        wb_v = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        names = list(wb_f.sheetnames)[:max_sheets]
        requested = scope.get("sheets") or []
        if requested:
            names = [n for n in names if n in set(requested)][:max_sheets]
        sheets_out: list[dict[str, Any]] = []
        parts: list[str] = []
        total = 0
        truncated = False
        formulas = 0
        missing_cached = 0
        for name in names:
            if name not in wb_f.sheetnames:
                continue
            ws_f, ws_v = wb_f[name], wb_v[name]
            rows_out: list[list[Any]] = []
            coords: list[str] = []
            count = 0
            max_row = int(limits.get("max_xlsx_rows", 1000))
            max_col = int(limits.get("max_xlsx_cols", 100))
            from openpyxl.utils import get_column_letter

            for ri, (rf, rv) in enumerate(zip(
                    ws_f.iter_rows(max_row=max_row, max_col=max_col),
                    ws_v.iter_rows(max_row=max_row, max_col=max_col),
                    strict=True), start=1):
                row_out = []
                for ci, (cf, cv) in enumerate(zip(rf, rv, strict=True), start=1):
                    count += 1
                    if count > max_cells:
                        truncated = True
                        break
                    # read_only mode yields EmptyCell (no .coordinate) for blank cells
                    coord = getattr(cf, "coordinate", None) or f"{get_column_letter(ci)}{ri}"
                    if cf.data_type == "f":  # formula: never evaluated
                        formulas += 1
                        cached = cv.value
                        if cached is None:
                            missing_cached += 1
                        row_out.append({"cell": coord, "formula": str(cf.value)[:200],
                                        "cached": None if cached is None else str(cached)[:200],
                                        "cached_missing": cached is None})
                    else:
                        v = cf.value
                        row_out.append({"cell": coord, "value": None if v is None else str(v)[:200],
                                        "value_missing": v is None})
                    coords.append(coord)
                    if total >= max_chars:
                        truncated = True
                        break
                rows_out.append(row_out)
                total += 1
                if truncated:
                    break
            sheets_out.append({"sheet": name, "rows": rows_out, "cells_seen": count,
                               "sample_coords": coords[:50]})
            parts.append(f"[sheet {name}] " + json.dumps(rows_out, ensure_ascii=False, default=str))
            if truncated or total >= max_chars:
                truncated = True
                break
        text = "\n".join(parts)[:max_chars]
        return {"format": "xlsx", "parser": f"openpyxl {openpyxl.__version__}",
                "sheets": sheets_out, "sheet_names": list(wb_f.sheetnames),
                "text": text, "truncated": truncated,
                "formula_cells": formulas, "missing_cached_values": missing_cached,
                "unsupported": ["macro_workbooks(.xlsm)", "external_links", "evaluated_formulas"],
                "raw_sha256": raw_sha, "extract_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "bytes_in": len(data)}
    except DocumentParseError as exc:
        return {"error": "bad_scope", "detail": str(exc)[:200], "raw_sha256": raw_sha}
    except Exception as exc:
        return {"error": "parse_failed", "detail": type(exc).__name__, "raw_sha256": raw_sha}


# ---- main-process simple formats (bounded stdlib parsers) ----------------------------

def parse_text_doc(data: bytes, fmt: str, limits: dict[str, Any]) -> dict[str, Any]:
    max_chars = int(limits.get("max_extract_chars", 20_000))
    if fmt == "html":
        from .web_retrieval import html_to_text

        ex = html_to_text(data, charset="utf-8", max_chars=max_chars,
                          max_tags=int(limits.get("max_html_tags", 5000)),
                          deadline_s=float(limits.get("extraction_deadline_seconds", 10)))
        return {"format": "html", "parser": "stdlib-html.parser 1.0",
                "text": ex.text, "metadata": {"title": ex.title},
                "headings": ex.headings, "truncated": ex.truncated,
                "unsupported": ["javascript", "css_execution", "resource_fetching"]}
    text_full = data.decode("utf-8", "replace")
    truncated = len(text_full) > max_chars
    text = text_full[:max_chars]
    return {"format": fmt, "parser": "stdlib-text", "text": text,
            "metadata": {}, "truncated": truncated, "unsupported": []}


def parse_csv_doc(data: bytes, limits: dict[str, Any]) -> dict[str, Any]:
    max_rows = int(limits.get("max_csv_rows", 5000))
    max_field = int(limits.get("max_csv_field_bytes", 100_000))
    max_chars = int(limits.get("max_extract_chars", 20_000))
    input_bound = max_chars * 4 + 65536
    old = csv.field_size_limit()
    # the reader limit is the INPUT bound (memory safety); oversized FIELDS are
    # truncated to max_field in the output, never fatal, never unbounded
    csv.field_size_limit(input_bound)
    try:
        text = data.decode("utf-8", "replace")
        reader = csv.reader(io.StringIO(text[:input_bound]))
        rows: list[list[str]] = []
        header: list[str] = []
        truncated = len(text) > input_bound
        for i, row in enumerate(reader):
            if i == 0:
                header = [c[:max_field] for c in row[:100]]
                continue
            if len(rows) >= max_rows:
                truncated = True
                break
            cells = [c[:max_field] for c in row[:100]]
            if any(len(c) == max_field for c in row[:100]):
                truncated = True
            rows.append(cells)
        out_text = json.dumps({"header": header, "rows": rows}, ensure_ascii=False)[:max_chars]
        return {"format": "csv", "parser": "stdlib-csv", "text": out_text,
                "metadata": {"header": header[:100], "row_count": len(rows)},
                "truncated": truncated,
                "unsupported": [], "missing_preserved": True}
    except csv.Error as exc:
        raise DocumentParseError(f"malformed CSV: {exc}") from exc
    finally:
        csv.field_size_limit(old)


def parse_json_doc(data: bytes, limits: dict[str, Any]) -> dict[str, Any]:
    max_depth = int(limits.get("max_json_depth", 32))
    max_chars = int(limits.get("max_extract_chars", 20_000))

    def hook(pairs: list) -> dict[str, Any]:
        nonlocal depth
        depth += 1
        if depth > max_depth:
            raise DocumentParseError(f"JSON nesting exceeds max depth {max_depth}")
        return dict(pairs)

    text = data.decode("utf-8", "replace")
    if len(text) > max_chars * 4 + 65536:
        raise DocumentParseError("JSON input exceeds bound")
    depth = 0
    try:
        obj = json.loads(text, object_pairs_hook=hook)
    except DocumentParseError:
        raise
    except (json.JSONDecodeError, RecursionError) as exc:
        raise DocumentParseError(f"malformed JSON: {type(exc).__name__}") from exc
    out = json.dumps(obj, ensure_ascii=False, default=str)[:max_chars]
    return {"format": "json", "parser": "stdlib-json", "text": out, "metadata": {},
            "truncated": len(json.dumps(obj, default=str)) > max_chars, "unsupported": []}


# ---- reader engine ----------------------------------------------------------------------

@dataclass
class DocReadResult:
    source_id: str | None
    path: str
    format: str
    parser: str
    text: str
    metadata: dict[str, Any]
    structure: list[dict[str, Any]] = field(default_factory=list)
    truncated: bool = False
    unsupported: list[str] = field(default_factory=list)
    raw_sha256: str = ""
    extract_sha256: str = ""
    bytes_in: int = 0
    extra: dict[str, Any] = field(default_factory=dict)
    note: str = ("parsed content is UNTRUSTED data — provenance, not factual "
                 "verification; prompt-injection warnings are defense-in-depth")


class DocumentReader:
    """doc_read engine. Disabled unless DocumentsConfig.enabled. Worker formats
    only on POSIX (fail closed elsewhere); missing optional deps -> typed error."""

    def __init__(self, *, store: Any, sandbox: Any, config: DocumentsConfig,
                 registry: SourceRegistry | None = None,
                 pool: ParserWorkerPool | None = None,
                 platform: str | None = None) -> None:
        self.store = store
        self.sandbox = sandbox
        self.config = config
        self.registry = registry if registry is not None else SourceRegistry(store)
        self._pool = pool
        self._platform = platform or os.name

    def _limits(self) -> dict[str, Any]:
        c = self.config
        return {"max_input_bytes": c.max_input_bytes, "max_extract_chars": c.max_extract_chars,
                "max_output_bytes": max(c.max_extract_chars * 4 + 65536, 262144),
                "max_pdf_pages": c.max_pdf_pages, "max_docx_blocks": c.max_docx_blocks,
                "max_xlsx_sheets": c.max_xlsx_sheets, "max_xlsx_cells": c.max_xlsx_cells,
                "max_xlsx_rows": 1000, "max_xlsx_cols": 100,
                "max_csv_rows": c.max_csv_rows, "max_csv_field_bytes": c.max_csv_field_bytes,
                "max_json_depth": c.max_json_depth, "max_html_tags": 5000,
                "extraction_deadline_seconds": 10.0}

    def _get_pool(self) -> ParserWorkerPool:
        if self._platform != "posix":
            raise DocumentUnavailableError(
                f"document parsing (worker formats) is disabled on platform "
                f"{self._platform!r}: parser isolation requires POSIX rlimits + "
                "process groups; fail closed (Windows Job Objects = future path)")
        if self._pool is None:
            c = self.config
            self._pool = ParserWorkerPool(
                max_workers=c.worker_max_workers, max_pending=c.worker_max_pending,
                op_timeout_seconds=c.worker_op_timeout_seconds,
                limits=ParserResourceLimits(
                    max_memory_bytes=c.worker_max_memory_bytes,
                    max_cpu_seconds=int(c.worker_op_timeout_seconds),
                    max_output_file_bytes=max(c.max_extract_chars * 4 + 65536, 262144),
                    max_input_bytes=c.max_input_bytes,
                    max_response_bytes=4 << 20))
        return self._pool

    def _read_bytes_capped(self, relpath: str) -> bytes:
        cap = self.config.max_input_bytes
        try:
            fd, _resolved = self.sandbox.open_read(relpath, max_bytes=cap)
        except HiveError as exc:
            raise DocumentParseError(
                f"cannot read {relpath!r} within max_input_bytes {cap}: {exc}") from exc
        try:
            chunks = []
            remaining = cap + 1
            while remaining > 0:
                chunk = os.read(fd, min(1 << 20, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
        finally:
            os.close(fd)
        if len(data) > cap:
            raise DocumentParseError(
                f"document exceeds max_input_bytes {cap} — refusing to parse")
        return data

    def read(self, path: str, *, format: str = "auto", scope: dict[str, Any] | None = None,
             max_chars: int | None = None, agent: str = "system",
             run_id: str | None = None, task_id: str | None = None,
             lease_owner: str | None = None, lease_generation: int | None = None,
             register_source: bool | None = None) -> DocReadResult:
        if not self.config.enabled:
            raise DocumentError("document reading is disabled by policy (documents.enabled)")
        if format != "auto" and format not in SUPPORTED_FORMATS:
            raise DocumentParseError(f"unsupported format {format!r}")
        if format != "auto" and format not in self.config.allowed_formats:
            raise DocumentError(f"format {format!r} not allowed by operator policy")
        eff_max_chars = min(max_chars or self.config.max_extract_chars,
                            self.config.max_extract_chars)  # models cannot raise limits
        limits = self._limits()
        limits["max_extract_chars"] = eff_max_chars
        data = self._read_bytes_capped(path)
        raw_sha = hashlib.sha256(data).hexdigest()
        fmt = detect_format(path, data) if format == "auto" else format
        if fmt not in self.config.allowed_formats:
            raise DocumentError(f"detected format {fmt!r} not allowed by operator policy")
        if fmt in ("docx", "xlsx"):
            # parent-side pure prescan before any bytes leave the process
            zip_keyword_prescan(data, ZipCaps(
                max_archive_bytes=limits["max_input_bytes"] * 2, max_total_expanded=400 << 20))

        if fmt in WORKER_FORMATS:
            pool = self._get_pool()
            try:
                res = pool.run(f"parse_{fmt}", data,
                               args={"scope": scope or {}, "limits": limits},
                               timeout=self.config.worker_op_timeout_seconds)
            except ParserIsolationError as exc:
                raise DocumentUnavailableError(f"parser worker unavailable: {exc}") from exc
            if not res.ok:
                raise DocumentParseError(f"parser worker failed: {res.error}")
            if not res.output:
                raise DocumentParseError("parser worker returned no result")
            out = json.loads(res.output.decode("utf-8"))  # bounded JSON only
        elif fmt == "csv":
            out = parse_csv_doc(data, limits)
        elif fmt == "json":
            out = parse_json_doc(data, limits)
        else:
            out = parse_text_doc(data, fmt, limits)

        if out.get("error"):
            err = out["error"]
            if err == "missing_dependency":
                raise DocumentUnavailableError(
                    f"optional dependency {out.get('dep')!r} is not installed — "
                    "format disabled (fail closed)")
            raise DocumentParseError(f"{err}: {out.get('detail', '')}".strip())
        if out.get("raw_sha256") and out["raw_sha256"] != raw_sha:
            raise DocumentParseError(
                "hash mismatch: parsed bytes differ from the sandbox-read source bytes")
        text = out.get("text", "")
        extract_sha = out.get("extract_sha256") or hashlib.sha256(text.encode()).hexdigest()

        source_id: str | None = None
        do_register = self.config.register_sources if register_source is None else register_source
        if do_register and self.registry is not None and run_id is not None:
            origin = self.registry.trusted_origin(
                run_id=run_id, lease_owner=lease_owner,  # type: ignore[arg-type]
                lease_generation=lease_generation,  # type: ignore[arg-type]
                agent_id=agent, task_id=task_id)
            prov = {"format": fmt, "scope_requested": scope or {}, "parser": out.get("parser", ""),
                    "bytes_in": out.get("bytes_in", len(data)),
                    "truncated": out.get("truncated", False),
                    "unsupported": out.get("unsupported", [])}
            for k in ("page_count", "pages_selected", "sheet_names", "formula_cells",
                      "missing_cached_values", "metadata"):
                if out.get(k) is not None:
                    prov[k] = out[k] if k != "metadata" else {
                        kk: str(vv)[:200] for kk, vv in out[k].items()}
            if out.get("blocks"):
                prov["blocks_summary"] = out["blocks"][:100]
            if out.get("sheets"):
                prov["sheets_summary"] = [
                    {k: v for k, v in s.items() if k != "rows"} for s in out["sheets"]][:20]
            if out.get("pages"):
                prov["pages_provenance"] = out["pages"][:100]
            source_id = self.registry.register(
                origin=origin, source_type=SourceType.DOCUMENT, provider="local-document",
                locator=f"workspace:{path}", raw_bytes=data, content_sha256=extract_sha,
                extract_method=out.get("parser", "unknown"), extract_version="1",
                provenance=prov, license=None, attribution="")
        return DocReadResult(
            source_id=source_id, path=path, format=fmt, parser=out.get("parser", ""),
            text=text, metadata=out.get("metadata", {}),
            structure=out.get("blocks") or out.get("pages") or out.get("sheets") or [],
            truncated=bool(out.get("truncated")), unsupported=out.get("unsupported", []),
            raw_sha256=raw_sha, extract_sha256=extract_sha, bytes_in=out.get("bytes_in", len(data)),
            extra={k: out[k] for k in ("headings", "formula_cells", "missing_cached_values",
                                       "sheet_names", "page_count") if out.get(k) is not None})


# ---- agent-facing tool (narrow registration; W6 wires it broadly) -----------------------

class DocReadInput(BaseModel):
    """Strict typed input (extra=forbid): relative sandbox path, format, narrow
    scope, and a char limit that is CLAMPED to the operator policy maximum."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(pattern=r"^[^/\\\x00-\x1f][^\x00-\x1f]{0,511}$")
    format: str = Field(default="auto", pattern=r"^(auto|pdf|docx|xlsx|csv|json|txt|md|html)$")
    pages: str | None = Field(default=None, pattern=r"^[0-9,\- all]{0,64}$")
    sheets: list[str] | None = Field(default=None, max_length=10)
    max_chars: int | None = Field(default=None, ge=1, le=200_000)


class DocReadAdapter:
    def __init__(self, reader: DocumentReader, spec: ToolSpec, *,
                 lease_lookup: Callable[[str], Any] | None = None) -> None:
        self.reader = reader
        self.spec = spec
        self._lease_lookup = lease_lookup

    def _lease_binding(self, ctx: Any) -> tuple[str | None, str | None, int | None]:
        run_id = getattr(ctx, "run_id", None)
        if run_id is None:
            return None, None, None
        if self._lease_lookup is None:
            from .errors import ToolPermissionError

            raise ToolPermissionError(f"run {run_id!r} has no lease identity — fail closed")
        rc = self._lease_lookup(run_id)
        owner = getattr(rc, "lease_owner", None) if rc is not None else None
        gen = getattr(rc, "lease_generation", None) if rc is not None else None
        if owner is None or gen is None:
            from .errors import ToolPermissionError

            raise ToolPermissionError(f"run {run_id!r} has no lease identity — fail closed")
        return run_id, owner, gen

    def execute(self, args: dict[str, Any], ctx: Any) -> str:
        run_id, lease_owner, lease_gen = self._lease_binding(ctx)
        scope: dict[str, Any] = {}
        if args.get("pages"):
            scope["pages"] = args["pages"]
        if args.get("sheets"):
            scope["sheets"] = args["sheets"]
        res = self.reader.read(
            args["path"], format=args.get("format", "auto"), scope=scope,
            max_chars=args.get("max_chars"), agent=getattr(ctx, "agent", "system"),
            run_id=run_id, task_id=getattr(ctx, "task_id", None),
            lease_owner=lease_owner, lease_generation=lease_gen)
        out: dict[str, Any] = {
            "source_id": res.source_id, "path": res.path, "format": res.format,
            "parser": res.parser, "text": res.text, "metadata": res.metadata,
            "truncated": res.truncated, "unsupported": res.unsupported,
            "raw_sha256": res.raw_sha256, "extract_sha256": res.extract_sha256,
            "bytes_in": res.bytes_in, "note": res.note, **res.extra,
        }
        if res.source_id:
            try:
                out["citation"] = self.reader.registry.cite(res.source_id)
            except Exception:  # noqa: BLE001
                pass
        return redact_secrets(json.dumps(out, ensure_ascii=False, default=str))

    async def aexecute(self, args: dict[str, Any], ctx: Any) -> str:
        import asyncio

        return await asyncio.to_thread(self.execute, args, ctx)


def doc_tools(reader: DocumentReader, *, lease_lookup: Callable[[str], Any] | None = None
              ) -> list[tuple[ToolSpec, DocReadAdapter]]:
    spec = ToolSpec(
        name="doc_read",
        description=("Read ONE document inside the workspace sandbox (pdf/docx/xlsx "
                     "parsed in an isolated worker; csv/json/txt/md/html via bounded "
                     "stdlib parsers). Returns bounded extracted text with page/sheet/cell "
                     "provenance and a registered source citation. Content is untrusted."),
        version="1", category="documents", side_effect=SideEffect.READ_ONLY,
        input_model=DocReadInput, max_output_bytes=160_000,
    )
    return [(spec, DocReadAdapter(reader, spec, lease_lookup=lease_lookup))]
