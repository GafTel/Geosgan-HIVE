from __future__ import annotations

import csv
import io
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .errors import ToolValidationError
from .tool_gateway import ToolExecutionContext
from .tool_registry import ToolCatalog
from .tool_spec import SideEffect, ToolSpec

#: Hard bounds enforced BEFORE expensive parsing/allocation.
MAX_JSON_NODES = 10_000
MAX_JSON_DEPTH = 64
MAX_CSV_ROWS = 10_000
MAX_CSV_COLS = 256
MAX_CSV_CELL_BYTES = 65_536
MAX_PREVIEW_CHARS = 2_000


class _TextArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)


class _CsvArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    delimiter: str = Field(default=",", min_length=1, max_length=1)


def _check_json_bounds(value: Any, *, nodes: list[int], depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        raise ToolValidationError(f"JSON nesting deeper than {MAX_JSON_DEPTH}")
    nodes[0] += 1
    if nodes[0] > MAX_JSON_NODES:
        raise ToolValidationError(f"JSON has more than {MAX_JSON_NODES} nodes")
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise ToolValidationError("JSON object keys must be strings")
            _check_json_bounds(v, nodes=nodes, depth=depth + 1)
    elif isinstance(value, list):
        for item in value:
            _check_json_bounds(item, nodes=nodes, depth=depth + 1)


def _preview(value: Any, limit: int = MAX_PREVIEW_CHARS) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "...[truncated]"


def _summarize(value: Any) -> str:
    def kind(v: Any) -> str:
        if isinstance(v, dict):
            return f"object({len(v)} keys)"
        if isinstance(v, list):
            return f"array({len(v)} items)"
        return type(v).__name__

    parts = [f"type={kind(value)}"]
    if isinstance(value, dict):
        parts.append("keys=" + json.dumps(sorted(value.keys())[:50], ensure_ascii=False))
    elif isinstance(value, list) and value:
        parts.append(f"first_item_type={kind(value[0])}")
    parts.append("preview=" + _preview(value))
    return "\n".join(parts)


class _JsonTool:
    def __init__(self, spec: ToolSpec, mode: str) -> None:
        self.spec = spec
        self.mode = mode

    def execute(self, args: dict[str, Any], ctx: ToolExecutionContext) -> str:
        try:
            value = json.loads(args["text"])
        except json.JSONDecodeError as exc:
            if self.mode == "validate":
                return f"INVALID JSON: {exc.msg} (line {exc.lineno}, col {exc.colno})"
            raise ToolValidationError(f"invalid JSON: {exc.msg}") from exc
        _check_json_bounds(value, nodes=[0])
        if self.mode == "validate":
            return f"VALID JSON: {_summarize(value)}"
        return _summarize(value)

    async def aexecute(self, args: dict[str, Any], ctx: ToolExecutionContext) -> str:
        import asyncio

        return await asyncio.to_thread(self.execute, args, ctx)


def _parse_csv_rows(text: str, delimiter: str) -> list[list[str]]:
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows: list[list[str]] = []
    for i, row in enumerate(reader):
        if i >= MAX_CSV_ROWS:
            raise ToolValidationError(f"CSV has more than {MAX_CSV_ROWS} rows")
        if len(row) > MAX_CSV_COLS:
            raise ToolValidationError(f"CSV row {i} has more than {MAX_CSV_COLS} columns")
        if any(len(cell.encode("utf-8")) > MAX_CSV_CELL_BYTES for cell in row):
            raise ToolValidationError(f"CSV row {i} has an oversized cell")
        rows.append(row)
    return rows


class _CsvTool:
    def __init__(self, spec: ToolSpec, mode: str) -> None:
        self.spec = spec
        self.mode = mode

    def execute(self, args: dict[str, Any], ctx: ToolExecutionContext) -> str:
        rows = _parse_csv_rows(args["text"], args.get("delimiter", ","))
        if not rows:
            return "EMPTY CSV (0 rows)"
        header, data = rows[0], rows[1:]
        widths = {len(r) for r in rows}
        if self.mode == "validate":
            if len(widths) != 1:
                return (f"INVALID CSV: ragged rows (column counts {sorted(widths)}), "
                        f"{len(rows)} rows")
            return (f"VALID CSV: {len(data)} data rows x {len(header)} columns; "
                    f"header={json.dumps(header, ensure_ascii=False)[:500]}")
        preview_rows = data[:20]
        out = [f"CSV: {len(data)} data rows x {len(header)} columns",
               f"header={json.dumps(header, ensure_ascii=False)[:500]}"]
        for r in preview_rows:
            out.append(json.dumps(r, ensure_ascii=False)[:MAX_PREVIEW_CHARS])
        if len(data) > len(preview_rows):
            out.append(f"...[{len(data) - len(preview_rows)} more rows not shown]")
        return "\n".join(out)

    async def aexecute(self, args: dict[str, Any], ctx: ToolExecutionContext) -> str:
        import asyncio

        return await asyncio.to_thread(self.execute, args, ctx)


def register_data_tools(catalog: ToolCatalog) -> None:
    specs = [
        ToolSpec(name="json_parse", description="Parse JSON text and return a bounded "
                   "structural summary (type, keys, preview).", category="data",
                 side_effect=SideEffect.READ_ONLY, input_model=_TextArgs),
        ToolSpec(name="json_validate", description="Validate JSON text; returns VALID/"
                   "INVALID with details and a bounded summary.", category="data",
                 side_effect=SideEffect.READ_ONLY, input_model=_TextArgs),
        ToolSpec(name="csv_parse", description="Parse CSV text; returns header and a "
                   "bounded row preview (max 20 rows shown).", category="data",
                 side_effect=SideEffect.READ_ONLY, input_model=_CsvArgs),
        ToolSpec(name="csv_validate", description="Validate CSV shape (rectangular, "
                   "bounded rows/columns/cells).", category="data",
                 side_effect=SideEffect.READ_ONLY, input_model=_CsvArgs),
    ]
    catalog.register(specs[0], _JsonTool(specs[0], "parse"))
    catalog.register(specs[1], _JsonTool(specs[1], "validate"))
    catalog.register(specs[2], _CsvTool(specs[2], "parse"))
    catalog.register(specs[3], _CsvTool(specs[3], "validate"))
