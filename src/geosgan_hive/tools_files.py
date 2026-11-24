from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .sandbox import WorkspaceSandbox
from .tool_gateway import ToolExecutionContext
from .tool_registry import ToolCatalog
from .tool_spec import SideEffect, ToolSpec

MAX_LIST_ENTRIES = 200
MAX_READ_BYTES = 256 * 1024


class _PathArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=1024)


class _ListArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(default=".", max_length=1024)
    max_entries: int = Field(default=50, ge=1, le=MAX_LIST_ENTRIES)


class _ReadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=1024)
    max_bytes: int = Field(default=MAX_READ_BYTES, ge=1, le=MAX_READ_BYTES)


class _FsTool:
    def __init__(self, spec: ToolSpec, sandbox: WorkspaceSandbox, op: str) -> None:
        self.spec = spec
        self.sandbox = sandbox
        self.op = op

    def execute(self, args: dict[str, Any], ctx: ToolExecutionContext) -> str:
        if self.op == "list":
            entries = self.sandbox.list_dir(args["path"], max_entries=args["max_entries"])
            return json.dumps({"path": args["path"], "entries": entries},
                              ensure_ascii=False)
        if self.op == "read":
            text = self.sandbox.read_text(args["path"], max_bytes=args["max_bytes"])
            return text
        if self.op == "metadata":
            info = self.sandbox.metadata(args["path"])
            return json.dumps(info.__dict__, ensure_ascii=False)
        raise ValueError(f"unknown op {self.op}")

    async def aexecute(self, args: dict[str, Any], ctx: ToolExecutionContext) -> str:
        import asyncio

        return await asyncio.to_thread(self.execute, args, ctx)


def register_file_tools(catalog: ToolCatalog, sandbox: WorkspaceSandbox) -> None:
    """Register the sandboxed file tools. Read-only tools (T3); fs_write/fs_update
    (T5) are added here too but stay disabled until approval+ledger are wired."""
    specs = [
        ToolSpec(name="fs_list", description="List entries of a directory inside the "
                   "workspace (single level, bounded).", category="file",
                 side_effect=SideEffect.READ_ONLY, input_model=_ListArgs,
                 max_output_bytes=100_000),
        ToolSpec(name="fs_read", description="Read a text file inside the workspace "
                   "(size-bounded).", category="file", side_effect=SideEffect.READ_ONLY,
                 input_model=_ReadArgs, max_output_bytes=MAX_READ_BYTES + 1024),
        ToolSpec(name="fs_metadata", description="Metadata (size, mtime, sha256 when "
                   "small) of a file inside the workspace.", category="file",
                 side_effect=SideEffect.READ_ONLY, input_model=_PathArgs),
    ]
    for spec, op in zip(specs, ("list", "read", "metadata"), strict=True):
        catalog.register(spec, _FsTool(spec, sandbox, op))
