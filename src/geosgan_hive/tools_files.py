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


class _WriteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=1024)
    content: str = Field(max_length=1024 * 1024)


class _UpdateArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=1024)
    content: str = Field(max_length=1024 * 1024)
    #: sha256 of the ORIGINAL file content; bound into the approval and re-verified
    #: immediately before the write (file changed after approval -> rejected)
    expected_sha256: str = Field(min_length=64, max_length=64)


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
        if self.op == "write":
            sha = self.sandbox.write_text(args["path"], args["content"],
                                          require_exists=False)
            return json.dumps({"path": args["path"], "created": True, "sha256": sha},
                              ensure_ascii=False)
        if self.op == "update":
            sha = self.sandbox.write_text(args["path"], args["content"],
                                          expected_sha256=args["expected_sha256"],
                                          require_exists=True)
            return json.dumps({"path": args["path"], "updated": True, "sha256": sha},
                              ensure_ascii=False)
        raise ValueError(f"unknown op {self.op}")

    def approval_context(self, args: dict[str, Any]) -> dict[str, Any]:
        """Bounded operation/target/preconditions stored with the approval record; the
        operator sees these in the CLI preview and they are revalidated pre-execution."""
        if self.op == "update":
            return {"operation": "update_file", "target": args["path"],
                    "preconditions": {"expected_sha256": args["expected_sha256"]}}
        return {"operation": "create_file", "target": args["path"],
                "preconditions": {"target_exists": False}}

    async def aexecute(self, args: dict[str, Any], ctx: ToolExecutionContext) -> str:
        import asyncio

        return await asyncio.to_thread(self.execute, args, ctx)


def register_file_tools(catalog: ToolCatalog, sandbox: WorkspaceSandbox, *,
                        allow_write: bool = False) -> None:
    """Register the sandboxed file tools. Read-only tools (fs_list/fs_read/fs_metadata)
    are always registered; the MUTATING tools (fs_write/fs_update) are only registered
    when `allow_write` is explicitly enabled in config \u2014 they additionally require
    human approval (derived from LOCAL_MUTATION) and a durable ledger."""
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
    ops = ["list", "read", "metadata"]
    if allow_write:
        specs += [
            ToolSpec(name="fs_write", description="Create a NEW plaintext file inside "
                       "the workspace (atomic replace; fails if the target exists). "
                       "Requires human approval.", category="file",
                     side_effect=SideEffect.LOCAL_MUTATION, idempotent=False,
                     requires_approval=True, input_model=_WriteArgs, timeout_seconds=10),
            ToolSpec(name="fs_update", description="Update an EXISTING plaintext file "
                       "inside the workspace (atomic replace). Requires human approval "
                       "bound to expected_sha256 of the original content; if the file "
                       "changed after approval the write is rejected.", category="file",
                     side_effect=SideEffect.LOCAL_MUTATION, idempotent=False,
                     requires_approval=True, input_model=_UpdateArgs, timeout_seconds=10),
        ]
        ops += ["write", "update"]
    for spec, op in zip(specs, ops, strict=True):
        catalog.register(spec, _FsTool(spec, sandbox, op))
