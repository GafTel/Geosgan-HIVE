"""W6 (v0.8.0): read-only source inspection tools for agents.

`source_list` / `source_get` / `source_excerpt` expose the W2 SourceRegistry
through the existing ToolGateway/ToolPolicy/lease machinery. Deny-by-default:
registered on the catalog whenever a durable store is wired, but usable only
when the operator lists them in `tools.enabled` AND grants them per agent.

Trust model (unchanged from W2): a source id is only trusted when it comes
from the durable registry — never from model output. Every read is lease-fenced
(read_binding), scope-checked (run / team visibility), expiry-checked and, for
excerpts, integrity-verified against the stored content hash. Citations
establish PROVENANCE only, never factual correctness; generated artifacts are
always marked generated=true and can never be relabeled as retrieved evidence.
Metadata-only by default: excerpts are only available for sources stored under
an explicitly authorized workspace storage policy; nothing is ever reconstructed.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .errors import CheckpointError, ToolPermissionError, ToolValidationError
from .sources import SourceRegistry
from .tool_gateway import redact_secrets
from .tool_spec import SideEffect, ToolSpec

SOURCE_ID_PATTERN = r"^src_[0-9a-f]{16}$"


class SourceListInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit: int = Field(default=50, ge=1, le=200)


class SourceGetInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(pattern=SOURCE_ID_PATTERN)


class SourceExcerptInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(pattern=SOURCE_ID_PATTERN)
    max_chars: int = Field(default=500, ge=1, le=2_000)


def _summary(d: dict[str, Any]) -> dict[str, Any]:
    """Bounded display metadata for one source (no content, no raw locator paths
    beyond the sanitized registration-time locator, capped at 256 chars)."""
    try:
        prov = json.loads(d.get("provenance_json") or "{}")
    except (ValueError, TypeError):
        prov = {}
    return {
        "source_id": d["source_id"],
        "source_type": d["source_type"],
        "provider": d.get("provider") or "unknown",
        "locator": (d.get("locator") or "unknown")[:256],
        "retrieved_at": d.get("retrieved_at"),
        "content_sha256": (d.get("content_sha256") or d.get("raw_sha256") or "")[:16],
        "license": d.get("license") or "unknown",
        "attribution": (d.get("attribution") or "")[:200],
        "origin_kind": d.get("origin_kind"),
        "generated": bool(prov.get("generated", False)),
        "expired": bool(d.get("expired", False)),
        "visibility": d.get("visibility"),
        "truncated": bool(prov.get("truncated", False)),
    }


class _SourceAdapter:
    """Shared lease-fenced binding resolution for the source tools."""

    def __init__(self, registry: SourceRegistry, spec: ToolSpec, *,
                 lease_lookup: Callable[[str], Any] | None = None) -> None:
        self.registry = registry
        self.spec = spec
        self._lease_lookup = lease_lookup

    def _binding(self, ctx: Any):
        run_id = getattr(ctx, "run_id", None)
        if not run_id:
            raise ToolPermissionError(
                "source tools require a run-bound context (run_id) — fail closed")
        rc = self._lease_lookup(run_id) if self._lease_lookup is not None else None
        owner = getattr(rc, "lease_owner", None) if rc is not None else None
        gen = getattr(rc, "lease_generation", None) if rc is not None else None
        if owner is None or gen is None:
            raise ToolPermissionError(f"run {run_id!r} has no lease identity — fail closed")
        try:
            return self.registry.read_binding(run_id=run_id, lease_owner=owner,
                                              lease_generation=gen)
        except CheckpointError as exc:
            raise ToolPermissionError(str(exc)) from exc

    def _run(self, fn: Callable[[Any], Any], ctx: Any) -> str:
        binding = self._binding(ctx)
        try:
            out = fn(binding)
        except CheckpointError as exc:
            msg = str(exc)
            if "cross-run" in msg or "access" in msg or "lease" in msg:
                raise ToolPermissionError(msg) from exc
            raise ToolValidationError(msg) from exc
        return redact_secrets(json.dumps(out, ensure_ascii=False, default=str))

    def execute(self, args: dict[str, Any], ctx: Any) -> str:
        raise NotImplementedError

    async def aexecute(self, args: dict[str, Any], ctx: Any) -> str:
        import asyncio

        return await asyncio.to_thread(self.execute, args, ctx)


class SourceListAdapter(_SourceAdapter):
    def execute(self, args: dict[str, Any], ctx: Any) -> str:
        def fn(b):
            rows = self.registry.list(binding=b, limit=args["limit"])
            return {"count": len(rows), "sources": [_summary(d) for d in rows],
                    "note": "provenance references only — not factual verification"}

        return self._run(fn, ctx)


class SourceGetAdapter(_SourceAdapter):
    def execute(self, args: dict[str, Any], ctx: Any) -> str:
        sid = args["source_id"]

        def fn(b):
            d = self.registry.get(sid, binding=b)
            if d is None:
                raise ToolValidationError(f"unknown source id {sid!r}")
            out = _summary(d)
            out["detail"] = {
                "storage": d.get("storage"),
                "content_bytes": d.get("content_bytes"),
                "raw_sha256": (d.get("raw_sha256") or "")[:16],
                "extract_method": (d.get("extract_method") or "")[:100],
                "extract_version": (d.get("extract_version") or "")[:20],
                "task_id": d.get("task_id"),
                "agent_id": d.get("agent_id"),
                "expires_at": d.get("expires_at"),
            }
            return out

        return self._run(fn, ctx)


class SourceExcerptAdapter(_SourceAdapter):
    def execute(self, args: dict[str, Any], ctx: Any) -> str:
        sid = args["source_id"]

        def fn(b):
            d = self.registry.get(sid, binding=b)
            if d is None:
                raise ToolValidationError(f"unknown source id {sid!r}")
            ex = self.registry.excerpt(sid, binding=b, max_chars=args["max_chars"])
            return {"source_id": sid, "generated": _summary(d)["generated"],
                    "available": ex["available"], "integrity": ex["integrity"],
                    "excerpt": ex["excerpt"],
                    "note": ("excerpt is integrity-verified stored content; unavailable "
                             "means metadata-only — never reconstructed")}

        return self._run(fn, ctx)


def source_tools(registry: SourceRegistry, *,
                 lease_lookup: Callable[[str], Any] | None = None) -> list[tuple[ToolSpec, Any]]:
    """Register the read-only source inspection tools (deny-by-default via policy)."""
    list_spec = ToolSpec(
        name="source_list", version="1", category="sources",
        side_effect=SideEffect.READ_ONLY, requires_approval=False,
        input_model=SourceListInput, max_output_bytes=100_000,
        description=("List registered sources visible to the current run/lease "
                     "(run-scoped + team visibility). Bounded metadata; provenance only."))
    get_spec = ToolSpec(
        name="source_get", version="1", category="sources",
        side_effect=SideEffect.READ_ONLY, requires_approval=False,
        input_model=SourceGetInput, max_output_bytes=100_000,
        description=("Metadata for one registered source id (lease-fenced, scope-checked, "
                     "expiry-checked). Never content."))
    ex_spec = ToolSpec(
        name="source_excerpt", version="1", category="sources",
        side_effect=SideEffect.READ_ONLY, requires_approval=False,
        input_model=SourceExcerptInput, max_output_bytes=100_000,
        description=("Bounded, integrity-verified excerpt of a source whose content is "
                     "stored under an authorized workspace policy. Unavailable "
                     "(metadata-only) is reported explicitly; missing content is never "
                     "invented."))
    return [
        (list_spec, SourceListAdapter(registry, list_spec, lease_lookup=lease_lookup)),
        (get_spec, SourceGetAdapter(registry, get_spec, lease_lookup=lease_lookup)),
        (ex_spec, SourceExcerptAdapter(registry, ex_spec, lease_lookup=lease_lookup)),
    ]
