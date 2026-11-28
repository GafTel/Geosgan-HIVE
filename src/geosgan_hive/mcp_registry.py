"""M1 (v0.9.0): configuration-only MCP registry (schema v13 preserved — no
durable storage, no migration).

Construction and every method here are PURE: no network, no subprocess, no
package installation, no model involvement. The registry is built ONLY from
operator-owned, strictly validated configuration (mcp_config.McpConfig, wired
into HiveConfig as the optional `mcp:` block). There is no public or
agent-facing API that can register, enable or modify servers — models cannot
reach this code through any tool in M1 (no MCP tools are registered on the
ToolGateway catalog at all during M1).

Fail-closed rules (amendments A/B/D):
- unknown server id / unknown tool name -> error, never auto-registered;
- server-reported annotations are ignored entirely; only the operator-owned
  classification in the config counts;
- capability bindings bind to digests (mcp_contracts.ToolCapability); a reported
  tool set that differs from the operator allowlist, or a schema that fails
  validation or mismatches a pinned digest, is a capability CHANGE -> error and
  invalidation of previous bindings;
- dispatch decisions evaluate the operator rules with zero I/O; DENIED can never
  be turned into allowed by a human approval.

Snapshots are immutable: all config models are frozen, so no caller can mutate
the registry or its snapshots.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .errors import HiveError
from .mcp_config import (
    MAX_SERVERS,
    McpConfig,
    McpServerConfig,
    McpToolPolicy,
    load_mcp_config_file,
)
from .mcp_contracts import (
    POLICY_VERSION,
    SUPPORTED_PROTOCOL_REVISIONS,
    DispatchDecision,
    ToolCapability,
    dispatch_decision,
    effective_cost_class,
    effective_requires_approval,
    namespaced_tool_name,
    server_identity_digest,
    transport_identity,
    validate_input_schema,
)

#: tool-name prefix reserved for MCP bridge tools; collisions fail closed
MCP_TOOL_PREFIX = "mcp_"


class McpRegistryError(HiveError):
    """Registry contract violation — always fail closed."""


@dataclass(frozen=True)
class McpServerEntry:
    """Immutable per-server registry entry with its identity digest."""

    server: McpServerConfig
    identity_digest: str
    transport_identity: str
    tools: Mapping[str, McpToolPolicy]


class McpRegistry:
    """Immutable, configuration-only MCP registry. Construction performs no
    network or subprocess activity (pure validation + hashing)."""

    def __init__(self, config: McpConfig | None = None, *,
                 reserved_names: Iterable[str] = ()) -> None:
        cfg = config if config is not None else McpConfig()
        if len(cfg.servers) > MAX_SERVERS:  # defensive; the model already caps it
            raise McpRegistryError(f"registry exceeds {MAX_SERVERS} servers")
        entries: dict[str, McpServerEntry] = {}
        namespaced: dict[str, str] = {}  # namespaced tool name -> server_id
        reserved = set(reserved_names) | {MCP_TOOL_PREFIX}
        for server in cfg.servers:
            if server.server_id in entries:
                raise McpRegistryError(f"duplicate server id {server.server_id!r}")
            tools: dict[str, McpToolPolicy] = {}
            for tool in server.tools:
                if tool.name in tools:
                    raise McpRegistryError(
                        f"duplicate tool {tool.name!r} in server {server.server_id!r}")
                ns = namespaced_tool_name(server.server_id, tool.name)
                if ns in namespaced:
                    raise McpRegistryError(
                        f"namespace collision: {ns!r} already bound to server "
                        f"{namespaced[ns]!r}")
                namespaced[ns] = server.server_id
                tools[tool.name] = tool
            entries[server.server_id] = McpServerEntry(
                server=server,
                identity_digest=server_identity_digest(server),
                transport_identity=transport_identity(server),
                tools=tools,
            )
        # collision against reserved/builtin tool names (e.g. the existing
        # catalog) — namespaced MCP names must never shadow or duplicate them
        for ns in namespaced:
            if ns in reserved:
                raise McpRegistryError(f"namespaced tool name {ns!r} collides with a "
                                       "reserved/builtin tool name")
        self._config = cfg
        self._entries = entries
        self._namespaced = namespaced

    # ---- construction helpers ----

    @classmethod
    def from_hive_config(cls, hive_config: Any, *,
                         reserved_names: Iterable[str] = ()) -> McpRegistry:
        """Build from a HiveConfig (uses its optional `mcp:` block; absent = the
        default disabled empty registry)."""
        mcp = getattr(hive_config, "mcp", None)
        return cls(mcp if isinstance(mcp, McpConfig) else None,
                   reserved_names=reserved_names)

    @classmethod
    def from_file(cls, path: str, *,
                  reserved_names: Iterable[str] = ()) -> McpRegistry:
        """Build from a standalone MCP registry YAML file (bounded size)."""
        return cls(load_mcp_config_file(path), reserved_names=reserved_names)

    # ---- access (read-only; models are frozen) ----

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    @property
    def policy_version(self) -> str:
        return POLICY_VERSION

    @property
    def supported_protocol_revisions(self) -> tuple[str, ...]:
        return SUPPORTED_PROTOCOL_REVISIONS

    @property
    def servers(self) -> Mapping[str, McpServerEntry]:
        return self._entries

    def snapshot(self) -> McpRegistry:
        """Return an isolated snapshot. Config models are frozen, so the
        snapshot cannot be mutated by any caller."""
        return McpRegistry(self._config.model_copy(deep=True))

    def summary(self) -> dict[str, Any]:
        """Serialized summary for operators/CLI: NEVER contains env reference
        names, executable arguments beyond counts, or any secret material —
        only kinds, counts and digests."""
        return {
            "enabled": self.enabled,
            "policy_version": POLICY_VERSION,
            "supported_protocol_revisions": list(SUPPORTED_PROTOCOL_REVISIONS),
            "server_count": len(self._entries),
            "servers": [
                {
                    "server_id": e.server.server_id,
                    "enabled": e.server.enabled,
                    "protocol_revision": e.server.protocol_revision,
                    "transport": e.server.transport.kind,
                    "endpoint_id": (e.server.transport.endpoint_id
                                    if e.server.transport.kind == "streamable_http"
                                    else None),
                    "cost_class": e.server.cost_class.value,
                    "identity_digest": e.identity_digest,
                    "tool_count": len(e.tools),
                    "tool_names": sorted(e.tools),
                    "env_ref_count": (len(e.server.transport.env_refs)
                                      if e.server.transport.kind == "stdio" else 0),
                }
                for e in self._entries.values()
            ],
        }

    def server(self, server_id: str) -> McpServerEntry:
        entry = self._entries.get(server_id)
        if entry is None:
            raise McpRegistryError(
                f"unknown MCP server {server_id!r} — fail closed (only operator-configured "
                "server ids exist; servers cannot register themselves)")
        return entry

    def tool(self, server_id: str, tool_name: str) -> McpToolPolicy:
        return self.server(server_id).tools.get(tool_name) or self._unknown_tool(
            server_id, tool_name)

    def _unknown_tool(self, server_id: str, tool_name: str) -> McpToolPolicy:
        raise McpRegistryError(
            f"unknown tool {tool_name!r} on server {server_id!r} — not in the operator "
            "allowlist; unknown tools are NEVER auto-registered as executable")

    def dispatch_decision(self, server_id: str, tool_name: str) -> DispatchDecision:
        """Pure operator-rule evaluation for one tool (amendments A/D)."""
        entry = self.server(server_id)
        tool = self.tool(server_id, tool_name)
        return dispatch_decision(entry.server, tool, globally_enabled=self.enabled)

    # ---- capability binding (amendment B) ----

    def bind_capabilities(self, server_id: str,
                          reported_tools: Iterable[dict[str, Any]]) -> list[ToolCapability]:
        """Bind a server-reported `tools/list` payload to operator-owned
        capabilities. `reported_tools` entries look like MCP Tool objects:
        {"name": ..., "inputSchema": {...}, "description": ..., "annotations": ...}.

        Fail closed on: unknown tool names, invalid/out-of-subset schemas,
        pinned-schema digest mismatch (capability change -> previous bindings
        invalidated), duplicate reported names. Server `annotations` (e.g.
        readOnlyHint) are recorded nowhere and NEVER influence classification.
        Returns one immutable ToolCapability per allowed tool; each carries a
        binding digest that changes if any bound field changes."""
        entry = self.server(server_id)
        server = entry.server
        if not server.enabled:
            raise McpRegistryError(f"server {server_id!r} is disabled — no capabilities bound")
        seen: set[str] = set()
        out: list[ToolCapability] = []
        for item in reported_tools:
            if not isinstance(item, dict):
                raise McpRegistryError("reported tool entries must be objects")
            name = item.get("name")
            if not isinstance(name, str) or not name:
                raise McpRegistryError("reported tool has no valid name")
            if name in seen:
                raise McpRegistryError(f"duplicate reported tool {name!r}")
            seen.add(name)
            tool = entry.tools.get(name)
            if tool is None:
                raise McpRegistryError(
                    f"server {server_id!r} reported tool {name!r} that is not in the "
                    "operator allowlist — capability change, fail closed (never "
                    "auto-registered)")
            schema_digest = validate_input_schema(item.get("inputSchema"))
            if tool.input_schema_sha256 is not None \
                    and tool.input_schema_sha256 != schema_digest:
                raise McpRegistryError(
                    f"tool {name!r} on server {server_id!r} reports an input schema whose "
                    "digest does not match the operator-pinned digest — capability CHANGE, "
                    "previous authorization invalidated")
            out.append(ToolCapability(
                server_id=server_id,
                server_identity=entry.identity_digest,
                transport_kind=server.transport.kind,
                transport_identity=entry.transport_identity,
                protocol_revision=server.protocol_revision,
                tool_name=name,
                namespaced_name=namespaced_tool_name(server_id, name),
                input_schema_sha256=schema_digest,
                side_effect=tool.side_effect,
                cost_class=effective_cost_class(server, tool),
                requires_approval=effective_requires_approval(tool),
                max_output_bytes=tool.max_output_bytes,
                policy_version=POLICY_VERSION,
            ))
        # every allowlisted tool must be reported (missing = capability change)
        missing = sorted(set(entry.tools) - seen)
        if missing:
            raise McpRegistryError(
                f"server {server_id!r} did not report allowlisted tools {missing} — "
                "capability change, previous authorization invalidated")
        return out


__all__ = [
    "MCP_TOOL_PREFIX",
    "McpRegistry",
    "McpRegistryError",
    "McpServerEntry",
]
