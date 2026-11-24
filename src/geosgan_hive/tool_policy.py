from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from .tool_spec import ToolSpec


@dataclass(frozen=True)
class ToolPolicyEntry:
    """Per-tool overrides on top of a ToolSpec (max calls, approval override)."""

    max_calls_per_run: int | None = None
    requires_approval: bool | None = None


@dataclass
class ToolPolicy:
    """Deny-by-default permission resolution for the ToolGateway.

    - `global_enabled`: tools available at all (None never happens in production;
      the factory passes the built-ins in legacy mode or the configured list).
    - `per_agent`: per-agent allowlists from `Permissions` (allow_tools=False or an
      explicitly empty `allowed_tools` denies ALL tools for that agent).
    - Absent configuration (legacy mode) is distinguished from an explicitly empty
      configuration by the factory, not here.

    The supervisor has no tools unless explicitly configured; delegation never
    transfers privileges because the gateway always resolves the policy for the
    EXECUTING agent identity from the trusted context.
    """

    global_enabled: frozenset[str] = field(default_factory=frozenset)
    per_agent: dict[str, frozenset[str]] = field(default_factory=dict)
    per_tool: dict[str, ToolPolicyEntry] = field(default_factory=dict)

    def allows(self, agent: str, tool: str) -> bool:
        if tool not in self.global_enabled:
            return False
        return tool in self.per_agent.get(agent, frozenset())

    def entry(self, tool: str) -> ToolPolicyEntry:
        return self.per_tool.get(tool, ToolPolicyEntry())

    def requires_approval(self, spec: ToolSpec, agent: str, tool: str) -> bool:
        entry = self.per_tool.get(tool)
        if entry is not None and entry.requires_approval is not None:
            return entry.requires_approval
        return spec.approval_required

    def max_calls(self, tool: str) -> int | None:
        entry = self.per_tool.get(tool)
        return entry.max_calls_per_run if entry is not None else None


def build_policy(
    *,
    global_enabled: Iterable[str],
    agent_permissions: dict[str, tuple[bool, list[str]]],
    per_tool: dict[str, ToolPolicyEntry] | None = None,
) -> ToolPolicy:
    """Build a ToolPolicy from the config. `agent_permissions` maps agent name to
    (allow_tools, allowed_tools)."""
    per_agent = {
        name: (frozenset(tools) if allow else frozenset())
        for name, (allow, tools) in agent_permissions.items()
    }
    return ToolPolicy(
        global_enabled=frozenset(global_enabled),
        per_agent=per_agent,
        per_tool=dict(per_tool or {}),
    )
