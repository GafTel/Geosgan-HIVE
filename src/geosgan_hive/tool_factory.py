from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .agent_config import HiveConfig
from .tool_gateway import BuiltinToolAdapter, ToolGateway
from .tool_policy import build_policy
from .tool_registry import ToolCatalog
from .tool_spec import SideEffect, ToolSpec
from .tools import CalculatorTool, CurrentDateTimeTool, WordCountTool

if TYPE_CHECKING:
    from .approvals import ApprovalManager

#: Legacy built-in tools (available in legacy mode when no `tools:` block is set).
BUILTIN_TOOL_NAMES = ("calculator", "current_datetime", "word_count")


def _builtin_specs() -> list[tuple[ToolSpec, Any]]:
    def spec_for(tool: Any) -> ToolSpec:
        return ToolSpec(
            name=tool.name,
            description=tool.description,
            category="builtin",
            side_effect=SideEffect.PURE if tool.name == "calculator" else SideEffect.READ_ONLY,
            idempotent=True,
            requires_approval=False,
            input_model=None,
        )

    tools = [CalculatorTool(), CurrentDateTimeTool(), WordCountTool()]
    return [(spec_for(t), t) for t in tools]


def register_builtin_tools(catalog: ToolCatalog) -> None:
    for spec, tool in _builtin_specs():
        catalog.register(spec, BuiltinToolAdapter(tool, spec))


def build_tool_catalog() -> ToolCatalog:
    """Catalog with all compiled-in tools. Registration != availability: the
    ToolPolicy decides what each agent may use (deny-by-default)."""
    catalog = ToolCatalog()
    register_builtin_tools(catalog)
    # T2/T3 extend this with data and file tools (registered but denied unless
    # enabled in the policy)
    return catalog


def build_tool_gateway(
    config: HiveConfig,
    store: Any = None,
    approval_manager: ApprovalManager | None = None,
) -> ToolGateway:
    """Build the production ToolGateway. ALWAYS enforced by the factory.

    Legacy mode (config.tools is None): the previously authorized built-ins remain
    available per agent permissions. Explicit `tools:` block: only `enabled` tools
    are available globally (empty list = deny all); new capabilities are off unless
    listed. `strict` fails closed when the durable store is wired but the run
    context is missing.
    """
    catalog = build_tool_catalog()
    if config.tools is None:
        global_enabled = list(BUILTIN_TOOL_NAMES)
    else:
        global_enabled = list(config.tools.enabled)
    agent_permissions = {
        spec.name: (spec.permissions.allow_tools, list(spec.permissions.allowed_tools))
        for spec in config.agents
    }
    agent_permissions[config.supervisor.name] = (
        config.supervisor.permissions.allow_tools,
        list(config.supervisor.permissions.allowed_tools),
    )
    policy = build_policy(global_enabled=global_enabled, agent_permissions=agent_permissions)
    return ToolGateway(
        catalog,
        policy,
        store=store,
        approval_manager=approval_manager,
        strict=store is not None,
    )
