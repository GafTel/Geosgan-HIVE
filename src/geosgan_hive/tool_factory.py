from __future__ import annotations

from typing import Any

from .agent_config import HiveConfig
from .approvals import ApprovalManager
from .sandbox import WorkspaceSandbox
from .tool_gateway import BuiltinToolAdapter, ToolGateway
from .tool_policy import build_policy
from .tool_registry import ToolCatalog
from .tool_spec import SideEffect, ToolSpec
from .tools import CalculatorTool, CurrentDateTimeTool, WordCountTool

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


def build_tool_catalog(config: HiveConfig | None = None) -> tuple[ToolCatalog, WorkspaceSandbox]:
    """Catalog with all compiled-in tools. Registration != availability: the
    ToolPolicy decides what each agent may use (deny-by-default)."""
    catalog = ToolCatalog()
    register_builtin_tools(catalog)
    from .tools_data import register_data_tools

    register_data_tools(catalog)  # registered but denied unless enabled
    from .sandbox import WorkspaceSandbox
    from .tools_files import register_file_tools

    has_tools = config is not None and config.tools is not None
    root = config.tools.workspace_root if has_tools else "workspace"
    deny = config.tools.deny_name_patterns if has_tools else None
    write_ext = config.tools.write_extensions if has_tools else None
    # W5: when document generation is enabled, its output extensions join the
    # sandbox write allowlist (operator opt-in via documents.enabled)
    docs_on = (config is not None and getattr(config, "documents", None) is not None
               and config.documents.enabled)
    if docs_on:
        extra = {".pdf", ".docx", ".xlsx", ".csv", ".json", ".md", ".txt"}
        write_ext = sorted(set(write_ext or [".txt", ".md"]) | extra)
    sandbox = WorkspaceSandbox(root, deny, write_extensions=write_ext)
    # Fail closed where race-resistant containment cannot be guaranteed (no
    # post-open fd re-verification, e.g. Windows/macOS): file tools are not even
    # registered there unless the operator explicitly accepted the residual risk.
    fs_ok = sandbox.race_resistant or sandbox.allow_insecure_fs
    if fs_ok:
        register_file_tools(catalog, sandbox,
                            allow_write=bool(has_tools and config.tools.allow_write))
    return catalog, sandbox


def build_tool_gateway(
    config: HiveConfig,
    store: Any = None,
    approval_manager: ApprovalManager | None = None,
    lease_lookup: Any = None,
) -> ToolGateway:
    """Build the production ToolGateway. ALWAYS enforced by the factory.

    Legacy mode (config.tools is None): the previously authorized built-ins remain
    available per agent permissions. Explicit `tools:` block: only `enabled` tools
    are available globally (empty list = deny all); new capabilities are off unless
    listed. `strict` fails closed when the durable store is wired but the run
    context is missing.

    A6: when `config.apis` enables public API providers, their typed read-only
    tools are registered on the same catalog (still deny-by-default via the tool
    policy AND the per-provider outbound policy). All external API tools are
    disabled by default.
    """
    catalog, _sandbox = build_tool_catalog(config)
    # W4: document reading tools — registered ONLY when documents.enabled=true
    # (default disabled; deny-by-default via the same ToolPolicy). PDF/DOCX/XLSX
    # parse inside the isolated POSIX parser worker (fail closed elsewhere).
    docs_cfg = getattr(config, "documents", None)
    if docs_cfg is not None and docs_cfg.enabled and store is not None:
        from .docgen import DocumentWriter, docgen_tools
        from .documents import DocumentReader, doc_tools

        reader = DocumentReader(store=store, sandbox=_sandbox, config=docs_cfg)
        for spec, adapter in doc_tools(reader, lease_lookup=lease_lookup):
            catalog.register(spec, adapter)
        writer = DocumentWriter(store=store, sandbox=_sandbox, config=docs_cfg)
        for spec, adapter in docgen_tools(writer, lease_lookup=lease_lookup):
            catalog.register(spec, adapter)
    # W6: read-only source inspection tools — registered whenever a durable store
    # is wired (the registry is core W2), still deny-by-default through the same
    # ToolPolicy (legacy mode: built-ins only; explicit mode: operator allowlist)
    if store is not None:
        from .source_tools import source_tools
        from .sources import SourceRegistry

        for spec, adapter in source_tools(SourceRegistry(store), lease_lookup=lease_lookup):
            catalog.register(spec, adapter)
    if approval_manager is None and store is not None and config.tools is not None:
        approval_manager = ApprovalManager(
            store,
            expiry_seconds=config.tools.approval.expiry_seconds,
            preview_bytes=config.tools.approval.preview_bytes,
        )
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
    # A6: register specialized public API tools for explicitly enabled providers
    # (default: none — every external API tool stays disabled). The session is
    # built through the strict production path (endpoint binding enforced).
    apis = getattr(config, "apis", None)
    if apis is not None and apis.enabled and store is not None:
        from .api_adapters import register_api_tools
        from .outbound import build_api_session

        session = build_api_session(config, store, approval_manager)
        if session is not None:
            register_api_tools(catalog, session, lease_lookup=lease_lookup,
                               providers=apis.enabled)
            # W3: secure web retrieval tool (only when the `web` provider is
            # enabled AND operator web endpoint policies exist; deny-by-default
            # via the same ToolPolicy; outbound auth + budgets via ApiSession)
            from .web_retrieval import build_web_engine, web_tools

            engine = build_web_engine(config, session, store)
            if engine is not None:
                for spec, adapter in web_tools(engine, lease_lookup=lease_lookup):
                    catalog.register(spec, adapter)
    return ToolGateway(
        catalog,
        policy,
        store=store,
        approval_manager=approval_manager,
        strict=store is not None,
    )
