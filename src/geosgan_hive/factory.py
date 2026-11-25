from __future__ import annotations

from typing import Any

from .agent_config import AgentSpec, HiveConfig, SupervisorSpec
from .agents import Agent
from .checkpoints import CheckpointStore
from .cloud_gate import ZeroCostGate
from .cloud_profiles import CloudProfileRegistry
from .collaboration import CollaborationEngine
from .errors import ConfigError
from .fallback import FallbackPolicy, effective_policy
from .gateway import ModelGateway
from .orchestrator import Supervisor
from .tool_factory import build_tool_gateway
from .tools import ToolRegistry, default_registry
from .usage import UsageTracker


def build_agent(
    spec: AgentSpec,
    gateway: ModelGateway,
    tracker: UsageTracker,
    tools: ToolRegistry | None = None,
    collaboration: CollaborationEngine | None = None,
    global_fallback: FallbackPolicy | None = None,
    tool_gateway: Any = None,
) -> Agent:
    """Build one specialist agent from its spec: own model, permissions, limits."""
    registry = tools if tools is not None else default_registry()
    policy = effective_policy(spec.fallback, global_fallback or FallbackPolicy())
    llm = gateway.bind(
        spec.model,
        agent=spec.name,
        temperature=spec.temperature,
        max_tokens=spec.limits.max_tokens,
        timeout_seconds=spec.limits.timeout_seconds,
        fallback=policy,
        needs_tools=spec.permissions.allow_tools and bool(spec.permissions.allowed_tools),
    )
    return Agent(
        spec.name,
        spec.description,
        spec.system_prompt,
        llm,
        registry,
        permissions=spec.permissions,
        limits=spec.limits,
        usage_tracker=tracker,
        temperature=spec.temperature,
        collaboration=collaboration,
        tool_gateway=tool_gateway,
    )


def build_supervisor(
    spec: SupervisorSpec,
    agents: dict[str, Agent],
    gateway: ModelGateway,
    tracker: UsageTracker,
    collaboration: CollaborationEngine | None = None,
    global_fallback: FallbackPolicy | None = None,
    checkpoint_store: CheckpointStore | None = None,
    checkpoint_config_json: str | None = None,
) -> Supervisor:
    policy = effective_policy(spec.fallback, global_fallback or FallbackPolicy())
    llm = gateway.bind(
        spec.model,
        agent=spec.name,
        temperature=spec.temperature,
        max_tokens=spec.limits.max_tokens,
        timeout_seconds=spec.limits.timeout_seconds,
        fallback=policy,
        needs_tools=False,
    )
    return Supervisor(
        agents,
        llm,
        max_steps=spec.max_steps,
        temperature=spec.temperature,
        usage_tracker=tracker,
        limits=spec.limits,
        collaboration=collaboration,
        checkpoint_store=checkpoint_store,
        checkpoint_config_json=checkpoint_config_json,
        suspend_cooldown_seconds=policy.cooldown_seconds,
        max_wait_seconds=policy.max_wait_seconds,
    )


def build_team(
    config: HiveConfig,
    gateway: ModelGateway,
    tracker: UsageTracker | None = None,
    *,
    checkpoint_store: CheckpointStore | None = None,
    cloud_adapter: Any = None,
    cloud_default_max_output_tokens: int | None = None,
) -> tuple[Supervisor, dict[str, Agent]]:
    """Build the whole team from a validated config, wiring the engine.

    The CollaborationEngine is shared by the supervisor and all agents; agents
    never reference each other directly — all handoffs go through the engine.
    `checkpoint_store` may be injected (e.g. by the CLI so management commands and
    the team share ONE store connection); otherwise one is created from
    `config.checkpoint` when present.
    """
    """Build the whole HIVE team from a validated config, wiring the engine.

    The CollaborationEngine is shared by the supervisor and all agents; agents
    never reference each other directly — all handoffs go through the engine.
    """
    tracker = tracker if tracker is not None else gateway.tracker
    engine = CollaborationEngine(policy=config.collaboration, tracker=tracker)
    # wire the gateway to the run contexts so every dispatch reserves its physical
    # attempt and request slot atomically at the dispatch boundary
    if hasattr(gateway, "set_context_lookup"):
        gateway.set_context_lookup(engine.context_for)
    # the global fallback policy also configures the gateway health registry
    # (in-memory; in-process cooldowns use a monotonic clock)
    if hasattr(gateway, "health"):
        gateway.health.failure_threshold = config.fallback.failure_threshold
        gateway.health.cooldown_seconds = config.fallback.cooldown_seconds
        gateway.health.half_open_max_probes = config.fallback.half_open_max_probes
    store = checkpoint_store
    if store is None and config.checkpoint is not None:
        store = CheckpointStore(
            config.checkpoint.path,
            persist_mode=config.checkpoint.persist_mode,
            lease_ttl_seconds=config.checkpoint.lease_ttl_seconds,
        )
    if store is not None and hasattr(gateway, "set_durable_store"):
        # every physical dispatch reservation is durably committed to SQLite
        # BEFORE the provider request is sent
        gateway.set_durable_store(store)
    # Phase 4 C4: wire the cloud registry + zero-cost gate through the factory so a
    # YAML zero_cost=true can never silently run in legacy mode
    if config.cloud is not None and hasattr(gateway, "set_cloud_registry"):
        reg = CloudProfileRegistry.from_config(config.cloud)
        gateway.set_cloud_registry(reg)
        if reg.zero_cost:
            if store is None or getattr(store, "path", None) == ":memory:":
                raise ConfigError(
                    "cloud.zero_cost=true requires a durable shared quota store "
                    "(config.checkpoint.path, not ':memory:') — refusing to run in "
                    "legacy mode"
                )
            gate = ZeroCostGate(
                reg, store, adapter=cloud_adapter,
                default_max_output_tokens=cloud_default_max_output_tokens,
            )
            gateway.set_cloud_gate(gate)
    tools = default_registry()
    # Phase 3: the ToolGateway is ALWAYS wired in production; every tool call goes
    # through it (policy + budgets + validation). Legacy registry path remains only
    # for standalone fixtures that build agents without a gateway.
    tool_gw = build_tool_gateway(config, store)
    if config.tools is not None:
        engine.tool_budgets = config.tools.budgets
    agents = {
        spec.name: build_agent(spec, gateway, tracker, tools, engine, config.fallback,
                               tool_gateway=tool_gw)
        for spec in config.agents
    }
    engine.register(agents)
    supervisor = build_supervisor(
        config.supervisor, agents, gateway, tracker, engine, config.fallback,
        checkpoint_store=store,
        checkpoint_config_json=config.model_dump_json(),
    )
    return supervisor, agents
