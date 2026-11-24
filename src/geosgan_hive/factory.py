from __future__ import annotations

from .agent_config import AgentSpec, HiveConfig, SupervisorSpec
from .agents import Agent
from .collaboration import CollaborationEngine
from .fallback import FallbackPolicy, effective_policy
from .gateway import ModelGateway
from .orchestrator import Supervisor
from .tools import ToolRegistry, default_registry
from .usage import UsageTracker


def build_agent(
    spec: AgentSpec,
    gateway: ModelGateway,
    tracker: UsageTracker,
    tools: ToolRegistry | None = None,
    collaboration: CollaborationEngine | None = None,
    global_fallback: FallbackPolicy | None = None,
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
    )


def build_supervisor(
    spec: SupervisorSpec,
    agents: dict[str, Agent],
    gateway: ModelGateway,
    tracker: UsageTracker,
    collaboration: CollaborationEngine | None = None,
    global_fallback: FallbackPolicy | None = None,
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
    )


def build_team(
    config: HiveConfig, gateway: ModelGateway, tracker: UsageTracker | None = None
) -> tuple[Supervisor, dict[str, Agent]]:
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
    tools = default_registry()
    agents = {
        spec.name: build_agent(spec, gateway, tracker, tools, engine, config.fallback)
        for spec in config.agents
    }
    engine.register(agents)
    supervisor = build_supervisor(
        config.supervisor, agents, gateway, tracker, engine, config.fallback
    )
    return supervisor, agents
