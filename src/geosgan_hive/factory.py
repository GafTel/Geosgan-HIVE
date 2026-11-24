from __future__ import annotations

from .agent_config import AgentSpec, HiveConfig, SupervisorSpec
from .agents import Agent
from .gateway import ModelGateway
from .orchestrator import Supervisor
from .tools import ToolRegistry, default_registry
from .usage import UsageTracker


def build_agent(
    spec: AgentSpec,
    gateway: ModelGateway,
    tracker: UsageTracker,
    tools: ToolRegistry | None = None,
) -> Agent:
    """Build one specialist agent from its spec: own model, permissions, limits."""
    registry = tools if tools is not None else default_registry()
    llm = gateway.bind(
        spec.model,
        agent=spec.name,
        temperature=spec.temperature,
        max_tokens=spec.limits.max_tokens,
        timeout_seconds=spec.limits.timeout_seconds,
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
    )


def build_supervisor(
    spec: SupervisorSpec,
    agents: dict[str, Agent],
    gateway: ModelGateway,
    tracker: UsageTracker,
) -> Supervisor:
    llm = gateway.bind(
        spec.model,
        agent=spec.name,
        temperature=spec.temperature,
        max_tokens=spec.limits.max_tokens,
        timeout_seconds=spec.limits.timeout_seconds,
    )
    return Supervisor(
        agents,
        llm,
        max_steps=spec.max_steps,
        temperature=spec.temperature,
        usage_tracker=tracker,
        limits=spec.limits,
    )


def build_team(
    config: HiveConfig, gateway: ModelGateway, tracker: UsageTracker | None = None
) -> tuple[Supervisor, dict[str, Agent]]:
    """Build the whole HIVE team (supervisor + agents) from a validated config."""
    tracker = tracker if tracker is not None else gateway.tracker
    tools = default_registry()
    agents = {spec.name: build_agent(spec, gateway, tracker, tools) for spec in config.agents}
    supervisor = build_supervisor(config.supervisor, agents, gateway, tracker)
    return supervisor, agents
