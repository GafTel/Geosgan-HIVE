from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from .errors import ConfigError, HiveError
from .fallback import FallbackPolicy
from .providers import parse_model
from .run_context import CollaborationPolicy

#: Core roles of the HIVE. The default config defines exactly these five.
CORE_ROLES = ("supervisor", "researcher", "developer", "analyst", "reviewer")


class Permissions(BaseModel):
    """Least privilege by default: NO tools and NO delegation unless explicit."""

    allow_tools: bool = False
    allowed_tools: list[str] = Field(default_factory=list)
    allow_delegation: bool = False
    allowed_agents: list[str] = Field(default_factory=list)  # empty = no targets


class Limits(BaseModel):
    """Per-agent execution limits.

    `max_cost_usd_per_run` is a soft budget: it is only enforced while costs are
    reliably known (cost_usd is None for unknown/local models, in which case no
    budget claim is made).
    """

    max_tool_steps: int = Field(default=5, ge=0)
    max_calls_per_run: int = Field(default=50, ge=1)
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_tokens: int | None = Field(default=None, ge=1)
    max_cost_usd_per_run: float | None = Field(default=None, ge=0)


class AgentSpec(BaseModel):
    """Pydantic-validated configuration of one specialist agent."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    description: str
    system_prompt: str
    model: str  # '<provider>/<model>', e.g. 'mistral/mistral-large-latest'
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    permissions: Permissions = Field(default_factory=Permissions)
    limits: Limits = Field(default_factory=Limits)
    # None = inherit the global `fallback` block; an explicit block (even mode=off)
    # overrides it for this agent. Fallback is opt-in: OFF by default.
    fallback: FallbackPolicy | None = None

    @field_validator("model")
    @classmethod
    def _validate_model(cls, value: str) -> str:
        try:
            parse_model(value)
        except HiveError as exc:
            raise ValueError(str(exc)) from exc
        return value


class SupervisorSpec(AgentSpec):
    max_steps: int = Field(default=8, ge=1)


class HiveConfig(BaseModel):
    """Whole team configuration: one supervisor plus the specialist agents."""

    supervisor: SupervisorSpec
    agents: list[AgentSpec] = Field(min_length=1)
    collaboration: CollaborationPolicy = Field(default_factory=CollaborationPolicy)
    fallback: FallbackPolicy = Field(default_factory=FallbackPolicy)  # global default: OFF

    @model_validator(mode="after")
    def _check_names(self) -> HiveConfig:
        names = [a.name for a in self.agents]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate agent names: {sorted(names)}")
        if self.supervisor.name in names:
            raise ValueError(
                f"supervisor name {self.supervisor.name!r} collides with an agent name"
            )
        known = set(names)
        for spec in self.agents:
            unknown = [t for t in spec.permissions.allowed_agents if t not in known]
            if unknown:
                raise ValueError(
                    f"agent '{spec.name}' may delegate to unknown agents: {unknown}"
                )
        return self

    def agent(self, name: str) -> AgentSpec:
        for spec in self.agents:
            if spec.name == name:
                return spec
        raise ConfigError(f"no agent named {name!r} in config")


#: Scalar fields that can be overridden via env vars (nested models stay YAML-only).
_AGENT_FIELDS = ("name", "description", "system_prompt", "model", "temperature")
_SUPERVISOR_FIELDS = _AGENT_FIELDS + ("max_steps",)
_COLLABORATION_FIELDS = tuple(CollaborationPolicy.model_fields)


def apply_env_overrides(data: dict[str, Any], environ: Mapping[str, str]) -> dict[str, Any]:
    """Apply explicit environment-variable overrides to a raw YAML dict.

    - `HIVE_SUPERVISOR_<FIELD>` overrides a supervisor field (e.g. HIVE_SUPERVISOR_MODEL)
    - `HIVE_AGENT_<NAME>_<FIELD>` overrides an agent field (e.g. HIVE_AGENT_DEVELOPER_MODEL)

    Overrides apply even when the field is absent from the YAML (pydantic
    defaults fill it); values are strings and pydantic coerces them on final
    validation. API keys are never read from YAML: they come from the provider
    env vars at call time.
    """
    data = copy.deepcopy(data)
    sections: list[tuple[str, tuple[str, ...], dict[str, Any]]] = []
    supervisor = data.get("supervisor")
    if isinstance(supervisor, dict):
        sections.append(("HIVE_SUPERVISOR", _SUPERVISOR_FIELDS, supervisor))
    agents = data.get("agents")
    if isinstance(agents, list):
        for agent in agents:
            if isinstance(agent, dict) and isinstance(agent.get("name"), str):
                prefix = "HIVE_AGENT_" + agent["name"].upper().replace("-", "_")
                sections.append((prefix, _AGENT_FIELDS, agent))
    # always present so HIVE_COLLABORATION_* overrides apply even without a YAML block
    collaboration = data.setdefault("collaboration", {})
    if isinstance(collaboration, dict):
        sections.append(("HIVE_COLLABORATION", _COLLABORATION_FIELDS, collaboration))
    for prefix, fields, section in sections:
        for key in fields:
            env_var = f"{prefix}_{key.upper()}"
            if env_var in environ:
                section[key] = environ[env_var]
    return data


def load_hive_config(path: str | Path, *, environ: Mapping[str, str] | None = None) -> HiveConfig:
    """Load and validate the team YAML config, then apply env overrides."""
    env = os.environ if environ is None else environ
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(
            f"Config file not found: {config_path}. "
            "Pass --agents-config or set HIVE_CONFIG to a YAML file."
        )
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(
            f"Config file {config_path} must contain a YAML mapping at the top level."
        )
    data = apply_env_overrides(data, env)
    try:
        return HiveConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration in {config_path}:\n{exc}") from exc
