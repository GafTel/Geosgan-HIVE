from __future__ import annotations

import copy
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from .checkpoints import PersistMode
from .errors import ConfigError, HiveError
from .fallback import FallbackMode, FallbackPolicy
from .providers import parse_model
from .run_context import CollaborationPolicy
from .tool_spec import ToolBudgets

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


class CheckpointConfig(BaseModel):
    """Durable checkpoint store (SQLite, stdlib only). No credentials are ever stored.

    `persist_mode=metadata_only` stores state/counters/completed-step metadata but
    NOT the transcript (no durable-resume claim); `resumable` additionally stores
    the transcript in plaintext SQLite (protect the file locally; no encryption at
    rest is claimed). Durable resume requires `resumable`.
    """

    path: str = ".geosgan_hive/checkpoints.db"
    persist_mode: PersistMode = PersistMode.METADATA_ONLY
    lease_ttl_seconds: float = Field(default=60.0, gt=0)
    worker_poll_seconds: float = Field(default=5.0, ge=1.0)  # recovery worker cadence

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("checkpoint.path must be non-empty")
        return value


class ApprovalConfig(BaseModel):
    """Human-approval policy for sensitive tool actions (CLI-only in Phase 3)."""

    expiry_seconds: float = Field(default=600.0, gt=0)
    preview_bytes: int = Field(default=2_000, ge=1)  # bounded preview shown to the operator


class ToolsConfig(BaseModel):
    """Tool Engine configuration (Phase 3).

    Distinguishes ABSENT configuration (HiveConfig.tools is None -> legacy mode:
    previously authorized built-in tools keep working) from an EXPLICIT block
    (enabled defaults to [] -> deny ALL tools unless listed). New capabilities are
    disabled by default. No credentials ever live here.
    """

    enabled: list[str] = Field(default_factory=list)  # empty = deny all tools
    workspace_root: str = "workspace"  # file tools are sandboxed to this root
    allow_write: bool = False  # fs_write/fs_update additionally need this + approval
    budgets: ToolBudgets = Field(default_factory=ToolBudgets)
    approval: ApprovalConfig = Field(default_factory=ApprovalConfig)
    #: name patterns file tools must never expose (credential files, checkpoint DBs)
    deny_name_patterns: list[str] = Field(
        default_factory=lambda: [".env", ".env.*", "*.db", "*.db-wal", "*.db-shm", ".git"]
    )

    @field_validator("enabled")
    @classmethod
    def _validate_enabled(cls, value: list[str]) -> list[str]:
        for name in value:
            if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                raise ValueError(f"invalid tool name {name!r}")
        return value


class HiveConfig(BaseModel):
    """Whole team configuration: one supervisor plus the specialist agents."""

    supervisor: SupervisorSpec
    agents: list[AgentSpec] = Field(min_length=1)
    collaboration: CollaborationPolicy = Field(default_factory=CollaborationPolicy)
    fallback: FallbackPolicy = Field(default_factory=FallbackPolicy)  # global default: OFF
    checkpoint: CheckpointConfig | None = None  # required for WAIT/HYBRID modes
    tools: ToolsConfig | None = None  # None = legacy built-in tools only; block = explicit

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
        # incompatible configurations are rejected at load time, never silently fixed
        if self.fallback.mode in (FallbackMode.WAIT, FallbackMode.HYBRID):
            if self.checkpoint is None:
                raise ValueError(
                    f"fallback mode {self.fallback.mode.value!r} suspends runs and therefore "
                    "requires a checkpoint store: set a 'checkpoint:' block in the config "
                    "(durable resume additionally requires checkpoint.persist_mode=resumable)"
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
_FALLBACK_FIELDS = tuple(FallbackPolicy.model_fields)
_CHECKPOINT_FIELDS = tuple(CheckpointConfig.model_fields)
#: List-valued fields arrive as comma-separated env strings and are split pre-validation.
_LIST_FIELDS = {"fallback_models", "allowed_providers", "allowed_fallback_providers"}
#: Per-agent fallback overrides (Phase 2.5): HIVE_AGENT_<NAME>_FALLBACK_MODE and
#: HIVE_AGENT_<NAME>_FALLBACK_MODELS (comma-separated) -> agent fallback block
_AGENT_FALLBACK_ENV = (("MODE", "mode"), ("MODELS", "fallback_models"))



def apply_env_overrides(data: dict[str, Any], environ: Mapping[str, str]) -> dict[str, Any]:
    """Apply explicit environment-variable overrides to a raw YAML dict.

    - `HIVE_SUPERVISOR_<FIELD>` overrides a supervisor field (e.g. HIVE_SUPERVISOR_MODEL)
    - `HIVE_AGENT_<NAME>_<FIELD>` overrides an agent field (e.g. HIVE_AGENT_DEVELOPER_MODEL)
    - Phase 2.5: `HIVE_FALLBACK_<FIELD>` (global fallback block),
      `HIVE_CHECKPOINT_<FIELD>` (checkpoint block; only applied when at least one
      HIVE_CHECKPOINT_* var is set), and per-agent `HIVE_AGENT_<NAME>_FALLBACK_MODE` /
      `HIVE_AGENT_<NAME>_FALLBACK_MODELS` (comma-separated)

    Overrides apply even when the field is absent from the YAML (pydantic
    defaults fill it); values are strings and pydantic coerces them on final
    validation (list fields are split on commas first). API keys are never read
    from YAML: they come from the provider env vars at call time.
    """
    data = copy.deepcopy(data)
    sections: list[tuple[str, tuple[str, ...], dict[str, Any]]] = []
    supervisor = data.get("supervisor")
    if isinstance(supervisor, dict):
        sections.append(("HIVE_SUPERVISOR", _SUPERVISOR_FIELDS, supervisor))
    agents = data.get("agents")
    agent_prefixes: list[tuple[str, dict[str, Any]]] = []
    if isinstance(agents, list):
        for agent in agents:
            if isinstance(agent, dict) and isinstance(agent.get("name"), str):
                prefix = "HIVE_AGENT_" + agent["name"].upper().replace("-", "_")
                sections.append((prefix, _AGENT_FIELDS, agent))
                agent_prefixes.append((prefix, agent))
    # always present so HIVE_COLLABORATION_* overrides apply even without a YAML block
    collaboration = data.setdefault("collaboration", {})
    if isinstance(collaboration, dict):
        sections.append(("HIVE_COLLABORATION", _COLLABORATION_FIELDS, collaboration))
    # Phase 2.5 global blocks
    fallback = data.setdefault("fallback", {})
    if isinstance(fallback, dict):
        sections.append(("HIVE_FALLBACK", _FALLBACK_FIELDS, fallback))
    if any(k.startswith("HIVE_CHECKPOINT_") for k in environ):
        checkpoint = data.setdefault("checkpoint", {})
        if isinstance(checkpoint, dict):
            sections.append(("HIVE_CHECKPOINT", _CHECKPOINT_FIELDS, checkpoint))
    # Phase 3: HIVE_TOOLS_* overrides create an explicit tools block (absent block
    # stays absent -> legacy mode is preserved)
    if any(k.startswith("HIVE_TOOLS_") for k in environ):
        tools = data.setdefault("tools", {})
        if isinstance(tools, dict):
            for key in ("enabled", "workspace_root", "allow_write"):
                env_var = f"HIVE_TOOLS_{key.upper()}"
                if env_var in environ:
                    value = environ[env_var]
                    tools[key] = (
                        [v.strip() for v in value.split(",") if v.strip()]
                        if key == "enabled"
                        else value
                    )
            if "HIVE_TOOLS_APPROVAL_EXPIRY_SECONDS" in environ:
                tools.setdefault("approval", {})["expiry_seconds"] = environ[
                    "HIVE_TOOLS_APPROVAL_EXPIRY_SECONDS"
                ]
            if "HIVE_TOOLS_BUDGETS_MAX_TOOL_CALLS_PER_RUN" in environ:
                tools.setdefault("budgets", {})["max_tool_calls_per_run"] = environ[
                    "HIVE_TOOLS_BUDGETS_MAX_TOOL_CALLS_PER_RUN"
                ]
    for prefix, fields, section in sections:
        for key in fields:
            env_var = f"{prefix}_{key.upper()}"
            if env_var in environ:
                value = environ[env_var]
                section[key] = (
                    [v.strip() for v in value.split(",") if v.strip()]
                    if key in _LIST_FIELDS
                    else value
                )
    # per-agent fallback overrides; an agent block without an explicit mode
    # inherits the global fallback mode (default: off)
    global_mode = (
        fallback.get("mode", FallbackMode.OFF.value)
        if isinstance(fallback, dict)
        else FallbackMode.OFF.value
    )
    for prefix, agent in agent_prefixes:
        fb: dict[str, Any] | None = None
        for env_key, field in _AGENT_FALLBACK_ENV:
            env_var = f"{prefix}_FALLBACK_{env_key}"
            if env_var in environ:
                fb = agent.setdefault("fallback", {})
                value = environ[env_var]
                fb[field] = (
                    [v.strip() for v in value.split(",") if v.strip()]
                    if field == "fallback_models"
                    else value
                )
        if fb is not None:
            fb.setdefault("mode", global_mode)
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
