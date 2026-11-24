"""GEOSGAN HIVE: base of a hybrid multi-agent AI system."""

from .agent_config import (
    CORE_ROLES,
    AgentSpec,
    HiveConfig,
    Limits,
    Permissions,
    SupervisorSpec,
    apply_env_overrides,
    load_hive_config,
)
from .agents import Agent, AgentResult, ToolCall, check_run_limits, parse_tool_call
from .config import Settings, load_settings
from .errors import (
    AuthenticationError,
    BadRequestError,
    BudgetExceededError,
    ConfigError,
    GatewayConnectionError,
    GatewayError,
    GatewayTimeoutError,
    HiveError,
    MissingCredentialsError,
    PermanentGatewayError,
    RateLimitError,
    ServiceUnavailableError,
    TransientGatewayError,
    UnknownProviderError,
)
from .factory import build_agent, build_supervisor, build_team
from .gateway import GatewayLLMClient, LLMResponse, ModelGateway
from .llm import FakeLLM, LLMClient
from .messages import Message
from .orchestrator import FINISH, OrchestratorResult, Supervisor
from .providers import Credentials, Provider, ProviderSpec, parse_model, resolve_credentials
from .retry import RetryPolicy, is_retryable
from .tools import ToolRegistry, default_registry
from .usage import UsageRecord, UsageSummary, UsageTracker

__version__ = "0.2.0"

__all__ = [
    "Agent",
    "AgentResult",
    "AgentSpec",
    "AuthenticationError",
    "BadRequestError",
    "BudgetExceededError",
    "CORE_ROLES",
    "ConfigError",
    "Credentials",
    "FakeLLM",
    "FINISH",
    "GatewayConnectionError",
    "GatewayError",
    "GatewayLLMClient",
    "GatewayTimeoutError",
    "HiveConfig",
    "HiveError",
    "LLMClient",
    "LLMResponse",
    "Limits",
    "Message",
    "MissingCredentialsError",
    "ModelGateway",
    "OrchestratorResult",
    "PermanentGatewayError",
    "Permissions",
    "Provider",
    "ProviderSpec",
    "RateLimitError",
    "RetryPolicy",
    "ServiceUnavailableError",
    "Settings",
    "Supervisor",
    "SupervisorSpec",
    "ToolCall",
    "ToolRegistry",
    "TransientGatewayError",
    "UnknownProviderError",
    "UsageRecord",
    "UsageSummary",
    "UsageTracker",
    "apply_env_overrides",
    "build_agent",
    "build_supervisor",
    "build_team",
    "check_run_limits",
    "default_registry",
    "is_retryable",
    "load_hive_config",
    "load_settings",
    "parse_model",
    "parse_tool_call",
    "resolve_credentials",
]
