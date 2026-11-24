"""GEOSGAN HIVE: hybrid multi-agent AI system with a collaboration engine."""

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
from .agents import (
    Agent,
    AgentResult,
    ToolCall,
    arun_agent,
    check_run_limits,
    parse_delegation_call,
    parse_tool_call,
)
from .collaboration import (
    CollaborationEngine,
    DelegationProposal,
    DelegationStatus,
    HandoffRequest,
    HandoffResult,
)
from .config import Settings, load_settings
from .errors import (
    AuthenticationError,
    BadRequestError,
    BudgetExceededError,
    ConfigError,
    DelegationError,
    DelegationRejectedError,
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
from .llm import AsyncLLMClient, FakeLLM, LLMClient
from .messages import Message
from .orchestrator import FINISH, OrchestratorResult, Supervisor
from .providers import Credentials, Provider, ProviderSpec, parse_model, resolve_credentials
from .retry import RetryPolicy, is_retryable
from .run_context import (
    CollaborationEvent,
    CollaborationPolicy,
    EventType,
    RunContext,
    TaskNode,
    TaskStatus,
)
from .tools import ToolRegistry, default_registry
from .usage import UsageRecord, UsageSummary, UsageTracker

__version__ = "0.3.0"

__all__ = [
    "Agent",
    "AgentResult",
    "AgentSpec",
    "AsyncLLMClient",
    "AuthenticationError",
    "BadRequestError",
    "BudgetExceededError",
    "CORE_ROLES",
    "CollaborationEngine",
    "CollaborationEvent",
    "CollaborationPolicy",
    "ConfigError",
    "Credentials",
    "DelegationError",
    "DelegationProposal",
    "DelegationRejectedError",
    "DelegationStatus",
    "EventType",
    "FakeLLM",
    "FINISH",
    "GatewayConnectionError",
    "GatewayError",
    "GatewayLLMClient",
    "GatewayTimeoutError",
    "HandoffRequest",
    "HandoffResult",
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
    "RunContext",
    "ServiceUnavailableError",
    "Settings",
    "Supervisor",
    "SupervisorSpec",
    "TaskNode",
    "TaskStatus",
    "ToolCall",
    "ToolRegistry",
    "TransientGatewayError",
    "UnknownProviderError",
    "UsageRecord",
    "UsageSummary",
    "UsageTracker",
    "apply_env_overrides",
    "arun_agent",
    "build_agent",
    "build_supervisor",
    "build_team",
    "check_run_limits",
    "default_registry",
    "is_retryable",
    "load_hive_config",
    "load_settings",
    "parse_delegation_call",
    "parse_model",
    "parse_tool_call",
    "resolve_credentials",
]
