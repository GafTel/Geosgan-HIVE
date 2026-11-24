"""GEOSGAN HIVE: base of a hybrid multi-agent AI system."""

from .agents import Agent, AgentResult, ToolCall, parse_tool_call
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
from .gateway import GatewayLLMClient, LLMResponse, ModelGateway
from .llm import FakeLLM, LiteLLMClient, LLMClient
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
    "AuthenticationError",
    "BadRequestError",
    "BudgetExceededError",
    "ConfigError",
    "Credentials",
    "FakeLLM",
    "FINISH",
    "GatewayConnectionError",
    "GatewayError",
    "GatewayLLMClient",
    "GatewayTimeoutError",
    "HiveError",
    "LLMClient",
    "LLMResponse",
    "LiteLLMClient",
    "Message",
    "MissingCredentialsError",
    "ModelGateway",
    "OrchestratorResult",
    "PermanentGatewayError",
    "Provider",
    "ProviderSpec",
    "RateLimitError",
    "RetryPolicy",
    "ServiceUnavailableError",
    "Settings",
    "Supervisor",
    "ToolCall",
    "ToolRegistry",
    "TransientGatewayError",
    "UnknownProviderError",
    "UsageRecord",
    "UsageSummary",
    "UsageTracker",
    "default_registry",
    "is_retryable",
    "load_settings",
    "parse_model",
    "parse_tool_call",
    "resolve_credentials",
]
