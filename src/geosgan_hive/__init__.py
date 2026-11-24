"""GEOSGAN HIVE: base of a hybrid multi-agent AI system."""

from .agents import Agent, AgentResult, ToolCall, parse_tool_call
from .config import Settings, load_settings
from .llm import FakeLLM, LiteLLMClient, LLMClient
from .messages import Message
from .orchestrator import FINISH, OrchestratorResult, Supervisor
from .tools import ToolRegistry, default_registry

__version__ = "0.2.0"

__all__ = [
    "Agent",
    "AgentResult",
    "FakeLLM",
    "FINISH",
    "LLMClient",
    "LiteLLMClient",
    "Message",
    "OrchestratorResult",
    "Settings",
    "Supervisor",
    "ToolCall",
    "ToolRegistry",
    "default_registry",
    "load_settings",
    "parse_tool_call",
]
