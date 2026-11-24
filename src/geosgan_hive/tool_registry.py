from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from .tool_spec import ToolSpec

if TYPE_CHECKING:
    from .tool_gateway import ToolExecutionContext
    from .tool_policy import ToolPolicy


class ToolAdapter(Protocol):
    """Execution interface for one tool. `spec` carries all security metadata.
    Sync `execute` is mandatory; `aexecute` defaults to a thread fallback and is
    overridden by natively-async adapters (never the other way around)."""

    spec: ToolSpec

    def execute(self, args: dict[str, Any], ctx: ToolExecutionContext) -> str: ...

    async def aexecute(self, args: dict[str, Any], ctx: ToolExecutionContext) -> str: ...


@dataclass
class RegisteredTool:
    spec: ToolSpec
    adapter: ToolAdapter


class ToolCatalog:
    """Registration and discovery of tools (spec + adapter). The legacy
    `tools.ToolRegistry` is untouched for backward compatibility; production
    execution always goes through the ToolGateway backed by this catalog."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, spec: ToolSpec, adapter: ToolAdapter) -> None:
        if spec.name in self._tools:
            raise ValueError(f"duplicate tool registration: {spec.name!r}")
        self._tools[spec.name] = RegisteredTool(spec=spec, adapter=adapter)

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def specs_for(self, policy: ToolPolicy, agent: str) -> list[ToolSpec]:
        """Only tools the policy currently allows for this agent (for prompts)."""
        return [s for s in self.specs() if policy.allows(agent, s.name)]

    def schemas_for(self, policy: ToolPolicy, agent: str) -> list[dict[str, Any]]:
        """Prompt-facing schemas of allowed tools (name/description/parameters)."""
        out = []
        for spec in self.specs_for(policy, agent):
            schema: dict[str, Any] = {"type": "object"}
            if spec.input_model is not None:
                schema = spec.input_model.model_json_schema()
            out.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": schema,
                }
            )
        return out
