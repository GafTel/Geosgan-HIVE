from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .agent_config import Limits, Permissions
from .errors import ConfigError
from .llm import LLMClient
from .messages import Message
from .tools import ToolRegistry
from .usage import UsageTracker

TOOL_CALL_INSTRUCTIONS = """\
When you need to use a tool, reply ONLY with a JSON like:
{"tool": "<tool_name>", "args": {"<parameter>": "<value>"}}
Do not add any text outside the JSON. When you have the final answer, reply in plain text."""


@dataclass
class ToolCall:
    tool: str
    args: dict[str, Any]
    result: str


@dataclass
class AgentResult:
    agent: str
    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    steps: int = 0


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:]).removesuffix("```").strip()
    if not (text.startswith("{") and text.endswith("}")):
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        text = text[start : end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def parse_tool_call(text: str) -> dict[str, Any] | None:
    data = parse_json_object(text)
    if data and isinstance(data.get("tool"), str) and isinstance(data.get("args"), dict):
        return data
    return None


def check_run_limits(
    tracker: UsageTracker | None,
    run_id: str | None,
    calls: int,
    limits: Limits,
) -> str | None:
    """Return a stop message when a hard per-run limit is reached.

    The cost budget is a soft limit: it is only enforced while the run's cost is
    reliably known (`cost_for_run` returns None when any call of the run has an
    unknown cost, e.g. a local model), so we never claim budget enforcement on
    unreliable cost data.
    """
    if calls >= limits.max_calls_per_run:
        return f"Execution limit reached: max {limits.max_calls_per_run} model calls per run."
    if tracker is not None and run_id is not None and limits.max_cost_usd_per_run is not None:
        cost = tracker.cost_for_run(run_id)
        if cost is not None and cost >= limits.max_cost_usd_per_run:
            return (
                f"Cost budget reached: known spend ${cost:.4f} >= "
                f"limit ${limits.max_cost_usd_per_run:.4f}."
            )
    return None


class Agent:
    """Individual agent with a bounded tool-calling loop until it answers.

    Least privilege: only the tools explicitly allowed in `permissions` are
    visible to the model. Any other tool call is rejected by the filtered
    registry ("unknown tool"), never executed.
    """

    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: str,
        llm: LLMClient,
        tools: ToolRegistry | None = None,
        *,
        permissions: Permissions | None = None,
        limits: Limits | None = None,
        usage_tracker: UsageTracker | None = None,
        temperature: float = 0.2,
    ) -> None:
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.llm = llm
        self.permissions = permissions or Permissions()
        self.limits = limits or Limits()
        self.usage_tracker = usage_tracker
        self.temperature = temperature
        self.tools = self._effective_registry(tools or ToolRegistry())

    def _effective_registry(self, registry: ToolRegistry) -> ToolRegistry:
        if not self.permissions.allow_tools:
            return ToolRegistry()
        allowed = []
        for name in self.permissions.allowed_tools:
            tool = registry.get(name)
            if tool is None:
                raise ConfigError(
                    f"Agent '{self.name}' allows unknown tool '{name}'. "
                    f"Registered tools: {registry.names()}"
                )
            allowed.append(tool)
        return ToolRegistry(allowed)

    def _system_message(self) -> Message:
        prompt = self.system_prompt
        schemas = self.tools.schemas()
        if schemas:
            prompt += "\n\nAvailable tools:\n"
            prompt += json.dumps(schemas, ensure_ascii=False, indent=2)
            prompt += "\n\n" + TOOL_CALL_INSTRUCTIONS
        return Message(role="system", content=prompt, agent=self.name)

    def run(
        self, task: str, context: Sequence[Message] = (), *, run_id: str | None = None
    ) -> AgentResult:
        messages = [self._system_message(), *context, Message(role="user", content=task)]
        tool_calls: list[ToolCall] = []
        calls = 0
        max_iterations = self.limits.max_tool_steps + 2
        raw = ""
        for step in range(1, max_iterations + 1):
            stop = check_run_limits(self.usage_tracker, run_id, calls, self.limits)
            if stop is not None:
                return AgentResult(
                    agent=self.name, answer=stop, tool_calls=tool_calls, steps=step - 1
                )
            raw = self.llm.complete(messages, temperature=self.temperature, run_id=run_id)
            calls += 1
            call = parse_tool_call(raw)
            if call is None:
                return AgentResult(agent=self.name, answer=raw, tool_calls=tool_calls, steps=step)
            if len(tool_calls) >= self.limits.max_tool_steps:
                messages.append(Message(role="assistant", content=raw, agent=self.name))
                messages.append(
                    Message(
                        role="user",
                        agent="system",
                        content=(
                            "You have reached the maximum number of tool calls. "
                            "Reply now with the final answer in plain text."
                        ),
                    )
                )
                continue
            result = self.tools.run(call["tool"], call["args"])
            tool_calls.append(ToolCall(tool=call["tool"], args=call["args"], result=result))
            messages.append(Message(role="assistant", content=raw, agent=self.name))
            messages.append(
                Message(
                    role="user",
                    agent="tool",
                    content=f"Result of '{call['tool']}': {result}",
                )
            )
        return AgentResult(agent=self.name, answer=raw, tool_calls=tool_calls, steps=max_iterations)
