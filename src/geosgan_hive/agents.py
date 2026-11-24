from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .llm import LLMClient
from .messages import Message
from .tools import ToolRegistry

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


class Agent:
    """Individual agent with a tool-calling loop until it gives its answer."""

    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: str,
        llm: LLMClient,
        tools: ToolRegistry | None = None,
        *,
        max_tool_steps: int = 5,
        temperature: float = 0.2,
    ) -> None:
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.max_tool_steps = max_tool_steps
        self.temperature = temperature

    def _system_message(self) -> Message:
        prompt = self.system_prompt
        schemas = self.tools.schemas()
        if schemas:
            prompt += "\n\nAvailable tools:\n"
            prompt += json.dumps(schemas, ensure_ascii=False, indent=2)
            prompt += "\n\n" + TOOL_CALL_INSTRUCTIONS
        return Message(role="system", content=prompt, agent=self.name)

    def run(self, task: str, context: Sequence[Message] = ()) -> AgentResult:
        messages = [self._system_message(), *context, Message(role="user", content=task)]
        tool_calls: list[ToolCall] = []
        max_iterations = self.max_tool_steps + 2
        raw = ""
        for step in range(1, max_iterations + 1):
            raw = self.llm.complete(messages, temperature=self.temperature)
            call = parse_tool_call(raw)
            if call is None:
                return AgentResult(agent=self.name, answer=raw, tool_calls=tool_calls, steps=step)
            if len(tool_calls) >= self.max_tool_steps:
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
