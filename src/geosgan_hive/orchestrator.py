from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from .agent_config import Limits
from .agents import Agent, check_run_limits, parse_json_object
from .errors import GatewayError
from .llm import LLMClient
from .messages import Message
from .usage import UsageTracker

FINISH = "FINISH"

SUPERVISOR_INSTRUCTIONS = """\
You are the supervisor of a team of specialist agents.
You decide which agent handles the task at each step.
Reply ONLY with a JSON:
- To delegate: {"next": "<agent_name>", "reason": "<why>"}
- To finish: {"next": "FINISH", "answer": "<final answer to the user>"}
Do not add any text outside the JSON."""


@dataclass
class OrchestratorResult:
    task: str
    final_answer: str
    route: list[str] = field(default_factory=list)
    transcript: list[Message] = field(default_factory=list)
    steps: int = 0
    run_id: str | None = None


def parse_decision(text: str) -> dict[str, Any] | None:
    data = parse_json_object(text)
    if data and isinstance(data.get("next"), str):
        return data
    return None


class Supervisor:
    """Orchestrator that routes the task between agents until a final answer.

    Runs on its own model (bound via the gateway) and shares the run's
    usage tracker so per-run limits and cost tracking cover the whole team.
    """

    def __init__(
        self,
        agents: Mapping[str, Agent],
        llm: LLMClient,
        *,
        max_steps: int = 8,
        temperature: float = 0.2,
        usage_tracker: UsageTracker | None = None,
        limits: Limits | None = None,
    ) -> None:
        if not agents:
            raise ValueError("At least one agent is required")
        self.agents = dict(agents)
        self.llm = llm
        self.max_steps = max_steps
        self.temperature = temperature
        self.usage_tracker = usage_tracker
        self.limits = limits or Limits()

    def _system_message(self) -> Message:
        roster = "\n".join(f"- {name}: {agent.description}" for name, agent in self.agents.items())
        content = f"Available agents:\n{roster}\n\n{SUPERVISOR_INSTRUCTIONS}"
        return Message(role="system", content=content, agent="supervisor")

    def run(self, task: str, *, run_id: str | None = None) -> OrchestratorResult:
        run_id = run_id or uuid4().hex
        transcript = [Message(role="user", content=task, agent="user")]
        route: list[str] = []
        calls = 0
        for step in range(1, self.max_steps + 1):
            stop = check_run_limits(self.usage_tracker, run_id, calls, self.limits)
            if stop is not None:
                return OrchestratorResult(
                    task=task, final_answer=stop, route=route, transcript=transcript,
                    steps=step, run_id=run_id,
                )
            try:
                raw = self.llm.complete(
                    [self._system_message(), *transcript],
                    temperature=self.temperature,
                    run_id=run_id,
                )
            except GatewayError as exc:
                return OrchestratorResult(
                    task=task, final_answer=f"Model gateway error: {exc}", route=route,
                    transcript=transcript, steps=step, run_id=run_id,
                )
            calls += 1
            transcript.append(Message(role="assistant", content=raw, agent="supervisor"))
            decision = parse_decision(raw)
            if decision is None:
                return OrchestratorResult(
                    task=task, final_answer=raw, route=route, transcript=transcript,
                    steps=step, run_id=run_id,
                )
            nxt = decision["next"]
            if nxt == FINISH:
                answer = str(decision.get("answer") or raw)
                return OrchestratorResult(
                    task=task, final_answer=answer, route=route, transcript=transcript,
                    steps=step, run_id=run_id,
                )
            agent = self.agents.get(nxt)
            if agent is None:
                valid = ", ".join(self.agents)
                transcript.append(
                    Message(
                        role="user",
                        agent="system",
                        content=f"Error: agent '{nxt}' does not exist. Valid: {valid}.",
                    )
                )
                continue
            route.append(agent.name)
            try:
                result = agent.run(task, context=transcript, run_id=run_id)
            except GatewayError as exc:
                transcript.append(
                    Message(
                        role="user",
                        agent="system",
                        content=f"Error from agent '{nxt}': {exc}",
                    )
                )
                continue
            transcript.append(Message(role="assistant", content=result.answer, agent=agent.name))
        return OrchestratorResult(
            task=task,
            final_answer="Maximum number of steps reached without a final answer.",
            route=route,
            transcript=transcript,
            steps=self.max_steps,
            run_id=run_id,
        )
