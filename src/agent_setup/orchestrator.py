from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .agents import Agent, parse_json_object
from .llm import LLMClient
from .messages import Message

FINISH = "FINISH"

SUPERVISOR_INSTRUCTIONS = """\
Eres el supervisor de un equipo de agentes especialistas.
Decides qué agente atiende la tarea en cada paso.
Responde ÚNICAMENTE con un JSON:
- Para delegar: {"next": "<nombre_del_agente>", "reason": "<por qué>"}
- Para terminar: {"next": "FINISH", "answer": "<respuesta final al usuario>"}
No añadas texto fuera del JSON."""


@dataclass
class OrchestratorResult:
    task: str
    final_answer: str
    route: list[str] = field(default_factory=list)
    transcript: list[Message] = field(default_factory=list)
    steps: int = 0


def parse_decision(text: str) -> dict[str, Any] | None:
    data = parse_json_object(text)
    if data and isinstance(data.get("next"), str):
        return data
    return None


class Supervisor:
    """Orquestador que enruta la tarea entre agentes hasta obtener una respuesta final."""

    def __init__(
        self,
        agents: Mapping[str, Agent],
        llm: LLMClient,
        *,
        max_steps: int = 8,
        temperature: float = 0.2,
    ) -> None:
        if not agents:
            raise ValueError("Se necesita al menos un agente")
        self.agents = dict(agents)
        self.llm = llm
        self.max_steps = max_steps
        self.temperature = temperature

    def _system_message(self) -> Message:
        roster = "\n".join(f"- {name}: {agent.description}" for name, agent in self.agents.items())
        content = f"Agentes disponibles:\n{roster}\n\n{SUPERVISOR_INSTRUCTIONS}"
        return Message(role="system", content=content, agent="supervisor")

    def run(self, task: str) -> OrchestratorResult:
        transcript = [Message(role="user", content=task, agent="user")]
        route: list[str] = []
        for step in range(1, self.max_steps + 1):
            raw = self.llm.complete(
                [self._system_message(), *transcript], temperature=self.temperature
            )
            transcript.append(Message(role="assistant", content=raw, agent="supervisor"))
            decision = parse_decision(raw)
            if decision is None:
                return OrchestratorResult(
                    task=task, final_answer=raw, route=route, transcript=transcript, steps=step
                )
            nxt = decision["next"]
            if nxt == FINISH:
                answer = str(decision.get("answer") or raw)
                return OrchestratorResult(
                    task=task, final_answer=answer, route=route, transcript=transcript, steps=step
                )
            agent = self.agents.get(nxt)
            if agent is None:
                valid = ", ".join(self.agents)
                transcript.append(
                    Message(
                        role="user",
                        agent="system",
                        content=f"Error: no existe el agente '{nxt}'. Válidos: {valid}.",
                    )
                )
                continue
            route.append(agent.name)
            result = agent.run(task, context=transcript)
            transcript.append(Message(role="assistant", content=result.answer, agent=agent.name))
        return OrchestratorResult(
            task=task,
            final_answer="Se alcanzó el número máximo de pasos sin respuesta final.",
            route=route,
            transcript=transcript,
            steps=self.max_steps,
        )
