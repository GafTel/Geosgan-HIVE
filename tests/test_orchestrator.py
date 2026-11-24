import json

import pytest

from agent_setup.agents import Agent
from agent_setup.llm import FakeLLM
from agent_setup.orchestrator import Supervisor


def make_agents(llm):
    return {
        "investigador": Agent(
            name="investigador",
            description="investiga",
            system_prompt="prompt del investigador",
            llm=llm,
        ),
        "redactor": Agent(
            name="redactor",
            description="redacta",
            system_prompt="prompt del redactor",
            llm=llm,
        ),
    }


def routed_handler(messages):
    if "supervisor" in messages[0].content:
        delegations = sum(1 for m in messages if m.agent in ("investigador", "redactor"))
        if delegations == 0:
            return json.dumps({"next": "investigador", "reason": "hay que investigar"})
        if delegations == 1:
            return json.dumps({"next": "redactor", "reason": "ya hay datos"})
        return json.dumps({"next": "FINISH", "answer": "respuesta final del equipo"})
    return f"salida del agente con {len(messages)} mensajes"


def test_supervisor_routes_and_finishes():
    llm = FakeLLM(handler=routed_handler)
    supervisor = Supervisor(make_agents(llm), llm, max_steps=8)
    result = supervisor.run("Escribe un resumen sobre agentes")
    assert result.route == ["investigador", "redactor"]
    assert result.final_answer == "respuesta final del equipo"
    assert result.steps == 3
    agents_in_transcript = [
        m.agent for m in result.transcript if m.agent in ("investigador", "redactor")
    ]
    assert agents_in_transcript == ["investigador", "redactor"]


def test_supervisor_handles_unknown_agent():
    responses = iter(
        [
            json.dumps({"next": "fantasma"}),
            json.dumps({"next": "FINISH", "answer": "ok"}),
        ]
    )
    llm = FakeLLM(handler=lambda messages: next(responses))
    supervisor = Supervisor(make_agents(llm), llm, max_steps=5)
    result = supervisor.run("tarea cualquiera")
    assert result.final_answer == "ok"
    assert result.route == []
    assert any("no existe el agente" in m.content for m in result.transcript)


def test_supervisor_plain_text_is_final_answer():
    llm = FakeLLM(responses=["respuesta en texto plano"])
    supervisor = Supervisor(make_agents(llm), llm)
    result = supervisor.run("tarea")
    assert result.final_answer == "respuesta en texto plano"
    assert result.steps == 1


def test_supervisor_max_steps_is_bounded():
    llm = FakeLLM(handler=lambda messages: json.dumps({"next": "investigador"}))
    supervisor = Supervisor(make_agents(llm), llm, max_steps=3)
    result = supervisor.run("tarea infinita")
    assert result.steps == 3
    assert result.route == ["investigador"] * 3
    assert "máximo de pasos" in result.final_answer


def test_supervisor_requires_at_least_one_agent():
    with pytest.raises(ValueError):
        Supervisor({}, FakeLLM())
