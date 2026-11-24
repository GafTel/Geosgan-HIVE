import json

import pytest

from geosgan_hive.agents import Agent
from geosgan_hive.errors import GatewayError, RateLimitError
from geosgan_hive.llm import FakeLLM
from geosgan_hive.orchestrator import Supervisor


def make_agents(llm):
    return {
        "researcher": Agent(
            name="researcher",
            description="investigates",
            system_prompt="researcher prompt",
            llm=llm,
        ),
        "writer": Agent(
            name="writer",
            description="writes",
            system_prompt="writer prompt",
            llm=llm,
        ),
    }


def routed_handler(messages):
    if "supervisor" in messages[0].content:
        delegations = sum(1 for m in messages if m.agent in ("researcher", "writer"))
        if delegations == 0:
            return json.dumps({"next": "researcher", "reason": "need research"})
        if delegations == 1:
            return json.dumps({"next": "writer", "reason": "data ready"})
        return json.dumps({"next": "FINISH", "answer": "final team answer"})
    return f"agent output with {len(messages)} messages"


def test_supervisor_routes_and_finishes():
    llm = FakeLLM(handler=routed_handler)
    supervisor = Supervisor(make_agents(llm), llm, max_steps=8)
    result = supervisor.run("Write a summary about agents")
    assert result.route == ["researcher", "writer"]
    assert result.final_answer == "final team answer"
    assert result.steps == 3
    agents_in_transcript = [
        m.agent for m in result.transcript if m.agent in ("researcher", "writer")
    ]
    assert agents_in_transcript == ["researcher", "writer"]
    assert result.run_id  # generated when not provided


def test_supervisor_propagates_run_id_to_all_llm_calls():
    llm = FakeLLM(handler=routed_handler)
    supervisor = Supervisor(make_agents(llm), llm, max_steps=8)
    result = supervisor.run("task", run_id="run-xyz")
    assert result.run_id == "run-xyz"
    assert set(llm.run_ids) == {"run-xyz"}


def test_supervisor_handles_unknown_agent():
    responses = iter(
        [
            json.dumps({"next": "ghost"}),
            json.dumps({"next": "FINISH", "answer": "ok"}),
        ]
    )
    llm = FakeLLM(handler=lambda messages: next(responses))
    supervisor = Supervisor(make_agents(llm), llm, max_steps=5)
    result = supervisor.run("any task")
    assert result.final_answer == "ok"
    assert result.route == []
    assert any("does not exist" in m.content for m in result.transcript)


def test_supervisor_plain_text_is_final_answer():
    llm = FakeLLM(responses=["plain text answer"])
    supervisor = Supervisor(make_agents(llm), llm)
    result = supervisor.run("task")
    assert result.final_answer == "plain text answer"
    assert result.steps == 1


def test_supervisor_max_steps_is_bounded():
    llm = FakeLLM(handler=lambda messages: json.dumps({"next": "researcher"}))
    supervisor = Supervisor(make_agents(llm), llm, max_steps=3)
    result = supervisor.run("endless task")
    assert result.steps == 3
    assert result.route == ["researcher"] * 3
    assert "Maximum number of steps" in result.final_answer


def test_supervisor_requires_at_least_one_agent():
    with pytest.raises(ValueError):
        Supervisor({}, FakeLLM())


def test_supervisor_surfaces_gateway_errors_as_final_answer():
    def boom(messages):
        raise RateLimitError("429 after retries")

    llm = FakeLLM(handler=boom)
    supervisor = Supervisor(make_agents(llm), llm)
    result = supervisor.run("task")
    assert isinstance(result.final_answer, str)
    assert "gateway error" in result.final_answer.lower()


def test_supervisor_continues_when_an_agent_raises_gateway_error():
    state = {"supervisor_calls": 0}

    def handler(messages):
        if "Available agents" in messages[0].content:
            state["supervisor_calls"] += 1
            if state["supervisor_calls"] == 1:
                return json.dumps({"next": "researcher", "reason": "go"})
            return json.dumps({"next": "FINISH", "answer": "recovered"})
        raise GatewayError("provider down")

    llm = FakeLLM(handler=handler)
    supervisor = Supervisor(make_agents(llm), llm, max_steps=5)
    result = supervisor.run("task")
    assert result.final_answer == "recovered"
    assert result.route == ["researcher"]
    assert any("Error from agent 'researcher'" in m.content for m in result.transcript)
