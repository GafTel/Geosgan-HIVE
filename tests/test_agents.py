import json

from geosgan_hive.agents import Agent, parse_tool_call
from geosgan_hive.llm import FakeLLM
from geosgan_hive.tools import CalculatorTool, ToolRegistry


def make_agent(llm, **kwargs):
    return Agent(
        name="tester",
        description="agente de prueba",
        system_prompt="Eres un agente de prueba.",
        llm=llm,
        tools=ToolRegistry([CalculatorTool()]),
        **kwargs,
    )


def test_parse_tool_call():
    call = {"tool": "calculator", "args": {"expression": "1+1"}}
    assert parse_tool_call(json.dumps(call)) == call
    assert parse_tool_call("texto normal") is None
    assert parse_tool_call('{"tool": "x"}') is None
    assert parse_tool_call("[1, 2]") is None
    # los modelos reales a veces envuelven el JSON en cercas markdown o texto alrededor
    fenced = "```json\n" + json.dumps(call) + "\n```"
    assert parse_tool_call(fenced) == call
    assert parse_tool_call(f"Aquí está: {json.dumps(call)} fin") == call


def test_agent_uses_tool_then_answers():
    llm = FakeLLM(
        responses=[
            json.dumps({"tool": "calculator", "args": {"expression": "6 * 7"}}),
            "La respuesta es 42",
        ]
    )
    result = make_agent(llm).run("¿Cuánto es 6 por 7?")
    assert result.answer == "La respuesta es 42"
    assert result.steps == 2
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool == "calculator"
    assert result.tool_calls[0].result == "42"
    # el segundo llamado al LLM recibió el resultado de la herramienta
    assert any("42" in m.content for m in llm.calls[1])


def test_agent_without_tool_call():
    llm = FakeLLM(responses=["respuesta directa"])
    result = make_agent(llm).run("hola")
    assert result.answer == "respuesta directa"
    assert result.steps == 1
    assert result.tool_calls == []


def test_agent_tool_step_limit_is_bounded():
    llm = FakeLLM(
        handler=lambda messages: json.dumps(
            {"tool": "calculator", "args": {"expression": "1+1"}}
        )
    )
    result = make_agent(llm, max_tool_steps=3).run("suma repetidamente")
    assert len(result.tool_calls) == 3
    assert result.steps == 3 + 2
