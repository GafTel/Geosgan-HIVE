import json

from geosgan_hive.agent_config import Limits, Permissions
from geosgan_hive.agents import Agent, parse_tool_call
from geosgan_hive.llm import FakeLLM
from geosgan_hive.tools import CalculatorTool, ToolRegistry


def make_agent(llm, **kwargs):
    kwargs.setdefault(
        "permissions", Permissions(allow_tools=True, allowed_tools=["calculator"])
    )
    return Agent(
        name="tester",
        description="test agent",
        system_prompt="You are a test agent.",
        llm=llm,
        tools=ToolRegistry([CalculatorTool()]),
        **kwargs,
    )


def test_parse_tool_call():
    call = {"tool": "calculator", "args": {"expression": "1+1"}}
    assert parse_tool_call(json.dumps(call)) == call
    assert parse_tool_call("plain text") is None
    assert parse_tool_call('{"tool": "x"}') is None
    assert parse_tool_call("[1, 2]") is None
    # real models sometimes wrap the JSON in markdown fences or prose
    fenced = "```json\n" + json.dumps(call) + "\n```"
    assert parse_tool_call(fenced) == call
    assert parse_tool_call(f"Here you go: {json.dumps(call)} end") == call


def test_agent_uses_tool_then_answers():
    llm = FakeLLM(
        responses=[
            json.dumps({"tool": "calculator", "args": {"expression": "6 * 7"}}),
            "The answer is 42",
        ]
    )
    result = make_agent(llm).run("What is 6 times 7?")
    assert result.answer == "The answer is 42"
    assert result.steps == 2
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool == "calculator"
    assert result.tool_calls[0].result == "42"
    # the second LLM call received the tool result
    assert any("42" in m.content for m in llm.calls[1])


def test_agent_without_tool_call():
    llm = FakeLLM(responses=["direct answer"])
    result = make_agent(llm).run("hello")
    assert result.answer == "direct answer"
    assert result.steps == 1
    assert result.tool_calls == []


def test_agent_tool_step_limit_is_bounded():
    llm = FakeLLM(
        handler=lambda messages: json.dumps(
            {"tool": "calculator", "args": {"expression": "1+1"}}
        )
    )
    result = make_agent(llm, limits=Limits(max_tool_steps=3)).run("add repeatedly")
    assert len(result.tool_calls) == 3
    assert result.steps == 3 + 2


def test_run_id_is_propagated_to_the_llm():
    llm = FakeLLM(responses=["ok"])
    make_agent(llm).run("task", run_id="run-42")
    assert llm.run_ids == ["run-42"]


def test_max_calls_per_run_is_enforced():
    llm = FakeLLM(
        handler=lambda messages: json.dumps(
            {"tool": "calculator", "args": {"expression": "1+1"}}
        )
    )
    agent = make_agent(llm, limits=Limits(max_tool_steps=100, max_calls_per_run=2))
    result = agent.run("keep calling tools", run_id="r1")
    assert "max 2 model calls" in result.answer
    assert len(llm.calls) == 2
