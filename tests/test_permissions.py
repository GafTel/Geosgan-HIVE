import json
from datetime import UTC, datetime

from geosgan_hive.agent_config import Limits, Permissions
from geosgan_hive.agents import Agent
from geosgan_hive.errors import ConfigError
from geosgan_hive.llm import FakeLLM
from geosgan_hive.tools import CalculatorTool, ToolRegistry, WordCountTool, default_registry
from geosgan_hive.usage import UsageRecord, UsageTracker


def make_agent(llm, **kwargs):
    kwargs.setdefault("tools", default_registry())
    return Agent(
        name="tester",
        description="test agent",
        system_prompt="You are a test agent.",
        llm=llm,
        **kwargs,
    )


def tool_call(name, **args):
    return json.dumps({"tool": name, "args": args})


def test_least_privilege_default_no_tools_visible():
    llm = FakeLLM(responses=["answer"])
    agent = make_agent(llm)  # no permissions -> allow_tools=False
    result = agent.run("hello")
    assert result.answer == "answer"
    system_msg = llm.calls[0][0].content
    assert "Available tools" not in system_msg
    assert agent.tools.names() == []


def test_disallowed_tool_call_is_rejected_not_executed():
    llm = FakeLLM(
        responses=[
            tool_call("calculator", expression="2+2"),
            "ok, I cannot use it",
        ]
    )
    agent = make_agent(llm)  # default: no tools allowed
    result = agent.run("use the calculator")
    assert len(result.tool_calls) == 1
    assert "unknown tool" in result.tool_calls[0].result  # rejected, never executed
    assert "2+2" not in llm.calls[1][-1].content or "unknown tool" in llm.calls[1][-1].content


def test_only_explicitly_allowed_tools_are_visible():
    llm = FakeLLM(responses=[tool_call("word_count", text="hello world"), "words=2"])
    agent = make_agent(
        llm, permissions=Permissions(allow_tools=True, allowed_tools=["word_count"])
    )
    result = agent.run("count words")
    system_msg = llm.calls[0][0].content
    assert "word_count" in system_msg
    assert "calculator" not in system_msg  # registered globally, but not for this agent
    assert "current_datetime" not in system_msg
    assert result.tool_calls[0].result == "words=2 characters=11"


def test_unknown_tool_in_allowed_tools_is_a_config_error():
    llm = FakeLLM(responses=["x"])
    try:
        make_agent(llm, permissions=Permissions(allow_tools=True, allowed_tools=["nuke"]))
    except ConfigError as exc:
        assert "nuke" in str(exc)
    else:
        raise AssertionError("expected ConfigError")


def make_cost_record(cost, run_id="r1", agent="tester"):
    return UsageRecord(
        agent=agent, model="openai/gpt-4o-mini", provider="openai",
        prompt_tokens=1, completion_tokens=1, total_tokens=2,
        cost_usd=cost, latency_ms=1.0, attempts=1,
        timestamp=datetime.now(UTC).isoformat(), run_id=run_id,
    )


def test_cost_budget_enforced_when_cost_is_known():
    tracker = UsageTracker()
    tracker.record(make_cost_record(2.0))  # known spend already above the limit
    llm = FakeLLM(responses=["should not be called"])
    agent = make_agent(
        llm,
        limits=Limits(max_cost_usd_per_run=1.0),
        usage_tracker=tracker,
        permissions=Permissions(allow_tools=True, allowed_tools=["calculator"]),
    )
    result = agent.run("task", run_id="r1")
    assert "Cost budget reached" in result.answer
    assert llm.calls == []  # stopped before any model call


def test_cost_budget_not_enforced_when_cost_unknown():
    tracker = UsageTracker()
    tracker.record(make_cost_record(None))  # unknown cost -> None, never 0
    llm = FakeLLM(responses=["called anyway"])
    agent = make_agent(
        llm,
        limits=Limits(max_cost_usd_per_run=0.0001),
        usage_tracker=tracker,
        permissions=Permissions(allow_tools=True, allowed_tools=["calculator"]),
    )
    result = agent.run("task", run_id="r1")
    assert result.answer == "called anyway"  # no budget claim on unreliable cost data
    assert len(llm.calls) == 1


def test_registry_still_contains_all_tools_globally():
    registry = ToolRegistry([CalculatorTool(), WordCountTool()])
    assert set(registry.names()) == {"calculator", "word_count"}
