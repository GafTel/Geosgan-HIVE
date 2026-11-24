"""Phase 3 T1: ToolGateway core — policy, validation, budgets, trusted context,
fail-closed behavior, redaction, timeout occupancy, audit events, agent hook."""

import asyncio
import threading
import time

import pytest
from pydantic import BaseModel, ConfigDict, Field

from geosgan_hive.agent_config import Permissions
from geosgan_hive.agents import Agent
from geosgan_hive.llm import FakeLLM
from geosgan_hive.run_context import EventType, RunContext
from geosgan_hive.tool_factory import BUILTIN_TOOL_NAMES, build_tool_gateway
from geosgan_hive.tool_gateway import (
    BuiltinToolAdapter,
    ToolExecutionContext,
    ToolGateway,
    redact_secrets,
)
from geosgan_hive.tool_policy import build_policy
from geosgan_hive.tool_registry import ToolCatalog
from geosgan_hive.tool_spec import SideEffect, ToolBudgets, ToolSpec, args_sha256
from geosgan_hive.tools import CalculatorTool, default_registry


class EchoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=1000)


def echo_spec() -> ToolSpec:
    return ToolSpec(
        name="echo", description="echoes text", category="data",
        side_effect=SideEffect.PURE, input_model=EchoArgs, timeout_seconds=5,
    )


class EchoAdapter:
    def __init__(self, spec):
        self.spec = spec

    def execute(self, args, ctx):
        return f"echo:{args['text']}|agent={ctx.agent}|run={ctx.run_id}"

    async def aexecute(self, args, ctx):
        return self.execute(args, ctx)


def make_gateway(**policy_kwargs) -> ToolGateway:
    catalog = ToolCatalog()
    catalog.register(echo_spec(), EchoAdapter(echo_spec()))
    spec = ToolSpec(name="calculator", description="calc", side_effect=SideEffect.PURE,
                    input_model=None)
    catalog.register(spec, BuiltinToolAdapter(CalculatorTool(), spec))
    policy = build_policy(
        global_enabled=["echo", "calculator"],
        agent_permissions={"alice": (True, ["echo", "calculator"]), "bob": (True, [])},
        **policy_kwargs,
    )
    return ToolGateway(catalog, policy)


def ctx(agent="alice", run_id="r1"):
    return ToolExecutionContext(agent=agent, run_id=run_id, task_id="t1")


# ---- policy ----

def test_policy_deny_by_default_and_empty_allowlist_denies_all():
    gw = make_gateway()
    assert gw.policy.allows("alice", "echo") is True
    assert gw.policy.allows("bob", "echo") is False  # explicitly empty allowlist
    assert gw.policy.allows("alice", "nope") is False  # not globally enabled
    assert gw.policy.allows("mallory", "echo") is False  # unknown agent


def test_legacy_mode_keeps_builtins_per_agent_permissions():
    from geosgan_hive.agent_config import AgentSpec, HiveConfig, SupervisorSpec

    config = HiveConfig(
        supervisor=SupervisorSpec(name="supervisor", description="d", system_prompt="s",
                                  model="ollama/s"),
        agents=[AgentSpec(name="a", description="d", system_prompt="s", model="ollama/a",
                          permissions=Permissions(allow_tools=True,
                                                  allowed_tools=["calculator"]))],
    )
    gw = build_tool_gateway(config)  # no tools block -> legacy
    assert gw.policy.allows("a", "calculator") is True
    assert gw.policy.allows("a", "echo") is False  # not a builtin
    assert gw.policy.allows("supervisor", "calculator") is False  # supervisor denied


def test_explicit_tools_block_empty_enabled_denies_even_builtins():
    from geosgan_hive.agent_config import AgentSpec, HiveConfig, SupervisorSpec, ToolsConfig

    config = HiveConfig(
        supervisor=SupervisorSpec(name="supervisor", description="d", system_prompt="s",
                                  model="ollama/s"),
        agents=[AgentSpec(name="a", description="d", system_prompt="s", model="ollama/a",
                          permissions=Permissions(allow_tools=True,
                                                  allowed_tools=["calculator"]))],
        tools=ToolsConfig(enabled=[]),  # explicit empty -> deny ALL
    )
    gw = build_tool_gateway(config)
    assert gw.policy.allows("a", "calculator") is False
    assert set(BUILTIN_TOOL_NAMES)  # sanity: builtins exist in legacy mode only


# ---- gateway enforcement ----

def test_unknown_tool_and_disallowed_tool_are_rejected_not_executed():
    gw = make_gateway()
    r = gw.execute("nope", {}, ctx=ctx())
    assert r.ok is False and r.status == "rejected" and "unknown tool" in r.error
    r = gw.execute("echo", {"text": "hi"}, ctx=ctx(agent="bob"))
    assert r.ok is False and "not allowed" in r.error


def test_strict_gateway_fails_closed_without_run_context():
    catalog = ToolCatalog()
    spec = echo_spec()
    catalog.register(spec, EchoAdapter(spec))
    from geosgan_hive.tool_policy import build_policy
    policy = build_policy(global_enabled=["echo"],
                          agent_permissions={"alice": (True, ["echo"])})
    gw = ToolGateway(catalog, policy, strict=True)
    r = gw.execute("echo", {"text": "hi"}, ctx=ctx(), run_context=None)
    assert r.ok is False and "failing closed" in r.error


def test_unexpected_arguments_rejected_and_input_size_capped():
    gw = make_gateway()
    r = gw.execute("echo", {"text": "hi", "evil": "x"}, ctx=ctx())
    assert r.ok is False and r.status == "rejected" and "invalid arguments" in r.error
    spec_small = ToolSpec(name="echo", description="", side_effect=SideEffect.PURE,
                          input_model=EchoArgs, max_input_bytes=10)
    gw.catalog._tools["echo"].spec = spec_small
    r = gw.execute("echo", {"text": "x" * 500}, ctx=ctx())
    assert r.ok is False and "max_input_bytes" in r.error


def test_trusted_runtime_identity_never_from_model_args():
    gw = make_gateway()
    r = gw.execute("echo", {"text": "hi", }, ctx=ctx(agent="alice", run_id="real-run"))
    assert r.ok is True
    # model-supplied identity inside args must not influence the context
    assert "run=real-run" in r.output
    r2 = gw.execute("echo", {"text": "hi"}, ctx=ToolExecutionContext(agent="alice"))
    assert "run=None" in r2.output


def test_tool_budgets_atomic_per_run_agent_tool():
    gw = make_gateway()
    from geosgan_hive.run_context import CollaborationPolicy
    rc = RunContext("r1", CollaborationPolicy(),
                    tool_budgets=ToolBudgets(max_tool_calls_per_run=3,
                                             max_tool_calls_per_agent=2,
                                             max_tool_calls_per_tool=1))
    ok_calls = 0
    for _ in range(5):
        r = gw.execute("echo", {"text": "x"}, ctx=ctx(), run_context=rc)
        if r.ok:
            ok_calls += 1
    assert ok_calls == 1  # per-tool cap = 1
    assert rc.tool_calls == 1
    r = gw.execute("echo", {"text": "x"}, ctx=ctx(), run_context=rc)
    assert r.status == "rejected" and "limit" in r.error


def test_concurrent_reservations_are_atomic():
    from geosgan_hive.run_context import CollaborationPolicy
    rc = RunContext("r", CollaborationPolicy(),
                    tool_budgets=ToolBudgets(max_tool_calls_per_run=50,
                                             max_tool_calls_per_agent=50,
                                             max_tool_calls_per_tool=50))
    gw = make_gateway()
    outcomes = []
    lock = threading.Lock()

    def worker():
        r = gw.execute("echo", {"text": "x"}, ctx=ctx(), run_context=rc)
        with lock:
            outcomes.append(r.ok)

    threads = [threading.Thread(target=worker) for _ in range(100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(outcomes) == 50
    assert rc.tool_calls == 50


def test_audit_events_recorded_without_payloads():
    from geosgan_hive.run_context import CollaborationPolicy
    rc = RunContext("r1", CollaborationPolicy())
    gw = make_gateway()
    gw.execute("echo", {"text": "secret-ish"}, ctx=ctx(), run_context=rc)
    gw.execute("nope", {}, ctx=ctx(), run_context=rc)
    events = rc.events()
    kinds = [e.event for e in events]
    assert EventType.TOOL_EXECUTED in kinds and EventType.TOOL_REJECTED in kinds
    blob = str([e.metadata for e in events])
    assert "secret-ish" not in blob  # no tool payloads in audit metadata


def test_redaction_and_output_cap():
    assert redact_secrets("key sk-abcdefghijklmnop1234 end") == "key ***REDACTED*** end"
    spec = ToolSpec(name="echo", description="", side_effect=SideEffect.PURE,
                    input_model=EchoArgs, max_output_bytes=10)
    gw = make_gateway()
    gw.catalog._tools["echo"].spec = spec
    r = gw.execute("echo", {"text": "x" * 500}, ctx=ctx())
    assert r.ok is True and r.truncated is True and r.output_bytes <= 30


def test_sync_timeout_keeps_occupancy_until_thread_terminates():
    from geosgan_hive.run_context import CollaborationPolicy

    class SlowAdapter:
        def __init__(self, spec):
            self.spec = spec

        def execute(self, args, ctx):
            time.sleep(0.4)
            return "slow done"

    spec = ToolSpec(name="slow", description="", side_effect=SideEffect.READ_ONLY,
                    input_model=EchoArgs, timeout_seconds=0.05)
    catalog = ToolCatalog()
    catalog.register(spec, SlowAdapter(spec))
    policy = build_policy(global_enabled=["slow"],
                          agent_permissions={"alice": (True, ["slow"])})
    gw = ToolGateway(catalog, policy)
    rc = RunContext("r", CollaborationPolicy(),
                    tool_budgets=ToolBudgets(max_concurrent_tool_executions=1))
    r = gw.execute("slow", {"text": "x"}, ctx=ctx(), run_context=rc)
    assert r.ok is False and r.status == "timeout"
    # occupancy is STILL held right after the timeout (thread still running)
    sem = rc.sync_tool_semaphore()
    assert sem.acquire(blocking=False) is False
    time.sleep(0.5)  # let the thread finish -> watcher releases the slot
    assert sem.acquire(blocking=False) is True
    sem.release()


def test_async_native_execution_and_cancellation():
    gw = make_gateway()

    async def go():
        r = await gw.aexecute("echo", {"text": "hi"}, ctx=ctx())
        return r

    r = asyncio.run(go())
    assert r.ok is True and "agent=alice" in r.output

    class SlowAsync:
        def __init__(self, spec):
            self.spec = spec

        def execute(self, args, ctx):
            time.sleep(5)
            return "never"

        async def aexecute(self, args, ctx):
            await asyncio.sleep(5)
            return "never"

    spec = ToolSpec(name="slows", description="", side_effect=SideEffect.READ_ONLY,
                    input_model=EchoArgs, timeout_seconds=30)
    gw.catalog.register(spec, SlowAsync(spec))
    gw.policy.global_enabled = frozenset({"echo", "calculator", "slows"})
    gw.policy.per_agent["alice"] = frozenset({"echo", "calculator", "slows"})

    async def cancel_go():
        task = asyncio.create_task(gw.aexecute("slows", {"text": "x"}, ctx=ctx()))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_go())  # cancellation propagates; no orphan crash


# ---- agent hook ----

def test_agent_executes_through_gateway_when_wired():
    gw = make_gateway()
    calls = []
    orig = gw.execute

    def spy(tool, args, *, ctx, run_context=None):
        calls.append((tool, ctx.agent, ctx.run_id))
        return orig(tool, args, ctx=ctx, run_context=run_context)

    gw.execute = spy
    agent = Agent("alice", "d", "s",
                  FakeLLM(responses=['{"tool": "echo", "args": {"text": "hi"}}', "done"]),
                  permissions=Permissions(allow_tools=True, allowed_tools=["echo"]),
                  tool_gateway=gw)
    result = agent.run("task", run_id="run-9")
    assert calls == [("echo", "alice", "run-9")]  # trusted runtime identity
    assert result.tool_calls[0].result.startswith("echo:hi")
    assert result.tool_calls[0].tool == "echo"


def test_agent_legacy_path_without_gateway_unchanged():
    agent = Agent("alice", "d", "s",
                  FakeLLM(responses=['{"tool": "calculator", "args": {"expression": "6*7"}}',
                                     "done"]),
                  permissions=Permissions(allow_tools=True, allowed_tools=["calculator"]),
                  tools=default_registry())
    result = agent.run("task")
    assert result.tool_calls[0].result == "42"  # legacy path intact


def test_args_sha256_stable_for_approval_binding():
    assert args_sha256({"b": 1, "a": [1, 2]}) == args_sha256({"a": [1, 2], "b": 1})
    assert args_sha256({"a": 1}) != args_sha256({"a": 2})


def test_factory_wires_gateway_into_agents():
    from geosgan_hive.agent_config import AgentSpec, HiveConfig, SupervisorSpec
    from geosgan_hive.factory import build_team
    from geosgan_hive.gateway import ModelGateway
    from geosgan_hive.usage import UsageTracker

    config = HiveConfig(
        supervisor=SupervisorSpec(name="supervisor", description="d", system_prompt="s",
                                  model="ollama/s"),
        agents=[AgentSpec(name="a", description="d", system_prompt="s", model="ollama/a",
                          permissions=Permissions(allow_tools=True,
                                                  allowed_tools=["calculator"]))],
    )
    gateway = ModelGateway(tracker=UsageTracker())
    _sup, agents = build_team(config, gateway)
    assert agents["a"].tool_gateway is not None  # production always enforces gateway
    assert agents["a"].tool_gateway.policy.allows("a", "calculator") is True
