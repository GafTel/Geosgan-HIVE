"""Offline demo of the Phase 2 HYBRID COLLABORATION ENGINE (no API keys).

A researcher agent delegates two independent subtasks in parallel to the
analyst and the reviewer through the CollaborationEngine; results come back
in deterministic order and the full run history (task tree + events) is printed.

Run from the repo root:  python examples/demo_collaboration.py
"""

import asyncio
import json

from geosgan_hive.agent_config import Permissions
from geosgan_hive.agents import Agent
from geosgan_hive.collaboration import CollaborationEngine, DelegationStatus
from geosgan_hive.llm import FakeLLM
from geosgan_hive.run_context import CollaborationPolicy, EventType
from geosgan_hive.usage import UsageTracker


async def researcher_handler(messages):
    if any("Delegation result from" in m.content for m in messages):
        return "Researcher final summary combining both delegated results."
    return json.dumps(
        {
            "delegations": [
                {"target": "analyst", "task": "Quantify the findings", "reason": "numbers"},
                {"target": "reviewer", "task": "Review the draft", "reason": "quality"},
            ]
        }
    )


def specialist_handler(name, delay):
    async def handler(messages):
        await asyncio.sleep(delay)  # simulate model latency; runs in parallel
        return f"{name} completed its subtask"

    return handler


def make_agent(name, llm, engine, *, allow_delegation=False, allowed_agents=()):
    return Agent(
        name=name,
        description=f"{name} agent",
        system_prompt=f"You are {name}.",
        llm=llm,
        permissions=Permissions(
            allow_delegation=allow_delegation, allowed_agents=list(allowed_agents)
        ),
        collaboration=engine,
    )


async def main() -> None:
    policy = CollaborationPolicy(allow_agent_to_agent=True, max_parallel_agents=3)
    engine = CollaborationEngine(policy=policy, tracker=UsageTracker())
    agents = {
        "researcher": make_agent(
            "researcher", FakeLLM(handler=researcher_handler), engine,
            allow_delegation=True, allowed_agents=["analyst", "reviewer"],
        ),
        "analyst": make_agent(
            "analyst", FakeLLM(handler=specialist_handler("analyst", 0.08)), engine
        ),
        "reviewer": make_agent(
            "reviewer", FakeLLM(handler=specialist_handler("reviewer", 0.02)), engine
        ),
    }
    engine.register(agents)

    ctx = engine.context_for("demo-run")
    ctx.create_task(
        task_id="t0", parent_task_id=None, agent="supervisor",
        description="Research and review a topic", depth=0,
    )
    result = await agents["researcher"].arun(
        "Research and review a topic", run_id="demo-run", task_id="t0", run_context=ctx
    )

    print("Final answer:", result.answer)
    print("\nHandoffs (in input order):")
    for h in result.handoffs:
        print(
            f"  {h.source_agent} -> {h.target_agent}: {h.status.value} "
            f"(depth {h.delegation_depth}, model_calls {h.model_calls})"
        )
    print("\nTask tree:")
    for t in ctx.tasks():
        parent = t.parent_task_id[:8] if t.parent_task_id else "-"
        print(
            f"  {t.task_id[:8]} parent={parent} agent={t.agent} "
            f"depth={t.depth} status={t.status.value} calls={t.model_calls}"
        )
    print("\nEvents:")
    for e in ctx.events():
        task = e.task_id[:8] if e.task_id else "-"
        print(f"  {e.event.value} task={task} {e.source_agent or ''}->{e.target_agent or ''}")
    budget = ctx.policy.max_model_calls_per_run
    print(f"\nRun-wide model calls: {ctx.model_calls} (budget {budget})")

    assert all(h.status == DelegationStatus.COMPLETED for h in result.handoffs)
    assert [h.target_agent for h in result.handoffs] == ["analyst", "reviewer"]
    assert any(e.event == EventType.RESULT_RETURNED for e in ctx.events())


if __name__ == "__main__":
    asyncio.run(main())
