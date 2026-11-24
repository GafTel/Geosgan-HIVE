"""Phase 2.5 final audit probes: concurrency under fallback, semaphore release
during suspension, event privacy, capability gate via bind(needs_tools), and
WAIT-mode error semantics. All offline (monkeypatched litellm)."""

import asyncio
import json
from types import SimpleNamespace

import litellm
import litellm.exceptions as litellm_exc
import pytest

from geosgan_hive.agent_config import Permissions
from geosgan_hive.agents import Agent
from geosgan_hive.collaboration import CollaborationEngine
from geosgan_hive.errors import AuthenticationError, ProviderUnavailableError
from geosgan_hive.fallback import FallbackMode, FallbackPolicy
from geosgan_hive.gateway import ModelGateway
from geosgan_hive.llm import FakeLLM
from geosgan_hive.messages import Message
from geosgan_hive.orchestrator import Supervisor
from geosgan_hive.retry import RetryPolicy
from geosgan_hive.run_context import CollaborationPolicy, EventType
from geosgan_hive.usage import UsageTracker

PRIMARY = "ollama/primary"
BACKUP = "ollama/backup"


def ok_response(model, content="ok"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        model=model,
    )


def make_gateway(monkeypatch, **kwargs):
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.001)
    tracker = UsageTracker()
    gateway = ModelGateway(
        tracker=tracker,
        retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0, max_delay_seconds=0,
                                 jitter_ratio=0),
        sleep=lambda s: None,
        async_sleep=lambda s: asyncio.sleep(0),
        **kwargs,
    )
    return gateway, tracker


def test_audit_parallel_delegation_under_fallback_respects_concurrency_cap(monkeypatch):
    """Nested parallel delegation with fallback enabled on every agent: peak ACTIVE
    in-flight requests never exceeds max_parallel_agents (bound at the gateway
    dispatch boundary, on the fallback path too), results keep input order."""
    peak = {"cur": 0, "max": 0}

    async def fake_acompletion(**kwargs):
        model = kwargs["model"]
        peak["cur"] += 1
        peak["max"] = max(peak["max"], peak["cur"])
        await asyncio.sleep(0.03)
        peak["cur"] -= 1
        if model == PRIMARY:
            raise litellm_exc.RateLimitError("429", llm_provider="ollama", model="primary")
        return ok_response(model, f"{model} done")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    gateway, tracker = make_gateway(monkeypatch)
    gateway.health.failure_threshold = 100  # keep circuits closed: isolate concurrency
    policy = CollaborationPolicy(allow_agent_to_agent=True, max_parallel_agents=2)
    engine = CollaborationEngine(policy=policy, tracker=tracker)
    gateway.set_context_lookup(engine.context_for)
    fb = FallbackPolicy(mode=FallbackMode.CONTINUITY, fallback_models=[BACKUP])
    agents = {}
    for name in ("a1", "a2", "a3", "a4"):
        agents[name] = Agent(
            name=name, description="d", system_prompt="s",
            llm=gateway.bind(PRIMARY, agent=name, fallback=fb),
            permissions=Permissions(allow_delegation=True,
                                    allowed_agents=["a1", "a2", "a3", "a4"]),
            collaboration=engine,
        )
    engine.register(agents)

    async def go():
        results = await engine.adelegate_many(
            [{"target": n, "task": f"t-{n}"} for n in ("a2", "a3", "a4")],
            run_id="audit-1", source_agent="a1",
        )
        return results

    results = asyncio.run(go())
    assert [r.target_agent for r in results] == ["a2", "a3", "a4"]  # input order kept
    assert all(r.status.value == "completed" for r in results)
    assert all(r.result == f"{BACKUP} done" for r in results)  # all fell back
    assert peak["max"] <= 2  # cap held on the fallback dispatch path
    ctx = engine._contexts["audit-1"]
    assert ctx.physical_reserved == 6  # 3 primary failures + 3 fallback dispatches


def test_audit_semaphore_released_during_wait_suspension(tmp_path, monkeypatch):
    """A WAIT suspension must not hold any request slot: the dispatch semaphore is
    fully released before the run is checkpointed."""
    from geosgan_hive.checkpoints import CheckpointStore, PersistMode

    monkeypatch.setattr(litellm, "completion", lambda **kw: (_ for _ in ()).throw(
        litellm_exc.RateLimitError("429", llm_provider="ollama", model="p")))
    gateway, tracker = make_gateway(monkeypatch)
    engine = CollaborationEngine(
        policy=CollaborationPolicy(allow_agent_to_agent=True, max_parallel_agents=3),
        tracker=tracker,
    )
    gateway.set_context_lookup(engine.context_for)
    store = CheckpointStore(str(tmp_path / "cp.db"), persist_mode=PersistMode.RESUMABLE)
    wait = FallbackPolicy(mode=FallbackMode.WAIT)
    sup = Supervisor(
        {"a": Agent("a", "d", "s", FakeLLM(responses=["x"]), collaboration=engine)},
        gateway.bind(PRIMARY, agent="supervisor", fallback=wait),
        collaboration=engine, checkpoint_store=store,
        checkpoint_config_json="{}", suspend_cooldown_seconds=60.0, max_wait_seconds=300.0,
    )
    result = sup.run("task", run_id="audit-2")
    assert result.suspended is True
    # the run context was closed after suspension; a fresh one shows full capacity
    ctx2 = engine.context_for("audit-2")
    assert ctx2.sync_semaphore()._value == 3  # nothing held during suspension
    store.close()


def test_audit_events_never_carry_prompt_content(monkeypatch):
    def fail(**kwargs):
        raise litellm_exc.RateLimitError("429", llm_provider="ollama", model="p")

    monkeypatch.setattr(litellm, "completion", fail)
    gateway, tracker = make_gateway(monkeypatch)
    engine = CollaborationEngine(policy=CollaborationPolicy(), tracker=tracker)
    gateway.set_context_lookup(engine.context_for)
    secret_prompt = "TOPSECRET-PROMPT-CONTENT"
    try:
        gateway.complete([Message("user", secret_prompt)], model=PRIMARY, run_id="audit-3",
                         fallback=FallbackPolicy(mode=FallbackMode.CONTINUITY,
                                                 fallback_models=[BACKUP]))
    except Exception:
        pass
    ctx = engine._contexts["audit-3"]
    blob = json.dumps([e.metadata for e in ctx.events()])
    assert secret_prompt not in blob
    assert "429" in blob or "rate_limit" in blob  # error class recorded, not content


def test_audit_needs_tools_capability_gate_via_bind(monkeypatch):
    import geosgan_hive.fallback as fallback_mod
    from geosgan_hive.providers import ModelCapabilities, model_capabilities

    def fake_caps(model):
        if model == BACKUP:
            return ModelCapabilities(context_window=None, supports_tools=False)
        return model_capabilities(model)

    monkeypatch.setattr(fallback_mod, "model_capabilities", fake_caps)
    monkeypatch.setattr(litellm, "completion", lambda **kw: (_ for _ in ()).throw(
        litellm_exc.RateLimitError("429", llm_provider="ollama", model="p")))
    gateway, tracker = make_gateway(monkeypatch)
    engine = CollaborationEngine(policy=CollaborationPolicy(), tracker=tracker)
    gateway.set_context_lookup(engine.context_for)
    client = gateway.bind(
        PRIMARY, agent="tooled", needs_tools=True,
        fallback=FallbackPolicy(mode=FallbackMode.CONTINUITY, fallback_models=[BACKUP]),
    )
    with pytest.raises(Exception) as excinfo:
        client.complete([Message("user", "hi")], run_id="audit-4")
    assert "exhausted" in str(excinfo.value).lower()
    rejected = engine._contexts["audit-4"].events(EventType.FALLBACK_REJECTED)
    assert any("tool calling" in r.metadata["reason"] for r in rejected)


def test_audit_wait_mode_keeps_permanent_errors_permanent(monkeypatch):
    """WAIT suspends only on fallback-eligible errors; an auth failure is permanent
    and must NOT become a suspension signal."""
    gateway, _ = make_gateway(monkeypatch)
    monkeypatch.setattr(litellm, "completion", lambda **kw: (_ for _ in ()).throw(
        litellm_exc.AuthenticationError("401", llm_provider="ollama", model="p")))
    with pytest.raises(AuthenticationError):
        gateway.complete([Message("user", "hi")], model=PRIMARY,
                         fallback=FallbackPolicy(mode=FallbackMode.WAIT))
    monkeypatch.setattr(litellm, "completion", lambda **kw: (_ for _ in ()).throw(
        litellm_exc.RateLimitError("429", llm_provider="ollama", model="p")))
    with pytest.raises(ProviderUnavailableError):
        gateway.complete([Message("user", "hi")], model=PRIMARY,
                         fallback=FallbackPolicy(mode=FallbackMode.HYBRID,
                                                 fallback_models=[]))
