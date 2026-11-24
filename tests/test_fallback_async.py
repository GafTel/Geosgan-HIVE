import asyncio
from types import SimpleNamespace

import litellm
import litellm.exceptions as litellm_exc
import pytest

from geosgan_hive.agent_config import Permissions
from geosgan_hive.agents import Agent
from geosgan_hive.collaboration import CollaborationEngine, DelegationStatus
from geosgan_hive.errors import (
    FallbackExhaustedError,
    ProviderUnavailableError,
    RateLimitError,
)
from geosgan_hive.fallback import FallbackMode, FallbackPolicy
from geosgan_hive.gateway import ModelGateway
from geosgan_hive.health import CircuitState, HealthRegistry
from geosgan_hive.llm import FakeLLM
from geosgan_hive.messages import Message
from geosgan_hive.retry import RetryPolicy
from geosgan_hive.run_context import CollaborationPolicy, EventType
from geosgan_hive.usage import UsageTracker

PRIMARY = "ollama/primary"
BACKUP = "ollama/backup"


def ok_response(model):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=f"answer from {model}"))],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        model=model,
    )


def make_gateway(monkeypatch, *, health=None, max_attempts=1, async_sleep=None):
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.001)
    tracker = UsageTracker()
    gateway = ModelGateway(
        tracker=tracker,
        retry_policy=RetryPolicy(
            max_attempts=max_attempts, base_delay_seconds=0, max_delay_seconds=0,
            jitter_ratio=0,
        ),
        sleep=lambda s: None,
        async_sleep=async_sleep or (lambda s: asyncio.sleep(0)),
        health=health,
    )
    return gateway, tracker


def policy(**kwargs):
    kwargs.setdefault("mode", FallbackMode.CONTINUITY)
    kwargs.setdefault("fallback_models", [BACKUP])
    return FallbackPolicy(**kwargs)


def rate_limit():
    return litellm_exc.RateLimitError("429", llm_provider="ollama", model="primary")


def test_async_primary_success(monkeypatch):
    async def run():
        async def fake_acompletion(**kwargs):
            return ok_response(kwargs["model"])

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
        gateway, _ = make_gateway(monkeypatch)
        return await gateway.acomplete(
            [Message("user", "hi")], model=PRIMARY, fallback=policy()
        )

    assert asyncio.run(run()) == f"answer from {PRIMARY}"


def test_async_ordered_fallback_and_events(monkeypatch):
    async def run():
        dispatched = []

        async def fake_acompletion(**kwargs):
            model = kwargs["model"]
            dispatched.append(model)
            if model == PRIMARY:
                raise rate_limit()
            return ok_response(model)

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
        gateway, tracker = make_gateway(monkeypatch)
        engine = CollaborationEngine(policy=CollaborationPolicy(), tracker=tracker)
        gateway.set_context_lookup(engine.context_for)
        resp = await gateway.acomplete_response(
            [Message("user", "hi")], model=PRIMARY, agent="a", run_id="ra",
            fallback=policy(mode=FallbackMode.HYBRID),
        )
        ctx = engine._contexts["ra"]
        return resp, dispatched, ctx, tracker

    resp, dispatched, ctx, tracker = asyncio.run(run())
    assert resp.model == BACKUP
    assert dispatched == [PRIMARY, BACKUP]
    assert ctx.physical_reserved == 2
    assert tracker.physical_requests("ra") == 2
    assert [e.event for e in ctx.events()] == [
        EventType.PROVIDER_FAILED, EventType.FALLBACK_SELECTED,
    ]


def test_async_all_down_continuity_vs_wait(monkeypatch):
    async def run():
        async def fake_acompletion(**kwargs):
            raise rate_limit()

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
        gateway, _ = make_gateway(monkeypatch)
        errors = []
        try:
            await gateway.acomplete(
                [Message("user", "hi")], model=PRIMARY, fallback=policy()
            )
        except Exception as exc:
            errors.append(type(exc))
        try:
            await gateway.acomplete(
                [Message("user", "hi")], model=PRIMARY,
                fallback=policy(mode=FallbackMode.WAIT),
            )
        except Exception as exc:
            errors.append(type(exc))
        return errors

    errors = asyncio.run(run())
    assert errors == [FallbackExhaustedError, ProviderUnavailableError]


def test_async_cancellation_during_retry_never_falls_back(monkeypatch):
    async def run():
        dispatched = []
        entered_sleep = asyncio.Event()

        async def fake_acompletion(**kwargs):
            dispatched.append(kwargs["model"])
            raise rate_limit()

        async def slow_sleep(_seconds):
            entered_sleep.set()
            await asyncio.sleep(30)

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
        gateway, _ = make_gateway(monkeypatch, max_attempts=3, async_sleep=slow_sleep)
        task = asyncio.create_task(
            gateway.acomplete([Message("user", "hi")], model=PRIMARY, fallback=policy())
        )
        await entered_sleep.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return dispatched

    dispatched = asyncio.run(run())
    assert dispatched == [PRIMARY]  # cancelled mid-backoff: no retry, no fallback


def test_async_recovery_probe_is_single_flight(monkeypatch):
    async def run():
        clock = {"now": 0.0}
        health = HealthRegistry(failure_threshold=1, cooldown_seconds=30.0,
                                half_open_max_probes=1, clock=lambda: clock["now"])
        dispatched = []
        probe_started = asyncio.Event()
        probe_gate = asyncio.Event()

        async def failing(**kwargs):
            if kwargs["model"] == PRIMARY:
                raise litellm_exc.Timeout("t", model="primary", llm_provider="ollama")
            return ok_response(kwargs["model"])

        async def probing(**kwargs):
            model = kwargs["model"]
            dispatched.append(model)
            if model == PRIMARY:
                probe_started.set()
                await probe_gate.wait()  # hold the probe open
            return ok_response(model)

        monkeypatch.setattr(litellm, "acompletion", failing)
        gateway, tracker = make_gateway(monkeypatch, health=health)
        engine = CollaborationEngine(policy=CollaborationPolicy(), tracker=tracker)
        gateway.set_context_lookup(engine.context_for)
        pol = policy()
        # open the primary MODEL circuit (timeout is model-scoped), backup serves
        await gateway.acomplete([Message("user", "x")], model=PRIMARY, run_id="rb",
                                fallback=pol)
        mkey = HealthRegistry.model_key(PRIMARY)
        assert health.state(mkey) == CircuitState.OPEN
        clock["now"] += 31.0  # cooldown elapsed -> HALF_OPEN
        monkeypatch.setattr(litellm, "acompletion", probing)
        dispatched.clear()

        async def second_call():
            await probe_started.wait()  # the first call now holds the probe claim
            await asyncio.sleep(0.05)  # still in flight (gate opens at 0.2s)
            return await gateway.acomplete([Message("user", "y")], model=PRIMARY,
                                           run_id="rb", fallback=pol)

        async def release_probe():
            await probe_started.wait()
            await asyncio.sleep(0.2)
            probe_gate.set()

        r1, r2, _ = await asyncio.gather(
            gateway.acomplete([Message("user", "z")], model=PRIMARY, run_id="rb",
                              fallback=pol),
            second_call(),
            release_probe(),
        )
        ctx = engine._contexts["rb"]
        return dispatched, r1, r2, ctx, health, mkey

    dispatched, r1, r2, ctx, health, mkey = asyncio.run(run())
    # exactly one probe was claimed; the concurrent call saw "probe in flight"
    # and went to the fallback instead of stampeding the provider
    assert dispatched.count(PRIMARY) == 1
    assert dispatched.count(BACKUP) == 1
    assert r1 == f"answer from {PRIMARY}"
    assert r2 == f"answer from {BACKUP}"
    assert len(ctx.events(EventType.RECOVERY_PROBE)) == 1
    # the concurrent call's primary candidate was rejected while the probe ran
    assert any(e.metadata.get("model") == PRIMARY
               for e in ctx.events(EventType.FALLBACK_REJECTED))
    assert health.state(mkey) == CircuitState.CLOSED


def test_async_nested_delegation_with_fallback(monkeypatch):
    """A delegated agent whose primary model fails is served by its fallback model
    through the real gateway dispatch (nested collaboration under fallback)."""
    async def run():
        dispatched = []

        async def fake_acompletion(**kwargs):
            model = kwargs["model"]
            dispatched.append(model)
            if model == PRIMARY:
                raise rate_limit()
            return ok_response(model)

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
        tracker = UsageTracker()
        gateway, _ = make_gateway(monkeypatch)
        engine = CollaborationEngine(
            policy=CollaborationPolicy(allow_agent_to_agent=True), tracker=tracker
        )
        gateway.set_context_lookup(engine.context_for)
        analyst_llm = gateway.bind(PRIMARY, agent="analyst", fallback=policy())
        researcher = Agent(
            name="researcher", description="d", system_prompt="s",
            llm=FakeLLM(responses=["done"]),
            permissions=Permissions(allow_delegation=True, allowed_agents=["analyst"]),
            collaboration=engine,
        )
        analyst = Agent(
            name="analyst", description="d", system_prompt="s", llm=analyst_llm,
            collaboration=engine,
        )
        engine.register({"researcher": researcher, "analyst": analyst})
        result = await engine.adelegate(
            {"target": "analyst", "task": "analyze"}, run_id="rc", source_agent="researcher"
        )
        ctx = engine._contexts["rc"]
        return result, dispatched, ctx

    result, dispatched, ctx = asyncio.run(run())
    assert result.status == DelegationStatus.COMPLETED
    assert result.result == f"answer from {BACKUP}"
    assert dispatched == [PRIMARY, BACKUP]
    assert ctx.physical_reserved == 2
    assert ctx.events(EventType.FALLBACK_SELECTED)


def test_async_off_mode_unchanged(monkeypatch):
    async def run():
        dispatched = []

        async def fake_acompletion(**kwargs):
            dispatched.append(kwargs["model"])
            raise rate_limit()

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
        gateway, _ = make_gateway(monkeypatch, max_attempts=2)
        with pytest.raises(RateLimitError):
            await gateway.acomplete(
                [Message("user", "hi")], model=PRIMARY,
                fallback=FallbackPolicy(mode=FallbackMode.OFF, fallback_models=[BACKUP]),
            )
        return dispatched

    assert asyncio.run(run()) == [PRIMARY, PRIMARY]
